"""
Whisper STT Docker Service — faster-whisper based transcription API.

Supports dual-device operation:
- CPU: runs in main process (permanent, no VRAM)
- GPU: runs in a separate child process (killed after TTL to fully release
  CUDA context + VRAM, enabling GPU P8 power state)

Device is selected per request. Model and parameters are changeable at
runtime via the Web-UI or /config endpoint.

Endpoints:
    GET  /          — Web-UI (status, model management, settings)
    GET  /health    — Health check (model status, device info)
    POST /transcribe — Transcribe audio file (multipart/form-data)
    GET  /status    — Detailed status (model info, memory usage)
    POST /unload    — Unload model(s) to free memory
    GET  /config    — Get current configuration
    POST /config    — Update configuration (JSON body)
"""

from __future__ import annotations

import gc
import multiprocessing
import os
import time
import tempfile
import threading
from pathlib import Path

from flask import Flask, request, jsonify

app = Flask(__name__)

# ── Configuration (mutable at runtime via /config) ───────────

AVAILABLE_MODELS = ["tiny", "base", "small", "medium", "large-v3"]
# Engines per request: Whisper (faster-whisper) or NVIDIA Parakeet TDT 0.6B v3
# (onnx-asr). Parakeet quality: fp32 (full) or int8 (smaller, more word errors).
ENGINES = ["whisper", "parakeet"]
PARAKEET_QUALITIES = ["fp32", "int8"]

_default_model = os.environ.get("WHISPER_MODEL", "medium")

_config = {
    "cpu_model": os.environ.get("WHISPER_CPU_MODEL", _default_model),
    "gpu_model": os.environ.get("WHISPER_GPU_MODEL", _default_model),
    "cpu_compute": os.environ.get("WHISPER_CPU_COMPUTE", "int8"),
    "gpu_compute": os.environ.get("WHISPER_GPU_COMPUTE", "float16"),
    "gpu_ttl_minutes": int(os.environ.get("WHISPER_GPU_TTL_MINUTES", "30")),
    "language": os.environ.get("WHISPER_LANGUAGE", "de"),
    "vad_filter": os.environ.get("WHISPER_VAD_FILTER", "1") in ("1", "true"),
    "beam_size": int(os.environ.get("WHISPER_BEAM_SIZE", "5")),
    # Carries the transcript so far into the next 30 s window. Without it the
    # initial_prompt below only styles the FIRST window and everything after
    # falls back to unpunctuated staccato (measured on a 2.5 h interview).
    # Known trade-off: on very long files this is what can trigger Whisper's
    # repetition loops — switch it off in the STT tab if that ever shows up.
    "condition_on_previous_text": os.environ.get(
        "WHISPER_CONDITION_ON_PREVIOUS", "1") in ("1", "true"),
    # "auto" = built-in per-language style primer, "" = off, anything else
    # is used verbatim. See _resolve_initial_prompt for why this exists.
    "initial_prompt": os.environ.get("WHISPER_INITIAL_PROMPT", "auto"),
    # Default speaker-count hint for diarization; 0 = let pyannote estimate.
    # Per-request num_speakers still wins. Set to 2 before transcribing an
    # interview and the greeting ping-pong stops collapsing into one speaker.
    "num_speakers": int(os.environ.get("WHISPER_NUM_SPEAKERS", "0")),
    # Default engine when a request names none — the API stays unchanged for
    # clients that never send this field. Parakeet quality is per device
    # (like cpu_model/gpu_model): int8 is the fast choice for the CPU, fp32 the
    # accurate one for the GPU.
    "engine": os.environ.get("STT_ENGINE", "whisper"),
    "cpu_quality": os.environ.get("STT_CPU_QUALITY", "fp32"),
    "gpu_quality": os.environ.get("STT_GPU_QUALITY", "fp32"),
}

# Whisper continues whatever "style" the transcript has so far. At the very
# start there is no transcript, so the first 30 s window decides — and fast
# dialogue ping-pong (greetings, "right", "yeah") reliably tips it into an
# unpunctuated staccato that then sticks. Seeding a well-punctuated sentence
# as fake history locks in written style with punctuation instead. The
# primer text itself never appears in the output.
_INITIAL_PROMPTS = {
    "de": "Schön, Sie wiederzusehen. Wir haben heute viel zu besprechen, nicht wahr? Ja, absolut.",
    "en": "Well, it's great to see you again. We have a lot to talk about today, don't we? Yes, absolutely.",
}


def _resolve_initial_prompt(language: str) -> str | None:
    """Effective initial_prompt for a request, or None to send nothing.

    With language "auto" no primer is used: Whisper detects the language
    from the first window, and a primer in the wrong language could tip
    that detection.
    """
    configured = _config["initial_prompt"]
    if not configured:
        return None
    if configured != "auto":
        return configured
    return _INITIAL_PROMPTS.get(language)
EAGER_LOAD = os.environ.get("WHISPER_EAGER_LOAD", "1") in ("1", "true", "True")

# Minimum free VRAM (MiB) to load GPU model. Whisper medium ≈ 1500 MiB.
_MIN_VRAM_MIB = int(os.environ.get("WHISPER_MIN_VRAM_MIB", "2000"))
# Smallest model the GPU degradation chain may fall back to (see _start_gpu_worker).
GPU_MIN_MODEL = os.environ.get("WHISPER_GPU_MIN_MODEL", "medium")

# Per-model VRAM demand (MiB, float16 incl. CUDA context headroom). Used by
# the degradation chain in _start_gpu_worker: the configured model is tried
# first, and if no GPU has room the next smaller model takes its place —
# loudly logged, never silent. Unknown names fall back to _MIN_VRAM_MIB.
_MODEL_VRAM_MIB = {
    "tiny": 600, "base": 700, "small": 1300, "medium": 2000, "large-v3": 4200,
}

# ── Speaker diarization (optional, own worker on its own GPU) ─
# Off by default: 99 % of requests are short voice commands where speaker
# labels are pointless and would only cost load time. Only long recordings
# (interviews, meetings) ask for it via diarize=1 per request.
# pyannote 4.x ships "community-1" as its current pipeline; the older 3.1
# identifier just redirects there, so name it directly.
DIARIZE_MODEL = os.environ.get("DIARIZE_MODEL", "pyannote/speaker-diarization-community-1")
# pyannote 3.1 ≈ 1 GiB VRAM — noticeably less than Whisper needs.
_DIARIZE_MIN_VRAM_MIB = int(os.environ.get("DIARIZE_MIN_VRAM_MIB", "1500"))
_DIARIZE_TTL_MINUTES = int(os.environ.get("DIARIZE_TTL_MINUTES", "10"))
# 0 = every GPU qualifies, which holds as long as torch is pinned to the
# CUDA 12 build (see Dockerfile): its cuDNN still supports Volta, so the
# V100s are usable. Should torch ever move to CUDA 13, sm_70 aborts with
# "cuDNN version ... not compatible with devices with SM < 7.5" — set this
# to 7.5 then to keep diarization on Turing or newer.
_DIARIZE_MIN_COMPUTE_CAP = float(os.environ.get("DIARIZE_MIN_COMPUTE_CAP", "0"))
_HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Max wait for one GPU transcription result. Long meeting recordings need
# minutes even on GPU (medium ≈ 10-20x realtime → 2 h audio ≈ 6-12 min);
# matches the AIfred-side WHISPER_TRANSCRIBE_TIMEOUT_S.
_GPU_RESULT_TIMEOUT_S = int(os.environ.get("WHISPER_GPU_RESULT_TIMEOUT_S", "1800"))


# ── CPU Model (main process, permanent) ──────────────────────

_model_cpu = None
_cpu_lock = threading.Lock()


_cpu_model_name: str = ""  # Track which model is actually loaded on CPU


def _load_cpu_model():
    """Load Whisper model on CPU in the main process."""
    global _model_cpu, _cpu_model_name
    from faster_whisper import WhisperModel

    model_name = _config["cpu_model"]
    compute = _config["cpu_compute"]
    t0 = time.time()
    print(f"[Whisper] Loading model '{model_name}' on cpu ({compute})...", flush=True)
    _model_cpu = WhisperModel(model_name, device="cpu", compute_type=compute)
    _cpu_model_name = model_name
    print(f"[Whisper] Model loaded on cpu in {time.time() - t0:.1f}s", flush=True)


def _transcribe_cpu(audio_path: str, language: str, return_segments: bool = False) -> dict:
    """Transcribe using CPU model (main process)."""
    global _model_cpu
    with _cpu_lock:
        if _model_cpu is None:
            _load_cpu_model()
        model = _model_cpu

    t0 = time.time()
    segments, info = model.transcribe(
        audio_path,
        language=language if language != "auto" else None,
        vad_filter=_config["vad_filter"],
        beam_size=_config["beam_size"],
        condition_on_previous_text=_config["condition_on_previous_text"],
        initial_prompt=_resolve_initial_prompt(language),
        # See the GPU worker: per-word stamps only when diarizing, so the
        # speaker merge can cut mid-segment where the speaker changes.
        word_timestamps=return_segments,
    )
    # Materialise once — ``segments`` is a generator; the timestamps are what
    # the speaker merge needs.
    seg_list = []
    text_parts = []
    for s in segments:
        text_parts.append(s.text)
        if return_segments and s.words:
            seg_list.extend((w.start, w.end, w.word) for w in s.words)
        elif return_segments:
            seg_list.append((s.start, s.end, s.text))
    text = " ".join(text_parts).strip()
    elapsed = time.time() - t0
    print(f"[Whisper] Transcribed (cpu, {elapsed:.2f}s): {text[:80]}...", flush=True)
    payload = {
        "text": text, "time": round(elapsed, 3), "device": "cpu",
        "language": info.language,
        "language_probability": round(info.language_probability, 3),
    }
    if return_segments:
        payload["segments"] = seg_list
    return payload


# ── Parakeet (onnx-asr) ──────────────────────────────────────
#
# Measured on German audiobooks (4 × 60 s): fp32 matches Whisper large-v3
# closely and never dropped text where Whisper skipped half a paragraph;
# CPU ≈ 5 s per audio minute (Whisper medium 12–23 s), V100 ≈ 0.25 s.

_PARAKEET_REPO = "istupakov/parakeet-tdt-0.6b-v3-onnx"
# A real directory inside the HF volume: onnxruntime refuses the fp32
# encoder's external weight file behind the HF cache's blob symlinks.
_PARAKEET_DIR = Path(os.environ.get("PARAKEET_DIR", "/root/.cache/huggingface/parakeet-tdt-0.6b-v3"))
# VRAM for one chunk (measured peak ≈ 3.9 GB on a 60 s chunk, fp32).
_PARAKEET_VRAM_MIB = int(os.environ.get("PARAKEET_MIN_VRAM_MIB", "4500"))
# Audio is cut at speech pauses into chunks of at most this length, so VRAM
# does not grow with the recording. Short pieces hurt: below ~20 s the model
# loses context (names garbled, stray English words), so pauses shorter
# than PARAKEET_MERGE_SILENCE_MS are merged up to the chunk length.
_PARAKEET_CHUNK_S = float(os.environ.get("PARAKEET_CHUNK_S", "60"))
_PARAKEET_MERGE_SILENCE_MS = float(os.environ.get("PARAKEET_MERGE_SILENCE_MS", "5000"))
_parakeet_download_lock = threading.Lock()


def _parakeet_model_dir() -> str:
    """The model files, downloaded once into the HF volume."""
    with _parakeet_download_lock:
        if not (_PARAKEET_DIR / "config.json").exists():
            from huggingface_hub import snapshot_download
            print(f"[Parakeet] Downloading {_PARAKEET_REPO} → {_PARAKEET_DIR}", flush=True)
            snapshot_download(_PARAKEET_REPO, local_dir=str(_PARAKEET_DIR))
    return str(_PARAKEET_DIR)


def _load_parakeet(quality: str, provider: str):
    """Parakeet with VAD chunking and timestamps on one onnxruntime provider.

    onnxruntime falls back to the CPU silently when CUDA cannot load (wrong
    CUDA/cuDNN) — checked here, so a "GPU" worker never computes on the CPU.
    """
    import onnx_asr
    t0 = time.time()
    model = onnx_asr.load_model(
        "nemo-parakeet-tdt-0.6b-v3", _parakeet_model_dir(),
        quantization="int8" if quality == "int8" else None, providers=[provider],
    )
    active = model.asr._encoder.get_providers()[0]
    if active != provider:
        raise RuntimeError(f"Parakeet runs on {active}, not {provider}")
    vad = onnx_asr.load_vad("silero", providers=["CPUExecutionProvider"])
    print(f"[Parakeet] Loaded {quality} on {provider} in {time.time() - t0:.1f}s", flush=True)
    return model.with_vad(vad, max_speech_duration_s=_PARAKEET_CHUNK_S,
                          min_silence_duration_ms=_PARAKEET_MERGE_SILENCE_MS).with_timestamps()


def _parakeet_transcribe(model, wav_path: str, want_words: bool) -> tuple[str, list]:
    """Text, plus (start, end, word) for the speaker merge when asked.

    Tokens are word pieces; a new word starts with a space. Token times are
    relative to their chunk, so the chunk start is added.
    """
    texts: list[str] = []
    words: list[tuple[float, float, str]] = []
    for seg in model.recognize(wav_path):
        if seg.text.strip():
            texts.append(seg.text.strip())
        if not want_words:
            continue
        current: list | None = None
        for token, rel in zip(seg.tokens, seg.timestamps):
            at = seg.start + rel
            if current is None or token.startswith(" "):
                if current:
                    words.append(tuple(current))
                current = [at, at, token]
            else:
                current[1] = at
                current[2] += token
        if current:
            words.append(tuple(current))
    return " ".join(texts), words


_parakeet_cpu = None
_parakeet_cpu_quality = ""
_parakeet_cpu_lock = threading.Lock()


def _transcribe_parakeet_cpu(wav_path: str, language: str, quality: str,
                             return_segments: bool = False) -> dict:
    """Parakeet on the CPU (main process, stays loaded like Whisper's)."""
    global _parakeet_cpu, _parakeet_cpu_quality
    with _parakeet_cpu_lock:
        if _parakeet_cpu is None or _parakeet_cpu_quality != quality:
            _parakeet_cpu = _load_parakeet(quality, "CPUExecutionProvider")
            _parakeet_cpu_quality = quality
        model = _parakeet_cpu
    t0 = time.time()
    text, words = _parakeet_transcribe(model, wav_path, return_segments)
    elapsed = time.time() - t0
    print(f"[Parakeet] Transcribed (cpu, {quality}, {elapsed:.2f}s): {text[:80]}...", flush=True)
    payload = {"text": text, "time": round(elapsed, 3), "device": "cpu",
               "engine": "parakeet", "quality": quality,
               # Parakeet detects the language itself and reports none.
               "language": language, "language_probability": None}
    if return_segments:
        payload["segments"] = words
    return payload


# ── GPU Worker (child process, killed after TTL) ─────────────

_gpu_process: multiprocessing.Process | None = None
_gpu_request_queue: multiprocessing.Queue | None = None
_gpu_result_queue: multiprocessing.Queue | None = None
_gpu_busy = False  # True while a transcription job is in flight
_gpu_lock = threading.Lock()
_gpu_ttl_timer: threading.Timer | None = None
_last_gpu_request = 0.0
_gpu_device_index: int | None = None
_gpu_uuid: str = ""  # UUID of the card Whisper sits on (diarization avoids it)
_gpu_model_name: str = ""  # Track which model is loaded on GPU
_gpu_engine: str = ""  # whisper | parakeet — the worker holds one engine
_gpu_quality: str = ""  # Parakeet quality of the loaded worker


def _find_best_gpu(min_vram_mib: int | None = None,
                   exclude_uuid: str | None = None,
                   min_compute_cap: float = 0.0,
                   tag: str = "Whisper") -> tuple[int, str] | None:
    """Find the GPU with the most free VRAM. Prefers completely empty GPUs.

    Returns (nvidia-smi index, GPU UUID). The worker pins the card via its
    UUID — same SSOT technique as the llama-swap profiles — so the CUDA
    enumeration order (FASTEST_FIRST vs PCI) can never select a different
    card than the one reported here.

    ``exclude_uuid`` skips a card already taken by another worker, so the
    diarization model lands on a different GPU than Whisper instead of both
    fighting over the same scraps of free VRAM.

    ``min_compute_cap`` filters by GPU generation. PyTorch's bundled cuDNN
    dropped Volta: anything below 7.5 aborts with "cuDNN version ... is not
    compatible with devices with SM < 7.5" the moment a model is moved to
    that card. Whisper/CTranslate2 is unaffected and still uses every GPU.
    """
    import subprocess
    min_vram = _MIN_VRAM_MIB if min_vram_mib is None else min_vram_mib
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,memory.free,memory.used,name,compute_cap",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        best: tuple[int, str] | None = None
        best_free, best_used = 0, float("inf")
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                idx, uuid = int(parts[0]), parts[1]
                free, used = int(parts[2]), int(parts[3])
                name = parts[4]
                cap = float(parts[5]) if len(parts) >= 6 else 0.0
                print(f"[{tag}]   GPU {idx}: {name} (sm_{cap}) — {free} MiB free, "
                      f"{used} MiB used", flush=True)
                if uuid == exclude_uuid:
                    continue
                if min_compute_cap and cap < min_compute_cap:
                    continue
                if free < min_vram:
                    continue
                if best is None:
                    best, best_free, best_used = (idx, uuid), free, used
                elif used == 0 and best_used > 0:
                    best, best_free, best_used = (idx, uuid), free, used
                elif (used == 0) == (best_used == 0) and free > best_free:
                    best, best_free, best_used = (idx, uuid), free, used
        if best is not None:
            print(f"[{tag}] Selected GPU {best[0]} ({best[1]}, "
                  f"free={best_free} MiB, used={int(best_used)} MiB)", flush=True)
        else:
            print(f"[{tag}] No GPU with >= {min_vram} MiB free VRAM", flush=True)
        return best
    except Exception as e:
        print(f"[{tag}] GPU detection failed: {e}", flush=True)
        return None


def _detect_gpu_compute(gpu_idx: int) -> str:
    """Detect best compute type for a GPU based on Compute Capability."""
    import subprocess
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,compute_cap",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2 and int(parts[0]) == gpu_idx:
                cc = float(parts[1])
                if cc >= 7.0:
                    return _config["gpu_compute"]
                print(f"[Whisper] GPU {gpu_idx} CC {cc} < 7.0 → using int8", flush=True)
                return "int8"
    except Exception:
        pass
    return "int8"


def _gpu_worker(req_queue: multiprocessing.Queue, res_queue: multiprocessing.Queue,
                gpu_idx: int, gpu_uuid: str, model_name: str, compute: str,
                engine: str = "whisper", quality: str = "fp32"):
    """Child process: load model on GPU, process requests until killed."""
    # Pin by UUID (SSOT, same as llama-swap profiles) — immune to CUDA's
    # enumeration order, which differs from nvidia-smi on mixed-GPU hosts.
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_uuid
    print(f"[Whisper/GPU] Worker started on GPU {gpu_idx} ({gpu_uuid}, PID {os.getpid()})", flush=True)

    t0 = time.time()
    try:
        if engine == "parakeet":
            model = _load_parakeet(quality, "CUDAExecutionProvider")
        else:
            from faster_whisper import WhisperModel
            print(f"[Whisper/GPU] Loading model '{model_name}' ({compute})...", flush=True)
            model = WhisperModel(model_name, device="cuda", compute_type=compute, device_index=0)
    except Exception as e:
        print(f"[Whisper/GPU] Loading {engine} failed: {e}", flush=True)
        res_queue.put({"status": "failed", "error": str(e)})
        return
    print(f"[Whisper/GPU] {engine} loaded in {time.time() - t0:.1f}s", flush=True)

    # Signal parent that we're ready
    res_queue.put({"status": "ready"})

    while True:
        try:
            job = req_queue.get(timeout=10)
        except Exception:
            continue  # Keep waiting

        if job is None:
            break  # Poison pill → exit

        audio_path = job["audio_path"]
        language = job["language"]

        try:
            t0 = time.time()
            want_words = bool(job.get("return_segments"))
            if engine == "parakeet":
                text, words = _parakeet_transcribe(model, audio_path, want_words)
                elapsed = time.time() - t0
                print(f"[Parakeet/GPU] Transcribed ({quality}, {elapsed:.2f}s): {text[:80]}...", flush=True)
                payload = {"text": text, "time": round(elapsed, 3), "device": "cuda",
                           "engine": "parakeet", "quality": quality,
                           "language": language, "language_probability": None}
                if want_words:
                    payload["segments"] = words
                res_queue.put(payload)
                continue
            segments, info = model.transcribe(
                audio_path,
                language=language if language != "auto" else None,
                vad_filter=job.get("vad_filter", True),
                beam_size=job.get("beam_size", 5),
                condition_on_previous_text=job.get("condition_on_previous_text", False),
                initial_prompt=job.get("initial_prompt"),
                # Word timestamps only when diarizing: a Whisper segment can
                # span 10+ seconds and contain BOTH speakers, so segment-level
                # assignment smears turns together. Per-word stamps let the
                # merge cut exactly where the speaker changes.
                word_timestamps=want_words,
            )
            # Materialise once — ``segments`` is a generator, a second pass
            # would be empty. Timestamps are what the speaker merge needs.
            seg_list = []
            text_parts = []
            for s in segments:
                text_parts.append(s.text)
                if want_words and s.words:
                    seg_list.extend((w.start, w.end, w.word) for w in s.words)
                elif want_words:
                    seg_list.append((s.start, s.end, s.text))
            text = " ".join(text_parts).strip()
            elapsed = time.time() - t0
            print(f"[Whisper/GPU] Transcribed ({elapsed:.2f}s): {text[:80]}...", flush=True)
            payload = {
                "text": text, "time": round(elapsed, 3), "device": "cuda",
                "language": info.language,
                "language_probability": round(info.language_probability, 3),
            }
            if job.get("return_segments"):
                payload["segments"] = seg_list
            res_queue.put(payload)
        except Exception as e:
            res_queue.put({"error": str(e)})

    print(f"[Whisper/GPU] Worker exiting (PID {os.getpid()})", flush=True)


def _start_gpu_worker(engine: str = "whisper", quality: str = "fp32") -> bool:
    """Start GPU child process on the best available GPU.

    Whisper: tries the configured model first; when no GPU has enough free
    VRAM for it, walks down the model list, but not below GPU_MIN_MODEL, and
    loads the largest one that fits (agreed degradation, logged loudly).
    Parakeet: one model, needs _PARAKEET_VRAM_MIB. When nothing fits it
    reports failure; the caller then decides about the CPU.
    """
    global _gpu_process, _gpu_request_queue, _gpu_result_queue, _gpu_device_index, _gpu_model_name, _gpu_uuid
    global _gpu_engine, _gpu_quality

    if engine == "parakeet":
        selected = _find_best_gpu(min_vram_mib=_PARAKEET_VRAM_MIB, tag="Parakeet")
        if selected is None:
            return False
        return _spawn_gpu_worker(selected, f"parakeet-{quality}", "", engine, quality)

    configured = _config["gpu_model"]
    chain_start = AVAILABLE_MODELS.index(configured) if configured in AVAILABLE_MODELS else 0
    # Never below the floor: smaller models transcribe too poorly to be worth it, the CPU
    # with its better model is the caller's choice then.
    chain_end = min(AVAILABLE_MODELS.index(GPU_MIN_MODEL), chain_start)
    candidates = list(reversed(AVAILABLE_MODELS[chain_end:chain_start + 1]))

    selected = None
    model_name = configured
    for model_name in candidates:
        need = _MODEL_VRAM_MIB.get(model_name, _MIN_VRAM_MIB)
        selected = _find_best_gpu(min_vram_mib=need)
        if selected is not None:
            if model_name != configured:
                print(f"[Whisper] No GPU fits '{configured}' "
                      f"({_MODEL_VRAM_MIB.get(configured, _MIN_VRAM_MIB)} MiB) — "
                      f"degraded to '{model_name}' ({need} MiB)", flush=True)
            break
    if selected is None:
        return False
    return _spawn_gpu_worker(selected, model_name, _detect_gpu_compute(selected[0]), engine, quality)


def _spawn_gpu_worker(selected: tuple[int, str], model_name: str, compute: str,
                      engine: str, quality: str) -> bool:
    """Start the worker process on the chosen card and wait until it is ready."""
    global _gpu_process, _gpu_request_queue, _gpu_result_queue, _gpu_device_index, _gpu_model_name, _gpu_uuid
    global _gpu_engine, _gpu_quality
    gpu_idx, gpu_uuid = selected
    _gpu_device_index = gpu_idx
    _gpu_uuid = gpu_uuid
    _gpu_model_name = model_name
    _gpu_engine = engine
    _gpu_quality = quality if engine == "parakeet" else ""

    _gpu_request_queue = multiprocessing.Queue()
    _gpu_result_queue = multiprocessing.Queue()

    _gpu_process = multiprocessing.Process(
        target=_gpu_worker,
        args=(_gpu_request_queue, _gpu_result_queue, gpu_idx, gpu_uuid, model_name, compute,
              engine, quality),
        daemon=True,
        name=f"{engine}-gpu-{gpu_idx}",
    )
    _gpu_process.start()

    # Wait for ready signal (the first Parakeet start also downloads the model)
    try:
        msg = _gpu_result_queue.get(timeout=600 if engine == "parakeet" else 120)
        if msg.get("status") == "ready":
            print(f"[Whisper] GPU worker ({engine}) ready on GPU {gpu_idx}", flush=True)
            return True
    except Exception:
        pass

    print(f"[Whisper] GPU worker ({engine}) failed to start", flush=True)
    _kill_gpu_worker()
    return False


def _kill_gpu_worker():
    """Kill the GPU child process to fully release CUDA context + VRAM."""
    global _gpu_process, _gpu_request_queue, _gpu_result_queue, _gpu_device_index, _gpu_model_name, _gpu_uuid
    global _gpu_engine, _gpu_quality

    if _gpu_process is not None:
        gpu_idx = _gpu_device_index
        pid = _gpu_process.pid
        print(f"[Whisper] Killing GPU worker (PID {pid}, GPU {gpu_idx})", flush=True)
        try:
            _gpu_process.kill()
            _gpu_process.join(timeout=5)
        except Exception:
            pass
        _gpu_process = None
        print("[Whisper] GPU worker killed — VRAM fully released", flush=True)

    if _gpu_request_queue is not None:
        try:
            _gpu_request_queue.close()
        except Exception:
            pass
        _gpu_request_queue = None
    if _gpu_result_queue is not None:
        if _gpu_busy:
            # Unblock a /transcribe request waiting on this queue — without
            # this it would sit out the full result timeout after a force
            # unload killed its worker mid-job. The queue is deliberately
            # not closed here; the waiter still reads the error from it.
            try:
                _gpu_result_queue.put({"error": "GPU worker was unloaded mid-transcription"})
            except Exception:
                pass
        else:
            try:
                _gpu_result_queue.close()
            except Exception:
                pass
        _gpu_result_queue = None
    _gpu_device_index = None
    _gpu_uuid = ""
    _gpu_model_name = ""
    _gpu_engine = ""
    _gpu_quality = ""


def _reset_gpu_ttl():
    """Reset the GPU auto-kill timer."""
    global _gpu_ttl_timer, _last_gpu_request
    _last_gpu_request = time.time()
    ttl = _config["gpu_ttl_minutes"]
    if ttl <= 0:
        return
    if _gpu_ttl_timer is not None:
        _gpu_ttl_timer.cancel()
    _gpu_ttl_timer = threading.Timer(ttl * 60, _kill_gpu_worker)
    _gpu_ttl_timer.daemon = True
    _gpu_ttl_timer.start()


def _transcribe_gpu(audio_path: str, language: str,
                    return_segments: bool = False,
                    engine: str = "whisper", quality: str = "fp32") -> dict | None:
    """Transcribe using GPU child process."""
    global _gpu_busy
    with _gpu_lock:
        alive = _gpu_process is not None and _gpu_process.is_alive()
        # The worker holds one engine; another one means a swap — unless a
        # job is still running on it, then this request gets the usual 503
        # and the caller falls back to the CPU.
        wanted = (engine, quality if engine == "parakeet" else "")
        if alive and (_gpu_engine, _gpu_quality) != wanted:
            if _gpu_busy:
                return None
            _kill_gpu_worker()
            alive = False
        # Start worker if not running
        if not alive:
            if not _start_gpu_worker(engine, quality):
                return None

        # Send job
        _gpu_busy = True
        _gpu_request_queue.put({
            "audio_path": audio_path,
            "language": language,
            "vad_filter": _config["vad_filter"],
            "beam_size": _config["beam_size"],
            "condition_on_previous_text": _config["condition_on_previous_text"],
            "initial_prompt": _resolve_initial_prompt(language),
            "return_segments": return_segments,
        })

    # Wait for result (outside lock so CPU requests aren't blocked)
    try:
        result = _gpu_result_queue.get(timeout=_GPU_RESULT_TIMEOUT_S)
        _reset_gpu_ttl()
        return result
    except Exception:
        return {"error": f"GPU worker timeout ({_GPU_RESULT_TIMEOUT_S}s)"}
    finally:
        _gpu_busy = False


def _decode_to_wav16k(src_path: str) -> str:
    """Decode audio once to 16 kHz mono WAV — the format BOTH engines want.

    Whisper and pyannote each resample to 16 kHz mono internally, so without
    this the same MP3 gets decoded twice in parallel: two CPU cores busy for
    the same result, and on a 2.5 h recording that is a minute of pure waste
    before either GPU sees data. Decoding once up front makes it a shared
    input (and both engines then just read PCM).

    Returns the WAV path, or the original path if conversion failed (logged
    loudly — the engines can still decode it themselves).
    """
    import subprocess

    wav_path = f"{src_path}.16k.wav"
    t0 = time.time()
    try:
        proc = subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", src_path,
             "-ar", "16000", "-ac", "1", "-f", "wav", "-y", wav_path],
            capture_output=True, text=True, timeout=1800,
        )
        if proc.returncode == 0 and Path(wav_path).exists():
            size_mb = Path(wav_path).stat().st_size / 1024 / 1024
            print(f"[Whisper] Decoded once to 16 kHz mono WAV "
                  f"({size_mb:.0f} MB, {time.time() - t0:.1f}s) — shared by both engines",
                  flush=True)
            return wav_path
        print(f"[Whisper] ffmpeg pre-decode FAILED (rc={proc.returncode}): "
              f"{proc.stderr[:200]} — engines will decode separately", flush=True)
    except Exception as e:
        print(f"[Whisper] ffmpeg pre-decode FAILED: {e} — engines will decode separately",
              flush=True)
    return src_path


# ── Speaker Diarization (own worker process, own GPU) ────────
#
# Transcription and diarization are fully independent: both only read the
# same audio FILE, no tensors are shared. That is why the pyannote model can
# sit on a different card than Whisper — and why both jobs can run at the
# same time. They only meet at the end, on the CPU, where each Whisper
# segment gets the speaker whose turn overlaps it most.

_diarize_process: multiprocessing.Process | None = None
_diarize_request_queue: multiprocessing.Queue | None = None
_diarize_result_queue: multiprocessing.Queue | None = None
_diarize_lock = threading.Lock()
_diarize_ttl_timer: threading.Timer | None = None
_diarize_device_index: int | None = None


def _diarize_worker(req_queue: multiprocessing.Queue, res_queue: multiprocessing.Queue,
                    gpu_idx: int, gpu_uuid: str, model_name: str, hf_token: str):
    """Child process: load the pyannote pipeline on its own GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_uuid
    print(f"[Diarize] Worker started on GPU {gpu_idx} ({gpu_uuid}, PID {os.getpid()})", flush=True)

    import wave

    import numpy as np
    import torch
    from pyannote.audio import Pipeline

    def load_wav16k(path: str):
        """Read the pre-decoded WAV into a (channel, time) float tensor.

        Deliberately bypasses pyannote's own decoder (torchcodec): that one
        is built against a different CUDA generation than the torch we pin
        for Volta support and fails on missing libs. Since ffmpeg already
        produced 16 kHz mono PCM upstream, handing the samples over directly
        is both simpler and one decode cheaper.
        """
        with wave.open(path, "rb") as w:
            sample_rate = w.getframerate()
            channels = w.getnchannels()
            raw = w.readframes(w.getnframes())
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if channels > 1:
            data = data.reshape(-1, channels).mean(axis=1)
        return torch.from_numpy(data.copy()).unsqueeze(0), sample_rate

    t0 = time.time()
    print(f"[Diarize] Loading pipeline '{model_name}'...", flush=True)
    # pyannote 4.x renamed the auth parameter to ``token``. Passing None
    # lets huggingface_hub fall back to the mounted token file.
    pipeline = Pipeline.from_pretrained(model_name, token=hf_token or None)
    if pipeline is None:
        res_queue.put({"error": f"Pipeline '{model_name}' could not be loaded "
                                "(gated model — check HF_TOKEN and licence acceptance)"})
        return
    pipeline.to(torch.device("cuda"))
    print(f"[Diarize] Pipeline loaded in {time.time() - t0:.1f}s", flush=True)

    res_queue.put({"status": "ready"})

    while True:
        try:
            job = req_queue.get(timeout=10)
        except Exception:
            continue
        if job is None:
            break

        try:
            t0 = time.time()
            kwargs = {}
            if job.get("num_speakers"):
                kwargs["num_speakers"] = int(job["num_speakers"])
            waveform, sample_rate = load_wav16k(job["audio_path"])
            output = pipeline({"waveform": waveform, "sample_rate": sample_rate},
                              **kwargs)
            # pyannote 4 returns a DiarizeOutput. Prefer its "exclusive"
            # variant: overlapping speech is resolved to a single speaker per
            # instant, which is exactly what assigning whole transcript
            # segments to one speaker needs.
            annotation = getattr(output, "exclusive_speaker_diarization", None)
            if annotation is None:
                annotation = getattr(output, "speaker_diarization", output)
            turns = [(seg.start, seg.end, spk)
                     for seg, _, spk in annotation.itertracks(yield_label=True)]
            speakers = sorted({t[2] for t in turns})
            elapsed = time.time() - t0
            print(f"[Diarize] {len(turns)} turns, {len(speakers)} speaker(s) "
                  f"({elapsed:.1f}s)", flush=True)
            res_queue.put({"turns": turns, "speakers": speakers,
                           "time": round(elapsed, 3)})
        except Exception as e:
            res_queue.put({"error": str(e)})

    print(f"[Diarize] Worker exiting (PID {os.getpid()})", flush=True)


def _start_diarize_worker() -> bool:
    """Start the diarization child process, avoiding Whisper's GPU."""
    global _diarize_process, _diarize_request_queue, _diarize_result_queue, _diarize_device_index

    selected = _find_best_gpu(min_vram_mib=_DIARIZE_MIN_VRAM_MIB,
                              exclude_uuid=_gpu_uuid or None,
                              min_compute_cap=_DIARIZE_MIN_COMPUTE_CAP, tag="Diarize")
    if selected is None and _gpu_uuid:
        # No second card with room — sharing Whisper's GPU is still better
        # than failing outright, as long as it has enough headroom left.
        print("[Diarize] No separate GPU free — retrying on Whisper's card", flush=True)
        selected = _find_best_gpu(min_vram_mib=_DIARIZE_MIN_VRAM_MIB,
                                  min_compute_cap=_DIARIZE_MIN_COMPUTE_CAP, tag="Diarize")
    if selected is None:
        return False
    gpu_idx, gpu_uuid = selected

    _diarize_device_index = gpu_idx
    _diarize_request_queue = multiprocessing.Queue()
    _diarize_result_queue = multiprocessing.Queue()

    _diarize_process = multiprocessing.Process(
        target=_diarize_worker,
        args=(_diarize_request_queue, _diarize_result_queue, gpu_idx, gpu_uuid,
              DIARIZE_MODEL, _HF_TOKEN),
        daemon=True,
        name=f"diarize-gpu-{gpu_idx}",
    )
    _diarize_process.start()

    try:
        msg = _diarize_result_queue.get(timeout=300)  # first run downloads models
        if msg.get("status") == "ready":
            print(f"[Diarize] Worker ready on GPU {gpu_idx}", flush=True)
            return True
        print(f"[Diarize] Worker failed: {msg.get('error')}", flush=True)
    except Exception:
        print("[Diarize] Worker failed to start (timeout)", flush=True)

    _kill_diarize_worker()
    return False


def _kill_diarize_worker():
    """Kill the diarization process to release its VRAM."""
    global _diarize_process, _diarize_request_queue, _diarize_result_queue, _diarize_device_index

    if _diarize_process is not None:
        print(f"[Diarize] Killing worker (PID {_diarize_process.pid}, "
              f"GPU {_diarize_device_index})", flush=True)
        try:
            _diarize_process.kill()
            _diarize_process.join(timeout=5)
        except Exception:
            pass
        _diarize_process = None

    for q in (_diarize_request_queue, _diarize_result_queue):
        if q is not None:
            try:
                q.close()
            except Exception:
                pass
    _diarize_request_queue = None
    _diarize_result_queue = None
    _diarize_device_index = None


def _reset_diarize_ttl():
    """Reset the diarization auto-kill timer."""
    global _diarize_ttl_timer
    if _DIARIZE_TTL_MINUTES <= 0:
        return
    if _diarize_ttl_timer is not None:
        _diarize_ttl_timer.cancel()
    _diarize_ttl_timer = threading.Timer(_DIARIZE_TTL_MINUTES * 60, _kill_diarize_worker)
    _diarize_ttl_timer.daemon = True
    _diarize_ttl_timer.start()


def _diarize_submit(audio_path: str, num_speakers: int = 0) -> bool:
    """Queue a diarization job — returns immediately so Whisper can run in
    parallel on its own card. Result is picked up by _diarize_collect()."""
    with _diarize_lock:
        if _diarize_process is None or not _diarize_process.is_alive():
            if not _start_diarize_worker():
                return False
        _diarize_request_queue.put({"audio_path": audio_path,
                                    "num_speakers": num_speakers})
    return True


def _diarize_collect(timeout: int) -> dict | None:
    """Wait for the queued diarization result."""
    if _diarize_result_queue is None:
        return None
    try:
        result = _diarize_result_queue.get(timeout=timeout)
        _reset_diarize_ttl()
        return result
    except Exception:
        return {"error": f"Diarization timeout ({timeout}s)"}


def _smooth_speakers(merged: list, min_run: int = 2) -> list:
    """Drop speaker runs shorter than ``min_run`` words.

    Word-level assignment reacts to every flicker in the diarization: a
    single "yeah" landing on the other speaker would split a sentence into
    three blocks. Runs below the threshold are absorbed into the surrounding
    speaker, which keeps real turn changes (always several words) intact.
    """
    if not merged:
        return merged

    # Group into consecutive runs of the same speaker
    runs: list[list] = [[merged[0]]]
    for item in merged[1:]:
        if item[3] == runs[-1][0][3]:
            runs[-1].append(item)
        else:
            runs.append([item])

    for i, run in enumerate(runs):
        # Words the diarization left uncovered (gaps between turns) always
        # join a neighbour — a bare "SPEAKER_?" block would only chop the
        # sentence apart without adding information.
        if run[0][3] is not None and len(run) >= min_run:
            continue
        # Absorb into whichever neighbour exists (prefer the previous one,
        # so a stray word joins the sentence it interrupts).
        neighbour = None
        if i > 0:
            neighbour = runs[i - 1][0][3]
        elif i + 1 < len(runs):
            neighbour = runs[i + 1][0][3]
        if neighbour is not None:
            runs[i] = [(s, e, t, neighbour) for s, e, t, _ in run]

    return [item for run in runs for item in run]


def _merge_speakers(segments: list, turns: list) -> list:
    """Attach a speaker label to each Whisper segment by largest overlap.

    Both lists are time-sorted, so a moving index keeps this linear instead
    of comparing every segment against every turn (a 2.5 h recording has
    thousands of each).
    """
    turns = sorted(turns, key=lambda t: t[0])
    merged = []
    idx = 0
    for start, end, text in segments:
        # Turns that end before this segment starts can never match a later
        # segment either — skip them permanently.
        while idx < len(turns) and turns[idx][1] < start:
            idx += 1
        best_speaker, best_overlap = None, 0.0
        probe = idx
        while probe < len(turns) and turns[probe][0] < end:
            overlap = min(end, turns[probe][1]) - max(start, turns[probe][0])
            if overlap > best_overlap:
                best_overlap, best_speaker = overlap, turns[probe][2]
            probe += 1
        merged.append((start, end, text, best_speaker))
    return merged


def _format_speaker_text(merged: list) -> str:
    """Render merged pieces as speaker-labelled blocks.

    Consecutive pieces of the same speaker are joined into one block, so the
    result reads like a dialogue transcript instead of one label per word.

    Pieces are single words when diarizing (Whisper's word timestamps carry
    their own leading space, so they are concatenated verbatim — inserting
    another space would double every gap).
    """
    blocks: list[str] = []
    current: str | None = None
    buf: list[str] = []

    def flush() -> None:
        if buf:
            blocks.append(f"[{current}] {''.join(buf).strip()}")

    for _start, _end, text, speaker in merged:
        label = speaker or "SPEAKER_?"
        if label != current:
            flush()
            current, buf = label, []
        # Whisper words start with a space; segment fallbacks may not, so
        # add one where it would otherwise glue two pieces together.
        piece = text if (text.startswith(" ") or not buf) else f" {text}"
        buf.append(piece)
    flush()
    return "\n\n".join(blocks)


# ── Web-UI ───────────────────────────────────────────────────

def _log_client(action: str, **details: str) -> None:
    """Who asked for what — the service has no access log, and the caller of a GPU load
    would otherwise stay unknown."""
    fields = " ".join(f"{key}={value}" for key, value in details.items())
    print(f"[Request] {action} {fields} from={request.remote_addr} "
          f"agent={request.user_agent.string!r}", flush=True)


@app.route("/", methods=["GET"])
def index():
    """Web-UI for status, model management, and settings."""
    cpu_loaded = _model_cpu is not None
    gpu_alive = _gpu_process is not None and _gpu_process.is_alive()
    cpu_badge = '<span style="color:#4CAF50">loaded</span>' if cpu_loaded else '<span style="color:#999">idle</span>'
    gpu_badge = '<span style="color:#4CAF50">loaded</span>' if gpu_alive else '<span style="color:#999">idle</span>'

    cpu_model_options = "".join(
        f'<option value="{m}"{" selected" if m == _config["cpu_model"] else ""}>{m}</option>'
        for m in AVAILABLE_MODELS
    )
    gpu_model_options = "".join(
        f'<option value="{m}"{" selected" if m == _config["gpu_model"] else ""}>{m}</option>'
        for m in AVAILABLE_MODELS
    )

    engine_options = "".join(
        f'<option value="{e}"{" selected" if e == _config["engine"] else ""}>{e}</option>'
        for e in ENGINES
    )
    cpu_quality_options = "".join(
        f'<option value="{q}"{" selected" if q == _config["cpu_quality"] else ""}>{q}</option>'
        for q in PARAKEET_QUALITIES
    )
    gpu_quality_options = "".join(
        f'<option value="{q}"{" selected" if q == _config["gpu_quality"] else ""}>{q}</option>'
        for q in PARAKEET_QUALITIES
    )

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Whisper STT</title>
<style>
body {{ font-family: sans-serif; background: #1a1a1a; color: #fff; max-width: 640px; margin: 40px auto; padding: 20px; }}
h1 {{ font-size: 20px; margin-bottom: 4px; }}
h2 {{ font-size: 14px; color: #FFD700; margin: 16px 0 8px 0; }}
.card {{ background: #252525; border-radius: 8px; padding: 16px; margin: 8px 0; }}
.row {{ display: flex; justify-content: space-between; align-items: center; margin: 6px 0; }}
label {{ color: #aaa; font-size: 13px; }}
select, input[type=number] {{ background: #333; color: #fff; border: 1px solid #555; border-radius: 4px; padding: 4px 8px; font-size: 13px; }}
select {{ width: 140px; }}
input[type=number] {{ width: 70px; text-align: right; }}
.toggle {{ position: relative; width: 40px; height: 22px; }}
.toggle input {{ opacity: 0; width: 0; height: 0; }}
.toggle .slider {{ position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0; background: #555; border-radius: 11px; transition: 0.2s; }}
.toggle .slider:before {{ content: ""; position: absolute; height: 16px; width: 16px; left: 3px; bottom: 3px; background: #fff; border-radius: 50%; transition: 0.2s; }}
.toggle input:checked + .slider {{ background: #4CAF50; }}
.toggle input:checked + .slider:before {{ transform: translateX(18px); }}
.btn {{ padding: 8px 14px; border: none; border-radius: 6px; cursor: pointer; font-weight: 600; font-size: 12px; margin: 2px; }}
.btn:hover {{ opacity: 0.85; }}
.btn-load {{ background: #66bb6a; color: #fff; }}
.btn-unload {{ background: #ff6f00; color: #fff; }}
.btn-save {{ background: #42a5f5; color: #fff; }}
.btn-row {{ display: flex; flex-wrap: wrap; gap: 4px; justify-content: center; margin-top: 8px; }}
#msg {{ margin-top: 12px; padding: 8px 12px; border-radius: 6px; display: none; font-size: 13px; }}
</style></head><body>
<h1>Whisper STT</h1>
<p style="color:#888; font-size:12px; margin-top:0;">faster-whisper Docker Service</p>

<div class="card">
  <div class="row"><label>CPU Model:</label> {cpu_badge}</div>
  <div class="row"><label>GPU Worker:</label> {gpu_badge}{f' (GPU {_gpu_device_index})' if gpu_alive and _gpu_device_index is not None else ''}</div>
  <div class="btn-row">
    <button class="btn btn-load" onclick="load('cpu')">Load CPU</button>
    <button class="btn btn-unload" onclick="unload('cpu')">Unload CPU</button>
    <button class="btn btn-load" onclick="load('cuda')">Load GPU</button>
    <button class="btn btn-unload" onclick="unload('cuda')">Unload GPU</button>
  </div>
</div>

<h2>Models</h2>
<div class="card">
  <div class="row"><label>Engine</label> <select id="cfg-engine">{engine_options}</select></div>
  <p style="color:#666; font-size:11px; margin:8px 0 0 0;">Whisper uses the models below, Parakeet has one model with a quality per device.</p>
</div>
<div style="display:flex; gap:12px;">
  <div class="card" style="flex:1;">
    <div style="font-size:13px; font-weight:600; color:#4CAF50; margin-bottom:8px;">CPU Engine</div>
    <div class="row"><label>Whisper model</label> <select id="cfg-cpu-model">{cpu_model_options}</select></div>
    <div class="row"><label>Parakeet quality</label> <select id="cfg-cpu-quality">{cpu_quality_options}</select></div>
  </div>
  <div class="card" style="flex:1;">
    <div style="font-size:13px; font-weight:600; color:#FF9800; margin-bottom:8px;">GPU Engine</div>
    <div class="row"><label>Whisper model</label> <select id="cfg-gpu-model">{gpu_model_options}</select></div>
    <div class="row"><label>Parakeet quality</label> <select id="cfg-gpu-quality">{gpu_quality_options}</select></div>
    <div class="row"><label>TTL (min)</label> <input type="number" id="cfg-ttl" value="{_config["gpu_ttl_minutes"]}" min="0" max="1440"></div>
  </div>
</div>

<h2>Transcription</h2>
<div class="card">
  <div class="row"><label>Beam Size</label> <input type="number" id="cfg-beam" value="{_config["beam_size"]}" min="1" max="20"></div>
  <div class="row"><label>Default Language</label>
    <select id="cfg-lang">
      <option value="de"{"" if _config["language"] != "de" else " selected"}>Deutsch</option>
      <option value="en"{"" if _config["language"] != "en" else " selected"}>English</option>
      <option value="auto"{"" if _config["language"] != "auto" else " selected"}>Auto-Detect</option>
    </select>
  </div>
  <div class="row"><label>VAD Filter</label>
    <label class="toggle"><input type="checkbox" id="cfg-vad" {"checked" if _config["vad_filter"] else ""}><span class="slider"></span></label>
  </div>
  <div class="row"><label>Condition on Previous</label>
    <label class="toggle"><input type="checkbox" id="cfg-cond" {"checked" if _config["condition_on_previous_text"] else ""}><span class="slider"></span></label>
  </div>
  <div class="row"><label>Initial Prompt (auto/leer/Text)</label> <input type="text" id="cfg-prompt" value="{_config["initial_prompt"]}" style="width:220px"></div>
  <div class="row"><label>Sprecheranzahl (0 = auto)</label> <input type="number" id="cfg-numspk" value="{_config["num_speakers"]}" min="0" max="10"></div>
  <div class="btn-row">
    <button class="btn btn-save" onclick="saveConfig()">Save Settings</button>
  </div>
  <p style="color:#666; font-size:11px; margin:8px 0 0 0;">Model change requires unload + reload to take effect.</p>
</div>

<div id="msg"></div>

<script>
async function load(device) {{
  msg('Loading ' + device + '...', '#01579b');
  const fd = new FormData();
  const wav = new Uint8Array([0x52,0x49,0x46,0x46,0x24,0,0,0,0x57,0x41,0x56,0x45,0x66,0x6D,0x74,0x20,
    0x10,0,0,0,1,0,1,0,0x80,0x3E,0,0,0,0x7D,0,0,2,0,0x10,0,0x64,0x61,0x74,0x61,0,0,0,0]);
  fd.append('file', new Blob([wav], {{type:'audio/wav'}}), 'load.wav');
  fd.append('device', device);
  fd.append('language', 'de');
  try {{
    const r = await fetch('/transcribe', {{method:'POST', body:fd}});
    if (r.ok) {{ msg(device.toUpperCase() + ' loaded', '#1b5e20'); setTimeout(()=>location.reload(), 800); }}
    else {{ msg('Error: ' + (await r.json()).error, '#b71c1c'); }}
  }} catch(e) {{ msg('Connection error', '#b71c1c'); }}
}}
async function unload(device) {{
  try {{
    const r = await fetch('/unload?device=' + device, {{method:'POST'}});
    if (r.ok) {{ msg(device.toUpperCase() + ' unloaded', '#1b5e20'); setTimeout(()=>location.reload(), 500); }}
    else {{ msg('Error', '#b71c1c'); }}
  }} catch(e) {{ msg('Connection error', '#b71c1c'); }}
}}
async function saveConfig() {{
  const cfg = {{
    engine: document.getElementById('cfg-engine').value,
    cpu_quality: document.getElementById('cfg-cpu-quality').value,
    gpu_quality: document.getElementById('cfg-gpu-quality').value,
    cpu_model: document.getElementById('cfg-cpu-model').value,
    gpu_model: document.getElementById('cfg-gpu-model').value,
    gpu_ttl_minutes: parseInt(document.getElementById('cfg-ttl').value) || 30,
    beam_size: parseInt(document.getElementById('cfg-beam').value) || 5,
    language: document.getElementById('cfg-lang').value,
    vad_filter: document.getElementById('cfg-vad').checked,
    condition_on_previous_text: document.getElementById('cfg-cond').checked,
    initial_prompt: document.getElementById('cfg-prompt').value,
    num_speakers: parseInt(document.getElementById('cfg-numspk').value) || 0,
  }};
  try {{
    const r = await fetch('/config', {{method:'POST', headers:{{'Content-Type':'application/json'}}, body:JSON.stringify(cfg)}});
    if (r.ok) {{ msg('Settings saved', '#1b5e20'); setTimeout(()=>location.reload(), 500); }}
    else {{ msg('Error saving', '#b71c1c'); }}
  }} catch(e) {{ msg('Connection error', '#b71c1c'); }}
}}
function msg(text, bg) {{
  const el = document.getElementById('msg');
  el.textContent = text; el.style.display = 'block'; el.style.background = bg;
  setTimeout(() => {{ el.style.display = 'none'; }}, 4000);
}}
</script>
</body></html>"""


# ── API Endpoints ────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """Health check — reports model status per device."""
    cpu_loaded = _model_cpu is not None or _parakeet_cpu is not None
    gpu_alive = _gpu_process is not None and _gpu_process.is_alive()
    if cpu_loaded or gpu_alive:
        status, model_loaded = "ok", True
    elif EAGER_LOAD:
        status, model_loaded = "loading", False
    else:
        status, model_loaded = "idle", False

    return jsonify({
        "status": status,
        "model_loaded": model_loaded,
        "cpu_loaded": cpu_loaded,
        "cpu_model": _cpu_model_name if _model_cpu is not None else _config["cpu_model"],
        "gpu_loaded": gpu_alive,
        "gpu_model": _gpu_model_name if gpu_alive else _config["gpu_model"],
        "gpu_device_index": _gpu_device_index,
        "gpu_ttl_minutes": _config["gpu_ttl_minutes"],
        "engine": _config["engine"],
        "gpu_engine": _gpu_engine if gpu_alive else None,
    })


@app.route("/transcribe", methods=["POST"])
def transcribe():
    """Transcribe audio file.

    Form data:
        file:         Audio file (WAV, MP3, M4A, OGG, FLAC, WebM)
        device:       "cpu" or "cuda" (default: "cpu")
        language:     Language code, e.g. "de", "en" (default: from config)
        diarize:      "1" to label speakers (default: off — pointless for
                      short voice commands, only worth it for interviews)
        num_speakers: Optional hint if the speaker count is known
        engine:       "whisper" or "parakeet" (default: STT_ENGINE)
        quality:      Parakeet only — "fp32" or "int8" for this request
                      (default: STT_CPU_QUALITY / STT_GPU_QUALITY, by device)
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    audio_file = request.files["file"]
    device = request.form.get("device", "cpu")
    language = request.form.get("language", _config["language"])
    diarize = request.form.get("diarize", "0") in ("1", "true", "True")
    try:
        num_speakers = int(request.form.get("num_speakers", "0"))
    except ValueError:
        num_speakers = 0
    # Request wins; otherwise the configured default (UI: "Sprecheranzahl").
    if num_speakers <= 0:
        num_speakers = _config["num_speakers"]

    if device not in ("cpu", "cuda"):
        return jsonify({"error": f"Invalid device: {device}. Use 'cpu' or 'cuda'"}), 400
    engine = request.form.get("engine", _config["engine"])
    quality = request.form.get("quality", _config[f"{'gpu' if device == 'cuda' else 'cpu'}_quality"])
    if engine not in ENGINES:
        return jsonify({"error": f"Invalid engine: {engine}. Use one of {ENGINES}"}), 400
    if quality not in PARAKEET_QUALITIES:
        return jsonify({"error": f"Invalid quality: {quality}. Use one of {PARAKEET_QUALITIES}"}), 400

    _log_client("transcribe", device=device, engine=engine, diarize=str(diarize))

    suffix = Path(audio_file.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        audio_file.save(tmp)
        tmp_path = tmp.name

    work_path = tmp_path
    try:
        # With diarization both engines need the same 16 kHz mono audio —
        # decode it once here instead of twice in parallel (SSOT for the
        # decoded audio). Without diarization Whisper decodes internally as
        # before, so nothing changes for the common short-command case.
        # Parakeet (onnx-asr) reads only WAV, so it always gets the decoded file.
        if diarize or engine == "parakeet":
            work_path = _decode_to_wav16k(tmp_path)
            if engine == "parakeet" and work_path == tmp_path:
                return jsonify({"error": "Audio could not be decoded to 16 kHz WAV"}), 500

        # Queue diarization FIRST so it runs on its own card while Whisper
        # transcribes — the two jobs never touch, so the wall-clock cost of
        # speaker labels is close to zero. It needs the decoded WAV (the
        # worker reads PCM directly instead of using pyannote's decoder), so
        # a failed conversion means no speaker labels — never a broken run.
        wav_ready = diarize and work_path != tmp_path
        diarize_queued = _diarize_submit(work_path, num_speakers) if wav_ready else False
        if diarize and not diarize_queued:
            print("[Diarize] Could not start — returning plain transcript", flush=True)

        if device == "cuda":
            result = _transcribe_gpu(work_path, language, return_segments=diarize_queued,
                                     engine=engine, quality=quality)
        elif engine == "parakeet":
            result = _transcribe_parakeet_cpu(work_path, language, quality,
                                              return_segments=diarize_queued)
        else:
            result = _transcribe_cpu(work_path, language, return_segments=diarize_queued)

        if result is None:
            # 503 (not 500): tells the client "GPU temporarily unavailable,
            # CPU fallback is a valid option" — distinct from real errors.
            if diarize_queued:
                _diarize_collect(timeout=10)  # drain, worker already ran
            return jsonify({"error": "Failed to start GPU worker — no GPU with enough VRAM"}), 503
        if "error" in result:
            return jsonify({"error": result["error"]}), 500

        if diarize_queued:
            dia = _diarize_collect(timeout=_GPU_RESULT_TIMEOUT_S)
            segments = result.pop("segments", None)
            if dia and "turns" in dia and segments:
                merged = _smooth_speakers(_merge_speakers(segments, dia["turns"]))
                result["text_speakers"] = _format_speaker_text(merged)
                result["speakers"] = dia["speakers"]
                result["diarize_time"] = dia["time"]
            else:
                # Diarization failed — the plain transcript is still valid,
                # so report the reason instead of failing the whole request.
                result["diarize_error"] = (dia or {}).get("error", "no diarization result")
        result.pop("segments", None)

        return jsonify(result)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
        if work_path != tmp_path:
            Path(work_path).unlink(missing_ok=True)


@app.route("/status", methods=["GET"])
def status():
    """Detailed status including memory usage and full config."""
    gpu_alive = _gpu_process is not None and _gpu_process.is_alive()
    result = {
        **_config,
        "cpu_loaded": _model_cpu is not None,
        "parakeet_cpu_loaded": _parakeet_cpu_quality if _parakeet_cpu is not None else None,
        "gpu_loaded": gpu_alive,
        "gpu_engine": _gpu_engine if gpu_alive else None,
        "gpu_busy": _gpu_busy,
        "gpu_device_index": _gpu_device_index,
        "gpu_worker_pid": _gpu_process.pid if gpu_alive else None,
        # Actually loaded model — differs from gpu_model when the VRAM
        # degradation chain picked a smaller one.
        "gpu_model_loaded": _gpu_model_name if gpu_alive else None,
        # SSOT for clients building a model dropdown (AIfred STT tab).
        "available_models": AVAILABLE_MODELS,
        "engines": ENGINES,
        "parakeet_qualities": PARAKEET_QUALITIES,
    }

    if _last_gpu_request > 0:
        idle = time.time() - _last_gpu_request
        result["gpu_idle_seconds"] = round(idle, 0)
        if _config["gpu_ttl_minutes"] > 0:
            result["gpu_ttl_remaining_seconds"] = round(
                max(0, _config["gpu_ttl_minutes"] * 60 - idle), 0)

    return jsonify(result)


@app.route("/unload", methods=["POST"])
def unload():
    """Unload model(s). Query param device: cpu, gpu, cuda, or all (default)."""
    global _model_cpu, _parakeet_cpu, _parakeet_cpu_quality
    device = request.args.get("device", "all")
    unloaded = []
    _log_client("unload", device=device, force=request.args.get("force", "0"))

    if device in ("gpu", "cuda", "all"):
        # A transcription in flight is not killed silently — callers get a
        # 409 and may retry (or pass force=1 to kill regardless, e.g. after
        # a patience deadline before an LLM cold start).
        if _gpu_busy and request.args.get("force") != "1":
            return jsonify({"success": False, "busy": True}), 409
        # Report "gpu" only when a worker was actually alive — callers use
        # this to log an honest "released before model load" message.
        gpu_was_alive = _gpu_process is not None and _gpu_process.is_alive()
        _kill_gpu_worker()
        if gpu_was_alive:
            unloaded.append("gpu")
        # The diarization worker sits on a second card and would otherwise
        # keep computing for a caller that is already gone (e.g. AIfred was
        # restarted mid-transcription) — free it in the same sweep.
        diarize_was_alive = _diarize_process is not None and _diarize_process.is_alive()
        _kill_diarize_worker()
        if diarize_was_alive:
            unloaded.append("diarize")
    if device in ("cpu", "all") and _model_cpu is not None:
        _model_cpu = None
        gc.collect()
        unloaded.append("cpu")
    if device in ("cpu", "all") and _parakeet_cpu is not None:
        _parakeet_cpu = None
        _parakeet_cpu_quality = ""
        gc.collect()
        unloaded.append("parakeet-cpu")

    return jsonify({"success": True, "unloaded": unloaded})


@app.route("/config", methods=["GET", "POST"])
def config_endpoint():
    """Get or update runtime configuration."""
    if request.method == "GET":
        return jsonify({**_config, "available_models": AVAILABLE_MODELS,
                        "engines": ENGINES, "parakeet_qualities": PARAKEET_QUALITIES})

    data = request.get_json(silent=True) or {}
    changed = []

    # Per-device model selection
    if "cpu_model" in data and data["cpu_model"] in AVAILABLE_MODELS:
        _config["cpu_model"] = data["cpu_model"]
        changed.append("cpu_model")
    if "gpu_model" in data and data["gpu_model"] in AVAILABLE_MODELS:
        _config["gpu_model"] = data["gpu_model"]
        changed.append("gpu_model")
    # Legacy: "model" sets both
    if "model" in data and data["model"] in AVAILABLE_MODELS:
        _config["cpu_model"] = data["model"]
        _config["gpu_model"] = data["model"]
        changed.append("model (both)")
    if "gpu_ttl_minutes" in data:
        _config["gpu_ttl_minutes"] = max(0, int(data["gpu_ttl_minutes"]))
        changed.append("gpu_ttl_minutes")
    if "beam_size" in data:
        _config["beam_size"] = max(1, min(20, int(data["beam_size"])))
        changed.append("beam_size")
    if "language" in data:
        _config["language"] = data["language"]
        changed.append("language")
    if "vad_filter" in data:
        _config["vad_filter"] = bool(data["vad_filter"])
        changed.append("vad_filter")
    if "condition_on_previous_text" in data:
        _config["condition_on_previous_text"] = bool(data["condition_on_previous_text"])
        changed.append("condition_on_previous_text")
    if "initial_prompt" in data:
        _config["initial_prompt"] = str(data["initial_prompt"])
        changed.append("initial_prompt")
    if "num_speakers" in data:
        _config["num_speakers"] = max(0, min(10, int(data["num_speakers"])))
        changed.append("num_speakers")
    if data.get("engine") in ENGINES:
        _config["engine"] = data["engine"]
        changed.append("engine")
    for key in ("cpu_quality", "gpu_quality"):
        if data.get(key) in PARAKEET_QUALITIES:
            _config[key] = data[key]
            changed.append(key)

    print(f"[Whisper] Config updated: {', '.join(changed)}", flush=True)
    return jsonify({"success": True, "changed": changed, "config": _config})


# ── Startup ──────────────────────────────────────────────────

if (_config["engine"] not in ENGINES or _config["cpu_quality"] not in PARAKEET_QUALITIES
        or _config["gpu_quality"] not in PARAKEET_QUALITIES):
    raise SystemExit(f"STT_ENGINE must be one of {ENGINES}, STT_CPU_QUALITY and STT_GPU_QUALITY one of "
                     f"{PARAKEET_QUALITIES} (got {_config['engine']!r}, {_config['cpu_quality']!r}, "
                     f"{_config['gpu_quality']!r})")

if EAGER_LOAD:
    def _eager_load():
        global _parakeet_cpu, _parakeet_cpu_quality
        time.sleep(2)
        # Only the default engine stays loaded from the start; the other one
        # loads on its first request.
        if _config["engine"] == "parakeet":
            with _parakeet_cpu_lock:
                _parakeet_cpu = _load_parakeet(_config["cpu_quality"], "CPUExecutionProvider")
                _parakeet_cpu_quality = _config["cpu_quality"]
        else:
            _load_cpu_model()
    threading.Thread(target=_eager_load, daemon=True).start()

print(f'[Whisper] Server starting — engine={_config["engine"]}, '
      f'cpu={_config["cpu_model"]}/{_config["cpu_quality"]}, gpu={_config["gpu_model"]}/{_config["gpu_quality"]}, '
      f'eager_load={EAGER_LOAD}, gpu_ttl={_config["gpu_ttl_minutes"]}min', flush=True)
