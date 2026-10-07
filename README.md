# whisper-stt

Speech-to-text as a small HTTP service in Docker, built on
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) and NVIDIA
[Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3). One running container serves
every client on the machine — [AIfred Intelligence](https://github.com/Peuqui/AIfred-Intelligence)
and [Agent-Orc](https://github.com/Peuqui/Agent-Orc) use it for voice input.

[Deutsche Version](README.de.md)

- **Two engines, chosen per request:** Whisper or Parakeet (see below).
- **CPU and GPU, chosen per request.** The CPU model stays loaded all the time (no load time,
  no VRAM). The GPU model is loaded on demand into the card with the most free VRAM and
  released again after a configurable idle time.
- **No silent device switch.** If no GPU has room, `/transcribe` answers `503`; the client
  decides whether to use the CPU instead.
- **VRAM-aware model choice.** The GPU tries the configured model (default `large-v3`) and
  steps down to smaller ones when VRAM is short, but never below `WHISPER_GPU_MIN_MODEL`
  (default `medium`; smaller models transcribe too poorly to be worth it).
- **Optional speaker diarization** with pyannote, on a second GPU, per request.
- A small web page at `/` shows the status and changes the runtime configuration.

## Requirements

- Docker with Compose
- For GPU use: an NVIDIA GPU and the NVIDIA Container Toolkit
- For speaker diarization only: a Hugging Face token with access to the pyannote models,
  stored in `~/.cache/huggingface/token` (for example via `huggingface-cli login`)

The compose file mounts `~/.cache/huggingface` read-only and reads the token from there.
Without a token file everything except diarization works as usual; add the token any time
and restart the container.

## Start

```
git clone https://github.com/Peuqui/whisper-stt.git && cd whisper-stt
docker compose up -d --build
curl http://localhost:5080/health
```

The service listens on port **5080**. Models are cached in the Docker volume `whisper_models`
and survive rebuilds. The container restarts with Docker (`restart: unless-stopped`).

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/transcribe` | Transcribe an audio file (multipart form, see below) |
| GET | `/health` | Readiness: `status`, `model_loaded`, loaded models per device |
| GET | `/status` | Detailed status and the full runtime configuration |
| GET/POST | `/config` | Read or change the runtime configuration (JSON) |
| POST | `/unload?device=cpu\|gpu\|cuda\|all` | Unload models; `409` while a GPU transcription runs, `force=1` kills it anyway |
| GET | `/` | Web page with status and configuration |

`POST /transcribe` takes a multipart form:

| Field | Meaning |
|---|---|
| `file` | Audio file (WAV, MP3, M4A, OGG, FLAC, WebM) |
| `device` | `cpu` (default) or `cuda` |
| `language` | Language code such as `de` or `en`; default from the configuration |
| `diarize` | `1` to label speakers (default off) |
| `num_speakers` | Optional hint when the number of speakers is known |
| `engine` | `whisper` or `parakeet`; default `STT_ENGINE` |
| `quality` | Parakeet only: `fp32` or `int8`; default `STT_CPU_QUALITY` or `STT_GPU_QUALITY`, by device |

The answer is JSON with `text`, the time taken and the device used. Errors: `400` for a
missing file or an invalid device, `503` when no GPU has room for a model, `500` otherwise.

```
curl -F file=@speech.webm -F device=cuda http://localhost:5080/transcribe
```

## Configuration

Set in `docker-compose.yml` or as environment variables when starting compose:

| Variable | Default | Meaning |
|---|---|---|
| `WHISPER_MODEL` | `medium` | Model on the CPU (and the GPU default when `WHISPER_GPU_MODEL` is unset) |
| `WHISPER_CPU_MODEL` | `WHISPER_MODEL` | Model on the CPU |
| `WHISPER_GPU_MODEL` | `large-v3` | First choice on the GPU |
| `WHISPER_GPU_MIN_MODEL` | `medium` | Smallest model the GPU steps down to |
| `WHISPER_GPU_TTL_MINUTES` | `30` | Idle minutes until the GPU worker is stopped and its VRAM freed; `0` keeps it |
| `WHISPER_LANGUAGE` | `de` | Default language (`auto` detects it) |
| `WHISPER_EAGER_LOAD` | `1` | Load the CPU model at start |
| `WHISPER_CPU_COMPUTE` / `WHISPER_GPU_COMPUTE` | `int8` / `float16` | Compute types |
| `DIARIZE_MODEL` | `pyannote/speaker-diarization-community-1` | Diarization pipeline |
| `STT_ENGINE` | `whisper` | Engine when a request names none: `whisper` or `parakeet` |
| `STT_CPU_QUALITY` / `STT_GPU_QUALITY` | `fp32` / `fp32` | Parakeet quality on the CPU / the GPU when a request names none: `fp32` or `int8` |
| `PARAKEET_CHUNK_S` / `PARAKEET_MERGE_SILENCE_MS` | `60` / `5000` | Parakeet cuts audio at speech pauses into chunks of at most this length, merging pauses shorter than this |
| `PARAKEET_MIN_VRAM_MIB` | `4500` | Free VRAM a card needs for Parakeet |

Available models: `tiny`, `base`, `small`, `medium`, `large-v3`.

## Choosing an engine

Measured on four one-minute excerpts of German audiobooks (different narrators); seconds per
minute of audio, lower is better:

| Engine | Device | Time (s) |
|---|---|---|
| Parakeet fp32 | GPU (V100) | 0.25 |
| Whisper large-v3 | GPU | 2.0–3.7 |
| Parakeet int8 | CPU | 2.5–3.8 |
| Parakeet fp32 | CPU | 2.8–5.1 |
| Whisper medium | CPU | 12.5–22.7 |

Quality, read by hand: Whisper large-v3 is best on names and grammar, but on one excerpt it
(and Whisper medium) dropped half the text. Parakeet fp32 came close to large-v3 and never
dropped anything; int8 makes more word errors (about Whisper medium's level). Parakeet covers
25 European languages and detects the language itself — for other languages use Whisper.

Parakeet needs no Hugging Face token; its model (≈3 GB, both qualities) is downloaded into
the model volume on first use.

## License

[PolyForm Noncommercial 1.0.0](LICENSE)
