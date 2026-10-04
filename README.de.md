# whisper-stt

Spracherkennung als kleiner HTTP-Dienst in Docker, auf Basis von
[faster-whisper](https://github.com/SYSTRAN/faster-whisper). Ein laufender Container bedient alle
Programme auf dem Rechner – [AIfred Intelligence](https://github.com/Peuqui/AIfred-Intelligence)
und [Agent-Orc](https://github.com/Peuqui/Agent-Orc) nutzen ihn für die Spracheingabe.

[English version](README.md)

- **CPU und GPU, pro Anfrage gewählt.** Das CPU-Modell bleibt dauerhaft geladen (keine
  Ladezeit, kein VRAM). Das GPU-Modell wird bei Bedarf auf die Karte mit dem meisten freien
  VRAM geladen und nach einer einstellbaren Leerlaufzeit wieder freigegeben.
- **Kein stiller Gerätewechsel.** Hat keine GPU Platz, antwortet `/transcribe` mit `503`; der
  Aufrufer entscheidet, ob er stattdessen die CPU nimmt.
- **Modellwahl nach freiem VRAM.** Die GPU versucht das eingestellte Modell (Standard
  `large-v3`) und steigt bei knappem VRAM auf kleinere ab, aber nie unter
  `WHISPER_GPU_MIN_MODEL` (Standard `medium`; kleinere Modelle erkennen zu schlecht).
- **Optionale Sprechererkennung** mit pyannote, auf einer zweiten GPU, pro Anfrage.
- Eine kleine Webseite unter `/` zeigt den Status und ändert die Laufzeit-Konfiguration.

## Voraussetzungen

- Docker mit Compose
- Für die GPU: eine NVIDIA-Karte und das NVIDIA Container Toolkit
- Nur für die Sprechererkennung: ein Hugging-Face-Token mit Zugriff auf die pyannote-Modelle,
  abgelegt in `~/.cache/huggingface/token` (z. B. per `huggingface-cli login`)

Die Compose-Datei bindet `~/.cache/huggingface/token` in den Container ein. Fehlt die Datei,
legt Docker an ihrer Stelle einen leeren Ordner an; für die Sprechererkennung also vorher die
Token-Datei anlegen.

## Start

```
git clone https://github.com/Peuqui/whisper-stt.git && cd whisper-stt
docker compose up -d --build
curl http://localhost:5080/health
```

Der Dienst hört auf Port **5080**. Die Modelle liegen im Docker-Volume `whisper_models` und
überstehen einen Neubau. Der Container startet mit Docker neu (`restart: unless-stopped`).

## API

| Methode | Pfad | Zweck |
|---|---|---|
| POST | `/transcribe` | Audiodatei transkribieren (Multipart-Formular, siehe unten) |
| GET | `/health` | Bereitschaft: `status`, `model_loaded`, geladene Modelle je Gerät |
| GET | `/status` | Ausführlicher Status und die vollständige Laufzeit-Konfiguration |
| GET/POST | `/config` | Laufzeit-Konfiguration lesen oder ändern (JSON) |
| POST | `/unload?device=cpu\|gpu\|cuda\|all` | Modelle entladen; `409`, solange eine GPU-Transkription läuft, `force=1` beendet sie trotzdem |
| GET | `/` | Webseite mit Status und Konfiguration |

`POST /transcribe` erwartet ein Multipart-Formular:

| Feld | Bedeutung |
|---|---|
| `file` | Audiodatei (WAV, MP3, M4A, OGG, FLAC, WebM) |
| `device` | `cpu` (Standard) oder `cuda` |
| `language` | Sprachcode wie `de` oder `en`; Standard aus der Konfiguration |
| `diarize` | `1`, um Sprecher zu kennzeichnen (Standard: aus) |
| `num_speakers` | Optionaler Hinweis, wenn die Sprecherzahl bekannt ist |

Die Antwort ist JSON mit `text`, der benötigten Zeit und dem verwendeten Gerät. Fehler: `400`
bei fehlender Datei oder ungültigem Gerät, `503`, wenn keine GPU Platz für ein Modell hat,
sonst `500`.

```
curl -F file=@sprache.webm -F device=cuda http://localhost:5080/transcribe
```

## Konfiguration

In `docker-compose.yml` oder als Umgebungsvariablen beim Start von Compose:

| Variable | Standard | Bedeutung |
|---|---|---|
| `WHISPER_MODEL` | `medium` | Modell auf der CPU (und GPU-Standard, wenn `WHISPER_GPU_MODEL` fehlt) |
| `WHISPER_CPU_MODEL` | `WHISPER_MODEL` | Modell auf der CPU |
| `WHISPER_GPU_MODEL` | `large-v3` | Erste Wahl auf der GPU |
| `WHISPER_GPU_MIN_MODEL` | `medium` | Kleinstes Modell, auf das die GPU absteigt |
| `WHISPER_GPU_TTL_MINUTES` | `30` | Leerlauf-Minuten, bis der GPU-Worker endet und sein VRAM frei wird; `0` behält ihn |
| `WHISPER_LANGUAGE` | `de` | Standardsprache (`auto` erkennt sie) |
| `WHISPER_EAGER_LOAD` | `1` | CPU-Modell beim Start laden |
| `WHISPER_CPU_COMPUTE` / `WHISPER_GPU_COMPUTE` | `int8` / `float16` | Rechengenauigkeit |
| `DIARIZE_MODEL` | `pyannote/speaker-diarization-community-1` | Pipeline der Sprechererkennung |

Verfügbare Modelle: `tiny`, `base`, `small`, `medium`, `large-v3`.

## Lizenz

[PolyForm Noncommercial 1.0.0](LICENSE)
