# host-assistant

Demo / starting-point code for a **local LAN voice host** that lets
[portal-assistant](https://github.com/rudysev/portal-assistant) run against a machine on your LAN
instead of the cloud Gemini Live API. Point the app at **Settings → Backend → Local server** and
exercise the full voice + tool-calling path. Not a shipped product — use it to demo, test, or fork.

It re-creates what Gemini's server did for the app: VAD, STT, LLM (with tools), TTS, and web-search
grounding, behind the WebSocket the app already speaks.

---

## Pipeline

```
Portal (Android) ── 16 kHz PCM (binary over wss) ──►  host_assistant (LAN host)
                 ◄─ 24 kHz PCM + JSON control ──

  transport.input (TLS)
    → Whisper-MLX (STT)
    → InputTranscriptEmitter
    → Silero VAD (turn end)
    → Qwen3.5 via Ollama (LLM + tools)
    → Kokoro (TTS; markdown + emoji stripped)
    → ControlBridge → transport.output
```

1. **VAD** ends the user turn → **Whisper** transcribes.
2. **Ollama** runs the LLM with portal-assistant tools from `setup`, plus host-only `get_weather`
   (Open-Meteo) and `web_search` (Tavily or DuckDuckGo).
3. **Kokoro** streams 24 kHz speech back; non-host tool calls round-trip to the app via
   `tool_call` / `tool_result`.

Wire: one `wss://` socket. Binary = raw PCM; text = JSON control (`setup`, `user_text`,
`tool_result`, `ready`, `input_transcript`, `output_transcript`, `model_generating`,
`turn_complete`, `interrupted`, `tool_call`, `error`). Host tools are an allowlist
(`web_search`, `get_weather`); everything else runs on the Portal.

TLS: self-signed cert under `~/.host-assistant/tls/` on first boot (`openssl` required). The app
uses trust-all for this LAN demo — not production PKI.

---

## Frameworks

| Layer | Stack |
|---|---|
| Orchestration | [Pipecat](https://github.com/pipecat-ai/pipecat) 1.5.0 |
| LLM | Qwen3.5 via [Ollama](https://ollama.com) (OpenAI-compatible API) |
| ASR | Whisper on MLX (`WhisperSTTServiceMLX`) — **Apple Silicon today** |
| VAD | Silero |
| TTS | [Kokoro](https://github.com/thewh1teagle/kokoro-onnx) |
| Weather | [Open-Meteo](https://open-meteo.com) (no key) |
| Web search | [Tavily](https://tavily.com) or duckduckgo-search |
| Config | python-dotenv |

Runtime: **Python 3.12**. Everything except STT is portable; Whisper here uses **MLX** (Apple
Silicon). On Linux, swap `WhisperSTTServiceMLX` for a non-MLX Whisper backend (Pipecat’s
faster-whisper path is the natural drop-in).

---

## Requirements

- Python 3.12 (ML wheels lag on 3.13+)
- [Ollama](https://ollama.com) running
- `openssl` (for the self-signed TLS cert)
- **Apple Silicon** if you keep the default MLX Whisper STT (or replace STT for Linux)
- `TAVILY_API_KEY` only if `WEB_SEARCH_BACKEND=tavily` (default); use `duckduckgo` to skip
- `ffmpeg` only for `scripts/test_client.py`

---

## Install

```bash
ollama pull qwen3.5:9b
OLLAMA_KEEP_ALIVE=-1 ollama serve   # terminal 1

cd host-assistant
python3.12 -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -r requirements.txt
cp .env.example .env
# edit TAVILY_API_KEY, or set WEB_SEARCH_BACKEND=duckduckgo
```

`.env` is git-ignored. Useful knobs (defaults in `.env.example`): `HOST`, `PORT`, `OLLAMA_MODEL`,
`OLLAMA_MAX_TOKENS`, `WEB_SEARCH_BACKEND`, `WHISPER_MODEL`, `KOKORO_VOICE`, `VAD_STOP_SECS`,
warm-up flags.

---

## Run

```bash
source .venv/bin/activate
python -m host_assistant
# → Local voice host listening on wss://0.0.0.0:8080 …
```

On the Portal: **Settings → Backend → Local server (LAN)** → `<host-lan-ip>:8080`.

One WebSocket client at a time. Same Wi-Fi; allow inbound TCP 8080 on the host firewall.

### Verify without the Portal

```bash
python -m scripts.smoke_test          # typed text, no mic
python -m unittest discover -s tests -v
```

Optional audio round-trip (`ffmpeg`; any 16 kHz mono WAV works as input):

```bash
ffmpeg -y -i your-speech.wav -ac 1 -ar 16000 sample.wav
python -m scripts.test_client sample.wav
ffmpeg -y -f s16le -ar 24000 -ac 1 -i reply.pcm reply.wav
```

---

## Layout

```
host_assistant/     # python -m host_assistant
  __main__.py       # Pipecat pipeline + TLS WebSocket
  config.py         # .env
  protocol.py       # JSON wire helpers
  pipeline/         # session, bridge, serializer, input transcript
  transport/        # tls + wss://
  tools/            # get_weather, web_search
  text/             # emoji / TTS filters
  warmup/           # Ollama, Kokoro, Whisper
scripts/            # smoke_test, test_client, benchmark_weather
tests/
```

---

## Notes

- **Latency:** STT/TTS ~1 s warm; LLM often 10–20 s on a tool turn (two passes). Prefer `qwen3.5:9b`,
  `OLLAMA_KEEP_ALIVE=-1`, and `get_weather` over `web_search` for weather.
- **`OLLAMA_MAX_TOKENS=1024`** default — lower values can miss tool calls when many tools are registered.
- Warm-ups (`WARM_*_AT_STARTUP`) preload models at boot.
- DuckDuckGo needs no key; Tavily is usually better for grounding when `WEB_SEARCH_ANSWER_ONLY=true`.

---

## License

[MIT](LICENSE) — Copyright (c) 2026 Rudy.

## Disclaimer

host-assistant is an independent community project — **not affiliated with,
endorsed by, or sponsored by Meta**. "Meta Portal" and "Portal" are trademarks of
Meta Platforms, Inc., used here only to identify compatible hardware. Demo /
LAN-only software for discontinued devices — **use at your own risk**. See
[DISCLAIMER.md](DISCLAIMER.md) for the full text and privacy notes.
