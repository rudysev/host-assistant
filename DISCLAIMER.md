# Disclaimer

**host-assistant is an independent, community-built project. It is not affiliated
with, authorized by, endorsed by, or sponsored by Meta Platforms, Inc.**

"Meta", "Meta Portal", and "Portal" are trademarks of Meta Platforms, Inc. They
are used here only to identify the hardware that
[portal-assistant](https://github.com/rudysev/portal-assistant) (the companion
Android app) runs on (nominative use). host-assistant is not a Meta product and
ships no Meta code.

## Use at your own risk

host-assistant is demo / starting-point software that you run yourself on a
machine on your LAN. It is meant for testing portal-assistant against a local
model. Meta Portal devices are discontinued and receive no official support. By
using this software you accept that:

- Installing and running third-party apps on a Portal may **void any remaining
  warranty** or violate the device's terms of use.
- Running a LAN voice host that accepts microphone audio always carries some
  risk. We are not aware of this project causing any harm, but **no outcome is
  guaranteed**.
- You are responsible for how you deploy it (network exposure, firewall, who can
  reach the WebSocket). The host accepts a single client and is intended for a
  trusted LAN only — not the public internet.

The software is provided "AS IS", without warranty of any kind, under the terms
of the [MIT License](LICENSE). To the maximum extent permitted by law, the
authors and contributors accept no liability for any damage, data loss, or other
harm arising from its use.

## Privacy

host-assistant has no analytics and no accounts. It is a local process on your
LAN host. When portal-assistant is pointed at this host, microphone audio for an
active conversation is streamed over your LAN (`wss://`) to that machine, where
speech-to-text, the LLM, and text-to-speech run locally (Whisper, Ollama, Kokoro).

- Audio stays on your LAN for the local-backend path; it is not sent to Google
  Gemini by this project. (portal-assistant's default cloud backend is separate —
  see that app's disclaimer.)
- Optional host tools may call the network: **Open-Meteo** (`get_weather`) and
  **Tavily** or **DuckDuckGo** (`web_search`). Those requests, and any data those
  services receive, are governed by **their** terms and privacy policies.
- If you set `TAVILY_API_KEY`, it lives in your local `.env` and is sent only to
  Tavily to authenticate search requests. The project never receives it.
- A self-signed TLS certificate is stored under `~/.host-assistant/tls/` on the
  host. No personal data is collected by the project.

## Reporting issues

If you believe any content here infringes your rights, or you represent Meta and
have concerns, please open an issue or contact the maintainers; we will respond
promptly.
