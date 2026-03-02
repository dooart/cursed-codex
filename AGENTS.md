# AGENTS.md

Instructions for AI coding agents working with this codebase.

## Architecture

```
Codex (TCP:4222) --> tts/server.py (TCP client + event filter + LLM + TTS) --> Audio
```

- **Codex**: Patched build (`vendor/codex/`) exposes a TCP event tap on port 4222
- **tts/server.py**: Main entry point. Runs TCP listener + HTTP server (port 8080)
- **tts/tcp_listener.py**: TCP client with auto-reconnect and newline-delimited JSON parsing
- **tts/event_filter.py**: Classifies events (skip/high/medium), per-type cooldown, content dedup
- **tts/llm.py**: Router that selects the active LLM experiment module
- **tts/llm_01 through llm_07**: Prompt experiments. Active one is set in `llm.py` (default: `07_direct`)
- **tts/voice.safetensors**: Cloned voice weights for Pocket TTS

## Key conventions

- Python code lives in `tts/`. No package structure — flat files, run from `tts/` directory.
- LLM modules export `generate(event: str) -> str` and `normalize(text: str) -> str`.
- The server uses a single `threading.Lock` (`busy`) to ensure only one TTS generation runs at a time. Events arriving while busy are dropped.
- All errors are `print()` logs. No alerts, no dialogs, no exceptions surfaced to the user.
- The `app/` directory contains an old Electron middleman layer that is no longer used. The Python server connects to Codex directly.

## Event flow

1. Codex emits newline-delimited JSON over TCP: `{"id": "...", "msg": {"type": "...", ...}}`
2. `tcp_listener.py` extracts `msg` and calls back with the raw event dict
3. `event_filter.py` classifies by type, applies 5s per-type cooldown and content dedup
4. Passing events become a string like `"[task_started] Task started"` and enter `handle_event()`
5. `handle_event()` truncates to 200 chars, acquires the busy lock, spawns a thread for LLM + TTS
6. LLM generates commentary, `normalize()` cleans it for TTS, Pocket TTS renders audio, `afplay` plays it

## Adding a new LLM experiment

1. Create `tts/llm_NN_name.py` with `generate(event: str) -> str` and `normalize(text: str) -> str`
2. Update `tts/llm.py`: change `ACTIVE = "NN_name"`
3. Test standalone: `python llm_NN_name.py "[task_started] Task started"`
4. Test in server: `python server.py` then `curl -X POST http://localhost:8080 -d '{"text":"[task_started] Task started"}'`

## Environment

- Python 3.11+ with venv at `tts/venv/`
- Key deps: `pocket-tts`, `scipy`, `openai`, `python-dotenv`, `requests`
- API key in `tts/.env`: `OPENROUTER_API_KEY=...`
- macOS only (uses `afplay` for audio playback)

<!-- opensrc:start -->

## Source Code Reference

Source code for dependencies is available in `opensrc/` for deeper understanding of implementation details.

See `opensrc/sources.json` for the list of available packages and their versions.

Use this source code when you need to understand how a package works internally, not just its types/interface.

### Fetching Additional Source Code

To fetch source code for a package or repository you need to understand, run:

```bash
npx opensrc <package>           # npm package (e.g., npx opensrc zod)
npx opensrc pypi:<package>      # Python package (e.g., npx opensrc pypi:requests)
npx opensrc crates:<package>    # Rust crate (e.g., npx opensrc crates:serde)
npx opensrc <owner>/<repo>      # GitHub repo (e.g., npx opensrc vercel/ai)
```

<!-- opensrc:end -->
