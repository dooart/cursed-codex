# Codex Commentator — Spec

An intentionally useless desktop sidecar that reacts to Codex runtime events
with sports-style commentary voice lines.

---

## Approach: patch-and-build

We do **not** fork the Codex repo. Instead we maintain a minimal `.patch` file
applied on top of a pinned Codex commit. This keeps our code separate and the
upstream delta trivially small.

### Repo structure

```
codex-commentator/
├── specs/
│   ├── research.md           # findings from source code analysis
│   └── spec.md               # this file
├── patches/
│   └── event-tap.patch       # the Rust patch (applied to pinned codex commit)
├── setup.sh                  # clone → checkout → patch → build
├── opensrc/                  # reference copy of codex source (for research)
└── app/                      # Electron commentator app (future)
```

### Setup flow

```bash
#!/bin/bash
CODEX_COMMIT="<pinned-sha>"   # pin to a known-good commit

git clone https://github.com/openai/codex.git vendor/codex
cd vendor/codex
git checkout $CODEX_COMMIT
git apply ../../patches/event-tap.patch
cd codex-rs
cargo build -p codex-cli --release

echo ""
echo "Done. Run codex with:"
echo "  $(pwd)/target/release/codex"
```

User runs the patched binary instead of the stock `codex` command. Everything
works identically except a local WebSocket is opened for event streaming.

---

## The patch

### What it does

Adds a **local WebSocket server** inside the TUI process that broadcasts every
`EventMsg` as JSON to any connected client.

### Where it hooks in

File: `codex-rs/tui/src/chatwidget/agent.rs`

The event loop at line 64 of `spawn_agent()`:

```rust
while let Ok(event) = thread.next_event().await {
    app_event_tx.send(AppEvent::CodexEvent(event));
    // ...
}
```

The patch adds a broadcast **before** the existing send:

```rust
while let Ok(event) = thread.next_event().await {
    // --- patch: broadcast to sidecar ---
    if let Ok(json) = serde_json::to_string(&event.msg) {
        let _ = tap_tx.send(json);
    }
    // --- end patch ---
    app_event_tx.send(AppEvent::CodexEvent(event));
    // ...
}
```

The same change is applied to `spawn_agent_from_existing()` (same file,
same pattern).

### Side-channel details

- **Transport**: WebSocket on `127.0.0.1`
- **Port**: chosen from env var `CODEX_TAP_PORT` (default: `4222`)
- **Protocol**: one JSON message per event, newline-delimited
- **Direction**: server → client only (read-only tap, no commands accepted)
- **Lifecycle**: server starts when TUI starts, stops when TUI exits
- **Dependencies**: `tokio-tungstenite` (already in the workspace)

### Port discovery

The patched binary writes the WebSocket URL to a known file:

```
~/.codex/commentator.sock
```

Contents: `ws://127.0.0.1:4222`

The Electron app reads this file on startup to find the connection.

---

## Event mapping

Events from `EventMsg` mapped to commentary categories:

| EventMsg variant | Commentary category | Example line |
|-----------------|---------------------|-------------|
| `TurnStarted` | turn_started | "WE ARE UNDERWAY." |
| `TurnComplete` | turn_completed | "AND THAT'S THE WHISTLE." |
| `TurnAborted` | turn_interrupted | "HARD RESET FROM THE SIDELINE." |
| `ExecCommandBegin` | item_started_command | "TESTS ARE UP. NOBODY BREATHE." |
| `ExecCommandEnd` | item_completed | "THAT ONE LANDS." |
| `PatchApplyBegin` | item_started_edit | "LINES ARE MOVING." |
| `PatchApplyEnd` | item_completed | "CLEAN FINISH ON THAT SEQUENCE." |
| `Error` | turn_failed | "OFF THE POST." |
| `StreamError` | turn_failed | "NOT TODAY. BACK TO BUILD-UP." |
| `McpToolCallBegin` | item_started_command | "A BIG COMMAND AT A BIG MOMENT." |
| `McpToolCallEnd` | item_completed | "AND IT GETS OVER THE LINE." |
| `AgentMessageDelta` | (ignored — too noisy) | — |
| `TokenCount` | (ignored) | — |

### Canonical event model (Electron side)

```ts
type CommentaryEvent =
  | 'TURN_STARTED'
  | 'TURN_COMPLETED'
  | 'TURN_INTERRUPTED'
  | 'TURN_FAILED'
  | 'ITEM_STARTED_READ'
  | 'ITEM_STARTED_EDIT'
  | 'ITEM_STARTED_COMMAND'
  | 'ITEM_COMPLETED';
```

The Electron app normalizes raw `EventMsg` variants into these categories
before passing to the phrase engine.

---

## Electron app (MVP)

### Components

1. **WebSocket listener** — connects to the tap, receives JSON events
2. **Event normalizer** — maps `EventMsg` → `CommentaryEvent`
3. **Phrase engine** — picks a line from the phrase bank per event + persona
4. **TTS adapter** — speaks the line (provider TBD, see linked TTS note)
5. **Audio queue** — prevents overlaps, interrupt policy for high-priority events
6. **UI** — minimal window showing live timeline + controls

### Anti-annoyance rules

- Cooldown per event type (5-10s)
- Max line length: 12 words
- No repeated phrase within last N utterances
- High-signal events only (deltas and token counts are ignored)
- Mute toggle / quiet hours

### Personas (starter pack)

- **Classic Sportscaster** — measured, professional, British football style
- **Overdramatic Esports Caster** — hype, screaming, "WHAT A PLAY"
- **Deadpan Analyst** — dry, understated, "that's a command, I suppose"

### Phrase bank

See the original note for the full phrase bank. Each persona gets its own
pool of lines per `CommentaryEvent`.

---

## Stretch goals

### Jumbotron mode

Visual broadcast overlay with:
- Live scoreboard (files touched, lines changed, commands run, error count)
- Session clock
- ESPN-style scrolling ticker
- Replay animations for diffs
- Crowd noise meter
- Fake sponsored messages

### Other

- Soundboard mode (pre-recorded clips instead of TTS)
- Team mode (multiple commentator personas arguing)
- Robot/IoT webhook (`onEvent -> POST /robot/dance`)
