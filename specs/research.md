# Codex Commentator — Research Findings

Research into how the OpenAI Codex CLI works internally, to determine how a
desktop sidecar app can listen to agent events in real time.

Source code examined: `./opensrc/repos/github.com/openai/codex/`

---

## 1. Codex CLI architecture

The `codex` command is a **thin Node.js wrapper** (`codex-cli/bin/codex.js`)
that spawns a **Rust binary** with `stdio: "inherit"`. The Node layer does
nothing interesting — it resolves the correct platform binary, spawns it, and
forwards signals.

All logic lives in the Rust binary, which is a Cargo workspace of ~67 crates
rooted at `codex-rs/`.

### Key crates

| Crate | Role |
|-------|------|
| `cli/` | Entry point. Dispatches subcommands (`app-server`, `exec`, interactive TUI, etc.) |
| `tui/` | Ratatui-based fullscreen terminal UI |
| `core/` | Business logic — `ThreadManager`, `Codex::spawn`, session management |
| `protocol/` | Event and operation types (`EventMsg`, `Op`) |
| `app-server/` | JSON-RPC 2.0 server (stdio or WebSocket transport) |
| `app-server-protocol/` | Wire protocol types for the app-server (v2) |

### Two runtime modes

1. **Interactive TUI** (`codex` with no subcommand)
   - Rust binary runs Ratatui TUI
   - Creates `ThreadManager` **in-process**
   - TUI ↔ core communication via **Rust async channels** (not stdio, not sockets)
   - No subprocess, no pipe, no network — everything is one process

2. **App-server mode** (`codex app-server`)
   - Runs a standalone JSON-RPC 2.0 server
   - Transport: `--listen stdio://` (default) or `--listen ws://IP:PORT`
   - Designed for IDE extensions (VS Code) and programmatic clients
   - Broadcasts event notifications to all connected clients

## 2. How the TUI processes events

File: `codex-rs/tui/src/chatwidget/agent.rs`

There is a single event loop (line 64):

```rust
while let Ok(event) = thread.next_event().await {
    let is_shutdown_complete = matches!(event.msg, EventMsg::ShutdownComplete);
    app_event_tx.send(AppEvent::CodexEvent(event));
    if is_shutdown_complete {
        break;
    }
}
```

**Every** event flows through this point. The `event` is a `codex_protocol::protocol::Event`
containing an `EventMsg` enum. This is the tap point.

Both `spawn_agent()` and `spawn_agent_from_existing()` have the same loop pattern.

## 3. Available events

From `codex-rs/protocol/src/protocol.rs`, the `EventMsg` enum:

### High-signal events (useful for commentary)

| Event | Description |
|-------|-------------|
| `TurnStarted` | Agent begins a turn |
| `TurnComplete` | Agent finishes a turn |
| `TurnAborted` | Turn was aborted |
| `ExecCommandBegin` | About to execute a shell command |
| `ExecCommandEnd` | Shell command finished |
| `ExecCommandOutputDelta` | Incremental command output |
| `PatchApplyBegin` | About to apply a code patch |
| `PatchApplyEnd` | Patch application finished |
| `McpToolCallBegin` | MCP tool call started |
| `McpToolCallEnd` | MCP tool call finished |
| `WebSearchBegin` | Web search started |
| `WebSearchEnd` | Web search finished |
| `Error` | Error during execution |
| `StreamError` | Model stream error/disconnect |
| `ShutdownComplete` | Agent shutting down |

### Lower-signal events (context / streaming)

| Event | Description |
|-------|-------------|
| `AgentMessage` | Complete agent text output |
| `AgentMessageDelta` | Streamed text chunk |
| `AgentReasoning` / `AgentReasoningDelta` | Reasoning output |
| `UserMessage` | What was sent to the model |
| `SessionConfigured` | Session ack |
| `TokenCount` | Usage stats |
| `TurnDiff` | Diff of changes in a turn |
| `ExecApprovalRequest` | Asking user to approve a command |
| `ApplyPatchApprovalRequest` | Asking user to approve a patch |
| `BackgroundEvent` | Background processing |
| `UndoStarted` / `UndoCompleted` | Undo operations |
| `PlanUpdate` | Plan step changes |

## 4. Can the CLI connect to an external app-server?

**No.** There is no `--connect`, `--server`, or hidden flag. The interactive
TUI always creates its own `ThreadManager` in-process. The app-server mode is
server-only (`--listen`), not a client mode.

We confirmed by searching all CLI args, env vars, and config options in the
Rust source.

## 5. Can we proxy the stdio pipe?

**No.** The Node wrapper spawns the Rust binary with `stdio: "inherit"` —
meaning the Rust process owns the terminal directly for Ratatui rendering.
The data flowing over stdio is raw terminal I/O (escape codes, cursor
movements), not JSON-RPC messages.

JSON-RPC only flows when running `codex app-server` mode explicitly, which
does not provide the interactive TUI.

## 6. Build requirements

- **Rust 1.93.0** — auto-installed via `rust-toolchain.toml` + rustup
- **macOS**: may need `brew install openssl` (for `openssl-sys` dep)
- **First build**: ~5-10 min (67 crates + git-patched dependencies)
- **Incremental builds**: seconds (touching only the patched file)
- Forked deps handled automatically by Cargo via `[patch.crates-io]`

Build command:
```bash
cd codex-rs
cargo build -p codex-cli --release
# binary: target/release/codex
```

## 7. Conclusion

The only viable approach for tapping into the TUI's event stream is a **small
Rust patch** to `agent.rs` that opens a local side-channel (WebSocket or Unix
socket) and broadcasts serialized events. The patch touches one file in a
single event loop. The rest of the Codex codebase remains untouched.
