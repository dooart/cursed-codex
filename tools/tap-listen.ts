#!/usr/bin/env bun
/**
 * tap-listen — connects to the codex event-tap TCP server and prints events.
 *
 * The server sends newline-delimited JSON over a plain TCP socket.
 *
 * Usage:
 *   bun tools/tap-listen.ts [host:port]
 *
 * If no address is given, reads from ~/.codex/commentator.sock
 * Falls back to 127.0.0.1:4222
 */

import { readFileSync } from "fs";
import { homedir } from "os";
import { join } from "path";
import { createConnection } from "net";
import { createInterface } from "readline";

const SOCK_FILE = join(homedir(), ".codex", "commentator.sock");
const DEFAULT_ADDR = "127.0.0.1:4222";

function resolveAddr(): { host: string; port: number } {
  let raw: string | undefined;

  // explicit arg
  if (process.argv[2]) {
    raw = process.argv[2];
  }

  // sock file
  if (!raw) {
    try {
      const contents = readFileSync(SOCK_FILE, "utf-8").trim();
      if (contents) {
        // strip protocol prefix if present (tcp://host:port or ws://host:port)
        raw = contents.replace(/^(tcp|ws):\/\//, "");
      }
    } catch {
      // ignore
    }
  }

  if (!raw) raw = DEFAULT_ADDR;

  const [host, portStr] = raw.split(":");
  return { host: host || "127.0.0.1", port: parseInt(portStr, 10) || 4222 };
}

const addr = resolveAddr();
let reconnectDelay = 1000;

// Events to skip entirely (streaming deltas and noise)
const SKIP_EVENTS = new Set([
  "agent_message_delta",
  "agent_message_content_delta",
  "agent_reasoning_delta",
  "agent_reasoning_raw_content_delta",
  "agent_reasoning_section_break",
  "reasoning_content_delta",
  "reasoning_raw_content_delta",
  "plan_delta",
  "exec_command_output_delta",
  "raw_response_item",
  "token_count",
]);

function handleLine(line: string) {
  try {
    const data = JSON.parse(line);
    // Event shape: { id: string, msg: { type: "task_started", ... } }
    const msg = data.msg;
    const eventType: string =
      (msg && typeof msg === "object" && msg.type) ||
      (typeof msg === "string" ? msg : "unknown");

    if (SKIP_EVENTS.has(eventType)) return;

    const ts = new Date().toISOString().slice(11, 23);
    console.log(`[${ts}] ${eventType}`);
    console.log(`         ${JSON.stringify(msg).slice(0, 200)}`);
  } catch {
    // not JSON — dump raw so we can debug the shape
    if (line.trim()) console.log(`[raw] ${line.slice(0, 300)}`);
  }
}

function connect() {
  console.log(`Connecting to ${addr.host}:${addr.port} ...`);

  const socket = createConnection(addr.port, addr.host, () => {
    console.log("Connected. Waiting for events...\n");
    reconnectDelay = 1000;
  });

  const rl = createInterface({ input: socket });
  rl.on("line", handleLine);

  socket.on("close", () => {
    rl.close();
    console.log(
      `\nDisconnected. Reconnecting in ${reconnectDelay / 1000}s...`
    );
    setTimeout(connect, reconnectDelay);
    reconnectDelay = Math.min(reconnectDelay * 2, 10000);
  });

  socket.on("error", () => {
    // suppress — close handler will reconnect
  });
}

connect();
