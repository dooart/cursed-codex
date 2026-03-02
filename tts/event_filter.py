"""Event classification, summarization, and filtering for Codex events."""

import time

COOLDOWN_S = 5.0

SKIP = frozenset([
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
    "item_started",
    "item_completed",
    "session_configured",
    "agent_reasoning",
    "agent_reasoning_raw_content",
])

HIGH = frozenset([
    "task_started",
    "task_complete",
    "agent_message",
    "user_message",
    "exec_command_begin",
    "exec_command_end",
    "patch_apply_begin",
    "patch_apply_end",
    "error",
    "warning",
    "stream_error",
    "turn_aborted",
])

MEDIUM = frozenset([
    "mcp_tool_call_begin",
    "mcp_tool_call_end",
    "web_search_begin",
    "web_search_end",
    "exec_approval_request",
    "apply_patch_approval_request",
    "model_reroute",
])


def classify(event_type):
    """Classify an event type as 'skip', 'high', or 'medium'."""
    if event_type in SKIP:
        return "skip"
    if event_type in HIGH:
        return "high"
    if event_type in MEDIUM:
        return "medium"
    return "skip"


def summarize(event):
    """Turn a raw event dict into a human-readable summary string."""
    t = event.get("type", "")

    if t == "task_started":
        return "Task started"
    if t == "task_complete":
        return "Task complete"
    if t == "agent_message":
        content = str(event.get("content") or event.get("message") or "")[:120]
        return f"Agent: {content}"
    if t == "user_message":
        content = str(event.get("content") or event.get("message") or "")[:120]
        return f"User: {content}"
    if t == "exec_command_begin":
        cmd = str(event.get("command") or event.get("call_id") or "")[:80]
        return f"Running: {cmd}"
    if t == "exec_command_end":
        code = event.get("exit_code") or event.get("exitCode") or "?"
        return f"Command finished (exit {code})"
    if t == "patch_apply_begin":
        return "Applying patch"
    if t == "patch_apply_end":
        return "Patch applied"
    if t in ("error", "stream_error"):
        msg = str(event.get("message") or event.get("error") or t)[:120]
        return f"Error: {msg}"
    if t == "warning":
        msg = str(event.get("message") or t)[:120]
        return f"Warning: {msg}"
    if t == "turn_aborted":
        return "Turn aborted"
    if t == "mcp_tool_call_begin":
        tool = str(event.get("tool_name") or event.get("name") or "")[:60]
        return f"MCP tool call: {tool}"
    if t == "mcp_tool_call_end":
        return "MCP tool call complete"
    if t == "web_search_begin":
        return "Web search started"
    if t == "web_search_end":
        return "Web search complete"
    if t == "exec_approval_request":
        return "Approval needed: exec command"
    if t == "apply_patch_approval_request":
        return "Approval needed: apply patch"
    if t == "model_reroute":
        return "Model rerouted"

    return t


class EventFilter:
    """Applies per-type cooldown and content dedup to raw events."""

    def __init__(self):
        self._last_seen = {}       # event_type -> timestamp
        self._recent_summaries = {}  # summary -> timestamp

    def filter(self, raw):
        """Returns a summary string for events that pass filtering, or None."""
        priority = classify(raw.get("type", ""))
        if priority == "skip":
            return None

        now = time.monotonic()

        # Per-type cooldown
        event_type = raw.get("type", "")
        last_time = self._last_seen.get(event_type, 0)
        if now - last_time < COOLDOWN_S:
            return None
        self._last_seen[event_type] = now

        summary = summarize(raw)

        # Content dedup
        last_summary_time = self._recent_summaries.get(summary, 0)
        if now - last_summary_time < COOLDOWN_S:
            return None
        self._recent_summaries[summary] = now

        # Prune old entries
        if len(self._recent_summaries) > 200:
            self._prune(now)

        return f"[{event_type}] {summary}"

    def _prune(self, now):
        stale = [k for k, ts in self._recent_summaries.items() if now - ts > COOLDOWN_S * 2]
        for k in stale:
            del self._recent_summaries[k]
