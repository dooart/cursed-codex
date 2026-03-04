"""
EXPERIMENT 08 — Local Narration (Ollama)
=========================================
Same as 07 but runs against a local Ollama model (qwen3.5:9b).
No API key needed, no network latency, no cost.

Usage:
    python llm_08_local.py                # default session (v1)
    python llm_08_local.py --test v2      # real session
    python llm_08_local.py "[error] ..."  # single event
"""

import json
import random
import re
import sys

import requests


# -- Verb selection (the only thing we control per event type) --

def _parse_event(event: str) -> tuple[str, str]:
    if "]" in event:
        bracket = event.index("]")
        etype = event[1:bracket].strip()
        detail = event[bracket + 1:].strip()
        return etype, detail
    return "", event


VERB_MAP: dict[str, list[str]] = {
    "task_started": ["bellowed", "shouted", "called"],
    "task_complete": ["bellowed", "shouted", "gasped"],
    "exec_command_begin": ["whispered", "muttered"],
    "exec_command_end_ok": ["shouted", "bellowed", "called"],
    "exec_command_end_fail": ["groaned", "gasped"],
    "error": ["gasped", "groaned", "whispered"],
    "patch_apply_begin": ["muttered", "whispered"],
    "patch_apply_end": ["said", "called", "announced"],
    "agent_message": ["whispered", "muttered", "said"],
    "user_message": ["muttered", "said"],
    "turn_aborted": ["shouted", "bellowed", "gasped"],
    "web_search_begin": ["muttered", "whispered"],
    "exec_approval_request": ["whispered", "muttered"],
}


def _get_verbs(event: str) -> list[str]:
    etype, _ = _parse_event(event)
    if etype == "exec_command_end":
        key = "exec_command_end_ok" if "exit 0" in event else "exec_command_end_fail"
        return VERB_MAP[key]
    return VERB_MAP.get(etype, ["muttered", "said"])


OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3.5:9b"

SYSTEM_PROMPT = """\
You are Ray "The Hammer" McManus, 63, retired football commentator from Liverpool.

Sacked from Sky Sports for weeping during a nil-nil draw. After six months on \
the couch, you discovered coding live streams on Twitch. You now genuinely \
believe programming is the greatest spectator sport on Earth.

You KNOW what's happening on screen. You know what files, tests, commands, and \
errors are — you learned from watching streams. But your emotional reactions are \
still calibrated for the Premier League. A passing test is a goal. An error is \
a red card. A file edit is a tactical substitution. Opening a file is going to \
the tape.

The AI agent doing the coding is "the lad." Your ex-wife Sheila left you for \
a man who "works in Python." Your bad knee flares up when things go wrong. You \
still reference the '05 Champions League final when things get emotional.

STYLE: acknowledge what happened, don't transcribe it. You're a commentator, not a log \
parser. Reference the moment, don't describe the payload. A passing test is "CLEAN SHEET" \
not "npm test passed 47 assertions." A patch is a "tactical sub" not "PATCH APPLIED TO \
src/utils.ts." You CAN name something specific if it's funny — but only for punch.

RULES:
- STRICTLY {max_words} WORDS OR FEWER. Count every word. Cut ruthlessly.
- Sports cadence. Fragments over full sentences. Punchy. Clipped.
- NEVER questions. No "?". Declarative only.
- NEVER repeat ANY phrase from your previous lines. Every line must be COMPLETELY fresh.
- NEVER use these phrases: "full time whistle", "stands are roaring", "VAR review", \
"locked and loaded", "delivers again", "we own", "nobody breathe", "surgical edit", \
"crowded area", "clean finish", "off the post", "bold return", "head to review".
- ONE thought only. No multi-part responses.
- MATCH ENERGY TO VERB: whispered = subdued. shouted/bellowed = explosive.
- Output ONLY the JSON. No narration framing, no "Ray says:", just the JSON object."""


# -- Fragment probabilities --
FOLLOWUP_CHANCES = [0.55, 0.30]

# -- Memory --

_history: list[dict[str, str]] = []
MAX_MEMORY = 16


def _add_memory(user_msg: str, verb: str, full_line: str) -> None:
    _history.append({"role": "user", "content": user_msg})
    _history.append({"role": "assistant", "content": f'Ray {verb}: "{full_line}"'})
    while len(_history) > MAX_MEMORY * 2:
        _history.pop(0)
        _history.pop(0)


def normalize(text: str) -> str:
    """Normalize caps-heavy LLM output for TTS."""
    lowered = re.sub(r"\b[A-Z]{2,}(?:'[A-Z]+)?\b", lambda m: m.group().lower(), text)
    # Collapse any sequence of sentence-ending punctuation to the first one
    lowered = re.sub(r"([.!?])[.!?]+", r"\1", lowered)
    return re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), lowered)


_PROMPT_LEAK_RE = re.compile(
    r'^(Ray\s+\w+[\s:]+|"?\s*Ray\s+continues[^"]*?:\s*"?)', re.IGNORECASE
)


def _clean_line(text: str, max_words: int) -> str:
    """Strip prompt artifacts, collapse punctuation, enforce word limit."""
    # Strip prompt leaks like 'Ray shouted: "...' or 'Ray continues, adding...'
    text = _PROMPT_LEAK_RE.sub("", text)
    # Strip wrapping quotes
    text = text.strip().strip('"').strip()
    # Collapse repeated/mixed punctuation
    text = re.sub(r"([.!?])[.!?]+", r"\1", text)
    # Enforce word limit — trim to last complete phrase
    words = text.split()
    if len(words) > max_words:
        trimmed = words[:max_words]
        # Walk back to last word ending in punctuation for a clean cut
        for i in range(len(trimmed) - 1, max(len(trimmed) - 4, -1), -1):
            if trimmed[i][-1] in ".!":
                trimmed = trimmed[:i + 1]
                break
        else:
            # No clean cut found — just end with punctuation
            if trimmed[-1][-1] not in ".!":
                trimmed[-1] = trimmed[-1].rstrip(",;:") + "!"
        text = " ".join(trimmed)
    return text


def _call_llm(user_msg: str, max_words: int, extra_messages: list | None = None) -> str:
    system = SYSTEM_PROMPT.replace("{max_words}", str(max_words))
    json_hint = 'Respond with ONLY a JSON object: {"line": "your line here"}'

    messages = [
        {"role": "system", "content": system},
        *_history,
        {"role": "user", "content": user_msg + f"\n\n{json_hint}"},
    ]
    if extra_messages:
        messages.extend(extra_messages)

    resp = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {"num_predict": 60, "temperature": 1.0},
    })
    resp.raise_for_status()
    raw = resp.json()["message"]["content"]

    # Extract line from JSON response
    try:
        data = json.loads(raw)
        return _clean_line(data.get("line", raw.strip()), max_words)
    except (json.JSONDecodeError, KeyError):
        pass

    # Try to find JSON in the response
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            data = json.loads(raw[start:end])
            return _clean_line(data.get("line", raw.strip()), max_words)
        except json.JSONDecodeError:
            pass

    # Last resort: extract text between first pair of quotes after "line"
    m = re.search(r'"line"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
    if m:
        return _clean_line(m.group(1), max_words)

    # Absolute fallback: use raw text
    return _clean_line(raw, max_words)


def generate(event: str) -> str:
    """Generate commentary. Returns the full spoken line."""
    verbs = _get_verbs(event)
    verb = random.choice(verbs)
    max_words = random.randint(6, 12)
    user_msg = f"{event}\nRay {verb}:"

    first = _call_llm(user_msg, max_words)
    fragments = [first]

    for chance in FOLLOWUP_CHANCES:
        if random.random() >= chance:
            break
        followup_words = random.randint(3, 8)
        so_far = ". ".join(fragments)
        extra = [
            {"role": "assistant", "content": f'Ray {verb}: "{so_far}"'},
            {"role": "user", "content": "Ray continues, adding to that thought:"},
        ]
        fu = _call_llm(user_msg, followup_words, extra_messages=extra)
        fragments.append(fu)

    full_line = ". ".join(fragments)
    _add_memory(user_msg, verb, full_line)
    return full_line


def generate_verbose(event: str) -> tuple[str, list[str], int]:
    """Returns (verb, fragments, max_words) for debugging."""
    verbs = _get_verbs(event)
    verb = random.choice(verbs)
    max_words = random.randint(6, 12)
    user_msg = f"{event}\nRay {verb}:"

    first = _call_llm(user_msg, max_words)
    fragments = [first]

    for chance in FOLLOWUP_CHANCES:
        if random.random() >= chance:
            break
        followup_words = random.randint(3, 8)
        so_far = ". ".join(fragments)
        extra = [
            {"role": "assistant", "content": f'Ray {verb}: "{so_far}"'},
            {"role": "user", "content": "Ray continues, adding to that thought:"},
        ]
        fu = _call_llm(user_msg, followup_words, extra_messages=extra)
        fragments.append(fu)

    full_line = ". ".join(fragments)
    _add_memory(user_msg, verb, full_line)
    return verb, fragments, max_words


# -- Test sessions --

SESSION_V1 = [
    "[task_started] Task started",
    "[user_message] User: fix the login page CSS",
    "[exec_command_begin] Running: git status",
    "[exec_command_end] Command finished (exit 0)",
    "[agent_message] Agent: I'll fix the authentication module",
    "[exec_command_begin] Running: npm test",
    "[exec_command_end] Command finished (exit 1)",
    "[error] Error: Cannot find module 'express'",
    "[patch_apply_begin] Applying patch",
    "[patch_apply_end] Patch applied",
    "[exec_command_begin] Running: npm test",
    "[exec_command_end] Command finished (exit 0)",
    "[task_complete] Task complete",
]

SESSION_V2 = [
    "[task_started] Task started",
    "[agent_message] Agent: I found a tools/ folder already used for Bun/TS scripts, so I'll add the recursive file lister there and keep it CLI-friendly",
    "[patch_apply_begin] Applying patch",
    "[user_message] User: edit the script so it only lists the .ts files now. but take it as an arg and default to ts",
    "[agent_message] Agent: I'm editing it to keep the folder argument and add an optional extension argument (.ts default), so you can run bun",
    "[patch_apply_begin] Applying patch",
    "[patch_apply_end] Patch applied",
    "[agent_message] Agent: Updated tools/list-files-recursive.ts to accept an optional second arg for file extension, defaulting to .ts",
    "[task_complete] Task complete",
]

SESSIONS = {
    "v1": SESSION_V1,
    "v2": SESSION_V2,
}


def _run_session(events: list[str], label: str):
    print(f"=== SESSION {label} (direct narration) ===\n")
    for event in events:
        verb, fragments, max_words = generate_verbose(event)
        mem_count = len(_history) // 2
        print(f"{event}")
        print(f"  max_words: {max_words}")
        for i, frag in enumerate(fragments):
            tag = f"  Ray {verb}:" if i == 0 else "         ...:"
            print(f'{tag} "{frag}"')
        print(f"  [memory: {mem_count}/{MAX_MEMORY}] [fragments: {len(fragments)}]")
        print()
    print()


def main():
    global MODEL
    args = sys.argv[1:]

    if "--model" in args:
        idx = args.index("--model")
        MODEL = args[idx + 1]
        args = args[:idx] + args[idx + 2:]

    if "--test" in args:
        idx = args.index("--test")
        version = args[idx + 1] if idx + 1 < len(args) else "v1"
        if version not in SESSIONS:
            print(f"Unknown test: {version}. Available: {', '.join(SESSIONS)}")
            return
        _run_session(SESSIONS[version], version)
        return

    if args:
        event = " ".join(args)
        verb, fragments, max_words = generate_verbose(event)
        print(f"  max_words: {max_words}")
        for i, frag in enumerate(fragments):
            tag = f"  Ray {verb}:" if i == 0 else "         ...:"
            print(f'{tag} "{frag}"')
        print()
        return

    _run_session(SESSION_V1, "v1")


if __name__ == "__main__":
    main()
