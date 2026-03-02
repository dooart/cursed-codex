"""
EXPERIMENT 07 — Direct Narration
=================================
Drop the perception layer entirely. Pass raw events to the LLM.
Ray understands what's happening — he just treats it like sport.

The comedy comes from: accurate description + absurd energy.
"THAT VARIABLE HAS BEEN RENAMED WITH INTENT." — literally true, absurdly grave.

Uses gpt-4.1-mini.

Usage:
    python llm_07_direct.py                # default session (v1)
    python llm_07_direct.py --test v2      # real session
    python llm_07_direct.py "[error] ..."  # single event
"""

import os
import random
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

# Load .env
for p in [Path(__file__).parent / ".env", Path(__file__).parent.parent / ".env"]:
    if p.exists():
        load_dotenv(p)
        break


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


# -- Schema --

class Reaction(BaseModel):
    line: str = Field(description="What Ray says. Punchy fragment. Specific to the event. No questions.")


class FollowUp(BaseModel):
    line: str = Field(description="Ray's next fragment. Extends or adds to his last line. Short. No questions.")


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

STYLE GUIDE — match this energy. NEVER copy these verbatim:
"AND IT OPENS THE THIRD FILE."
"A BOLD RETURN TO THE SAME FILE."
"THAT VARIABLE HAS BEEN RENAMED WITH INTENT."
"TESTS ARE UP. NOBODY BREATHE."
"SHOT AWAY. NOW WE WAIT."
"OFF THE POST."
"THAT MOVE BREAKS DOWN LATE."
"A SURGICAL EDIT IN A CROWDED AREA."
"CLEAN FINISH ON THAT SEQUENCE."
"AND WE HEAD TO REVIEW."

Notice the style: these lines ACKNOWLEDGE what happened but don't TRANSCRIBE it. \
"TESTS ARE UP" not "npm test RUNNING WITH 47 ASSERTIONS." \
"A SURGICAL EDIT" not "PATCH APPLIED TO src/utils.ts LINES 14-28." \
You're a commentator, not a log parser. Reference the moment, don't describe the payload.

You CAN mention a specific name if it's funny or dramatic — "EXPRESS IS DOWN" or \
"GIT STATUS AT A TIME LIKE THIS" — but only when it adds punch, not for accuracy.

RULES:
- MAX {max_words} WORDS.
- You are commentating, not documenting. Sound like a man in a booth, not a CI pipeline.
- Sports cadence and energy. Treat code events like match-defining moments.
- NEVER questions. No "?". Declarative only.
- NEVER repeat a line from the conversation history.
- Fragments over full sentences. Punchy. Clipped.
- MATCH ENERGY TO VERB: whispered = subdued. shouted/bellowed = explosive."""


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


# -- Token tracking --

class Usage:
    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def add(self, usage):
        if usage:
            self.prompt_tokens += usage.prompt_tokens or 0
            self.completion_tokens += usage.completion_tokens or 0

    @property
    def total(self):
        return self.prompt_tokens + self.completion_tokens

    def __str__(self):
        return f"prompt={self.prompt_tokens} completion={self.completion_tokens} total={self.total}"


_usage = Usage()


def _get_client() -> OpenAI:
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )


def _is_openai_model(model: str) -> bool:
    return model.startswith("openai/")


def _call_llm(user_msg: str, max_words: int, schema=Reaction, extra_messages: list | None = None):
    client = _get_client()
    system = SYSTEM_PROMPT.replace("{max_words}", str(max_words))

    messages = [
        {"role": "system", "content": system},
        *_history,
        {"role": "user", "content": user_msg},
    ]
    if extra_messages:
        messages.extend(extra_messages)

    if _is_openai_model(MODEL):
        # OpenAI models: native structured output
        resp = client.beta.chat.completions.parse(
            model=MODEL,
            messages=messages,
            max_tokens=500,
            temperature=1.0,
            response_format=schema,
        )
        _usage.add(resp.usage)
        parsed = resp.choices[0].message.parsed
        if parsed is None:
            raise ValueError(f"No parsed response. Raw: {resp.choices[0].message.content}")
        return parsed
    else:
        # Non-OpenAI models (Claude etc): JSON in prompt + manual parse
        json_hint = 'Respond with ONLY a JSON object: {"line": "your line here"}'
        messages[-1] = {"role": messages[-1]["role"], "content": messages[-1]["content"] + f"\n\n{json_hint}"}
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=500,
            temperature=1.0,
        )
        _usage.add(resp.usage)
        raw = resp.choices[0].message.content or ""
        # Extract JSON from response
        import json
        try:
            # Try direct parse first
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Try to find JSON in the response
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(raw[start:end])
            else:
                raise ValueError(f"Could not parse JSON from: {raw}")
        return schema(**data)


def generate(event: str) -> str:
    """Generate commentary. Returns the full spoken line."""
    verbs = _get_verbs(event)
    verb = random.choice(verbs)
    max_words = random.randint(6, 12)
    user_msg = f"{event}\nRay {verb}:"

    first = _call_llm(user_msg, max_words)
    fragments = [first.line]

    for chance in FOLLOWUP_CHANCES:
        if random.random() >= chance:
            break
        followup_words = random.randint(3, 8)
        so_far = ". ".join(fragments)
        extra = [
            {"role": "assistant", "content": f'Ray {verb}: "{so_far}"'},
            {"role": "user", "content": "Ray continues, adding to that thought:"},
        ]
        fu = _call_llm(user_msg, followup_words, schema=FollowUp, extra_messages=extra)
        fragments.append(fu.line)

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
    fragments = [first.line]

    for chance in FOLLOWUP_CHANCES:
        if random.random() >= chance:
            break
        followup_words = random.randint(3, 8)
        so_far = ". ".join(fragments)
        extra = [
            {"role": "assistant", "content": f'Ray {verb}: "{so_far}"'},
            {"role": "user", "content": "Ray continues, adding to that thought:"},
        ]
        fu = _call_llm(user_msg, followup_words, schema=FollowUp, extra_messages=extra)
        fragments.append(fu.line)

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
    print(f"--- tokens: {_usage} ---")


MODEL = "openai/gpt-4.1-mini"


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
        print(f"\n--- tokens: {_usage} ---")
        return

    _run_session(SESSION_V1, "v1")


if __name__ == "__main__":
    main()
