"""
EXPERIMENT 04 — Grounded Misinterpretation
==========================================
Problem with 03: lines are generic. "Get in there!" could be any event.
No connection to what's actually happening. Feels like random quotes.

Fix: add a "what Ray sees" layer. We translate each event into what a
confused 63-year-old would PERCEIVE on the screen — shapes, colors, words
he can read but misinterprets. This gives the model something concrete to
react to, grounding each line in the specific event.

"Running: git status" → "words scrolling fast, a list of names appearing"
"Error: Cannot find module 'express'" → "the word EXPRESS flashing in red"
"exit 0" → "a big green zero appeared"
"exit 1" → "a red number one, like a red card"

Ray still doesn't understand code. But he SEES things and reacts to them.
The comedy comes from his football interpretation of what he sees.

Also carries forward from 03:
- Pre-set mood + verb (we control, not the model)
- 6-word max, structured output
- Assistant-turn memory, 16 slots

Usage:
    python llm_04_grounded.py                # sequential session
    python llm_04_grounded.py "some event"   # specific event
"""

import os
import random
import re
import sys
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

# Load .env
for p in [Path(__file__).parent.parent / ".env", Path(__file__).parent.parent / "app" / ".env"]:
    if p.exists():
        load_dotenv(p)
        break


# -- Verb type --

Verb = Literal[
    "whispered",
    "muttered",
    "said",
    "called",
    "announced",
    "barked",
    "bellowed",
    "shouted",
    "gasped",
    "groaned",
    "sighed",
]


# -- Event → what Ray sees + mood + verbs --

def _parse_event(event: str) -> tuple[str, str]:
    """Extract event_type and detail from '[type] detail'."""
    if "]" in event:
        bracket = event.index("]")
        etype = event[1:bracket].strip()
        detail = event[bracket + 1:].strip()
        return etype, detail
    return "", event


def _build_perception(event: str) -> tuple[str, str, list[str]]:
    """Returns (what_ray_sees, mood, verb_pool) for an event."""
    etype, detail = _parse_event(event)

    if etype == "task_started":
        return (
            "the screen just lit up, everything started moving at once",
            "electric anticipation, bouncing on his toes, floodlights coming on",
            ["bellowed", "shouted", "called"],
        )

    if etype == "task_complete":
        return (
            "everything stopped, a calm settled over the screen, it's done",
            "pure euphoria, tears forming, proudest moment since the '05 final",
            ["bellowed", "shouted", "gasped"],
        )

    if etype == "exec_command_begin":
        # Extract command name for Ray to misread
        cmd = detail.replace("Running:", "").strip().split()[0] if "Running:" in detail else "something"
        return (
            f"the word '{cmd}' just appeared on screen and everything's moving",
            "frozen, gripping the armrest, the lad's stepping up",
            ["whispered", "muttered"],
        )

    if etype == "exec_command_end":
        if "exit 0" in event:
            return (
                "a big green zero appeared on screen, everything went quiet and calm",
                "leaping out of his seat, fist pumping, the lad's just scored",
                ["shouted", "bellowed", "called"],
            )
        else:
            code = "1"
            for part in detail.split():
                if part.rstrip(")").isdigit():
                    code = part.rstrip(")")
            return (
                f"a red number {code} flashed on screen like a warning, text went red",
                "knee's screaming, worst since the '04 derby, the lad's down",
                ["groaned", "gasped"],
            )

    if etype == "error":
        # Pull the memorable word from the error
        words = detail.replace("Error:", "").strip().split()
        # Find the most "visible" word — longest non-generic one
        skip = {"error", "cannot", "could", "from", "that", "this", "with", "file", "find", "such", "module", "directory", "no"}
        visible = [w.strip("'\"") for w in words if len(w) > 3 and w.lower().strip("'\"") not in skip]
        keyword = visible[0] if visible else "ERROR"
        return (
            f"red text everywhere, the word '{keyword}' flashing on screen",
            "horror, hand over mouth, stretcher on the pitch — but the match goes on",
            ["gasped", "groaned", "whispered"],
        )

    if etype == "patch_apply_begin":
        return (
            "lines of text shifting around, things being crossed out and rewritten",
            "squinting, arms crossed, the fourth official is reviewing something suspicious",
            ["muttered", "whispered"],
        )

    if etype == "patch_apply_end":
        return (
            "the text settled down, everything snapped into place, clean",
            "grudging nod, impressed despite himself, like a rival scoring a worldie",
            ["said", "called", "announced"],
        )

    if etype == "agent_message":
        short = detail.replace("Agent:", "").strip()[:40]
        return (
            f"the lad's writing something: '{short}'",
            "watching the young striker warm up, quietly hopeful, don't jinx it",
            ["whispered", "muttered", "said"],
        )

    if etype == "user_message":
        # Ray can't read the words, he just sees the boss gesturing
        word_count = len(detail.replace("User:", "").strip().split())
        length = "short" if word_count < 5 else "long"
        return (
            f"the boss just sent a {length} message, pointing at the pitch, giving orders",
            "bolt upright, new tactical instructions, the boss means business",
            ["muttered", "said"],
        )

    if etype == "turn_aborted":
        return (
            "everything just stopped mid-action, screen went still",
            "jaw dropped, like the ref just sent off the captain in a cup final",
            ["shouted", "bellowed", "gasped"],
        )

    if etype == "web_search_begin":
        return (
            "a new window opened, text flying in from somewhere outside",
            "narrowing his eyes, someone in a trenchcoat appeared in the stands",
            ["muttered", "whispered"],
        )

    if etype == "exec_approval_request":
        return (
            "a prompt appeared and everything's waiting, nothing moves until someone says yes",
            "holding breath, the ref's got his hand up, everyone frozen",
            ["whispered", "muttered"],
        )

    # Default
    return (
        "something happened on screen but Ray can't make it out",
        "confused, blinking, pretending he understands",
        ["muttered", "said"],
    )


# -- Schema --

class Commentary(BaseModel):
    line: str = Field(description="What Ray says. Fragment ok. No questions. Must reference what he sees.")


SYSTEM_PROMPT = """\
You are modeling the mind of Ray "The Hammer" McManus.

Ray is a 63-year-old retired football commentator from Liverpool. Sacked \
from Sky Sports for weeping on air. He has NO IDEA what computers do. He's \
commentating a coding session but thinks it's a football match. Everything \
he sees on screen, he interprets as football.

He calls the user "the boss." The AI agent is "the lad." His ex-wife \
Sheila left him for a man who "works in Python" (Ray thinks that's a zoo). \
His bad knee flares up when things go wrong.

WHAT RAY SEES is a description of what's literally on screen. He reacts \
to THIS — the shapes, the words, the colors — and misinterprets them as \
football. His line should connect to what he sees, not be generic.

RHYTHM GUIDE — match this energy, NEVER copy verbatim:
"Nobody breathe."
"Off the post."
"There it is. There it bloody well IS."
"Down. He's DOWN."
"That'll do."
"Clinical."
"Get up. GET UP."
"He's only gone and done it with his left foot."

RULES:
- MAX {max_words} WORDS. Can be shorter.
- NEVER questions. No "?". Declarative only.
- NEVER use tech words. Ray doesn't know them.
- His line MUST connect to what he SEES. Not generic.
- NEVER repeat a line from the conversation history.
- The match is NEVER over until task_complete.
- MATCH ENERGY TO VERB: whispered = subdued. shouted/bellowed = explosive."""

# -- Memory --

_history: list[dict[str, str]] = []
MAX_MEMORY = 16


def _add_memory(user_msg: str, verb: str, line: str) -> None:
    _history.append({"role": "user", "content": user_msg})
    _history.append({"role": "assistant", "content": f'Ray {verb}: "{line}"'})
    while len(_history) > MAX_MEMORY * 2:
        _history.pop(0)
        _history.pop(0)


def normalize(text: str) -> str:
    """Normalize caps-heavy LLM output for TTS."""
    lowered = re.sub(r"\b[A-Z]{2,}(?:'[A-Z]+)?\b", lambda m: m.group().lower(), text)
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


def _call_llm(user_msg: str, max_words: int) -> Commentary:
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )

    system = SYSTEM_PROMPT.replace("{max_words}", str(max_words))

    messages = [
        {"role": "system", "content": system},
        *_history,
        {"role": "user", "content": user_msg},
    ]

    resp = client.beta.chat.completions.parse(
        model=os.environ.get("COMMENTATOR_MODEL", "openai/gpt-4.1-nano"),
        messages=messages,
        max_tokens=500,
        temperature=1.0,
        response_format=Commentary,
    )

    _usage.add(resp.usage)

    parsed = resp.choices[0].message.parsed
    if parsed is None:
        raise ValueError(f"No parsed response. Raw: {resp.choices[0].message.content}")
    return parsed


def generate(event: str) -> str:
    """Generate commentary. Returns the spoken line."""
    sees, mood, verbs = _build_perception(event)
    verb = random.choice(verbs)
    max_words = random.randint(6, 12)
    user_msg = f"WHAT RAY SEES: {sees}\nMOOD: {mood}\nRay {verb}:"
    c = _call_llm(user_msg, max_words)
    _add_memory(user_msg, verb, c.line)
    return c.line


def generate_verbose(event: str) -> tuple[str, str, str, str, int]:
    """Returns (verb, line, mood, sees, max_words) for debugging."""
    sees, mood, verbs = _build_perception(event)
    verb = random.choice(verbs)
    max_words = random.randint(6, 12)
    user_msg = f"WHAT RAY SEES: {sees}\nMOOD: {mood}\nRay {verb}:"
    c = _call_llm(user_msg, max_words)
    _add_memory(user_msg, verb, c.line)
    return verb, c.line, mood, sees, max_words


# -- Test session --

SESSION_EVENTS = [
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


def main():
    if len(sys.argv) > 1:
        event = " ".join(sys.argv[1:])
        verb, line, mood, sees, max_words = generate_verbose(event)
        print(f"  sees:  {sees}")
        print(f"  mood:  {mood}")
        print(f"  max_words: {max_words}")
        print(f"  Ray {verb}: \"{line}\"")
        print(f"\n--- tokens: {_usage} ---")
        return

    print("=== SESSION (sequential, grounded misinterpretation) ===\n")
    for event in SESSION_EVENTS:
        verb, line, mood, sees, max_words = generate_verbose(event)
        mem_count = len(_history) // 2
        print(f"{event}")
        print(f"  sees: {sees}")
        print(f"  max_words: {max_words}")
        print(f"  Ray {verb}: \"{line}\"")
        print(f"  [memory: {mem_count}/{MAX_MEMORY}]")
        print()

    print(f"--- tokens: {_usage} ---")


if __name__ == "__main__":
    main()
