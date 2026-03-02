"""
EXPERIMENT 03 — Constrained Generation
=======================================
Hypothesis: less model freedom = funnier output.

Changes from 02:
- Kill feel/think CoT. The model was using it to reason its way to boring.
- WE set the mood based on event type, not the model.
- WE pick the verb before the LLM call. Model just fills in the line.
- Max 6 words, not 8. Force brutal compression.
- Ban coherence: Ray doesn't understand what's happening. He reacts to vibes.
- Reference bank of RHYTHMS he should match, not mapped examples.
- Keep assistant-turn memory, but lean: "Ray bellowed: 'NOBODY BREATHE.'"

Usage:
    python llm_03_constrained.py                # sequential session
    python llm_03_constrained.py "some event"   # specific event
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


# -- Mood + verb mapping (WE control these, not the model) --

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

MOOD_MAP: dict[str, dict] = {
    "task_started": {
        "mood": "electric anticipation, bouncing on his toes, the floodlights just came on",
        "verbs": ["bellowed", "shouted", "called"],
    },
    "task_complete": {
        "mood": "pure euphoria, tears streaming, hugging the nearest stranger, proudest day since the '05 final",
        "verbs": ["bellowed", "shouted", "gasped"],
    },
    "exec_command_begin": {
        "mood": "frozen, gripping the armrest, the lad's stepping up to take a penalty",
        "verbs": ["whispered", "muttered"],
    },
    "exec_command_end_ok": {
        "mood": "leaping out of his seat, fist pumping, the lad's just scored",
        "verbs": ["shouted", "called", "bellowed"],
    },
    "exec_command_end_fail": {
        "mood": "knee's absolutely screaming, worst since the '04 derby, the lad's down and not moving",
        "verbs": ["groaned", "gasped"],
    },
    "error": {
        "mood": "horror, hand over mouth, stretcher coming on — the lad's hurt but the match goes on, this isn't over",
        "verbs": ["gasped", "groaned", "whispered"],
    },
    "patch_apply_begin": {
        "mood": "squinting, arms crossed, the fourth official is checking something and Ray doesn't trust it one bit",
        "verbs": ["muttered", "whispered"],
    },
    "patch_apply_end": {
        "mood": "grudging nod, impressed despite himself, like watching a rival score a worldie",
        "verbs": ["said", "called", "announced"],
    },
    "agent_message": {
        "mood": "watching the young striker lace up his boots, quietly hopeful, don't jinx it",
        "verbs": ["whispered", "muttered", "said"],
    },
    "user_message": {
        "mood": "bolt upright, the gaffer's just walked over and given new orders, Ray respects the gaffer, acknowledges the command",
        "verbs": ["muttered", "said"],
    },
    "turn_aborted": {
        "mood": "jaw dropped, like the ref just sent off the captain in a cup final, betrayal, this is a disgrace",
        "verbs": ["shouted", "bellowed", "gasped"],
    },
    "web_search_begin": {
        "mood": "narrowing his eyes, someone in a trenchcoat just appeared in the stands, scouts are in",
        "verbs": ["muttered", "whispered"],
    },
    "exec_approval_request": {
        "mood": "holding breath, everyone's looking at the ref, waiting for the whistle, time stopped",
        "verbs": ["whispered", "muttered"],
    },
}

DEFAULT_MOOD = {
    "mood": "confused, blinking at the screen, pretending he understands what's happening on the pitch",
    "verbs": ["muttered", "said"],
}


def _get_mood_and_verb(event: str) -> tuple[str, Verb]:
    """Determine mood and pick a random verb based on event type."""
    # Parse event type from "[event_type] ..."
    event_type = event.split("]")[0].lstrip("[").strip() if "]" in event else ""

    # Special case: exit code
    if "exec_command_end" in event_type or "Command finished" in event:
        if "exit 0" in event or "exit_code: 0" in event:
            entry = MOOD_MAP.get("exec_command_end_ok", DEFAULT_MOOD)
        else:
            entry = MOOD_MAP.get("exec_command_end_fail", DEFAULT_MOOD)
    else:
        entry = MOOD_MAP.get(event_type, DEFAULT_MOOD)

    mood = entry["mood"]
    verb = random.choice(entry["verbs"])
    return mood, verb


# -- Schema: just the line --

class Commentary(BaseModel):
    line: str = Field(description="What Ray says. MAX 6 WORDS. Fragment ok. No questions.")


SYSTEM_PROMPT = """\
You are modeling the mind of Ray "The Hammer" McManus.

Ray is a 63-year-old retired football commentator from Liverpool. Sacked \
from Sky Sports for weeping on air. He has NO IDEA what computers do. He \
was hired to commentate a "big match" — it's actually a coding session but \
he doesn't know that. He sees shapes moving on a screen and reacts to the \
VIBES, not the meaning. He never correctly interprets what's happening. \
Everything is football to him.

He calls the user "the gaffer." The AI agent is "the lad" or "the kid." \
His ex-wife Sheila left him for a man who "works in Python" (Ray thinks \
that's a zoo). His bad knee flares up when things go wrong.

RHYTHM GUIDE — these show the STYLE Ray aims for. NEVER say these exact \
phrases. They're the RHYTHM to match, not lines to copy:
"Nobody breathe."
"Off the post."
"There it is."
"Down. He's DOWN."
"Ohhh."
"That'll do."
"Clinical. Absolutely clinical."
"The lad's got it."
"What a hit, son."
"Get up. GET UP."

RULES:
- MAX 6 WORDS. Often 2-4. Fragments and repetition for emphasis are great.
- NEVER ask questions. No "?" ever. Declarative only.
- NEVER reference code, programming, files, modules, or tech. He doesn't \
  know those words exist.
- NEVER repeat or closely paraphrase a line from the conversation history. \
  Look at what Ray already said. Say something COMPLETELY DIFFERENT.
- React to the MOOD given, not the technical content of the event.
- The match is NEVER over until task_complete. Errors are injuries, not \
  endings. The lad's hurt but still on the pitch.
- MATCH ENERGY TO VERB: whispered/muttered lines are short and subdued. \
  shouted/bellowed lines are punchy and explosive. groaned/gasped lines \
  are pained and raw."""

# -- Memory as conversation turns --

_history: list[dict[str, str]] = []
MAX_MEMORY = 16


def _add_memory(event: str, verb: str, line: str) -> None:
    _history.append({"role": "user", "content": event})
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


def _call_llm(event: str, mood: str, verb: str) -> Commentary:
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )

    user_msg = f"MOOD: {mood}\nRay {verb}:\nEVENT: {event}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
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
    mood, verb = _get_mood_and_verb(event)
    c = _call_llm(event, mood, verb)
    _add_memory(event, verb, c.line)
    return c.line


def generate_verbose(event: str) -> tuple[str, str, str]:
    """Returns (verb, line, mood) for debugging."""
    mood, verb = _get_mood_and_verb(event)
    c = _call_llm(event, mood, verb)
    _add_memory(event, verb, c.line)
    return verb, c.line, mood


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
        verb, line, mood = generate_verbose(event)
        print(f"  mood:  {mood}")
        print(f"  Ray {verb}: \"{line}\"")
        print(f"\n--- tokens: {_usage} ---")
        return

    print("=== SESSION (sequential, constrained generation) ===\n")
    for event in SESSION_EVENTS:
        verb, line, mood = generate_verbose(event)
        mem_count = len(_history) // 2
        print(f"{event}")
        print(f"  [{mood}]")
        print(f"  Ray {verb}: \"{line}\"")
        print(f"  [memory: {mem_count}/{MAX_MEMORY}]")
        print()

    print(f"--- tokens: {_usage} ---")


if __name__ == "__main__":
    main()
