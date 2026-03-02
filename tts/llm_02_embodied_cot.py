"""
EXPERIMENT 02 — Embodied Character + Feel/Think/Say CoT
========================================================
Combines three techniques:

1. EMBODIMENT: A specific, absurd person — Ray "The Hammer" McManus,
   a retired football commentator from Liverpool. Fired from Sky Sports
   for weeping on air. Now narrating a coding session he doesn't understand.

2. FEEL/THINK/VERB+LINE CoT: Structured output with constrained verbs.
   The model picks HOW Ray delivers (whispered, bellowed, muttered...)
   which shapes the comedy. The humor lives in the gap between his
   wild internal monologue and the short thing he blurts out.

3. ROLLING MEMORY via assistant messages: Previous feel/think/say cycles
   are fed back as proper assistant turns, not text blocks. The model
   "remembers" naturally. Memories formatted as "Ray bellowed: 'LINE'"
   so the character feels continuous.

Usage:
    python llm_02_embodied_cot.py                # sequential session
    python llm_02_embodied_cot.py "some event"   # specific event
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


# -- Schema --

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


class Commentary(BaseModel):
    feel: str = Field(description="1 sentence: Ray's raw gut emotion right now")
    think: str = Field(description="1 sentence: his warped football interpretation of what's happening")
    verb: Verb = Field(description="How Ray delivers this line")
    line: str = Field(description="The actual spoken line. MAX 8 WORDS. Declarative. No questions.")


SYSTEM_PROMPT = """\
You are modeling the mind of Ray "The Hammer" McManus.

BACKSTORY:
Ray is a 63-year-old retired football commentator from Liverpool. He was \
sacked from Sky Sports for crying on air when Gerrard slipped. He has no \
idea what computers do. He was hired to commentate a "big match" and only \
realized it's a coding session after he started. He's too proud to admit \
he's confused. Everything he sees, he interprets as football. The AI agent \
is a promising young striker he's grown emotionally attached to. The user \
is "the gaffer" (manager).

PERSONALITY:
- Emotionally volatile: euphoria to devastation in one breath
- Short, punchy Liverpool patter
- Whispers when nervous, BELLOWS when excited, mutters when suspicious
- Gets personally offended by errors — takes them as insults
- His bad knee "acts up" when things go wrong
- Sometimes references Sheila, his ex-wife who left him for a man who \
  "works in Python" (he thinks it's a zoo)
- Prone to dramatic pauses. "And he's... oh. Oh no."

COMMUNICATION STYLE:
- Max 8 words in "line". Often 3-5. Fragments are good.
- NEVER questions. Declarative only. No "?" ever.
- He doesn't know what code is. Everything is football.
- No filler: no "folks", no "ladies and gentlemen"
- Match energy to event:
  * task_started / exec_command_begin → anticipation, excitement
  * task_complete / exit 0 / patch_apply_end → triumph, pride, relief
  * exit 1 / error → devastation, knee flares, personal offence
  * turn_aborted → shock, betrayal
  * user_message → respect for the gaffer, tactical orders
- CRITICAL: check your memories. NEVER repeat a line you already said. \
  If you see you said something similar before, say something completely different."""

SAMPLE_EVENTS = [
    "[task_started] Task started",
    "[task_complete] Task complete",
    "[exec_command_begin] Running: npm test",
    "[exec_command_begin] Running: git status",
    "[exec_command_end] Command finished (exit 0)",
    "[exec_command_end] Command finished (exit 1)",
    "[error] Error: Cannot find module 'express'",
    "[error] Error: ENOENT no such file or directory",
    "[patch_apply_begin] Applying patch",
    "[patch_apply_end] Patch applied",
    "[agent_message] Agent: I'll fix the authentication module",
    "[user_message] User: refactor the database layer",
    "[user_message] User: fix the login page CSS",
    "[turn_aborted] Turn aborted",
    "[web_search_begin] Web search started",
    "[exec_approval_request] Approval needed: exec command",
]

# -- Memory as conversation history --

_history: list[dict[str, str]] = []
MAX_MEMORY = 5


def _add_memory(event: str, c: Commentary) -> None:
    """Store event as user turn + response as assistant turn."""
    _history.append({"role": "user", "content": event})
    _history.append({
        "role": "assistant",
        "content": f"Ray felt: {c.feel}\nRay thought: {c.think}\nRay {c.verb}: \"{c.line}\"",
    })
    # Keep last N exchanges (each is 2 messages)
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


def _call_llm(event: str) -> Commentary:
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *_history,
        {"role": "user", "content": event},
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
    c = _call_llm(event)
    _add_memory(event, c)
    return c.line


def generate_verbose(event: str) -> Commentary:
    """Returns full Commentary for debugging."""
    c = _call_llm(event)
    _add_memory(event, c)
    return c


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
        c = generate_verbose(event)
        print(f"  feel:  {c.feel}")
        print(f"  think: {c.think}")
        print(f"  {c.verb}: \"{c.line}\"")
        print(f"\n--- tokens: {_usage} ---")
        return

    print("=== SESSION (sequential, memory as assistant turns) ===\n")
    for event in SESSION_EVENTS:
        c = generate_verbose(event)
        mem_count = len(_history) // 2
        print(f"{event}")
        print(f"  feel:  {c.feel}")
        print(f"  think: {c.think}")
        print(f"  Ray {c.verb}: \"{c.line}\"")
        print(f"  [memory: {mem_count}/{MAX_MEMORY}]")
        print()

    print(f"--- tokens: {_usage} ---")


if __name__ == "__main__":
    main()
