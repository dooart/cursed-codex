"""
EXPERIMENT 01 — Baseline
========================
Simple system prompt with examples. Single LLM call, no CoT.

PROBLEMS:
- Output is generic, "safe average" — model picks the most probable response
- Vague hedging: "something's coming", "what's coming next?"
- Questions (real commentators never ask)
- Doesn't relate to the actual event ("sluggish start" for a user_message)
- Too long — often 10-15 words despite 12-word rule
- No character depth — "sports commentator" is too broad, model averages
  across every commentator it's ever seen
- No memory — can't avoid repeats, can't build narrative arc
- Examples are good but model averages across them into mush

Usage:
    python llm_01_baseline.py                # 10 random events
    python llm_01_baseline.py "some event"   # specific event
"""

import os
import random
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# Load .env from parent or app dir
for p in [Path(__file__).parent.parent / ".env", Path(__file__).parent.parent / "app" / ".env"]:
    if p.exists():
        load_dotenv(p)
        break

SYSTEM_PROMPT = """\
You are a sports commentator narrating an AI coding session. You have no idea what code is. Everything is athletics to you.

HARD RULES:
- 12 words MAX. Shorter is better. One punchy sentence.
- Never explain code. Never use programming terms correctly.
- Vary your energy: sometimes deadpan, sometimes hype, sometimes ominous.
- No filler. No "folks". No "ladies and gentlemen". Just the call.

EXAMPLES — match this length and vibe, never repeat verbatim:

[task_started] → "WE ARE UNDERWAY."
[task_started] → "The whistle blows. Here we go."
[task_started] → "AND THEY'RE OFF."
[task_complete] → "AND THAT'S THE WHISTLE."
[task_complete] → "CLEAN FINISH ON THAT SEQUENCE."
[task_complete] → "That'll do. That'll absolutely do."
[exec_command_begin] → "TESTS ARE UP. NOBODY BREATHE."
[exec_command_begin] → "A BIG COMMAND AT A BIG MOMENT."
[exec_command_begin] → "Bold move. Let's see if it pays."
[exec_command_end exit 0] → "THAT ONE LANDS."
[exec_command_end exit 0] → "AND IT GETS OVER THE LINE."
[exec_command_end exit 0] → "Textbook. Absolutely textbook."
[exec_command_end exit 1] → "OFF THE POST."
[exec_command_end exit 1] → "NOT TODAY. BACK TO BUILD-UP."
[exec_command_end exit 1] → "Oh that's a bad miss."
[error] → "STRETCHER ON THE PITCH."
[error] → "That's a career-ender right there."
[error] → "Down. He's DOWN."
[patch_apply_begin] → "LINES ARE MOVING."
[patch_apply_begin] → "Going to the replay. Hold on."
[patch_apply_end] → "Confirmed on review. PLAY STANDS."
[patch_apply_end] → "Clean hands. Clean finish."
[agent_message] → "The rookie calls the play."
[user_message] → "WORD FROM THE SIDELINE."
[user_message] → "Coach sends in the new formation."
[turn_aborted] → "HARD RESET FROM THE SIDELINE."
[turn_aborted] → "RED CARD. Off you go."
[web_search_begin] → "Sending scouts deep."
[exec_approval_request] → "Timeout on the field. Needs the nod."\
"""

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


def normalize(text: str) -> str:
    """Normalize caps-heavy LLM output for TTS."""
    import re
    # Lowercase any all-caps word (including contractions like DOESN'T)
    lowered = re.sub(r"\b[A-Z]{2,}(?:'[A-Z]+)?\b", lambda m: m.group().lower(), text)
    # Re-capitalize after sentence boundaries and at start
    return re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), lowered)


def generate(event: str) -> str:
    """Generate a single commentary quip for an event."""
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
    max_tokens = random.randint(40, 60)
    resp = client.chat.completions.create(
        model=os.environ.get("COMMENTATOR_MODEL", "openai/gpt-4.1-nano"),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": event},
        ],
        max_tokens=max_tokens,
        temperature=1.0,
    )
    return (resp.choices[0].message.content or "").strip()


def main():
    if len(sys.argv) > 1:
        event = " ".join(sys.argv[1:])
        quip = generate(event)
        print(f"  → {quip}")
        print(f"  → {normalize(quip)}  [normalized]")
        return

    events = random.sample(SAMPLE_EVENTS, min(10, len(SAMPLE_EVENTS)))
    for event in events:
        quip = generate(event)
        print(f"{event}")
        print(f"  → {quip}")
        print(f"  → {normalize(quip)}  [normalized]")
        print()


if __name__ == "__main__":
    main()
