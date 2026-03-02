"""
EXPERIMENT 06 — Fragment Bursts (Mini)
=======================================
Same as 05 but using gpt-4.1-mini instead of nano. Testing whether
a slightly bigger model produces better commentary.

Fix: after Ray's first reaction, the LLM decides whether Ray has more to say.
If yes, it generates 1-3 more sentence fragments that extend the thought,
like someone live-commenting in bursts:

  "Off the post." → "Wait." → "No, it's IN."

or just:
  "Clinical."  (no follow-up)

Each event can trigger a sequence of fragments with varying lengths and energy.

Carries forward from 04:
- "What Ray sees" perception layer
- Pre-set mood + verb
- Randomized max words (6-12)
- Assistant-turn memory, 16 slots
- Pydantic schemas

Usage:
    python llm_05_fragments.py                # sequential session
    python llm_05_fragments.py "some event"   # specific event
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


# -- Event → what Ray sees + mood + verbs (same as 04) --

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
        sees = random.choice([
            "blank screen, cursor blinking, dead silent — about to kick off",
            "empty terminal, nothing yet, the calm before it all starts",
            "screen just appeared, cursor waiting, everyone holding their breath",
            "fresh window opened, completely empty, the pitch is bare",
            "dark screen with a single blinking cursor, like a ref's whistle about to blow",
        ])
        return sees, "electric anticipation, bouncing on his toes, floodlights coming on", ["bellowed", "shouted", "called"]

    if etype == "task_complete":
        sees = random.choice([
            "everything stopped, the cursor came back, it's over — silence",
            "screen went still, no more scrolling, just a quiet prompt sitting there",
            "all the activity died down, cursor blinking peacefully, finished",
            "nothing moving anymore, the terminal is calm, like the final whistle just blew",
            "the screen settled, one last line appeared, and then nothing — done",
        ])
        return sees, "pure euphoria, tears forming, proudest moment since the '05 final", ["bellowed", "shouted", "gasped"]

    if etype == "exec_command_begin":
        cmd = detail.replace("Running:", "").strip()[:100] if "Running:" in detail else "something"
        sees = random.choice([
            f"'{cmd}' just appeared and text started scrolling fast",
            f"the screen lit up with '{cmd}' and now output is flying by",
            f"'{cmd}' popped up and lines of text are pouring down the screen",
            f"someone typed '{cmd}' and the whole screen started moving",
            f"'{cmd}' kicked off and now there's text everywhere, scrolling down",
        ])
        return sees, "frozen, gripping the armrest, the lad's stepping up", ["whispered", "muttered"]

    if etype == "exec_command_end":
        if "exit 0" in event:
            sees = random.choice([
                "the scrolling stopped, output finished cleanly, prompt came back — looks like it worked",
                "everything went quiet, no red text, just a clean finish and the cursor waiting",
                "the command finished, no errors visible, screen is calm again",
                "output stopped flowing, everything looks green, the prompt returned",
                "it all settled down, ran clean, no complaints — back to the blinking cursor",
            ])
            return sees, "leaping out of his seat, fist pumping, the lad's just scored", ["shouted", "bellowed", "called"]
        else:
            sees = random.choice([
                "the command choked, red text splashed across the screen, something went wrong",
                "output stopped abruptly, angry red lines appeared, it failed",
                "error text everywhere, the command didn't finish clean, screen looks bad",
                "red warnings flooded the terminal, the command crashed out",
                "it broke — red text, the output cut short, the screen looks like a crime scene",
            ])
            return sees, "knee's screaming, worst since the '04 derby, the lad's down", ["groaned", "gasped"]

    if etype == "error":
        words = detail.replace("Error:", "").strip().split()
        skip = {"error", "cannot", "could", "from", "that", "this", "with", "file", "find", "such", "module", "directory", "no"}
        visible = [w.strip("'\"") for w in words if len(w) > 3 and w.lower().strip("'\"") not in skip]
        keyword = visible[0] if visible else "something"
        sees = random.choice([
            f"red error text on screen, the word '{keyword}' stands out in the mess",
            f"the terminal went red, '{keyword}' is the big word in the error message",
            f"angry red text with '{keyword}' right in the middle of it",
            f"error splashed across the screen, '{keyword}' keeps appearing in the red text",
            f"wall of red, and '{keyword}' is the word that jumps out most",
        ])
        return sees, "horror, hand over mouth, stretcher on the pitch — but the match goes on", ["gasped", "groaned", "whispered"]

    if etype == "patch_apply_begin":
        sees = random.choice([
            "code is changing on screen — lines highlighted, some green, some red, things being rewritten",
            "the file is being edited, lines getting added and removed, green and red diffs everywhere",
            "text shifting around on screen, old lines crossed out in red, new ones appearing in green",
            "a diff appeared — chunks of code being swapped out, the screen is half green half red",
            "lines of code flickering, deletions in red, additions in green, the file is being rewritten",
        ])
        return sees, "squinting, arms crossed, the fourth official is reviewing something suspicious", ["muttered", "whispered"]

    if etype == "patch_apply_end":
        sees = random.choice([
            "the changes landed, the diff disappeared, the file looks clean now",
            "editing finished, the green and red are gone, code settled into place",
            "the patch went through, screen is calm again, new code sitting there quietly",
            "all the changes applied, no more flickering, the file looks different but stable",
            "done editing, the code snapped into its new shape, everything still",
        ])
        return sees, "grudging nod, impressed despite himself, like a rival scoring a worldie", ["said", "called", "announced"]

    if etype == "agent_message":
        short = detail.replace("Agent:", "").strip()[:100]
        sees = random.choice([
            f"the AI is typing: '{short}'",
            f"text appearing on screen, the lad's writing: '{short}'",
            f"a message is being written out: '{short}'",
            f"words forming on screen from the AI: '{short}'",
            f"the lad just wrote: '{short}'",
        ])
        return sees, "watching the young striker warm up, quietly hopeful, don't jinx it", ["whispered", "muttered", "said"]

    if etype == "user_message":
        msg = detail.replace("User:", "").strip()[:100]
        if len(msg) > 80:
            sees = random.choice([
                f"a long message appeared out of nowhere: '{msg}'",
                f"a wall of text just landed on screen from somewhere: '{msg}'",
                f"instructions came in, a whole paragraph: '{msg}'",
                f"a big block of words appeared from the tunnel: '{msg}'",
                f"someone sent a long note onto the pitch: '{msg}'",
            ])
        else:
            sees = random.choice([
                f"a message appeared out of nowhere: '{msg}'",
                f"words just showed up on screen: '{msg}'",
                f"new instructions came in from somewhere: '{msg}'",
                f"a note appeared, like it was passed from the stands: '{msg}'",
                f"something was typed in: '{msg}'",
            ])
        return sees, "bolt upright, new instructions from somewhere, no idea who sent them", ["muttered", "said"]

    if etype == "turn_aborted":
        sees = random.choice([
            "everything just stopped mid-sentence, the screen froze, nothing's moving",
            "the AI was in the middle of something and it all just cut off",
            "screen went dead still, mid-action, like someone pulled the plug",
            "abrupt stop — the output was flowing and then just... nothing",
            "cut off mid-flow, the cursor is just sitting there, abandoned",
        ])
        return sees, "jaw dropped, like the ref just sent off the captain in a cup final", ["shouted", "bellowed", "gasped"]

    if etype == "web_search_begin":
        sees = random.choice([
            "something is being looked up, search results loading on screen",
            "a web search just kicked off, new text streaming in from outside",
            "the lad's searching for something online, results pouring in",
            "search results appearing, text flying in from the internet",
            "a lookup started, external content loading onto the screen",
        ])
        return sees, "narrowing his eyes, someone in a trenchcoat appeared in the stands", ["muttered", "whispered"]

    if etype == "exec_approval_request":
        sees = random.choice([
            "a prompt appeared asking for permission, everything's paused, waiting for a yes",
            "the screen is frozen with a question — nothing moves until someone approves",
            "a confirmation dialog popped up, the whole thing is on hold",
            "everything stopped, there's a yes/no prompt, waiting for the user to decide",
            "the lad wants to do something but needs permission first, screen is waiting",
        ])
        return sees, "holding breath, the ref's got his hand up, everyone frozen", ["whispered", "muttered"]

    # Default
    sees = random.choice([
        "something happened on screen but it's hard to tell what",
        "the screen flickered, something changed but unclear what",
        "text appeared briefly, couldn't quite make it out",
        "something shifted on screen, Ray's not sure what he's looking at",
        "a flash of activity on screen, gone before he could read it",
    ])
    return sees, "confused, blinking, pretending he understands", ["muttered", "said"]


# -- Schemas --

class Reaction(BaseModel):
    line: str = Field(description="What Ray says. Fragment ok. No questions. Must reference what he sees.")


class FollowUp(BaseModel):
    line: str = Field(description="Ray's next fragment. Extends, corrects, or builds on his last line. Short. No questions.")


SYSTEM_PROMPT = """\
You are modeling the mind of Ray "The Hammer" McManus.

Ray is a 63-year-old retired football commentator from Liverpool. Sacked \
from Sky Sports for weeping on air. He has NO IDEA what computers do. He's \
commentating a coding session but thinks it's a football match. Everything \
he sees on screen, he interprets as football.

The AI agent is "the lad." Ray has no idea who's giving instructions — \
they just appear from somewhere, like a voice from the stands or a note \
passed from the tunnel. He never names or addresses the user. His ex-wife \
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
- MAX {max_words} WORDS per fragment. Can be shorter.
- NEVER questions. No "?". Declarative only.
- NEVER use tech words. Ray doesn't know them.
- His line MUST connect to what he SEES. Not generic.
- NEVER repeat a line from the conversation history.
- The match is NEVER over until task_complete.
- MATCH ENERGY TO VERB: whispered = subdued. shouted/bellowed = explosive.

FRAGMENT BURSTS:
Sometimes Ray adds a trailing thought — a correction, a memory, a mutter.
  "Off the post." → "Wait." → "It's IN."
  "He's down." → "Knee. Same one as mine."
  "Clinical." (done)
  "Gaffer means business." (done)
Most events: 1 fragment. That's the default — punchy, done.
Sometimes 2 if he trails off or corrects himself.
Rarely 3 — only for the biggest moments (goals, disasters, final whistle).
NEVER more than 3 total."""


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


# -- Fragment probabilities --
# After first line: 40% chance of a 2nd fragment
# After 2nd: 15% chance of a 3rd
FOLLOWUP_CHANCES = [0.55, 0.30]


def _call_llm(user_msg: str, max_words: int, schema=Reaction, extra_messages: list | None = None):
    """Generic LLM call with a pydantic schema."""
    client = _get_client()
    system = SYSTEM_PROMPT.replace("{max_words}", str(max_words))

    messages = [
        {"role": "system", "content": system},
        *_history,
        {"role": "user", "content": user_msg},
    ]
    if extra_messages:
        messages.extend(extra_messages)

    resp = client.beta.chat.completions.parse(
        model="openai/gpt-4.1-mini",
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


def generate(event: str) -> str:
    """Generate commentary. Returns the full spoken line (fragments joined)."""
    sees, mood, verbs = _build_perception(event)
    verb = random.choice(verbs)
    max_words = random.randint(6, 12)
    user_msg = f"WHAT RAY SEES: {sees}\nMOOD: {mood}\nRay {verb}:"

    first = _call_llm(user_msg, max_words)
    fragments = [first.line]

    # Coin flip for follow-ups
    for chance in FOLLOWUP_CHANCES:
        if random.random() >= chance:
            break
        followup_words = random.randint(3, 8)
        so_far = " ... ".join(fragments)
        extra = [
            {"role": "assistant", "content": f'Ray {verb}: "{so_far}" ...'},
            {"role": "user", "content": "Ray continues, adding to that thought:"},
        ]
        fu = _call_llm(user_msg, followup_words, schema=FollowUp, extra_messages=extra)
        fragments.append(fu.line)

    full_line = " ... ".join(fragments)
    _add_memory(user_msg, verb, full_line)
    return full_line


def generate_verbose(event: str) -> tuple[str, list[str], str, str, int]:
    """Returns (verb, fragments, mood, sees, max_words) for debugging."""
    sees, mood, verbs = _build_perception(event)
    verb = random.choice(verbs)
    max_words = random.randint(6, 12)
    user_msg = f"WHAT RAY SEES: {sees}\nMOOD: {mood}\nRay {verb}:"

    first = _call_llm(user_msg, max_words)
    fragments = [first.line]

    for chance in FOLLOWUP_CHANCES:
        if random.random() >= chance:
            break
        followup_words = random.randint(3, 8)
        so_far = " ... ".join(fragments)
        extra = [
            {"role": "assistant", "content": f'Ray {verb}: "{so_far}" ...'},
            {"role": "user", "content": "Ray continues, adding to that thought:"},
        ]
        fu = _call_llm(user_msg, followup_words, schema=FollowUp, extra_messages=extra)
        fragments.append(fu.line)

    full_line = " ... ".join(fragments)
    _add_memory(user_msg, verb, full_line)
    return verb, fragments, mood, sees, max_words


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
    print(f"=== SESSION {label} (fragment bursts) ===\n")
    for event in events:
        verb, fragments, mood, sees, max_words = generate_verbose(event)
        mem_count = len(_history) // 2
        print(f"{event}")
        print(f"  sees: {sees}")
        print(f"  max_words: {max_words}")
        for i, frag in enumerate(fragments):
            tag = f"  Ray {verb}:" if i == 0 else "         ...:"
            print(f'{tag} "{frag}"')
        print(f"  [memory: {mem_count}/{MAX_MEMORY}] [fragments: {len(fragments)}]")
        print()
    print(f"--- tokens: {_usage} ---")


def main():
    args = sys.argv[1:]

    # --test v1 / --test v2 / (default = v1)
    if "--test" in args:
        idx = args.index("--test")
        version = args[idx + 1] if idx + 1 < len(args) else "v1"
        if version not in SESSIONS:
            print(f"Unknown test: {version}. Available: {', '.join(SESSIONS)}")
            return
        _run_session(SESSIONS[version], version)
        return

    # Single event
    if args:
        event = " ".join(args)
        verb, fragments, mood, sees, max_words = generate_verbose(event)
        print(f"  sees:  {sees}")
        print(f"  mood:  {mood}")
        print(f"  max_words: {max_words}")
        for i, frag in enumerate(fragments):
            tag = f"  Ray {verb}:" if i == 0 else "         ...:"
            print(f'{tag} "{frag}"')
        print(f"\n--- tokens: {_usage} ---")
        return

    # Default: v1
    _run_session(SESSION_V1, "v1")


if __name__ == "__main__":
    main()
