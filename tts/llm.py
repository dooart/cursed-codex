"""
LLM commentary router.

Selects which experiment module to use. Change ACTIVE to switch.
Run directly to test: python llm.py

Experiments:
  01_baseline      — Simple prompt + examples. Generic output.
  02_embodied_cot  — Embodied character (Ray McManus) + feel/think/say CoT + memory.
  03_constrained   — Pre-set mood/verb, 6-word max, no CoT, rhythm bank.
  04_grounded      — "What Ray sees" layer. Lines connect to specific events.
  05_fragments     — Multi-fragment bursts. Ray trails off, adds thoughts, builds tension.
  06_fragments_mini — Same as 05 but with gpt-4.1-mini.
  07_direct        — Raw events, no perception layer. gpt-4.1-mini.
  08_local         — Same as 07 but local Ollama (qwen3.5:9b). No API key needed.
"""

# ---- ACTIVE EXPERIMENT ----
ACTIVE = "07_direct"
# ---------------------------

import importlib
import sys

_module = importlib.import_module(f"llm_{ACTIVE}")

generate = _module.generate
normalize = _module.normalize

if __name__ == "__main__":
    _module.main()
