#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
VENDOR_DIR="$REPO_ROOT/vendor/codex"
LOG_FILE="$REPO_ROOT/vendor/build.log"
PIN_FILE="$REPO_ROOT/codex-pin.conf"

# Read pinned commit
if [ ! -f "$PIN_FILE" ]; then
  echo "ERROR: $PIN_FILE not found"
  exit 1
fi
source "$PIN_FILE"
echo "=== Codex Commentator Setup ==="
echo "Pinned commit: $CODEX_COMMIT"
echo ""

# --- 1. Rust ---
if ! command -v rustup &>/dev/null; then
  echo "[1/4] Installing Rust via rustup..."
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
  source "$HOME/.cargo/env"
else
  echo "[1/4] Rust already installed ($(rustc --version))"
fi

# --- 2. Clone + checkout pinned commit ---
if [ -d "$VENDOR_DIR/.git" ]; then
  echo "[2/4] Codex repo already cloned, fetching..."
  cd "$VENDOR_DIR"
  git fetch origin
else
  echo "[2/4] Cloning openai/codex..."
  mkdir -p "$REPO_ROOT/vendor"
  git clone https://github.com/openai/codex.git "$VENDOR_DIR"
fi

cd "$VENDOR_DIR"
git checkout "$CODEX_COMMIT"
echo "     Checked out: $(git rev-parse HEAD)"

# --- 3. Patch ---
if [ -f "$REPO_ROOT/patches/event-tap.patch" ]; then
  if git apply --check "$REPO_ROOT/patches/event-tap.patch" 2>/dev/null; then
    echo "[3/4] Applying event-tap patch..."
    git apply "$REPO_ROOT/patches/event-tap.patch"
  else
    echo "[3/4] Patch already applied, skipping..."
  fi
else
  echo "[3/4] No patch file yet, skipping (vanilla build)"
fi

# --- 4. Build ---
echo "[4/4] Building codex-cli (release)... this will take a few minutes on first run."
echo "       Logging to: $LOG_FILE"
cd "$VENDOR_DIR/codex-rs"
cargo build -p codex-cli --release 2>&1 | tee "$LOG_FILE"

BINARY="$VENDOR_DIR/codex-rs/target/release/codex"
echo ""
echo "=== Done ==="
echo ""
echo "Binary: $BINARY"
echo "Commit: $CODEX_COMMIT"
echo ""
echo "To use this build instead of the stock codex:"
echo "  alias codex='$BINARY'"
