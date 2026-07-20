#!/usr/bin/env bash
# Fetch the CEO authors' stack (Branchini et al., AISTATS 2023) into
# third_party/CEO, pinned to the commit the misspec adapter was written
# against. third_party/ is gitignored, like CausalBO_Benchmark.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$REPO_ROOT/third_party/CEO"
PIN=35dd277

if [ -d "$DEST/.git" ]; then
    echo "already present: $DEST"
else
    git clone https://github.com/nicola144/CEO "$DEST"
fi
git -C "$DEST" checkout -q "$PIN"
echo "CEO at $(git -C "$DEST" rev-parse --short HEAD)"
