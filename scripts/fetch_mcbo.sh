#!/usr/bin/env bash
# Fetch the MCBO authors' stack (Sussex et al., ICLR 2023;
# github.com/ssethz/mcbo) into third_party/mcbo, pinned to the commit the
# QMCBO quotient layer was written against. third_party/ is gitignored.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$REPO_ROOT/third_party/mcbo"
PIN=${MCBO_PIN:-0d0650e}

if [ -d "$DEST/.git" ]; then
    echo "already present: $DEST"
else
    git clone https://github.com/ssethz/mcbo "$DEST"
fi
if [ "$PIN" != "HEAD" ]; then
    git -C "$DEST" checkout -q "$PIN"
fi
echo "mcbo at $(git -C "$DEST" rev-parse --short HEAD)"
