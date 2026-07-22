#!/usr/bin/env bash
# Fetch the DCBO authors' stack (Aglietti et al., NeurIPS 2021;
# github.com/neildhir/DCBO) into third_party/DCBO, pinned to the commit the
# QDCBO quotient layer was written against. third_party/ is gitignored.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$REPO_ROOT/third_party/DCBO"
PIN=${DCBO_PIN:-85a9bdf}

if [ -d "$DEST/.git" ]; then
    echo "already present: $DEST"
else
    git clone https://github.com/neildhir/DCBO "$DEST"
fi
if [ "$PIN" != "HEAD" ]; then
    git -C "$DEST" checkout -q "$PIN"
fi
echo "DCBO at $(git -C "$DEST" rev-parse --short HEAD)"
