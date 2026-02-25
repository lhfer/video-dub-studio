#!/usr/bin/env bash
set -euo pipefail

REPO_NAME="${1:-video-dub-studio}"
VISIBILITY="${2:-public}"

if ! command -v gh >/dev/null 2>&1; then
  echo "gh command not found. Install with: brew install gh"
  exit 1
fi

if ! gh auth status >/dev/null 2>&1; then
  echo "GitHub not authenticated. Run: gh auth login"
  exit 1
fi

if [[ "$VISIBILITY" != "public" && "$VISIBILITY" != "private" ]]; then
  echo "visibility must be public or private"
  exit 1
fi

# create repo if not exists
if ! gh repo view "$REPO_NAME" >/dev/null 2>&1; then
  gh repo create "$REPO_NAME" --"$VISIBILITY" --source=. --remote=origin --description "Convert YouTube/local videos into multilingual dubbed audio with Qwen ASR + translation + TTS."
else
  if git remote get-url origin >/dev/null 2>&1; then
    echo "origin exists: $(git remote get-url origin)"
  else
    gh repo set-default "$REPO_NAME"
    git remote add origin "https://github.com/$(gh api user -q .login)/$REPO_NAME.git"
  fi
fi

git branch -M main
git push -u origin main

echo "Published: https://github.com/$(gh api user -q .login)/$REPO_NAME"
