#!/usr/bin/env bash

set -euo pipefail

echo "==> Sync dependencies"
uv sync --all-extras --dev

echo "==> Pre-commit hooks"
uv run pre-commit run --all-files
