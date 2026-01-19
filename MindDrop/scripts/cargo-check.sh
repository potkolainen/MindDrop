#!/usr/bin/env bash
set -euo pipefail

# Backend-only compile/typecheck.
cd "$(dirname "${BASH_SOURCE[0]}")/../src-tauri"

cargo check
