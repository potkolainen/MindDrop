#!/usr/bin/env bash
set -euo pipefail

# Run the full Tauri + Vite dev app from the repo root.
cd "$(dirname "${BASH_SOURCE[0]}")/.."

npm run tauri:dev
