#!/bin/bash
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN_PATH="${PROJECT_DIR}/bin/edge_arm_control"

if [ ! -f "${BIN_PATH}" ]; then
    echo "[Error] Binary not found at ${BIN_PATH}. Please run ./scripts/build.sh first."
    exit 1
fi

echo "=== [Starting EdgeAI-Robotic-Arm-Control Pipeline] ==="
"${BIN_PATH}"
