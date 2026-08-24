#!/bin/bash
set -e

echo "=== [Building EdgeAI-Robotic-Arm-Control] ==="

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${PROJECT_DIR}/build"

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

echo "=== [Build Finished Successfully!] ==="
echo "Binary output: ${PROJECT_DIR}/bin/edge_arm_control"
