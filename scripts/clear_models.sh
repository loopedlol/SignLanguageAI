#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints_30"

echo "This will remove:"
echo "  ${CHECKPOINT_DIR}"
echo
read -r -p "Are you sure you want to delete trained checkpoints? Type YES to continue: " CONFIRMATION

if [[ "${CONFIRMATION}" != "YES" ]]; then
  echo "Cancelled. Trained checkpoints were not deleted."
  exit 0
fi

rm -rf "${CHECKPOINT_DIR}"

mkdir -p "${CHECKPOINT_DIR}"

echo "Trained checkpoints deleted and empty folder recreated."
