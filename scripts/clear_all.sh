#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PROCESSED_DIR="${PROJECT_ROOT}/data/processed_landmarks"
NORMALIZED_DIR="${PROJECT_ROOT}/data/normalized_landmarks"
CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints_30"

echo "This will remove:"
echo "  ${PROCESSED_DIR}"
echo "  ${NORMALIZED_DIR}"
echo "  ${CHECKPOINT_DIR}"
echo
read -r -p "Are you sure you want to delete recorded landmark data and trained checkpoints? Type YES to continue: " CONFIRMATION

if [[ "${CONFIRMATION}" != "YES" ]]; then
  echo "Cancelled. Recorded landmark data and trained checkpoints were not deleted."
  exit 0
fi

rm -rf "${PROCESSED_DIR}"
rm -rf "${NORMALIZED_DIR}"
rm -rf "${CHECKPOINT_DIR}"

mkdir -p "${PROCESSED_DIR}"
mkdir -p "${NORMALIZED_DIR}"
mkdir -p "${CHECKPOINT_DIR}"

echo "Recorded landmark data and trained checkpoints deleted; empty folders recreated."
