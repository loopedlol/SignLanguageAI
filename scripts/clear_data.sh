#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PROCESSED_DIR="${PROJECT_ROOT}/data/processed_landmarks"
NORMALIZED_DIR="${PROJECT_ROOT}/data/normalized_landmarks"

echo "This will remove:"
echo "  ${PROCESSED_DIR}"
echo "  ${NORMALIZED_DIR}"
echo
read -r -p "Are you sure you want to delete recorded landmark data? Type YES to continue: " CONFIRMATION

if [[ "${CONFIRMATION}" != "YES" ]]; then
  echo "Cancelled. Recorded landmark data was not deleted."
  exit 0
fi

rm -rf "${PROCESSED_DIR}"
rm -rf "${NORMALIZED_DIR}"

mkdir -p "${PROCESSED_DIR}"
mkdir -p "${NORMALIZED_DIR}"

echo "Recorded landmark data deleted and empty folders recreated."
