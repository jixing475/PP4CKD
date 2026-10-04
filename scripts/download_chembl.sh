#!/usr/bin/env bash
# ==============================================================================
# download_chembl.sh: Download & Extract Official ChEMBL SQLite Database
# ==============================================================================
set -euo pipefail

CHEMBL_VERSION="${1:-36}"
TARGET_DIR="${2:-./raw_data}"
CHEMBL_TAR="chembl_${CHEMBL_VERSION}_sqlite.tar.gz"
DOWNLOAD_URL="https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_${CHEMBL_VERSION}/${CHEMBL_TAR}"

echo "================================================================================"
echo "Downloading ChEMBL ${CHEMBL_VERSION} SQLite Database"
echo "URL:         ${DOWNLOAD_URL}"
echo "Destination: ${TARGET_DIR}"
echo "================================================================================"

mkdir -p "${TARGET_DIR}"

ARCHIVE_PATH="${TARGET_DIR}/${CHEMBL_TAR}"

if [ -f "${ARCHIVE_PATH}" ]; then
    echo "Existing archive detected: ${ARCHIVE_PATH}"
    echo "Resuming or verifying download..."
else
    echo "Starting download..."
fi

if command -v curl >/dev/null 2>&1; then
    curl -C - -L --retry 5 --retry-delay 3 -o "${ARCHIVE_PATH}" "${DOWNLOAD_URL}"
elif command -v wget >/dev/null 2>&1; then
    wget -c -O "${ARCHIVE_PATH}" "${DOWNLOAD_URL}"
else
    echo "Error: Neither curl nor wget found in PATH." >&2
    exit 1
fi

echo "Download completed: ${ARCHIVE_PATH} ($(du -h "${ARCHIVE_PATH}" | cut -f1))"

EXTRACT_DIR="${TARGET_DIR}/chembl_${CHEMBL_VERSION}"
echo "Extracting archive to ${EXTRACT_DIR}..."
mkdir -p "${EXTRACT_DIR}"
tar -xzf "${ARCHIVE_PATH}" -C "${EXTRACT_DIR}"

DB_FILE=$(find "${EXTRACT_DIR}" -name "chembl_${CHEMBL_VERSION}.db" | head -n 1)

if [ -n "${DB_FILE}" ] && [ -f "${DB_FILE}" ]; then
    echo "================================================================================"
    echo "Success! ChEMBL ${CHEMBL_VERSION} SQLite database ready at:"
    echo "${DB_FILE} ($(du -h "${DB_FILE}" | cut -f1))"
    echo "================================================================================"
else
    echo "Error: chembl_${CHEMBL_VERSION}.db not found after extraction." >&2
    exit 1
fi
