#!/usr/bin/env bash
set -o errexit
set -o nounset
set -o pipefail

# Go to git project root
cd "$(cd -P -- "$(dirname -- "$0")" && pwd -P)"
cd "$(git rev-parse --show-toplevel)"

# Check if flake8 is installed
if ! command -v flake8 1>/dev/null 2>&1; then
    printf "%s\n" "Flake8 not installed!"
    exit 1
fi

black {balrogo,tests}
