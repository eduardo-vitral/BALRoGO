#!/usr/bin/env bash
set -o errexit
set -o nounset
set -o pipefail

# Go to git project root
cd "$(cd -P -- "$(dirname -- "$0")" && pwd -P)"
cd "$(git rev-parse --show-toplevel)"

# Check if poetry is installed
if ! command -v poetry 1>/dev/null 2>&1; then
    printf "%s\n" "Poetry not installed!"
    exit 1
fi

poetry build
