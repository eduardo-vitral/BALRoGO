#!/usr/bin/env bash
set -o errexit
set -o nounset
set -o pipefail

# Go to git project root
cd "$(cd -P -- "$(dirname -- "$0")" && pwd -P)"
cd "$(git rev-parse --show-toplevel)"

python -m unittest 
