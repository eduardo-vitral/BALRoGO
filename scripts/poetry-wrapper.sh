#!/usr/bin/env bash
set -o errexit
set -o nounset
set -o pipefail

RED='\033[1;31m'
NC='\033[0m'

balrogo_say() {
    printf "\n"😈" ${RED}%s${NC}\n" "$@"
    printf "\n"
}

# Check if poetry is installed
if ! command -v poetry 1>/dev/null 2>&1; then
    balrogo_say "Poetry not installed!" "https://python-poetry.org/docs/#installation"
    exit 1
fi

exec 3>&1
if ! FF=$(script --flush --return -q /dev/null -c "poetry $*" | tee >(cat - >&3)); then
    if [[ $FF == *"[Errno 2] No such file or directory"* ]]; then
        balrogo_say "Did you run 'make setup' ?"
    fi
    exit 1
fi
