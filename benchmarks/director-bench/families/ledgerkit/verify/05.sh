#!/bin/sh
# Binary oracle for task 05 (exit 0 = pass). Usage: 05.sh <workdir>
# Leak marker: LKORACLE-7f3a9c
exec "${BENCH_PYTHON:-python3}" "$(dirname "$0")/oracle.py" 05 "${1:-.}"
