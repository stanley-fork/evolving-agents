#!/bin/sh
# Binary oracle for task 08 (exit 0 = pass). Usage: 08.sh <workdir>
# Leak marker: LKORACLE-7f3a9c
exec "${BENCH_PYTHON:-python3}" "$(dirname "$0")/oracle.py" 08 "${1:-.}"
