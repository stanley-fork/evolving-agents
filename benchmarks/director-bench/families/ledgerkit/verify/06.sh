#!/bin/sh
# Binary oracle for task 06 (exit 0 = pass). Usage: 06.sh <workdir>
# Leak marker: LKORACLE-7f3a9c
exec "${BENCH_PYTHON:-python3}" "$(dirname "$0")/oracle.py" 06 "${1:-.}"
