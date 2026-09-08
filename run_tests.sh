#!/usr/bin/env bash
# Runs every test file in its own pytest process.
# Per-file isolation avoids the legacy stub-pollution some test modules
# introduce into sys.modules when sharing a single process.
set -u
cd "$(dirname "$0")"
overall=0
for f in tests/test_*.py; do
    printf '%-72s' "$f"
    if python -m pytest "$f" -q >/tmp/pytest_out.txt 2>&1; then
        echo "PASS  ($(grep -oE '[0-9]+ passed' /tmp/pytest_out.txt | head -1))"
    else
        overall=1
        echo "FAIL"
        tail -20 /tmp/pytest_out.txt
    fi
done
if [ "$overall" -eq 0 ]; then
    echo "ALL TEST FILES PASSED"
fi
exit "$overall"
