#!/usr/bin/env bash
# Run the studies of one sweep script in parallel (resumable: completed runs are skipped).
# Usage: examples/run_parallel.sh <sweep script> [study ...]
#   e.g. examples/run_parallel.sh examples/lorenz/sweeps_lorenz63.py ensemble_size inflation
# Default: every study in the STUDIES of the script. Logs: results/<script>_<study>.log next to the script.
# Exits with 1 if any study fails.
set -e
[ $# -ge 1 ] && [ -f "$1" ] || { echo "Usage: $0 <sweep script> [study ...]" >&2; exit 1; }
cd "$(dirname "$1")"
script=$(basename "$1" .py)
shift
[ $# -eq 0 ] && set -- $(python -c "import $script as s; print(*s.STUDIES)" | tail -n 1)

mkdir -p results
pids=()
for study in "$@"; do
    MPLBACKEND=Agg python "$script.py" --study "$study" > "results/${script}_$study.log" 2>&1 &
    pids+=($!)
done
rc=0
for pid in "${pids[@]}"; do
    wait "$pid" || rc=1
done
exit $rc
