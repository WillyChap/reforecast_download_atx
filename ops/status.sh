#!/usr/bin/env bash
# Progress: reforecast dates on S3, bias fields built, and the end of the latest log.
REPO="$(cd "$(dirname "$0")/.." && pwd)"
source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-reforecast}"
cd "$REPO/src"
python - <<'PY'
import common as C
from download import reforecast_dates, FIRST_50R1
import pandas as pd
dates = reforecast_dates(FIRST_50R1, pd.Timestamp.now("UTC").tz_localize(None))
done = C.hindcast_folders()
print(f"reforecast dates on S3: {len(done)} of {len(dates)} published so far")
for var in C.VARIABLES:
    src = C.read_json(C.source_manifest_path(var), {})
    days = C.read_json(C.final_manifest_path(var), {})
    print(f"  {var:5s} bias: {len(src):3d} start dates, {len(days):3d} of 365 calendar days")
PY
running=$(flock -n "$REPO/logs/.lock" true 2>/dev/null && echo no || echo yes)
echo "run in progress: $running"
latest=$(ls -t "$REPO"/logs/run_*.log 2>/dev/null | head -1)
[ -n "$latest" ] && { echo "--- $latest"; tail -n 15 "$latest"; }
