#!/usr/bin/env bash
# Download new reforecasts from MARS, then update the bias fields they affect.
# Used for the first backfill and by cron every two weeks. Safe to re-run: every
# step skips finished work. Only one run at a time (lock file).
#
#   bash ops/run.sh
#
# Settings (environment): CONDA_BASE (default ~/miniconda3), CONDA_ENV (default reforecast),
# REFCAST_WORKERS / REFCAST_WORKER_MEMORY (dask, default 2 x 6GB).
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$REPO/logs"
exec 9>"$REPO/logs/.lock"
if ! flock -n 9; then
    echo "$(date -u '+%F %T') another run is active; exiting"
    exit 0
fi
LOG="$REPO/logs/run_$(date -u +%Y%m%d_%H%M).log"
exec >>"$LOG" 2>&1
export PYTHONUNBUFFERED=1
source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-reforecast}"
cd "$REPO/src"
echo "$(date -u '+%F %T') run start"

# A step is tried 3 times, 10 minutes apart. A failed download does not stop the
# bias update (dates already on S3 are still used); a failed bias step does.
# Expired AWS credentials (an SSO login lasts only hours) are waited out, not counted
# as attempts: the run checks every 5 minutes until someone logs in again.
wait_for_aws() {
    aws sts get-caller-identity >/dev/null 2>&1 && return 0
    echo "$(date -u '+%F %T') AWS credentials expired or missing; waiting (run 'aws sso login' or fix the instance role)"
    until aws sts get-caller-identity >/dev/null 2>&1; do sleep 300; done
    echo "$(date -u '+%F %T') AWS credentials valid again"
}

run_step() {
    attempt=1
    while [ "$attempt" -le 3 ]; do
        wait_for_aws
        python "$@" && return 0
        if ! aws sts get-caller-identity >/dev/null 2>&1; then
            echo "$(date -u '+%F %T') $1 stopped by expired AWS credentials"
            continue
        fi
        echo "$(date -u '+%F %T') $1 failed (attempt $attempt)"
        attempt=$((attempt + 1))
        [ "$attempt" -le 3 ] && sleep 600
    done
    return 1
}

status=0
run_step download.py || { echo "$(date -u '+%F %T') download incomplete; continuing with the dates on S3"; status=1; }
for step in step1_era5.py step2_bias.py step3_bias.py; do
    if ! run_step "$step"; then
        echo "$(date -u '+%F %T') $step FAILED; stopping"
        exit 2
    fi
done
echo "$(date -u '+%F %T') run end (status $status)"
exit $status
