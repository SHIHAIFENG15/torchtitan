#!/usr/bin/env bash
# Wait for a free 8-GPU window, then exec the given command.
#
# Window criteria: every card has less than FREE_MIB_GIB*1024 MiB in use.
# Historical bug this guards against: requiring `nvidia-smi --query-compute-apps`
# count to be zero, which is never true on this box because of a long-lived
# atk server (~644 MiB) on device 0.
#
# Optional env:
#   POLL_SECONDS  poll interval            (default 60)
#   FREE_MIB_GIB  per-card budget in GiB   (default 2)
#   MAX_WAIT_MIN  give up after N minutes  (default 0 = no limit)
#   RUN_NOW       if 1, exec the command immediately, no wait
#
# Usage: scripts/watch_and_run.sh <cmd> [args...]
# Exits with the executed command's code.
set -u
POLL_SECONDS=${POLL_SECONDS:-60}
FREE_MIB=$(( ${FREE_MIB_GIB:-2} * 1024 ))
MAX_WAIT_MIN=${MAX_WAIT_MIN:-0}
RUN_NOW=${RUN_NOW:-0}

if [ "$#" -lt 1 ]; then
    echo "usage: $0 <cmd> [args...]" >&2
    exit 2
fi

log(){ echo "[watch_and_run] $*"; }

busy_cards(){
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk -v lim="$FREE_MIB" '$1>lim' | wc -l
}

if [ "$RUN_NOW" != "1" ]; then
    started=$(date +%s)
    while :; do
        n=$(busy_cards)
        if [ "$n" -eq 0 ]; then
            log "all GPUs under ${FREE_MIB} MiB — running: $*"
            break
        fi
        log "$(date +%H:%M:%S) — $n card(s) above ${FREE_MIB} MiB, waiting ${POLL_SECONDS}s"
        if [ "$MAX_WAIT_MIN" != "0" ]; then
            if [ $(( ( $(date +%s) - started ) / 60 )) -ge "$MAX_WAIT_MIN" ]; then
                log "giving up after ${MAX_WAIT_MIN} min"
                exit 3
            fi
        fi
        sleep "$POLL_SECONDS"
    done
else
    log "RUN_NOW=1 — skipping wait, running: $*"
fi

exec "$@"
