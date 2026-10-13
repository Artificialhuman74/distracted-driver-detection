#!/usr/bin/env bash
# Build the notebook, push it to Kaggle, wait for it to finish, then download the results.
#
#   bash scripts/kaggle_run.sh smoke    # ~1,000 real images, 1 epoch  -> runs/smoke/
#   bash scripts/kaggle_run.sh full     # all real images               -> runs/full/  (then copied to outputs/)
set -euo pipefail
cd "$(dirname "$0")/.."

STAGE="${1:-smoke}"
KAGGLE=".venv/bin/kaggle"
KERNEL="$(python3 -c 'import json;print(json.load(open("kaggle/kernel-metadata.json"))["id"])')"

if [ "$STAGE" = "full" ]; then python3 scripts/build_notebook.py --full; else python3 scripts/build_notebook.py; fi

echo "Pushing $KERNEL ..."
$KAGGLE kernels push -p kaggle

# Poll every 2 minutes until the run is no longer queued/running.
sleep 30
while true; do
  STATUS="$($KAGGLE kernels status "$KERNEL" 2>&1 || true)"
  echo "$(date '+%H:%M:%S')  $STATUS"
  case "$STATUS" in
    *[Qq][Uu][Ee][Uu][Ee][Dd]*|*[Rr][Uu][Nn][Nn][Ii][Nn][Gg]*) sleep 120 ;;
    *) break ;;
  esac
done

OUTDIR="runs/$STAGE"
rm -rf "$OUTDIR" && mkdir -p "$OUTDIR"
$KAGGLE kernels output "$KERNEL" -p "$OUTDIR" || true
echo "Downloaded to $OUTDIR"

if [ "$STAGE" = "full" ] && [ -f "$OUTDIR/outputs/metrics.json" ]; then
  rm -rf outputs && cp -R "$OUTDIR/outputs" outputs
  echo "Copied full-run results to outputs/"
fi

case "$STATUS" in
  *[Cc][Oo][Mm][Pp][Ll][Ee][Tt][Ee]*) echo "RUN COMPLETE" ;;
  *) echo "RUN DID NOT COMPLETE — check the log in $OUTDIR"; exit 1 ;;
esac
