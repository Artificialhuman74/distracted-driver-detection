#!/usr/bin/env bash
# Driver-grouped cross-validation on Google Colab (scripts/cv_experiment.py), robust to disconnects.
#   bash scripts/colab_cv.sh [GPU] [SEEDS...]      e.g.  bash scripts/colab_cv.sh T4 1 2
# The VM loads the data once, then trains ONE FOLD at a time; results are downloaded and merged into
# runs/cv/results_seed<S>.jsonl after every fold. If the connection drops, re-running resumes at the first missing fold.
set -uo pipefail
cd "$(dirname "$0")/.."
COLAB=".venv/bin/colab"; GPU="${1:-T4}"; shift || true
SEEDS=("$@"); [ "${#SEEDS[@]}" -eq 0 ] && SEEDS=(0 1 2)
mkdir -p runs/cv
TOKEN_FILE="$HOME/.kaggle/access_token"; [ -f "$TOKEN_FILE" ] || TOKEN_FILE="$HOME/.kaggle/kaggle.json"
limit() { local secs="$1"; shift; perl -e 'alarm shift; exec @ARGV' "$secs" "$@"; }   # stop any CLI call that hangs

missing_folds() {   # folds of this seed that do not yet have all 3 variants locally
  python3 - "$1" <<'EOF'
import json, sys, pathlib
s = int(sys.argv[1]); f = pathlib.Path(f"runs/cv/results_seed{s}.jsonl")
rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()] if f.exists() else []
done = {r["fold"] for r in rows if r.get("seed") == s and r["variant"] != "imagenet"
        and sum(1 for q in rows if q["fold"] == r["fold"] and q.get("seed") == s and q["variant"] in ("baseline", "A", "C")) == 3}
print(" ".join(str(k) for k in range(5) if k not in done))
EOF
}
merge() {   # merge downloaded lines into the local file, keeping one line per (fold, seed, variant)
  python3 - "$1" "$2" <<'EOF'
import json, sys, pathlib
local, new = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
rows = {}
for f in (local, new):
    if f.exists():
        for l in f.read_text().splitlines():
            if l.strip():
                r = json.loads(l); rows[(r["fold"], r.get("seed"), r["variant"])] = l
local.write_text("\n".join(rows.values()) + "\n")
print(f"{len(rows)} lines in {local}")
EOF
}

for SEED in "${SEEDS[@]}"; do
  for ATTEMPT in 1 2 3; do
    TODO="$(missing_folds "$SEED")"
    [ -z "$TODO" ] && { echo "== seed $SEED complete"; break; }
    S="dd-f28-cv-s$SEED-a$ATTEMPT"
    echo "== seed $SEED attempt $ATTEMPT: folds to run: $TODO (session $S)"
    sed "s/^SEEDS = \[0\]/SEEDS = [$SEED]/" scripts/cv_experiment.py > "runs/cv/cv_seed$SEED.py"
    limit 600 $COLAB new -s "$S" --gpu "$GPU" || { echo "SESSION FAILED"; sleep 60; continue; }
    limit 300 $COLAB upload -s "$S" "$TOKEN_FILE" /content/kaggle_access_token
    limit 1200 $COLAB exec -s "$S" -f scripts/colab_setup.py
    echo 'SETUP_ONLY = True' | limit 120 $COLAB exec -s "$S"
    limit 900 $COLAB exec -s "$S" -f "runs/cv/cv_seed$SEED.py" 2>&1 | tee -a "runs/cv/log_seed$SEED.txt" | grep -E "loaded|rror"
    for K in $TODO; do
      echo "== seed $SEED fold $K"
      printf 'for _n, _s, _l in VARIANTS:\n    run(%s, %s, _n, _s, _l)\n' "$K" "$SEED" | limit 2400 $COLAB exec -s "$S" 2>&1 | tee -a "runs/cv/log_seed$SEED.txt" | grep -E "fold [0-9] seed|rror"
      if limit 300 $COLAB download -s "$S" /content/cv/results.jsonl "runs/cv/tmp_seed$SEED.jsonl"; then
        merge "runs/cv/results_seed$SEED.jsonl" "runs/cv/tmp_seed$SEED.jsonl"
      else
        echo "DOWNLOAD FAILED (connection lost?) after seed $SEED fold $K"; break
      fi
    done
    limit 120 $COLAB stop -s "$S" < /dev/null || true
  done
  echo "== seed $SEED finished: missing folds now: [$(missing_folds "$SEED")]"
done
echo "CV ALL SEEDS DONE"
