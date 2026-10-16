#!/usr/bin/env bash
# Regenerate every number, table and figure of the paper from results/ and compile the PDFs.
#   ./make_paper.sh                 (uses ./results and ./logs)
#   RESULTS=/path/results LOGS=/path/logs ./make_paper.sh   (e.g. the original runs)
set -e
cd "$(dirname "$0")"
ROOT=$PWD
PY=$ROOT/.venv/bin/python
RES=${RESULTS:-$ROOT/results}
LOGD=${LOGS:-$ROOT/logs}
STAGE=$ROOT/paper/_stage
GEN=$ROOT/paper/generated
rm -rf "$STAGE"; mkdir -p "$STAGE/results" "$GEN/tables" "$GEN/figs"
ln -s "$ROOT/data" "$STAGE/data"; ln -s "$LOGD" "$STAGE/logs"
for f in "$RES"/*; do [ "$(basename "$f")" = e1_solomon.csv ] || ln -s "$f" "$STAGE/results/"; done

# headroom table needs the per-configuration format of the paper (re-run it once if needed: 3-4 minutes)
if ! head -1 "$RES/headroom.csv" | grep -q '^cfg'; then (cd "$ROOT" && $PY scripts/tune.py headroom > logs/tune_headroom.log); fi

# greedy actions of the iteration-level agents (not stored by evaluate.py for K=1): a few minutes, once
[ -f "$RES/iter_actions.csv" ] || (cd "$ROOT" && $PY scripts/check_iter_actions.py "$RES/iter_gnn/last.pt" "$RES/iter_mlp/last.pt" > "$LOGD/iter_actions.log" 2>&1 && mv results/iter_actions.csv "$RES/" 2>/dev/null || true)
[ -e "$STAGE/results/iter_actions.csv" ] || ln -s "$RES/iter_actions.csv" "$STAGE/results/" 2>/dev/null || true

# split the review runs (collapse-sensitivity policy and the two schedules) into e7_review.csv as in the paper
$PY - "$RES" "$STAGE/results" <<'PYEOF'
import sys, pandas as pd, os
res, out = sys.argv[1:]
e1 = pd.read_csv(f"{res}/e1_solomon.csv", keep_default_na=False)
rv = open(f"{res}/dyn_headroom_best.txt").read().split()
review = e1.method.isin(["sg:main_ent"] + rv)
if not os.path.exists(f"{res}/e7_review.csv"):
    e1[review].to_csv(f"{out}/e7_review.csv", index=False)
e1[~review].to_csv(f"{out}/e1_solomon.csv", index=False)
PYEOF

cd "$STAGE"
export PNG_DIR=$STAGE MPLBACKEND=Agg
# embed TrueType (Type 42) fonts in every figure: IEEE PDF eXpress rejects Type 3 fonts
printf "pdf.fonttype: 42\nps.fonttype: 42\n" > "$STAGE/matplotlibrc"
for s in make_paper_tables make_extra_tables make_review_tables make_analysis_figure make_more_figures make_fig_cd_entropy; do
  $PY "$ROOT/scripts/paper/$s.py" --out "$GEN" > "$STAGE/$s.log" 2>&1 || { echo "FAILED $s"; tail -20 "$STAGE/$s.log"; exit 1; }
done
$PY "$ROOT/scripts/paper/make_revision_tables.py" --out "$GEN" || exit 1     # overrides compare.tex and behaviour.pdf
$PY "$ROOT/scripts/paper/make_review2_tables.py" --out "$GEN" || exit 1     # second review: probe, tuning, reward, ...
$PY "$ROOT/scripts/paper/fit_tables.py" "$GEN/tables"
grep -l '{?}' "$GEN"/*.tex && echo "WARNING: unresolved numbers ({?}) in the files above (run scripts/verify_bks.py?)"

cd "$ROOT/paper"
for f in main_anonymous main supplementary_anonymous supplementary main_lncs; do
  latexmk -pdf -interaction=nonstopmode -halt-on-error $f.tex > /dev/null 2>&1 || { echo "LaTeX failed: $f"; grep -a -A3 '^!' $f.log | head; exit 1; }
  echo "$f.pdf: $(grep -a 'Output written' $f.log | grep -o '[0-9]* pages')$(grep -a -c 'Overfull' $f.log | sed 's/^/, overfull boxes: /')"
done
