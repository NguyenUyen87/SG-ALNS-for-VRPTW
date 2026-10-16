#!/usr/bin/env bash
# Package the final submission: LaTeX sources + PDFs, and the complete source code with data, raw results and
# trained models of the paper (symlinks are resolved). Output: release/SG-ALNS_<date>/ and release/SG-ALNS_<date>.zip
#   ./make_release.sh
set -e
cd "$(dirname "$0")"
ROOT=$PWD
NAME=SG-ALNS_$(date +%Y%m%d)
OUT=$ROOT/release/$NAME
rm -rf "$OUT" "$OUT.zip"
mkdir -p "$OUT/latex" "$OUT/source"

# ---- LaTeX: everything needed to compile, no build artefacts
cd "$ROOT/paper"
rsync -a --exclude '_stage' --exclude '*.aux' --exclude '*.log' --exclude '*.out' --exclude '*.fls' \
      --exclude '*.fdb_latexmk' --exclude '*.blg' --exclude '*.synctex.gz' --exclude '.DS_Store' \
      --exclude '*_v1.pdf' --exclude 'SG_ALNS__*.pdf' --exclude '*.docx' --exclude 'references_with_doi.bib' \
      ./ "$OUT/latex/"
mkdir -p "$OUT/pdf"
cp main.pdf main_anonymous.pdf supplementary.pdf supplementary_anonymous.pdf main_lncs.pdf "$OUT/pdf/"

# ---- source code, data, raw results and models of the paper
cd "$ROOT"
rsync -a --exclude '__pycache__' --exclude '.DS_Store' sgalns scripts configs tests data "$OUT/source/"
cp README.md requirements.txt run_all.sh run_revision.sh run_revision2.sh run_revision2b.sh run_revision2c.sh run_revision3.sh \
   make_paper.sh make_release.sh "$OUT/source/"
mkdir -p "$OUT/source/results_paper"
rsync -aL --exclude '.DS_Store' --exclude 'pilot*' --exclude 'smoke*' --exclude 'probe_solomon.csv' \
      results_paper/results results_paper/logs "$OUT/source/results_paper/"
ln -s ../latex "$OUT/source/paper" 2>/dev/null || true

cp "$ROOT/paper/Doi_chieu_nhan_xet.md" "$OUT/"
cd "$ROOT/release"
zip -qry "$NAME.zip" "$NAME"
du -sh "$NAME" "$NAME.zip"
