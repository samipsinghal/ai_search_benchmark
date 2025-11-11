#!/usr/bin/env bash
# Benchmark all runs across dev/eval qrels -> clean TSV + Markdown table.
# Compatible with macOS default shell (bash 3.x / zsh).
set -euo pipefail
shopt -s nullglob

QRELS_DIR="data"
RUNS_DIR="runs"
REPORT_DIR="reports"
OUTFILE="$REPORT_DIR/metrics_summary.tsv"
OUTCLEAN="$REPORT_DIR/metrics_summary.clean.tsv"
OUTMD="$REPORT_DIR/metrics_summary.md"

mkdir -p "$REPORT_DIR"

# Write header once
echo -e "run\tqrels_type\tMRR@10\tRecall@100\tNDCG@10\tNDCG@100\tMAP" > "$OUTFILE"

# ---------- helper functions ----------
find_qrels() {
  local c
  for c in "$@"; do
    if [ -f "$QRELS_DIR/$c" ]; then echo "$c"; return 0; fi
  done
  return 1
}

run_block() {
  local qrels_file="$1"
  local pattern="$2"
  local matched=()
  local f
  for f in $RUNS_DIR/$pattern; do matched+=("$f"); done
  if [ ${#matched[@]} -eq 0 ]; then
    echo "[WARN] No runs match pattern $pattern" | tee -a "$OUTFILE" >/dev/null
    return 0
  fi
  local run_args=()
  for f in "${matched[@]}"; do run_args+=("--run" "$f"); done
  echo "Evaluating $qrels_file on ${#matched[@]} run(s)..."
  python -m scripts.report_effectiveness --qrels "$QRELS_DIR/$qrels_file" "${run_args[@]}" >> "$OUTFILE"
}

# ---------- evaluate all splits ----------
echo "=============================================================="
echo " Evaluating DEV (binary qrels)"
echo "=============================================================="
DEV=$(find_qrels "qrels.dev.tsv" "qrels.dev.txt" || true)
[ -n "${DEV:-}" ] && run_block "$DEV" "*dev*.trec" || echo "[WARN] No DEV qrels found"

echo "=============================================================="
echo " Evaluating EVAL1 (graded qrels)"
echo "=============================================================="
EVAL1=$(find_qrels "qrels.eval.one.tsv" "qrel.eval.one.tsv" "qrels.dl19-passage.txt" || true)
[ -n "${EVAL1:-}" ] && run_block "$EVAL1" "*eval1*.trec" || echo "[WARN] No EVAL1 qrels found"

echo "=============================================================="
echo " Evaluating EVAL2 (graded qrels)"
echo "=============================================================="
EVAL2=$(find_qrels "qrels.eval.two.tsv" "qrel.eval.two.tsv" "qrels.dl20-passage.txt" || true)
[ -n "${EVAL2:-}" ] && run_block "$EVAL2" "*eval2*.trec" || echo "[WARN] No EVAL2 qrels found"

# ---------- clean + make Markdown ----------
awk -F'\t' '
BEGIN {have_header=0}
# skip comments
/^#/ {next}
# keep first header
$1=="run" {if(!have_header){print $0; have_header=1}; next}
# data rows
{print $0}
' "$OUTFILE" > "$OUTCLEAN"

# Pretty Markdown
{
  echo "### Retrieval Benchmark Summary (BM25 vs HNSW vs Rerank)"
  echo
  echo "| Split | System | MRR@10 | Recall@100 | NDCG@10 | NDCG@100 | MAP |"
  echo "|-------|---------|--------|-------------|----------|-----------|------|"

  awk -F'\t' '
  NR>1 {
    # infer split name
    split($1, arr, "/")
    fname = arr[length(arr)]
    split(fname, parts, ".")
    sys = parts[1]
    split_tag=""
    if (fname ~ /dev/) split_tag="Dev (binary)"
    else if (fname ~ /eval1/) split_tag="Eval 1 (graded)"
    else if (fname ~ /eval2/) split_tag="Eval 2 (graded)"
    # system label
    if (fname ~ /bm25/) label="BM25"
    else if (fname ~ /hnsw/) label="HNSW"
    else if (fname ~ /rerank/) label="Rerank"
    else label=parts[1]

    printf("| %s | %s | %.3f | %.3f | %.3f | %.3f | %.3f |\n",
           split_tag, label, $3, $4, $5, $6, $7)
  }' "$OUTCLEAN"
} > "$OUTMD"

echo "=============================================================="
echo " Outputs generated:"
echo " - $OUTFILE   (raw TSV)"
echo " - $OUTCLEAN  (clean TSV)"
echo " - $OUTMD     (Markdown table)"
echo "--------------------------------------------------------------"
column -t -s $'\t' "$OUTCLEAN" || true
echo
cat "$OUTMD"
