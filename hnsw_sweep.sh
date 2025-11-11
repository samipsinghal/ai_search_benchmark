#!/usr/bin/env bash
set -euo pipefail
SAMPLE=data/queries.dev.sample100.tsv

declare -a CFGS=(
  "index/faiss_hnsw_M4C50 50 runs/S2_hnsw.sample100.M4C50S50.trec"
  "index/faiss_hnsw_M4C80 100 runs/S2_hnsw.sample100.M4C80S100.trec"
  "index/faiss_hnsw_M8C200 200 runs/S2_hnsw.sample100.M8C200S200.trec"
)

for cfg in "${CFGS[@]}"; do
  read IDX EFS OUT <<<"$cfg"
  echo "==> HNSW: $IDX efS=$EFS -> $OUT"
  python -m scripts.search_hnsw \
    --query_h5 data/msmarco_queries_dev_eval_embeddings.h5 \
    --qid_key id --qemb_key embedding \
    --index_dir "$IDX" \
    --qid_list_tsv "$SAMPLE" \
    --topk 1000 --efS "$EFS" \
    --run_out "$OUT" --tag FAISS
done
