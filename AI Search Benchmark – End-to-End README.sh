AI Search Benchmark – End-to-End README
0) Overview
Compare three IR systems on MSMARCO:
S1: BM25 (lexical)
S2: Dense retrieval (FAISS HNSW, inner-product)
S3: BM25→Dense re-rank (hybrid)
Evaluate on:
Dev (binary qrels) → MAP, Recall@100
Eval-1 (TREC DL’19) → MRR@10, NDCG@10/100, Recall@100
Eval-2 (TREC DL’20) → MRR@10, NDCG@10/100, Recall@100
1) Data layout (what you need in data/)
collection.tsv
msmarco_passages_subset.tsv
msmarco_passages_embeddings_subset.h5
msmarco_queries_dev_eval_embeddings.h5
queries.dev.tsv
queries.eval.tsv
qrels.dev.tsv
qrels.eval.one.tsv
qrels.eval.two.tsv
2) Environment
python3 -m venv .venv
source .venv/bin/activate
pip install faiss-cpu h5py numpy tqdm pytrec_eval
3) Build BM25 index (S1)
PYTHONPATH=. python -m src.index_build \
  --input data/collection.tsv \
  --outdir index/final \
  --batch_docs 50000
4) Validate subset alignment & create page table (critical)
PYTHONPATH=. python -m scripts.validate_subset_alignment \
  --collection data/collection.tsv \
  --subset data/msmarco_passages_subset.tsv \
  --h5 data/msmarco_passages_embeddings_subset.h5 \
  --write_mapping data/page_table.tsv    # writes internal→external map if needed
5) Build FAISS HNSW (S2 index)
PYTHONPATH=. python -m scripts.build_hnsw \
  --emb_h5 data/msmarco_passages_embeddings_subset.h5 \
  --out_dir index/faiss_hnsw_M4C80 \
  --M 4 --efC 80
(You can later try M=8, efC up to 200.)
6) Prepare filtered query lists (fast & aligned)
Dev (1k judged qids):
awk '{print $1}' data/qrels.dev.tsv | sort -u > data/qids.dev.txt
awk 'NR==FNR{keep[$1];next} ($1 in keep)' data/qids.dev.txt data/queries.dev.tsv > data/queries.dev.filtered.tsv
Eval-1 (DL’19):
awk '{print $1}' data/qrels.eval.one.tsv | sort -u > data/qids.eval1.txt
awk 'NR==FNR{keep[$1];next} ($1 in keep)' data/qids.eval1.txt data/queries.eval.tsv > data/queries.eval1.tsv
Eval-2 (DL’20):
awk '{print $1}' data/qrels.eval.two.tsv | sort -u > data/qids.eval2.txt
awk 'NR==FNR{keep[$1];next} ($1 in keep)' data/qids.eval2.txt data/queries.eval.tsv > data/queries.eval2.tsv
7) Convert qrels to TREC format
# Dev: binary (qid, 0, pid, rel)
awk '{print $1,"0",$2,$3}' data/qrels.dev.tsv > data/qrels.dev.trec

# Eval-1/Eval-2: graded (rel is 4th column)
awk '{print $1,"0",$3,$4}' data/qrels.eval.one.tsv > data/qrels.eval.one.trec
awk '{print $1,"0",$3,$4}' data/qrels.eval.two.tsv > data/qrels.eval.two.trec
8) Runs & Evaluation
8.1 BM25 (S1)
Dev
PYTHONPATH=. python -m scripts.search_to_run \
  --index_dir index/final \
  --queries data/queries.dev.filtered.tsv \
  --run_out runs/S1_bm25.dev.trec \
  --k1 0.9 --b 0.4 --mode disj --topk 1000 \
  --page_table data/page_table.tsv

trec_eval -m map -m recall.100 data/qrels.dev.trec runs/S1_bm25.dev.trec
Eval-1
PYTHONPATH=. python -m scripts.search_to_run \
  --index_dir index/final \
  --queries data/queries.eval1.tsv \
  --run_out runs/S1_bm25.eval1.trec \
  --k1 0.9 --b 0.4 --mode disj --topk 100 \
  --page_table data/page_table.tsv

trec_eval -m recip_rank.10 -m ndcg_cut.10 -m ndcg_cut.100 -m recall.100 \
  data/qrels.eval.one.trec runs/S1_bm25.eval1.trec
Eval-2
PYTHONPATH=. python -m scripts.search_to_run \
  --index_dir index/final \
  --queries data/queries.eval2.tsv \
  --run_out runs/S1_bm25.eval2.trec \
  --k1 0.9 --b 0.4 --mode disj --topk 100 \
  --page_table data/page_table.tsv

trec_eval -m recip_rank.10 -m ndcg_cut.10 -m ndcg_cut.100 -m recall.100 \
  data/qrels.eval.two.trec runs/S1_bm25.eval2.trec
8.2 Dense / HNSW (S2)
Dev
PYTHONPATH=. python -m scripts.search_hnsw \
  --query_h5 data/msmarco_queries_dev_eval_embeddings.h5 \
  --qid_key id --qemb_key embedding \
  --index_dir index/faiss_hnsw_M4C80 \
  --qid_list_tsv data/queries.dev.filtered.tsv \
  --topk 1000 --efS 100 \
  --run_out runs/S2_hnsw.dev.M4C80S100.trec --tag FAISS

# If trec_eval complains about duplicate (qid,docid), dedup then eval:
awk '!seen[$1,$3]++' runs/S2_hnsw.dev.M4C80S100.trec > runs/S2_hnsw.dev.M4C80S100_nodup.trec
trec_eval -m map -m recall.100 data/qrels.dev.trec runs/S2_hnsw.dev.M4C80S100_nodup.trec
Eval-1
PYTHONPATH=. python -m scripts.search_hnsw \
  --query_h5 data/msmarco_queries_dev_eval_embeddings.h5 \
  --qid_key id --qemb_key embedding \
  --index_dir index/faiss_hnsw_M4C80 \
  --qid_list_tsv data/queries.eval1.tsv \
  --topk 100 --efS 100 \
  --run_out runs/S2_hnsw.eval1.M4C80S100.trec --tag FAISS

trec_eval -m recip_rank.10 -m ndcg_cut.10 -m ndcg_cut.100 -m recall.100 \
  data/qrels.eval.one.trec runs/S2_hnsw.eval1.M4C80S100.trec
Eval-2
PYTHONPATH=. python -m scripts.search_hnsw \
  --query_h5 data/msmarco_queries_dev_eval_embeddings.h5 \
  --qid_key id --qemb_key embedding \
  --index_dir index/faiss_hnsw_M4C80 \
  --qid_list_tsv data/queries.eval2.tsv \
  --topk 100 --efS 100 \
  --run_out runs/S2_hnsw.eval2.M4C80S100.trec --tag FAISS

trec_eval -m recip_rank.10 -m ndcg_cut.10 -m ndcg_cut.100 -m recall.100 \
  data/qrels.eval.two.trec runs/S2_hnsw.eval2.M4C80S100.trec
8.3 BM25→Dense Re-rank (S3)
Dev
PYTHONPATH=. python -m scripts.rerank_bm25_dense \
  --bm25_run runs/S1_bm25.dev.trec \
  --query_h5 data/msmarco_queries_dev_eval_embeddings.h5 \
  --passage_h5 data/msmarco_passages_embeddings_subset.h5 \
  --qid_list_tsv data/queries.dev.filtered.tsv \
  --topk_in 1000 --topk_out 1000 --fusion dense \
  --run_out runs/S3_rerank.dev.M4C80S100.trec --tag RERANK_DENSE

trec_eval -m map -m recall.100 data/qrels.dev.trec runs/S3_rerank.dev.M4C80S100.trec
Eval-1
PYTHONPATH=. python -m scripts.rerank_bm25_dense \
  --bm25_run runs/S1_bm25.eval1.trec \
  --query_h5 data/msmarco_queries_dev_eval_embeddings.h5 \
  --passage_h5 data/msmarco_passages_embeddings_subset.h5 \
  --qid_list_tsv data/queries.eval1.tsv \
  --topk_in 100 --topk_out 100 --fusion dense \
  --run_out runs/S3_rerank.eval1.M4C80S100.trec --tag RERANK_DENSE

trec_eval -m recip_rank.10 -m ndcg_cut.10 -m ndcg_cut.100 -m recall.100 \
  data/qrels.eval.one.trec runs/S3_rerank.eval1.M4C80S100.trec
Eval-2
PYTHONPATH=. python -m scripts.rerank_bm25_dense \
  --bm25_run runs/S1_bm25.eval2.trec \
  --query_h5 data/msmarco_queries_dev_eval_embeddings.h5 \
  --passage_h5 data/msmarco_passages_embeddings_subset.h5 \
  --qid_list_tsv data/queries.eval2.tsv \
  --topk_in 100 --topk_out 100 --fusion dense \
  --run_out runs/S3_rerank.eval2.M4C80S100.trec --tag RERANK_DENSE

trec_eval -m recip_rank.10 -m ndcg_cut.10 -m ndcg_cut.100 -m recall.100 \
  data/qrels.eval.two.trec runs/S3_rerank.eval2.M4C80S100.trec
9) Troubleshooting quickies
0.0000 scores on Eval: you forgot --page_table or used wrong qrels columns; regenerate BM25 Eval runs and qrels .trec with col-4 relevance.
trec_eval: duplicate docs on Dev Dense: dedup (qid,docid) then re-rank numbers (see awk above).
Runs too slow: use filtered query lists (queries.eval1.tsv, queries.eval2.tsv) and smaller --topk during iteration; or pass --max_queries N to search_to_run.py.
Faiss missing: pip install faiss-cpu.
tqdm missing: pip install tqdm.
10) Report (what to include)
Setup: dataset subset (1M), 384-d embeddings, dot-product, BM25 params, HNSW params (M, efC, efS), re-rank @K.
Results Table: Dev (MAP/R@100) + Eval-1/Eval-2 (MRR@10, NDCG@10/100, R@100) for BM25 vs Dense vs Re-rank.
Analysis: dense improves top-10 quality; BM25 often higher recall; hybrid dominates top-k while keeping recall; latency & memory trade-offs.
Efficiency: build time, index size, query p50/p95.
That’s the whole pipeline—from blank repo to final tables.