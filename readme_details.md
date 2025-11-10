AI Search Benchmark – BM25 vs Dense vs Re-rank
Benchmarks three IR systems on the MSMARCO passage task:
S1 — BM25 (lexical)
S2 — Dense retrieval (FAISS HNSW, inner-product)
S3 — Hybrid re-rank (BM25 candidates → dense scores)
Evaluate on:
Dev (binary labels): MAP, Recall@100
Eval-1 (TREC DL’19): MRR@10, NDCG@10, NDCG@100, Recall@100
Eval-2 (TREC DL’20): same as Eval-1
Assignment constraints and file definitions come from the course handout and data README
ps3

README
.
Data layout (expected in data/)
From professor
msmarco_passages_embeddings_subset.h5 — 1M passage IDs + 384-d vectors
msmarco_queries_dev_eval_embeddings.h5 — query IDs + vectors (dev+eval)
msmarco_passages_subset.tsv — list of external passage IDs (for alignment)
qrels.dev.tsv — binary (0/1) judgments for ~1k dev queries
qrels.eval.one.tsv — graded (0–3) DL’19
qrels.eval.two.tsv — graded (0–3) DL’20
read_h5.py — snippet for H5 reading
From MSMARCO site
queries.dev.tsv, queries.eval.tsv
Your corpus
collection.tsv — the 1M subset passages (id<TAB>text)
Environment
python3 -m venv .venv
source .venv/bin/activate
pip install faiss-cpu h5py numpy tqdm pytrec_eval
High-level pipeline
Index BM25 on the 1M subset
Validate ID alignment (internal vs external IDs) and write page_table.tsv if needed
Build FAISS HNSW (inner-product) over 1M vectors
Create filtered query lists that match judged qids (Dev/Eval-1/Eval-2)
Run S1/S2/S3 and evaluate
Write report (quality + efficiency)
Commands (canonical)
1) Build BM25 index (S1)
PYTHONPATH=. python -m src.index_build \
  --input data/collection.tsv \
  --outdir index/final \
  --batch_docs 50000
2) Validate subset alignment & (optionally) write page table
PYTHONPATH=. python -m scripts.validate_subset_alignment \
  --collection data/collection.tsv \
  --subset data/msmarco_passages_subset.tsv \
  --h5 data/msmarco_passages_embeddings_subset.h5 \
  --write_mapping data/page_table.tsv
3) Build FAISS HNSW (S2 index)
PYTHONPATH=. python -m scripts.build_hnsw \
  --emb_h5 data/msmarco_passages_embeddings_subset.h5 \
  --out_dir index/faiss_hnsw_M4C80 \
  --M 4 --efC 80
4) Filter queries to judged qids
Dev (≈1k):
awk '{print $1}' data/qrels.dev.tsv | sort -u > data/qids.dev.txt
awk 'NR==FNR{keep[$1];next} ($1 in keep)' data/qids.dev.txt data/queries.dev.tsv > data/queries.dev.filtered.tsv
Eval-1 (DL’19):
awk '{print $1}' data/qrels.eval.one.tsv | sort -u > data/qids.eval1.txt
awk 'NR==FNR{keep[$1];next} ($1 in keep)' data/qids.eval1.txt data/queries.eval.tsv > data/queries.eval1.tsv
Eval-2 (DL’20):
awk '{print $1}' data/qrels.eval.two.tsv | sort -u > data/qids.eval2.txt
awk 'NR==FNR{keep[$1];next} ($1 in keep)' data/qids.eval2.txt data/queries.eval.tsv > data/queries.eval2.tsv
5) Convert qrels to TREC format
# Dev (qid, 0, pid, rel)
awk '{print $1,"0",$2,$3}' data/qrels.dev.tsv > data/qrels.dev.trec

# Eval sets (relevance is 4th column)
awk '{print $1,"0",$3,$4}' data/qrels.eval.one.tsv > data/qrels.eval.one.trec
awk '{print $1,"0",$3,$4}' data/qrels.eval.two.tsv > data/qrels.eval.two.trec
6) Run BM25 (S1) + evaluate
Dev
PYTHONPATH=. python -m scripts.search_to_run \
  --index_dir index/final \
  --queries data/queries.dev.filtered.tsv \
  --run_out runs/S1_bm25.dev.trec \
  --k1 0.9 --b 0.4 --mode disj --topk 1000 \
  --page_table data/page_table.tsv

trec_eval -m map -m recall.100 data/qrels.dev.trec runs/S1_bm25.dev.trec
Eval-1 / Eval-2 (use queries.eval1.tsv / queries.eval2.tsv, typically topk 100)
trec_eval -m recip_rank.10 -m ndcg_cut.10 -m ndcg_cut.100 -m recall.100 \
  data/qrels.eval.one.trec runs/S1_bm25.eval1.trec
trec_eval -m recip_rank.10 -m ndcg_cut.10 -m ndcg_cut.100 -m recall.100 \
  data/qrels.eval.two.trec runs/S1_bm25.eval2.trec
7) Run Dense / HNSW (S2) + evaluate
Dev
PYTHONPATH=. python -m scripts.search_hnsw \
  --query_h5 data/msmarco_queries_dev_eval_embeddings.h5 \
  --qid_key id --qemb_key embedding \
  --index_dir index/faiss_hnsw_M4C80 \
  --qid_list_tsv data/queries.dev.filtered.tsv \
  --topk 1000 --efS 100 \
  --run_out runs/S2_hnsw.dev.M4C80S100.trec --tag FAISS

# If duplicates occur:
awk '!seen[$1,$3]++' runs/S2_hnsw.dev.M4C80S100.trec > runs/S2_hnsw.dev.M4C80S100_nodup.trec
trec_eval -m map -m recall.100 data/qrels.dev.trec runs/S2_hnsw.dev.M4C80S100_nodup.trec
Eval-1 / Eval-2 (use their filtered query lists, e.g., topk 100, efS 100)
8) Re-rank (S3) + evaluate
Dev
PYTHONPATH=. python -m scripts.rerank_bm25_dense \
  --bm25_run runs/S1_bm25.dev.trec \
  --query_h5 data/msmarco_queries_dev_eval_embeddings.h5 \
  --passage_h5 data/msmarco_passages_embeddings_subset.h5 \
  --qid_list_tsv data/queries.dev.filtered.tsv \
  --topk_in 1000 --topk_out 1000 --fusion dense \
  --run_out runs/S3_rerank.dev.M4C80S100.trec --tag RERANK_DENSE

trec_eval -m map -m recall.100 data/qrels.dev.trec runs/S3_rerank.dev.M4C80S100.trec
Eval-1 / Eval-2 (use their BM25 runs + filtered query lists, typically topk_in/out 100)
Code map (what each file does)
src/
index_build.py
Reads collection.tsv and builds a compact inverted index under index/final/:
lexicon.tsv (term → posting offset, length, df)
postings.bin (docIDs + tfs)
doclen.bin (per-doc lengths)
Focus: fast disk-backed BM25 retrieval on the 1M subset.
query_bm25.py
Core BM25 components used by search_to_run.py:
load_lexicon, load_doclens, read_term_postings
BM25 class (k1, b, idf)
score_disjunctive / score_conjunctive over merged postings
Returns (docid → score) maps per query.
scripts/
search_to_run.py
BM25 search driver. Reads queries (qid<TAB>text), scores with BM25, emits a TREC run.
Key features:
Robust query parser (tabs or single-space; skips headers/blanks)
--mode disj|conj (OR vs AND)
--page_table maps internal BM25 docIDs → external MSMARCO IDs (critical for eval)
--max_queries N for quick smoke runs
Output: qid Q0 docid rank score BM25
validate_subset_alignment.py
Verifies consistency across:
collection.tsv (IDs in the index)
msmarco_passages_subset.tsv (professor’s chosen 1M external IDs)
msmarco_passages_embeddings_subset.h5 (id in same order)
Detects internal (0..N-1) vs external IDs; optionally writes page_table.tsv.
This prevents ID-mismatch zeros at evaluation.
build_hnsw.py
Builds FAISS HNSW (inner-product) from the passage .h5 vectors.
Inputs: --emb_h5, optional --pid_key, --emb_key (auto-detects).
Params: --M, --efC.
Outputs:
index/.../hnsw.index (FAISS)
index/.../pids.npy (passage external IDs, aligned to index rows)
meta.json (config + counts)
search_hnsw.py
Dense retrieval driver.
Loads FAISS index + pids.npy, reads query embeddings from .h5, takes a qid list (*.tsv) to stay aligned with judged queries, performs index.search(), and writes TREC runs with external passage IDs (from pids.npy).
Param --efS controls HNSW search quality/time.
rerank_bm25_dense.py
Hybrid ranker.
Reads a BM25 run, takes top-K candidate docIDs per query, scores them via dot(query, passage) using provided .h5 embeddings, re-sorts, and writes a new TREC run.
Flags:
--topk_in (BM25 candidates to rescore), --topk_out (how many to emit)
--fusion dense (pure dense score; linear with --alpha also supported if present)
Requires consistent ID space (BM25 run must already be in external passage IDs; that’s why --page_table is applied at BM25 stage, not here).
(Optional) reporting utilities
If present: report_effectiveness.py, plot_ir_graphs.py help aggregate results and plot metrics.
Common pitfalls & fixes
Eval zeros everywhere → Your BM25 run used internal IDs.
Regenerate BM25 Eval runs with --page_table data/page_table.tsv, and ensure qrels .trec uses the 4th column for relevance.
Dev Dense MAP fails due to duplicates → HNSW run emitted duplicate (qid, docid) lines.
Dedup before eval:
awk '!seen[$1,$3]++' runs/S2_hnsw.dev...trec > runs/S2_hnsw.dev..._nodup.trec
Runs are huge/slow → Always filter queries to judged qids (the awk filter steps).
You can also reduce --topk and tweak HNSW (M, efC, efS) for iteration speed.
Metric sets
Dev: MAP, Recall@100 (binary)
Eval: MRR@10, NDCG@10/100, Recall@100 (graded)
What “good” looks like (your completed numbers, example)
Dev
BM25: MAP 0.3809, R@100 0.7573
HNSW: MAP 0.3890, R@100 0.6462 (after de-dup)
Re-rank: MAP 0.5655, R@100 0.8630
Eval-1 (DL’19, top-100)
BM25: MRR@10 0.8120, NDCG@10 0.4647, R@100 0.5004
HNSW: MRR@10 0.7682, NDCG@10 0.5387, R@100 0.4507
Re-rank: MRR@10 0.9271, NDCG@10 0.6625, R@100 0.5004
Eval-2 (DL’20, top-100)
BM25: MRR@10 0.8007, NDCG@10 0.4939, R@100 0.5023
HNSW: MRR@10 0.8397, NDCG@10 0.5882, R@100 0.5225
Re-rank: MRR@10 0.8988, NDCG@10 0.6225, R@100 0.5023
(Your exact values may vary slightly by params.)
Efficiency notes (what to record for the report)
Index build
BM25: disk size of postings; build wall-time
HNSW: index size, M, efC; build wall-time
Query latency
Per-query p50/p95 for S1, S2 (efS), S3 (topk_in x dot-products)
Memory
HNSW resident memory vs BM25 disk IO
Why ID alignment matters (quick mental model)
BM25 index may assign internal ids 0..N-1 even if your corpus lines begin with external MSMARCO ids.
Dense embeddings and qrels always use external ids.
validate_subset_alignment.py + --page_table ensure BM25 output matches external ids so trec_eval can join runs and qrels consistently.
HNSW uses external ids via pids.npy, so its runs already align.
Minimal reproducibility checklist
 index/final/ exists (BM25)
 index/faiss_hnsw_M*/hnsw.index and pids.npy exist
 data/page_table.tsv generated if collection used internal ids
 data/qrels.*.trec built correctly (Dev uses col-3; Eval uses col-4)
 Filtered query lists exist: queries.dev.filtered.tsv, queries.eval1.tsv, queries.eval2.tsv
 Runs present for all (S1/S2/S3 × Dev/Eval-1/Eval-2)
 trec_eval metrics recorded in a single table in the report