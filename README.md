AI Search Benchmark – End-to-End README

0) Overview

Compare three information retrieval (IR) systems on the MS MARCO Passage Ranking (v1) dataset:

System	Description
S1 – BM25	Sparse lexical retrieval (term-based inverted index)
S2 – Dense Retrieval (HNSW)	Approximate nearest-neighbor search using FAISS HNSW (inner-product)
S3 – BM25 → Dense Rerank	Hybrid pipeline: BM25 retrieval followed by dense embedding–based re-ranking

Evaluated on:

Split	Relevance	Metrics
Dev	Binary (qrels.dev.tsv)	MAP, Recall@100
Eval-1 (TREC DL 2019)	Graded (0-3)	MRR@10, NDCG@10/100, Recall@100
Eval-2 (TREC DL 2020)	Graded (0-3)	MRR@10, NDCG@10/100, Recall@100
1) Data Layout (data/)
collection.tsv
msmarco_passages_subset.tsv
msmarco_passages_embeddings_subset.h5
msmarco_queries_dev_eval_embeddings.h5
queries.dev.tsv
queries.dev.filtered.tsv         # subset aligned with qrels.dev.tsv
queries.eval.tsv
qrels.dev.tsv
qrel.eval.one.tsv                # TREC DL’19
qrel.eval.two.tsv                # TREC DL’20

2) Environment Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install faiss-cpu h5py numpy tqdm psutil pytrec_eval

3) Build BM25 Index (S1)
PYTHONPATH=. python -m src.index_build \
  --input data/collection.tsv \
  --outdir index/final \
  --batch_docs 50000

4) Validate Subset Alignment & Create Page Table
PYTHONPATH=. python -m scripts.validate_subset_alignment \
  --collection data/collection.tsv \
  --subset data/msmarco_passages_subset.tsv \
  --h5 data/msmarco_passages_embeddings_subset.h5 \
  --write_mapping data/page_table.tsv


This ensures that passage IDs in the H5 embedding file match those in collection.tsv.

5) Build FAISS HNSW Index (S2)
PYTHONPATH=. python -m scripts.build_hnsw \
  --emb_h5 data/msmarco_passages_embeddings_subset.h5 \
  --out_dir index/faiss_hnsw_M4C80 \
  --M 4 --efC 80


You can later tune M (4–8), efConstruction (50–200).

6) Prepare Filtered Query Lists

To align queries with their judged qrels:

awk '{print $1}' data/qrels.dev.tsv | sort -u > data/qids.dev.txt
awk 'NR==FNR{keep[$1];next} ($1 in keep)' data/qids.dev.txt data/queries.dev.tsv > data/queries.dev.filtered.tsv


Repeat similarly for Eval-1 and Eval-2 if needed.

7) Convert Qrels to TREC Format
awk '{print $1,"0",$2,$3}' data/qrels.dev.tsv > data/qrels.dev.trec
awk '{print $1,"0",$3,$4}' data/qrel.eval.one.tsv > data/qrels.eval.one.trec
awk '{print $1,"0",$3,$4}' data/qrel.eval.two.tsv > data/qrels.eval.two.trec

8) Runs & Evaluation
8.1 BM25 (S1)

Dev:

PYTHONPATH=. python -m scripts.search_to_run \
  --index_dir index/final \
  --queries data/queries.dev.filtered.tsv \
  --run_out runs/S1_bm25.dev.trec \
  --k1 0.9 --b 0.4 --mode disj --topk 1000 \
  --page_table data/page_table.tsv
trec_eval -m map -m recall.100 data/qrels.dev.trec runs/S1_bm25.dev.trec


Run analogous commands for Eval-1 and Eval-2.

8.2 Dense Retrieval / HNSW (S2)

Dev Example:

PYTHONPATH=. python -m scripts.search_hnsw \
  --query_h5 data/msmarco_queries_dev_eval_embeddings.h5 \
  --qid_key id --qemb_key embedding \
  --index_dir index/faiss_hnsw_M4C80 \
  --qid_list_tsv data/queries.dev.filtered.tsv \
  --topk 1000 --efS 100 \
  --run_out runs/S2_hnsw.dev.M4C80S100.trec --tag FAISS


Parameter Tuning:

M	efConstruction	efSearch	Trend
4	50	50	Fastest / Lower recall
4	80	100	Balanced (default)
8	200	200	Best recall / Highest latency
8.3 BM25 → Dense Rerank (S3)

Re-rank BM25 candidates with dense embeddings:

PYTHONPATH=. python -m scripts.rerank_bm25_dense \
  --bm25_run runs/S1_bm25.dev.trec \
  --query_h5 data/msmarco_queries_dev_eval_embeddings.h5 \
  --passage_h5 data/msmarco_passages_embeddings_subset.h5 \
  --qid_list_tsv data/queries.dev.filtered.tsv \
  --fusion dense \
  --run_out runs/S3_rerank.dev.M4C80S100.trec --tag RERANK_DENSE
trec_eval -m map -m recall.100 data/qrels.dev.trec runs/S3_rerank.dev.M4C80S100.trec

9) Aggregate All Results

The provided script summarizes all .trec outputs:

./benchmark_all.sh | tee results_summary.txt


Produces a table like:

Run	Qrels Type	MRR@10	Recall@100	NDCG@10	NDCG@100	MAP
S1_bm25.dev	binary	0.379	0.757	0.425	0.462	0.381
S2_hnsw.dev	binary	0.391	0.646	0.424	0.447	0.389
S3_rerank.dev	binary	0.569	0.863	0.610	0.634	0.566
10) Efficiency Measurement

Each command can be timed with:

/usr/bin/time -v python -m scripts.search_to_run ...


and internally the scripts log:

[TIME] 35.24s | [MEM] 680.3 MB


To estimate average latency:

NUM=$(wc -l < data/queries.dev.filtered.tsv)
echo "scale=3; <elapsed_seconds> / $NUM" | bc

11) Troubleshooting
Issue	Cause / Fix
0.0000 scores	Wrong qrels format → check columns and --page_table
Duplicate (qid, docid)	Run dedup: awk '!seen[$1,$3]++'
declare: -A error	Use bash ./benchmark_all.sh instead of sh
Missing psutil	pip install psutil
FAISS not found	pip install faiss-cpu
