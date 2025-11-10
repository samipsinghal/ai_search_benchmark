AI Search Benchmark — Assignment #3
CS 6913: Information Retrieval and Search Systems (Fall 2024)
Author: Samip Singhal, NYU Tandon School of Engineering
1. Introduction
This repository implements and benchmarks three retrieval systems on the MS MARCO passage ranking dataset:
BM25 (Sparse Retrieval) — lexical matching using an inverted index.
HNSW (Dense Retrieval) — vector-based search using approximate nearest neighbor indexing.
Hybrid Re-Ranking (BM25 + Dense) — first-stage lexical retrieval followed by semantic re-ranking.
The primary goal is to compare retrieval quality (MRR, MAP, NDCG, Recall) and efficiency (indexing time, query latency, memory) across these methods, using the same 1 million-passage subset provided for this assignment.
2. Dataset Setup
The instructor provided a pre-filtered subset of the MS MARCO passage corpus, along with corresponding embeddings and evaluation files.
File	Description
msmarco_passages_embeddings_subset.h5	Embeddings (384-D) of the 1 M passages used for dense retrieval.
msmarco_dev_eval_embeddings.h5	Query embeddings for both dev and eval splits.
msmarco_passages_subset.tsv	List of 1 M passage IDs from the original 8.8 M dataset.
qrels.eval.one.tsv	TREC DL 2019 relevance labels (graded 0–3).
qrels.eval.two.tsv	TREC DL 2020 relevance labels (graded 0–3).
qrels.dev.tsv	MSMARCO dev labels (binary 0/1).
read_h5.py	Reference snippet for reading the .h5 files.
All experiments are run on this fixed subset so that BM25, FAISS, and the hybrid reranker operate over exactly the same set of passages and maintain ID consistency.
3. Rebuilding the Subset-Aligned Corpus
Script: scripts/download_data.py
This step reconstructs a text collection (collection.tsv) and a mapping file (page_table.tsv) that align perfectly with the provided subset and embedding order.
Command
python -m scripts.download_data \
  --subset_ids data/msmarco_passages_subset.tsv \
  --out data/collection.tsv \
  --page_table data/page_table.tsv \
  --verify
Functionality
Reads the 1 M passage IDs from msmarco_passages_subset.tsv.
Fetches only those passages from the Hugging Face dataset
sentence-transformers/msmarco-corpus (config = "passage").
Writes:
collection.tsv → internal_docid<TAB>text
page_table.tsv → internal_docid<TAB>official_passage_id
Example Output
[INFO] collected 1,000,000 / 1,000,000 required passages
[OK] wrote 1,000,000 rows -> data/collection.tsv
[OK] wrote mapping internal->external -> data/page_table.tsv
[VERIFY] first 3 ids mapping (internal->external):
    0 -> 350
    1 -> 448
    2 -> 466
This confirms one-to-one alignment between the internal document IDs used by BM25 and the official passage IDs used in the .h5 files and qrels.
4. Building the BM25 Index
Scripts: src/index_build.py, src/index_merge.py
Commands
rm -rf index/tmp index/final
python -m src.index_build --input data/collection.tsv --outdir index/tmp --batch_docs 50000
python -m src.index_merge  --tmpdir index/tmp --outdir index/final
Description
index_build.py tokenizes documents and writes partial posting runs.
index_merge.py merges all runs using an external k-way merge.
Postings are stored with delta-encoded docIDs and VarByte compression.
Resulting Files
index/final/postings.bin   # compressed postings
index/final/lexicon.tsv    # term → offset, length, df
index/final/doclen.bin     # document lengths
index/final/docmap.tsv     # internal bookkeeping
Sample Log
[run 0–19] wrote ~2M postings each
[OK] total documents processed: 1,000,000
[INFO] Merging 20 runs ...
[OK] Wrote postings -> index/final/postings.bin
[OK] Wrote lexicon  -> index/final/lexicon.tsv
5. Querying and Verification
To confirm index integrity and mapping:
python -m src.query_bm25 \
  --index_dir index/final \
  --mode disj --k 5 --k1 0.9 --b 0.4 \
  --page_table data/page_table.tsv \
  --collection data/collection.tsv \
  --snippet --show_text
This interactive mode reads queries from standard input, returns top-K passages, highlights matched terms, and prints text snippets.
Results display external passage IDs via page_table.tsv, verifying that mapping works correctly.
6. Generating BM25 Run Files (Evaluation Phase)
Script: scripts/search_to_run.py
This script runs BM25 for a full query set and outputs a TREC-formatted run file.
Example Commands
mkdir -p runs

# TREC DL 2019
python -m scripts.search_to_run \
  --index_dir index/final \
  --queries   data/queries.eval.tsv \
  --run_out   runs/S1_bm25.eval1.trec \
  --k1 0.9 --b 0.4 --mode disj --topk 1000 \
  --page_table data/page_table.tsv

# TREC DL 2020
python -m scripts.search_to_run \   
  --index_dir index/final \
  --queries   data/queries.eval.tsv \
  --run_out   runs/S1_bm25.eval2.trec \
  --k1 0.9 --b 0.4 --mode disj --topk 1000 \
  --page_table data/page_table.tsv

# MS MARCO Dev
python -m scripts.search_to_run \
  --index_dir index/final \
  --queries   data/queries.dev.tsv \
  --run_out   runs/S1_bm25.dev.trec \
  --k1 0.9 --b 0.4 --mode disj --topk 1000 \
  --page_table data/page_table.tsv
Each output line follows the TREC format:
qid  Q0  passage_id  rank  score  BM25
7. Evaluation Metrics
Using trec_eval (C tool)
# DL’19 and DL’20 (graded)
trec_eval -m recip_rank.10 -m recall.100 -m ndcg_cut.10 -m ndcg_cut.100 \
  data/qrels.eval.one.tsv runs/S1_bm25.eval1.trec
trec_eval -m recip_rank.10 -m recall.100 -m ndcg_cut.10 -m ndcg_cut.100 \
  data/qrels.eval.two.tsv runs/S1_bm25.eval2.trec

# MSMARCO dev (binary)
trec_eval -m map -m recall.100 \
  data/qrels.dev.tsv runs/S1_bm25.dev.trec
Using Python (pytrec_eval)
import pytrec_eval, numpy as np

def load_qrels(path):
    qrels = {}
    for line in open(path):
        q, _, d, rel = line.split()
        qrels.setdefault(q, {})[d] = int(rel)
    return qrels

def load_run(path):
    run = {}
    for line in open(path):
        q, _, d, rank, score, _ = line.split()
        run.setdefault(q, {})[d] = float(score)
    return run

qrels = load_qrels('data/qrels.eval.one.tsv')
run   = load_run('runs/S1_bm25.eval1.trec')
evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'ndcg_cut', 'recip_rank', 'recall'})
res = evaluator.evaluate(run)
ndcg10  = np.mean([r['ndcg_cut_10'] for r in res.values()])
mrr10   = np.mean([r['recip_rank'] for r in res.values()])
print(f"NDCG@10={ndcg10:.4f}, MRR@10={mrr10:.4f}")
8. Dense Retrieval (HNSW / FAISS)
This stage constructs an approximate nearest-neighbor index on the same 1 M passage embeddings.
Input Files
msmarco_passages_embeddings_subset.h5
msmarco_dev_eval_embeddings.h5
Planned Workflow
Load the passage embeddings (float32, 384-D).
Build a FAISS HNSW index using dot-product similarity:
import faiss, h5py
data = h5py.File("data/msmarco_passages_embeddings_subset.h5")["embedding"][:]
index = faiss.IndexHNSWFlat(384, 64)
index.hnsw.efConstruction = 150
index.hnsw.efSearch = 100
index.add(data)
faiss.write_index(index, "index/faiss_hnsw.index")
For each query embedding, perform search(k=1000) to get top candidates.
Write results in TREC format:
qid  Q0  passage_id  rank  score  FAISS
Evaluate with the same metrics as BM25.
Parameter Tuning:
Parameter	Typical Range	Effect
M	4–8	number of bi-directional links per node
efConstruction	50–200	indexing accuracy vs build time
efSearch	50–200	retrieval accuracy vs latency
9. Hybrid Re-Ranking (BM25 + Dense)
The hybrid system reranks top BM25 results using vector similarity from the dense model.
Planned Steps
Use BM25 to get top K (100 or 1000) candidates.
Retrieve embeddings for the corresponding passages and the query.
Compute dot-product similarity.
Re-rank the BM25 candidates by dense score.
Write new run files:
runs/S3_rerank.eval1.trec
runs/S3_rerank.eval2.trec
runs/S3_rerank.dev.trec
Evaluate using the same trec_eval setup.
The expectation is that re-ranking will improve NDCG and MRR at the cost of additional computation per query.
10. Progress Summary
Component	Status
Subset alignment (collection.tsv, page_table.tsv)	✅ Done
BM25 index build and validation	✅ Done
Query interface	✅ Done
Batch run generation	⬜ Pending
Evaluation (trec_eval, pytrec_eval)	⬜ Pending
Dense retrieval (FAISS HNSW)	⬜ Pending
Hybrid re-ranking	⬜ Pending
11. References
MS MARCO Passage Ranking Dataset — https://microsoft.github.io/msmarco/
Anserini Toolkit — Open-source BM25 and evaluation framework.
FAISS Library (Facebook AI Research) — https://github.com/facebookresearch/faiss
TREC Deep Learning 2019 & 2020 Tracks
CS 6913 Course Materials, NYU Tandon School of Engineering
Notes
All experiments are performed on the instructor-provided 1 M subset.
BM25 results will serve as the baseline for subsequent FAISS and hybrid evaluations