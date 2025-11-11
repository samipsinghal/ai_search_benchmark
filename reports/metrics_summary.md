### Retrieval Benchmark Summary (BM25 vs HNSW vs Rerank)

| Split | System | MRR@10 | Recall@100 | NDCG@10 | NDCG@100 | MAP |
|-------|---------|--------|-------------|----------|-----------|------|
| Dev (binary) | BM25 | 0.379 | 0.757 | 0.425 | 0.462 | 0.381 |
| Dev (binary) | HNSW | 0.391 | 0.646 | 0.424 | 0.447 | 0.389 |
| Dev (binary) | Rerank | 0.569 | 0.863 | 0.610 | 0.634 | 0.566 |
| Eval 1 (graded) | BM25 | 0.810 | 0.500 | 0.392 | 0.496 | 0.339 |
| Eval 1 (graded) | HNSW | 0.768 | 0.451 | 0.472 | 0.509 | 0.348 |
| Eval 1 (graded) | Rerank | 0.927 | 0.500 | 0.592 | 0.562 | 0.381 |
| Eval 1 (graded) | Rerank | 0.965 | 0.545 | 0.615 | 0.622 | 0.516 |
| Eval 2 (graded) | BM25 | 0.798 | 0.502 | 0.451 | 0.496 | 0.301 |
| Eval 2 (graded) | HNSW | 0.840 | 0.523 | 0.543 | 0.542 | 0.377 |
| Eval 2 (graded) | Rerank | 0.898 | 0.502 | 0.577 | 0.556 | 0.376 |
