#!/usr/bin/env python3
"""
rerank_bm25_dense.py
--------------------
Re-rank BM25 candidates using dense embeddings (dot product).
Supports:
  --fusion dense       → ignore BM25 scores, use dense similarity only
  --fusion linear      → weighted fusion: alpha * dense + (1-alpha) * bm25
Normalization: optional minmax or zscore.
"""

import os, sys, json, argparse, h5py, numpy as np
from tqdm import tqdm

def load_embeddings(h5_path, id_key, emb_key):
    with h5py.File(h5_path, "r") as f:
        ids = f[id_key][:]
        vecs = f[emb_key][:].astype("float32")
    # convert byte/string IDs to int if necessary
    if ids.dtype.kind not in "iu":
        ids = ids.astype(str).astype(np.int64)
    return ids, vecs

def load_bm25_run(path, topk_in=1000):
    """Parse TREC run file -> {qid: [(docid, score), ...]}"""
    runs = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            qid, _, docid, rank, score, tag = line.strip().split()
            if qid not in runs:
                runs[qid] = []
            if len(runs[qid]) < topk_in:
                runs[qid].append((int(docid), float(score)))
    return runs

def normalize_scores(scores, method="none"):
    if method == "none":
        return scores
    s = np.array(list(scores.values()), dtype=np.float32)
    if len(s) == 0:
        return scores
    if method == "minmax":
        lo, hi = np.min(s), np.max(s)
        rng = hi - lo if hi > lo else 1e-9
        for k in scores: scores[k] = (scores[k]-lo)/rng
    elif method == "zscore":
        mu, sd = np.mean(s), np.std(s) if np.std(s)>0 else 1e-9
        for k in scores: scores[k] = (scores[k]-mu)/sd
    return scores

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bm25_run", required=True)
    ap.add_argument("--query_h5", required=True)
    ap.add_argument("--passage_h5", required=True)
    ap.add_argument("--qid_list_tsv", required=True)
    ap.add_argument("--topk_in", type=int, default=1000)
    ap.add_argument("--topk_out", type=int, default=1000)
    ap.add_argument("--fusion", choices=["dense", "linear"], default="dense")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--norm_dense", choices=["none","minmax","zscore"], default="none")
    ap.add_argument("--norm_bm25", choices=["none","minmax","zscore"], default="none")
    ap.add_argument("--qid_key", default="id")
    ap.add_argument("--qemb_key", default="embedding")
    ap.add_argument("--pid_key", default="id")
    ap.add_argument("--pemb_key", default="embedding")
    ap.add_argument("--run_out", required=True)
    ap.add_argument("--tag", default="RERANK_DENSE")
    args = ap.parse_args()

    qids, qvecs = load_embeddings(args.query_h5, args.qid_key, args.qemb_key)
    pids, pvecs = load_embeddings(args.passage_h5, args.pid_key, args.pemb_key)
    pid_to_idx = {pid: i for i, pid in enumerate(pids)}
    bm25_runs = load_bm25_run(args.bm25_run, args.topk_in)

    with open(args.qid_list_tsv, "r", encoding="utf-8") as f:
        valid_qids = {line.strip().split("\t")[0] for line in f if line.strip()}

    with open(args.run_out, "w", encoding="utf-8") as outf:
        for qid in tqdm(bm25_runs, desc="[Rerank]"):
            if qid not in valid_qids: continue
            try:
                qidx = np.where(qids == int(qid))[0][0]
            except Exception:
                continue
            qv = qvecs[qidx]
            bm25_pairs = bm25_runs[qid]
            dense_scores = {}
            for pid, _ in bm25_pairs:
                if pid not in pid_to_idx: continue
                pvec = pvecs[pid_to_idx[pid]]
                dense_scores[pid] = float(np.dot(qv, pvec))
            dense_scores = normalize_scores(dense_scores, args.norm_dense)

            # make sure bm25 scores are normalized consistently
            bm25_scores = {pid: s for pid, s in bm25_pairs}
            bm25_scores = normalize_scores(bm25_scores, args.norm_bm25)

            fused = {}
            if args.fusion == "dense":
                fused = dense_scores
            else:
                for pid in dense_scores:
                    fused[pid] = args.alpha*dense_scores[pid] + (1-args.alpha)*bm25_scores.get(pid, 0.0)

            topk = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:args.topk_out]
            for rank, (pid, score) in enumerate(topk, 1):
                outf.write(f"{qid}\tQ0\t{pid}\t{rank}\t{score:.4f}\t{args.tag}\n")

    print(f"[OK] wrote reranked run -> {args.run_out}")

if __name__ == "__main__":
    main()