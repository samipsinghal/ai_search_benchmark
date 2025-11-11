#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
report_effectiveness.py

Compute effectiveness metrics for TREC-style runs:
- MRR@10
- Recall@100
- NDCG@10
- NDCG@100
- MAP

Robust to:
- qrels in 3-col (qid docid rel) OR 4-col TREC (qid 0 docid rel)
- non-integer rel tokens (coerced to int if possible), blank lines, comments
- .gz inputs for qrels and runs
- multiple --run flags
"""

import argparse
import math
import gzip
import glob
import os
from collections import defaultdict, OrderedDict
from typing import Dict, List, Tuple, Iterable


def open_text_auto(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return open(path, "r", encoding="utf-8", newline="")


def parse_int(x: str) -> int:
    # Coerce "1", "1.0", "01", etc. to int; return 0 if it truly can't parse
    try:
        return int(x)
    except ValueError:
        try:
            return int(float(x))
        except ValueError:
            return 0


def read_qrels(path: str) -> Dict[str, Dict[str, int]]:
    """
    Supports:
      - 3 columns: qid docid rel
      - 4 columns (TREC): qid 0 docid rel
    Returns {qid: {docid: rel_int}}
    """
    qrels: Dict[str, Dict[str, int]] = defaultdict(dict)
    with open_text_auto(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            # TREC 4-col
            if len(parts) >= 4 and parts[1] in {"0", "Q0"}:
                qid, _zero, docid, rel = parts[0], parts[1], parts[2], parts[3]
            elif len(parts) >= 3:
                qid, docid, rel = parts[0], parts[1], parts[2]
            else:
                continue
            r = parse_int(rel)
            # ignore negative labels just in case
            if r < 0:
                r = 0
            qrels[qid][docid] = r
    return qrels


def read_run(path: str) -> Dict[str, List[Tuple[str, float]]]:
    """
    TREC run format (tolerant):
      qid Q0 docid rank score tag
    or compact:
      qid docid score
    Returns {qid: [(docid, score), ...]} sorted by score desc then docid
    """
    per_q: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    with open_text_auto(path) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 6:
                qid, _, docid, _, score, _ = parts[:6]
            elif len(parts) >= 3:
                qid, docid, score = parts[:3]
            else:
                continue
            try:
                sc = float(score)
            except ValueError:
                continue
            per_q[qid].append((docid, sc))
    for qid in per_q:
        per_q[qid].sort(key=lambda x: (-x[1], x[0]))
    return per_q


def average(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def dcg(rel_list: List[int], k: int) -> float:
    # ensure numeric
    s = 0.0
    for i, rel in enumerate(rel_list[:k], start=1):
        r = float(rel)
        # log2(i+1) is always >= 1 because i starts at 1
        s += (2.0**r - 1.0) / math.log2(i + 1)
    return s


def ndcg_at_k(ranked_docids: List[str], qrels_q: Dict[str, int], k: int) -> float:
    gains = [qrels_q.get(docid, 0) for docid in ranked_docids[:k]]
    ideal = sorted(qrels_q.values(), reverse=True)
    idcg = dcg(ideal, k)
    if idcg == 0.0:
        return 0.0
    return dcg(gains, k) / idcg


def ap(ranked_docids: List[str], relevant_set: set) -> float:
    if not relevant_set:
        return 0.0
    hits = 0
    precisions = []
    for i, docid in enumerate(ranked_docids, start=1):
        if docid in relevant_set:
            hits += 1
            precisions.append(hits / i)
    return sum(precisions) / len(relevant_set) if relevant_set else 0.0


def recall_at_k(ranked_docids: List[str], relevant_set: set, k: int) -> float:
    if not relevant_set:
        return 0.0
    retrieved = set(ranked_docids[:k])
    return len(retrieved & relevant_set) / len(relevant_set)


def mrr_at_k(ranked_docids: List[str], relevant_set: set, k: int) -> float:
    for i, docid in enumerate(ranked_docids[:k], start=1):
        if docid in relevant_set:
            return 1.0 / i
    return 0.0


def compute_metrics_for_run(qrels: Dict[str, Dict[str, int]],
                            run: Dict[str, List[Tuple[str, float]]],
                            k_mrr: int = 10,
                            k_recall: int = 100,
                            k_ndcg_small: int = 10,
                            k_ndcg_large: int = 100) -> Dict[str, float]:
    max_rel = 0
    for q in qrels.values():
        if q:
            max_rel = max(max_rel, max(q.values()))
    graded = max_rel > 1

    mrrs, recalls100, ndcg10s, ndcg100s, maps = [], [], [], [], []

    for qid, rels in qrels.items():
        ranked = [d for (d, _) in run.get(qid, [])]
        binary_rel = {d for d, r in rels.items() if r > 0}

        mrrs.append(mrr_at_k(ranked, binary_rel, k_mrr))
        recalls100.append(recall_at_k(ranked, binary_rel, k_recall))
        ndcg10s.append(ndcg_at_k(ranked, rels, k_ndcg_small))
        ndcg100s.append(ndcg_at_k(ranked, rels, k_ndcg_large))
        maps.append(ap(ranked, binary_rel))

    summary = OrderedDict()
    summary["MRR@10"] = round(average(mrrs), 6)
    summary["Recall@100"] = round(average(recalls100), 6)
    summary["NDCG@10"] = round(average(ndcg10s), 6)
    summary["NDCG@100"] = round(average(ndcg100s), 6)
    summary["MAP"] = round(average(maps), 6)
    summary["_qrels_type"] = "graded" if graded else "binary"
    return summary


def expand_run_args(run_args: Iterable[str]) -> List[str]:
    expanded: List[str] = []
    for arg in run_args:
        matches = glob.glob(arg)
        if matches:
            expanded.extend(sorted(matches))
        elif os.path.exists(arg):
            expanded.append(arg)
    # dedupe, preserve order
    seen = set()
    ordered = []
    for p in expanded:
        if p not in seen:
            seen.add(p)
            ordered.append(p)
    return ordered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qrels", required=True, help="Path to qrels (supports 3- or 4-col, .gz ok)")
    ap.add_argument("--run", action="append", required=True,
                    help="Path or glob to TREC run file (repeatable). Supports .gz")
    args = ap.parse_args()

    qrels = read_qrels(args.qrels)
    run_paths = expand_run_args(args.run)

    if not run_paths:
        print("[ERROR] No valid run files found to evaluate. Check your --run arguments.")
        return

    print(f"# qrels: {args.qrels}")
    print("run\tqrels_type\tMRR@10\tRecall@100\tNDCG@10\tNDCG@100\tMAP")

    for run_path in run_paths:
        try:
            run = read_run(run_path)
        except Exception as e:
            print(f"[WARN] Failed reading '{run_path}': {e}  (skipping)")
            continue
        metrics = compute_metrics_for_run(qrels, run)
        print(
            f"{run_path}\t{metrics['_qrels_type']}\t"
            f"{metrics['MRR@10']}\t{metrics['Recall@100']}\t"
            f"{metrics['NDCG@10']}\t{metrics['NDCG@100']}\t{metrics['MAP']}"
        )


if __name__ == "__main__":
    main()
