#!/usr/bin/env python3
"""
search_to_run.py
----------------
Batch-evaluate BM25 over a query set.
Outputs a TREC-style run file: 'qid Q0 docid rank score tag'.

Supports --page_table to map internal docIDs -> external passage IDs.
Tolerates messy TSVs (blank lines, space-delimited rows, headers).
"""

import os, sys, heapq, argparse

# --- ensure repo root on sys.path so 'from src...' works either as -m or script ---
_ROOT = os.path.dirname(os.path.abspath(os.path.join(__file__, "..")))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
# -------------------------------------------------------------------------------

from src.query_bm25 import (
    load_lexicon, load_doclens, read_term_postings, BM25,
    score_disjunctive, score_conjunctive
)

def load_page_table(path):
    """Load optional internal->external mapping: 'internal_docid[TAB]external_id'."""
    if not path:
        return None
    mapping = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or "\t" not in s:
                continue
            k, v = s.split("\t", 1)
            try:
                mapping[int(k)] = v
            except ValueError:
                # skip malformed row
                continue
    return mapping

def iter_queries(queries_path):
    """
    Yield (qid:str, qtext:str) from a file containing either:
      - 'qid<TAB>query text'
      - 'qid<SPACE>query text'
    Skips blank/malformed rows and very short lines.
    """
    with open(queries_path, "r", encoding="utf-8") as qf:
        for ln, raw in enumerate(qf, 1):
            s = raw.strip()
            if not s:
                continue
            qid, qtext = None, None
            if "\t" in s:
                a, b = s.split("\t", 1)
                qid, qtext = a.strip(), b.strip()
            else:
                parts = s.split(None, 1)  # split on any whitespace
                if len(parts) == 2:
                    qid, qtext = parts[0].strip(), parts[1].strip()
                else:
                    # likely a header line or malformed
                    if ln <= 5:
                        print(f"[WARN] Skipping malformed query line {ln}: {s!r}", file=sys.stderr)
                    continue
            if not qid or not qtext:
                if ln <= 5:
                    print(f"[WARN] Empty qid or query at line {ln}: {s!r}", file=sys.stderr)
                continue
            yield qid, qtext

def search_all_queries(index_dir, queries_path, run_out, k1, b, mode, topk=1000, page_table=None):
    lexicon = load_lexicon(os.path.join(index_dir, "lexicon.tsv"))
    doclens  = load_doclens(os.path.join(index_dir, "doclen.bin"))
    bm25     = BM25(doclens, k1=k1, b=b)
    postings_path = os.path.join(index_dir, "postings.bin")

    total = kept = 0
    with open(postings_path, "rb") as pf, open(run_out, "w", encoding="utf-8") as outf:
        for qid, qtext in iter_queries(queries_path):
            total += 1
            terms = [t for t in qtext.lower().split() if t]
            if not terms:
                # no usable tokens
                continue

            # collect postings per in-vocab term
            term_postings = []
            for t in terms:
                meta = lexicon.get(t)
                if not meta:
                    continue
                off, ln, df = meta
                docs, tfs = read_term_postings(pf, off, ln)
                if docs:
                    term_postings.append((df, docs, tfs, bm25.idf(df)))

            if not term_postings:
                # all terms OOV for this query
                continue

            scores = (score_disjunctive if mode == "disj" else score_conjunctive)(bm25, term_postings)
            topK = heapq.nlargest(topk, scores.items(), key=lambda x: x[1])
            if not topK:
                continue
            kept += 1

            # Write TREC format with SPACES (trec_eval expects spaces or tabs; we standardize)
            for rank, (doc, score) in enumerate(topK, 1):
                ext_docid = page_table.get(doc, str(doc)) if page_table else str(doc)
                outf.write(f"{qid} Q0 {ext_docid} {rank} {score:.4f} BM25\n")

    print(f"[OK] wrote run file -> {run_out}  (read {total} queries, produced {kept})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_dir", required=True, help="Directory with postings.bin, lexicon.tsv, doclen.bin")
    ap.add_argument("--queries",   required=True, help="Path to queries file (qid<TAB>text or qid SPACE text)")
    ap.add_argument("--run_out",   required=True, help="Output TREC run path")
    ap.add_argument("--k1",  type=float, default=0.9)
    ap.add_argument("--b",   type=float, default=0.4)
    ap.add_argument("--mode", choices=["disj", "conj"], default="disj")
    ap.add_argument("--topk", type=int, default=1000)
    ap.add_argument("--page_table", default=None, help="Optional internal->external docid map (TSV)")
    ap.add_argument("--max_queries", type=int, default=None, help="Optional cap for a quick smoke test")
    args = ap.parse_args()

    mapping = load_page_table(args.page_table) if args.page_table else None

    # Optional: cap to first N queries without editing the file
    qpath = args.queries
    if args.max_queries is not None:
        import tempfile
        n = 0
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as tmp, \
             open(args.queries, "r", encoding="utf-8") as src:
            for line in src:
                if line.strip():
                    tmp.write(line)
                    n += 1
                    if n >= args.max_queries:
                        break
            qpath = tmp.name
        print(f"[INFO] Using first {n} queries -> {qpath}")

    search_all_queries(
        args.index_dir, qpath, args.run_out,
        args.k1, args.b, args.mode, args.topk, mapping
    )

if __name__ == "__main__":
    main()
