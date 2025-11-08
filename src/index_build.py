"""
index_build.py
---------------
Builds intermediate posting lists (run files) from the input collection.
Each run file contains sorted (term, docid, tf) lines.

Usage:
    python -m src.index_build --input data/collection.sample100000.tsv --outdir index/tmp --batch_docs 50000
"""

import os, argparse
from collections import Counter
from src.common import tokenize  # our tokenizer from common.py

def write_run(run_id, postings, outdir):
    """Write postings sorted by (term, docid) into a run file."""
    os.makedirs(outdir, exist_ok=True)
    run_path = os.path.join(outdir, f"run_{run_id:06d}.tsv")
    postings.sort(key=lambda x: (x[0], x[1]))  # (term, docid)
    with open(run_path, "w", encoding="utf-8") as f:
        for term, docid, tf in postings:
            f.write(f"{term}\t{docid}\t{tf}\n")
    print(f"[run {run_id}] wrote {len(postings)} postings -> {run_path}")

def build_index(input_path, outdir, batch_docs=50000):
    postings = []           # list[(term, docid, tf)]
    doc_lens = []           # list[int] length per doc
    run_id = 0
    doc_counter = 0

    # optional: record processing order -> passage_id (helps audits)
    docmap_dir = os.path.join(os.path.dirname(outdir), "final")
    os.makedirs(docmap_dir, exist_ok=True)
    docmap_path = os.path.join(docmap_dir, "docmap.tsv")
    docmap_fp = open(docmap_path, "w", encoding="utf-8")

    with open(input_path, "r", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip() or "\t" not in line:
                continue

            docid_str, text = line.split("\t", 1)
            docid_str = docid_str.strip()
            if not docid_str:
                continue

            # Preserve canonical MS MARCO passage_id (they are integers)
            try:
                docid = int(docid_str)
            except ValueError as e:
                raise ValueError(f"Non-integer docid encountered: {docid_str!r}") from e

            terms = list(tokenize(text))
            tf_counts = Counter(terms)
            doc_lens.append(len(terms))

            # write processing-order map: ordinal -> passage_id
            # (ordinal is the 0-based index of appearance in collection.tsv)
            docmap_fp.write(f"{doc_counter}\t{docid}\n")
            doc_counter += 1

            for term, tf in tf_counts.items():
                postings.append((term, docid, tf))

            if doc_counter % batch_docs == 0:
                write_run(run_id, postings, outdir)
                postings.clear()
                run_id += 1

    if postings:
        write_run(run_id, postings, outdir)

    docmap_fp.close()

    # doc lengths sidecar (binary little-endian uint32)
    doclen_path = os.path.join(docmap_dir, "doclen.bin")
    with open(doclen_path, "wb") as f:
        for L in doc_lens:
            f.write(L.to_bytes(4, "little", signed=False))

    print(f"[OK] wrote {len(doc_lens)} doc lengths -> {doclen_path}")
    print(f"[OK] wrote docmap -> {docmap_path}")
    print(f"[OK] total documents processed: {doc_counter}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to collection.tsv (passage_id<TAB>text)")
    parser.add_argument("--outdir", required=True, help="Where to write run files")
    parser.add_argument("--batch_docs", type=int, default=50000, help="Docs per run before spilling")
    args = parser.parse_args()
    build_index(args.input, args.outdir, batch_docs=args.batch_docs)
