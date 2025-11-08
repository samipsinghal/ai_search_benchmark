#!/usr/bin/env python3
"""
validate_subset_alignment.py

Verifies that your BM25 collection.tsv uses the same 1M passages
as the professor’s msmarco_passages_subset.tsv and h5 file.

Checks:
  1) Set equality between collection.tsv and subset.tsv
  2) Order alignment between subset.tsv and h5 embeddings
  3) Detects internal (0..N-1) vs external passage IDs
  4) Optionally writes page_table.tsv (internal_docid → external_id)
"""

import argparse, io, os, random, sys

def read_subset_ids(path):
    ids = []
    with io.open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                ids.append(int(s))
            except ValueError:
                raise ValueError(f"[subset] Non-integer ID at line {ln}: {s}")
    return ids


def read_collection_ids(path):
    ids = []
    with io.open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            if not line.strip():
                continue
            if "\t" not in line:
                raise ValueError(f"[collection] Missing TAB at line {ln}")
            first, _ = line.split("\t", 1)
            try:
                ids.append(int(first))
            except ValueError:
                raise ValueError(f"[collection] Non-integer ID at line {ln}: {first}")
    return ids


def read_h5_ids(path):
    try:
        import h5py
    except ImportError:
        print("Install h5py first: pip install h5py", file=sys.stderr)
        sys.exit(1)

    with h5py.File(path, "r") as h5:
        if "id" not in h5:
            raise KeyError(f"[h5] Dataset 'id' not found in {path}")
        raw = h5["id"][:]

    out = []
    for v in raw:
        if isinstance(v, (bytes, bytearray)):
            v = v.decode("utf-8", errors="replace")
        out.append(int(v))
    return out


def detect_internal_ids(ids):
    return min(ids) == 0 and max(ids) == len(ids) - 1 and len(set(ids)) == len(ids)


def sample_alignment(a, b, k=2000):
    n = min(len(a), len(b))
    if n == 0:
        return 0, 0
    k = min(k, n)
    idx = random.sample(range(n), k)
    matches = sum(1 for i in idx if a[i] == b[i])
    return matches, k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", required=True, help="collection.tsv file")
    ap.add_argument("--subset", required=True, help="msmarco_passages_subset.tsv file")
    ap.add_argument("--h5", required=True, help="msmarco_passages_embeddings_subset.h5 file")
    ap.add_argument("--write_mapping", default="", help="optional: path to write page_table.tsv")
    args = ap.parse_args()

    for p in [args.collection, args.subset, args.h5]:
        if not os.path.exists(p):
            print(f"[ERR] Missing: {p}")
            sys.exit(1)

    print(f"[INFO] Reading {args.subset} …")
    subset_ids = read_subset_ids(args.subset)
    print(f"[OK] Subset count: {len(subset_ids):,}")

    print(f"[INFO] Reading {args.collection} …")
    coll_ids = read_collection_ids(args.collection)
    print(f"[OK] Collection count: {len(coll_ids):,}")

    # --- Set equality
    sset, cset = set(subset_ids), set(coll_ids)
    diff1, diff2 = sset - cset, cset - sset
    if not diff1 and not diff2:
        print("[OK] collection.tsv contains exactly the same passages as subset.tsv ✅")
    else:
        print(f"[WARN] Mismatch: {len(diff1)} missing_in_collection, {len(diff2)} missing_in_subset")
        print("Example missing_in_collection:", list(diff1)[:5])
        print("Example missing_in_subset:", list(diff2)[:5])

    # --- Detect internal vs external
    internal = detect_internal_ids(coll_ids)
    print(f"[INFO] Collection uses {'internal (0..N-1)' if internal else 'external'} passage IDs")

    # --- Check H5 order
    print(f"[INFO] Checking H5 order …")
    h5_ids = read_h5_ids(args.h5)
    if subset_ids == h5_ids:
        print("[OK] H5 IDs are in the same order as subset.tsv ✅")
    else:
        matches, k = sample_alignment(subset_ids, h5_ids)
        print(f"[WARN] H5 order differs ({matches}/{k} positions match). You may need explicit mapping.")

    # --- Optional mapping file
    if args.write_mapping and internal:
        print(f"[INFO] Writing mapping file {args.write_mapping} …")
        with io.open(args.write_mapping, "w", encoding="utf-8") as f:
            for i, ext_id in enumerate(subset_ids[:len(coll_ids)]):
                f.write(f"{i}\t{ext_id}\n")
        print("[OK] page_table.tsv written successfully ✅")
    elif args.write_mapping and not internal:
        print("[SKIP] No mapping needed: collection already uses external passage IDs")

if __name__ == "__main__":
    main()
