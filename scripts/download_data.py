#!/usr/bin/env python3
"""
download_data.py (subset-aligned, robust)

Builds a BM25-ready collection that EXACTLY matches the professor’s 1M subset:
- Input:  data/msmarco_passages_subset.tsv  (external passage IDs, one per line)
- Source: Hugging Face "sentence-transformers/msmarco-corpus", config="passage"
- Output:
    data/collection.tsv       (internal_id [0..N-1] <TAB> text)
    data/page_table.tsv       (internal_id <TAB> external_passage_id)

Run (from repo root):
  python -m scripts.download_data \
    --subset_ids data/msmarco_passages_subset.tsv \
    --out data/collection.tsv \
    --page_table data/page_table.tsv \
    --verify
"""

import argparse
import io
import os
import sys

# ---------- utils ----------

def repo_root() -> str:
    # Resolve to project root (directory containing this script's parent)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def abspath_from_root(path: str) -> str:
    # If already absolute, return; else resolve relative to repo root
    return path if os.path.isabs(path) else os.path.join(repo_root(), path)

def sanitize(text: str) -> str:
    return (text or "").replace("\t", " ").replace("\n", " ").strip()

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
                raise ValueError(f"[subset] non-integer id on line {ln}: {s!r}")
    if not ids:
        raise ValueError(f"[subset] no ids found in {path}")
    return ids

def to_int(x):
    if x is None:
        return None
    if isinstance(x, (bytes, bytearray)):
        x = x.decode("utf-8", errors="replace")
    try:
        return int(x)
    except Exception:
        return None

def get_pid(row):
    # Try common field names across MS MARCO variants
    for k in ("passage_id", "pid", "docid", "corpus_id", "id"):
        if k in row:
            v = to_int(row[k])
            if v is not None:
                return v
    return None

def get_text(row):
    # Prefer 'passage', then 'text'; fallback to title+text if present
    t = row.get("passage")
    if not t:
        t = row.get("text")
    if not t and "title" in row and "text" in row:
        t = f"{row['title']} {row['text']}"
    return sanitize(t or "")

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset_ids", default="data/msmarco_passages_subset.tsv",
                    help="Professor's subset ids (one external passage_id per line)")
    ap.add_argument("--out", default="data/collection.tsv",
                    help="Output: internal_id<TAB>text (written relative to repo root if not absolute)")
    ap.add_argument("--page_table", default="data/page_table.tsv",
                    help="Output: internal_id<TAB>external_passage_id")
    ap.add_argument("--hf_dataset", default="sentence-transformers/msmarco-corpus",
                    help="HF dataset repo (default: sentence-transformers/msmarco-corpus)")
    ap.add_argument("--hf_config", default="passage",
                    help="HF dataset config (default: 'passage')")
    ap.add_argument("--split", default="train",
                    help="HF dataset split (default: 'train')")
    ap.add_argument("--verify", action="store_true",
                    help="Print simple sanity checks after writing")
    args = ap.parse_args()

    subset_path   = abspath_from_root(args.subset_ids)
    out_path      = abspath_from_root(args.out)
    page_table    = abspath_from_root(args.page_table)

    # ensure parent dirs exist
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    os.makedirs(os.path.dirname(page_table), exist_ok=True)

    # 1) Read professor’s subset ids (ordered)
    subset_ids = read_subset_ids(subset_path)
    want = set(subset_ids)
    print(f"[INFO] subset ids: {len(subset_ids):,}")

    # 2) Load HF dataset
    try:
        from datasets import load_dataset
    except Exception:
        print("[ERR] Please install: pip install datasets", file=sys.stderr)
        sys.exit(2)
    try:
        from tqdm import tqdm  # progress bar
    except Exception:
        def tqdm(x, **k): return x

    print(f"[INFO] loading HF dataset: {args.hf_dataset} ({args.hf_config}/{args.split}) …")
    ds = load_dataset(args.hf_dataset, args.hf_config, split=args.split)

    # 3) Collect the exact subset into pid->text (stop early when done)
    keep = {}
    for row in tqdm(ds, desc="Scanning HF dataset"):
        pid = get_pid(row)
        if pid is None:
            continue
        if pid in want and pid not in keep:
            keep[pid] = get_text(row)
            if len(keep) == len(subset_ids):
                break

    print(f"[INFO] collected {len(keep):,} / {len(subset_ids):,} required passages")
    missing = [pid for pid in subset_ids if pid not in keep]
    if missing:
        print(f"[WARN] {len(missing):,} subset IDs not found in HF dataset. "
              f"First few: {missing[:10]}", file=sys.stderr)

    # 4) Write outputs in the EXACT subset order:
    #    - collection.tsv: internal_id  text
    #    - page_table.tsv: internal_id  external_id
    written = 0
    with io.open(out_path, "w", encoding="utf-8") as fout, \
         io.open(page_table, "w", encoding="utf-8") as ptab:
        for internal_id, pid in enumerate(subset_ids):
            text = keep.get(pid)
            if text is None:
                continue
            fout.write(f"{internal_id}\t{text}\n")
            ptab.write(f"{internal_id}\t{pid}\n")
            written += 1

    print(f"[OK] wrote {written:,} rows -> {out_path}")
    print(f"[OK] wrote mapping internal->external -> {page_table}")

    # 5) Optional quick sanity check
    if args.verify:
        print("[VERIFY] quick checks:")
        print("  - constructed in subset order (guaranteed)")
        print("  - first 3 ids mapping (internal->external):")
        for i in range(min(3, len(subset_ids))):
            print(f"    {i} -> {subset_ids[i]}")

if __name__ == "__main__":
    main()
