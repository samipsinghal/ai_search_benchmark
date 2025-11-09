#!/usr/bin/env python3
import os, sys, json, argparse, h5py, numpy as np

# so "src" is importable if you later need it
ROOT = os.path.dirname(os.path.abspath(os.path.join(__file__, "..")))
if ROOT not in sys.path: sys.path.insert(0, ROOT)

def autodetect_keys(f):
    pid_key = emb_key = None
    for k in f.keys():
        d = f[k]
        if len(d.shape) == 1 and d.dtype.kind in "iu": pid_key = pid_key or k
        if len(d.shape) == 2 and d.dtype.kind == "f":  emb_key = emb_key or k
    # common fallback names seen in course dumps
    if pid_key is None and "id" in f: pid_key = "id"
    if emb_key is None and "embedding" in f: emb_key = "embedding"
    return pid_key, emb_key

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_h5", required=True, help="H5 file with passage embeddings")
    ap.add_argument("--out_dir", required=True, help="Output directory for FAISS index")
    ap.add_argument("--M", type=int, default=8, help="HNSW M (neighbors)")
    ap.add_argument("--efC", type=int, default=200, help="HNSW efConstruction")
    ap.add_argument("--pid_key", default=None, help="dataset key for passage ids (auto if omitted)")
    ap.add_argument("--emb_key", default=None, help="dataset key for passage vectors (auto if omitted)")
    args = ap.parse_args()

    import faiss  # import here to keep import errors obvious
    os.makedirs(args.out_dir, exist_ok=True)

    with h5py.File(args.emb_h5, "r") as f:
        pid_key, emb_key = args.pid_key, args.emb_key
        if pid_key is None or emb_key is None:
            ad_pid, ad_emb = autodetect_keys(f)
            pid_key = pid_key or ad_pid
            emb_key = emb_key or ad_emb
        if pid_key is None or emb_key is None:
            raise RuntimeError(f"Could not find id/embedding datasets in {args.emb_h5}. Found {list(f.keys())}")

        pid_ds = f[pid_key]
        # accept ints or strings
        if pid_ds.dtype.kind in "iu":
            pids = pid_ds[:]
        else:
            pids = pid_ds[:].astype(str).astype(np.int64)

        vecs = f[emb_key][:].astype("float32")

    d = int(vecs.shape[1])
    index = faiss.IndexHNSWFlat(d, args.M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = int(args.efC)
    index.add(vecs)

    faiss.write_index(index, os.path.join(args.out_dir, "hnsw.index"))
    np.save(os.path.join(args.out_dir, "pids.npy"), pids)
    with open(os.path.join(args.out_dir, "meta.json"), "w") as g:
        json.dump({
            "M": args.M, "efConstruction": args.efC, "metric": "inner_product",
            "pid_key": pid_key, "emb_key": emb_key, "count": int(vecs.shape[0]), "dim": d
        }, g, indent=2)

    print(f"[OK] HNSW(IP) built for {len(pids)} passages @ dim={d} → {args.out_dir} "
          f"(pid_key={pid_key}, emb_key={emb_key})")

if __name__ == "__main__":
    main()
