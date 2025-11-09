#!/usr/bin/env python3
import os, sys, argparse, h5py, numpy as np
ROOT = os.path.dirname(os.path.abspath(os.path.join(__file__, "..")))
if ROOT not in sys.path: sys.path.insert(0, ROOT)

def autodetect_q_keys(f):
    qid_key = emb_key = None
    for k in f.keys():
        d = f[k]
        if len(d.shape) == 1 and d.dtype.kind in "iu": qid_key = qid_key or k
        if len(d.shape) == 2 and d.dtype.kind == "f":  emb_key = emb_key or k
    # fallback names
    if qid_key is None and "qid" in f: qid_key = "qid"
    if emb_key is None and "embedding" in f: emb_key = "embedding"
    if emb_key is None and "emb" in f: emb_key = "emb"
    return qid_key, emb_key

def qid_order(tsv):
    out=[]
    with open(tsv,"r",encoding="utf-8") as f:
        for line in f:
            s=line.strip()
            if not s: continue
            out.append(s.split("\t",1)[0] if "\t" in s else s.split(None,1)[0])
    return [str(x) for x in out]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query_h5", required=True)
    ap.add_argument("--index_dir", required=True)
    ap.add_argument("--qid_list_tsv", required=True)
    ap.add_argument("--topk", type=int, default=1000)
    ap.add_argument("--efS", type=int, default=200)
    ap.add_argument("--run_out", required=True)
    ap.add_argument("--tag", default="FAISS_HNSW_IP")
    ap.add_argument("--qid_key", default=None)
    ap.add_argument("--qemb_key", default=None)
    args = ap.parse_args()

    import faiss
    index = faiss.read_index(os.path.join(args.index_dir, "hnsw.index"))
    index.hnsw.efSearch = int(args.efS)
    pids = np.load(os.path.join(args.index_dir, "pids.npy"))

    with h5py.File(args.query_h5,"r") as f:
        qid_key, qemb_key = args.qid_key, args.qemb_key
        if qid_key is None or qemb_key is None:
            ad_qid, ad_emb = autodetect_q_keys(f)
            qid_key = qid_key or ad_qid
            qemb_key = qemb_key or ad_emb
        if qid_key is None or qemb_key is None:
            raise RuntimeError(f"Could not find qid/embedding in {args.query_h5}. Found {list(f.keys())}")
        qids = f[qid_key][:]
        # accept ints or strings
        qids = qids.astype(str)
        qvecs = f[qemb_key][:].astype("float32")

    row = {str(q): i for i, q in enumerate(qids)}
    wanted = qid_order(args.qid_list_tsv)

    wrote=0
    with open(args.run_out, "w", encoding="utf-8") as out:
        for q in wanted:
            i = row.get(str(q))
            if i is None: 
                continue
            qv = qvecs[i:i+1]
            D, I = index.search(qv, int(args.topk))
            for rank,(score, idx) in enumerate(zip(D[0], I[0]), 1):
                out.write(f"{q} Q0 {int(pids[idx])} {rank} {float(score):.6f} {args.tag}\n")
            wrote += 1

    print(f"[OK] wrote {wrote} queries → {args.run_out} (qid_key={qid_key}, emb_key={qemb_key})")

if __name__ == "__main__":
    main()
