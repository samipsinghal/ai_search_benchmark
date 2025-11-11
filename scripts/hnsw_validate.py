#!/usr/bin/env python3
import os, sys, shlex, subprocess, re, csv, pathlib

# --------- Config (edit if you use different paths) ----------
SAMPLE_TSV = "data/queries.dev.sample100.tsv"         # 100-query subset
QRELS_TREC = "data/qrels.dev.trec"                    # dev qrels converted to TREC format
QUERY_H5   = "data/msmarco_queries_dev_eval_embeddings.h5"
INDEXES    = [
    ("index/faiss_hnsw_M4C50",  4,  50,  50),
    ("index/faiss_hnsw_M4C80",  4,  80, 100),
    ("index/faiss_hnsw_M8C200", 8, 200, 200),
]
TOPK       = 1000
OUTDIR     = "runs"
CSV_OUT    = "hnsw_validation.csv"

# --------- Helpers ----------
TIME_RE = re.compile(r"\[TIME\]\s+([0-9.]+)s\s+\|\s+\[MEM\]\s+([0-9.]+)\s+MB")

def run(cmd):
    print(f"\n>>> {cmd}")
    p = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out_lines = []
    for line in p.stdout:
        print(line, end="")
        out_lines.append(line)
    p.wait()
    out = "".join(out_lines)
    if p.returncode != 0:
        sys.exit(f"ERROR: command failed: {cmd}")
    return out

def ensure_index(idx_dir, M, efC):
    idx_file = os.path.join(idx_dir, "hnsw.index")
    if os.path.exists(idx_file):
        print(f"[OK] Index exists: {idx_file}")
        return
    pathlib.Path(idx_dir).mkdir(parents=True, exist_ok=True)
    cmd = (
        f"python -m scripts.build_hnsw "
        f"--emb_h5 data/msmarco_passages_embeddings_subset.h5 "
        f"--out_dir {idx_dir} --M {M} --efC {efC}"
    )
    run(cmd)
    if not os.path.exists(idx_file):
        sys.exit(f"ERROR: failed to build index at {idx_file}")

def search(idx_dir, efS, tag):
    out_path = os.path.join(OUTDIR, f"S2_hnsw.sample100.{tag}.trec")
    pathlib.Path(OUTDIR).mkdir(parents=True, exist_ok=True)
    cmd = (
        f"python -m scripts.search_hnsw "
        f"--query_h5 {QUERY_H5} --qid_key id --qemb_key embedding "
        f"--index_dir {idx_dir} --qid_list_tsv {SAMPLE_TSV} "
        f"--topk {TOPK} --efS {efS} --run_out {out_path} --tag FAISS"
    )
    out = run(cmd)
    m = TIME_RE.search(out)
    if not m:
        sys.exit("ERROR: Did not find [TIME] ... | [MEM] ... in output. "
                 "Ensure psutil logger is added to search_hnsw.py")
    total_s = float(m.group(1))
    mem_mb  = float(m.group(2))
    return out_path, total_s, mem_mb

def trec_eval(qrels_trec, run_trec):
    # Needs trec_eval binary in PATH; if not present, guide user.
    # Metrics: MAP, Recall@100, NDCG@10
    try:
        cmd = f"trec_eval -m map -m recall.100 -m ndcg_cut.10 {qrels_trec} {run_trec}"
        out = run(cmd)
    except SystemExit:
        print("\n[HINT] If trec_eval isn't installed, install it or use pytrec_eval.")
        raise

    def parse_metric(name):
        for line in out.splitlines():
            parts = line.strip().split()
            if len(parts) == 3 and parts[0] == name and parts[1] in ("all",""):
                try:
                    return float(parts[2])
                except:
                    pass
        return None

    return {
        "MAP":         parse_metric("map"),
        "Recall@100":  parse_metric("recall_100"),
        "NDCG@10":     parse_metric("ndcg_cut_10"),
    }

# --------- Main ----------
def main():
    rows = [("M","efC","efS","MAP","Recall@100","NDCG@10","Total Time (s)","Avg Time/Query (s)","Peak Mem (MB)","Run File")]
    for idx_dir, M, efC, efS in INDEXES:
        tag = f"M{M}C{efC}S{efS}"
        print(f"\n=== Validating HNSW {tag} ===")
        ensure_index(idx_dir, M, efC)
        run_file, total_s, mem_mb = search(idx_dir, efS, tag)

        # evaluate
        if not os.path.exists(QRELS_TREC):
            sys.exit(f"ERROR: Missing qrels TREC file: {QRELS_TREC} "
                     f"(convert from TSV as in README)")
        metrics = trec_eval(QRELS_TREC, run_file)

        avg_s = round(total_s / 100.0, 6)  # 100-query sample
        rows.append((M, efC, efS, 
                     f"{metrics['MAP']:.6f}" if metrics['MAP'] is not None else "NA",
                     f"{metrics['Recall@100']:.6f}" if metrics['Recall@100'] is not None else "NA",
                     f"{metrics['NDCG@10']:.6f}" if metrics['NDCG@10'] is not None else "NA",
                     f"{total_s:.2f}",
                     f"{avg_s:.6f}",
                     f"{mem_mb:.1f}",
                     run_file))

    with open(CSV_OUT, "w", newline="") as f:
        csv.writer(f).writerows(rows)

    print("\n=== Table 3: Effect of HNSW parameter tuning on quality and efficiency ===")
    hdr, *data = rows
    widths = [max(len(str(x)) for x in col) for col in zip(*rows)]
    print(" | ".join(str(h).ljust(w) for h, w in zip(hdr, widths)))
    print("-+-".join("-"*w for w in widths))
    for r in data:
        print(" | ".join(str(v).ljust(w) for v, w in zip(r, widths)))
    print(f"\n[CSV] Wrote {CSV_OUT}")

if __name__ == "__main__":
    main()
