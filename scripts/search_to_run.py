#!/usr/bin/env python3
"""
search_to_run.py
----------------
Batch-evaluate BM25 over a query set.
Outputs a TREC-style run file for evaluation (qid Q0 docid rank score tag).

Supports --page_table to map internal docIDs -> external passage IDs.
Handles messy query TSVs (blank lines, space-delimited lines, headers).
"""
 
-import os, sys, heapq, argparse
+import os, sys, heapq, argparse
+
+# --- repo-root on sys.path so `from src...` works regardless of how you invoke ---
+_ROOT = os.path.dirname(os.path.abspath(os.path.join(__file__, "..")))
+if _ROOT not in sys.path:
+    sys.path.insert(0, _ROOT)
+# ---------------------------------------------------------------------------------

 from src.query_bm25 import (
     load_lexicon, load_doclens, read_term_postings, BM25,
     score_disjunctive, score_conjunctive
 )

@@
 def search_all_queries(index_dir, queries_path, run_out, k1, b, mode, topk=1000, page_table=None):
     lexicon = load_lexicon(os.path.join(index_dir, "lexicon.tsv"))
     doclens = load_doclens(os.path.join(index_dir, "doclen.bin"))
     bm25 = BM25(doclens, k1=k1, b=b)
     postings_path = os.path.join(index_dir, "postings.bin")
 
-    total = 0
+    total = kept = 0
     with open(postings_path, "rb") as pf, open(run_out, "w", encoding="utf-8") as outf:
         for qid, qtext in iter_queries(queries_path):
             total += 1
             terms = [t for t in qtext.lower().split() if t]
             if not terms:
                 continue
 
             # collect postings
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
                 # no in-vocab terms for this query
                 continue
 
             scores = (score_disjunctive if mode == "disj" else score_conjunctive)(bm25, term_postings)
             topK = heapq.nlargest(topk, scores.items(), key=lambda x: x[1])
+            if not topK:
+                continue
+            kept += 1
 
             for rank, (doc, score) in enumerate(topK, 1):
                 ext_docid = page_table.get(doc, str(doc)) if page_table else str(doc)
-                outf.write(f"{qid}\tQ0\t{ext_docid}\t{rank}\t{score:.4f}\tBM25\n")
+                # TREC format = SPACE-separated (trec_eval is picky)
+                outf.write(f"{qid} Q0 {ext_docid} {rank} {score:.4f} BM25\n")
 
-    print(f"[OK] wrote run file -> {run_out}  (processed {total} queries)")
+    print(f"[OK] wrote run file -> {run_out}  (read {total} queries, produced {kept})")
 
 if __name__ == "__main__":
     ap = argparse.ArgumentParser()
     ap.add_argument("--index_dir", required=True)
     ap.add_argument("--queries", required=True)
     ap.add_argument("--run_out", required=True)
     ap.add_argument("--k1", type=float, default=0.9)
     ap.add_argument("--b", type=float, default=0.4)
     ap.add_argument("--mode", choices=["disj","conj"], default="disj")
     ap.add_argument("--topk", type=int, default=1000)
     ap.add_argument("--page_table", help="Optional mapping internal->external doc IDs", default=None)
+    ap.add_argument("--max_queries", type=int, default=None, help="Optional cap for smoke tests")
     args = ap.parse_args()
 
     mapping = load_page_table(args.page_table) if args.page_table else None
-    search_all_queries(
-        args.index_dir, args.queries, args.run_out,
-        args.k1, args.b, args.mode, args.topk, mapping
-    )
+    if args.max_queries:
+        # quick smoke test on first N lines without editing the file
+        import tempfile, itertools
+        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as tmp:
+            n = 0
+            for line in open(args.queries, "r", encoding="utf-8"):
+                if line.strip():
+                    tmp.write(line)
+                    n += 1
+                    if n >= args.max_queries:
+                        break
+            qpath = tmp.name
+    else:
+        qpath = args.queries
+
+    search_all_queries(
+        args.index_dir, qpath, args.run_out,
+        args.k1, args.b, args.mode, args.topk, mapping
+    )
