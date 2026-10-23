%%writefile src/run_minilm.py
"""pretrained sentence-embedding baseline, no fine tuning.
    python src/run_minilm.py
    python src/run_minilm.py --model sentence-transformers/all-mpnet-base-v2 --tag mpnet
Same corpus, same 500 queries, same two protocols and same evaluate() as run_baselines.py.
"""
import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import rexdict as R

OUTDIR = os.environ.get("REXDICT_RESULTS", "results")
CACHE = os.environ.get("REXDICT_CACHE", "emb_cache")
TOP = 1000


def encode(df, model_name, tag, max_len, batch):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, tag + ".npy")
    if os.path.isfile(path):
        D = np.load(path)
        if D.shape[0] == len(df):
            print("loaded cached embeddings", path, D.shape, flush=True)
            return D
        print("cache has %d rows, corpus has %d. re-encoding" % (D.shape[0], len(df)))
    from sentence_transformers import SentenceTransformer
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"
    print("encoding %d definitions with %s on %s" % (len(df), model_name, device), flush=True)
    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = max_len
    D = model.encode(df["definition"].tolist(), batch_size=batch,
                     normalize_embeddings=True, show_progress_bar=True,
                     convert_to_numpy=True).astype(np.float32)
    np.save(path, D)
    return D


def rank_rows(Q, D, qidx, exclude_self, k=TOP, chunk=50):
    k = min(k, len(D) - (1 if exclude_self else 0))
    out = np.zeros((len(qidx), k), dtype=np.int64)
    for st in range(0, len(qidx), chunk):
        en = min(st + chunk, len(qidx))
        sims = Q[st:en] @ D.T
        if exclude_self:
            for j in range(en - st):
                sims[j, qidx[st + j]] = -np.inf
        part = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        for j in range(en - st):
            p = part[j]
            out[st + j] = p[np.lexsort((p, -sims[j, p]))]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--tag", default="minilm")
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--batch", type=int, default=256)
    a = ap.parse_args()

    t0 = time.time()
    df, meta = R.load()
    tr, va, te = R.make_word_split(df)
    qidx = R.pick_queries(df, te)
    gold = df["wid"].to_numpy()[qidx]
    wid = df["wid"].to_numpy()
    print("corpus rows=%d words=%d queries=%d" % (len(df), meta["n_words"], len(qidx)))

    D = encode(df, a.model, a.tag, a.max_len, a.batch)
    Q = D[qidx]
    print("encoding done %.0fs" % (time.time() - t0), flush=True)

    rows_per_word = df.groupby("wid").size().reindex(gold).to_numpy()
    out = {"model": a.model, "tag": a.tag, "max_seq_length": a.max_len,
           "n_queries": int(len(qidx)), "meta": meta,
           "ceilings": R.ceilings(df, qidx)}
    table, ranked_loo = [], None
    for proto, ex in (("in_dictionary", False), ("leave_one_out", True)):
        rows = rank_rows(Q, D, qidx, ex)
        words = wid[rows]
        strict = R.evaluate(words, df, qidx, mode="strict")
        tol = R.evaluate(words, df, qidx, mode="tolerant")
        self1 = round(float((rows[:, 0] == qidx).mean()) * 100, 2)
        strict["pct_self_row_at_1"] = self1
        out.setdefault(a.tag, {})[proto] = {"strict": strict, "tolerant": tol}
        table.append(dict(method=a.tag, protocol=proto, **strict,
                          r1_tol=tol["recall@1"], r10_tol=tol["recall@10"],
                          r100_tol=tol["recall@100"]))
        print("  %-14s R@1=%.3f R@10=%.3f R@100=%.3f medrank=%s | tol R@1=%.3f R@10=%.3f R@100=%.3f | self@1=%.1f%%"
              % (proto, strict["recall@1"], strict["recall@10"], strict["recall@100"],
                 strict["median_rank"], tol["recall@1"], tol["recall@10"], tol["recall@100"], self1))
        if ex:
            ranked_loo = rows

    if not table[0]["recall@1"] > table[1]["recall@1"] + 0.2:
        print("WARNING: in_dictionary is not clearly above leave_one_out. protocol looks broken.")
    if len(qidx) != 500:
        print("WARNING: n_queries is %d, expected 500" % len(qidx))

    words = wid[ranked_loo]
    fails = []
    for i in range(len(qidx)):
        hit = np.where(words[i] == gold[i])[0]
        rank = int(hit[0]) + 1 if len(hit) else -1
        if rank == 1:
            continue
        top = [df.at[int(ranked_loo[i, j]), "word"] for j in range(5)]
        fails.append(dict(query_row=int(qidx[i]), gold=df.at[qidx[i], "word"],
                          definition=df.at[qidx[i], "definition"], gold_rank=rank,
                          n_rows_for_gold=int(rows_per_word[i]),
                          n_shared_def=int(df.at[qidx[i], "n_accept"]),
                          top1=top[0], top2=top[1], top3=top[2], top4=top[3], top5=top[4]))
    os.makedirs(OUTDIR, exist_ok=True)
    pd.DataFrame(fails).to_csv(os.path.join(OUTDIR, "failures_%s_leave_one_out.csv" % a.tag),
                               index=False)
    out["n_failures_leave_one_out"] = len(fails)
    out["pct_queries_unreachable_without_self_row"] = round(
        float((rows_per_word <= 1).mean()) * 100, 2)
    out["runtime_sec"] = round(time.time() - t0, 1)
    out["table"] = table
    R.write_json(out, os.path.join(OUTDIR, "%s.json" % a.tag))
    write_md(out, a)
    print("\nwrote results/%s.md, .json and failures csv (%.0fs)" % (a.tag, time.time() - t0))


def write_md(out, a):
    L = ["# %s, pretrained embeddings, no fine tuning" % a.tag, "",
         "from src/run_minilm.py. model %s, max_seq_length %d, cosine over L2 normalised "
         "definition embeddings, %d fixed queries, seed %d. took %ss."
         % (a.model, a.max_len, out["n_queries"], R.SEED, out["runtime_sec"]), "",
         "| method | protocol | R@1 | R@10 | R@100 | medrank | R@1 tol | R@10 tol | R@100 tol | own row at rank 1 |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in out["table"]:
        L.append("| %s | %s | %.3f | %.3f | %.3f | %s | %.3f | %.3f | %.3f | %s%% |"
                 % (r["method"], r["protocol"], r["recall@1"], r["recall@10"], r["recall@100"],
                    r["median_rank"], r["r1_tol"], r["r10_tol"], r["r100_tol"],
                    r["pct_self_row_at_1"]))
    L += ["", "notes", "",
          "- leave_one_out is the number to quote. in_dictionary keeps the query's own row in the "
          "index, so it is lookup and not understanding.",
          "- %.1f%% of queries have no other row for their word, so leave_one_out cannot reach 1.0. "
          "it is a floor." % out["pct_queries_unreachable_without_self_row"],
          "- strict R@1 cannot pass %.3f even with a perfect ranker (shared definitions)."
          % out["ceilings"]["ceiling_strict_top1"],
          "- tol counts any word with the same definition string, a string rule and not wordnet.",
          "- %d of %d queries missed at rank 1 under leave_one_out, see failures_%s_leave_one_out.csv."
          % (out["n_failures_leave_one_out"], out["n_queries"], a.tag)]
    with open(os.path.join(OUTDIR, "%s.md" % a.tag), "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()