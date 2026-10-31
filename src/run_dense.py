"""pretrained embedding models, nothing fine tuned. python src/run_dense.py
writes results/embeddings.md and .json, one row per model.

the two protocols are the ones in run_baselines.py so the tables sit next to each
other. in_dictionary keeps the query's own row in the index, leave_one_out removes it.
the models are compared as they come, no training and no reranker, because this is the
step before deciding whether either is worth doing.

models come from models.txt in the repo root, one sentence-transformers id per line, or
from the default pair if that file is missing. a run appends to results/embeddings.json
instead of replacing it, so the person who runs the second model does not erase the row
the first one wrote.

on kaggle set the accelerator to GPU and turn internet on, the weights download the
first time. the vectors are cached as .npy in results, which is gitignored, so a rerun
does not encode again. encoding 135,907 definitions on a T4 took 66s for MiniLM and
164s for MPNet.
"""
import os
import re
import sys
import time
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import rexdict as R
from run_baselines import topk_from_scores, rows_to_words, summarise, TOP

OUTDIR = os.environ.get("REXDICT_RESULTS", "results")
DEFAULT = ["sentence-transformers/all-MiniLM-L6-v2",
           "sentence-transformers/all-mpnet-base-v2"]


def model_list():
    env = os.environ.get("REXDICT_MODELS")
    if env:
        return [m.strip() for m in env.split(",") if m.strip()]
    if os.path.exists("models.txt"):
        names = [l.strip() for l in open("models.txt") if l.strip() and not l.startswith("#")]
        if names:
            return names
    return DEFAULT


def tag(name):
    return re.sub(r"[^a-zA-Z0-9]+", "_", name.split("/")[-1]).strip("_").lower()


def failures_csv(model_tag, df, qidx, ranked, gold):
    """same columns and same rank rule as the bm25 file in run_baselines.py, so
    error_analysis.py can read either one without knowing which is which.

    ranked holds row numbers and gold holds word numbers. those are different id spaces
    because a word can have more than one row, so they are only comparable after
    rows_to_words, which is the map the metrics use. comparing them directly found no
    hit for any query and wrote all 500 rows as misses.
    """
    import pandas as pd
    n_rows = df.groupby("wid").size()
    words = rows_to_words(df, ranked)
    gold = np.asarray(gold)
    out = []
    for i, qi in enumerate(qidx):
        hit = words[i] == gold[i]
        rank = int(np.where(hit)[0][0]) + 1 if hit.any() else -1
        if rank == 1:
            continue
        tops = [df.at[int(ranked[i, j]), "word"] if ranked[i, j] >= 0 else "-"
                for j in range(min(3, ranked.shape[1]))]
        tops += ["-"] * (3 - len(tops))
        qi = int(qi)
        out.append(dict(query_row=qi, gold=df.at[qi, "word"], definition=df.at[qi, "definition"],
                        gold_rank=rank,
                        n_rows_for_gold=int(n_rows.get(df.at[qi, "wid"], 1)),
                        n_shared_def=int(df.at[qi, "n_accept"]),
                        top1=tops[0], top2=tops[1], top3=tops[2]))
    pd.DataFrame(out).to_csv(os.path.join(OUTDIR, "failures_%s_leave_one_out.csv" % model_tag),
                             index=False)
    # the check from next_week_goals.md, done here so the file cannot disagree with the
    # table again: a query is missing from this file exactly when it was a hit at 1.
    at1 = int((words[:, 0] == gold).sum())
    ok = "ok" if len(out) + at1 == len(qidx) else "MISMATCH"
    print("      failures written: %d of %d queries, %d hit at rank 1, %d + %d = %d  %s"
          % (len(out), len(qidx), at1, len(out), at1, len(qidx), ok))
    return len(out)


def encode(model, texts, cachepath):
    if os.path.exists(cachepath):
        e = np.load(cachepath)
        if e.shape[0] == len(texts):
            print("      using cached vectors %s" % (e.shape,))
            return e
    t0 = time.time()
    e = model.encode(texts, batch_size=256, normalize_embeddings=True,
                     convert_to_numpy=True, show_progress_bar=False)
    e = np.asarray(e, dtype=np.float32)
    np.save(cachepath, e)
    print("      encoded %d rows, dim %d, %.0fs" % (e.shape[0], e.shape[1], time.time() - t0))
    return e


def rank_dense(E, qidx, exclude_self, k=TOP, chunk=32):
    """cosine over unit vectors. returns row ids, best first, -1 padding."""
    n_docs = E.shape[0]
    k = min(k, n_docs - (1 if exclude_self else 0))
    qidx = np.asarray(qidx)
    out = np.full((len(qidx), max(k, 1)), -1, dtype=np.int64)
    for st in range(0, len(qidx), chunk):
        en = min(st + chunk, len(qidx))
        # one column per query, so transpose to [queries, docs] before ranking
        Eq = np.ascontiguousarray(E[qidx[st:en]], dtype=np.float32)
        sims = (E @ Eq.T).T.astype(np.float32, copy=True)   # [queries, docs]
        for j in range(en - st):
            if exclude_self:
                sims[j, qidx[st + j]] = -np.inf
            idx = topk_from_scores(sims[j], k)
            out[st + j, :len(idx)] = idx
        del sims
    return out


def main(models=None, max_len=64):
    from sentence_transformers import SentenceTransformer
    os.makedirs(OUTDIR, exist_ok=True)
    names = models or model_list()
    df, meta = R.load()
    tr, va, te = R.make_word_split(df)
    qidx = R.pick_queries(df, te)
    texts = df["definition"].astype(str).tolist()
    print("corpus rows=%d words=%d queries=%d  models=%s"
          % (len(df), meta["n_words"], len(qidx), ", ".join(names)))

    ref = {}
    bj = os.path.join(OUTDIR, "baselines.json")
    if os.path.exists(bj):
        b = json.load(open(bj))
        for m in ("bm25", "tfidf"):
            if m in b:
                ref[m] = {p: b[m][p]["strict"] for p in b[m]}

    rows_out = []
    for name in names:
        print("\n[%s]" % name, flush=True)
        t0 = time.time()
        try:
            model = SentenceTransformer(name, device="cuda")
        except Exception as e:
            print("      could not load on cuda (%s), falling back to cpu" % e)
            model = SentenceTransformer(name)
        model.max_seq_length = max_len
        E = encode(model, texts, os.path.join(OUTDIR, "emb_%s.npy" % tag(name)))
        for proto in ("in_dictionary", "leave_one_out"):
            ranked = rank_dense(E, qidx, exclude_self=(proto == "leave_one_out"))
            words = rows_to_words(df, ranked)
            strict = summarise(ranked, words, df, qidx, "strict")
            tol = summarise(ranked, words, df, qidx, "tolerant")
            row = dict(model=name, short=tag(name), protocol=proto, dim=int(E.shape[1]),
                       **{k: v for k, v in strict.items()},
                       recall_at_1_tolerant=tol["recall@1"],
                       recall_at_10_tolerant=tol["recall@10"],
                       recall_at_100_tolerant=tol["recall@100"],
                       secs=round(time.time() - t0, 1))
            if proto == "leave_one_out":
                failures_csv(tag(name), df, qidx, ranked, df["wid"].to_numpy()[qidx])
            rows_out.append(row)
            print("  %-28s %-14s R@1=%.3f R@10=%.3f R@100=%.3f  medrank=%s  self@1=%.1f%%"
                  % (tag(name), proto, strict["recall@1"], strict["recall@10"],
                     strict["recall@100"], strict["median_rank"], strict["pct_self_row_at_1"]))

    # two people run this file on different models, so whatever is already there for a
    # model we did not run today is kept. without this the later run erases the earlier row
    jp = os.path.join(OUTDIR, "embeddings.json")
    kept = []
    if os.path.exists(jp):
        try:
            prior = json.load(open(jp))
            done = {r["model"] for r in rows_out}
            kept = [r for r in prior.get("rows", []) if r["model"] not in done]
        except Exception:
            kept = []
    allrows = sorted(kept + rows_out, key=lambda r: (r["model"], r["protocol"]))
    order = sorted({r["model"] for r in allrows})
    # "table" is the key every other result file uses, so the notebook and the report
    # can read all of them the same way
    table = [dict(method=r["short"], protocol=r["protocol"], recall_1=r["recall@1"],
                  recall_10=r["recall@10"], recall_100=r["recall@100"],
                  median_rank=r["median_rank"], n_queries=int(len(qidx)))
             for r in allrows]
    for r in table:
        r["recall@1"], r["recall@10"], r["recall@100"] = (r.pop("recall_1"), r.pop("recall_10"),
                                                          r.pop("recall_100"))
    res = dict(models=order, n_queries=int(len(qidx)), rows=allrows, table=table,
               meta={k: meta[k] for k in ("rows", "n_words", "rows_raw") if k in meta},
               reference=ref)
    R.write_json(res, jp)

    L = ["# pretrained embeddings, no fine tuning", "",
         "from src/run_dense.py. %d fixed queries, cleaned corpus of %s rows over %s "
         "headwords, seed %d. same two protocols as results/baselines.md."
         % (len(qidx), f"{meta['rows']:,}", f"{meta['n_words']:,}", R.SEED), "",
         "| model | dim | protocol | R@1 | R@10 | R@100 | medrank | R@1 tol | own row at rank 1 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for name in order:
        for proto in ("in_dictionary", "leave_one_out"):
            r = [x for x in allrows if x["model"] == name and x["protocol"] == proto]
            if r:
                r = r[0]
                L.append("| %s | %d | %s | %.3f | %.3f | %.3f | %s | %.3f | %.1f%% |"
                         % (r["short"], r["dim"], proto, r["recall@1"], r["recall@10"],
                            r["recall@100"], r["median_rank"], r["recall_at_1_tolerant"],
                            r["pct_self_row_at_1"]))
    L += ["", "notes", ""]
    if "bm25" in ref and "leave_one_out" in ref["bm25"]:
        b = ref["bm25"]["leave_one_out"]
        L.append("- bm25 on this same query set is R@1 %.3f and R@100 %.3f leave_one_out, "
                 "from results/baselines.json. the rows above are meant to be read against "
                 "that column, not in isolation." % (b["recall@1"], b["recall@100"]))
    L.append("- no prompt, no fine tuning, the definition text goes in as it stands. that is "
             "what he asked for this week. if a model ships a query prompt and does better "
             "with it, that is a finding for the report, not this run")
    L.append("- these are modern English models over 1913 definitions, which is why he "
             "expects a fine tuned model later. any gap here is the size of the domain shift, "
             "and it is the argument for CP4")
    with open(os.path.join(OUTDIR, "embeddings.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\nwrote results/embeddings.md and .json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=None)
    ap.add_argument("--max_len", type=int, default=64)
    a = ap.parse_args()
    main([m for m in a.models.split(",") if m.strip()] if a.models else None, a.max_len)
