"""95% bootstrap intervals for the model table. python src/bootstrap_ci.py

one row per method on the frozen 500 queries, leave_one_out, strict scoring. the 500
queries are resampled with replacement B times (same resample for every method, so the
differences below are paired) and the 2.5th and 97.5th percentiles are reported.

methods: bm25 and tfidf, rebuilt here with the same code as run_baselines.py, and every
dense model named in results/embeddings.json (or --models). dense vectors come from the
emb_*.npy files that run_dense.py caches in results. if a model has no cache the weights
are downloaded and the corpus is encoded, which is slow on a CPU.

the intervals only show how much the numbers move if a different 500 queries had been
drawn. they say nothing about training noise, and they cannot be compared across
different query sets.

writes results/bootstrap_ci.json and results/bootstrap_ci.md
"""
import os
import sys
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import rexdict as R
from run_baselines import BM25, rank_built, rank_tfidf, rows_to_words
from run_dense import rank_dense, tag, model_list, OUTDIR

KS = (1, 10, 100)
SEED = 2026


def gold_ranks(words, gold):
    """1-based position of the gold word in each ranked word list, a huge number if absent.
    -1 padding never equals a gold id, so a short list counts as not found."""
    big = np.iinfo(np.int32).max
    hit = words == np.asarray(gold)[:, None]
    return np.where(hit.any(axis=1), hit.argmax(axis=1) + 1, big).astype(np.int64)


def metrics(ranks):
    """ranks [..., n] -> recall@k and median rank (capped at MAX_RANK like evaluate())."""
    out = {"recall@%d" % k: (ranks <= k).mean(axis=-1) for k in KS}
    out["median_rank"] = np.median(np.minimum(ranks, R.MAX_RANK), axis=-1)
    return out


def interval(v):
    lo, hi = np.percentile(v, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def load_dense(name, texts, max_len):
    path = os.path.join(OUTDIR, "emb_%s.npy" % tag(name))
    if os.path.exists(path):
        E = np.load(path)
        if E.shape[0] == len(texts):
            print("      using cached vectors %s" % (E.shape,), flush=True)
            return E
        print("      cache has %d rows, corpus has %d, encoding again" % (E.shape[0], len(texts)))
    from sentence_transformers import SentenceTransformer
    from run_dense import encode
    try:
        model = SentenceTransformer(name, device="cuda")
    except Exception as e:
        print("      could not load on cuda (%s), falling back to cpu" % e)
        model = SentenceTransformer(name)
    model.max_seq_length = max_len
    return encode(model, texts, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=None, help="comma separated, default: embeddings.json")
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--no_lexical", action="store_true", help="skip bm25 and tfidf")
    a = ap.parse_args()

    t0 = time.time()
    df, meta = R.load()
    tr, va, te = R.make_word_split(df)
    qidx = R.pick_queries(df, te)
    gold = df["wid"].to_numpy()[qidx]
    n_docs, n = len(df), len(qidx)
    texts = df["definition"].astype(str).tolist()
    print("corpus rows=%d words=%d queries=%d" % (n_docs, meta["n_words"], n), flush=True)

    ranks, check = {}, {}

    def add(name, rows):
        words = rows_to_words(df, rows)
        ranks[name] = gold_ranks(words, gold)
        # same numbers as the other result files, or this script is not measuring the same thing
        check[name] = R.evaluate(words, df, qidx, mode="strict")

    if not a.no_lexical:
        print("[bm25]", flush=True)
        tok = df["definition"].map(R.tokens).tolist()
        bm = BM25(tok)
        add("bm25", rank_built(lambda qi: bm.scores(tok[qi]), qidx, True, n_docs))
        print("[tfidf]", flush=True)
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(sublinear_tf=True, dtype=np.float32, token_pattern=r"[a-z']{2,}")
        D = vec.fit_transform(df["definition"].str.lower())
        Q = vec.transform(df.loc[qidx, "definition"].str.lower())
        add("tfidf", rank_tfidf(Q, D, qidx, True, n_docs))

    names = [m.strip() for m in a.models.split(",")] if a.models else None
    if names is None:
        jp = os.path.join(OUTDIR, "embeddings.json")
        names = json.load(open(jp)).get("models") if os.path.exists(jp) else None
    for name in names or model_list():
        print("[%s]" % name, flush=True)
        E = load_dense(name, texts, a.max_len)
        add(tag(name), rank_dense(E, qidx, exclude_self=True))

    # ---- bootstrap. one resample of the query set, shared by every method
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, n, size=(a.B, n))
    boot = {m: metrics(r[idx]) for m, r in ranks.items()}
    point = {m: {k: float(v) for k, v in metrics(r).items()} for m, r in ranks.items()}

    for m in ranks:
        for k in ("recall@1", "recall@10", "recall@100"):
            if abs(point[m][k] - check[m][k]) > 1e-3:
                print("WARNING: %s %s is %.4f here but %.4f from evaluate()"
                      % (m, k, point[m][k], check[m][k]))
        if int(point[m]["median_rank"]) != check[m]["median_rank"]:
            print("WARNING: %s median rank %s here but %s from evaluate()"
                  % (m, point[m]["median_rank"], check[m]["median_rank"]))

    ref = "bm25" if "bm25" in ranks else next(iter(ranks))
    res = {"n_queries": int(n), "B": int(a.B), "seed": SEED, "protocol": "leave_one_out",
           "mode": "strict", "reference": ref, "methods": {}, "paired_vs_reference": {}}
    for m in ranks:
        res["methods"][m] = {k: {"value": round(point[m][k], 4),
                                 "ci95": interval(boot[m][k])} for k in point[m]}
        if m == ref:
            continue
        d = {}
        for k in point[m]:
            diff = boot[m][k] - boot[ref][k]
            lo, hi = interval(diff)
            d[k] = {"diff": round(point[m][k] - point[ref][k], 4), "ci95": [lo, hi],
                    "excludes_zero": bool(lo > 0 or hi < 0)}
        res["paired_vs_reference"][m] = d
    res["runtime_sec"] = round(time.time() - t0, 1)

    os.makedirs(OUTDIR, exist_ok=True)
    R.write_json(res, os.path.join(OUTDIR, "bootstrap_ci.json"))
    write_md(res)
    print("\nwrote %s and %s (%.0fs)" % (os.path.join(OUTDIR, "bootstrap_ci.md"),
                                       os.path.join(OUTDIR, "bootstrap_ci.json"), res["runtime_sec"]))


def fmt(c, digits=3):
    return "%.*f [%.*f, %.*f]" % (digits, c["value"], digits, c["ci95"][0], digits, c["ci95"][1])


def write_md(res):
    L = ["# model table with 95% bootstrap intervals", "",
         "from src/bootstrap_ci.py. %d fixed queries, leave_one_out, strict scoring, %d resamples "
         "of the queries with replacement, seed %d. the same resample is used for every method."
         % (res["n_queries"], res["B"], res["seed"]), "",
         "| method | R@1 | R@10 | R@100 | median rank |", "|---|---|---|---|---|"]
    for m, v in res["methods"].items():
        mr = v["median_rank"]
        L.append("| %s | %s | %s | %s | %d [%d, %d] |"
                 % (m, fmt(v["recall@1"]), fmt(v["recall@10"]), fmt(v["recall@100"]),
                    mr["value"], mr["ci95"][0], mr["ci95"][1]))
    L += ["", "## paired difference against %s" % res["reference"], "",
          "the difference is taken inside each resample, so it is tighter than comparing the "
          "intervals above by eye. an interval that contains 0 means the two methods cannot be "
          "told apart on these 500 queries.", "",
          "| method | R@1 diff | R@10 diff | R@100 diff | median rank diff |", "|---|---|---|---|---|"]
    for m, d in res["paired_vs_reference"].items():
        cells = []
        for k in ("recall@1", "recall@10", "recall@100", "median_rank"):
            x = d[k]
            dg = 0 if k == "median_rank" else 3
            cells.append("%+.*f [%+.*f, %+.*f]%s" % (dg, x["diff"], dg, x["ci95"][0], dg, x["ci95"][1],
                                                      " *" if x["excludes_zero"] else ""))
        L.append("| %s | %s |" % (m, " | ".join(cells)))
    L += ["", "\\* the interval excludes 0.", "",
          "notes", "",
          "- the intervals show how much a number moves if a different 500 queries had been drawn. "
          "they do not include training noise or the choice of cleaning rules.",
          "- all methods are scored on the same queries, so differences are paired.",
          "- the median rank is capped at %d, so a median of %d means more than half the queries "
          "are not found in the top %d." % (R.MAX_RANK, R.MAX_RANK, R.MAX_RANK)]
    with open(os.path.join(OUTDIR, "bootstrap_ci.md"), "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()