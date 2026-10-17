"""bm25 and tfidf over the definitions. python src/run_baselines.py
writes results/baselines.md and .json
each method runs twice. in_dictionary keeps the query's own row in the index, so a
lexical baseline can win by string matching. leave_one_out removes that row and the
word has to be reached through a sibling sense or a neighbour definition. both numbers
go in the table, only reporting the first one would be misleading.
"""
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import rexdict as R

OUTDIR = os.environ.get("REXDICT_RESULTS", "results")
TOP = 1000  # how deep we keep the ranking; MAX_RANK is the cap used for reporting


class BM25:
    """Okapi BM25. k1=1.2, b=0.75, the usual defaults. Written here instead of
    imported from rank_bm25 so that the indexing details are ours."""

    def __init__(self, token_lists, k1=1.2, b=0.75):
        self.k1 = k1
        self.b = b
        self.N = len(token_lists)
        self.doc_len = np.fromiter((len(t) for t in token_lists), np.float32, self.N)
        self.avgdl = float(self.doc_len.mean()) or 1.0
        df = Counter()
        post = {}
        for di, toks in enumerate(token_lists):
            tf = Counter(toks)
            for t, f in tf.items():
                df[t] += 1
                p = post.setdefault(t, ([], []))
                p[0].append(di)
                p[1].append(f)
        self.post = {t: (np.array(a, np.int32), np.array(v, np.float32))
                     for t, (a, v) in post.items()}
        self.idf = {t: np.log(1.0 + (self.N - df[t] + 0.5) / (df[t] + 0.5))
                    for t in self.post}
        self.n_terms = len(self.post)
        self.n_query_free = sum(1 for t, c in df.items() if c == 1)

    def scores(self, q_tokens):
        s = np.zeros(self.N, dtype=np.float32)
        for t in set(q_tokens):
            p = self.post.get(t)
            if p is None:
                continue
            di, tf = p
            norm = tf + self.k1 * (1.0 - self.b + self.b * self.doc_len[di] / self.avgdl)
            s[di] += self.idf[t] * (tf * (self.k1 + 1.0)) / norm
        return s


def topk_from_scores(s, k):
    """indices of the k highest scores, best first. avoids a full sort when it can."""
    n = len(s)
    k = min(k, n)
    if k >= n:
        idx = np.lexsort((np.arange(n), -s))
    else:
        part = np.argpartition(-s, k)[:k]
        idx = part[np.lexsort((part, -s[part]))]
    return idx[np.isfinite(s[idx])][:k]


def rank_built(score_fn, qidx, exclude_self, n_docs, k=TOP):
    """score_fn(row) -> dense scores over all rows.
    dead rows are dropped, so a short list means the answer is not in it. padding is -1."""
    k = min(k, n_docs - (1 if exclude_self else 0))
    out = np.full((len(qidx), max(k, 1)), -1, dtype=np.int64)
    for i, qi in enumerate(qidx):
        s = np.asarray(score_fn(int(qi)), dtype=np.float32)
        if exclude_self:
            s[qi] = -np.inf
        idx = topk_from_scores(s, k)
        out[i, :len(idx)] = idx
    return out


def rank_tfidf(Q, D, qidx, exclude_self, n_docs, k=TOP, chunk=32):
    """cosine over tf-idf. Q is [nq,V] sparse, D is [ndocs,V] sparse."""
    k = min(k, n_docs - (1 if exclude_self else 0))
    out = np.full((len(qidx), max(k, 1)), -1, dtype=np.int64)
    for st in range(0, len(qidx), chunk):
        en = min(st + chunk, len(qidx))
        sims = np.asarray((Q[st:en] @ D.T).todense(), dtype=np.float32)
        for j in range(en - st):
            if exclude_self:
                sims[j, qidx[st + j]] = -np.inf
            idx = topk_from_scores(sims[j], k)
            out[st + j, :len(idx)] = idx
        del sims
    return out


def rows_to_words(df, rows):
    """row ids to word ids. -1 means the ranking ran out before this depth."""
    w = df["wid"].to_numpy()[np.where(rows >= 0, rows, 0)].astype(np.int64)
    w[rows < 0] = -1
    return w


def summarise(ranked_rows, ranked_words, df, qidx, mode):
    """Recall numbers plus how often rank 1 is the query's own row."""
    res = R.evaluate(ranked_words, df, qidx, mode=mode)
    self_at_1 = float((ranked_rows[:, 0] == np.asarray(qidx)).mean()) * 100
    res["pct_self_row_at_1"] = round(self_at_1, 2)
    return res


def main():
    t0 = time.time()
    df, meta = R.load()
    tr, va, te = R.make_word_split(df)
    qidx = R.pick_queries(df, te)
    gold = df["wid"].to_numpy()[qidx]
    n_docs = len(df)

    print("corpus rows=%d words=%d queries=%d  (clean: -%d bad, -%d crossref, -%d dupes)"
          % (n_docs, meta["n_words"], len(qidx), meta.get("rows_dropped_bad_def", 0),
             meta.get("rows_dropped_crossref", 0), meta.get("rows_dropped_dupes", 0)))

    tok = df["definition"].map(R.tokens).tolist()

    # ---- how many queries lose their answer entirely when the self row is removed
    rows_per_word = df.groupby("wid").size().reindex(gold).to_numpy()
    only_row = float((rows_per_word <= 1).mean()) * 100

    methods = {}

    print("[1/4] BM25 indexing ...", flush=True)
    bm = BM25(tok)
    bm25_scores = lambda qi, _b=bm, _t=tok: _b.scores(_t[qi])
    print("      %d terms, %.0fs" % (bm.n_terms, time.time() - t0), flush=True)
    for proto, ex in (("in_dictionary", False), ("leave_one_out", True)):
        rows = rank_built(bm25_scores, qidx, ex, n_docs)
        methods[("bm25", proto)] = (rows, rows_to_words(df, rows))

    print("[2/4] TF-IDF fit ...", flush=True)
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(sublinear_tf=True, dtype=np.float32,
                          token_pattern=r"[a-z']{2,}")
    D = vec.fit_transform(df["definition"].str.lower())
    Q = vec.transform(df.loc[qidx, "definition"].str.lower())
    empty = int((np.asarray((Q != 0).sum(1)).ravel() == 0).sum())
    print("      vocab=%d, queries with no in-vocab term: %d" %
          (len(vec.vocabulary_), empty), flush=True)
    for proto, ex in (("in_dictionary", False), ("leave_one_out", True)):
        rows = rank_tfidf(Q, D, qidx, ex, n_docs)
        methods[("tfidf", proto)] = (rows, rows_to_words(df, rows))

    print("[3/4] floors ...", flush=True)
    # random guess over the word vocabulary, and "always the most frequent words",
    # both fitted on the train split only
    rng = np.random.default_rng(0)
    nq = len(qidx)
    K = 100
    rand_rows = -np.ones((nq, K), dtype=np.int64)
    rand_words = np.stack([rng.integers(0, meta["n_words"], K) for _ in range(nq)])
    methods[("random", "in_dictionary")] = (rand_rows, rand_words)
    # always answer the most frequent words in the training rows. Uses the Count column
    # of the file, which is how often the word appears in the source text.
    train_mask = df["wid"].isin(set(tr.tolist()))
    by_word = df.loc[train_mask].groupby("wid")["count"].sum().sort_values(ascending=False)
    prior = by_word.index.to_numpy()
    if len(prior) == 0:
        prior = np.arange(K)
    reps = np.resize(prior, K)[:K]
    methods[("word_freq", "in_dictionary")] = (-np.ones((nq, K), np.int64),
                                                np.tile(reps, (nq, 1)))

    print("[4/4] scoring ...", flush=True)
    out = {"meta": meta,
           "ceiling_strict_top1": R.ceilings(df, qidx)["ceiling_strict_top1"],
           "pct_queries_unreachable_without_self_row": round(only_row, 2),
           "n_queries": int(nq), "runtime_sec": 0}
    na = df.loc[qidx, "n_accept"]
    out["pct_queries_with_tie"] = round(float((na > 1).mean()) * 100, 1)
    short = df["ntok"] < 3
    out["pct_rows_tied_under_3_tokens"] = round(float((df.loc[short, "n_accept"] > 1).mean()) * 100, 1)
    out["pct_rows_tied_3_tokens_or_more"] = round(float((df.loc[~short, "n_accept"] > 1).mean()) * 100, 1)
    table = []
    for (name, proto), (rows, words) in methods.items():
        strict = summarise(rows, words, df, qidx, "strict")
        tol = summarise(rows, words, df, qidx, "tolerant")
        row = dict(method=name, protocol=proto, **{k: v for k, v in strict.items()},
                   recall_at_1_tolerant=tol["recall@1"],
                   recall_at_10_tolerant=tol["recall@10"],
                   recall_at_100_tolerant=tol["recall@100"])
        table.append(row)
        out.setdefault(name, {})[proto] = {"strict": strict, "tolerant": tol}
        print("  %-8s %-14s R@1=%.3f R@10=%.3f R@100=%.3f  medrank=%s  self@1=%.1f%%"
              % (name, proto, strict["recall@1"], strict["recall@10"],
                 strict["recall@100"], strict["median_rank"],
                 0.0 if name in ("random", "word_freq") else strict["pct_self_row_at_1"]))

    # a small failure file so the error analysis script does not have to rerank
    rows = methods[("bm25", "leave_one_out")][0]
    words = rows_to_words(df, rows)
    fails = []
    for i in range(len(qidx)):
        first = int(np.where(words[i] == gold[i])[0][0]) + 1 if (words[i] == gold[i]).any() else -1
        if first == 1:
            continue
        top = [df.at[int(rows[i, j]), "word"] if rows[i, j] >= 0 else "-" for j in range(min(5, rows.shape[1]))]
        fails.append(dict(query_row=int(qidx[i]), gold=df.at[qidx[i], "word"],
                          definition=df.at[qidx[i], "definition"],
                          gold_rank=first, n_rows_for_gold=int(rows_per_word[i]),
                          n_shared_def=int(df.at[qidx[i], "n_accept"]),
                          top1=top[0], top2=top[1] if len(top) > 1 else "-",
                          top3=top[2] if len(top) > 2 else "-"))
    fl = pd.DataFrame(fails)
    os.makedirs(OUTDIR, exist_ok=True)
    fl.to_csv(os.path.join(OUTDIR, "failures_bm25_leave_one_out.csv"), index=False)
    print("  failures written: %d of %d queries" % (len(fl), len(qidx)))
    out["n_failures"] = int(len(fl))

    out["table"] = table
    out["runtime_sec"] = round(time.time() - t0, 1)
    R.write_json(out, os.path.join(OUTDIR, "baselines.json"))
    write_md(out, meta)
    print("\nwrote results/baselines.md and .json  (%.0fs)" % (time.time() - t0))


def write_md(out, meta):
    order = [("bm25", "in_dictionary"), ("bm25", "leave_one_out"),
             ("tfidf", "in_dictionary"), ("tfidf", "leave_one_out"),
             ("word_freq", "in_dictionary"), ("random", "in_dictionary")]
    tab = {(r["method"], r["protocol"]): r for r in out["table"]}
    L = []
    L.append("# baselines, bm25 and tfidf")
    L.append("")
    L.append("from src/run_baselines.py. cleaned corpus is %s rows over %s headwords, "
             "%d fixed queries from the test split, seed %d. took %ss."
             % (f"{meta['rows']:,}", f"{meta['n_words']:,}", out["n_queries"], R.SEED,
                out["runtime_sec"]))
    L.append("")
    L.append("| method | protocol | R@1 | R@10 | R@100 | medrank | R@1 tol | own row at rank 1 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for key in order:
        r = tab.get(key)
        if not r:
            continue
        selfstr = "-" if r["method"] in ("random", "word_freq") else str(r["pct_self_row_at_1"]) + "%"
        L.append("| %s | %s | %.3f | %.3f | %.3f | %s | %.3f | %s |"
                 % (r["method"], r["protocol"], r["recall@1"], r["recall@10"],
                    r["recall@100"], r["median_rank"], r["recall_at_1_tolerant"], selfstr))
    L.append("")
    L.append("notes")
    L.append("")
    L.append("- in_dictionary cant be quoted on its own. the query string is sitting in the "
             "index, so a lexical method gets it by matching text. that is lookup, not "
             "understanding the definition.")
    L.append("- leave_one_out takes that row out. %.1f%% of these queries have no other row "
             "for their word at all, so that side cant reach 1.0 either and the number is a "
             "floor not a ceiling."
             % out["pct_queries_unreachable_without_self_row"])
    L.append("- tol column counts any word sharing the same definition string. string rule, "
             "not wordnet, so it undercounts. details in notes.md")
    L.append("- strict R@1 cant pass %.3f on this set, because %.1f%% of these queries share the "
             "exact string with another word. the file as a whole is %.2f%% tied, but a tied "
             "string is nearly always a short one: %.1f%% tied among definitions under 3 tokens "
             "against %.1f%% among those of 3 or more. our query rule asks for 3 tokens, so the "
             "corpus figure and the query figure are different denominators and not a "
             "contradiction."
             % (out["ceiling_strict_top1"], out["pct_queries_with_tie"], meta["pct_shared_def"],
                out["pct_rows_tied_under_3_tokens"], out["pct_rows_tied_3_tokens_or_more"]))
    L.append("- word_freq is 0 because it is fitted on train words and every query here is a "
             "held out word. a row level split would have hidden that, which is the main "
             "argument for splitting by word.")
    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, "baselines.md"), "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
