"""checks on the metric code. python tests/test_metrics.py
no csv needed, it builds a tiny frame by hand so the right answer is obvious.
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import rexdict as R

# four words, five rows. "big" and "large" share one definition string.
ROWS = [
    ("big", "Of great size"),
    ("large", "Of great size"),
    ("tiny", "Of very small size"),
    ("run", "To move quickly on foot"),
    ("run", "A sequence of steps"),
]
df = pd.DataFrame(ROWS, columns=["word", "definition"])
df["pos_raw"] = ""
df["count"] = 1.0

words = np.array(sorted(df["word"].unique()))
wid = pd.Series(np.arange(len(words)), index=words)
df["wid"] = df["word"].map(wid).astype(np.int32)
dnorm = df["definition"].str.lower()
grp = dnorm.map(df.groupby(dnorm)["wid"].agg(lambda s: tuple(sorted(set(s)))))
df["accept"] = grp
df["n_accept"] = df["accept"].map(len)
df["ntok"] = dnorm.map(lambda s: len(R.tokens(s)))


def check(name, got, want):
    # evaluate() rounds to 4 decimals, so compare at that precision
    ok = abs(got - round(want, 4)) < 1e-9 if isinstance(want, float) else got == want
    print("%-42s %-10s %s" % (name, "ok" if ok else "FAIL", got))
    assert ok, "%s: got %r want %r" % (name, got, want)


# 1. gold word first for every query
qidx = np.array([0, 2, 3])
gold = df.loc[qidx, "wid"].to_numpy()
check("setup: 3 queries, ids distinct", len(set(gold.tolist())), 3)
ranked = gold.reshape(-1, 1)
res = R.evaluate(ranked, df, qidx, mode="strict")
check("perfect ranking, recall@1", res["recall@1"], 1.0)
check("perfect ranking, median rank", res["median_rank"], 1)
check("perfect ranking, nothing missing", res["pct_not_in_top1"], 0.0)

# 2. gold at position 5 -> outside recall@1, inside recall@10
ranked = np.full((3, 10), -1, dtype=np.int64)
ranked[:, 4] = gold
res = R.evaluate(ranked, df, qidx, mode="strict")
check("rank 5, recall@1", res["recall@1"], 0.0)
check("rank 5, recall@10", res["recall@10"], 1.0)
check("rank 5, median rank", res["median_rank"], 5)

# 3. gold never returned -> capped median rank, counted as missing
ranked = np.full((3, 10), 99, dtype=np.int64)
res = R.evaluate(ranked, df, qidx, mode="strict")
check("miss, recall@100", res["recall@100"], 0.0)
check("miss, median rank capped", res["median_rank"], R.MAX_RANK)
check("miss, pct not in top", res["pct_not_in_top10"], 100.0)

# 4. tolerant scoring: a synonym with the same definition string counts as a hit.
#    Rows 0 and 1 are big and large, which share one definition.
big = np.array([[wid["big"]], [wid["big"]]])
qpair = np.array([0, 1])
res_s = R.evaluate(big, df, qpair, mode="strict")
res_t = R.evaluate(big, df, qpair, mode="tolerant")
check("one answer, two tied rows, strict@1", res_s["recall@1"], 0.5)
check("one answer, two tied rows, tolerant@1", res_t["recall@1"], 1.0)

# 5. padding of -1 must never count as a hit
ranked = np.tile([-1, -1, wid["tiny"]], (3, 1))
res = R.evaluate(ranked, df, np.array([0, 1, 2]), mode="strict")
check("padded ranks do not create hits", res["recall@1"], 0.0)
check("padded ranks: tiny found at 3", res["recall@10"], 1 / 3)

# 6. split by word keeps senses together
tr, va, te = R.make_word_split(df, ratios=(0.8, 0.1, 0.1), seed=R.SEED)
union = set(tr) | set(va) | set(te)
check("split covers every word exactly once",
      (len(tr) + len(va) + len(te), len(union)), (df["wid"].nunique(), df["wid"].nunique()))
rows_tr = set(df.loc[df["wid"].isin(set(tr.tolist())), "wid"])
check("no test word appears in the train rows", len(rows_tr & (set(va) | set(te))), 0)
run_id = int(wid["run"])
check("both senses of run land in the same split",
      run_id in set(tr) or run_id in set(va) or run_id in set(te), True)
q = R.pick_queries(df, np.array([int(wid["tiny"]), run_id]), min_toks=3)
check("queries come only from the given words",
      set(df.loc[list(q), "wid"].tolist()) <= {int(wid["tiny"]), run_id}, True)

# 7. ceilings
c = R.ceilings(df, np.array([0, 1, 2]))
check("ceiling with one tied pair", round(c["ceiling_strict_top1"], 4), round((0.5 + 0.5 + 1.0) / 3, 4))

print("\nall metric checks passed")
