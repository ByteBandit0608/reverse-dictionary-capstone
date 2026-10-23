"""sorts the failures dumped by run_baselines into categories.
run after run_baselines.py, needs results/failures_bm25_leave_one_out.csv
writes results/error_analysis.md
the categories are automatic, the meaning is not, so the last column of the table is
blank and we fill the 20 rows by hand. that part is the error analysis in the brief.
the 20 are picked as 5 from each category, preferring the ones where rank 1 is a word
the definition itself offers, because those are the interesting confusions.
"""
import os
import re
import json
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import rexdict as R
from explore_pos import classify, cue_of, tag_class, would_be_dropped

OUTDIR = os.environ.get("REXDICT_RESULTS", "results")
IN = os.path.join(OUTDIR, "failures_bm25_leave_one_out.csv")

STEM = re.compile(r"^(.+?)(ations?|ings?|ers?|ors?|ists?|ies|ing|ed|es|s)$")
LONG = 24     # definitions at or above the corpus p90, the multi clause ones
RARE_DF = 2   # a term used twice or less in the whole file counts as rare
RARE_FRAC = 0.5


def stem(w):
    m = STEM.match(w)
    return m.group(1) if m and len(m.group(1)) > 3 else w


def words_of(s):
    return [w for w in re.findall(r"[a-z]+", str(s).lower()) if len(w) > 3]


def named_in_query(top1, defn):
    """rank 1 is a word the definition itself lists as a synonym of the answer,
    like Abolition for 'the act of abolishing; abolition; destruction'."""
    if not str(top1) or top1 == "-":
        return False
    d = {stem(w) for w in words_of(defn)}
    return any(stem(w) in d for w in words_of(top1))


def cls_of(w):
    """class from the headword ending. a failure file can carry a missing word, either
    an empty cell read back as NaN or the - placeholder for nothing at that rank, so
    the guard is here and not inside classify."""
    return classify(w) if isinstance(w, str) and w and w != "-" else "-"


def category(row):
    if row["gold_rank"] == -1 and row["n_rows_for_gold"] <= 1:
        return "unreachable"        # the only row for that word was the one we removed
    if row["n_shared_def"] > 1:
        return "tied definition"    # another word has this exact string, so rank 1 is a coin flip
    if row["gold_rank"] == -1:
        return "no lexical overlap"  # nothing in the query text matched the surviving rows
    if row["gold_rank"] <= 10:
        return "right word, deep"    # retrieved but outranked
    return "outranked"               # retrieved, but past rank 10


def phenomenon(row, rare_frac):
    """which phenomenon the miss is an instance of. five labels, each one computed
    from fields already in the row, so the count is reproducible and not a judgement
    call. thresholds are in LONG and RARE_DF."""
    if row["gold_rank"] == -1 and row["n_rows_for_gold"] <= 1:
        return "structural, the protocol deleted the only row"
    if row["n_shared_def"] > 1:
        return "structural, two words share the exact string"
    if row["gold_rank"] == -1:
        if rare_frac >= RARE_FRAC:
            return "rare or technical wording in the query"
        if row["ntok"] >= LONG:
            return "long definition, several ideas at once"
        return "the dictionary words it up differently"
    if row["top1_named_in_definition"]:
        return "returned a synonym the definition itself names"
    if row["gold_rank"] <= 10:
        return "right word found, ranked just below"
    return "right word found, buried past the top 10"


def main():
    if not os.path.exists(IN):
        raise SystemExit("missing %s - run python src/run_baselines.py first" % IN)
    fl = pd.read_csv(IN).fillna({"top1": "-", "top2": "-", "top3": "-"})
    df, meta = R.load()
    fl["cat"] = fl.apply(category, axis=1)
    fl["cue"] = df.loc[fl["query_row"].tolist(), "definition"].map(cue_of).fillna("none").values
    qrows = df.loc[fl["query_row"].tolist()]
    fl["gold_pos"] = qrows["pos_raw"].map(tag_class).values
    fl["gold_cls"] = fl["gold"].map(cls_of)
    fl["top1_cls"] = fl["top1"].map(cls_of)
    fl["hard_filter_would_drop_gold"] = [would_be_dropped(c, p)
                                         for c, p in zip(fl["cue"], fl["gold_pos"])]
    fl["top1_named_in_definition"] = [named_in_query(a, b)
                                      for a, b in zip(fl["top1"], fl["definition"])]
    q = df.loc[fl["query_row"].tolist()]
    fl["ntok"] = q["ntok"].values
    doc_df = Counter(t for toks in df["definition"].map(R.tokens) for t in set(toks))
    rare = []
    for toks in q["definition"].map(R.tokens):
        keep = [t for t in toks if len(t) > 3]
        rare.append(sum(1 for t in keep if doc_df.get(t, 0) <= RARE_DF) / len(keep) if keep else 0.0)
    fl["rare_frac"] = rare
    fl["phenomenon"] = [phenomenon(r, f) for (_, r), f in zip(fl.iterrows(), fl["rare_frac"])]

    counts = Counter(fl["cat"])
    n = len(fl)
    total = n
    bm_loo = None
    bj = os.path.join(OUTDIR, "baselines.json")
    if os.path.exists(bj):
        b = json.load(open(bj))
        total = int(b.get("n_queries", n))
        for r in b.get("table", []):
            if r["method"] == "bm25" and r["protocol"] == "leave_one_out":
                bm_loo = r
    named = float(fl["top1_named_in_definition"].mean()) * 100
    dropped = float(fl["hard_filter_would_drop_gold"].mean()) * 100
    deep = fl[(fl["gold_rank"] > 1) & (fl["gold_rank"] <= 100)]
    med_deep = int(deep["gold_rank"].median()) if len(deep) else 0

    take = pd.concat([g.sort_values(["top1_named_in_definition", "query_row"],
                                    ascending=[False, True]).head(5)
                      for _, g in fl.groupby("cat")])
    lines = ["# failures, bm25 leave_one_out", "",
             "goes with results/baselines.md. %d of %d queries miss at rank 1 here." % (n, total),
             "",
             "categories, from the script:", ""]
    for cat, c in counts.most_common():
        lines.append("- %s: %d" % (cat, c))
    lines += ["",
              "two of these are not the model failing. unreachable means the word had one row "
              "and that row is the query, so removing it deletes the answer. tied definition "
              "means two words share the string, so rank 1 is a coin flip no matter what the "
              "ranker is. both are properties of the file.",
              "",
              "three things the categories do not say on their own",
              ""]
    lines.append("- %.1f%% of these misses have a rank 1 that is a word the definition itself "
                 "names as a synonym. so the model is often not lost, it returned a different "
                 "dictionary word for the same thing and the strict metric calls it wrong. that "
                 "is the same argument as the tolerant column, just counted on the failures."
                 % named)
    tail = "" if bm_loo is None else (" and it is why R@100 of %.3f and R@1 of %.3f are not the "
                                      "same statement." % (bm_loo["recall@100"],
                                                            bm_loo["recall@1"]))
    lines.append("- %d queries had the gold word inside the top 100 but not at rank 1, median "
                 "rank %d. that is the pool a dense model or a reranker has to reorder,%s"
                 % (len(deep), med_deep, tail))
    lines.append("- on the misses the opening cue disagrees with the tag in the file for %.1f%%, "
                 "which is the same rule pos_roles.md uses, computed there over the whole query "
                 "set and here over the failures only." % dropped)
    phen = Counter(fl["phenomenon"])
    lines += ["",
              "phenomena, counted over the whole failure set instead of listed as examples",
              "",
              "| what the model is actually missing | queries | mean tokens | mean rare terms |",
              "|---|---|---|---|"]
    for name, c in phen.most_common():
        g = fl[fl["phenomenon"] == name]
        lines.append("| %s | %d | %.1f | %.0f%% |"
                     % (name, c, float(g["ntok"].mean()), float(g["rare_frac"].mean()) * 100))
    others = [f for f in sorted(os.listdir(OUTDIR))
              if f.startswith("failures_") and f.endswith("_leave_one_out.csv")
              and f != os.path.basename(IN)]
    if others:
        here = set(fl["query_row"].tolist())
        lines += ["", "how much the misses overlap with the other methods, same %d queries"
                  % total, ""]
        for f in others:
            tag = f.replace("failures_", "").replace("_leave_one_out.csv", "")
            try:
                other = set(pd.read_csv(os.path.join(OUTDIR, f))["query_row"].tolist())
            except Exception:
                continue
            both = len(here & other)
            lines.append("- %s: %d of its %d misses are also bm25 misses, %.0f%% of the union of "
                         "the two sets. a high share means both methods go wrong on the same "
                         "queries and a reranker over the two has little to add. a low one means "
                         "they are looking at different things."
                         % (tag, both, len(other), 100.0 * both / max(1, len(here | other))))
    lines += ["",
              "%d failures to read by hand, 5 from each category" % len(take), ""]
    lines.append("| query | gold | rank 1 | rank 2 | category | cue | gold pos | rank of gold | what happened |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for _, r in take.iterrows():
        d = str(r["definition"]).replace("|", "/")
        d = d[:60] + "..." if len(d) > 60 else d
        rk = "not in top 100" if r["gold_rank"] == -1 else str(int(r["gold_rank"]))
        lines.append("| %s | **%s** | %s | %s | %s | %s | %s | %s |  |" %
                     (d, r["gold"], r["top1"], r["top2"], r["cat"], r["cue"], r["gold_pos"], rk))
    R.write_json({"n_queries_total": int(n), "by_category": {k: int(v) for k, v in counts.items()},
                  "pct_misses_where_rank1_is_named_in_the_definition": round(named, 2),
                  "n_gold_in_top100_but_not_first": int(len(deep)),
                  "median_rank_of_those": med_deep,
                  "pct_misses_where_a_hard_pos_filter_would_drop_gold": round(dropped, 2),
                  "by_phenomenon": {k: int(v) for k, v in phen.items()},
                  "mean_ntok_by_phenomenon": {k: round(float(fl[fl["phenomenon"] == k]["ntok"].mean()), 1)
                                              for k in phen}},
                 os.path.join(OUTDIR, "error_analysis.json"))
    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, "error_analysis.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    fl.to_csv(os.path.join(OUTDIR, "failures_annotated.csv"), index=False)
    print("\n".join(lines[3:20]))
    print("\nwrote results/error_analysis.md, error_analysis.json, failures_annotated.csv")


if __name__ == "__main__":
    main()
