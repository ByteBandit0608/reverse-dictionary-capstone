"""sorts the failures dumped by run_baselines into categories.
run after run_baselines.py, needs results/failures_bm25_leave_one_out.csv
writes results/error_analysis.md
the categories are automatic, the meaning is not, so the last column of the table is
blank and we fill about 20 rows by hand. that part is the error analysis in the brief.
"""
import os
import json
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import rexdict as R
from explore_pos import classify, cue_of

OUTDIR = os.environ.get("REXDICT_RESULTS", "results")
IN = os.path.join(OUTDIR, "failures_bm25_leave_one_out.csv")


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


def main():
    if not os.path.exists(IN):
        raise SystemExit("missing %s - run python src/run_baselines.py first" % IN)
    fl = pd.read_csv(IN).fillna({"top1": "-", "top2": "-", "top3": "-"})
    df, meta = R.load()
    fl["cat"] = fl.apply(category, axis=1)
    fl["cue"] = df.loc[fl["query_row"].tolist(), "definition"].map(cue_of).fillna("none").values
    fl["gold_cls"] = fl["gold"].map(classify)
    fl["top1_cls"] = fl["top1"].map(lambda w: classify(w) if w != "-" else "-")
    fl["wrong_class"] = (fl["cue"] != "none") & (fl["top1_cls"] != "-") & \
                        (fl["cue"] != fl["gold_cls"])

    counts = Counter(fl["cat"])
    n = len(fl)
    total = n
    bj = os.path.join(OUTDIR, "baselines.json")
    if os.path.exists(bj):
        total = int(json.load(open(bj)).get("n_queries", n))
    take = fl.head(20)
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
              "first %d failures, to read by hand" % len(take), ""]
    lines.append("| query | gold | rank 1 | rank 2 | category | cue | gold class | what happened |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for _, r in take.iterrows():
        d = str(r["definition"]).replace("|", "/")
        d = d[:60] + "..." if len(d) > 60 else d
        lines.append("| %s | **%s** | %s | %s | %s | %s | %s |  |" %
                     (d, r["gold"], r["top1"], r["top2"], r["cat"], r["cue"], r["gold_cls"]))
    R.write_json({"n_queries_total": int(n), "by_category": {k: int(v) for k, v in counts.items()},
                  "pct_queries_where_cue_disagrees_with_gold_class":
                      round(float(fl["wrong_class"].mean()) * 100, 2) if n else 0.0},
                 os.path.join(OUTDIR, "error_analysis.json"))
    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, "error_analysis.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    fl.to_csv(os.path.join(OUTDIR, "failures_annotated.csv"), index=False)
    print("\n".join(lines[:14]))
    print("\nwrote results/error_analysis.md, error_analysis.json, failures_annotated.csv")


if __name__ == "__main__":
    main()
