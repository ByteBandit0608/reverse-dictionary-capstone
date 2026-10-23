"""by-word split + the fixed 500 queries, written to splits/
python src/make_splits.py
writes train_words.txt val_words.txt test_words.txt test_query_ids.txt manifest.json
the id file is the point: it is what makes this week's baseline and the week 6 model
comparable instead of two different numbers that happen to look alike.
"""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rexdict as R

OUT = os.environ.get("REXDICT_SPLITS", "splits")


def main():
    df, meta = R.load()
    tr, va, te = R.make_word_split(df)
    qidx = R.pick_queries(df, te)
    if len(qidx) < 25:
        raise SystemExit("only %d rows survive as queries. the cleaning rules are too "
                         "aggressive or the corpus is tiny, check rows_dropped_self_ref "
                         "before trusting anything downstream" % len(qidx))

    words = sorted(df["word"].unique())
    os.makedirs(OUT, exist_ok=True)

    def dump(name, ids):
        with open(os.path.join(OUT, name), "w") as f:
            f.write("\n".join(words[i] for i in sorted(ids.tolist())) + "\n")

    dump("train_words.txt", tr)
    dump("val_words.txt", va)
    dump("test_words.txt", te)
    pd.Series(qidx).to_csv(os.path.join(OUT, "test_query_ids.txt"),
                           index=False, header=False)

    def nrows(ids):
        return int(df["wid"].isin(set(ids.tolist())).sum())

    man = dict(meta)
    man.update(seed=R.SEED,
               ratios="80/10/10, split by word",
               n_words_train=len(tr), n_words_val=len(va), n_words_test=len(te),
               rows_train=nrows(tr), rows_val=nrows(va), rows_test=nrows(te),
               n_queries=len(qidx),
               min_query_tokens=R.MIN_QUERY_TOKENS,
               min_senses_per_query_word=R.MIN_QUERY_SENSES,
               max_definition_tie=meta["max_tie"])
    man.update(R.ceilings(df, qidx))
    R.write_json(man, os.path.join(OUT, "manifest.json"))

    print("wrote", OUT)
    for k in ("rows_raw", "rows", "rows_dropped_bad_def", "rows_dropped_named",
              "rows_named_in_word", "rows_named_in_definition", "rows_named_word_exact",
              "rows_dropped_empty_word",
              "rows_dropped_crossref", "rows_dropped_dupes", "rows_dropped_self_ref",
              "pct_rows_self_ref", "pct_rows_self_ref_exact", "n_words", "n_words_train", "n_words_val",
              "n_words_test", "rows_train", "rows_val", "rows_test",
              "n_queries", "pct_multi_sense", "pct_shared_def", "max_tie",
              "ceiling_strict_top1", "pct_reachable_when_self_removed"):
        print("%-32s %s" % (k, man.get(k)))

    shared = df.loc[qidx, "n_accept"].gt(1).mean() * 100
    print("\n%.1f%% of the chosen queries have at least one other word with the same "
          "definition string." % shared)
    print("Strict Recall@1 cannot pass %.3f on this set without breaking those ties."
          % man["ceiling_strict_top1"])


if __name__ == "__main__":
    main()
