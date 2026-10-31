"""Loader + metrics for the OPTED reverse dictionary.
The other scripts import this so cleaning and scoring live in one place.
Cleaning is by pattern, not by length (a length cut would remove 17.8% of the file).
Split is by word, so senses of one word never end up on two sides.
The index is the full dictionary, we hold out training pairs and not candidate words.
"""
import os
import re
import glob
import json
from functools import lru_cache
import numpy as np
import pandas as pd

SEED = 2026
N_QUERIES = 500
MIN_QUERY_TOKENS = 3
MIN_QUERY_SENSES = 2   # a query word must keep another row, or leave_one_out deletes its answer
TOPK = (1, 10, 100)
MAX_RANK = 1000  # ranks worse than this are capped, following the Sõnajaht paper

CROSSREF = re.compile(r"^(see|same as|same for)\b", re.I)
BAD_DEF = re.compile(r"^nan$|^$", re.I)
# the #NAME? damage in this file is in the Word and Count columns, not in the
# definition. it is an Excel error value left behind when the source was resaved,
# 207 rows whose headword is literally "#NAME?" and whose definition is a fragment
# like "of Baptistry". Count is numeric by the time we get here so the check runs
# on the headword, and on the definition text in case the same damage lands there.
NAMED = re.compile(r"#NAME", re.I)
# cross-reference stubs. a row that only sends you to another entry has nothing to match
# against, so it is useless as a query. the anchored CROSSREF above catches the ones that
# start with "See", and these three are the ones it does not: the headword can be
# repeated in front of the pointer, and the inflection pointers write the pointer as a
# dotted abbreviation followed by "of". the rule runs on the query side only, the index
# keeps every row, so the vocabulary counts and the earlier results stay comparable.
ECHO = re.compile(r"""^['"]?([A-Za-z][A-Za-z' -]{1,30}?)['"]?[.,;:]\s+(.*)$""", re.S)
POINTER = re.compile(r"^(see|same as|same for|compare|compare with|altered form of|"
                     r"alternate form of|alternative form of)\b", re.I)
# an inflection or alternate-spelling pointer written with dotted abbreviations in front:
# "imp. of Sit.", "Alt. of Hough", "a. & n. from Smoke.". the dots are what separate these
# from a real definition that happens to contain "of", so that is what the pattern keys on.
OF_POINTER = re.compile(r"^(?:[a-z]{1,6}\.\s*(?:&\s*)?){1,3}(?:of|from)\s+[^.]{1,40}\.?\s*$",
                        re.I)
# the same pointer words without the anchor, for the case where a fragment sits in front
POINTER_FREE = re.compile(r"\b(see also|see|same as|same for|compare with)\b", re.I)


def without_echo(word, defn):
    """'Rime. See Rhyme.' -> 'See Rhyme.'. only strips it when the front piece really is
    the headword, so an ordinary definition that starts with a short word is untouched."""
    d = str(defn).strip()
    w = re.sub(r"[^a-z]", "", str(word).lower())
    if not w:
        return d
    m = ECHO.match(d)
    if m and re.sub(r"[^a-z]", "", m.group(1).lower()) == w:
        return m.group(2).strip()
    return d


STOP = {"a", "an", "the", "of", "to", "in", "on", "for", "and", "or", "this", "that",
        "same", "as", "at", "with", "be", "is", "it", "its", "see", "also", "use"}


def is_pointer_stub(word, defn):
    """true when the definition carries no description of its own, only a direction to
    look somewhere else. three shapes show up in this file: a bare pointer at the start
    ("see Plummet"), an inflection pointer written as a dotted abbreviation
    ("imp. of Sit."), and a pointer with a fragment in front of it ("Rhyme. See Rhyme."
    and "The capuchin. See Capuchin  3 (a)."). the third one is the careful case, so the
    piece in front is only allowed one content word, otherwise a real definition that
    happens to end with "see also" gets taken for a stub."""
    d = without_echo(word, defn)
    if POINTER.match(d) or OF_POINTER.match(d):
        return True
    m = POINTER_FREE.search(d)
    if not m:
        return False
    pre = [t for t in tokens(d[:m.start()]) if t not in STOP]
    post = tokens(re.sub(r"\(.*?\)|\d+", " ", d[m.start():]))
    return len(pre) <= 1 and len(post) <= 6

# trailing inflections we strip off the headword before matching it inside its own
# definition. "Abolishment" has to catch "abolishing", otherwise the leak is
# counted as clean when it is only spelled differently.
INFLECTION = re.compile(r"(ments?|ments|es|ed|ing|s)$")
TAIL = "(?:s|es|ed|ing|ment|ments)?"


def self_ref_pattern(word):
    w = re.sub(r"[^a-z]", "", str(word).lower())
    if len(w) < 3:
        return None
    core = INFLECTION.sub("", w)
    if len(core) < 3:
        core = w
    return re.compile(r"\b%s%s\b" % (re.escape(core), TAIL))


@lru_cache(maxsize=200000)
def _pattern(word):
    return self_ref_pattern(word)


def contains_headword(word, defn):
    p = _pattern(word)
    return p is not None and p.search(str(defn).lower()) is not None


def exact_headword(word, defn):
    w = re.sub(r"[^a-z]", "", str(word).lower())
    return len(w) >= 3 and re.search(r"\b%s\b" % re.escape(w), str(defn).lower()) is not None


def find_csv():
    """find the csv. kaggle puts it under /kaggle/input, locally under data/"""
    env = os.environ.get("REXDICT_CSV")
    cands = ([env] if env else []) + ["data/OPTED-Dictionary.csv"]
    cands += sorted(glob.glob("data/*.csv"))
    cands += sorted(glob.glob("/kaggle/input/**/*.csv", recursive=True))
    for p in cands:
        if p and os.path.isfile(p):
            return p
    raise SystemExit("No CSV found. Set REXDICT_CSV=/path/to/OPTED-Dictionary.csv")


def tokens(s):
    return [t for t in re.split(r"[^a-z']+", str(s).lower()) if t]


def load(path=None, clean=True):
    """clean the corpus, return (frame, meta).
    frame columns: word, definition, wid, ntok, pos_raw, accept, n_accept.
    """
    path = path or find_csv()
    raw = pd.read_csv(path, dtype=str).fillna("")
    n_raw = len(raw)

    cols = list(raw.columns)
    lc = {c.lower(): c for c in cols}
    wcol = lc.get("word", cols[0])
    dcol = next((lc[k] for k in lc if "defin" in k), cols[-1])
    pcol = lc.get("pos")
    df = raw.rename(columns={wcol: "word", dcol: "definition"})
    for c in ("word", "definition"):
        df[c] = df[c].astype(str).str.strip().str.strip('"').str.strip()
    if pcol:
        df["pos_raw"] = df[pcol].astype(str).str.strip().str.strip('"')
    else:
        df["pos_raw"] = ""
    ccol = lc.get("count")
    if ccol:
        df["count"] = pd.to_numeric(df[ccol], errors="coerce").fillna(0)
    else:
        df["count"] = 0.0

    meta = {"csv": path, "rows_raw": n_raw}
    # how many rows in the file are a bare pointer to another entry. CROSSREF above only
    # looks at the start of the definition, so this counts a wider pattern than the one
    # that is applied. the gap between the two numbers is what the anchored rule misses.
    stub_all = np.array([is_pointer_stub(a, b) for a, b in zip(df["word"], df["definition"])])
    meta["rows_pointer_stub"] = int(stub_all.sum())
    if clean:
        self_hit = np.array([contains_headword(a, b)
                             for a, b in zip(df["word"], df["definition"])])
        exact_hit = np.array([exact_headword(a, b)
                              for a, b in zip(df["word"], df["definition"])])
        meta["pct_rows_self_ref"] = round(float(self_hit.mean()) * 100, 2)
        meta["pct_rows_self_ref_exact"] = round(float(exact_hit.mean()) * 100, 2)
        in_word = df["word"].map(lambda s: bool(NAMED.search(s)))
        in_def = df["definition"].map(lambda s: bool(NAMED.search(s)))
        named = in_word | in_def
        # a row with no headword can never be answered correctly, it is only noise in
        # the index. the csv writes these as an empty quoted cell.
        empty_word = df["word"].str.len() == 0
        meta["rows_named_in_word"] = int(in_word.sum())
        meta["rows_named_in_definition"] = int(in_def.sum())
        meta["rows_named_word_exact"] = int((df["word"] == "#NAME?").sum())
        meta["rows_dropped_named"] = int(named.sum())
        meta["rows_dropped_empty_word"] = int(empty_word.sum())
        bad_def = df["definition"].map(lambda s: bool(BAD_DEF.match(s))) | \
                  (df["definition"].str.len() < 2) | named | empty_word
        n_bad = int(bad_def.sum())
        df = df[~bad_def].copy()
        # "see XYZ" style entries. Pattern based. A length cut would also remove
        # short real definitions, which are 17.8% of the corpus, so we do not do that.
        cross = df["definition"].map(lambda s: bool(CROSSREF.match(s)))
        n_cross = int(cross.sum())
        df = df[~cross]
        dupes = df.duplicated(["word", "definition"])
        n_dupes = int(dupes.sum())
        df = df[~dupes]
        # rows where the definition uses the word it is defining. label leakage, a
        # model can score them by string copy alone, so they come out of the corpus
        # and not only out of the query set.
        keep = ~np.array([contains_headword(a, b)
                          for a, b in zip(df["word"], df["definition"])])
        n_self = int((~keep).sum())
        df = df[keep].copy()
        # pointers still sitting in the index. CROSSREF only matches at the start of a
        # definition, so this is the part of the file that the anchored rule leaves in.
        # they stay, an index entry does no harm and the query rule above is what matters,
        # but the number is reported so the difference is visible instead of assumed.
        n_stub = int(sum(is_pointer_stub(a, b)
                         for a, b in zip(df["word"], df["definition"])))
        meta.update(rows_dropped_bad_def=n_bad,
                    rows_dropped_crossref=n_cross,
                    rows_dropped_dupes=n_dupes,
                    rows_dropped_self_ref=n_self,
                    rows_pointer_stub_in_index=n_stub)

    df = df.reset_index(drop=True)
    words = np.array(sorted(df["word"].unique()))
    wid = pd.Series(np.arange(len(words)), index=words)
    df["wid"] = df["word"].map(wid).astype(np.int32)

    dnorm = df["definition"].str.lower().str.strip()
    df["ntok"] = dnorm.map(lambda s: len(tokens(s)))

    # words that share this exact definition string. Used for the tolerant metric
    # and for the tie ceiling. String based, not WordNet based (see notes.md).
    grp = df.groupby(dnorm)["wid"].agg(lambda s: tuple(sorted(set(s))))
    df["accept"] = dnorm.map(grp)
    df["n_accept"] = df["accept"].map(len)

    nmulti = int((df.groupby("wid").size() > 1).sum())
    meta.update(
        rows=len(df),
        n_words=int(len(words)),
        n_def_strings=int(dnorm.nunique()),
        rows_per_word=round(len(df) / len(words), 3),
        n_words_multi_sense=nmulti,
        pct_multi_sense=round(100.0 * nmulti / len(words), 2),
        pct_shared_def=round(float((df["n_accept"] > 1).mean()) * 100, 2),
        max_tie=int(df["n_accept"].max()),
        mean_ntok=round(float(df["ntok"].mean()), 2),
        median_ntok=int(df["ntok"].median()),
        pct_pos_filled=round(float((df["pos_raw"] != "").mean()) * 100, 2),
    )
    return df, meta


def make_word_split(df, ratios=(0.8, 0.1, 0.1), seed=SEED):
    """split the headwords. three arrays of word ids: train, val, test."""
    w = np.array(sorted(df["wid"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(w)
    n = len(w)
    a = int(round(n * ratios[0]))
    b = a + int(round(n * ratios[1]))
    return w[:a], w[a:b], w[b:]


def query_pool(df, test_wids, min_toks=MIN_QUERY_TOKENS, min_senses=MIN_QUERY_SENSES):
    """every row that is allowed to be a query, as row indices. pick_queries draws from
    this. the rules here are unchanged from the run that produced the committed 500, so
    that draw reproduces exactly. pointer stubs are handled after the draw and not here,
    because shrinking the pool would re-roll all 500 instead of the few bad slots."""
    cand = df[df["wid"].isin(set(test_wids.tolist())) & (df["ntok"] >= min_toks)]
    if min_senses > 1:
        # counted over the whole cleaned corpus, not over the candidate rows. the
        # sibling sense only has to be in the index, it does not have to be long
        sizes = df.groupby("wid").size()
        cand = cand[cand["wid"].isin(set(sizes[sizes >= min_senses].index))]
    return cand.index.to_numpy()


def is_stub(df, row):
    """the query side test. word and definition come from the frame so every caller
    uses the same judgement as the counts in the manifest."""
    return is_pointer_stub(df.at[int(row), "word"], df.at[int(row), "definition"])


def pick_queries(df, test_wids, n=N_QUERIES, min_toks=MIN_QUERY_TOKENS, seed=SEED,
                 min_senses=MIN_QUERY_SENSES, force_draw=False):
    """the fixed 500 queries as row indices into df, sorted, so they can be
    committed as a text file and everyone scores on the same set.
    min_senses keeps only words with at least two rows. under leave_one_out a
    single sense word loses its only row, so the answer is gone and no model can be
    scored on it at all. that made 42.6% of the earlier query set dead weight.

    only the three rows named in the task are replaced. the rest of the original draw is
    left alone, even if the wider detector recognizes another pointer-shaped entry. the
    replacement rows come from the same candidate pool, using the next draw from the
    same seeded random stream and skipping other detected stubs. once the query ID file
    exists, normal model runs use it instead of drawing again. force_draw is for the split
    script to check that the frozen file still matches this version of the corpus."""
    split_dir = os.environ.get("REXDICT_SPLITS", "splits")
    frozen_path = os.path.join(split_dir, "test_query_ids.txt")
    fixed_rules = (len(df) > N_QUERIES and n == N_QUERIES and
                   min_toks == MIN_QUERY_TOKENS and min_senses == MIN_QUERY_SENSES and
                   seed == SEED)
    if fixed_rules and not force_draw and os.path.isfile(frozen_path):
        try:
            frozen = np.array([int(x) for x in open(frozen_path).read().split()], dtype=np.int64)
        except ValueError as e:
            raise SystemExit("bad query ID in %s: %s" % (frozen_path, e))
        if len(frozen) != n or len(np.unique(frozen)) != n:
            raise SystemExit("%s must contain exactly %d distinct row IDs" % (frozen_path, n))
        pool = set(query_pool(df, test_wids, min_toks, min_senses).tolist())
        if not set(frozen.tolist()).issubset(pool):
            raise SystemExit("frozen query IDs do not match the current query pool; "
                             "do not redraw the 500 silently")
        named_bad = {"sate", "rime", "weeper"}
        if any(str(df.at[int(i), "word"]).strip().lower() in named_bad and is_stub(df, i)
               for i in frozen):
            raise SystemExit("frozen file still contains one of the three named bad queries")
        print("using frozen query IDs from %s" % frozen_path)
        return np.sort(frozen)

    idx = query_pool(df, test_wids, min_toks, min_senses)
    if len(idx) <= n:
        return idx
    rng = np.random.default_rng(seed)
    drawn = rng.choice(idx, size=n, replace=False)
    named_bad = {"sate", "rime", "weeper"}
    bad = [int(i) for i in drawn
           if str(df.at[int(i), "word"]).strip().lower() in named_bad and is_stub(df, i)]
    if not bad:
        return np.sort(drawn)
    used = set(int(i) for i in drawn)
    pool = np.array([int(i) for i in idx
                     if int(i) not in used and not is_stub(df, i)], dtype=np.int64)
    extra = rng.choice(pool, size=len(bad), replace=False)
    names = ", ".join(str(df.at[i, "word"]) for i in bad)
    print("query set: %d named bad rows of %d (%s), swapped for %d rows from the same "
          "seeded pool, %d queries unchanged"
          % (len(bad), n, names, len(extra), n - len(bad)))
    return np.sort(np.array(sorted((used - set(bad)) | set(int(i) for i in extra))))




def evaluate(ranked, df, qidx, mode="strict"):
    """ranked: int array [n_queries, K] of word ids, best first.

    mode="strict" counts only the gold word.
    mode="tolerant" counts any word that shares the gold definition string.
    Returns recall@1/10/100, median rank (capped at MAX_RANK), % never found.
    """
    gold = df.loc[qidx, "wid"].to_numpy()
    accs = df.loc[qidx, "accept"].to_numpy()
    n = len(gold)
    first = np.zeros(n, dtype=np.int32)  # 0 means not found inside the ranked list
    for qi in range(n):
        g = gold[qi]
        accset = None if mode == "strict" else set(accs[qi])
        for pos, w in enumerate(ranked[qi]):
            if w == g or (accset is not None and w in accset):
                first[qi] = pos + 1
                break
    out = {}
    big = np.iinfo(np.int32).max
    ranks = np.where(first > 0, first, big)
    for k in TOPK:
        out[f"recall@{k}"] = round(float((ranks <= k).mean()), 4)
    capped = np.minimum(ranks, MAX_RANK)
    out["median_rank"] = int(np.median(capped))
    out["pct_not_in_top{0}".format(ranked.shape[1])] = round(float((first == 0).mean()) * 100, 2)
    return out


def ceilings(df, qidx):
    """two numbers that bound what any system can score on these queries.
    strict_top1: if a definition is shared by k words and you rank perfectly, the best
      you can do at 1 is 1/k, so the expected ceiling is mean(1/k).
    leave_one_out: if the query row is removed and the gold word has no other row,
      that word is unreachable and the query cannot be solved at all.
    """
    gold = df.loc[qidx, "wid"].to_numpy()
    n_acc = df.loc[qidx, "n_accept"].to_numpy()
    rows_per_word = df.groupby("wid").size().reindex(gold).to_numpy()
    strict = float((1.0 / n_acc).mean())
    reach = float((rows_per_word > 1).mean())
    return {"ceiling_strict_top1": round(strict, 4),
            "pct_reachable_when_self_removed": round(reach * 100, 2)}


def write_json(obj, path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True, default=str)
