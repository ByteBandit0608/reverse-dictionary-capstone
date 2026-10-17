"""what pos / role signal can we get out of this file. python src/explore_pos.py
writes results/pos_roles.md

the POS column in the csv has 1913 abbreviations in it (n., v. t., prep., a. and so on),
97.9% of rows are filled, so when it is readable we take the tag from the file instead of
guessing it. the suffix rule stays as the fallback and as the morphology channel, which is
the part that actually separates astronomer from astronomy since both of those are nouns.

one number is why this exists: how many gold answers a hard POS filter would throw away. if
it is not small, pos stays a feature and a rerank bonus and never a filter.

on roles: Ekansh said on 3 Oct he means agent/patient, karak, "not the subject-object thing",
so no nsubj/dobj here. a dictionary states the role in the first few words of the definition
("a person who", "that which is used by"), that is where we read it.
"""
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import rexdict as R

OUTDIR = os.environ.get("REXDICT_RESULTS", "results")

# classes from the ending of a headword. order matters, first hit wins.
SUFFIX = [
    ("place",   r"(arium|ery|ory|ium)$"),
    ("agent",   r"(ers|er|ors|or|ists|ist|ians|ian|eers|eer|ees|ee)$"),
    ("noun_thing", r"(tion|sion|ment|ness|ity|ance|ence|ship|dom|hood|acy|ure|ism)$"),
    ("adj",     r"(ful|ous|ive|able|ible|less|ish|ary|ant|ent|ic|al)$"),
    ("verb",    r"(ize|ise|ify|ate)$"),
]

# the tags in this file are Webster abbreviations: n. v. i. v. t. a. adv. prep. conj.
# interj. pron. s. imp. & p. pr. so we take the first token and look it up, which avoids
# the trap where "adv." and "a." both start with a.
TAG_MAP = {
    "n": "noun", "s": "noun", "v": "verb", "a": "adj", "adj": "adj", "al": "adj",
    "adv": "adv", "prep": "prep", "conj": "conj", "interj": "interj", "pron": "pron",
    "imp": "partic", "p": "partic", "ind": "pron", "def": "pron", "pref": "other",
    "suffix": "other", "v.": "verb",
}

AGENTIVE = re.compile(r"(ers?|ors?|ists?|ians?|eers?|ees?|ants?|ents?)$", re.I)

# what the opening of a definition says the answer should be, in real POS terms
EXPECT = {"agent": "noun", "tool": "noun", "place": "noun", "noun_thing": "noun",
          "adj": "adj", "verb": "verb"}

# what the opening of a definition says the answer should be, in suffix class terms.
# only used on the fallback path where there is no POS column to read.
CUE_OK = {
    "agent": {"agent", "other"},
    "tool": {"agent", "other", "noun_thing"},
    "place": {"place", "noun_thing"},
    "noun_thing": {"noun_thing", "other"},
    "adj": {"adj", "other"},
    "verb": {"verb", "other"},
}

# cues only count at the start of the definition. that is where a dictionary states what
# kind of thing the word is, and matching further in picks up phrases like "in the manner
# of one who", which say nothing about the headword.
CUES = [
    ("agent",      r"^(a |the |one |any )?(person|persons|one|party|individual|being|man|woman|child|boy|girl)s?\s+(who|that)\b"
                   r"|^(one|a person|the person)\s+who\b"
                   r"|^(a |the )?(writer|maker|breaker|keeper|student|lover)\s+of\b"),
    ("tool",       r"^(a |the )?(thing|instrument|device|apparatus|machine|utensil|tool|implement)\b"
                   r"|^(that|something)\s+(which|by means of which)\b"),
    ("place",      r"^(a |the )?(place|spot|part|region|country|town|city|building|room)\b"
                   r"|^the place where\b"),
    ("noun_thing", r"^(a |the )?(act|action|process|state|quality|condition|result|art|science|study|practice|feeling|emotion|degree|manner)\b"),
    ("adj",        r"^(characterized by|having|of the nature of|of the quality of|tending to|inclined to|marked by|being|like)\b"
                   r"|^of the\b"),
    ("verb",       r"^to\s+[a-z]+"),
]

# Role of the thing being defined, read off the opening of the definition. Order matters.
ROLES = [
    ("agent",      r"^(a |the |one |any )?(person|persons|one|people|individual|being)s?\s+(who|that)\b"
                   r"|^one who\b|^that which does\b"),
    ("patient",    r"^(a |the )?(one|person|thing|object|that which)\s+(who|that|which)\s+is\s+[a-z]+ed\b"
                   r"|^(a |the )?(one|person)\s+to\s+be\b|^that which is (given|made|done|used|found)\b"),
    ("instrument", r"^(a |the )?(instrument|tool|device|implement|machine|utensil|appliance|thing)\b"
                   r"|by means of which\b|^that which is used\b"),
    ("location",   r"^(a |the )?(place|spot|part|region|building|room|town|country)\b"
                   r"|^the place where\b|\bwhere .* is (kept|found|made|done)\b"),
    ("quality",    r"^(having|of the nature of|of the quality of|characterized by|marked by|like|pertaining to)\b"),
    ("action",     r"^(a |the )?(act|action|process|manner|practice)\b|^to\s+[a-z]+"),
]


def classify(word):
    w = word.lower()
    for name, pat in SUFFIX:
        if re.search(pat, w):
            return name
    return "other"


def tag_class(raw):
    t = str(raw).lower().strip().strip('"').strip()
    if t in ("", "nan", "none"):
        return "missing"
    first = re.split(r"[.\s&]+", t)[0]
    return TAG_MAP.get(first, "other")


def is_agentive(word):
    return bool(AGENTIVE.search(str(word).strip()))


def role_of(defn):
    d = str(defn).lower().strip()
    for name, pat in ROLES:
        if re.search(pat, d):
            return name
    return "none"


def compatible(cue, cls):
    return cls == cue or cls in CUE_OK.get(cue, set())


def would_be_dropped(cue, cls):
    """Would a hard filter on the cue's class hide the gold answer?
    "other" and "missing" are unknown, not contradictory, so they are not counted."""
    if cls in ("other", "missing"):
        return False
    if cls not in EXPECT.values():
        return False
    return cls != EXPECT.get(cue, cls)


def spacy_tags(words):
    """Return {word: POS} or None if spaCy is not usable here."""
    try:
        import spacy
    except ImportError:
        return None
    try:
        nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "lemmatizer"])
    except Exception:
        print("spaCy is installed but en_core_web_sm is not. "
              "Fix: python -m spacy download en_core_web_sm")
        return None
    out = {}
    for i, doc in enumerate(nlp.pipe(list(words), batch_size=4096)):
        t = next(iter(doc), None)
        out[words[i]] = t.pos_ if t is not None else "X"
    return out


def cue_of(defn):
    d = str(defn).lower().strip()
    for name, pat in CUES:
        if re.search(pat, d):
            return name
    return None


def main():
    df, meta = R.load()
    tr, va, te = R.make_word_split(df)

    # ---- 0. where the pos comes from. the column is mostly filled in this file, so read
    #         it and say so in the output. fall back to endings only when it is not usable.
    raw_tags = df.loc[df["pos_raw"] != "", "pos_raw"].value_counts()
    top_tags = {str(k): int(v) for k, v in raw_tags.head(15).items()}
    use_col = meta.get("pct_pos_filled", 0) > 50 and len(raw_tags) > 0
    df["pos_cls"] = df["pos_raw"].map(tag_class) if use_col else df["word"].map(classify)
    src = "csv column" if use_col else "headword suffix"

    # senses of one word do not have to share a pos. "run" as a verb and as a noun is the
    # normal case in this dictionary, and it matters, a per word feature is not per sense
    per_word = df.groupby("wid")["pos_cls"].nunique()
    pct_word_multi_pos = float((per_word > 1).mean()) * 100

    qidx = R.pick_queries(df, te)
    q = df.loc[qidx].copy()
    q["cue"] = q["definition"].map(cue_of)
    q["role"] = q["definition"].map(role_of)
    q["gold_cls"] = q["pos_cls"]
    q["suffix_cls"] = q["word"].map(classify)

    # ---- 1. how much of the vocabulary the suffix rule can name at all
    one_row_per_word = df.drop_duplicates("wid").reset_index(drop=True)
    one_row_per_word["cls"] = one_row_per_word["word"].map(classify)
    dist = one_row_per_word["cls"].value_counts()
    cov = float((one_row_per_word["cls"] != "other").mean()) * 100
    pct_nouns_agentive = float(
        one_row_per_word.loc[one_row_per_word["pos_cls"] == "noun", "word"].map(is_agentive)
        .mean()) * 100

    # ---- 2. spaCy on the headwords, so we know if the tagger is worth trusting here
    sp = None
    agree = None
    try:
        import spacy  # noqa: F401
        have_spacy = True
    except ImportError:
        have_spacy = False
    if have_spacy:
        sample = one_row_per_word["word"].sample(min(20000, len(one_row_per_word)),
                                                 random_state=R.SEED).tolist()
        sp = spacy_tags(sample)
        if sp is not None:
            tmp = pd.DataFrame({"word": sample, "pos": [sp[w] for w in sample]})
            mapx = {"NOUN": "noun", "PROPN": "noun", "ADJ": "adj", "VERB": "verb"}
            tmp["pos_cls"] = tmp["pos"].map(lambda p: mapx.get(p, "other"))
            mine = df.drop_duplicates("wid").set_index("word")["pos_cls"]
            tmp["ours"] = tmp["word"].map(mine).fillna("missing")
            agree = float((tmp["pos_cls"] == tmp["ours"]).mean()) * 100

    # ---- 3. does the opening of the definition tell us the pos of the answer, and what
    #         would a hard filter cost. this is the number that decides the design.
    cued = q[q["cue"].notna()]
    cued_share = float(len(cued) / len(q)) * 100
    if len(cued):
        exp = cued["cue"].map(EXPECT)
        exact = float((cued["gold_cls"] == exp).mean()) * 100
        lost = float(cued.apply(lambda r: would_be_dropped(r["cue"], r["gold_cls"]),
                               axis=1).mean()) * 100
        if use_col:
            ok = float(cued.apply(lambda r: compatible(r["cue"], r["suffix_cls"]),
                                 axis=1).mean()) * 100
        else:
            ok = exact
    else:
        exact = ok = lost = float("nan")

    # ---- 4. the astronomer case, in numbers. POS is the wrong tool for it, morphology is
    #         the right one, and this is the evidence for that claim.
    ag = q[q["cue"] == "agent"]
    ag_noun = float((ag["gold_cls"] == "noun").mean()) * 100 if len(ag) else float("nan")
    ag_agentive = float(ag["word"].map(is_agentive).mean()) * 100 if len(ag) else float("nan")
    ag_role = q[q["role"] == "agent"]
    ag_role_agentive = (float(ag_role["word"].map(is_agentive).mean()) * 100
                        if len(ag_role) else float("nan"))

    # ---- 5. role distribution, queries and corpus
    role_q = q["role"].value_counts()
    roles_all = df["definition"].map(role_of)
    all_roles = roles_all.value_counts()
    pct_role = float((roles_all != "none").mean()) * 100
    pat_n = int(role_q.get("patient", 0) + role_q.get("instrument", 0))

    # ---- 6. multi word answers, which no pos filter can handle
    multi = float(q["word"].str.contains(" ").mean()) * 100

    def nz(v):
        return None if v is None or (isinstance(v, float) and v != v) else round(v, 2)

    res = dict(
        queries=len(q),
        pos_source=src,
        pct_pos_column_filled=nz(meta.get("pct_pos_filled", 0)),
        top_pos_tags=top_tags,
        gold_pos_distribution={str(k): int(v) for k, v in q["gold_cls"].value_counts().items()},
        pct_words_whose_senses_differ_in_pos=nz(pct_word_multi_pos),
        headword_suffix_coverage_pct=nz(cov),
        headword_class_distribution={str(k): int(v) for k, v in dist.items()},
        spacy_used=bool(sp is not None),
        spacy_vs_column_agree_pct=nz(agree),
        pct_queries_with_a_cue=nz(cued_share),
        cue_matches_gold_pos_pct=nz(exact),
        cue_matches_gold_suffix_pct=nz(ok),
        pct_lost_by_hard_filter=nz(lost),
        agent_cue_queries=len(ag),
        pct_agent_cue_gold_is_noun=nz(ag_noun),
        pct_agent_cue_headword_is_agentive=nz(ag_agentive),
        pct_agent_role_headword_is_agentive=nz(ag_role_agentive),
        pct_nouns_that_are_agentive=nz(pct_nouns_agentive),
        patient_or_instrument_cue_queries=pat_n,
        role_counts_queries={str(k): int(v) for k, v in role_q.items()},
        role_counts_corpus={str(k): int(v) for k, v in all_roles.items()},
        pct_corpus_with_a_role=round(pct_role, 2),
        pct_multiword_answers=nz(multi),
    )
    R.write_json(res, os.path.join(OUTDIR, "pos_roles.json"))

    lines = ["# pos and roles", "",
             "from src/explore_pos.py. numbers on the fixed %d test queries unless said "
             "otherwise. pos source used: %s." % (len(q), src), "",
             "| finding | value |", "|---|---|"]
    for k, v in res.items():
        lines.append("| %s | %s |" % (k, v))
    lines += ["", "notes", ""]
    lines.append("- the POS column is not empty, %.1f%% of rows have a tag in it. our EDA read "
                 "it as empty because the check we ran only accepted a single letter and this "
                 "file writes them as Webster abbreviations, `n.`, `v. t.`, `prep.`, `a.`. "
                 "so tags come from the file when it is there and from the ending when not."
                 % meta.get("pct_pos_filled", 0))
    lines.append("- %d distinct tag strings in the file, top ones: %s."
                 % (len(raw_tags), ", ".join("%s (%d)" % (k, v) for k, v in list(top_tags.items())[:6])))
    lines.append("- %.1f%% of headwords have senses with more than one POS in this corpus. so "
                 "POS is a per sense property here, which is one more reason the feature goes "
                 "next to the model and not as a filter over the whole word." % pct_word_multi_pos)
    if use_col:
        lines.append("- headword endings still name a class for %.1f%% of the %s vocabulary. "
                     "kept because that is the morphology channel, not because it beats the "
                     "column." % (cov, f"{meta['n_words']:,}"))
    if agree is not None:
        lines.append("- spaCy and the file agree on %.1f%% of the sampled headwords. where they "
                     "differ the file is usually right about the sense and the tagger is guessing "
                     "on archaic 1913 text, so the tagger is a cross check and not the source."
                     % agree)
    else:
        lines.append("- no spaCy model in this run. only needed if we want to compare its tags "
                     "with the column, rerun with en_core_web_sm installed.")
    if len(cued):
        lines.append("- %.1f%% of queries open with a cue that states the answer type. the cue "
                     "matches the gold POS %.1f%% of the time. a hard filter on that would have "
                     "lost the gold answer for %.1f%% of them, which is why the class stays a "
                     "feature and a rerank bonus." % (cued_share, exact, lost))
    if len(ag):
        lines.append("- on the astronomer question: of the %d \"person who\" queries, %.1f%% of "
                     "the gold answers are nouns. astronomy is a noun too, so POS cannot "
                     "separate them. the ending can: %.1f%% of those gold words are agentive and "
                     "only %.1f%% of all nouns in the file are. that is the whole argument for "
                     "the morphology channel." % (len(ag), ag_noun, ag_agentive,
                                                   pct_nouns_agentive))
    lines.append("- roles over the %d queries: %s. %.1f%% of the whole corpus opens with one of "
                 "these phrases, so the role is written in the text and needs no parser."
                 % (len(q), ", ".join("%s %d" % (k, v) for k, v in role_q.items()), pct_role))
    lines.append("- %.1f%% of the test answers are multi word entries which no POS filter can "
                 "handle. they stay in the index and count as a limitation." % multi)
    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, "pos_roles.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
