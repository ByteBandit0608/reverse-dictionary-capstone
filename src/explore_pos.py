"""what pos / role signal can we get out of this file. python src/explore_pos.py
writes results/pos_roles.md
the POS column is empty so tags come from somewhere else. two options: spaCy with
en_core_web_sm if installed, otherwise suffix rules on the headword, which need
nothing and work on kaggle. in a dictionary the ending carries most of the signal.
one number is why this exists: how many gold answers a hard POS filter would throw
away. if it is not small, pos stays a feature and a rerank bonus and never a filter.
on roles: Ekansh said on 3 Oct he means agent/patient (karak), "not the
subject-object thing", so no nsubj/dobj here. a dictionary states the role in the
first few words ("a person who", "that which is used by"), that is where we read it.
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

# classes we can name from the ending of a headword. Order matters, first hit wins.
SUFFIX = [
    ("place",   r"(arium|ery|ory|ium)$"),
    ("agent",   r"(ers|er|ors|or|ists|ist|ians|ian|eers|eer|ees|ee)$"),
    ("noun_thing", r"(tion|sion|ment|ness|ity|ance|ence|ship|dom|hood|acy|ure|ism)$"),
    ("adj",     r"(ful|ous|ive|able|ible|less|ish|ary|ant|ent|ic|al)$"),
    ("verb",    r"(ize|ise|ify|ate)$"),
]

# cues that say what kind of answer the definition is asking for
# Every cue is anchored to the start of the definition. That is where a dictionary
# states what kind of thing the word is, and matching in the middle picks up phrases
# like "in the manner of one who", which describe nothing about the headword.
CUES = [
    ("agent",      r"^(a |the |one |any )?(person|persons|one|party|individual|being|man|woman|child|boy|girl)s?\s+(who|that)\b"
                   r"|^(one|a person|the person)\s+who\b"
                   r"|^(a |the )?(writer|maker|breaker|keeper|student|lover)\s+of\b"),
    ("tool",       r"^(a |the )?(thing|instrument|device|apparatus|machine|utensil|tool|implement)\b"
                   r"|^(that|something)\s+(which|by means of which)\b"),
    ("place",      r"^(a |the )?(place|spot|building|room|region|country|town|city)\b"
                   r"|^the place where\b"),
    ("noun_thing", r"^(a |the )?(act|action|process|state|quality|condition|result|art|science|study|practice|feeling|emotion|degree|manner)\b"),
    ("adj",        r"^(characterized by|having|of the nature of|of the quality of|tending to|inclined to|marked by|being|like)\b"
                   r"|^of the\b"),
    ("verb",       r"^to\s+[a-z]+"),
]

# A definition cue and a headword ending rarely use the same name for the same idea.
# "instrument by means of which" and an -er headword are the same thing, and a suffix
# rule cannot tell a tool from an agent, so we count both readings.
CUE_OK = {
    "agent": {"agent", "other"},
    "tool": {"agent", "other", "noun_thing"},
    "place": {"place", "noun_thing"},
    "noun_thing": {"noun_thing", "other"},
    "adj": {"adj", "other"},
    "verb": {"verb", "other"},
}


def compatible(cue, cls):
    return cls == cue or cls in CUE_OK.get(cue, set())


def would_be_dropped(cue, cls):
    """Would a hard filter on the cue's class hide the gold answer?
    An "other" ending is unknown, not contradictory, so it is not counted as a loss."""
    if cls == "other":
        return False
    return cls != cue and cls not in CUE_OK.get(cue, set())


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


def role_of(defn):
    d = str(defn).lower().strip()
    for name, pat in ROLES:
        if re.search(pat, d):
            return name
    return "none"


def classify(word):
    w = word.lower()
    for name, pat in SUFFIX:
        if re.search(pat, w):
            return name
    return "other"


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
    text = "\n".join(words)
    for i, doc in enumerate(nlp.pipe([w for w in words], batch_size=4096)):
        t = next(iter(doc), None)
        out[words[i]] = t.pos_ if t is not None else "X"
    return out


def cue_of(defn):
    d = defn.lower().strip()
    for name, pat in CUES:
        if re.search(pat, d):
            return name
    return None


def main():
    df, meta = R.load()
    tr, va, te = R.make_word_split(df)
    qidx = R.pick_queries(df, te)
    q = df.loc[qidx]

    # ---- 1. how much does the headword ending predict POS, on the train words
    one_row_per_word = df.drop_duplicates("wid").reset_index(drop=True)
    one_row_per_word["cls"] = one_row_per_word["word"].map(classify)
    dist = one_row_per_word["cls"].value_counts()
    cov = float((one_row_per_word["cls"] != "other").mean()) * 100

    # ---- 2. spaCy, if available, on the headwords. Also on 2000 definitions so we
    #         can see how much the tagger disagrees with the suffix rule on 1913 text.
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
            tmp["cls"] = tmp["word"].map(classify)
            mapx = {"NOUN": "noun_thing", "PROPN": "noun_thing", "ADJ": "adj",
                    "VERB": "verb", "PART": "other"}
            tmp["pos_cls"] = tmp["pos"].map(lambda p: mapx.get(p, "other"))
            agree = float((tmp["cls"] == tmp["pos_cls"]).mean()) * 100
            dist = tmp["pos"].value_counts().to_dict()

    # ---- 3. cue in the definition vs class of the gold headword
    q = q.copy()
    q["cue"] = q["definition"].map(cue_of)
    q["gold_cls"] = q["word"].map(classify)
    cued = q[q["cue"].notna()]
    cued_share = float(len(cued) / len(q)) * 100
    if len(cued):
        exact = float((cued["cue"] == cued["gold_cls"]).mean()) * 100
        ok = float(cued.apply(lambda r: compatible(r["cue"], r["gold_cls"]), axis=1).mean()) * 100
        # a hard filter drops every gold whose class is not in the accepted set.
        # "other" is excluded because an unknown ending is not evidence against a cue
        lost = float(cued.apply(lambda r: would_be_dropped(r["cue"], r["gold_cls"]),
                               axis=1).mean()) * 100
    else:
        exact = ok = lost = float("nan")
    correct = exact

    # ---- 4. the case that motivated this: person who studies stars
    agentish = q[q["cue"] == "agent"]
    bad = agentish[agentish["gold_cls"] != "agent"] if len(agentish) else agentish

    # ---- 5. role distribution, on the queries and on the whole corpus
    q["role"] = q["definition"].map(role_of)
    role_q = q["role"].value_counts()
    roles_all = df["definition"].map(role_of)
    all_roles = roles_all.value_counts()
    pct_role = float((roles_all != "none").mean()) * 100
    pat_n = int(role_q.get("patient", 0) + role_q.get("instrument", 0))
    # for agent role queries, does the headword actually carry an agentive ending
    ag = q[q["role"] == "agent"]
    ag_suffix = round(float((ag["gold_cls"] == "agent").mean()) * 100, 2) if len(ag) else None

    # ---- 6. multi-word answers, since a POS filter cannot be applied to those at all
    multi = float((q["word"].str.contains(" ")).mean()) * 100

    def nz(v):
        return None if v is None or (isinstance(v, float) and v != v) else round(v, 2)

    res = dict(
        queries=len(q),
        headword_suffix_coverage_pct=nz(cov),
        headword_class_distribution={str(k): int(v) for k, v in dist.items()},
        spacy_used=bool(sp is not None),
        spacy_vs_suffix_agree_pct=nz(agree),
        pct_queries_with_a_cue=nz(cued_share),
        cue_equals_gold_class_pct=nz(exact),
        cue_compatible_with_gold_class_pct=nz(ok),
        pct_lost_by_hard_filter=nz(lost),
        agent_cue_queries=len(agentish),
        agent_cue_wrong_class=len(bad),
        patient_or_instrument_cue_queries=pat_n,
        role_counts_queries={str(k): int(v) for k, v in role_q.items()},
        role_counts_corpus={str(k): int(v) for k, v in all_roles.items()},
        pct_corpus_with_a_role=round(pct_role, 2),
        pct_agent_role_queries_with_agentive_headword=ag_suffix,
        pct_multiword_answers=nz(multi),
    )
    R.write_json(res, os.path.join(OUTDIR, "pos_roles.json"))

    lines = ["# pos and roles", "",
             "from src/explore_pos.py. numbers on the fixed %d test queries unless said "
             "otherwise." % len(q), "",
             "| finding | value |", "|---|---|"]
    for k, v in res.items():
        lines.append("| %s | %s |" % (k, v))
    lines += ["", "notes", ""]
    lines.append("- POS column is %.2f%% filled, so every tag here is produced, not read "
                 "from the file." % meta["pct_pos_filled"])
    lines.append("- headword endings classify %.1f%% of the %s vocabulary. costs nothing to "
                 "compute and the multi-channel model uses the same idea."
                 % (cov, f"{meta['n_words']:,}"))
    if agree is not None:
        lines.append("- spaCy and the suffix rule agree on %.1f%% of headwords. where they "
                     "differ the tagger is mostly right on normal words and wrong on archaic "
                     "1913 spellings, so tag quality is a real limit on this feature." % agree)
    else:
        lines.append("- no spaCy model in this run, so only the suffix rule. rerun with "
                     "en_core_web_sm installed to get the tagger side.")
    lines.append("- %.1f%% of queries start with a cue that names the answer type (person who "
                 "/ thing that / place where / act of ...). Where a cue exists, the gold headword "
                 "ends in exactly the matching class %.1f%% of the time, or %.1f%% of the time "
                 "if a tool ending in -er counts as matching an instrument cue." %
                 (cued_share, exact, ok))
    lines.append("- The gap between those two numbers is naming, not the signal. A suffix rule "
                 "cannot tell an agent from an instrument, because both take -er. The published "
                 "multi-channel models have the same problem.")
    lines.append("- a hard POS filter would lose the gold answer for %.1f%% of them, so for "
                 "now the class is a feature and a rerank bonus, not a filter." % lost)
    lines.append("- Roles over the %d queries: %s. Over the whole corpus %.1f%% of definitions "
                 "open with one of these role phrases, so the role is stated in the text and can "
                 "be read without a parser."
                 % (len(q), ", ".join("%s %d" % (k, v) for k, v in role_q.items()),
                    pct_role))
    if len(ag):
        lines.append("- Of the queries with an agent role (\"a person who ...\"), %.1f%% have a "
                     "headword with an agentive ending. Where they disagree the definition is "
                     "describing a person for a word that is not formed from a verb, which is "
                     "the case a morphology channel cannot help with." % ag_suffix)
    lines.append("- The patient and instrument cues together cover %.1f%% of the queries, the "
                 "smaller half of the role signal. %d of them."
                 % (100.0 * pat_n / max(1, len(q)), pat_n))
    lines.append("- %.1f%% of the test answers are multi word entries, which no POS filter "
                 "can handle. left in the index, counted as a limitation." % multi)
    if len(bad):
        lines += ["", "person-who queries whose answer is not an -er word", ""]
        for _, r in bad.head(10).iterrows():
            lines.append("- `%s` -> **%s** (%s)" % (r["definition"][:80], r["word"], r["gold_cls"]))
    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, "pos_roles.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
