# notes

Project notes for the reverse dictionary capstone. Numbers in here come from the scripts, so
if you want to check one, run the script and look at the file it writes. Nothing in this file
is hand-typed from a notebook run except the parts marked as measured on the raw csv.

## the data

`data/OPTED-Dictionary.csv`, 16.6 MB. OPTED is the Online Plain Text English Dictionary, made
from Webster's Revised Unabridged, 1913. Public domain, so it can sit in a public repo.

Columns are Word, Count, POS, Definition. Count is how often the word shows up in the source
text. The POS column exists but only 0.02% of rows have anything in it, so tags have to be
produced and not read.

176,004 rows over 111,597 distinct headwords, 166,065 different definition strings. The brief
says about 96,200 pairs which is a different version of the file. We asked, Ekansh said on
3 Oct that this one is fine and he'll look at it once it is pushed, so the question is closed.

measured on the raw csv, before cleaning:
- 25,309 words (22.68% of the vocabulary) have more than one definition. run has 72, set has 59
- 18,321 rows (10.41%) share their definition string with at least one other word. worst case
  is one string shared by 9 words
- 101 exact duplicate rows, 206 rows where the definition is literally `#NAME?`, 363 rows with
  doubled quotes inside the string
- 1.34% of headwords have characters other than letters. a lot of them are real entries like
  `bank of a river`, so they stay
- 5,622 rows (3.19%) start with "see" or "same as". cross references, no meaning in them
- 23,776 rows (13.54%) have the headword inside its own definition, counting only headwords of
  3 letters or more. one letter words like "a" match anything so they are excluded from the count
- definition length in tokens: median 8, p90 24, p95 32, p99 49, max 148. cutting at 64 tokens
  loses something for 0.28% of rows, at 32 it is 4.57%. so max_seq_length is 64 everywhere
- floors: random word from the vocabulary gives R@1 of 8.96e-06, always answering the single
  most frequent word gives 0.00117, top 50 frequent words gives 0.0111

## cleaning

`src/rexdict.py:load()` does three things and nothing else:
1. drop rows where the definition is `#NAME?`, empty, or under 2 characters
2. drop rows starting with "see" / "same as" / "same for"
3. drop exact duplicate (word, definition) rows

no length filter. in the last meeting one of us said the cross references would get cleared by
a length limit, and they would, along with 17.83% of the file, because that many definitions are
shorter than 3 tokens. so the rule is by pattern instead. no dedup to one sense per word either,
the extra senses are what the polysemy question is about.

## split and metrics

80/10/10 by word, seed 2026, `src/make_splits.py`. Three things inside that choice:
- by word and not by row, because 22.68% of headwords have several definitions. split by row and
  a test query has a sibling sense of its own word in the training data
- the index at query time is the whole dictionary. the words are the resource, what is held out
  is the training pairs
- no cross validation. he said it is less critical at this size and 176k rows is plenty

the 500 queries are rows of test-split words with at least 3 definition tokens, picked with the
same seed, ids committed in `splits/test_query_ids.txt`. the brief asks for 500, so anyone can
rerun on the exact same set.

metrics: Recall@1, @10, @100 and median rank, capped at 1000 when the answer is not in the top
1000 (same convention as the Sõnajaht paper). accuracy is not the headline because it is a
ranking problem, a model can get the gold word into the top 5 without its top-1 exact match
moving at all, and also because the right answer is often a set here. we still report strict
Recall@1 since the brief asks for it.

two ways to count a hit. strict only takes the gold word. tol takes any word sharing the exact
definition string. tol is the fairer number on this corpus but it is a string rule and not a
wordnet rule, so it is a floor. "a person who studies stars" should accept both astronomer and
stargazer, and if their 1913 definitions are worded differently the tolerant rule does not know
that. moving it to synsets is a week 4 job, then about 100 queries judged by hand.

## the self match thing

our test queries are rows of the same file the index is built from. so bm25 gets handed the
answer as a string and can approach 1.0 at Recall@1 without understanding anything. that number
means nothing on its own. so `run_baselines.py` runs both protocols:
- in_dictionary: the query's own row is in the index. it is in-dictionary lookup
- leave_one_out: that row is removed. now the word has to come from a sibling sense or from a
  neighbour's definition

we print how often rank 1 is literally the query row, next to every number. leave_one_out has a
ceiling too: a word with a single row becomes impossible to reach when that row is the query,
and `splits/manifest.json` reports that share as pct_reachable_when_self_removed.

he said human written / colloquial queries are not needed for now, the held out set is enough,
and that colloquial queries would be harder for the model. so we are not building any. if the
leave_one_out numbers end up dominated by unreachable queries, the option is to generate
modern paraphrases of these definitions, which breaks the self match and gives us the colloquial
set at the same time. that is a proposal, not a plan.

## features

`src/explore_pos.py`. POS from the headword ending (spaCy on top when en_core_web_sm is
installed). roles read off the opening of the definition, because he said on 3 Oct he means
agent/patient, karak, "not the subject-object thing". so no nsubj/dobj parsing. a dictionary
writes the role in the text: "a person who" is an agent, "that which is used by" is a patient or
an instrument, "the place where" is a location. he also said other roles are there, so we count
instrument, location, action and manner too.

the same script answers the question that decides the design: how many gold answers a hard POS
filter would throw away. if that is not small, POS stays a feature and a rerank bonus.

his repo reference (WantWords / the multi-channel model) has three extra channels, sememes,
WordNet lexnames and root-affix morphology. ours line up with them roughly: definition cue for
the meaning class, POS class for the lexname, headword suffix for the root-affix.

## baselines

he said two: WantWords ("FontWords" in the notes, it is the thunlp/WantWords repo), and BM25 or
TF-IDF. we do both of the lexical pair since it is cheap, that is `src/run_baselines.py`.

WantWords is not done. checked the repo on 16 Oct. what is published there is the website code
plus a pretrained model and data as a separate download from a Tsinghua cloud link, deps pinned
to 2019 (torch 1.2.0, numpy 1.17.2, pytorch-transformers 1.2.0), no training code. reading
model_en.py: for operation "test" it returns a score over their whole class list, so Recall@10
and @100 are computable, no beam search needed, but their classes are their own dictionary and
not OPTED so the comparison needs a word level mapping. three ways to go:
- real inference on our 500 queries, a day or two, breaks if the download link is dead
- the live site, which returns a few candidates, so only R@1 and R@5 and we say so
- if neither runs, a written note on what blocked it, which he accepted as an outcome

needs one owner from the team. whatever happens it goes to results/wantwords.md.

## running it

local, from the repo root, needs numpy pandas scikit-learn (spacy optional):

    pip install -r requirements.txt
    python src/run_all.py

kaggle, in the notebook with the OPTED dataset attached, no gpu needed for any of this:

    !git clone -q https://github.com/ByteBandit0608/reverse-dictionary-capstone /kaggle/working/repo
    %cd /kaggle/working/repo
    !python src/run_all.py
    !zip -qr /kaggle/working/week3_outputs.zip splits results

if the loader cant find the csv, run `!ls /kaggle/input/*/` and put REXDICT_CSV=<that path> in
front of the python command. then check two things before committing: the manifest should have
n_queries 500, and in_dictionary should be clearly above leave_one_out for bm25. if those two are
close, the protocol is broken and the run is not worth committing.

the scripts are deterministic, same seed same splits same queries same tables.

## open

- wantwords baseline, assigned to someone, this week
- the 20 failures need a human read. the script does the categories, not the judgement
- ask: which number is the headline one for the report, in_dictionary or leave_one_out
- check whether their released model can be used for inference only or whether he expects us to
  retrain their architecture
- he will push reference papers to the repo once the dataset is up. nothing there yet
