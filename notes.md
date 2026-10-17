# notes

Project notes for the reverse dictionary capstone. Numbers in here come from the scripts, so
if you want to check one, run the script and look at the file it writes. Nothing in this file
is hand-typed from a notebook run except the parts marked as measured on the raw csv.

## the data

`data/OPTED-Dictionary.csv`, 16.6 MB. OPTED is the Online Plain Text English Dictionary, made
from Webster's Revised Unabridged, 1913. Public domain, so it can sit in a public repo.

Columns are Word, Count, POS, Definition. Count is how often the word shows up in the source
text.

The POS column is filled, 97.9% of rows. It holds Webster's own abbreviations, `n.`, `v. i.`,
`v. t.`, `a.`, `adv.`, `prep.`, `conj.`, `interj.`, `pron.`, `s.`, and some compounds like
`imp. & p. pr.` and `a. & superl.`. Empty cells are written as `` in the csv, which pandas
reads as a pair of quote characters, so a "is it empty" check has to strip quotes or it lies.

We got this wrong at first. In the meeting the answer given to him was that the column is empty,
that came from our EDA which only accepted a single letter as a tag. It isn't empty, we were
reading it badly. So POS is read from the file and the suffix rule is kept only as the fallback
and as the morphology channel.

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

one consequence to know about: 4,565 headwords exist only as cross references or duplicates, so
dropping those rows takes them out of the answer space too, 111,597 distinct words in the raw
file becomes 107,032 after cleaning. that is the vocabulary every Recall number is over.

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
answer as a string and can approach 1.0 at Recall@1 without understanding anything. so
`run_baselines.py` runs both protocols:
- in_dictionary: the query's own row is in the index. it is in-dictionary lookup
- leave_one_out: that row is removed. now the word has to come from a sibling sense or from a
  neighbour's definition

we print how often rank 1 is literally the query row, next to every number.

two numbers from `splits/manifest.json` set the ceilings for what the table can look like.
- pct_reachable_when_self_removed is 57.4. under leave_one_out, 42.6% of the queries lose their
  word completely, because that word had exactly one row and that row is the query. so the honest
  ceiling on that protocol is about 0.574 and not 1.0
- ceiling_strict_top1 is 0.996 on our 500 queries and that is not an off by one. tie rate by
  definition length, measured: 10.2% over all 170,140 rows, 56.4% among rows under 3 tokens,
  1.1% among rows of 3 tokens or more. so repeated definition strings are the very short ones,
  and our query rule which asks for 3 tokens or more filters them out. that is why the query set
  is nearly tie free while the corpus is not. quote 0.996 for the metric table, and 0.944 only
  when talking about the file as a whole, they are different denominators

he said human written / colloquial queries are not needed for now, the held out set is enough,
and that colloquial queries would be harder for the model. so we are not building any. if the
leave_one_out numbers end up dominated by unreachable queries, the option is to generate modern
paraphrases of these definitions, which breaks the self match and gives us the colloquial set at
the same time. that is a proposal, not a plan.

## features

`src/explore_pos.py`. POS comes from the column now, `tag_class()` maps the Webster
abbreviations to noun / verb / adj / adv / prep / conj / interj / pron. the suffix rule stays as
the fallback and as the morphology channel. spaCy, when installed, is only a cross check. roles read off the opening of the definition, because he said on 3 Oct he means
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

## baseline results, run of 17 Oct on the pushed split

| method | protocol | R@1 | R@10 | R@100 | medrank | self@1 |
|---|---|---|---|---|---|---|
| bm25 | in_dictionary | 0.998 | 1.000 | 1.000 | 1 | 99.8% |
| bm25 | leave_one_out | 0.032 | 0.132 | 0.264 | 1000 | 0.0% |
| tfidf | in_dictionary | 0.998 | 1.000 | 1.000 | 1 | 99.8% |
| tfidf | leave_one_out | 0.022 | 0.120 | 0.274 | 1000 | 0.0% |
| random | in_dictionary | 0.000 | 0.000 | 0.004 | 1000 | 0.0% |
| word_freq | in_dictionary | 0.000 | 0.000 | 0.000 | 1000 | 0.0% |

the in_dictionary row is the self match and not skill, 499 of 500 had their own definition row at
rank 1. leave_one_out is the column a learned model has to beat.

over the 287 queries still answerable after the row is removed, bm25 is R@1 5.6% and R@100 46.0%.
so there is somewhere for a dense model to go. 100 of those 287 share no term at all with any
surviving row, which no lexical method can fix, that is the argument for the neural model rather
than tuning bm25 further. median rank on leave_one_out is reported as 1000 because that is where
we cap it, 368 of 500 never appear in the top 100 at all.

bm25 0.032 against tfidf 0.022 is 16 queries against 11. not a finding, do not present it as one.

the failure buckets add back to the recall columns exactly, which is how we checked the two
scripts agree: 16 at rank 1, plus 50 more by rank 10 which is the right word but deep bucket,
plus 66 more by rank 100, leaves 368 unretrieved, which is 213 unreachable plus 100 with no
lexical overlap plus 55 out ranked past 100.

## what the pos and role features came back with

run of 17 Oct, from `results/pos_roles.md`. the short version is that POS is a real column and the
ending is a real signal, and neither one separates astronomer from astronomy.

- 97.91% of rows carry a tag, 273 distinct strings, top ones `n.` 81,092 then `a.` 34,596 and
  `v. t.` 17,977. the EDA claim that the column was empty was our parsing error, not the data
- 9.86% of headwords have senses with more than one POS, so the tag is per sense. that is why it
  goes in as a feature beside the model and not as a filter over the whole word
- 31.6% of the queries open with a cue that states the answer type, and the cue names the gold POS
  95.6% of the time, yet a hard filter would still drop the gold answer for 4.4% of them
- the karak point he asked about: 17 "person who" queries, gold is a noun in 100% of them, and
  only 17.0% of all nouns in the file are agentive while 88.2% of those 17 gold words are. so the
  ending carries about a 5x lift that the POS tag cannot see. small sample, say the 17 out loud
- 24.2% of the whole corpus opens with a role phrase, so the role is written in the text and needs
  no parser. agent 5,017 rows, patient 38, which is worth knowing: OPTED almost never writes a
  patient style definition, so the patient feature will be sparse and should not be oversold
- spaCy agrees with the column on 52.7% of sampled headwords. the tagger sees a bare word with no
  sentence and picks the most common usage, the file knows which sense the row is. so the tagger is
  a cross check and the column is the source
- 0.6% of the answers are multi word entries which no POS filter can handle at all

## what the failures look like, read by hand

the 20 query failure analysis in the brief. `src/error_analysis.py` takes 5 from each category now
instead of the first 20 rows, and inside a category it prefers the cases where rank 1 is a word the
definition itself lists, because those are the ones worth a sentence.

two numbers from `results/failures_annotated.csv` that change how the 0.032 reads:

- 14.5% of the 484 misses returned, at rank 1, a word that appears in the query definition as a
  synonym. Abolishment came back as Abolition, Admissible as Permissible, Amorphous as Crystalline.
  so the retriever is mostly not lost, it hands back a different dictionary word for the same idea
  and the strict metric calls that wrong. this is the argument for a synonym aware metric, and it
  is the same argument as the tolerant column but counted where it matters
- 116 queries had the gold word inside the top 100 but not at rank 1, median rank 13, 90th
  percentile 68. that is the reorder pool. a dense model does not have to find these, only rank
  them higher, which is why R@100 0.264 next to R@1 0.032 is not a contradiction and why the
  reranker is on the plan

the rest of the misses are structural. 213 unreachable because the protocol deleted the only row,
100 with no shared term at all, which is the part a lexical method can never fix.

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
- the 20 failures need a human read. the script picks them now, the judgement column is
  still blank and that is ours to write
- rerun run_baselines.py and error_analysis.py after the note wording fix, results/ has
  the old sentence about the 0.996 ceiling in it
- ask: which number is the headline one for the report, in_dictionary or leave_one_out
- check whether their released model can be used for inference only or whether he expects us to
  retrain their architecture
- he will push reference papers to the repo once the dataset is up. nothing there yet
