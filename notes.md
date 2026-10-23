# notes

Project notes for the reverse dictionary capstone. Every number here was written out by a script
into `results/` or `splits/manifest.json`, so if one looks wrong, run the script and compare. The
parts measured straight off the csv say so.

Two query set versions are in this file. v1 is the 500 queries that the 17 Oct run scored. v2 adds
the two cleaning rules under `## query set v2`. Numbers from the two are not comparable and are
labelled everywhere they appear.

## the data

`data/OPTED-Dictionary.csv`, 16.6 MB. OPTED is the Online Plain Text English Dictionary, made from
Webster's Revised Unabridged, 1913. Public domain, so it can sit in a public repo.

Columns are Word, Count, POS, Definition. Count is how often the word shows up in the source text.

176,009 rows over 111,597 distinct headwords before cleaning (`rows_raw` in
`splits/manifest.json`), 166,065 different definition strings. The brief says about 96,200 pairs,
which is a different version of the file. The larger one is what we have and what the numbers below
are over.

measured on the raw csv, before cleaning:
- 25,309 words (22.68% of the vocabulary) have more than one definition. `run` has 72, `set` has 59
- 18,321 rows (10.41%) share their definition string with at least one other word. worst case is one
  string shared by 9 words
- 101 exact duplicate rows, 363 rows with doubled quotes inside the string
- 207 lines of the raw file contain `#NAME?` and every one of them has it in the Word and Count
  columns, none in the Definition column. Those rows are an Excel error value that came in when the
  source was resaved, the headword is `#NAME?` and the definition is a fragment like "of Baptistry"
- 1.34% of headwords have characters other than letters. many of them are real entries like
  `bank of a river`, so they stay
- 5,622 rows (3.19%) start with "see" or "same as". cross references, no meaning in them
- 23,776 rows (13.54%) have the headword inside its own definition, counting headwords of 3 letters
  or more. single letter words like "a" match anything, so they are outside that count
- definition length in tokens: median 8, p90 24, p95 32, p99 49, max 148. cutting at 64 tokens
  loses something for 0.28% of rows, at 32 it is 4.57%. `max_seq_length` is 64 everywhere for that
  reason
- floors: a random word from the vocabulary gives R@1 of 8.96e-06, always answering the single most
  frequent word gives 0.00117, always answering from the top 50 gives 0.0111

## the POS column

Filled on 97.91% of rows. It holds Webster's own abbreviations, `n.`, `v. i.`, `v. t.`, `a.`,
`adv.`, `prep.`, `conj.`, `interj.`, `pron.`, and compounds like `imp. & p. pr.` and
`a. & superl.` Empty cells are written as `""` in the csv, which pandas reads as two quote
characters, so an "is it empty" check has to strip quotes or it lies.

The first version of the EDA reported this column as empty. That was a parsing error, the check
accepted a single letter as a tag and these tags are mostly longer than that. Nothing is wrong with
the file. POS is read from the column, and the headword ending stays as a fallback and as the
morphology channel.

## cleaning

`src/rexdict.py:load()` drops five things:
1. rows whose definition is empty or under 2 characters
2. rows where `#NAME?` appears in the headword or in the definition
3. rows with no headword at all, 4 of them. A row with an empty word can never be answered
   correctly, it is only noise in the index, and it breaks the failure report because the empty cell
   comes back through a csv read as a NaN
4. rows starting with "see" / "same as" / "same for", and exact duplicate (word, definition) rows
5. rows whose definition contains the headword it is defining, matched after stripping common
   inflections so `Abolishment` catches `abolishing`

On 2, the first version of the rule looked only at the definition column and found nothing there,
then the fix looked at the start of the string and found 5 rows, and both were the wrong question.
The damage is in the headword column, which is why the count was invisible to a definition based
rule. The manifest now prints `rows_named_in_word`, `rows_named_in_definition` and
`rows_named_word_exact` beside `rows_dropped_named`, so the 206 in the old notes and the 207 from a
grep of the file are compared from the output rather than from memory.

On 1 and 3, `rows_dropped_bad_def` is the union of the short definitions, the `#NAME?` rows
and the rows with no headword, so it reads 215 and not 5 plus 207 plus 4. The drops add back to the
total exactly, 176,009 minus 215 minus 5,616 minus 232 minus 34,039 is 135,907, which is `rows` in
the manifest, and the vocabulary loses only two headwords because all 207 named rows share the one
fake entry `#NAME?`.

On 5, this is the label leakage case. A definition that contains its own word can be scored by
string copy alone, so it goes out of the corpus and not only out of the query set.
`pct_rows_self_ref` and `pct_rows_self_ref_exact` report how much of the file that is, 19.45% and
13.33% on this corpus, and the EDA notebook prints them on their own.

One consequence worth knowing: 4,565 headwords exist only as cross references or duplicates, so
dropping those rows removes those words from the answer space too. 111,597 distinct words raw
becomes 107,032 after cleaning. That is the vocabulary every Recall number below is over.

There is deliberately no length filter. A 3 token cut would clear the cross references, along with
17.83% of the file, because that many definitions are shorter than 3 tokens. Cleaning is by pattern.
Nor do we collapse a word to one sense, the extra senses are what the polysemy question is about.

What the self reference rule costs has to be said as plainly as what it fixes: 34,039 rows go out,
and roughly 11,700 headwords with them, because those words are written in this file only in a way
that names themselves. They are no longer in the answer space at all. The row and word totals are in
`splits/manifest.json` rather than repeated here, since they move every time a rule changes and this
file has already been wrong once about a count that the scripts could read out better.

## query set v2

Two rules on top of the v1 rule set, both in `src/make_splits.py` and `src/rexdict.py`:
- a query word needs at least 2 rows in the cleaned corpus (`MIN_QUERY_SENSES = 2`)
- a query cannot be a definition that names its own headword, because such rows no longer exist

The first one matters more than it looks. On v1, 42.6% of the queries were words with exactly one
row, and under leave_one_out removing that row removes the answer, so no method could ever be
scored on them. Their ceiling is 1.0 now and the v2 manifest reads `pct_reachable_when_self_removed
100.0`, which is the same fact as `unreachable: 0` in the failure categories, seen from the other
end.

Side effect to expect: some words lose their second sense to the self reference rule and leave the
query set entirely. That is why cleaning runs before the query pick, never after.

The pool stayed deep enough. `n_queries` is 500 under both rules, so the brief's number still
holds and nobody had to pad anything. If the pool ever is smaller, `pick_queries` returns the whole
of it rather than padding, and the manifest prints the real count, so a short query set shows up in
the output instead of being smoothed over.

## split and metrics

80/10/10 by word, seed 2026, `src/make_splits.py`. Three things inside that choice:
- by word and not by row, because 22.68% of headwords have several definitions. split by row and a
  test query has a sibling sense of its own word in the training data
- the index at query time is the whole dictionary. the words are the resource, what is held out is
  the training pairs
- no cross validation. at 170k rows and a fixed held out query set the second fold buys nothing

The queries are rows of test split words with at least 3 definition tokens, picked with the same
seed, and the ids are committed in `splits/test_query_ids.txt` so anyone can score on the exact same
set.

Metrics are Recall@1, @10, @100 and median rank, capped at 1000 when the answer never reaches the
top 1000, the convention the Sõnajaht paper uses. Accuracy is not the headline because this is a
ranking problem: a model can put the gold word at 5 without its top-1 exact match moving at all.
Strict Recall@1 is still reported since the brief asks for it.

A hit is counted two ways. Strict takes only the gold word. Tolerant takes any word sharing the
exact definition string. Tolerant is the fairer number on this corpus but it is a string rule and
not a wordnet rule, so treat it as a floor. "a person who studies stars" should accept both
`astronomer` and `stargazer`, and if their 1913 definitions are worded differently the tolerant rule
does not know that.

Two protocols, both from `run_baselines.py`, and the dense runner uses the same two so the tables sit
next to each other:
- in_dictionary: the query's own row is in the index
- leave_one_out: that row is removed, so the word has to come from a sibling sense or from a
  neighbour's definition

The test queries are rows of the same file the index is built from, so in_dictionary hands the answer
over as a string and a lexical method can approach 1.0 at rank 1 without understanding anything.
That column is a check that the index is wired up, not a result. How often rank 1 is literally the
query row is printed next to every number, and it should be about 0% under leave_one_out.

The two manifest numbers that bound the table are `pct_reachable_when_self_removed` and
`ceiling_strict_top1`. The first is 100.0 on v2 and was 57.4 on v1. The second is 0.995 on v2
against 0.996 on v1, and it is not an off by one either time: 1.0% of the v2 queries share their
definition string with another word, so rank 1 is a coin flip on those. Tie rate by definition
length, measured on the cleaned file, is 56.4% among rows under 3 tokens and 1.1% among rows of 3
tokens or more, which is why a query set that asks for 3 tokens or more is nearly tie free while the
corpus is not. Quote 0.995 for the metric table and not the whole file number, they have different
denominators.

## baselines and the dense model, v2 queries, run of 23 Oct

This run is over 135,907 cleaned rows, 95,286 headwords and the 500 v2 queries, written by
`src/run_all.py` and `src/run_dense.py` into `results/baselines.md` and `results/embeddings.md`.

| method | protocol | R@1 | R@10 | R@100 | medrank | self@1 |
|---|---|---|---|---|---|---|
| bm25 | in_dictionary | 0.990 | 1.000 | 1.000 | 1 | 99.0% |
| tfidf | in_dictionary | 0.992 | 1.000 | 1.000 | 1 | 99.2% |
| mpnet | in_dictionary | 0.992 | 1.000 | 1.000 | 1 | 99.2% |
| bm25 | leave_one_out | 0.046 | 0.152 | 0.322 | 945 | 0.0% |
| tfidf | leave_one_out | 0.048 | 0.162 | 0.350 | 884 | 0.0% |
| mpnet | leave_one_out | 0.050 | 0.170 | 0.394 | 319 | 0.0% |
| random | in_dictionary | 0.000 | 0.002 | 0.004 | 1000 | 0.0% |
| word_freq | in_dictionary | 0.000 | 0.000 | 0.000 | 1000 | 0.0% |

`mpnet` is `sentence-transformers/all-mpnet-base-v2`, 768 dim, no fine tuning, `max_seq_length` 64,
and it encoded 135,907 definitions in 164 s on a Kaggle T4. The tolerant columns are in the two
result files and are not copied here.

How to read the leave_one_out block:

- all three are near the floor, and the dense model is the best of them at every cutoff, 0.050
  against 0.046 at 1, 0.170 against 0.152 at 10, 0.394 against 0.322 at 100
- the distance grows with k and median rank falls from 945 to 319, so the dense model is putting the
  right word in the list far more often without getting it to the front. That is a ranking problem
  sitting on top of a retrieval problem, and it is the argument for fine tuning and a reranker
- bm25 and tfidf change places between the two runs and by two or three queries at a time. Neither
  ordering is a finding about the scoring functions
- in the in_dictionary block all three sit at 0.99 or 0.992 against a 0.995 ceiling, which is that
  column measuring the index rather than the method

The v1 table below is kept so the effect of the query rule is visible, but a method looking better in
the v2 column has not necessarily improved, it is answering a different set of 500 questions where
every question can actually be answered.

## baselines, v1 queries, run of 17 Oct, superseded

| method | protocol | R@1 | R@10 | R@100 | medrank | self@1 |
|---|---|---|---|---|---|---|
| bm25 | in_dictionary | 0.998 | 1.000 | 1.000 | 1 | 99.8% |
| bm25 | leave_one_out | 0.032 | 0.132 | 0.264 | 1000 | 0.0% |
| tfidf | in_dictionary | 0.998 | 1.000 | 1.000 | 1 | 99.8% |
| tfidf | leave_one_out | 0.022 | 0.120 | 0.274 | 1000 | 0.0% |
| random | in_dictionary | 0.000 | 0.000 | 0.004 | 1000 | 0.0% |
| word_freq | in_dictionary | 0.000 | 0.000 | 0.000 | 1000 | 0.0% |

leave_one_out is the column a learned model has to beat.

Over the 287 queries still answerable once the row is removed, bm25 is R@1 5.6% and R@100 46.0%.
100 of those 287 share no term at all with any surviving row, which no lexical method can fix. That
is the argument for a dense model rather than more bm25 tuning.

bm25 0.032 against tfidf 0.022 is 16 queries against 11 out of 500. Not a finding. Five queries on
this set is inside what sampling noise does, so the two are a tie and the interesting part is that
both are near the floor.

The failure buckets add back to the recall columns exactly, which is how the two scripts were
checked against each other: 16 at rank 1, plus 50 more by rank 10, plus 66 more by rank 100, leaves
368 unretrieved, and 368 is 213 unreachable plus 100 with no lexical overlap plus 55 outranked past
100.

## failure phenomena, v2 queries

`src/error_analysis.py` labels every miss from fields already in the row, so the count is
reproducible and not a judgement call. All five categories are printed in a fixed order including the
empty ones, which is how the page shows `unreachable: 0` where it said 213 on v1. The bm25 misses are
477 of 500, and the buckets are 244 with no lexical overlap, 176 outranked, 52 right word deep and 5
tied definitions, which is 477.

| what the model is actually missing | queries | mean tokens | mean rare terms |
|---|---|---|---|
| the dictionary words it up differently | 220 | 10.2 | 3% |
| right word found, buried past the top 10 | 162 | 12.6 | 2% |
| right word found, ranked just below | 47 | 16.1 | 2% |
| returned a synonym the definition itself names | 19 | 14.0 | 5% |
| long definition, several ideas at once | 18 | 30.8 | 4% |
| rare or technical wording in the query | 6 | 3.5 | 75% |
| structural, two words share the exact string | 5 | 4.6 | 20% |

Those add to 477 as well, so no query is counted twice and none is lost.

The largest bucket is the dictionary simply using other words, 220 of them, and those queries are
short, 10.2 tokens with almost no rare terms. It is not that the query is hard to read, it is that a
1913 definition of `Bolt` is a crossbow bolt and the index answers `Dart`. The two right word found
rows together are 209 queries where the answer was in the list, 162 of them past rank 10, which is
the same fact the 0.046 against 0.322 gap states. Rare or technical wording is only 6 queries and
they are the shortest and strangest rows in the set, mean 3.5 tokens with 75% of their terms used
twice or less in the whole file.

Two numbers beside that table:

- 6.5% of the misses had a rank 1 that the query definition itself names as a synonym, `Acinus`
  answered with `Drupelet`, `Prolocutor` with `Spokesman`. The v1 figure was 14.5% and the two are
  not a comparison, the v1 set had hundreds of queries whose answer the protocol had deleted
- 138 queries had the gold word inside the top 100 and not first, median rank 16. That is 138 of 477
  misses, and a median of 16 is one reorder away, which is where a cross encoder reranker would go

The script writes a 20 row table, 5 from each category that has them, preferring the cases where rank
1 is a word the definition names, with the last column left empty. That column is the human read and
it is the part of this task that is not a script.

## POS and role features

`src/explore_pos.py` writes `results/pos_roles.md`. POS comes from the column, `tag_class()` maps
the Webster abbreviations to noun / verb / adj / adv / prep / conj / interj / pron, and the ending
stays as the morphology channel. spaCy is installed only as a cross check and it agrees with the
column on 52.7% of sampled headwords, which is expected: the tagger sees a bare word with no
sentence and picks the most common usage, while the file knows which sense the row is.

Roles are read off the opening of the definition, not from dependency parsing. The brief asks for
karak style roles, agent and patient and instrument, and a dictionary writes those in the text, so
"a person who" is an agent, "that which is used by" is an instrument, "the place where" is a
location.

What came back, the v2 run of 23 Oct:
- 7.09% of headwords have senses with more than one POS, so the tag is per sense. That is why it
  goes in beside the model and not as a filter over the whole word
- 34.8% of queries open with a cue that states the answer type and the cue names the gold POS 90.8%
  of the time, yet a hard filter would still drop the gold answer for 6.9% of them. On v1 that loss
  was 4.4%, so the number depends on which queries are in the set and should be quoted with the set
- on the `astronomer` style question there are 16 "person who" queries. 87.5% of their gold answers
  are nouns, so not all of them, and `astronomy` is a noun too, which is the point, the tag cannot
  separate the two. The ending can: 81.2% of those gold words are agentive against 16.9% of all
  nouns in the file, about a 5x lift. The sample is 16 and it should be said out loud as 16
- 22.66% of the corpus opens with a role phrase. agent appears on 4,739 rows and patient on 26, so
  OPTED almost never writes a patient style definition and that feature will be sparse. The 248
  distinct tag strings, the 40.0% of headwords whose ending names a class and the 55.8% spaCy
  agreement are all in `results/pos_roles.md`
- 0.6% of the answers are multi word entries which no POS filter can handle. They stay in the index
  and are reported as a limitation

## embedding models

`src/run_dense.py` encodes every cleaned definition row with a pretrained sentence transformer and
scores the same fixed queries under both protocols, no fine tuning and no reranker. Per model
vectors are cached as `results/emb_<tag>.npy`, which `.gitignore` keeps out of the repo, so a rerun
that adds one model reuses what is already encoded. It writes
`results/embeddings.md` and `.json` with a `table` key in the same shape as the baseline files,
plus `results/failures_<tag>_leave_one_out.csv` in the baselines' columns so the phenomenon counts
and the miss overlap cover it too.

Models so far:
- `sentence-transformers/all-MiniLM-L6-v2`, 6 layers, 384 dim, runs on a Kaggle P100 in a few
  minutes for the full corpus
- `sentence-transformers/all-mpnet-base-v2`, 12 layers, 768 dim, roughly three times the encode time
- a third id, still to be added. `models.txt` in the repo root takes one id per line and the script
  appends a row rather than overwriting, so adding it is one edit and one run

`max_seq_length` is set to 64 in the script to match the truncation the corpus statistics above were
measured with. If the third model wants a different window, `--max_len` is there and the number
belongs in the table next to the model name rather than being silently different.

The comparison is a comparison, not a win condition. These are modern English encoders pointed at
1913 definitions, so a zero shot dense model landing near the lexical baselines is the expected
result and is the case for fine tuning next week rather than a failure of the run.

Three things about what is in this table and what is not. MiniLM was run on the v1 query set and
its numbers are in `results/minilm.md` from that run, 0.028, 0.132, 0.284 on leave_one_out against
that split's bm25 of 0.032, 0.132, 0.264. So the third row of the comparison he asked for is on
different queries than the other two, and it has to be said that way rather than put in one table as
if it were matched. Rerunning MiniLM through `run_dense.py` is three minutes on the same GPU and it
is the thing that makes the three model rows comparable. Second, a dense model beating two lexical
methods by two to seven points at 100 while losing at 1 is not a win to lead with, the honest lead is
median rank 945 to 319. Third, the corpus, index and query ids for every row here are in
`splits/manifest.json`, so any of these numbers can be recomputed and argued with.

## WantWords

The brief names it as the reference model. What is published there is the website code plus a
pretrained model behind a separate Tsinghua download, dependencies pinned to 2019 (torch 1.2.0,
numpy 1.17.2, pytorch-transformers 1.2.0) and no training code or training data in the repo. Reading
`model_en.py`: a BERT encoder into a 300 dim vector, scored against a frozen word embedding table,
with three auxiliary heads, sememe, root affix morphology and WordNet lexname, each added as a
bonus term over the candidate list. For their own class list that means Recall@k is computable
without beam search, but their classes are their dictionary and not OPTED, so running it needs a word
level mapping.

That mapping is not this week's work. The useful part for us is the feature design, which is written
up in `results/wantwords_features.md`, and the three channels line up with what we already compute:
definition cue for the meaning class, POS class for the lexname, headword suffix for root affix.
Fine tuning their architecture is out of scope for this capstone unless the download and the pin
breakage turn out to be cheap.

## running it

local, from the repo root, needs numpy pandas scikit-learn, spacy optional:

    pip install -r requirements.txt
    python src/run_all.py

kaggle, GPU off for all of the above, GPU on plus internet for the dense runner:

    !git clone -q https://github.com/ByteBandit0608/reverse-dictionary-capstone /kaggle/working/repo
    %cd /kaggle/working/repo
    !python src/run_all.py
    !python src/run_dense.py --models sentence-transformers/all-mpnet-base-v2
    !zip -qr /kaggle/working/outputs.zip splits results

If the loader cannot find the csv, `!ls /kaggle/input/*/` and put `REXDICT_CSV=<that path>` in front
of the command. Two checks before anything is committed: the manifest should show `n_queries` and
`pct_reachable_when_self_removed 100.0` for v2, and in_dictionary should sit clearly above
leave_one_out. If those two are close, the protocol is broken and the run is not worth committing.

The scripts are deterministic. Same seed, same split, same queries, same tables.

## not done yet

- MiniLM on the current query set, through `src/run_dense.py`, so that all three model rows are on
  the same 500 queries. Until then the model comparison is two rows on v2 and one on v1
- the third model id and its row
- the judgement column in the 20 row failure table, which is a read of each query and not a script
  output
- the tolerant metric moved from shared definition strings to WordNet synsets, which would need
  about 100 queries checked by hand
- the README, left to last so the numbers in it are the ones we keep
