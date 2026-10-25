# Next week goals: Reverse Dictionary

Decide among yourselves who does what before starting the work.

1. **Fix the failure files.** They list every query as missed, because `run_dense.py` compares row numbers with word numbers. Check: misses in the file = 500 minus the hits in the results table.
2. **Apply the self-reference rule to queries only.** Keep entries that mention their own word in the index, and keep them out of the test and validation queries only. Report vocabulary size and R@1/10/100 for both versions.
3. **Swap the 3 bad queries and freeze the 500.** Drop "imp. of Sit.", "Rime. See Rhyme." and "Weeper. See Capuchin", and take the next 3 from the same seeded pool. Keep the other 497 unchanged, then never change the 500 again.
4. **One model table.** BM25, TF-IDF and three dense models (MiniLM, MPNet and `BAAI/bge-base-en-v1.5`) on the same final 500 test queries, searching the full dictionary with the query's own entry removed. Give R@1/10/100 and median rank, each with a 95% bootstrap confidence interval. Also give the same numbers separately for test words with 2, 3-4 and 5+ meanings, and say how many of the 500 queries fall in each group.
5. **Fine-tune one model.** Fine-tune the best of the three dense models as described below, with 3 seeds (mean and standard deviation), and compare with the untrained model on the 500, with 95% intervals.
6. **Rerank the model's top words with simple hints.** For each query, take the model's top 100 words and re-sort them by a score: model similarity plus a small bonus for each hint that agrees. Choose the bonus sizes on validation. Report four rows: untrained and fine-tuned model, each with and without the hints, and each hint alone.
7. **Tidy the repo.** Remove informal wording from the notes, results and scripts. Fix references to files that don't exist. Add a README with the data, setup, how to run each step, the results table and the limitations.

## How to fine-tune (goal 5)

- **Triplets, from training words only:** anchor = a definition of a word with two or more senses, with the word itself (and its inflections) masked out if it appears in the definition, so the model can't learn to copy it; positive = another definition of the same word (random each epoch); hard negative = a BM25 result from a different word.
- **Hard negatives:** search the training definitions with BM25. For each anchor take the top 50, drop the anchor's own word and any identical text, and pick one from about ranks 5 to 50.
- **Train:** `sentence-transformers` with `MultipleNegativesRankingLoss` and "no duplicates" batches.
- **Choose settings on validation queries** (built from the validation words with the same rules as the 500), not on the 500 test queries.
- **Leakage check:** no training text comes from a validation or test word, or matches a validation or test query. Training uses training words only, but at evaluation the search still covers the full dictionary.

## How to rerank with hints (goal 6)

`src/explore_pos.py` in your repo already has most of this (`tag_class`, `classify`, `cue_of`, `role_of`).

- **Word class (POS):** for each candidate, read the POS column of the dictionary file and map it to noun, verb, adjective or adverb. For the query, read the opening words ("To ..." suggests a verb, "One who ..." a noun, "Having ..." or "Of or pertaining to ..." an adjective).
- **Word ending:** take the candidate's ending (-er, -or and -ist for a person; -tion and -ment for an action noun; -ous and -ful for adjectives). A query starting "A person who ..." suggests a person ending.
- **Role:** read it from the query's opening phrase: "a person who" (agent), "the place where" (location), "that which is used" (instrument). Match it to the candidate's class or ending.
- **Score:** model similarity + w_pos × (1 if the query's class and the candidate's agree) + w_end × (same for endings) + w_role × (same for role). Try small values for each weight (for example 0, 0.02, 0.05, 0.1) on the validation queries and keep the best combination.
