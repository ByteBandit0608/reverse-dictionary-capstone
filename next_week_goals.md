# Next goals: Reverse Dictionary (final week)

Decide among yourselves who does what. One week remains. Goals 1 to 8 complete the project. Do the extras only with time left, in order.

**Required**

1. **Frozen test set, clean rerun.** Store the 500 queries by word and definition text, not row numbers. Rerun every method on them and regenerate all result files, including the failure files (misses = 500 minus hits at rank 1). No old numbers may remain.
2. **One model table.** BM25, TF-IDF, MiniLM, MPNet and `BAAI/bge-base-en-v1.5`, searching the full dictionary with the query's own entry removed. Give Recall@1, @10, @100 and median rank with 95% bootstrap intervals.
3. **Neural model.** Fine-tune the best dense model with a contrastive loss and BM25 hard negatives (see below). Compare with the untrained model on the 500, with 95% intervals.
4. **Casual-description queries.** Build 100 casual descriptions of test words (half one meaning, half several), from an AI model (give it only the word, ask how a person would describe it without using the word), a real description dataset (for example the human descriptions in the multi-channel model's data), or both. Drop any that contain the target word, save each with its target, and read 20 by hand. Keep them out of all training and settings. Compare untrained and fine-tuned results with the 500, with 95% intervals. Say in the report where they came from.
5. **Failure analysis.** Take 20 failures from the final system, from both query sets, covering different kinds. One sentence each, an input type (paraphrase, colloquial, vague, multi-meaning, rare or archaic, multi-word), and whether it is a query problem or a model failure.
6. **Gradio demo.** Search box, top 10 words from the fine-tuned and untrained models side by side, and a tab with the results table (see below). It must start from the README instructions. Record a short backup video.
7. **System report.** Data and cleaning, methods, results, the casual-description results, the failure analysis, how WantWords was used (its hint ideas; its pretrained model is unavailable), deployment for a language-learning app (see below), and limitations (the 500 are dictionary-style, the casual set is small).
8. **Clean repository.** README with data, setup, how to run each step, results and limitations. Remove informal wording, fix references to missing files. Every script the README names is in the repo, and one command reruns the main results.

**Extras, in order**

9. **Rerank with hints** (see below). Report with and without, and each hint alone.
10. **Synonym-aware score.** Also count a WordNet synonym (same synset) as correct. Report strict, same-definition-text and WordNet scores.
11. **Self-reference rule on queries only.** Keep entries that mention their own word in the index, and leave them out of the queries only. Report vocabulary size and Recall@1, @10, @100 for both versions.
12. **Results by number of meanings.** The same numbers for test words with 2, 3-4 and 5+ meanings, with the query count in each group.

## How to fine-tune (goal 3)

- **Triplets, training words only.** Anchor: a definition of a word with two or more senses, with the word and its inflections masked out. Positive: another definition of the same word. Hard negative: a BM25 result from a different word.
- **Hard negatives.** BM25-search the training definitions, take the top 50, drop the anchor's own word and identical texts, and pick one from about ranks 5 to 50 (ranks 1 to 4 are often near-synonyms).
- **Train** with `sentence-transformers`, `MultipleNegativesRankingLoss` and "no duplicates" batches. One seed is enough.
- **Choose settings on validation queries** (built from validation words like the 500), not the test queries.
- **Leakage check.** No training text comes from a validation or test word or matches a validation or test query. Evaluation still searches the full dictionary.

## How to build the demo (goal 6)

- Load both models once at start. Encode all definitions once and save the vectors, so a search is one encode plus a dot product. Don't commit the vectors (about 400 MB); build or download them.
- Show each word once (its best-matching row) with that definition. Two lists side by side, 10 words each.
- Run with `python app.py`. `share=True` gives a temporary link, and a free Hugging Face Space a permanent one.

## What the deployment section covers (goal 7)

- **Speed:** median time per query on CPU and GPU, over 100 queries.
- **Index size:** rows × dimensions × bytes, and what half precision saves.
- **Adding words:** encode the new definition and append it, with no retraining.
- **Vocabulary gap:** the dictionary is from 1913 (archaic words, no "smartphone"), while learners type modern descriptions. Show the effect with the casual-description results.
- **Which model to ship:** accuracy against speed and size.
- **Limits:** near-synonyms count as wrong, and rank 1 is rare, so show the top 10.

## How to rerank with hints (goal 9)

Most of this is in `src/explore_pos.py` (`tag_class`, `classify`, `cue_of`, `role_of`).

- **Word class:** map the candidate's POS column to noun, verb, adjective or adverb. For the query, read the opening words ("To ..." verb, "One who ..." noun, "Having ..." adjective).
- **Ending:** -er, -or, -ist (person), -tion, -ment (action), -ous, -ful (adjective). "A person who ..." suggests a person ending.
- **Role:** "a person who" (agent), "the place where" (location), "that which is used" (instrument), matched to the candidate's class or ending.
- **Score:** model similarity + w_pos, w_end and w_role bonuses (1 if the hint agrees). Try 0, 0.02, 0.05, 0.1 for each weight on validation, keep the best, and re-sort the model's top 100.
