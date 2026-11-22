# Next goals: Reverse Dictionary

Decide among yourselves who does what. The frozen test set, the model table with intervals and the fine-tuning results are done. What remains is making the fine-tuning reproducible, testing casual descriptions, and finishing the demo, the presentation and the repository. Goals 1 to 5 complete the project. Do the extras only with time left, in order.

**Required**

1. **Make the fine-tuning reproducible and put it in the main table.** Commit the training and evaluation scripts, and say in the README where the fine-tuned model is stored (a shared link, not GitHub). Merge the two folders `Stage6_finetuning` and `stage6_finetuning` into one. Add the fine-tuned model as a row in the main table, with a 95% interval, and give the paired difference against the untrained MPNet and against BM25.
2. **Casual-description queries.** Build 100 casual descriptions of test words (half with one meaning and half with several), from an AI model (give it only the word and ask how a person would describe it without using the word), a real description dataset (for example the human descriptions in the multi-channel model's data), or both. Drop any that contain the target word, save each with its target, and read 20 by hand. Keep them out of all training and settings. Search the full dictionary including the target word's own definitions, because that is what a real user meets, and say so in the presentation. Compare the untrained and fine-tuned models and BM25 with the results on the 500, with 95% intervals. Say where the descriptions came from.
3. **Gradio demo.** Search box, top 10 words from the fine-tuned and untrained models side by side, and a tab with the results table (see below). It must start from the README instructions. Record a short backup video.
4. **Presentation.** About 15 slides: the problem, the data and cleaning, how the test works, the methods, the results table with intervals, the casual-description results, the demo, deployment for a language-learning app (speed, index size, adding new words, the 1913 vocabulary gap), and the limitations.
5. **Clean repository.** README with data, setup, how to run each step, results and limitations. Remove informal wording (for example in `results/embeddings.md` and `notes.md`). Fix references to missing files. Every script the README names is in the repo, and one command reruns the main results.

**Extras, in order**

6. **Rerank with hints** (see below). Report with and without, and each hint alone.
7. **Synonym-aware score.** Also count a WordNet synonym (same synset) as correct. Report strict, same-definition-text and WordNet scores.
8. **Results by number of meanings.** The same numbers for test words with 2, 3-4 and 5+ meanings, with the query count in each group.

## How to build the demo (goal 3)

- Load both models once at start. Encode all definitions once and save the vectors, so a search is one encode plus a dot product. Don't commit the vectors (about 400 MB); build or download them.
- Show each word once (its best-matching row) with that definition. Two lists side by side, 10 words each.
- Run with `python app.py`. `share=True` gives a temporary link, and a free Hugging Face Space a permanent one.

## How to rerank with hints (goal 6)

Most of this is in `src/explore_pos.py` (`tag_class`, `classify`, `cue_of`, `role_of`).

- **Word class:** map the candidate's POS column to noun, verb, adjective or adverb. For the query, read the opening words ("To ..." verb, "One who ..." noun, "Having ..." adjective).
- **Ending:** -er, -or, -ist (person), -tion, -ment (action), -ous, -ful (adjective). "A person who ..." suggests a person ending.
- **Role:** "a person who" (agent), "the place where" (location), "that which is used" (instrument), matched to the candidate's class or ending.
- **Score:** model similarity + w_pos, w_end and w_role bonuses (1 if the hint agrees). Try 0, 0.02, 0.05, 0.1 for each weight on validation, keep the best, and re-sort the model's top 100.
