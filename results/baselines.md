# baselines, bm25 and tfidf

from src/run_baselines.py. cleaned corpus is 170,140 rows over 107,032 headwords, 500 fixed queries from the test split, seed 2026. took 23.7s.

| method | protocol | R@1 | R@10 | R@100 | medrank | R@1 tol | own row at rank 1 |
|---|---|---|---|---|---|---|---|
| bm25 | in_dictionary | 0.998 | 1.000 | 1.000 | 1 | 1.000 | 99.8% |
| bm25 | leave_one_out | 0.032 | 0.132 | 0.264 | 1000 | 0.040 | 0.0% |
| tfidf | in_dictionary | 0.998 | 1.000 | 1.000 | 1 | 1.000 | 99.8% |
| tfidf | leave_one_out | 0.022 | 0.120 | 0.274 | 1000 | 0.030 | 0.0% |
| word_freq | in_dictionary | 0.000 | 0.000 | 0.000 | 1000 | 0.000 | - |
| random | in_dictionary | 0.000 | 0.000 | 0.004 | 1000 | 0.000 | - |

notes

- in_dictionary cant be quoted on its own. the query string is sitting in the index, so a lexical method gets it by matching text. that is lookup, not understanding the definition.
- leave_one_out takes that row out. 42.6% of these queries have no other row for their word at all, so that side cant reach 1.0 either and the number is a floor not a ceiling.
- tol column counts any word sharing the same definition string. string rule, not wordnet, so it undercounts. details in notes.md
- strict R@1 cant pass 0.996 on this set even with a perfect ranker, because 10.25% of rows share a definition with another word.
- word_freq is 0 because it is fitted on train words and every query here is a held out word. a row level split would have hidden that, which is the main argument for splitting by word.
