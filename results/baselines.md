# baselines, bm25 and tfidf

from src/run_baselines.py. cleaned corpus is 135,907 rows over 95,286 headwords, 500 fixed queries from the test split, seed 2026. took 30.8s.

| method | protocol | R@1 | R@10 | R@100 | medrank | R@1 tol | own row at rank 1 |
|---|---|---|---|---|---|---|---|
| bm25 | in_dictionary | 0.990 | 1.000 | 1.000 | 1 | 0.996 | 99.0% |
| bm25 | leave_one_out | 0.046 | 0.152 | 0.322 | 945 | 0.056 | 0.0% |
| tfidf | in_dictionary | 0.992 | 1.000 | 1.000 | 1 | 0.998 | 99.2% |
| tfidf | leave_one_out | 0.048 | 0.162 | 0.350 | 884 | 0.058 | 0.0% |
| word_freq | in_dictionary | 0.000 | 0.000 | 0.000 | 1000 | 0.000 | - |
| random | in_dictionary | 0.000 | 0.002 | 0.004 | 1000 | 0.000 | - |

notes

- in_dictionary cant be quoted on its own. the query string is sitting in the index, so a lexical method gets it by matching text. that is lookup, not understanding the definition.
- leave_one_out takes that row out. 0.0% of these queries have no other row for their word at all, so that side cant reach 1.0 either and the number is a floor not a ceiling.
- tol column counts any word sharing the same definition string. string rule, not wordnet, so it undercounts. details in notes.md
- strict R@1 cant pass 0.995 on this set, because 1.0% of these queries share the exact string with another word. the file as a whole is 7.86% tied, but a tied string is nearly always a short one: 42.9% tied among definitions under 3 tokens against 1.2% among those of 3 or more. our query rule asks for 3 tokens, so the corpus figure and the query figure are different denominators and not a contradiction.
- word_freq is 0 because it is fitted on train words and every query here is a held out word. a row level split would have hidden that, which is the main argument for splitting by word.
