# minilm, pretrained embeddings, no fine tuning

from src/run_minilm.py. model sentence-transformers/all-MiniLM-L6-v2, max_seq_length 64, cosine over L2 normalised definition embeddings, 500 fixed queries, seed 2026. took 66.2s.

| method | protocol | R@1 | R@10 | R@100 | medrank | R@1 tol | R@10 tol | R@100 tol | own row at rank 1 |
|---|---|---|---|---|---|---|---|---|---|
| minilm | in_dictionary | 0.998 | 1.000 | 1.000 | 1 | 1.000 | 1.000 | 1.000 | 99.8% |
| minilm | leave_one_out | 0.028 | 0.132 | 0.284 | 1000 | 0.036 | 0.140 | 0.290 | 0.0% |

notes

- leave_one_out is the number to quote. in_dictionary keeps the query's own row in the index, so it is lookup and not understanding.
- 42.6% of queries have no other row for their word, so leave_one_out cannot reach 1.0. it is a floor.
- strict R@1 cannot pass 0.996 even with a perfect ranker (shared definitions).
- tol counts any word with the same definition string, a string rule and not wordnet.
- 486 of 500 queries missed at rank 1 under leave_one_out, see failures_minilm_leave_one_out.csv.
