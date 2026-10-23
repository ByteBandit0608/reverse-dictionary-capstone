# pretrained embeddings, no fine tuning

from src/run_dense.py. 500 fixed queries, cleaned corpus of 135,907 rows over 95,286 headwords, seed 2026. same two protocols as results/baselines.md.

| model | dim | protocol | R@1 | R@10 | R@100 | medrank | R@1 tol | own row at rank 1 |
|---|---|---|---|---|---|---|---|---|
| all_mpnet_base_v2 | 768 | in_dictionary | 0.992 | 1.000 | 1.000 | 1 | 1.000 | 99.2% |
| all_mpnet_base_v2 | 768 | leave_one_out | 0.050 | 0.170 | 0.394 | 319 | 0.060 | 0.0% |

notes

- bm25 on this same query set is R@1 0.046 and R@100 0.322 leave_one_out, from results/baselines.json. the rows above are meant to be read against that column, not in isolation.
- no prompt, no fine tuning, the definition text goes in as it stands. that is what he asked for this week. if a model ships a query prompt and does better with it, that is a finding for the report, not this run
- these are modern English models over 1913 definitions, which is why he expects a fine tuned model later. any gap here is the size of the domain shift, and it is the argument for CP4
