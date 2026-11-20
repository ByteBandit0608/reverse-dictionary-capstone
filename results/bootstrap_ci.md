# model table with 95% bootstrap intervals

from src/bootstrap_ci.py. 500 fixed queries, leave_one_out, strict scoring, 10000 resamples of the queries with replacement, seed 2026. the same resample is used for every method.

| method | R@1 | R@10 | R@100 | median rank |
|---|---|---|---|---|
| bm25 | 0.044 [0.026, 0.062] | 0.148 [0.118, 0.180] | 0.320 [0.280, 0.362] | 958 [509, 1000] |
| tfidf | 0.046 [0.028, 0.066] | 0.158 [0.126, 0.190] | 0.348 [0.306, 0.390] | 912 [440, 1000] |
| bge_base_en_v1_5 | 0.038 [0.022, 0.056] | 0.172 [0.140, 0.206] | 0.382 [0.340, 0.426] | 367 [184, 545] |
| all_minilm_l6_v2 | 0.044 [0.026, 0.062] | 0.164 [0.132, 0.198] | 0.380 [0.338, 0.422] | 290 [200, 442] |
| all_mpnet_base_v2 | 0.050 [0.032, 0.070] | 0.168 [0.136, 0.202] | 0.392 [0.348, 0.436] | 335 [214, 518] |

## paired difference against bm25

the difference is taken inside each resample, so it is tighter than comparing the intervals above by eye. an interval that contains 0 means the two methods cannot be told apart on these 500 queries.

| method | R@1 diff | R@10 diff | R@100 diff | median rank diff |
|---|---|---|---|---|
| tfidf | +0.002 [-0.014, +0.018] | +0.010 [-0.014, +0.034] | +0.028 [+0.010, +0.048] * | -46 [-286, +119] |
| bge_base_en_v1_5 | -0.006 [-0.024, +0.012] | +0.024 [-0.008, +0.056] | +0.062 [+0.020, +0.102] * | -591 [-767, -204] * |
| all_minilm_l6_v2 | +0.000 [-0.020, +0.020] | +0.016 [-0.016, +0.050] | +0.060 [+0.020, +0.102] * | -668 [-751, -252] * |
| all_mpnet_base_v2 | +0.006 [-0.014, +0.026] | +0.020 [-0.014, +0.056] | +0.072 [+0.028, +0.114] * | -622 [-742, -230] * |

\* the interval excludes 0.

notes

- the intervals show how much a number moves if a different 500 queries had been drawn. they do not include training noise or the choice of cleaning rules.
- all methods are scored on the same queries, so differences are paired.
- the median rank is capped at 1000, so a median of 1000 means more than half the queries are not found in the top 1000.