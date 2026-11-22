
# Stage 6: Error Analysis and Interpretation

## Project
Reverse Dictionary: Retrieving Words from Meaning Descriptions

## Objective
Investigate the retrieval behaviour of pretrained and fine-tuned
MPNet models and understand improvements and failure cases.

## Evaluation Setup
- Dictionary definitions: 135,907
- Unique words: 95,286
- Test queries: 500
- Embedding dimension: 768
- Evaluation: Strict and tolerant retrieval

## Aggregate Results

| Metric | Original MPNet | Fine-tuned MPNet |
|---|---:|---:|
| Strict Recall@1 | 5.0% | 5.8% |
| Strict Recall@10 | 16.8% | 21.0% |
| Strict Recall@100 | 39.2% | 41.8% |
| Tolerant Recall@10 | 17.6% | 21.8% |
| Tolerant Recall@100 | 39.8% | 42.6% |
| Strict Median Rank | 335 | 250 |

## Query-level Analysis

- Improved queries: 208 (41.6%)
- Worsened queries: 156 (31.2%)
- Unchanged queries: 136 (27.2%)

Ranks not found in the top 1000 were represented by 1001
for this query-level comparison.

## Statistical Analysis

- Sign test p-value: 0.00743
- Recall@10 exact McNemar-style test p-value: 0.02753
- Recall@100 exact McNemar-style test p-value: 0.24143

The Recall@10 result is not significant at 0.05 after
the Holm correction for the two Recall tests.

These tests should be interpreted cautiously because multiple
queries may correspond to the same dictionary word.

## Error Analysis Examples

### Tyne
Definition: To become lost; to perish.

- Original MPNet rank: Not found in top 1000
- Fine-tuned MPNet rank: 4

### Eleventh
Definition: Of or pertaining to the interval of the octave and the fourth.

Alternative musical definition:
The interval consisting of ten conjunct degrees;
the interval made up of an octave and a fourth.

- Original MPNet rank: 1
- Fine-tuned MPNet rank: 7976
- Original cosine similarity: 0.7978
- Fine-tuned cosine similarity: 0.4183

This illustrates that fine-tuning improves some retrieval cases
while worsening others.

## Conclusion

Fine-tuning MPNet improved aggregate retrieval performance
on the evaluated test set, particularly Recall@10 and median rank.
However, the model showed mixed query-level behaviour.

Further comparison with the BM25 baseline is required before
drawing conclusions about the overall best retrieval approach.

## Limitations

- Evaluation uses 500 frozen test queries.
- Query-level observations may not be statistically independent.
- Some definition senses are difficult to distinguish.
- Target-word masking was conservative and applied to only
  98 of 20,000 training anchors.
- Results are specific to this dataset and experimental setup.
