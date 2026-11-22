# Stage 6: MPNet Fine-tuning

This directory contains the experimental results of our
task-specific MPNet fine-tuning for the Reverse Dictionary project.

## Contents

- STAGE_6_REPORT.md
- evaluation_results.json
- model_comparison.csv
- query_level_analysis.csv
- statistical_analysis.json
- training_triplets.csv

## Experiment Details

- Base model: all-mpnet-base-v2
- Training triplets: 20,000
- Validation triplets: 2,000
- Test queries: 500
- Training epochs: 2

The fine-tuned model and large embedding matrices will be
stored separately as large artifacts.
