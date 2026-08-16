#!/usr/bin/env bash
set -euo pipefail

# Command template. Replace paths with aligned prediction files containing
# the required ground-truth columns. Threshold fitting must use validation data.

BOOTSTRAP=${BOOTSTRAP:-1000}
SEED=${SEED:-42}
mkdir -p metrics

python evaluate.py --task triage \
  --predictions predictions/triage_with_labels.csv \
  --output metrics/triage.json \
  --n-bootstrap "$BOOTSTRAP" --seed "$SEED"

python evaluate.py --task generation \
  --predictions predictions/reports_with_references.csv \
  --output metrics/generation.json \
  --bertscore-model bert-base-chinese \
  --n-bootstrap "$BOOTSTRAP" --seed "$SEED"

# The manuscript retrieval tables were computed from top-200 ranked reports
# using the explicitly recorded fuzzy relevance rule.
python evaluate.py --task retrieval \
  --predictions predictions/retrieval_ranked.csv \
  --reference-column gt_answer \
  --candidates-column similar_valid_gt \
  --match-mode fuzzy --fuzzy-threshold 0.92 --min-match-length 8 \
  --ranked-miss-policy after-list \
  --output metrics/retrieval.json \
  --n-bootstrap "$BOOTSTRAP" --seed "$SEED"

python evaluate.py --task cq500 \
  --predictions predictions/cq500_test_with_labels.csv \
  --threshold-validation-predictions predictions/cq500_val_with_labels.csv \
  --threshold-objective youden \
  --save-thresholds metrics/cq500_thresholds.json \
  --output metrics/cq500.json \
  --n-bootstrap "$BOOTSTRAP" --seed "$SEED"

python evaluate.py --task zero_shot \
  --predictions predictions/zero_shot_test_with_labels.csv \
  --labels-json data/examples/zero_shot_labels_45.json \
  --threshold-validation-predictions predictions/zero_shot_val_with_labels.csv \
  --threshold-objective youden \
  --save-thresholds metrics/zero_shot_thresholds.json \
  --output metrics/zero_shot.json \
  --n-bootstrap "$BOOTSTRAP" --seed "$SEED"
