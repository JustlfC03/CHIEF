# CHIEF

This repository provides the reference implementation for **CHIEF**, a Chinese-language 3D vision–language model for non-contrast head CT interpretation in emergency settings. CHIEF is pretrained on paired volumetric CT examinations and radiology reports using bidirectional contrastive alignment, image-conditioned report generation, and representation decorrelation. The shared representation supports emergency triage classification, report generation, image-to-text retrieval, fine-tuned multilabel classification, and zero-shot abnormality detection.

The repository focuses on the core model, data, training, inference, and evaluation interfaces described in the study. Private clinical data, institution-specific splits, and trained model weights are not included.

<p align="center">
  <img src="assets/fig1_multicentre_cohort.png" width="760" alt="Multicentre head CT cohort construction and evaluation design">
</p>
<p align="center"><sub>Multicentre cohort construction, quality control, triage distributions, and centre-level distribution shifts.</sub></p>

Clinical data, private cohort splits, and trained model weights are not included in this release.

## Main capabilities

- **Emergency triage:** `negative`, `non-emergency-positive`, or `positive`.
- **Report generation:** 3D head CT to Chinese radiology report.
- **Image-to-text retrieval:** retrieval of semantically matched reports.
- **CQ500 classification:** fine-tuned 14-label abnormality classification.
- **Zero-shot detection:** language-guided abnormality scoring without task-specific fine-tuning.

## Installation

Python 3.10 is recommended.

```bash
conda create -n chief python=3.10 -y
conda activate chief
pip install -r requirements.txt
```

## Data preparation

The released pipeline uses two image stages:

```text
prepared NPY [32, 256, 256]
        ↓ piecewise intensity mapping and trilinear resize
model input [1, 128, 128, 128]
```

Convert a NIfTI, DICOM series, or other supported volume:

```bash
python preprocess.py input.nii.gz output.npy
```

`preprocess.py` is a reference volume converter. DICOM series selection, complete-head coverage, artefact review, de-identification, and image-report pairing must be completed before the manifest is created.

A manifest contains one examination per row. The minimum inference format is:

```csv
sample_id,image_path
case_0001,/path/to/case_0001.npy
```

Training manifests additionally contain the paired report or task label and a `split` column. The triage class order is fixed throughout the code and checkpoints:

```text
0 = negative
1 = non-emergency-positive
2 = positive
```

## Training

```bash
# Joint vision-language pretraining
python train.py --config configs/pretrain.yaml \
  --set data.manifest=/path/to/pretrain_manifest.csv

# Emergency triage
python train.py --config configs/triage.yaml \
  --init-checkpoint /path/to/pretrain_best.pt \
  --set data.manifest=/path/to/triage_manifest.csv

# Report generation
python train.py --config configs/report_generation.yaml \
  --init-checkpoint /path/to/pretrain_best.pt \
  --set data.manifest=/path/to/report_manifest.csv

# CQ500 multilabel classification
python train.py --config configs/cq500.yaml \
  --init-checkpoint /path/to/pretrain_best.pt \
  --set data.manifest=/path/to/cq500_manifest.csv
```

## Inference

Use the configuration matching the trained checkpoint:

```bash
python infer.py \
  --config configs/triage.yaml \
  --checkpoint /path/to/triage_best.pt \
  --output predictions/triage.csv \
  --set data.manifest=/path/to/test_manifest.csv
```

For report generation and CQ500, replace the configuration with `configs/report_generation.yaml` or `configs/cq500.yaml`. Retrieval and zero-shot settings are provided in `configs/inference.yaml` through `inference.mode`; the ordered 45-label zero-shot set is stored in `data/examples/zero_shot_labels_45.json`. Thresholded zero-shot or CQ500 outputs must reuse thresholds fitted on an independent validation cohort through `--threshold-validation-predictions` and `inference.thresholds_path`, rather than fitting thresholds on the test cohort.

## Evaluation

```bash
python evaluate.py \
  --task triage \
  --predictions predictions/triage_with_labels.csv \
  --output metrics/triage.json \
  --n-bootstrap 1000 \
  --seed 42
```

The evaluator accepts both the released column names and the original triage table names (`class_label`, `Pred`, `PROB_NEG`, `PROB_NON`, and `PROB_POS`). UTF-8 and GB18030 CSV files are supported.

The default confidence intervals use 1,000 examination-level bootstrap resamples. Triage G-mean is calculated per class as the geometric mean of sensitivity and specificity and then macro-averaged across classes for which the statistic is defined (all three classes in the internal cohort). For CIDEr, the corpus estimate and per-examination scores are computed once; the confidence interval bootstraps the fixed per-examination scores rather than recalculating CIDEr in every resample.

A complete command template for the released tasks is available in `scripts/reproduce_metrics.sh`.

## Repository structure

```text
configs/        reference training and inference configurations
src/chief/      model, data, training, inference, and evaluation code
scripts/        compact command templates
train.py        training entry point
infer.py        inference entry point
evaluate.py     metric and bootstrap evaluation
preprocess.py   volume conversion entry point
```

## Research-use statement

CHIEF is research software and is not approved for autonomous diagnosis, clinical triage, or treatment decisions. Prospective and site-specific validation is required before clinical use.

Citation information will be updated after publication. The code is released under [CC BY-NC-SA 4.0](LICENSE); third-party components retain their original licences as listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
