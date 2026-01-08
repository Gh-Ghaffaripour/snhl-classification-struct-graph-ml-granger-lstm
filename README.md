# Multimodal MRI Biomarkers for Hearing Loss  
**Granger Causality–Driven Network Analysis and LSTM Classification**

This repository provides a **fully reproducible implementation** of the analysis pipeline described in the accompanying manuscript:

*Multimodal MRI Biomarkers and Machine Learning for Objective Diagnosis of Hearing Loss*

All preprocessing, modeling, and evaluation steps are implemented **exactly as described in the Methods section**, with no additional heuristics or undocumented modifications.

---

## Overview of the Pipeline

The repository implements the following stages:

1. **Subject-level Granger causality analysis**
   - ADF stationarity testing
   - First-order differencing (if required)
   - Lag selection using AIC (1–10)
   - Fixed lag order **K = 5**
   - Pairwise Granger causality (F-test)
   - Benjamini–Hochberg FDR correction

2. **Group-level directed network construction**
   - Edge-frequency maps (NH / HL)
   - Thresholding at **frequency ≥ 0.70**
   - Directed graph visualization

3. **Granger-driven LSTM classification**
   - 20-fold stratified cross-validation
   - ROI selection **inside each training fold only**
   - Top-10 ROIs selected by mean out-degree (GC, train-only)
   - BiLSTM ×2 + Attention
   - Focal loss
   - SMOTE (training set only)
   - Test-Time Augmentation (TTA)

4. **Statistical evaluation**
   - AUROC, AUPRC, Balanced Accuracy
   - Threshold sweep for balanced accuracy (0.10–0.89)
   - Bootstrap 95% confidence intervals
   - Permutation tests

---

## Repository Structure

project/
├── README.md
├── requirements.txt
└── src/
├── 01_granger_subject_level.py
├── 02_granger_group_networks.py
├── 03_lstm_20fold_granger_driven.py
└── 04_lstm_metrics_ci_permutation.py

---

### ROI Time Series
- One CSV file per subject
- Filename: `subject_<ID>.csv`
- Shape: `ROI × Time` or `Time × ROI` (auto-detected)
- Number of ROIs: **116 (AAL atlas)**

### Labels
`data/labels.csv` must contain:

| column | description |
|------|-------------|
| subject_id | subject identifier matching `<ID>` |
| gp | group label (`nh` or `hl`) |

---

## Installation

Create a clean environment and install dependencies:

```bash
pip install -r requirements.txt
Required Python version: ≥ 3.9

Step-by-Step Execution
1️⃣ Subject-Level Granger Causality

python src/01_granger_subject_level.py
Outputs (outputs/granger_subject/):

gc_p_subject_<ID>.csv

gc_q_subject_<ID>.csv

gc_bin_fdr_subject_<ID>.csv

stationarity_<ID>.csv

lag_log.csv

This step implements:

ADF → diff(1) → ADF

AIC-based lag selection (1–10)

Final lag order fixed at K = 5

BH-FDR correction per subject

2️⃣ Group-Level Directed Networks
bash
Copy code
python src/02_granger_group_networks.py
Outputs (outputs/granger_group/):

NH_edge_frequency.csv

HL_edge_frequency.csv

NH_thr70_adj.csv

HL_thr70_adj.csv

NH_network_thr70.png

HL_network_thr70.png

Edges are retained if present in ≥70% of subjects within each group.

3️⃣ LSTM Classification (Granger-Driven)
bash
Copy code
python src/03_lstm_20fold_granger_driven.py
Key characteristics:

20-fold stratified CV

Granger causality computed inside each training fold

Top-10 ROIs selected by GC out-degree (train only)

2-layer BiLSTM + Attention

Focal loss

SMOTE applied only on training data

Test-Time Augmentation on test data

Balanced accuracy optimized via threshold sweep

Outputs (outputs/lstm/):

fold_metrics.csv

fold_predictions.csv

4️⃣ Statistical Evaluation (CI + Permutation Tests)
bash
Copy code
python src/04_lstm_metrics_ci_permutation.py
Reported statistics:

Mean AUROC / AUPRC with 95% bootstrap CI

Balanced accuracy (fold-specific thresholds)

Permutation test p-values

Reproducibility Notes
All random processes use fixed seeds.

ROI selection is strictly confined to training folds.

No information leakage occurs across folds.

All hyperparameters match those reported in the manuscript.

Citation
If you use this code, please cite the corresponding manuscript:

Multimodal MRI Biomarkers and Machine Learning for Objective Diagnosis of Hearing Loss.

Contact
For questions regarding implementation details or reproducibility, please contact the corresponding author.

License
This project is released for academic research use only.

yaml
Copy code
