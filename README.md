# XAI-ACF-MAAR: Explainable ACF-Based Drift Diagnosis for Data Stream Analysis

[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue)](https://www.python.org/)
[![License MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Under%20Review-orange)]()
[![PyCaret](https://img.shields.io/badge/PyCaret-4.0-1abc9c)](https://pycaret.org/)
[![River](https://img.shields.io/badge/River-0.21.2-4B8BBE)](https://riverml.xyz/)
[![JAIR](https://img.shields.io/badge/Journal-JAIR%202026-red)]()

> **Paper:** *Explainable ACF-Based Drift Diagnosis for Data Stream Analysis*  
> **Authors:** Shahad Palathingal and Ebin Deni Raj  
> **Affiliation:** Indian Institute of Information Technology, Kottayam, Kerala, India  
> **Journal:** Journal of Artificial Intelligence Research (JAIR), Vol. 4, Article 6, August 2026

---

## 🚀 Run in Google Colab

| Dataset | Notebook | Description |
|---|---|---|
| **Household** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1U8REo0b37cPXKIV71BGNQcB2q0Xmo7Pk) | Full XAI-ACF-MAAR pipeline: lag decomposition, taxonomy classification, NL diagnostic reports, ablation study |
| **Elec2** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1cQXwZmL4vs2I6mIxhzVaxWWQy_-DwFjV) | Cross-dataset validation: Australian electricity market (30-min intervals) |
| **Tetouan** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RMsknHx72GI8dMRfRTFd2qsTs-STHQIf) | Cross-dataset validation: Tetouan City power consumption (10-min intervals) |

---

## Overview

This repository implements **XAI-ACF-MAAR**, an explainable diagnostic framework that transforms
opaque sequential change-point detection signals into rich, human-readable diagnostic reports.

The framework answers three questions that existing drift detectors cannot:

- **What** has structurally changed in the data stream?
- **Which** autocorrelation lags are responsible?
- **What type** of structural shift has taken place?

---

## Three-Layer XAI Architecture

### Layer 1 — Lag Importance Decomposition

Decomposes the ACF distance $D^2_{\text{ACF}}$ into per-lag squared contributions $\Delta_k$,
with Bonferroni-corrected significance testing at level $\alpha/L$.
Identifies the exact lags responsible for each structural change.

### Layer 2 — Six-Category Structural Change Taxonomy

Automatically classifies detected changes into six interpretable categories:

| Category | Description |
|---|---|
| **Periodicity Shift** | ACF peaks appear or disappear — periodic pattern changed |
| **Memory Contraction** | Effective dependency range shortened |
| **Memory Expansion** | Effective dependency range lengthened |
| **Dominant Lag Migration** | Primary temporal dependency moved to different time scale |
| **Correlation Collapse** | Broad autocorrelation loss across all lags |
| **Correlation Emergence** | New autocorrelation structure appeared across all lags |

### Layer 3 — Five-Section NL Diagnostic Report

Template-based natural language reporter with domain-contextual interpretation:

```
====================================================================
  STRUCTURAL DRIFT DIAGNOSTIC REPORT
====================================================================
  Dataset : Household

[1] DRIFT SUMMARY
    Time index   : t = 897120
    ACF distance : D = 4.0160  (threshold τ = 2.5671,  p < 0.001)
    Severity     : SEVERE  (D/τ ratio = 1.56)

[2] LAG-LEVEL ATTRIBUTION
    Significant lags : |K*| = 59 of 60 monitored  (Bonferroni α/L = 0.00083)
    Primary contributors (top-3 by ωk):
      Lag  19 : ω = 0.0310  ρ̂: +0.5575 → -0.1500  ↓ (decreased)
      Lag  18 : ω = 0.0310  ρ̂: +0.5687 → -0.1383  ↓ (decreased)
      Lag  20 : ω = 0.0310  ρ̂: +0.5458 → -0.1607  ↓ (decreased)
    Cumulative top-3 importance : 0.093 of total D²_ACF

[3] STRUCTURAL CHANGE CLASSIFICATION
    Detected type(s) : Correlation Collapse
    ▶ Correlation Collapse
      Mean |ρ̂| collapsed: 0.4962 → 0.1075  (78.3% reduction)

[4] OPERATIONAL CONTEXT
    Severe autocorrelation loss (78.3%) suggests transition to
    near-random consumption — possible unoccupied or vacation
    period near Sat.

[5] ADAPTATION ACTION
    Tier 1 : Window resize (151 → 8 obs), model retained.
    Rationale: Prediction error within tolerance δ — existing
    model handles new regime adequately.
====================================================================
```

---

## Key Results

### Household Dataset

| Metric | Value |
|---|---|
| Taxonomy Macro F1 | **0.89** |
| Lag Identification F1 | **0.89** |
| Event-level Accuracy | **8/9 = 88.9%** |
| Streaming MAE | **0.055 kW** |
| Streaming R² | **0.954** |
| Amortised Overhead | **1.71%** (target < 7%) |
| Drift rate | **5.3%** (matches α = 0.05 exactly) |
| Inter-annotator agreement | **κ = 0.84** |

### Ablation Study (Household Dataset)

| Configuration | Type F1 | Lag F1 | Time (ms) |
|---|---|---|---|
| Full framework | 0.89 | 0.89 | 1.22 |
| − Lag decomposition | 0.71 | — | 0.46 |
| − Taxonomy | — | 0.89 | 0.76 |
| − NL reporting | 0.89 | 0.89 | 0.86 |
| − All interpretability | — | — | 0.00 |

### Computational Overhead (Household Dataset, median of 200 runs)

| Component | Time (ms) | Role |
|---|---|---|
| ACF distance + threshold | 1.75 | Base monitoring |
| Lag importance decomposition | 0.40 | Interpretability |
| Structural change taxonomy | 0.46 | Interpretability |
| NL report generation | 0.36 | Interpretability |
| **Total interpretability layer** | **1.22** | Sum of 3 above |
| **Amortised per observation** | **0.065** | At 5.3% drift rate |
| **Amortised overhead %** | **1.71%** | Of base monitoring |

---

## Setup

### Requirements

```
python >= 3.12
pycaret == 4.0.0a8
river == 0.21.2
scikit-learn >= 1.3.0
numpy >= 1.24.0
pandas >= 2.0.0
scipy >= 1.12.0
statsmodels >= 0.14.0
matplotlib >= 3.7.0
```

### Install

```bash
pip install pycaret==4.0.0a8 -q
pip install river==0.21.2 --no-deps -q
pip install statsmodels scipy scikit-learn -q
```

> **Note:** PyCaret 4.0.0a8 is required for Python 3.12 compatibility.
> PyCaret 3.x does not support Python 3.12 and requires complex patching.

---

## Quick Start (Google Colab)

1. Open the Household notebook via the badge above
2. Run **Cell 1a** — installs packages
3. **Runtime → Restart session** (mandatory after install)
4. Run **Cell 1c** — imports
5. Run **Cell 1d** — mount Google Drive
6. Run all remaining cells sequentially

Expected total runtime: **15–30 minutes** on Colab free tier.

---

## Repository Structure

```
XAI-ACF-MAAR/
├── notebooks/
│   ├── XAI_ACF_MAAR_Household.ipynb    # Full pipeline — Household dataset
│   ├── XAI_ACF_MAAR_Elec2.ipynb        # Cross-dataset — Elec2
│   └── XAI_ACF_MAAR_Tetouan.ipynb      # Cross-dataset — Tetouan
│
├── src/
│   ├── acf_utils.py                    # ACF computation, Bartlett bound, window sizing
│   ├── lag_decomposition.py            # Lag contribution, normalised importance, K*
│   ├── taxonomy.py                     # Six-category structural change classifier
│   ├── nl_reporter.py                  # NL report generator (5-section, domain context)
│   ├── automl_engine.py                # PyCaret 4.0 AutoML wrapper
│   └── streaming_pipeline.py           # Integrated XAI-ACF-MAAR pipeline
│
├── results/
│   ├── household_diagnostics.csv       # Per-event diagnostic outputs
│   ├── household_diagnostics.json      # Full structured diagnostic data
│   ├── household_nl_reports.txt        # All 9 NL diagnostic reports
│   └── figures/                        # Publication figures (PDF)
│       ├── fig_omega_heatmap.pdf
│       ├── fig_taxonomy_confusion.pdf
│       ├── fig_example_reports_acf.pdf
│       ├── fig_streaming_performance.pdf
│       ├── fig_acf_drift_timeline.pdf
│       └── fig_lag_importance_event1-9.pdf
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Core Functions

```python
# Lag importance decomposition
from src.lag_decomposition import (
    lag_contribution,            # Δk = (ρ̂_ref_k - ρ̂_cur_k)²
    normalised_lag_importance,   # ωk = Δk / D²_ACF
    significant_lag_set,         # K* with Bonferroni correction at α/L
    top_contributing_lags        # Ranked top-k by ωk
)

# Structural change taxonomy
from src.taxonomy import classify_structural_change

# NL diagnostic report
from src.nl_reporter import (
    generate_diagnostic_report_v2,   # 5-section report with domain context
    infer_domain_context              # Rule-based operational interpretation
)

# Example usage
categories, evidence = classify_structural_change(
    acf_ref, acf_cur, n=1440,
    alpha=0.05, gamma=1, eta=0.5
)

report, report_dict = generate_diagnostic_report_v2(
    time_index=191520,
    d_acf=2.343,
    tau=2.567,
    omega=omega,
    delta_k=delta_k,
    K_star=K_star,
    acf_ref=acf_ref,
    acf_cur=acf_cur,
    categories=categories,
    evidence=evidence,
    n=1440,
    adaptation_tier=1,
    window_change=(117, 151),
    dataset_name='Household'
)

print(report)
```

---

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `alpha` | 0.05 | Significance level for ACF tests and Bonferroni correction |
| `L` | 60 | Maximum lags monitored |
| `c` | 3 | Consecutive-below-threshold counter for window sizing |
| `gamma` | 1 | Memory contraction/expansion tolerance (lags) |
| `eta` | 0.5 | Collapse ratio threshold |
| `emergence_threshold` | 1.85 × ρ̄_ref | Empirically calibrated emergence threshold |

---

## Datasets

| Dataset | Instances | Frequency | Target | Source |
|---|---|---|---|---|
| Household | 2,075,259 | 1 min | Global active power (kW) | [UCI](https://archive.ics.uci.edu/dataset/235) |
| Tetouan | 52,417 | 10 min | Zone 1 power consumption | [UCI](https://archive.ics.uci.edu/dataset/849) |
| Elec2 | 45,312 | 30 min | Electricity pricing | [UCI](https://archive.ics.uci.edu/dataset/321) |

---

## How It Differs from Feature Importance Methods (SHAP etc.)

Existing interpretability methods like SHAP explain **model predictions** — which input
features drive a model's output at a given time step.

This framework explains **structural changes in the data-generating process itself** —
which temporal lags have shifted in the ACF profile between two windows, independently
of any predictive model.

A process can undergo structural change that is invisible to feature importance methods
(because no single prediction is anomalous) yet devastating to forecasting performance
(because the model's learned lag dependencies no longer match the current temporal
structure). The lag importance decomposition is therefore a **process diagnostic tool**,
not a model explainability tool.

---

## Citation

```bibtex
@article{palathingal2026xai,
  title   = {Explainable ACF-Based Drift Diagnosis for Data Stream Analysis},
  author  = {Palathingal, Shahad and Raj, Ebin Deni},
  journal = {Journal of Artificial Intelligence Research},
  volume  = {4},
  number  = {6},
  year    = {2026},
  doi     = {10.1613/jair.1.xxxxx}
}
```

---

## Contact

- **Shahad Palathingal** — shahadphd2019@iiitkottayam.ac.in
- **Ebin Deni Raj** — ebindeniraj@iiitkottayam.ac.in

Indian Institute of Information Technology, Kottayam, Kerala, India

---

## License

MIT License — see [LICENSE](LICENSE) for details.
