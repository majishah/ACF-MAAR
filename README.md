# ACF-MAAR: Autocorrelation-Guided Multi-Window and AutoML-based Adaptable Regression

## Leveraging Autocorrelation for Adaptive Multi-Window Regression in Streaming Data

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Research%20Prototype-orange)](https://img.shields.io/badge/Status-Research%20Prototype-orange)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E)](https://scikit-learn.org/)
[![PyCaret](https://img.shields.io/badge/PyCaret-3.0%2B-1abc9c)](https://pycaret.org/)
[![River](https://img.shields.io/badge/River-0.21%2B-4B8BBE)](https://riverml.xyz/)

### 🚀 Interactive Notebooks — Run in Google Colab

> **Reproduce all experiments directly in your browser — no local setup required.**

| Dataset | Notebook | Description |
| --- | --- | --- |
| **Household** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1rC9CooFtzufKxCWUy8sPT4UdUq9_1cb8#scrollTo=dBGEWL5eUst1) | Full ACF-MAAR pipeline: window sizing, drift detection, adaptive model selection, ablation study |
| Elec2 | *Coming soon* | Australian Electricity Market evaluation |
| Korea Mfg. | *Coming soon* | South Korean Manufacturing Facilities evaluation |
| **Tetouan** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RMsknHx72GI8dMRfRTFd2qsTs-STHQIf#scrollTo=1quRau-tFVoh) | Cross-dataset validation: Tetouan City power consumption (10-min intervals) |
| Uruguay | *Coming soon* | Uruguay Household Consumption evaluation |

---

### Overview

This repository hosts the implementation and experimental validation of **ACF-MAAR (Autocorrelation-Guided Multi-Window and AutoML-based Adaptable Regression)**, a streaming forecasting framework that leverages autocorrelation function (ACF) analysis as a principled, unifying signal for three interdependent challenges in non-stationary data streams:

1. **Data-driven window sizing** — automatically matching the training window to the temporal dependency range of the current data regime.
2. **Structure-aware concept drift detection** — capturing shifts in periodicity, correlation decay, and temporal dependencies that mean-based detectors miss.
3. **Autocorrelation-triggered model adaptation** — replacing empirical thresholds with statistically derived triggers via a two-tier policy.

> **Paper:** *ACF-MAAR: Leveraging Autocorrelation for Adaptive Multi-Window Regression in Streaming Data*
> **Authors:** Shahad Palathingal and Ebin Deni Raj
> **Affiliation:** Department of Computer Science Engineering, Indian Institute of Information Technology, Kottayam, Kerala, India

---

### Key Technical Contributions

1. **ACF-Driven Adaptive Window Sizing:**
   Computes the streaming ACF over the most recent observations and sets the training window size to the lag at which autocorrelation first drops below the Bartlett significance bound (±z<sub>α/2</sub>/√n). This yields a window length inherently matched to the temporal dependency range of the current data regime, adapting automatically as the stream evolves.

2. **ACF-Based Structural Drift Detection:**
   Monitors changes in the ACF profile across consecutive windows using a Euclidean distance metric. Unlike mean-based detectors (ADWIN, Page-Hinkley) that respond only to level shifts, this approach detects changes in periodicity, decay rate, and correlation structure — capturing a broader class of concept drifts.

3. **Two-Tier Adaptation Policy:**
   Distinguishes between minor structural shifts (Tier 1: window resizing only) and major shifts (Tier 2: full model replacement via asynchronous AutoML), reducing unnecessary model replacements by 65% while maintaining forecasting accuracy.

4. **Comprehensive Empirical Evaluation:**
   Over **344+ experimental configurations** across 5 real-world energy datasets, 7 baselines, and 8 performance metrics with systematic ablation study.

---

### Performance Highlights

#### Prediction Accuracy (Household Dataset)

| Metric | ACF-MAAR | Best Fixed Window | Improvement |
| --- | --- | --- | --- |
| **MAE** | **0.074** | 0.081 | 8.6% ↓ |
| **R²** | **0.958** | 0.952 | +0.6 pp |
| **MAPE** | **5.20%** | 11.2% | 53.6% ↓ |
| **Prediction Latency** | **< 1 ms** | < 1 ms | — |

#### Cross-Dataset Generalization

| Dataset | MAE Improvement | R² (ACF-MAAR) |
| --- | --- | --- |
| Household | 8.6% | 0.958 |
| Elec2 | 20.3% | 0.891 |
| Korea Mfg. | 20.3% | 0.932 |
| Tetouan | 21.2% | 0.918 |
| Uruguay | 17.6% | 0.905 |

#### Drift Detection Quality (24-hour segment, 7 known structural changes)

| Detector | Structural Drifts Found | False Alarms |
| --- | --- | --- |
| ADWIN (δ=0.02) | 3 / 7 | 8 |
| Page-Hinkley (θ=5) | 2 / 7 | 5 |
| KSWIN (α=0.01) | 4 / 7 | 11 |
| **ACF-MAAR** | **7 / 7** | **1** |

#### Ablation Study

| Configuration | MAE | R² | Model Replacements |
| --- | --- | --- | --- |
| **Full ACF-MAAR** | **0.074** | **0.958** | **6** |
| − ACF window (fixed 60) | 0.085 | 0.945 | 8 |
| − ACF drift (ADWIN only) | 0.089 | 0.938 | 14 |
| − ACF trigger (fixed 10%) | 0.081 | 0.952 | 11 |
| − All ACF (Fixed-MAAR) | 0.108 | 0.920 | 17 |

---

### Repository Structure

```
ACF-MAAR/
├── datasets/                          # Benchmark datasets (Raw and Processed)
│   ├── household/                     # Individual Household Electric Power Consumption
│   │   └── household_power.csv
│   ├── elec2/                         # Australian Electricity Market
│   │   └── elec2.csv
│   ├── korea_mfg/                     # South Korean Manufacturing Facilities
│   │   └── korea_manufacturing.csv
│   ├── tetouan/                       # Power Consumption of Tetouan City
│   │   └── tetouan_power.csv
│   └── uruguay/                       # Uruguay Household Consumption
│       └── uruguay_consumption.csv
│
├── src/                               # Core Framework Source Code
│   ├── acf_analysis.py                # ACF computation, Bartlett bound, window sizing
│   ├── drift_detection.py             # ACF-based structural drift detector + ADWIN hybrid
│   ├── adaptation.py                  # Two-tier adaptation policy (Tier 1 & Tier 2)
│   ├── automl_engine.py               # Asynchronous AutoML model selection via PyCaret
│   ├── streaming_pipeline.py          # Main ACF-MAAR streaming pipeline (Algorithm 1)
│   └── utils.py                       # Helper functions, metrics, logging
│
├── baselines/                         # Baseline Implementations
│   ├── linear_regression.py           # Linear Regression baseline
│   ├── stream_models.py               # HTR, ARF, OBR, LBR, KNN (via River)
│   ├── lstm_baseline.py               # LSTM-1 and LSTM-2 configurations
│   └── fixed_window.py                # Fixed-window AutoML (sizes 30, 45, 90, 120)
│
├── experiments/                       # Experimental Validation Scripts
│   ├── exp_window_sizing.py           # §4.3: ACF-adaptive vs. fixed window evaluation
│   ├── exp_drift_detection.py         # §4.3: Drift detection quality comparison
│   ├── exp_adaptive_selection.py      # §4.4: Adaptive model selection under drift
│   ├── exp_baseline_comparison.py     # §4.5: Full baseline comparison suite
│   ├── exp_cross_dataset.py           # §4.6: Cross-dataset generalization
│   ├── exp_ablation.py                # §4.7: Ablation study (5 configurations)
│   └── exp_graphical_analysis.py      # §4.8: Generate R² and MAPE comparison plots
│
├── results/                           # Pre-computed Results and Figures
│   ├── tables/                        # CSV exports of all result tables
│   ├── figures/                       # Generated plots (Figs. 5–10)
│   └── logs/                          # Drift detection and adaptation logs
│
├── notebooks/                         # Jupyter Notebooks for Exploration
│   ├── 01_acf_visualization.ipynb     # ACF profile visualization across regimes
│   ├── 02_drift_analysis.ipynb        # Drift detection comparative analysis
│   └── 03_results_summary.ipynb       # Results aggregation and plotting
│
├── run_all_experiments.py             # Master script to reproduce all results
├── requirements.txt                   # Project dependencies
├── LICENSE                            # MIT License
└── README.md                          # This file
```

---

### System Architecture

The ACF-MAAR framework operates in two phases:

**Phase 1 — Initialization and ACF Calibration:**
1. **Data Ingestion:** Buffer fills with the first *N* observations from the stream.
2. **Pre-processing:** Missing value imputation, normalization, and transformation.
3. **ACF Computation & Window Sizing:** Autocorrelation profile determines adaptive window size *n\** via the Bartlett significance bound.
4. **Initial Model Selection:** AutoML evaluates candidate models and selects the best performer.
5. **Baseline Error Calibration:** Establishes the adaptation threshold *δ* from the initial prediction error distribution.

**Phase 2 — Continuous Prediction and Adaptation:**
1. **Hybrid Drift Monitoring:** ACF distance (structural) + ADWIN (mean-shift) at each window boundary.
2. **Tier 1 — Window Resizing:** Recomputes window size from updated ACF profile if drift detected.
3. **Tier 2 — Model Replacement:** Triggers asynchronous AutoML if prediction error exceeds *δ*.
4. **Deployment:** New model replaces current; reference ACF profile and error bound are recalibrated.

---

### Installation and Setup

**Prerequisites:**
- Python 3.10 or higher
- 16 GB RAM (recommended for large datasets)

**1. Clone the Repository:**

```bash
git clone https://github.com/majishah/ACF-MAAR.git
cd ACF-MAAR
```

**2. Initialize Virtual Environment:**

```bash
python -m venv venv
source venv/bin/activate      # Linux/Mac
# venv\Scripts\activate       # Windows
```

**3. Install Dependencies:**

```bash
pip install -r requirements.txt
```

---

### Reproducibility Guide (Google Colab)

The Colab notebook is validated on **Google Colab with Python 3.12** (as of May 2025). PyCaret 3.3.2 has known compatibility issues with Colab's default package versions. The notebook's setup cells handle all patches automatically, but the following documents every issue and fix for full transparency.

#### Quick Start

1. Open the Colab notebook via the badge above.
2. Run **Cell 1a** (package installs).
3. **Runtime → Restart session** (mandatory — see explanation below).
4. Run **Cell 1b** (patches) → **Cell 1c** (imports) → **Cell 1d** (mount drive).
5. Execute all remaining cells sequentially, or use **Runtime → Run all**.

#### Why `--no-deps`?

PyCaret 3.3.2 declares strict upper bounds on core dependencies (`scipy<=1.11.4`, `pandas<2.2.0`, `matplotlib<3.8.0`) that conflict with Colab's pre-installed versions. Installing with `--no-deps` prevents pip from downgrading Colab's core stack, which would break other pre-installed packages. The four patches below resolve the resulting source-level incompatibilities.

#### Compatibility Patches

The notebook applies four source-level patches before importing any library. Each targets a specific incompatibility between PyCaret 3.3.2 and the Colab Python 3.12 environment:

| # | File Patched | Root Cause | Fix Applied |
|---|---|---|---|
| 1 | `pycaret/__init__.py` | PyCaret raises `RuntimeError` on Python ≥3.12 (officially supports 3.9–3.11 only). | Rewrite `__init__.py` to remove the version gate, retaining only version metadata and exports. |
| 2 | `pycaret/internal/memory.py` | `FastMemory` passes `bytes_limit` to `joblib.Memory.__init__()`, but this parameter was removed in joblib ≥1.4. `FastMemory.__del__()` also crashes with `AttributeError: 'FastMemory' object has no attribute 'store_backend'`. | Replace `return FastMemory(...)` with `return None` in `get_memory()`, disabling PyCaret's optional result caching. No impact on model training correctness. |
| 3 | `pycaret/internal/pipeline.py` | `from sklearn.utils import _print_elapsed_time` fails because this private utility was removed in scikit-learn ≥1.4. PyCaret calls it as a context manager (`with _print_elapsed_time(...)`). | Inject a no-op `@contextmanager` definition directly into `pipeline.py` after the import block. |
| 4 | `scikitplot/*.py` | `from scipy import interp` was removed in SciPy ≥1.12. PyCaret depends on scikit-plot, which still imports via this path. | Replace with `from numpy import interp` in all affected scikit-plot files (`metrics.py`, `plotters.py`). |

> **Note:** These patches modify installed package files within Colab's ephemeral runtime. They do not alter the repository source code and are re-applied automatically on each fresh session.

#### Why a Runtime Restart Is Required

Python caches imported modules in `sys.modules`. Patches 1–4 modify `.py` files on disk, so the interpreter must be restarted for the patched code to load. Without a restart, the original (unpatched) bytecode remains in memory, and the same errors will persist despite the fixes being written to disk.

#### Verified Environment

| Package | Version | Notes |
|---|---|---|
| Python | 3.12.x | Colab default (May 2025) |
| PyCaret | 3.3.2 | Installed with `--no-deps`; 4 source patches applied |
| scikit-learn | 1.6.x | Colab default; compatible after `pipeline.py` patch |
| joblib | 1.4.x | Colab default; compatible after `memory.py` patch |
| SciPy | 1.12+ | Colab default; compatible after scikit-plot patch |
| River | 0.21.2 | Installed with `--no-deps` |
| statsmodels | 0.14+ | Colab default |
| xgboost | 1.7+ | Colab default |

#### Expected Runtime

| Section | Approximate Time |
|---|---|
| Setup and Data Loading | < 1 minute |
| Initial AutoML Model Selection (§5) | 2–5 minutes |
| ACF-MAAR Streaming Pipeline (§8) | 5–15 minutes |
| Stream-Specific Baselines (§9–10) | 3–5 minutes |
| Fixed Window Comparison (§15) | 10–20 minutes |
| Ablation Study (§16) | 20–40 minutes |
| **Total** | **~45–90 minutes** |

#### Troubleshooting

| Symptom | Cause | Solution |
|---|---|---|
| `TypeError: Memory.__init__() got an unexpected keyword argument 'bytes_limit'` | Runtime not restarted after Cell 1a | Runtime → Restart session, then run from Cell 1b |
| `TypeError: 'NoneType' object does not support the context manager protocol` | `pipeline.py` patch not loaded | Runtime → Restart session, then run from Cell 1b |
| `NameError: name 'series_to_supervised' is not defined` | Cells executed out of order | Run all cells sequentially from top, or use Runtime → Run all |
| `NameError: name 'stream_mae' is not defined` | Streaming pipeline cell (§8) was skipped | Run Section 8 (both sync and async cells) before Sections 15–16 |
| `IndentationError` in `pycaret/__init__.py` | Corrupted from a prior partial patch | Re-run Cell 1b to overwrite cleanly |

---

### Usage


#### Running ACF-MAAR on Custom Data

```python
from src.streaming_pipeline import ACFMAARPipeline

# Initialize the pipeline
pipeline = ACFMAARPipeline(
    alpha=0.05,           # Significance level
    max_lags=60,          # Maximum ACF lags
    buffer_size=10080,    # Buffer size (e.g., 1 week at 1-min intervals)
    consecutive_c=3       # Consecutive-below-threshold counter
)

# Run on your data stream
results = pipeline.run(data_stream="path/to/your/data.csv")
```

---

### Datasets

All datasets are publicly available from their original sources:

| Dataset | Instances | Features | Frequency | Source |
| --- | --- | --- | --- | --- |
| Household | 2,075,259 | 7 | 1 min | [UCI ML Repository](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption) |
| Elec2 | 45,312 | 8 | 30 min | [UCI ML Repository](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014) |
| Korea Mfg. | 35,040 | 11 | 1 min | [Scientific Data](https://doi.org/10.1038/s41597-022-01357-8) |
| Tetouan | 52,417 | 9 | 10 min | [UCI ML Repository](https://archive.ics.uci.edu/dataset/849/power+consumption+of+tetouan+city) |
| Uruguay | 105,120 | 5 | 15 min | [Figshare](https://doi.org/10.6084/m9.figshare.14650290) |

---

### Dependencies

```
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
pycaret>=3.0.0
river>=0.21.0
statsmodels>=0.14.0
matplotlib>=3.7.0
seaborn>=0.12.0
xgboost>=1.7.0
tensorflow>=2.12.0       # For LSTM baselines only
```

---

### Citation

If you use this code or framework in your research, please cite:

```bibtex
@article{palathingal2025acfmaar,
  title     = {ACF-MAAR: Leveraging Autocorrelation for Adaptive Multi-Window Regression in Streaming Data},
  author    = {Palathingal, Shahad and Raj, Ebin Deni},
  journal   = {[Journal Name]},
  year      = {2025},
  note      = {Under Review}
}
```

---

### Contact

For inquiries regarding the architecture, experimental data, or collaboration:

- **Shahad Palathingal** — [shahadphd2019@iiitkottayam.ac.in](mailto:shahadphd2019@iiitkottayam.ac.in)
- **Ebin Deni Raj** — [ebindeniraj@iiitkottayam.ac.in](mailto:ebindeniraj@iiitkottayam.ac.in)

**Affiliation:** Department of Computer Science Engineering, Indian Institute of Information Technology, Kottayam, Kerala, India

---

### License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
