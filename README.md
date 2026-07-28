# 🔄 ACF-MAAR

**Autocorrelation-Guided Multi-Window and AutoML-based Adaptable Regression**

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-research-orange.svg)

A streaming-regression framework that uses a **single autocorrelation signal** to jointly drive window sizing, drift detection, and model adaptation — three tasks usually handled by separate heuristics.

---

## ✨ Highlights

- 📐 **Hyperparameter-free window sizing** from a lag-dependent Bartlett bound
- 🎯 **Structural drift detection** invisible to mean-based detectors, via a variance-standardised ACF statistic with a window-invariant chi-square threshold
- ⚙️ **Two-tier AutoML adaptation** — lightweight resize first, full retrain only when error exceeds a statistical threshold
- 📊 **Rigorous multi-day evaluation** that surfaces limitations single-segment protocols hide

---

## 📈 Key Results

| Aspect | Result |
|--------|--------|
| 🏆 Forecasting | **2nd of 10** stream baselines by MAE (0.170, R² 0.932) |
| ⚡ Peak regime | Strongest relative accuracy where it matters operationally (7.7%) |
| 📐 Window sizing | Competitive with best fixed window, **no grid search** |
| 🔍 Detection | Catches structural change mean-based detectors miss |
| 🔧 Adaptation | Controlled experiment recovers **+23%** with ACF-guided feature selection |

---

## 🗂️ Datasets

| Dataset | Type | Role |
|---------|------|------|
| 🏠 **UCI Household Power** | Real-world | Primary evaluation |
| 🌀 **Mackey-Glass** | Third-party synthetic | Generalisation benchmark (R² 0.997) |
| 🔀 **Lag-shift stream** | Controlled diagnostic | Isolates the adaptation mechanism |

---

## 📓 Notebooks

| Notebook | Link |
|----------|------|
| 🔬 **Main Pipeline (Household)** | [Open in Colab](https://colab.research.google.com/drive/1cYrdmkiSgrRFVOxE8knWwByVZrXx8Di6) |
| 🌀 **Mackey-Glass Benchmark** | [Open in Colab](https://colab.research.google.com/drive/1ZshjiPaYpwg66mK6aegCLTsjo6JcK0S_) |
| 🔀 **Lag-Shift Experiment** | [Open in Colab](https://colab.research.google.com/drive/13KaehSztAWl6kYm7tekD3WJOKG4-wKSI) |

---

## 🚀 Installation

```bash
pip install -r requirements.txt
```

**Core dependencies:** `numpy` · `pandas` · `scikit-learn` · `river` · `statsmodels` · `ruptures` · `scipy` · `matplotlib`

---

## ⚡ Quick Start

1. 📥 Download the [UCI Household Power dataset](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)
2. 🔬 Open the **Main Pipeline** notebook and *Run all*
3. 🌀🔀 The Mackey-Glass and Lag-Shift notebooks are **self-contained** — they generate their own data

---

## 🎛️ Parameters

| Parameter | Value | Role |
|-----------|-------|------|
| `ALPHA_W` | 0.05 | Window-sizing significance (Bartlett bound) |
| `ALPHA_D` | 0.10 | Drift-detection significance (chi-square) |
| `L_LAGS` | 12 | Effective lag count for drift statistic |
| `C_CONSEC` | 3 | Consecutive insignificant lags to set n* |
| `K_MIN` | 48 | Minimum window (4 × L_LAGS) |

---

## 📄 Citation

```bibtex
@article{acfmaar,
  title   = {ACF-MAAR: Autocorrelation-Guided Multi-Window and
             AutoML-based Adaptable Regression for Streaming Forecasting},
  author  = {Palathingal, Shahad and Raj, Ebin Deni},
  journal = {TBD},
  year    = {2026}
}
```

---

## 📜 License

Released under the MIT License — see [`LICENSE`](LICENSE).
