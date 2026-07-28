# ACF-MAAR: Autocorrelation-Guided Multi-Window and AutoML-based Adaptable Regression

ACF-MAAR is a streaming-regression framework that uses a single
autocorrelation (ACF) signal to jointly drive three tasks usually handled
separately: adaptive window sizing, structural drift detection, and model
adaptation. Window length is derived from a lag-dependent Bartlett
significance bound, drift is flagged by a variance-standardised ACF-profile
statistic with a chi-square threshold invariant to window size, and a
two-tier AutoML policy escalates from window resizing to full model
replacement only when local prediction error also exceeds a statistically
derived threshold.

## Key results

- **Forecasting:** ranks 2nd of 10 stream-specific baselines by MAE on the
  UCI Household Electric Power dataset (MAE 0.170, R² 0.932), and is
  strongest in the operationally critical peak-consumption regime.
- **Window sizing:** fully data-driven and hyperparameter-free; competitive
  with the best fixed window without grid search (validated on the
  Mackey-Glass benchmark, R² 0.997).
- **Detection:** the ACF statistic detects structural changes invisible to
  mean- and distribution-based detectors; combined with ADWIN it is
  reported directly against conventional baselines.
- **Adaptation:** a multi-day ablation finds naive retraining is a net cost
  on the Household data; a controlled lag-shift experiment localises this to
  recency-based feature selection and undersized retrain windows, and shows
  that ACF-guided feature selection on an adequate window recovers accuracy
  (23% improvement over a static model).

## Datasets

- **UCI Household Electric Power** (primary, real-world): minute-resolution
  household power consumption. Target: `Global_active_power`.
- **Mackey-Glass** (third-party synthetic benchmark): a standard chaotic
  time series with genuine temporal autocorrelation, used to validate
  generalisation and window sizing under known ground truth.
- **Lag-shift stream** (controlled diagnostic, generator included): a
  synthetic AR stream whose generating lag switches (1 → 30), used to isolate
  and demonstrate the adaptation mechanism under a genuine
  predictive-relationship change.

## Notebooks (Google Colab)

- **Main pipeline (Household):**
  https://colab.research.google.com/drive/1cYrdmkiSgrRFVOxE8knWwByVZrXx8Di6
- **Mackey-Glass benchmark:**
  https://colab.research.google.com/drive/1ZshjiPaYpwg66mK6aegCLTsjo6JcK0S_
- **Lag-shift adaptation experiment:**
  https://colab.research.google.com/drive/13KaehSztAWl6kYm7tekD3WJOKG4-wKSI

## Installation

```bash
pip install -r requirements.txt
```

Core dependencies: numpy, pandas, scikit-learn, river, statsmodels,
ruptures, scipy, matplotlib.

## Reproducing the results

1. Download the UCI Household Electric Power dataset and place it as
   indicated in the main notebook's data-loading cell.
2. Open the main pipeline notebook and run all cells in order. Key parameters
   (`ALPHA_W=0.05`, `ALPHA_D=0.10`, `L_LAGS=12`, `C_CONSEC=3`) are set in the
   global parameters cell.
3. The Mackey-Glass and lag-shift notebooks are self-contained: they generate
   their own data and run the same pipeline.

## Method parameters

| Parameter | Value | Role |
|-----------|-------|------|
| `ALPHA_W` | 0.05 | window-sizing significance level (Bartlett bound) |
| `ALPHA_D` | 0.10 | drift-detection significance level (chi-square threshold) |
| `L_LAGS`  | 12   | effective lag count for the drift statistic |
| `C_CONSEC`| 3    | consecutive insignificant lags to set n* |
| `K_MIN`   | 48   | minimum window (4 × L_LAGS) |

## Citation

If you use this work, please cite:

> Palathingal, S. and Raj, E. D. ACF-MAAR: Autocorrelation-Guided
> Multi-Window and AutoML-based Adaptable Regression for Streaming
> Forecasting. [venue / year — update on acceptance]

## License

[Add your license — e.g. MIT]
