# <a href="https://www.kaggle.com/competitions/vr-calibration-free-emg-challenge-pieeg/data?select=train">Aura: Calibration-free EMG Challenge</a>

## Training-data analysis

* **Notebook:** `notebooks/01_train_data_analysis.ipynb` — deep analysis of the 28
  training sessions (descriptive structure, signal quality, class separability,
  train/test shift, leakage, baseline, preprocessing recipe).
* **Report:** `docs/data_analysis_report.md` — condensed findings and recommendations.
* **Figures:** `graphs/data_analysis/01…15_*.png`.
* **Rules reference:** `docs/Aura_EMG_Competition_Guide.md`.

### Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/01_train_data_analysis.ipynb
```

Raw data goes in `data/raw/train/` and `data/raw/test/` (git-ignored).

### Run on Kaggle

Open the notebook and choose *Run All*. It locates the dataset under `/kaggle/input`
automatically, writes figures and caches to `/kaggle/working`, and displays every figure
inline. If the data lives elsewhere, set `AURA_TRAIN_DIR`, `AURA_TEST_DIR` or
`AURA_OUT_DIR`. No code changes are required between local and Kaggle runs.
