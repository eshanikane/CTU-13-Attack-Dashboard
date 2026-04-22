# CTU-13 Attack Traffic Dashboard

A small data analytics / machine learning miniproject that scores network flows from the CTU-13 style attack traffic dataset and explores them in an interactive Streamlit app. The main application lives in **`app.py`**.

## What it does

- Loads **`CTU13_Attack_Traffic.csv`** from the project directory (path is fixed in code).
- Cleans numeric features: coerces columns to numbers, drops rows with too many missing values, imputes remaining gaps with column medians, and maps the **`Label`** column to a binary target when possible (numeric labels or keyword heuristics such as “bot”, “normal”, etc.).
- **Scoring**
  - **Supervised** (when at least two distinct label values exist after parsing): trains a **Random Forest** (and **XGBoost** if installed) on an 80/20 stratified split; risk is based on the positive-class probability (XGBoost if available, otherwise Random Forest).
  - **Unsupervised** (insufficient labels): fits an **Isolation Forest** and converts anomaly scores into a 0–100 style risk scale.
- **Dashboard**: Plotly charts (top suspicious flows, risk histogram, scatter vs a chosen flow feature, bot vs normal pie), feature-importance or variance-based signals, correlation heatmap, summary metrics, expandable classification reports, and a **CSV download** of scored flows.

## Requirements

- Python 3.10+ recommended (compatible with the listed packages).
- Install dependencies:

```bash
pip install -r requirements.txt
```

`requirements.txt` includes `streamlit`, `pandas`, `numpy`, `scikit-learn`, `plotly`, `xgboost`, and `networkx`. If **XGBoost** fails to install on your system, the app still runs: supervised mode falls back to Random Forest probabilities for the risk score.

## Dataset

Place **`CTU13_Attack_Traffic.csv`** in the same folder as `app.py` (this is the default expected location). The app will not start without this file.

## Run the app

From the project directory:

```bash
streamlit run app.py
```

Then open the URL Streamlit prints in the browser (typically `http://localhost:8501`).

## Project layout

| File | Role |
|------|------|
| `app.py` | Streamlit dashboard and ML scoring pipeline |
| `CTU13_Attack_Traffic.csv` | Input flow-level dataset |
| `requirements.txt` | Python dependencies |
| `DAL_MiniProject.ipynb` | Notebook companion / exploratory work (optional) |

## Notes

- **Random seed**: `RANDOM_STATE = 42` in `app.py` for reproducible splits and models.
- **Risk bands**: scores are clipped to 0–100; **High** risk is used for flows classified as **Bot** in the summary `Status` column when risk is above 70.
