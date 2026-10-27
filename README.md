# SportAnalytics

Football recruitment decision-support tool. It helps a club's Head of Recruitment decide whether to bid for a player and up to what price, using two predictive KPIs:

| KPI | What it predicts | Model |
|---|---|---|
| 1 · Market value | Player's market value at the next yearly valuation | Log-scale regression with an age curve |
| 2 · Passes per 90 | Player's passes per 90 minutes next season | Gradient boosting (explained with SHAP) |

The app shows each model's accuracy, overall predictor importance, per-player contributions, and a what-if tool with a baseline, automatic summaries, and a downloadable scenario log.

## Setup

Requires **Python 3.10 or newer**.

```bash
git clone https://github.com/Dinahak/SportAnalytics.git
cd SportAnalytics/Predictive_Model
pip install -r requirements.txt
streamlit run app/app.py
```

The app opens at http://localhost:8501. All model outputs are already in the repo, so you can run the app right away.

To rebuild everything from the raw data (about 2 minutes):

```bash
cd Predictive_Model
python code/run_all.py
```

## Repository layout

```
Predictive_Model/
  app/app.py              Streamlit application (5 screens)
  code/                   Pipeline steps 01-10 and model code
    value_model.py        KPI 1 model
    performance_model.py  KPI 2 model
    explain.py            Importance, contributions, interpretation text, what-if summaries
  data/raw/               Week 5 synthetic data (unchanged)
  data/final/             Cleaned tables the app reads
  data/model/             Model files, training tables, forecasts
  data/withheld/          Simulated outcomes held back for validation
  outputs/                Model comparison, backtests, validation results
statbomb_data/            Scripts and docs used to calibrate the synthetic data
docs/                     Progress report, contributions, demo video outline
```

## Data

All data the app uses is **synthetic** (no real players or clubs) and is included in full.

Two external sources were used only to calibrate the synthetic data and are **not** in the repo:

- **StatsBomb Open Data** — its user agreement forbids redistribution. Anyone can download it with `python statbomb_data/01_fetch_statsbomb.py --out <folder>` (see `statbomb_data/LICENSE.pdf`).
- **Transfermarkt market values** (`transfermkt_data/`, git-ignored) — not shared because of Transfermarkt's terms of use. Authorized users can request the files from the group.

## Deliverables (Week 10)

- Progress report: [`docs/PROGRESS_REPORT.md`](docs/PROGRESS_REPORT.md)
- Group contributions: [`docs/CONTRIBUTIONS.md`](docs/CONTRIBUTIONS.md)
- Demo video: see link in [`docs/PROGRESS_REPORT.md`](docs/PROGRESS_REPORT.md)
