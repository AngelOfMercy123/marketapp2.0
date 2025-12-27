# MarketApp (Modular Upgrade)

This is a clean modular rebuild of your Streamlit market app, with:

- **Universe + Watchlist** management
- **Daily price fetching** (free, no API keys) via **Stooq**
- **Technical features + ML classifier** (next-horizon up/down probability)
- **Prediction logging + scoring** (tracks accuracy once horizon passes)
- **Demographic risk score** (your thesis inputs: age/physical labor/visa-dependence/stability)
- **Free RSS news window** (Google News RSS)
- Auto refresh option for "live" scanning

## Install

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Data files

- `universe.csv` (created automatically if missing)
- `watchlist.csv`
- `predictions.csv` (created when you generate signals)
- `data_prices/` (cached OHLCV CSVs)

### Universe schema (important)
These columns are supported out-of-the-box:

- `utility_score` (0-10)
- `policy_score` (0-10)
- `leadership_score` (0-10)
- `labor_exposure` (0-10)
- `trend_score` (0-10)

Demographics inputs:

- `avg_workforce_age` (years)
- `pct_physical_labor` (0-100)
- `pct_female` (0-100)
- `pct_immigrant` (0-100)
- `retirement_age` (years)
- `workforce_stability` (0-10)

> Note: the demographic score is **aggregated** and designed to represent *operational workforce sustainability*, not to judge individuals.

## How to use in practice
1. Add tickers in Universe.
2. Fill in thesis columns (utility/policy/leadership/etc and demographics).
3. Update prices.
4. Train model.
5. Generate signals.
6. Score matured predictions over time and iterate.
