# Cross-Border Spread Forecasting and Trading Strategies in the Nordic-Continental Power Market

Forecasting and backtesting day-ahead electricity price spreads across three Nordic-Continental interconnectors, built as the empirical core of a Master's thesis in Economics (Aarhus University, 2026).

This project forecasts the cross-border spread $S_t = P_{A,t} - P_{B,t}$ between coupled European bidding zones, converts those forecasts into trading signals, and backtests them under progressively more realistic execution-cost and capacity assumptions — then stress-tests whether the resulting "edge" survives that scrutiny, rather than presenting a single flattering backtest number.

## Why this project

Anyone can backtest a forecasting model and report a Sharpe ratio. The harder, more useful question is whether that Sharpe ratio would survive contact with reality: realistic transaction costs, finite interconnector capacity, and the fact that daily P&L is rarely as statistically independent as a naive annualization assumes. This project is built around answering that question honestly for three real Nordic-Continental borders, including reporting a headline result that got *smaller and more defensible* once properly diagnosed — a genuine finding, not a failure to hide.

## Key results

| Border | Robust edge under stress? | Headline finding |
|---|---|---|
| DK1 ↔ DE/LU | **Yes** | Profitable and Sharpe-positive under all three execution-cost stress scenarios (moderate/conservative/severe); degrades gracefully rather than collapsing |
| DK1 ↔ DK2 | **No** | Crosses into losses under conservative-to-severe stress; reported as a genuine negative result |
| DK1 ↔ SE3 | **Real, but overstated at short lags** | Strong day-over-day serial persistence is genuine (verified via a zero-parameter persistence-strategy autocorrelation test), but the naive 7-day HAC-Sharpe (13.16) is inflated — the effective number of independent trading days collapses to ~75 out of 815 raw days at longer lags. The properly HAC-corrected Sharpe (~7.1) is reported as the headline figure instead. |

Full methodology, diagnostics, and discussion are in the thesis (linked below / in `/docs`).

## What makes this more than a standard forecasting pipeline

- **Leakage-safe walk-forward validation.** Five-fold expanding-window CV with a 168-hour embargo, enforced consistently across every model's *own* internal validation split (including the LSTM's), not just at the outer fold boundary.
- **A custom sequence-windowing layer for the LSTM**, since this is a tabular pipeline, not a native sequence one — verified to correctly allow test-fold windows to look back into embargoed-out training rows (legitimate, since the embargo restricts *fitting*, not what a live model would know at inference time) while keeping the internal validation split strictly non-overlapping.
- **A quantile-calibration check**, not just quantile *point estimates* — empirical coverage of the LightGBM quantile model's [q10, q90] interval is measured directly, revealing miscalibration that turns out to correlate with which border's trading strategy actually loses money.
- **HAC-adjusted Sharpe ratios reported at six lags (1–90 days)**, alongside a Newey–West-implied *effective number of independent trading days* — because a single-lag Sharpe ratio can badly overstate significance under serially correlated daily P&L (Lo, 2002), and this project measures that risk directly rather than assuming it away.
- **A custom trading-payoff-derived LightGBM objective, built, verified, tested for robustness — and honestly abandoned.** Gradients and Hessians computed via PyTorch autograd and checked against finite differences to ~1e-10; a subsequent hyperparameter sensitivity analysis found the selected defaults were not robust across the full dataset, and the model was excluded from the final results on that basis. Documented as a negative methodological result rather than silently dropped.
- **Empirically investigated data limitations, not just disclosed ones.** A live diagnostic query against the ENTSO-E API confirmed its `createdDateTime` metadata cannot verify forecast-publication vintage; a follow-up robustness check (re-estimating the two core models with every forecast feature shifted an additional 24 hours) found no evidence this affects the reported results.
- **A full stylized execution-cost model** — BRP imbalance haircut, exchange fees (matched to Nord Pool's real published fee schedule), volatility-scaled slippage, margin financing, and an NTC-capacity feasibility cap — reported as a transparent cost waterfall (gross → net) rather than a single blended P&L figure.

## Models

| Model | Role |
|---|---|
| Persistence (t-24) / Seasonal Naive (t-168) | Naive baselines |
| Elastic Net | Interpretable linear benchmark |
| LightGBM (point forecast) | Primary non-linear model |
| LightGBM (quantile, q10/q50/q90) | Probabilistic forecast; drives confidence-weighted position sizing and the calibration analysis |
| LSTM | Sequence-aware model, custom windowing layer over the tabular pipeline |

All models are evaluated for statistical significance against the persistence baseline via Diebold–Mariano tests (HAC/Newey–West, Bartlett kernel).

## Data

- **ENTSO-E Transparency Platform**: day-ahead prices, load forecasts, wind/solar generation forecasts, day-ahead Net Transfer Capacity (NTC), for DK1, DK2, DE_LU, and SE3.
- **Yahoo Finance**: TTF gas futures and an EU ETS carbon proxy (KRBN), as demand/cost-driver context.
- Study period: 2022-01-01 to 2025-09-30 (ends before the October 2025 15-minute market time-unit transition).

## Project structure

```
├── Phase 2 — Data Pipeline           # ENTSO-E + Yahoo Finance ingestion, cleaning
├── Phase 3 — Feature Engineering     # Fundamentals, NTC interactions, calendar, AR lags, causal ffill
├── Phase 4 — Predictive Modeling     # Walk-forward CV, all forecasting models, calibration checks
├── Phase 5 — Trading Backtest        # Signal generation, cost waterfall, position sizing
├── Phase 5B — Execution Stress Tests # Joint moderate/conservative/severe scenario sweeps
├── Phase 5C — Serial-Dependence Diagnostics  # Multi-lag HAC Sharpe, effective independent days
└── Phase 6 — SHAP Interpretability   # Descriptive feature-importance analysis, per border
```

## Setup

```bash
pip install entsoe-py pandas numpy scikit-learn lightgbm torch yfinance matplotlib seaborn holidays shap tqdm
```

Requires a free ENTSO-E API token (register at [transparency.entsoe.eu](https://transparency.entsoe.eu)):

```bash
export ENTSOE_API_KEY="your-token"
```

## Tech stack

Python · pandas · scikit-learn · LightGBM · PyTorch · SHAP · entsoe-py · statsmodels-style HAC/Newey–West estimation (implemented directly)

## Limitations

Reported in full in the thesis — briefly: no published NTC data exists for two of the three borders (an ENTSO-E data-availability gap, verified rather than assumed); the backtest is a stylized execution framework, not a claim that reported position sizes were executable in the real auction; and the sample period spans the 2022 European gas crisis, whose disproportionate influence on the fitted models has not been separately tested.

## Author

Oliver Høyrup — Aarhus University, MSc Economics
