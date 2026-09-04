### 1. Electricity Price Spread Forecasting and Trading Strategies ⚡
This is the code of a 10 ECTS Topics project I've done as part of my masters in Economics. 
<details>
<summary><b>View Technical Details & Infrastructure Architecture</b></summary>

#### Objective
Forecasts day-ahead electricity price spreads across three Nordic/Continental European bidding-zone pairs (DK1↔DE_LU, DK1↔DK2, DK1↔SE3) and backtests a systematic trading strategy on the resulting edge, built with the statistical and execution-cost rigor of a real trading desk rather than a stylized backtest.

#### Data Pipeline & Feature Engineering
* **Multi-Source Ingestion:** Pulls hourly day-ahead prices, load, wind/solar generation, and net transfer capacity (NTC) from the ENTSO-E Transparency Platform across four bidding zones, plus TTF gas and EU carbon (KRBN) proxies via `yfinance`.
* **Feature Engineering:** Builds lagged spreads (t-24/48/168h), wind-forecast differentials, per-zone residual demand, calendar/holiday dummies, and hand-engineered wind×NTC / demand×NTC interaction terms — validated via SHAP as genuinely predictive rather than spurious.

#### Modeling & Statistical Rigor
* **Model Comparison:** Walk-forward (expanding-window, 168h embargoed to prevent leakage) evaluation spanning naive persistence, seasonal-naive, Elastic Net, LightGBM (point + quantile q10/q50/q90), XGBoost, LSTM, and an encoder-only Transformer — the sequence models using a 168h windowed-lookback layer.
* **Significance Testing:** HAC/Newey-West Diebold-Mariano tests and HAC-adjusted Sharpe ratios (7d/60d/90d) to check whether headline performance survives serial-dependence correction, not just a single point estimate.

#### Execution-Realistic Backtest
* **Risk-Adjusted Sizing:** Kelly and half-Kelly position sizing from realized (not model-quantile) volatility, avoiding circularity between the model's own confidence and its risk budget.
* **Full Cost Waterfall:** Sequential cost accounting from gross PnL → BRP imbalance haircut → exchange fees → slippage → VaR-style margin financing cost, plus an NTC-capacity feasibility cap (position capped at 10% of published day-ahead NTC) so backtested trades stay executable in principle.

---
#### 📊 Key Findings
| Border | HAC-Adjusted Sharpe | Note |
| :--- | :--- | :--- |
| **DK1–SE3** | ~13 (7d) → ~7-8 (60-90d) | Strongest raw edge, but the 7d figure overstates it — corrects down once serial dependence is priced in |
| **DK1–DE_LU** | ~7.7 | Most robust edge under execution stress-testing |
| **DK1–DK2** | ~2-3 | Weakest border; edge disappears under conservative/severe stress scenarios |

</details>
