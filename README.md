# Regime-Switching Factor Timing

Can you time equity style factors by conditioning on the macro regime?

This repository contains a full backtest suite that tries two independent
answers to that question, evaluates both against a strict pre-registered
pass/fail checklist, and reports the result honestly.

**Headline finding: no.** Every strategy tested produces a Fama-French
five-factor alpha that is zero or negative, several significantly so. Raw
Sharpe ratios look respectable (0.46-0.55), but that is compensation for
factor exposure, not skill. Performance also decays sharply after 2007.

---

## The two approaches

**1. Latent regimes — a Markov switching model**
`trial-and-error/Markov/markov_factor_timing.py`

Fits a two-state `MarkovRegression` to the spread between a cyclical sleeve
(½·SMB + ½·HML) and a defensive sleeve (½·RMW + ½·CMA). Regimes are
*inferred*, not observed. Filtered state probabilities are mapped to
portfolio weights through 0.60/0.40 thresholds and evaluated in an
expanding-window backtest with a 60-month minimum training sample, so there
is no look-ahead.

**2. Observable regimes — macro state classification**
`ALPHA/alpha_common.py`

Classifies each month as "good" or "bad" from observable macro variables
(term spread, default spread, VIX, z-scored) via median and tercile splits,
then rotates factor legs accordingly. A gradient-boosted classifier
(`4 S1 ML`) additionally tries to *predict* next month's regime.

Each strategy is run in two versions: **FF** (Fama-French published factors)
and **Self-built** (factors reconstructed from CRSP/Compustat).

---

## Results

### Markov model (reproducible from the included public data)

| Series | Ann. return | Vol | Sharpe | Max DD | FF5 alpha | t |
|---|---|---|---|---|---|---|
| Strategy (gross) | 3.58% | 6.55% | **0.546** | -16.9% | -0.12% | -2.19 |
| Equal-sleeve benchmark | 2.31% | 5.87% | 0.394 | -24.0% | -0.19% | -7.86 |
| Defensive sleeve only | 3.38% | 6.89% | 0.491 | -27.6% | -0.19% | -7.86 |
| Cyclical sleeve only | 1.24% | 8.13% | 0.153 | -39.2% | -0.19% | -7.86 |

Net of costs: 0.509 (25bps), 0.471 (50bps), 0.432 (75bps).

The strategy beats its benchmarks on Sharpe and drawdown and survives
transaction costs — but its FF5 alpha is *significantly negative*, and it
fails three of six pre-registered criteria (model stability, economic
interpretability, robustness support). Verdict recorded in
`trial-and-error/Markov/markov_final_decision.md`: **NO**.

### Macro-regime strategies (require WRDS data)

| Strategy | Version | Sharpe | FF5 alpha | p | Pre-2008 | Post-2007 |
|---|---|---|---|---|---|---|
| 1A Centerpiece | FF | 0.544 | -0.08% | 0.19 | 0.831 | 0.242 |
| 3A Dynamic timing | FF | 0.536 | -0.14% | 0.000 | 1.045 | 0.202 |
| 4 S1 ML | FF | 0.458 | -0.18% | 0.050 | 0.421 | 0.493 |
| 1A Centerpiece | Self-built | -0.052 | -0.43% | 0.36 | -0.224 | 0.007 |
| 3A Dynamic timing | Self-built | -0.352 | -0.32% | 0.67 | n/a | -0.352 |
| 4 S1 ML | Self-built | 0.549 | 0.60% | 0.14 | n/a | 0.549 |

The pre-2008 to post-2007 collapse is the most robust pattern in the table.
The gap between FF and Self-built versions is itself a finding: results that
depend on which factor construction you use are not results.

---

## Repository layout

```
├── ALPHA/                       Macro-regime strategy pipeline
│   ├── alpha_common.py            Shared engine (signals, backtest, stats, ML)
│   ├── 1A Centerpiece/            Baseline median-split rotation
│   ├── 3A T1 Dynamic timing/      Continuous timing overlay
│   ├── 4 S1 ML/                   Gradient-boosted regime prediction
│   └── academic_report_summary.*  Consolidated results
├── trial-and-error/             Model development and dead ends
│   ├── credit_factor_rotation_core.py   Data loaders, backtest core
│   ├── hml_market_regime_switch_strategies.py
│   ├── Markov/                    Markov switching model + diagnostics
│   └── FINAL/                     Long-only and HML/SMB switch variants
└── Data/                        Public inputs (see Data/README.md)
```

Sample period: 1990-01 to 2024-09, monthly. Costs tested at 0/25/50/75 bps.

---

## Quickstart

```bash
git clone <this-repo>
cd regime-switching-factor-timing
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Runs end-to-end on the included public data
cd trial-and-error/Markov && python markov_factor_timing.py
```

This writes model diagnostics, filtered-probability plots, a recursive
backtest, robustness tables and the final decision file into that directory.

---

## Data and licensing

The public inputs (Fama-French factors, FRED series) are committed. The
CRSP, Compustat and LSEG/Refinitiv extracts are **subscription-licensed and
deliberately excluded** — they cannot be redistributed.

`Data/README.md` documents exactly which files are missing, which vendor
table each comes from, and the fields needed to rebuild them from your own
institutional access. Loaders that need a missing file raise a clear
`FileNotFoundError` pointing back to those instructions.

The Markov model depends only on public data and runs without any of this.

---

## Caveats

- Single market (US), single sample period; no cross-country validation.
- The two-state Markov specification failed its own stability checks —
  state assignments move under re-estimation. Treat the regime labels as
  fragile.
- Self-built factor versions have short effective samples (15-226 months
  after burn-in) and their point estimates are not reliable.
- These are backtests. Nothing here is investment advice.

## Authors

Group 17 — Meher Paryani, Cillian, Beau.

Code released under the MIT Licence (see `LICENSE`). The licence covers the
code only, not any third-party data.
