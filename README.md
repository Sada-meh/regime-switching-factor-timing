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

## Why these four factors

The sleeves are not an arbitrary split. Both rest on one organising question:
**how dependent is each factor's long leg on the cost of external credit?**

| Factor | Long leg | Short leg | Sleeve | Behaviour when credit is cheap |
|---|---|---|---|---|
| SMB | Small caps | Large caps | Cyclical | Outperforms |
| HML | High book-to-market (value) | Low book-to-market (growth) | Cyclical | Outperforms |
| RMW | Robust profitability | Weak profitability | Defensive | Underperforms |
| CMA | Conservative investment | Aggressive investment | Defensive | Underperforms |

**Cyclical sleeve — SMB and HML.** Small firms borrow on worse terms than large
ones: they are more opaque, post less collateral and face steeper risk premia.
When credit is cheap or easing, that penalty shrinks and the small-cap leg
rallies. Value firms are credit-sensitive for a related reason. High
book-to-market is, in practice, a screen for firms that are leveraged,
low-margin or partway through distress — precisely the balance sheets that
benefit most when refinancing gets easier. This is the distress-risk reading of
HML that Fama and French (1993) originally advanced, and it makes both factors
long the part of the market that does best when money is loose.

**Defensive sleeve — RMW and CMA.** These load on firms that do not need the
credit market. Robustly profitable companies fund themselves from operating cash
flow, so when borrowing costs rise and refinancing becomes hard, their advantage
over weak-profitability peers widens. Conservative investors face the same
asymmetry from the other side: firms committing heavily to capital expenditure
must fund it externally, and expensive credit turns aggressive investment from a
growth signal into a liability. Investors reward balance-sheet caution exactly
when caution is scarce. Fama and French (2015) added both factors to capture
this profitability–investment dimension.

**The resulting spread.** The model does not trade the four factors separately.
It forms one series — the cyclical sleeve minus the defensive sleeve:

```
cyclical  = 0.5 * (SMB + HML)
defensive = 0.5 * (RMW + CMA)
spread    = cyclical - defensive
```

If credit conditions really do drive the rotation, this spread should be high in
easy-credit states and low in tight ones — persistently enough for a two-state
Markov model to recover the switch. Testing that claim is the point of this
repository; §Results reports what was actually found.

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



## Data 

The public inputs (Fama-French factors, FRED series) are committed. The
CRSP, Compustat and LSEG/Refinitiv extracts cannot be redistributed.

`Data/README.md` documents exactly which files are missing, which vendor
table each comes from, and the fields needed to rebuild them from your own
institutional access. Loaders that need a missing file raise a clear
`FileNotFoundError` pointing back to those instructions.

The Markov model depends only on public data and runs without any of this.

