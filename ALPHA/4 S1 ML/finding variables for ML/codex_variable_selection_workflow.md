# VARIABLE SELECTION WORKFLOW FOR DYNAMIC FACTOR TIMING — INSTRUCTIONS FOR CODEX

## Objective

Help construct a disciplined predictor-selection workflow for a monthly factor-timing strategy.

The goal is **not** to maximize the number of variables. The goal is to identify a **small, economically justified, non-redundant, and out-of-sample useful** predictor set for a model such as gradient boosting trees.

This workflow should be followed **before** finalizing the predictive model.

---

## Core Principle

Do **not** assume that more variables are better.

For monthly asset-pricing / factor-timing problems with limited sample size, too many weak or redundant predictors increase:

- noise
- data-mining risk
- instability
- overfitting
- false discoveries

Gradient boosting trees can model nonlinearities and interactions, but they do **not** require “as many variables as possible.” They perform best when fed a **carefully chosen set of informative predictors**.

Treat this as a **signal discovery and screening problem**, not a brute-force feature dump.

---

## Sample Context

Assume the strategy uses monthly data and a limited historical sample.

This implies:

- the effective sample size is modest
- highly parameterized or overly wide feature sets are dangerous
- predictors must be justifiable both economically and statistically

As a working default, prefer **8 to 12 strong variables** over 30 to 50 weak ones.

---

## Required Workflow

### Step 1 — Start from economic mechanism

For every candidate variable, first identify the economic channel it is supposed to proxy for.

Each variable should be linked to one of the following mechanisms, or another clearly stated mechanism:

- **credit conditions**
- **funding stress**
- **risk aversion / market stress**
- **growth expectations**
- **policy / rates regime**
- **inflation regime**
- **factor crowding**
- **factor momentum / factor state**
- **factor valuation or spread conditions**

For each candidate predictor, complete the sentence:

> “This variable should matter because it proxies for …”

If this sentence cannot be completed cleanly, flag the variable as weakly motivated.

#### Deliverable
Create a table with columns:

- variable name
- definition
- economic mechanism
- expected sign or qualitative effect
- data source
- publication lag / real-time availability concerns

---

### Step 2 — Screen the literature

Before any model fitting, compare the candidate variables against prior literature.

Focus on variables that are repeatedly used in:

- return prediction
- macro-finance timing
- factor timing
- business-cycle-sensitive asset pricing
- defensive / low-volatility / profitability / investment timing

Classify variables into:

- **core** = strong literature support and strong economic logic
- **optional** = plausible, but less central or more exploratory
- **drop** = weak justification, unclear interpretation, or obvious duplication

#### Strong baseline examples
These are examples of variables that are often defensible in this context:

- TERM / yield slope
- DEF / credit spread
- change in TERM
- change in DEF
- VIX
- recent market excess return
- trailing factor return
- trailing factor volatility
- factor spread / valuation spread

#### Deliverable
Produce a literature-screened shortlist and explicitly label each candidate as:

- core
- optional
- drop

---

### Step 3 — Run simple univariate screening

Before using gradient boosting, evaluate each candidate variable on its own.

This is not the final model. It is a filtering stage.

For each variable, test whether it has a sensible relationship with **next-month factor returns** or **next-month active strategy returns**.

Recommended checks:

1. correlation with next-month target return
2. predictive regression using only that variable
3. average next-month return in low / middle / high predictor states
4. quintile or tercile sort analysis
5. visual scatter or binned relationship plots
6. subsample stability check

The purpose is to identify variables that are:

- clearly useless
- unstable
- only active in one short window
- directionally inconsistent with the theory

#### Deliverable
For each variable, report:

- sign of relationship
- rough strength
- stability across subperiods
- preliminary keep / review / drop decision

---

### Step 4 — Inspect nonlinear and threshold behavior

Some predictors may look weak in a linear regression but matter in a state-dependent way.

Examples:

- VIX may matter mainly when already elevated
- credit spreads may matter mainly when widening sharply
- slope variables may matter more near inversion
- factor momentum may matter only in calm markets

Check for:

- threshold effects
- asymmetric effects
- interaction effects
- nonlinear monotonicity or non-monotonicity

Recommended tools:

- binned average return plots
- conditional return tables
- state-by-state performance comparisons
- interaction heatmaps for pairs of predictors

This step should happen **before** the tree model is trusted to discover everything automatically.

#### Deliverable
Flag which variables appear:

- approximately linear
- nonlinear but useful
- noisy / patternless

---

### Step 5 — Check redundancy and multicollinearity-by-proxy

Even though tree models can tolerate correlated predictors better than linear models, highly overlapping variables still make interpretation and stability worse.

Examples of likely overlap:

- TERM and related slope measures
- DEF and other credit-spread proxies
- VIX and realized market volatility
- short-horizon and highly similar momentum measures

Check:

- pairwise correlations
- conceptual overlap
- whether one variable dominates another in screening
- whether two variables are effectively the same signal expressed differently

When two variables serve the same role, keep the cleaner, more standard, or more interpretable one.

#### Deliverable
Produce a reduced predictor set with duplicate or near-duplicate variables removed.

---

### Step 6 — Fit the model only after screening

Only after completing the prior steps should gradient boosting be trained.

The goal is to learn nonlinear mappings and interactions among **already-screened predictors**, not to use boosting as a substitute for economic thinking.

Recommended practice:

- begin with the reduced predictor set
- use strict time-series train / validation / test logic
- respect all lags to avoid look-ahead bias
- standardize or transform predictors only using training information when needed
- keep the model shallow and conservative

If sample size is limited, prefer a smaller feature set and milder model complexity.

#### Deliverable
Train the model on the screened set and report out-of-sample performance relative to simpler baselines.

---

### Step 7 — Use model-based importance carefully

After model fitting, evaluate which variables matter most.

Use multiple importance diagnostics, not just one:

- permutation importance
- SHAP values
- split / gain importance
- rolling-window importance stability

A variable is more credible if it:

- appears important repeatedly across windows
- behaves in the expected direction
- improves out-of-sample results
- remains useful even when competing predictors are included

Do **not** keep a variable just because it appears important in one sample period.

#### Deliverable
Produce an importance table with:

- average importance
- importance stability across rolling periods
- interpretation of sign / direction when possible
- final recommendation: keep / optional / drop

---

### Step 8 — Force robustness before final inclusion

A candidate variable should only be admitted to the final feature set if it survives robustness checks.

Test whether the variable still helps when:

- the sample is split into subperiods
- crisis years are excluded
- standardization conventions change
- similar predictors are added or removed
- the target variable is slightly redefined
- evaluation is done strictly out of sample

If a variable only works in one narrow period or under one fragile specification, treat it as unreliable.

#### Deliverable
Create a final “robustness verdict” for each predictor.

---

## Final Classification Rules

At the end of the workflow, every variable must be assigned to one of three buckets:

### Core
Use in the main model.

Requirements:

- strong economic story
- literature support
- non-redundant
- useful in screening
- stable or at least defensible out of sample

### Optional
Use only in robustness or alternative specifications.

Requirements:

- plausible story
- some evidence of usefulness
- but weaker stability, weaker interpretability, or overlap with stronger variables

### Drop
Exclude from the final model.

Reasons may include:

- no clear theory
- weak or unstable screening results
- redundancy
- likely look-ahead / revision problems
- obvious data-mining candidate

---

## Preferred Predictor-Set Construction for This Project

Unless there is strong evidence otherwise, start from something like the following.

### Core macro variables

- TERM
- DEF
- change in TERM
- change in DEF
- VIX
- trailing 3-month market excess return

### Core factor-state variables

For each factor sleeve, consider:

- trailing 3-month factor return
- trailing 12-month factor return
- trailing factor volatility
- current factor spread or valuation-related measure

### Candidate rule of thumb

Start with this moderate set first. Only expand it if the added variables clearly improve out-of-sample behavior and pass robustness.

---

## Coding Requirements for Codex

Implement this as a reproducible feature-selection workflow.

### Required outputs

1. A master table of all candidate variables and their rationale.
2. A literature-screened shortlist.
3. Univariate screening results for each candidate variable.
4. Nonlinearity / threshold diagnostics.
5. Redundancy reduction summary.
6. Model-based importance diagnostics after fitting.
7. Final classification into core / optional / drop.
8. A final proposed predictor set for the main model.

### Required safeguards

- no look-ahead bias
- no use of revised future information in predictor construction
- all targets lagged correctly
- time-series validation only
- no random shuffling across time
- clear distinction between in-sample screening and out-of-sample testing

### Required coding style

- modular functions
- readable variable names
- comments explaining economic rationale where relevant
- tables saved to file
- plots saved to file
- all design choices documented

---

## What Codex Should Do First

Before writing the final model, first answer these questions explicitly:

1. Do we have enough data to run this workflow well?
2. Which candidate variables can already be built from the available files?
3. Which variables require extra data?
4. Are any proposed variables too redundant or too weakly justified?
5. Is the effective sample size too small for the proposed feature count?
6. What is the recommended final feature count given the data constraints?

If more data or clarification is needed, ask for it **before** implementing the final predictor set.

---

## Bottom Line

The task is **not**:

> “Find as many variables as possible.”

The task is:

> “Find the smallest set of variables that is economically motivated, statistically defensible, non-redundant, and useful out of sample.”

That is the correct workflow for feature selection in a monthly factor-timing strategy using gradient boosting trees.
