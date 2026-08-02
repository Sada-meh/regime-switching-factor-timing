# STRATEGY 3A-T1 UPGRADE LOOP
## Self-running improvement protocol for the dynamic timing model

This file is a **supplement** to the main strategy guide. It does **not** replace the locked baseline design. Start from the existing Strategy 3A-T1 implementation exactly as specified in the main guide:
- factor universe = `RMW, CMA, HML`
- baseline predictors = `z_TERM(t-1), z_DEF(t-1)`
- interaction design = 3 raw factors + 6 interactions = 9 total series
- optimiser = regularised mean-variance with shrinkage target `w0 = [1/3, 1/3, 1/3, 0, 0, 0, 0, 0, 0]`
- lambda grid baseline = `{1, 3, 10, 30, 100, 300, 1000, 10000, 100000}`
- expanding-window design = 120m initial train, 24m validation, 12m test
- gross exposure normalisation = `sum(abs(W_i)) = 1`

The main guide defines these baseline components and the interpretation of lambda extremes. Use them as the immutable reference implementation for this upgrade loop.

---

## OBJECTIVE

Improve the out-of-sample performance and stability of Strategy 3A-T1 **without** turning it into a data-mined or dimension-heavy model.

The loop must:
1. make **one change at a time**,
2. run the full evaluation,
3. compare results against the currently accepted version,
4. keep the change only if it improves the model under pre-committed rules,
5. otherwise revert it completely,
6. continue until one full pass produces no accepted improvements.

This is a deterministic hill-climbing loop, not a free-form research exercise.

---

## NON-NEGOTIABLE RULES

1. **No look-ahead bias.** All predictors remain lagged exactly as in the base guide.
2. **One change per iteration.** Never bundle multiple upgrades into one test.
3. **Same live sample unless explicitly unavoidable.** Do not accept a change that shortens the OOS sample unless the change is impossible otherwise.
4. **Same benchmarks and evaluation suite.** Reuse the exact Strategy 3A-T1 evaluation framework from the base guide.
5. **Keep TERM and DEF as the baseline predictor set.** Do not add more macro variables until the stabilisation upgrades have been tested first.
6. **Do not go below lambda = 1.**
7. **Always save both the candidate version and the last accepted version.**
8. **Every decision must be logged** in a machine-readable results table.

---

## WHY THIS LOOP EXISTS

If the selected lambda is often at the low end (`1`) or the high end (`1000+`), the model is likely unstable or the validation design is noisy. In the base guide, a very high lambda already implies the model is shrinking toward equal weights and that timing may not be adding value. Therefore, the first task is to stabilise the procedure, **not** to immediately add more predictors.

---

## MASTER ACCEPTANCE RULE

A candidate change is **accepted** only if **all** of the following are true relative to the currently accepted version:

### Primary condition
- `Net Sharpe at 25 bps` must improve.

### Secondary condition
At least **one** of these must also improve:
- `Net Sharpe at 50 bps`
- `FF5 alpha (annualised)`
- `lambda_extreme_share` decreases
- `weight_instability` decreases
- `turnover` decreases

### Guardrail condition
None of these may materially worsen:
- `Max drawdown` may not worsen by more than 10%
- `turnover` may not rise by more than 15%
- `lambda_extreme_share` may not rise by more than 10 percentage points
- `weight_instability` may not rise by more than 15%
- `live_months` may not shrink

If the candidate fails any of these conditions, **reject and revert**.

---

## METRICS TO COMPUTE AT EVERY ITERATION

For the candidate version and the current accepted version, compute and store:

- `gross_sharpe`
- `net_sharpe_25bps`
- `net_sharpe_50bps`
- `mean_ann_return`
- `vol_ann`
- `ff5_alpha_ann`
- `ff5_alpha_t`
- `max_drawdown`
- `turnover`
- `lambda_extreme_share`
- `lambda_low_share` where lambda in `{1, 3}`
- `lambda_high_share` where lambda in `{1000, 10000, 100000}`
- `weight_instability`
- `avg_abs_delta_weight`
- `max_abs_weight`
- `clip_rate` if clipping/caps are used
- `live_months`

Define:
- `lambda_extreme_share = share of rebalances where lambda is in {1, 3, 1000, 10000, 100000}`
- `weight_instability = mean over time of sum_i |W_i(t) - W_i(t-1)|`
- `avg_abs_delta_weight = average absolute monthly change in factor weights`
- `max_abs_weight = max over all months and factors of |W_i(t)|`

---

## LOOP STRUCTURE

### Step 0 — Freeze the baseline
1. Run the exact baseline Strategy 3A-T1 from the main guide.
2. Save results as `accepted_version = baseline`.
3. Save all metrics as `accepted_metrics`.
4. Save weight paths, lambda path, return series, and regression output.

### Step 1 — Create the ordered candidate queue
Test the following changes **in this exact order**.

#### Candidate 1 — longer validation window to 36 months
Change only:
- validation window from `24m` to `36m`
Keep everything else unchanged.

#### Candidate 2 — longer validation window to 48 months
Apply only if Candidate 1 is rejected or accepted and you are moving to the next step.
Change only:
- validation window from current accepted value to `48m`

#### Candidate 3 — lambda smoothing
Keep the same lambda grid, but instead of selecting the single best lambda from one validation slice, choose the lambda with the best **median** validation Sharpe across the most recent 3 validation folds available at each decision point.

#### Candidate 4 — z-score clipping
Clip all macro z-scores used in Strategy 3 to:
- `z = max(min(z, 3), -3)`
No other change.

#### Candidate 5 — factor weight cap
After mapping raw weights into final factor weights and before final normalisation, cap each factor weight:
- `|W_i| <= 0.60`
Then renormalise gross exposure back to `sum(abs(W_i)) = 1`.

#### Candidate 6 — damp interaction contribution
Replace
`W_i(t+1) = w_raw_i + sum_p [w_interaction(i,p) * z_p(t)]`
with
`W_i(t+1) = w_raw_i + gamma * sum_p [w_interaction(i,p) * z_p(t)]`
using `gamma = 0.50`.

#### Candidate 7 — stronger damping
Same as Candidate 6, but set:
- `gamma = 0.25`

#### Candidate 8 — covariance shrinkage
Shrink the sample covariance matrix toward its diagonal target:
- `Sigma_shrunk = (1 - rho) * Sigma + rho * diag(diag(Sigma))`
with `rho = 0.25`.
Use `Sigma_shrunk` in the optimiser. No other change.

#### Candidate 9 — stronger covariance shrinkage
Same as Candidate 8, but set:
- `rho = 0.50`

#### Candidate 10 — turnover-aware acceptance overlay
Do **not** alter returns directly inside the optimiser. Instead, when selecting among candidate lambdas during validation, rank lambdas by:
- primary = `net Sharpe at 25 bps`
- tie-breaker = lower turnover
This changes model selection, not portfolio arithmetic.

#### Candidate 11 — add VIX as the first extra predictor
Only test this **after** Candidates 1–10 have been processed.
Expand predictors from `{z_TERM, z_DEF}` to `{z_TERM, z_DEF, z_VIX}`.
This creates:
- 3 raw factor series
- 9 interaction series
- 12 total optimisation series
Keep all accepted stabilisation upgrades in place.

#### Candidate 12 — VIX plus clipping
Only test if Candidate 11 is accepted.
Add predictor `z_VIX` and keep z-score clipping from Candidate 4.

---

## ITERATION PROTOCOL

For each candidate in the queue:

1. Start from the **current accepted version**, not from the raw baseline.
2. Apply exactly one candidate change.
3. Rebuild the full strategy.
4. Run the full OOS backtest.
5. Compute all metrics.
6. Compare candidate metrics to `accepted_metrics` using the acceptance rules.
7. If accepted:
   - promote candidate to `accepted_version`
   - replace `accepted_metrics`
   - save all outputs
   - log decision = `ACCEPT`
8. If rejected:
   - discard the candidate implementation
   - keep the previous `accepted_version`
   - log decision = `REJECT`
9. Move to the next candidate.

After all candidates have been processed once:
- if **at least one** candidate was accepted during the pass, start a new pass from the beginning of the queue,
- if **zero** candidates were accepted, stop the loop.

This means the procedure continues until it reaches a local optimum under the pre-committed candidate set.

---

## OUTPUT LOG FORMAT

Create and update a CSV called:
- `strategy_3A_T1_upgrade_log.csv`

Required columns:
- `iteration_id`
- `pass_number`
- `candidate_name`
- `parent_version`
- `candidate_version`
- `accepted` (`YES`/`NO`)
- `reason`
- `gross_sharpe_old`
- `gross_sharpe_new`
- `net_sharpe_25_old`
- `net_sharpe_25_new`
- `net_sharpe_50_old`
- `net_sharpe_50_new`
- `ff5_alpha_old`
- `ff5_alpha_new`
- `max_dd_old`
- `max_dd_new`
- `turnover_old`
- `turnover_new`
- `lambda_extreme_share_old`
- `lambda_extreme_share_new`
- `weight_instability_old`
- `weight_instability_new`
- `live_months_old`
- `live_months_new`

The `reason` field must be short and explicit, for example:
- `accepted: net_sharpe_25 improved and lambda_extreme_share fell`
- `rejected: net_sharpe_25 worse`
- `rejected: max drawdown worsened beyond guardrail`

---

## REQUIRED DIAGNOSTICS AFTER EVERY ACCEPTED CHANGE

After each accepted change, regenerate:
1. lambda path plot
2. factor weight evolution plot
3. rolling 36m Sharpe vs equal-weight baseline
4. cumulative return plot vs equal-weight baseline
5. transaction-cost table at `0 / 25 / 50 / 75 bps`
6. FF5 alpha regression output

If an accepted change improves Sharpe but makes the model visually erratic, do **not** override the acceptance rule manually. Log the issue and continue. The loop must remain deterministic.

---

## STOPPING CONDITIONS

Stop immediately if any of these occur:
1. one full candidate pass produces zero accepted changes,
2. more than 3 consecutive accepted changes produce only trivial improvements (`net_sharpe_25` gain < 0.01 each),
3. live sample shrinks,
4. the strategy becomes impossible to evaluate consistently against the same benchmark set.

When the loop stops, label the final model:
- `Strategy_3A_T1_final_accepted`

---

## FINAL REPORT TEMPLATE

At the end, print a concise summary:

1. baseline metrics
2. final accepted metrics
3. accepted changes in chronological order
4. rejected changes and reason
5. final lambda distribution
6. final turnover and weight instability
7. whether VIX was needed or not

Then state one final verdict:
- `YES` if the final accepted version improves on the baseline under the acceptance rules
- `NO` if no accepted version beats the baseline

---

## IMPLEMENTATION NOTES

- Preserve reproducibility with fixed seeds where relevant.
- Reuse the same aligned monthly dataset as the main guide.
- Reuse the same FF5 alpha framework and transaction cost framework as the main guide.
- Do not introduce five-variable macro expansions in this loop. That belongs to a later robustness stage only.
- Do not add SMB to Strategy 3. The factor universe stays `RMW, CMA, HML`.

