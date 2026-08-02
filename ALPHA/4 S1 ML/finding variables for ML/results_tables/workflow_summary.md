# Variable Selection Workflow Summary

## Initial questions
1. Do we have enough data? Yes for a focused 5-feature FF-based workflow; no for a wide or self-built-heavy feature set.
2. Which variables can already be built? z_TERM, z_DEF, z_dTERM, z_dDEF, z_VIX, z_EPU, z_mkt_rf_3m, z_spread_3m, z_spread_12m, z_spread_vol_12m, z_cyclical_3m, z_defensive_3m
3. Which variables require extra data? IP growth, factor valuation spreads, and richer crowding/liquidity proxies require extra data.
4. Are any variables too redundant or weakly justified? z_cyclical_3m, z_defensive_3m, z_spread_12m, and z_EPU are the main overlap / weak-justification candidates.
5. Is the effective sample too small for a wide feature set? The aligned FF workflow supports about 413 usable monthly observations, so feature count should stay below 7.
6. What is the recommended final feature count? Recommended final feature count: 5 core variables, with at most 1 optional robustness variable.

## Recommended final predictor set

- z_mkt_rf_3m
- z_TERM
- z_dDEF
- z_dTERM
- z_spread_3m

## Out-of-sample note
The screened GBT model produced an out-of-sample Sharpe of 0.830 on the cyclical-versus-defensive timing task.
OOS months: 293. Robustness threshold met: Yes (minimum 60 months).