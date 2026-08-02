# Self-Built Variable Selection Workflow Summary

## Initial questions
1. Do we have enough data? Barely for a narrow workflow; not enough for a wide self-built feature set.
2. Which variables can already be built? z_TERM, z_DEF, z_dTERM, z_dDEF, z_VIX, z_EPU, z_mkt_rf_3m, z_spread_3m, z_spread_12m, z_spread_vol_12m, z_cyclical_3m, z_defensive_3m
3. Which variables require extra data? IP growth, factor valuation spreads, and richer crowding/liquidity proxies require extra data.
4. Are any variables too redundant or weakly justified? z_cyclical_3m, z_defensive_3m, z_spread_12m, and z_EPU remain the main overlap / weak-justification candidates.
5. Is the effective sample too small for a wide feature set? The aligned self-built workflow supports about 131 usable monthly observations, so feature count should stay well below 10.
6. What is the recommended final feature count? Recommended final feature count: 4 core variables, with at most 1-2 optional robustness variables.

## Recommended final predictor set

- z_mkt_rf_3m
- z_TERM
- z_dTERM
- z_spread_3m

## Out-of-sample note
The screened GBT model produced an out-of-sample Sharpe of -0.084 on the self-built cyclical-versus-defensive timing task.
OOS months: 11. Robustness threshold met: No (minimum 60 months).
The self-built workflow should be interpreted cautiously because the cyclical sleeve begins much later than the FF version.