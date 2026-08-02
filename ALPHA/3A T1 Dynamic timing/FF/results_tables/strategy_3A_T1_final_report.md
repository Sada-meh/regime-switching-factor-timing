# Strategy 3A-T1 Upgrade Loop Final Report

## Baseline Metrics
- gross_sharpe: 0.5364
- net_sharpe_25bps: 0.5359
- net_sharpe_50bps: 0.5353
- ff5_alpha_ann: -0.017259
- max_drawdown: -0.2467

## Final Accepted Metrics
- gross_sharpe: 0.5396
- net_sharpe_25bps: 0.5395
- net_sharpe_50bps: 0.5395
- ff5_alpha_ann: -0.017007
- ff5_alpha_t: -6.1138
- max_drawdown: -0.2466

## Accepted Changes
- candidate_2_validation_48m
- candidate_4_z_clip_3
- candidate_6_gamma_050
- candidate_7_gamma_025

## Rejected Changes
- candidate_1_validation_36m: rejected: net_sharpe_25 worse
- candidate_3_lambda_smoothing: rejected: net_sharpe_25 worse
- candidate_5_factor_cap_60: rejected: net_sharpe_25 worse
- stopped: more than 3 consecutive accepted changes produced trivial improvements

## Final Lambda Distribution
- lambda 1: 56.00%
- lambda 100000: 44.00%

## Final Turnover and Weight Instability
- turnover: 0.000182
- weight_instability: 0.000182
- avg_abs_delta_weight: 0.000061
- max_abs_weight: 0.337012
- clip_rate: 0.000000

## VIX Needed
- NO

## Final Verdict
- YES