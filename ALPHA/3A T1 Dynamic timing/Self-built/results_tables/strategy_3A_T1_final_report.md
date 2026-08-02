# Strategy 3A-T1 Upgrade Loop Final Report

## Baseline Metrics
- gross_sharpe: -0.3517
- net_sharpe_25bps: -0.3532
- net_sharpe_50bps: -0.3547
- ff5_alpha_ann: -0.038009
- max_drawdown: -0.0861

## Final Accepted Metrics
- gross_sharpe: -0.2975
- net_sharpe_25bps: -0.2991
- net_sharpe_50bps: -0.3006
- ff5_alpha_ann: -0.024667
- ff5_alpha_t: -0.2584
- max_drawdown: -0.0797

## Accepted Changes
- candidate_1_validation_36m
- candidate_2_validation_48m

## Rejected Changes
- candidate_3_lambda_smoothing: rejected: net_sharpe_25 worse
- candidate_4_z_clip_3: rejected: net_sharpe_25 worse
- candidate_5_factor_cap_60: rejected: net_sharpe_25 worse
- candidate_6_gamma_050: rejected: net_sharpe_25 worse
- candidate_7_gamma_025: rejected: net_sharpe_25 worse
- candidate_8_cov_shrink_025: rejected: net_sharpe_25 worse
- candidate_9_cov_shrink_050: rejected: net_sharpe_25 worse
- candidate_10_lambda_net25: rejected: net_sharpe_25 worse
- candidate_11_add_vix: rejected: turnover worsened beyond guardrail
- candidate_12_vix_plus_clip: rejected: candidate 11 not accepted previously
- candidate_1_validation_36m: rejected: net_sharpe_25 worse
- candidate_2_validation_48m: rejected: no change from accepted version
- candidate_3_lambda_smoothing: rejected: net_sharpe_25 worse
- candidate_4_z_clip_3: rejected: net_sharpe_25 worse
- candidate_5_factor_cap_60: rejected: net_sharpe_25 worse
- candidate_6_gamma_050: rejected: net_sharpe_25 worse
- candidate_7_gamma_025: rejected: net_sharpe_25 worse
- candidate_8_cov_shrink_025: rejected: net_sharpe_25 worse
- candidate_9_cov_shrink_050: rejected: net_sharpe_25 worse
- candidate_10_lambda_net25: rejected: net_sharpe_25 worse
- candidate_11_add_vix: rejected: turnover worsened beyond guardrail
- candidate_12_vix_plus_clip: rejected: candidate 11 not accepted previously

## Final Lambda Distribution
- lambda 1: 100.00%

## Final Turnover and Weight Instability
- turnover: 0.005147
- weight_instability: 0.005147
- avg_abs_delta_weight: 0.001838
- max_abs_weight: 0.369307
- clip_rate: 0.000000

## VIX Needed
- NO

## Final Verdict
- YES