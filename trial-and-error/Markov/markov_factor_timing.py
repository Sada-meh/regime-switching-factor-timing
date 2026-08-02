from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from credit_factor_rotation_core import compute_max_drawdown, load_ff5


OUTPUT_DIR = Path(__file__).resolve().parent
DATA_SUMMARY_CSV = OUTPUT_DIR / "markov_data_summary.csv"
SPREAD_DIAGNOSTICS_CSV = OUTPUT_DIR / "markov_spread_diagnostics.csv"
MODEL_SUMMARY_CSV = OUTPUT_DIR / "markov_model_summary.csv"
PERFORMANCE_CSV = OUTPUT_DIR / "markov_performance_summary.csv"
COST_CSV = OUTPUT_DIR / "markov_transaction_cost_summary.csv"
ROBUSTNESS_CSV = OUTPUT_DIR / "markov_robustness_summary.csv"
MODEL_RISK_CSV = OUTPUT_DIR / "markov_model_risk_diagnostics.csv"
BACKTEST_CSV = OUTPUT_DIR / "markov_recursive_backtest.csv"
REESTIMATION_CSV = OUTPUT_DIR / "markov_reestimation_robustness.csv"
DECISION_MD = OUTPUT_DIR / "markov_final_decision.md"
ROBUSTNESS_LOG_JSON = OUTPUT_DIR / "markov_robustness_log.json"

SPREAD_PLOT = OUTPUT_DIR / "markov_spread.png"
AUTOCORR_PLOT = OUTPUT_DIR / "markov_spread_autocorrelation.png"
PROBABILITY_PLOT = OUTPUT_DIR / "markov_filtered_probabilities.png"
CUMULATIVE_PLOT = OUTPUT_DIR / "markov_cumulative_returns.png"
COST_PLOT = OUTPUT_DIR / "markov_transaction_costs.png"

MIN_TRAIN_MONTHS = 60
BASELINE_UPPER = 0.60
BASELINE_LOWER = 0.40
TRADING_COSTS = {
    "0bps": 0.0000,
    "25bps": 0.0025,
    "50bps": 0.0050,
    "75bps": 0.0075,
}


@dataclass
class MarkovFitSummary:
    converged: bool
    llf: float
    aic: float
    bic: float
    cyclical_state: int
    defensive_state: int
    cyclical_mean: float
    defensive_mean: float
    mean_gap: float
    transition_matrix: np.ndarray
    expected_durations: np.ndarray
    filtered_probability: pd.Series
    smoothed_probability: pd.Series
    warnings: list[str]
    regime_share_cyclical: float
    regime_share_defensive: float
    avg_filtered_prob_cyclical: float
    avg_filtered_prob_defensive: float
    avg_abs_filtered_smoothed_gap: float


def load_factor_panel() -> pd.DataFrame:
    ff5 = load_ff5().copy()
    data = ff5[["date", "Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]].copy()
    data["cyclical"] = 0.5 * (data["SMB"] + data["HML"])
    data["defensive"] = 0.5 * (data["RMW"] + data["CMA"])
    data["spread"] = data["cyclical"] - data["defensive"]
    data["benchmark_equal_sleeves"] = 0.5 * data["cyclical"] + 0.5 * data["defensive"]
    data["benchmark_cyclical_only"] = data["cyclical"]
    data["benchmark_defensive_only"] = data["defensive"]
    data["benchmark_all_factor_equal"] = 0.25 * (data["SMB"] + data["HML"] + data["RMW"] + data["CMA"])
    return data.sort_values("date").reset_index(drop=True)


def validate_factor_panel(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = ["SMB", "HML", "RMW", "CMA"]
    missing_required = [col for col in required if col not in data.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    duplicates = int(data["date"].duplicated().sum())
    strictly_increasing = bool(data["date"].is_monotonic_increasing and not data["date"].duplicated().any())
    missing_by_column = data[["SMB", "HML", "RMW", "CMA", "RF", "Mkt-RF"]].isna().sum()

    cleaned = data.dropna(subset=required).copy()
    dropped_rows = int(len(data) - len(cleaned))

    summary = pd.DataFrame(
        [
            {"metric": "sample_start", "value": str(cleaned["date"].min().date())},
            {"metric": "sample_end", "value": str(cleaned["date"].max().date())},
            {"metric": "n_months", "value": int(len(cleaned))},
            {"metric": "duplicate_dates", "value": duplicates},
            {"metric": "strictly_increasing_dates", "value": strictly_increasing},
            {"metric": "rows_dropped_for_missing_required", "value": dropped_rows},
        ]
        + [{"metric": f"missing_{col}", "value": int(val)} for col, val in missing_by_column.items()]
    )
    return cleaned.reset_index(drop=True), summary


def annualized_performance(returns: pd.Series) -> dict[str, float]:
    ret = pd.Series(returns).dropna()
    if ret.empty:
        return {
            "n_months": 0,
            "mean_annual": np.nan,
            "vol_annual": np.nan,
            "sharpe_annual": np.nan,
            "max_drawdown": np.nan,
        }
    mean_m = float(ret.mean())
    vol_m = float(ret.std(ddof=1))
    sharpe = (mean_m / vol_m) * np.sqrt(12) if pd.notna(vol_m) and vol_m > 0 else np.nan
    return {
        "n_months": int(len(ret)),
        "mean_annual": mean_m * 12,
        "vol_annual": vol_m * np.sqrt(12) if pd.notna(vol_m) else np.nan,
        "sharpe_annual": sharpe,
        "max_drawdown": compute_max_drawdown(ret),
    }


def run_factor_regressions(returns_df: pd.DataFrame, ff5: pd.DataFrame) -> pd.DataFrame:
    merged = returns_df.merge(ff5, on="date", how="inner")
    models = {
        "CAPM": ["Mkt-RF"],
        "FF3": ["Mkt-RF", "SMB", "HML"],
        "FF5": ["Mkt-RF", "SMB", "HML", "RMW", "CMA"],
    }
    rows: list[dict[str, float | str | int]] = []
    import statsmodels.api as sm

    for series in [col for col in returns_df.columns if col != "date"]:
        sample = merged[["date", series, "RF", "Mkt-RF", "SMB", "HML", "RMW", "CMA"]].dropna().copy()
        if sample.empty:
            continue
        sample["excess_ret"] = sample[series] - sample["RF"]

        for model_name, regressors in models.items():
            if len(sample) <= len(regressors):
                continue
            X = sm.add_constant(sample[regressors])
            y = sample["excess_ret"]
            result = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
            rows.append(
                {
                    "series": series,
                    "model": model_name,
                    "n_obs": int(result.nobs),
                    "alpha": float(result.params.get("const", np.nan)),
                    "alpha_t": float(result.tvalues.get("const", np.nan)),
                    "alpha_p": float(result.pvalues.get("const", np.nan)),
                    "r_squared": float(result.rsquared),
                }
            )
    return pd.DataFrame(rows)


def summarize_performance(returns_df: pd.DataFrame, ff5: pd.DataFrame, turnover: dict[str, pd.Series] | None = None) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    reg = run_factor_regressions(returns_df, ff5)
    ff5_alpha = reg.loc[reg["model"] == "FF5"].set_index("series") if not reg.empty else pd.DataFrame()

    for series in [col for col in returns_df.columns if col != "date"]:
        perf = annualized_performance(returns_df[series])
        row = {"series": series, **perf}
        row["avg_monthly_turnover"] = float(turnover[series].mean()) if turnover is not None and series in turnover else np.nan
        if not ff5_alpha.empty and series in ff5_alpha.index:
            row["ff5_alpha"] = float(ff5_alpha.loc[series, "alpha"])
            row["ff5_alpha_t"] = float(ff5_alpha.loc[series, "alpha_t"])
            row["ff5_alpha_p"] = float(ff5_alpha.loc[series, "alpha_p"])
            row["ff5_r_squared"] = float(ff5_alpha.loc[series, "r_squared"])
        else:
            row["ff5_alpha"] = np.nan
            row["ff5_alpha_t"] = np.nan
            row["ff5_alpha_p"] = np.nan
            row["ff5_r_squared"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def spread_diagnostics(data: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    spread = data["spread"].dropna()
    acf1 = float(spread.autocorr(lag=1))
    lb = acorr_ljungbox(spread, lags=[1, 3, 6, 12], return_df=True)

    import statsmodels.api as sm

    ar_sample = pd.DataFrame({"y": spread, "lag1": spread.shift(1)}).dropna()
    ar1 = sm.OLS(ar_sample["y"], sm.add_constant(ar_sample["lag1"])).fit()

    diag_rows = [
        {"metric": "mean_monthly", "value": float(spread.mean())},
        {"metric": "std_monthly", "value": float(spread.std(ddof=1))},
        {"metric": "lag1_autocorrelation", "value": acf1},
        {"metric": "ar1_beta", "value": float(ar1.params["lag1"])},
        {"metric": "ar1_pvalue", "value": float(ar1.pvalues["lag1"])},
    ]
    for lag, row in lb.iterrows():
        diag_rows.append({"metric": f"ljungbox_stat_lag_{lag}", "value": float(row["lb_stat"])})
        diag_rows.append({"metric": f"ljungbox_pvalue_lag_{lag}", "value": float(row["lb_pvalue"])})

    meaningful_serial_dependence = bool(
        (abs(acf1) >= 0.10 and ar1.pvalues["lag1"] < 0.05)
        or ((lb["lb_pvalue"] < 0.05).sum() >= 2 and abs(acf1) >= 0.08)
    )
    diag_rows.append({"metric": "meaningful_serial_dependence", "value": meaningful_serial_dependence})
    return pd.DataFrame(diag_rows), meaningful_serial_dependence


def fit_markov_model(
    spread: pd.Series,
    k_regimes: int = 2,
    switching_variance: bool = False,
    exog: pd.DataFrame | None = None,
    start_params: np.ndarray | None = None,
    search_reps: int = 0,
    maxiter: int = 200,
) -> tuple[object | None, list[str]]:
    y = pd.Series(spread).astype(float)
    model = MarkovRegression(
        y,
        k_regimes=k_regimes,
        trend="c",
        exog=exog,
        switching_variance=switching_variance,
        switching_exog=False if exog is not None else True,
    )
    caught: list[str] = []
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        try:
            fit_kwargs: dict[str, object] = {"disp": False, "maxiter": maxiter}
            if start_params is not None:
                fit_kwargs["start_params"] = start_params
            elif search_reps > 0:
                fit_kwargs["search_reps"] = search_reps
            result = model.fit(**fit_kwargs)
        except Exception as exc:
            caught.append(str(exc))
            return None, caught
        for warning_record in records:
            caught.append(str(warning_record.message))
    return result, caught


def _extract_state_means(result: object, k_regimes: int) -> dict[int, float]:
    params = result.params
    means: dict[int, float] = {}
    for state in range(k_regimes):
        means[state] = float(params.get(f"const[{state}]", np.nan))
    return means


def summarize_markov_fit(result: object, warnings_list: list[str], k_regimes: int = 2) -> MarkovFitSummary:
    means = _extract_state_means(result, k_regimes=k_regimes)
    cyclical_state = int(max(means, key=means.get))
    defensive_state = int(min(means, key=means.get))

    filtered = pd.Series(result.filtered_marginal_probabilities.iloc[:, cyclical_state], index=result.data.dates)
    smoothed = pd.Series(result.smoothed_marginal_probabilities.iloc[:, cyclical_state], index=result.data.dates)
    assigned = result.filtered_marginal_probabilities.idxmax(axis=1)
    trans = np.asarray(result.regime_transition[:, :, 0], dtype=float)
    expected_durations = np.asarray(result.expected_durations, dtype=float)

    return MarkovFitSummary(
        converged=bool(getattr(result, "mle_retvals", {}).get("converged", True)),
        llf=float(result.llf),
        aic=float(result.aic),
        bic=float(result.bic),
        cyclical_state=cyclical_state,
        defensive_state=defensive_state,
        cyclical_mean=float(means[cyclical_state]),
        defensive_mean=float(means[defensive_state]),
        mean_gap=float(means[cyclical_state] - means[defensive_state]),
        transition_matrix=trans,
        expected_durations=expected_durations,
        filtered_probability=filtered,
        smoothed_probability=smoothed,
        warnings=warnings_list,
        regime_share_cyclical=float((assigned == cyclical_state).mean()),
        regime_share_defensive=float((assigned == defensive_state).mean()),
        avg_filtered_prob_cyclical=float(filtered.mean()),
        avg_filtered_prob_defensive=float((1.0 - filtered).mean()),
        avg_abs_filtered_smoothed_gap=float((filtered - smoothed).abs().mean()),
    )


def fit_baseline_full_sample(data: pd.DataFrame) -> MarkovFitSummary:
    result, warn = fit_markov_model(data["spread"], search_reps=20, maxiter=200)
    if result is None:
        raise RuntimeError("Baseline full-sample Markov model failed to fit.")
    return summarize_markov_fit(result, warn, k_regimes=2)


def fit_ar1_full_sample(data: pd.DataFrame) -> MarkovFitSummary | None:
    sample = data[["date", "spread"]].copy()
    sample["spread_lag1"] = sample["spread"].shift(1)
    sample = sample.dropna().reset_index(drop=True)
    result, warn = fit_markov_model(
        sample["spread"],
        exog=sample[["spread_lag1"]],
        search_reps=10,
        maxiter=200,
    )
    if result is None:
        return None
    return summarize_markov_fit(result, warn, k_regimes=2)


def fit_switching_variance_full_sample(data: pd.DataFrame) -> MarkovFitSummary | None:
    result, warn = fit_markov_model(data["spread"], switching_variance=True, search_reps=10, maxiter=200)
    if result is None:
        return None
    return summarize_markov_fit(result, warn, k_regimes=2)


def fit_three_state_full_sample(data: pd.DataFrame) -> MarkovFitSummary | None:
    result, warn = fit_markov_model(data["spread"], k_regimes=3, search_reps=10, maxiter=250)
    if result is None:
        return None
    return summarize_markov_fit(result, warn, k_regimes=3)


def compute_vol_scaled_sleeves(data: pd.DataFrame, lookback: int = 12) -> pd.DataFrame:
    scaled = data[["date", "SMB", "HML", "RMW", "CMA"]].copy()
    for col in ["SMB", "HML", "RMW", "CMA"]:
        scaled[f"vol_{col}"] = scaled[col].rolling(lookback, min_periods=lookback).std(ddof=1).shift(1)
        scaled[f"invvol_{col}"] = 1.0 / scaled[f"vol_{col}"]

    cyc_denom = scaled["invvol_SMB"] + scaled["invvol_HML"]
    def_denom = scaled["invvol_RMW"] + scaled["invvol_CMA"]
    scaled["cyclical_volscaled"] = (
        (scaled["invvol_SMB"] / cyc_denom) * scaled["SMB"]
        + (scaled["invvol_HML"] / cyc_denom) * scaled["HML"]
    )
    scaled["defensive_volscaled"] = (
        (scaled["invvol_RMW"] / def_denom) * scaled["RMW"]
        + (scaled["invvol_CMA"] / def_denom) * scaled["CMA"]
    )
    return scaled[["date", "cyclical_volscaled", "defensive_volscaled"]]


def probability_to_weights(prob: float, upper: float, lower: float) -> tuple[float, float]:
    if prob > upper:
        return 1.0, 0.0
    if prob < lower:
        return 0.0, 1.0
    return 0.5, 0.5


def recursive_backtest(
    data: pd.DataFrame,
    upper: float = BASELINE_UPPER,
    lower: float = BASELINE_LOWER,
    cyclical_col: str = "cyclical",
    defensive_col: str = "defensive",
    use_ar1: bool = False,
    min_train_months: int = MIN_TRAIN_MONTHS,
) -> pd.DataFrame:
    rows: list[dict[str, float | str | bool | pd.Timestamp]] = []
    prev_params: np.ndarray | None = None
    prev_prob = 0.5

    for end_idx in range(min_train_months - 1, len(data) - 1):
        train = data.iloc[: end_idx + 1].copy()
        signal_date = pd.Timestamp(train["date"].iloc[-1])
        next_row = data.iloc[end_idx + 1]

        if use_ar1:
            train["spread_lag1"] = train["spread"].shift(1)
            aligned = train.dropna(subset=["spread", "spread_lag1"]).reset_index(drop=True)
            if len(aligned) <= min_train_months - 1:
                continue
            result, fit_warnings = fit_markov_model(
                aligned["spread"],
                exog=aligned[["spread_lag1"]],
                start_params=prev_params,
                maxiter=100,
            )
        else:
            result, fit_warnings = fit_markov_model(
                train["spread"],
                start_params=prev_params,
                maxiter=100,
            )

        used_fallback = False
        converged = False
        mean_gap = np.nan

        if result is None:
            p_cyc = prev_prob
            used_fallback = True
            fit_warnings = fit_warnings + ["fit_failure_fallback"]
        else:
            fit_summary = summarize_markov_fit(result, fit_warnings)
            p_cyc = float(fit_summary.filtered_probability.iloc[-1])
            prev_prob = p_cyc
            prev_params = np.asarray(result.params, dtype=float)
            converged = fit_summary.converged
            mean_gap = fit_summary.mean_gap

        w_cyc, w_def = probability_to_weights(p_cyc, upper=upper, lower=lower)
        rows.append(
            {
                "signal_date": signal_date,
                "holding_date": pd.Timestamp(next_row["date"]),
                "p_cyclical": p_cyc,
                "w_cyclical": w_cyc,
                "w_defensive": w_def,
                "cyclical_next": float(next_row[cyclical_col]),
                "defensive_next": float(next_row[defensive_col]),
                "strategy_gross": w_cyc * float(next_row[cyclical_col]) + w_def * float(next_row[defensive_col]),
                "converged": converged,
                "used_fallback": used_fallback,
                "mean_gap_fit": mean_gap,
                "warning_count": len(fit_warnings),
            }
        )

    backtest = pd.DataFrame(rows).sort_values("holding_date").reset_index(drop=True)
    if backtest.empty:
        raise RuntimeError("Recursive backtest produced no observations.")
    backtest["turnover"] = (
        backtest["w_cyclical"].diff().abs().fillna(0.0) + backtest["w_defensive"].diff().abs().fillna(0.0)
    )
    for label, cost in TRADING_COSTS.items():
        backtest[f"strategy_{label}"] = backtest["strategy_gross"] - cost * backtest["turnover"]
    return backtest


def build_benchmark_frame(data: pd.DataFrame, backtest: pd.DataFrame) -> pd.DataFrame:
    dates = backtest[["holding_date"]].rename(columns={"holding_date": "date"})
    aligned = dates.merge(
        data[
            [
                "date",
                "benchmark_equal_sleeves",
                "benchmark_cyclical_only",
                "benchmark_defensive_only",
                "benchmark_all_factor_equal",
            ]
        ],
        on="date",
        how="left",
    )
    return aligned


def plot_spread(data: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(data["date"], data["spread"], color="#234f7d", linewidth=1.2)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_title("Cyclical Minus Defensive Spread")
    ax.set_ylabel("Monthly Return")
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(SPREAD_PLOT, dpi=180)
    plt.close(fig)


def plot_autocorrelation(data: pd.DataFrame) -> None:
    spread = data["spread"].dropna()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(spread, ax=axes[0], lags=24, zero=False)
    axes[0].set_title("ACF of Spread")
    plot_pacf(spread, ax=axes[1], lags=24, zero=False, method="ywm")
    axes[1].set_title("PACF of Spread")
    fig.tight_layout()
    fig.savefig(AUTOCORR_PLOT, dpi=180)
    plt.close(fig)


def plot_probabilities(data: pd.DataFrame, baseline_fit: MarkovFitSummary) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(data["date"], data["spread"], color="#234f7d", linewidth=1.0)
    axes[0].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axes[0].set_title("Spread and Markov Regime Probabilities")
    axes[0].set_ylabel("Spread")

    axes[1].plot(
        baseline_fit.filtered_probability.index,
        baseline_fit.filtered_probability.values,
        color="#1b9e77",
        label="Filtered P(cyclical)",
        linewidth=1.3,
    )
    axes[1].plot(
        baseline_fit.smoothed_probability.index,
        baseline_fit.smoothed_probability.values,
        color="#d95f02",
        label="Smoothed P(cyclical)",
        linewidth=1.0,
        alpha=0.75,
    )
    axes[1].axhline(BASELINE_UPPER, color="gray", linestyle="--", linewidth=0.8)
    axes[1].axhline(BASELINE_LOWER, color="gray", linestyle="--", linewidth=0.8)
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_ylabel("Probability")
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(PROBABILITY_PLOT, dpi=180)
    plt.close(fig)


def plot_cumulative(backtest: pd.DataFrame, benchmarks: pd.DataFrame) -> None:
    frame = backtest[["holding_date", "strategy_gross", "strategy_50bps"]].rename(columns={"holding_date": "date"})
    frame = frame.merge(benchmarks, on="date", how="left")
    fig, ax = plt.subplots(figsize=(12, 6))
    for col, color in [
        ("strategy_gross", "#0d3b66"),
        ("strategy_50bps", "#2a9d8f"),
        ("benchmark_equal_sleeves", "#6c757d"),
        ("benchmark_cyclical_only", "#e76f51"),
        ("benchmark_defensive_only", "#577590"),
    ]:
        wealth = (1.0 + frame[col].fillna(0.0)).cumprod()
        ax.plot(frame["date"], wealth, label=col, linewidth=1.3, color=color)
    ax.set_title("Cumulative Returns: Markov Strategy vs Benchmarks")
    ax.set_ylabel("Growth of $1")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(CUMULATIVE_PLOT, dpi=180)
    plt.close(fig)


def plot_costs(backtest: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    for col, color in [
        ("strategy_0bps", "#0d3b66"),
        ("strategy_25bps", "#2a9d8f"),
        ("strategy_50bps", "#e9c46a"),
        ("strategy_75bps", "#e76f51"),
    ]:
        wealth = (1.0 + backtest[col].fillna(0.0)).cumprod()
        ax.plot(backtest["holding_date"], wealth, label=col, linewidth=1.3, color=color)
    ax.set_title("Markov Strategy Under Transaction Costs")
    ax.set_ylabel("Growth of $1")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(COST_PLOT, dpi=180)
    plt.close(fig)


def fit_reestimation_robustness(data: pd.DataFrame) -> pd.DataFrame:
    sample = data["spread"]
    base_model = MarkovRegression(sample, k_regimes=2, trend="c", switching_variance=False)
    baseline_start = np.asarray(base_model.start_params, dtype=float)
    rows: list[dict[str, float | int | bool]] = []

    for seed in [11, 22, 33]:
        rng = np.random.default_rng(seed)
        start = baseline_start + rng.normal(0.0, 0.05, size=len(baseline_start))
        result, warn = fit_markov_model(sample, start_params=start, maxiter=200)
        if result is None:
            rows.append({"seed": seed, "converged": False, "llf": np.nan, "cyclical_mean": np.nan, "defensive_mean": np.nan, "mean_gap": np.nan})
            continue
        fit = summarize_markov_fit(result, warn)
        rows.append(
            {
                "seed": seed,
                "converged": fit.converged,
                "llf": fit.llf,
                "cyclical_mean": fit.cyclical_mean,
                "defensive_mean": fit.defensive_mean,
                "mean_gap": fit.mean_gap,
            }
        )
    return pd.DataFrame(rows)


def build_model_summary_rows(
    baseline: MarkovFitSummary,
    ar1_fit: MarkovFitSummary | None,
    switching_var_fit: MarkovFitSummary | None,
    three_state_fit: MarkovFitSummary | None,
) -> pd.DataFrame:
    rows = [
        {
            "model": "baseline_2state",
            "converged": baseline.converged,
            "llf": baseline.llf,
            "aic": baseline.aic,
            "bic": baseline.bic,
            "cyclical_mean": baseline.cyclical_mean,
            "defensive_mean": baseline.defensive_mean,
            "mean_gap": baseline.mean_gap,
            "expected_duration_cyclical": float(baseline.expected_durations[baseline.cyclical_state]),
            "expected_duration_defensive": float(baseline.expected_durations[baseline.defensive_state]),
            "avg_abs_filtered_smoothed_gap": baseline.avg_abs_filtered_smoothed_gap,
            "warning_count": len(baseline.warnings),
        }
    ]
    for name, fit in [
        ("ar1_2state", ar1_fit),
        ("switching_variance_2state", switching_var_fit),
        ("three_state_diagnostic", three_state_fit),
    ]:
        if fit is None:
            rows.append({"model": name, "converged": False})
            continue
        rows.append(
            {
                "model": name,
                "converged": fit.converged,
                "llf": fit.llf,
                "aic": fit.aic,
                "bic": fit.bic,
                "cyclical_mean": fit.cyclical_mean,
                "defensive_mean": fit.defensive_mean,
                "mean_gap": fit.mean_gap,
                "expected_duration_cyclical": float(fit.expected_durations[fit.cyclical_state]),
                "expected_duration_defensive": float(fit.expected_durations[fit.defensive_state]),
                "avg_abs_filtered_smoothed_gap": fit.avg_abs_filtered_smoothed_gap,
                "warning_count": len(fit.warnings),
            }
        )
    return pd.DataFrame(rows)


def build_threshold_robustness(backtest: pd.DataFrame, data: pd.DataFrame, ff5: pd.DataFrame) -> pd.DataFrame:
    probabilities = backtest[["holding_date", "p_cyclical"]].rename(columns={"holding_date": "date"})
    aligned_returns = probabilities.merge(data[["date", "cyclical", "defensive"]], on="date", how="left")
    strategies: dict[str, pd.Series] = {}

    for name, upper, lower in [
        ("Threshold_55_45", 0.55, 0.45),
        ("Threshold_60_40", 0.60, 0.40),
        ("Threshold_65_35", 0.65, 0.35),
    ]:
        weights = aligned_returns["p_cyclical"].apply(lambda p: probability_to_weights(float(p), upper, lower))
        frame = pd.DataFrame(weights.tolist(), columns=["w_cyc", "w_def"])
        strategies[name] = frame["w_cyc"] * aligned_returns["cyclical"] + frame["w_def"] * aligned_returns["defensive"]
    results = pd.concat([aligned_returns[["date"]], pd.DataFrame(strategies)], axis=1)
    return summarize_performance(results, ff5)


def build_volscaled_robustness(backtest: pd.DataFrame, data: pd.DataFrame, ff5: pd.DataFrame) -> pd.DataFrame:
    scaled = compute_vol_scaled_sleeves(data)
    probabilities = backtest[["holding_date", "p_cyclical"]].rename(columns={"holding_date": "date"})
    frame = probabilities.merge(scaled, on="date", how="left").dropna().reset_index(drop=True)
    weights = frame["p_cyclical"].apply(lambda p: probability_to_weights(float(p), BASELINE_UPPER, BASELINE_LOWER))
    w = pd.DataFrame(weights.tolist(), columns=["w_cyc", "w_def"])
    returns = pd.DataFrame(
        {
            "date": frame["date"],
            "VolScaled_Sleeves": w["w_cyc"] * frame["cyclical_volscaled"] + w["w_def"] * frame["defensive_volscaled"],
        }
    )
    return summarize_performance(returns, ff5)


def build_sample_split_summary(backtest: pd.DataFrame, ff5: pd.DataFrame) -> pd.DataFrame:
    split = pd.Timestamp("2007-12-31")
    rows: list[pd.DataFrame] = []
    base = backtest[["holding_date", "strategy_gross"]].rename(columns={"holding_date": "date"})
    for label, mask in [("pre_2008", base["date"] <= split), ("post_2007", base["date"] > split)]:
        perf = summarize_performance(base.loc[mask].copy(), ff5)
        perf["period"] = label
        rows.append(perf)
    return pd.concat(rows, ignore_index=True)


def build_model_risk_diagnostics(
    baseline: MarkovFitSummary,
    reestimation: pd.DataFrame,
    backtest: pd.DataFrame,
    threshold_summary: pd.DataFrame,
    cost_summary: pd.DataFrame,
) -> pd.DataFrame:
    threshold_6040 = threshold_summary.loc[threshold_summary["series"] == "Threshold_60_40", "sharpe_annual"]
    threshold_5545 = threshold_summary.loc[threshold_summary["series"] == "Threshold_55_45", "sharpe_annual"]
    threshold_6535 = threshold_summary.loc[threshold_summary["series"] == "Threshold_65_35", "sharpe_annual"]
    strategy_0 = cost_summary.loc[cost_summary["series"] == "strategy_0bps", "sharpe_annual"]
    strategy_50 = cost_summary.loc[cost_summary["series"] == "strategy_50bps", "sharpe_annual"]

    risk_rows = [
        {
            "check": "convergence_warnings",
            "flag": len(baseline.warnings) > 0,
            "details": "; ".join(baseline.warnings[:5]) if baseline.warnings else "none",
        },
        {
            "check": "nearly_identical_regime_means",
            "flag": baseline.mean_gap < 0.0025,
            "details": f"mean_gap={baseline.mean_gap:.4f}",
        },
        {
            "check": "low_regime_persistence",
            "flag": float(np.nanmin(baseline.expected_durations)) < 3.0,
            "details": f"durations={baseline.expected_durations.round(2).tolist()}",
        },
        {
            "check": "rare_regime",
            "flag": min(baseline.regime_share_cyclical, baseline.regime_share_defensive) < 0.15,
            "details": f"shares={[round(baseline.regime_share_cyclical, 3), round(baseline.regime_share_defensive, 3)]}",
        },
        {
            "check": "probabilities_oscillate_excessively",
            "flag": float((backtest["turnover"] > 1.0).mean()) > 0.25,
            "details": f"high_turnover_share={(backtest['turnover'] > 1.0).mean():.3f}",
        },
        {
            "check": "reestimation_instability",
            "flag": reestimation["llf"].dropna().std(ddof=1) > 5 or not bool(reestimation["converged"].fillna(False).all()),
            "details": f"llf_std={reestimation['llf'].dropna().std(ddof=1):.4f}",
        },
        {
            "check": "threshold_fragility",
            "flag": not (
                len(threshold_6040) == 1
                and len(threshold_5545) == 1
                and len(threshold_6535) == 1
                and abs(float(threshold_5545.iloc[0]) - float(threshold_6040.iloc[0])) <= 0.30
                and abs(float(threshold_6535.iloc[0]) - float(threshold_6040.iloc[0])) <= 0.30
            ),
            "details": f"sharpes={threshold_summary[['series','sharpe_annual']].to_dict(orient='records')}",
        },
        {
            "check": "cost_fragility_50bps",
            "flag": len(strategy_0) == 1 and len(strategy_50) == 1 and float(strategy_50.iloc[0]) <= 0.0,
            "details": f"sharpe_0bps={float(strategy_0.iloc[0]):.3f}, sharpe_50bps={float(strategy_50.iloc[0]):.3f}",
        },
    ]
    return pd.DataFrame(risk_rows)


def final_decision(
    baseline: MarkovFitSummary,
    meaningful_serial_dependence: bool,
    ar1_fit: MarkovFitSummary | None,
    performance: pd.DataFrame,
    cost_summary: pd.DataFrame,
    threshold_summary: pd.DataFrame,
    sample_split: pd.DataFrame,
    risk_diag: pd.DataFrame,
) -> tuple[str, pd.DataFrame, str]:
    perf = performance.set_index("series")
    cost = cost_summary.set_index("series")

    strategy_sharpe = float(perf.loc["strategy_gross", "sharpe_annual"])
    equal_sharpe = float(perf.loc["benchmark_equal_sleeves", "sharpe_annual"])
    defensive_sharpe = float(perf.loc["benchmark_defensive_only", "sharpe_annual"])
    net_50_sharpe = float(cost.loc["strategy_50bps", "sharpe_annual"])
    pre_sharpe = float(sample_split.loc[sample_split["period"] == "pre_2008", "sharpe_annual"].iloc[0])
    post_sharpe = float(sample_split.loc[sample_split["period"] == "post_2007", "sharpe_annual"].iloc[0])

    model_stability = (
        baseline.converged
        and baseline.mean_gap >= 0.0025
        and float(np.nanmin(baseline.expected_durations)) >= 3.0
        and not bool(risk_diag.loc[risk_diag["check"] == "reestimation_instability", "flag"].iloc[0])
    )
    economic_interpretability = (
        baseline.mean_gap >= 0.0025
        and min(baseline.regime_share_cyclical, baseline.regime_share_defensive) >= 0.15
        and baseline.avg_abs_filtered_smoothed_gap <= 0.25
    )
    autocorr_lag_choice = (
        (not meaningful_serial_dependence)
        or (ar1_fit is not None and ar1_fit.converged and ar1_fit.aic < baseline.aic)
    )
    out_of_sample_value_add = strategy_sharpe >= max(equal_sharpe + 0.10, defensive_sharpe + 0.05)
    transaction_cost_survival = net_50_sharpe > 0.0 and net_50_sharpe >= equal_sharpe
    threshold_sharpes = threshold_summary.set_index("series")["sharpe_annual"]
    robustness_support = (
        threshold_sharpes.min() > 0.0
        and pre_sharpe > 0.0
        and post_sharpe > 0.0
        and not bool(risk_diag["flag"].any())
    )

    checklist = pd.DataFrame(
        [
            {"criterion": "model stability", "pass": model_stability},
            {"criterion": "economic interpretability", "pass": economic_interpretability},
            {"criterion": "autocorrelation / lag choice", "pass": autocorr_lag_choice},
            {"criterion": "out-of-sample value add", "pass": out_of_sample_value_add},
            {"criterion": "transaction-cost survival", "pass": transaction_cost_survival},
            {"criterion": "robustness support", "pass": robustness_support},
        ]
    )
    decision = "YES" if bool(checklist["pass"].all()) else "NO"
    paragraph = (
        "The baseline 2-state spread model is evaluated on stability, interpretability, lag handling, "
        "out-of-sample value add, transaction-cost survival, and robustness support. "
        f"The recursive strategy posts a gross Sharpe of {strategy_sharpe:.3f} versus {equal_sharpe:.3f} "
        f"for the equal-sleeve benchmark and {defensive_sharpe:.3f} for the defensive sleeve, "
        f"with a 50 bps net Sharpe of {net_50_sharpe:.3f}. "
        "The final YES/NO line follows the strict pass-all-criteria rule from the implementation spec."
    )
    return decision, checklist, paragraph


def save_final_decision(paragraph: str, checklist: pd.DataFrame, decision: str) -> None:
    lines = ["## Final decision", "", paragraph, ""]
    for _, row in checklist.iterrows():
        label = "PASS" if bool(row["pass"]) else "FAIL"
        lines.append(f"- {row['criterion']}: {label}")
    lines.extend(["", decision, ""])
    DECISION_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> str:
    data = load_factor_panel()
    data, data_summary = validate_factor_panel(data)
    ff_for_eval = data[["date", "Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]].copy()

    plot_spread(data)
    plot_autocorrelation(data)

    diagnostics, meaningful_serial_dependence = spread_diagnostics(data)
    baseline_fit = fit_baseline_full_sample(data)
    ar1_fit = fit_ar1_full_sample(data) if meaningful_serial_dependence else None
    switching_var_fit = fit_switching_variance_full_sample(data)
    three_state_fit = fit_three_state_full_sample(data)
    plot_probabilities(data, baseline_fit)

    backtest = recursive_backtest(data)
    benchmarks = build_benchmark_frame(data, backtest)
    plot_cumulative(backtest, benchmarks)
    plot_costs(backtest)

    performance_frame = backtest[["holding_date", "strategy_gross"]].rename(columns={"holding_date": "date"})
    for benchmark_col in [
        "benchmark_equal_sleeves",
        "benchmark_cyclical_only",
        "benchmark_defensive_only",
        "benchmark_all_factor_equal",
    ]:
        performance_frame = performance_frame.merge(benchmarks[["date", benchmark_col]], on="date", how="left")
    turnover_frame = {"strategy_gross": backtest["turnover"]}
    performance_summary = summarize_performance(performance_frame, ff_for_eval, turnover=turnover_frame)

    cost_frame = backtest[["holding_date"] + [f"strategy_{label}" for label in TRADING_COSTS]].rename(columns={"holding_date": "date"})
    cost_turnover = {series: backtest["turnover"] for series in [col for col in cost_frame.columns if col != "date"]}
    cost_summary = summarize_performance(cost_frame, ff_for_eval, turnover=cost_turnover)

    threshold_summary = build_threshold_robustness(backtest, data, ff_for_eval)
    volscaled_summary = build_volscaled_robustness(backtest, data, ff_for_eval)
    sample_split = build_sample_split_summary(backtest, ff_for_eval)
    reestimation = fit_reestimation_robustness(data)
    risk_diag = build_model_risk_diagnostics(baseline_fit, reestimation, backtest, threshold_summary, cost_summary)

    robustness_rows = [
        {"check": "threshold_robustness", "status": "pass" if threshold_summary["sharpe_annual"].min() > 0 else "fail", "details": threshold_summary[["series", "sharpe_annual"]].round(4).to_dict(orient="records")},
        {"check": "sleeve_definition_volscaled", "status": "pass" if float(volscaled_summary["sharpe_annual"].iloc[0]) > 0 else "fail", "details": volscaled_summary.round(4).to_dict(orient="records")},
        {"check": "lag_robustness", "status": "pass" if (not meaningful_serial_dependence) or (ar1_fit is not None and ar1_fit.converged) else "fail", "details": "AR(1) tested" if meaningful_serial_dependence else "lag extension not warranted by diagnostics"},
        {"check": "three_state_diagnostic", "status": "pass" if (three_state_fit is not None and three_state_fit.converged) else "fail", "details": "diagnostic only"},
        {"check": "switching_variance", "status": "pass" if (switching_var_fit is not None and switching_var_fit.converged) else "fail", "details": "diagnostic only"},
        {"check": "sample_split", "status": "pass" if (sample_split["sharpe_annual"] > 0).all() else "fail", "details": sample_split.round(4).to_dict(orient="records")},
        {"check": "benchmark_robustness", "status": "pass" if float(performance_summary.loc[performance_summary["series"] == "strategy_gross", "sharpe_annual"].iloc[0]) > float(performance_summary.loc[performance_summary["series"] == "benchmark_equal_sleeves", "sharpe_annual"].iloc[0]) else "fail", "details": performance_summary[["series", "sharpe_annual"]].round(4).to_dict(orient="records")},
        {"check": "probability_source_revision", "status": "pass" if baseline_fit.avg_abs_filtered_smoothed_gap < 0.20 else "fail", "details": {"avg_abs_gap": round(baseline_fit.avg_abs_filtered_smoothed_gap, 4)}},
        {"check": "reestimation_robustness", "status": "pass" if not bool(risk_diag.loc[risk_diag["check"] == "reestimation_instability", "flag"].iloc[0]) else "fail", "details": reestimation.round(4).to_dict(orient="records")},
    ]
    robustness_summary = pd.DataFrame(robustness_rows)
    model_summary = build_model_summary_rows(baseline_fit, ar1_fit, switching_var_fit, three_state_fit)
    backtest_export = backtest.rename(columns={"holding_date": "date"})

    data_summary.to_csv(DATA_SUMMARY_CSV, index=False)
    diagnostics.to_csv(SPREAD_DIAGNOSTICS_CSV, index=False)
    model_summary.to_csv(MODEL_SUMMARY_CSV, index=False)
    performance_summary.to_csv(PERFORMANCE_CSV, index=False)
    cost_summary.to_csv(COST_CSV, index=False)
    robustness_summary.to_csv(ROBUSTNESS_CSV, index=False)
    risk_diag.to_csv(MODEL_RISK_CSV, index=False)
    backtest_export.to_csv(BACKTEST_CSV, index=False)
    reestimation.to_csv(REESTIMATION_CSV, index=False)
    ROBUSTNESS_LOG_JSON.write_text(json.dumps(robustness_rows, indent=2, default=str), encoding="utf-8")

    decision, checklist, paragraph = final_decision(
        baseline=baseline_fit,
        meaningful_serial_dependence=meaningful_serial_dependence,
        ar1_fit=ar1_fit,
        performance=performance_summary,
        cost_summary=cost_summary,
        threshold_summary=threshold_summary,
        sample_split=sample_split,
        risk_diag=risk_diag,
    )
    save_final_decision(paragraph, checklist, decision)

    print("Saved outputs:")
    for path in [
        DATA_SUMMARY_CSV,
        SPREAD_DIAGNOSTICS_CSV,
        MODEL_SUMMARY_CSV,
        PERFORMANCE_CSV,
        COST_CSV,
        ROBUSTNESS_CSV,
        MODEL_RISK_CSV,
        BACKTEST_CSV,
        REESTIMATION_CSV,
        DECISION_MD,
        SPREAD_PLOT,
        AUTOCORR_PLOT,
        PROBABILITY_PLOT,
        CUMULATIVE_PLOT,
        COST_PLOT,
    ]:
        print(f"  {path}")

    print("\nPerformance summary:")
    print(performance_summary.round(4).to_string(index=False))
    print("\nCost summary:")
    print(cost_summary.round(4).to_string(index=False))
    print("\nFinal decision:")
    print(decision)
    return decision


if __name__ == "__main__":
    main()
