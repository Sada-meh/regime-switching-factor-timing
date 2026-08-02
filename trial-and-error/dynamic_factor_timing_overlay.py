from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from credit_factor_rotation_core import (
    PROJECT_ROOT,
    descriptive_statistics,
    load_ff5,
    load_macro_feature_frame,
    prepare_alpha_table,
    prepare_summary_table,
    run_core_strategy,
    run_factor_regressions,
)


OUTPUT_RETURNS_CSV = PROJECT_ROOT / "dynamic_factor_timing_returns.csv"
OUTPUT_WEIGHTS_CSV = PROJECT_ROOT / "dynamic_factor_timing_weights.csv"
OUTPUT_SUMMARY_CSV = PROJECT_ROOT / "dynamic_factor_timing_summary_table.csv"
OUTPUT_ALPHA_CSV = PROJECT_ROOT / "dynamic_factor_timing_alpha_table.csv"
OUTPUT_COMPARISON_PLOT = PROJECT_ROOT / "dynamic_factor_timing_comparison.png"

FACTOR_COLS = ["R_prof", "R_inv", "R_lowvol"]
DEFAULT_WEIGHTS = np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)
LAMBDA_GRID = np.array([0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0], dtype=float)


@dataclass
class DynamicTimingArtifacts:
    predictor_panel: pd.DataFrame
    strategy: pd.DataFrame
    comparison: pd.DataFrame
    summary_table: pd.DataFrame
    alpha_table: pd.DataFrame


def _expanding_zscore(series: pd.Series, min_history: int = 24) -> pd.Series:
    mean_hist = series.expanding().mean().shift(1)
    std_hist = series.expanding().std(ddof=1).shift(1)
    z = (series - mean_hist) / std_hist.replace(0.0, np.nan)
    enough_history = np.arange(len(series)) >= min_history
    return z.where(enough_history)


def _abs_normalize(weights: np.ndarray) -> np.ndarray:
    gross = np.abs(weights).sum()
    if not np.isfinite(gross) or gross <= 0:
        return DEFAULT_WEIGHTS.copy()
    return weights / gross


def _solve_regularized_weights(mu: np.ndarray, sigma: np.ndarray, lam: float) -> np.ndarray:
    penalized = sigma + lam * np.eye(len(mu))
    rhs = mu + lam * DEFAULT_WEIGHTS
    try:
        raw = np.linalg.solve(penalized, rhs)
    except np.linalg.LinAlgError:
        raw = np.linalg.pinv(penalized) @ rhs
    return _abs_normalize(raw)


def _fit_linear_forecaster(x: pd.DataFrame, y: pd.Series) -> np.ndarray:
    x_design = np.column_stack([np.ones(len(x)), x.to_numpy()])
    beta, *_ = np.linalg.lstsq(x_design, y.to_numpy(), rcond=None)
    return beta


def _predict_linear_forecaster(beta: np.ndarray, x_row: pd.Series) -> float:
    x_vec = np.concatenate(([1.0], x_row.to_numpy(dtype=float)))
    return float(x_vec @ beta)


def build_predictor_panel() -> pd.DataFrame:
    core = run_core_strategy(write_csv=False)
    ff5 = load_ff5()[["date", "Mkt-RF", "RF"]]
    macro = load_macro_feature_frame(include_vix=True)

    panel = (
        core.factor_returns.merge(macro, on="date", how="inner")
        .merge(ff5, on="date", how="inner")
        .sort_values("date")
        .reset_index(drop=True)
    )

    panel["mktexcess_3m"] = panel["Mkt-RF"].rolling(3, min_periods=3).sum()

    for factor in FACTOR_COLS:
        short = factor.replace("R_", "")
        panel[f"{short}_ret_3m"] = panel[factor].rolling(3, min_periods=3).sum()
        panel[f"{short}_ret_12m"] = panel[factor].rolling(12, min_periods=12).sum()
        panel[f"{short}_vol_12m"] = panel[factor].rolling(12, min_periods=12).std(ddof=1)

    predictor_cols = [
        "TERM",
        "DEF",
        "dTERM",
        "dDEF",
        "vix",
        "mktexcess_3m",
        "prof_ret_3m",
        "prof_ret_12m",
        "prof_vol_12m",
        "inv_ret_3m",
        "inv_ret_12m",
        "inv_vol_12m",
        "lowvol_ret_3m",
        "lowvol_ret_12m",
        "lowvol_vol_12m",
    ]

    for col in predictor_cols:
        panel[f"z_{col}"] = _expanding_zscore(panel[col])

    z_cols = [f"z_{col}" for col in predictor_cols]
    lagged = panel[["date"] + z_cols].copy()
    lagged[z_cols] = lagged[z_cols].shift(1)

    model_panel = panel[["date"] + FACTOR_COLS + ["RF"]].merge(lagged, on="date", how="left")
    model_panel = model_panel.dropna(subset=FACTOR_COLS + z_cols).reset_index(drop=True)
    return model_panel


def run_dynamic_factor_timing_strategy(
    min_train_months: int = 60,
    validation_months: int = 24,
    lambda_grid: np.ndarray | None = None,
) -> DynamicTimingArtifacts:
    lambda_grid = LAMBDA_GRID if lambda_grid is None else np.asarray(lambda_grid, dtype=float)
    panel = build_predictor_panel()
    feature_cols = [col for col in panel.columns if col.startswith("z_")]

    results: list[dict[str, float | str | pd.Timestamp]] = []

    for i in range(min_train_months + validation_months, len(panel)):
        train = panel.iloc[: i - validation_months].copy()
        validation = panel.iloc[i - validation_months : i].copy()
        history = panel.iloc[:i].copy()
        current = panel.iloc[i]

        betas_train = {
            factor: _fit_linear_forecaster(train[feature_cols], train[factor])
            for factor in FACTOR_COLS
        }
        sigma_train = train[FACTOR_COLS].cov().to_numpy()

        best_lambda = float(lambda_grid[0])
        best_score = -np.inf
        for lam in lambda_grid:
            val_returns = []
            for _, row in validation.iterrows():
                mu_val = np.array(
                    [_predict_linear_forecaster(betas_train[factor], row[feature_cols]) for factor in FACTOR_COLS],
                    dtype=float,
                )
                w_val = _solve_regularized_weights(mu_val, sigma_train, float(lam))
                val_returns.append(float(w_val @ row[FACTOR_COLS].to_numpy(dtype=float)))
            val_returns = np.asarray(val_returns, dtype=float)
            if val_returns.size == 0 or np.isclose(val_returns.std(ddof=1), 0.0):
                score = -np.inf
            else:
                score = float(np.sqrt(12) * val_returns.mean() / val_returns.std(ddof=1))
            if score > best_score:
                best_score = score
                best_lambda = float(lam)

        betas_full = {
            factor: _fit_linear_forecaster(history[feature_cols], history[factor])
            for factor in FACTOR_COLS
        }
        sigma_full = history[FACTOR_COLS].cov().to_numpy()
        mu_current = np.array(
            [_predict_linear_forecaster(betas_full[factor], current[feature_cols]) for factor in FACTOR_COLS],
            dtype=float,
        )
        weights = _solve_regularized_weights(mu_current, sigma_full, best_lambda)
        realized = float(weights @ current[FACTOR_COLS].to_numpy(dtype=float))

        results.append(
            {
                "date": current["date"],
                "ret": realized,
                "lambda": best_lambda,
                "forecast_prof": mu_current[0],
                "forecast_inv": mu_current[1],
                "forecast_lowvol": mu_current[2],
                "w_prof": weights[0],
                "w_inv": weights[1],
                "w_lowvol": weights[2],
                "R_prof": float(current["R_prof"]),
                "R_inv": float(current["R_inv"]),
                "R_lowvol": float(current["R_lowvol"]),
                "RF": float(current["RF"]),
            }
        )

    strategy = pd.DataFrame(results)
    if strategy.empty:
        raise ValueError("Dynamic timing strategy could not generate any out-of-sample observations.")

    core = run_core_strategy(write_csv=False)
    binary = core.strategy_returns.copy()
    binary["date"] = pd.to_datetime(binary["date"])
    comparison = (
        strategy[["date", "ret"]]
        .rename(columns={"ret": "Dynamic_Timing"})
        .merge(binary.rename(columns={"ret": "Binary_Strategy"}), on="date", how="inner")
    )
    comparison["Equal_Weight"] = (
        strategy["R_prof"].to_numpy() + strategy["R_inv"].to_numpy() + strategy["R_lowvol"].to_numpy()
    ) / 3.0

    ff5 = load_ff5()
    summary = prepare_summary_table(
        descriptive_statistics(comparison),
        run_factor_regressions(comparison, ff5),
    )
    alpha = prepare_alpha_table(run_factor_regressions(comparison, ff5))

    return DynamicTimingArtifacts(
        predictor_panel=panel,
        strategy=strategy,
        comparison=comparison,
        summary_table=summary,
        alpha_table=alpha,
    )


def export_dynamic_timing_outputs(artifacts: DynamicTimingArtifacts) -> dict[str, Path]:
    artifacts.strategy.to_csv(OUTPUT_RETURNS_CSV, index=False)
    artifacts.strategy[
        ["date", "lambda", "w_prof", "w_inv", "w_lowvol", "forecast_prof", "forecast_inv", "forecast_lowvol"]
    ].to_csv(OUTPUT_WEIGHTS_CSV, index=False)
    artifacts.summary_table.to_csv(OUTPUT_SUMMARY_CSV, index=False)
    artifacts.alpha_table.to_csv(OUTPUT_ALPHA_CSV, index=False)
    plot_dynamic_timing_comparison(artifacts.comparison)
    return {
        "returns": OUTPUT_RETURNS_CSV,
        "weights": OUTPUT_WEIGHTS_CSV,
        "summary": OUTPUT_SUMMARY_CSV,
        "alpha": OUTPUT_ALPHA_CSV,
        "comparison_plot": OUTPUT_COMPARISON_PLOT,
    }


def plot_dynamic_timing_comparison(
    comparison: pd.DataFrame,
    output_path: Path | None = None,
) -> Path:
    output_path = output_path or OUTPUT_COMPARISON_PLOT
    fig, ax = plt.subplots(figsize=(14, 7))

    for column in ["Dynamic_Timing", "Binary_Strategy", "Equal_Weight"]:
        sample = comparison[["date", column]].dropna().copy()
        sample["cum_growth"] = (1.0 + sample[column]).cumprod()
        ax.plot(sample["date"], sample["cum_growth"], linewidth=2, label=column.replace("_", " "))

    ax.set_title("Dynamic Timing vs Binary Strategy and Equal Weight")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    artifacts = run_dynamic_factor_timing_strategy()
    output_paths = export_dynamic_timing_outputs(artifacts)
    print(
        f"Built {len(artifacts.strategy):,} dynamic timing returns "
        f"from {artifacts.strategy['date'].iloc[0].date()} to {artifacts.strategy['date'].iloc[-1].date()}."
    )
    for name, path in output_paths.items():
        print(f"{name}: {path}")
    print("\nSummary table:")
    print(artifacts.summary_table.to_string(index=False))
