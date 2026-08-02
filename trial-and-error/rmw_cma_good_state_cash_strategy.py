from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from credit_factor_rotation_core import (
    PROJECT_ROOT,
    build_macro_signal,
    descriptive_statistics,
    load_ff5,
    load_macro_feature_frame,
    plot_cumulative_returns,
    prepare_alpha_table,
    prepare_summary_table,
    run_factor_regressions,
    run_subperiod_analysis,
)


OUTPUT_RETURNS_CSV = PROJECT_ROOT / "rmw_cma_good_state_cash_returns.csv"
OUTPUT_SUMMARY_CSV = PROJECT_ROOT / "rmw_cma_good_state_cash_summary_table.csv"
OUTPUT_ALPHA_CSV = PROJECT_ROOT / "rmw_cma_good_state_cash_alpha_table.csv"
OUTPUT_ROBUSTNESS_CSV = PROJECT_ROOT / "rmw_cma_good_state_cash_robustness_summary.csv"
OUTPUT_COMPARISON_PLOT = PROJECT_ROOT / "rmw_cma_good_state_cash_comparison.png"
OUTPUT_ROBUSTNESS_PLOT = PROJECT_ROOT / "rmw_cma_good_state_cash_robustness.png"


@dataclass
class RmwCmaCashArtifacts:
    main_strategy: pd.DataFrame
    comparison: pd.DataFrame
    descriptive_stats: pd.DataFrame
    regression_results: pd.DataFrame
    summary_table: pd.DataFrame
    alpha_table: pd.DataFrame
    subperiod_results: pd.DataFrame
    tercile_strategy: pd.DataFrame
    vix_strategy: pd.DataFrame
    ml_strategy: pd.DataFrame
    robustness_summary: pd.DataFrame


def _build_risky_leg(ff5: pd.DataFrame) -> pd.DataFrame:
    panel = ff5[["date", "RMW", "CMA", "RF"]].copy()
    panel["risky_leg"] = 0.5 * (panel["RMW"] + panel["CMA"])
    return panel


def build_good_state_cash_strategy(
    ff5: pd.DataFrame,
    macro_signal: pd.DataFrame,
) -> pd.DataFrame:
    risky = _build_risky_leg(ff5)
    signal = macro_signal.loc[macro_signal["signal_ready"], ["holding_date", "regime_label", "M_t"]].rename(
        columns={"holding_date": "date"}
    )

    panel = risky.merge(signal, on="date", how="inner")
    panel = panel.dropna(subset=["risky_leg", "RF", "regime_label"]).copy()
    panel["ret"] = np.where(panel["regime_label"] == "good", panel["risky_leg"], panel["RF"])
    return panel[["date", "regime_label", "M_t", "RMW", "CMA", "RF", "risky_leg", "ret"]]


def run_tercile_strategy(ff5: pd.DataFrame) -> pd.DataFrame:
    macro = build_macro_signal(threshold_mode="tercile", include_vix=False)
    panel = build_good_state_cash_strategy(ff5, macro)
    panel["ret"] = np.where(panel["regime_label"] == "good", panel["risky_leg"], panel["RF"])
    return panel


def run_vix_strategy(ff5: pd.DataFrame) -> pd.DataFrame:
    macro = build_macro_signal(threshold_mode="median", include_vix=True)
    return build_good_state_cash_strategy(ff5, macro)


def run_ml_strategy(
    ff5: pd.DataFrame,
    min_train_months: int = 24,
) -> pd.DataFrame:
    try:
        from sklearn.ensemble import GradientBoostingClassifier
    except ImportError as exc:
        raise ImportError("ML robustness requires scikit-learn.") from exc

    features = load_macro_feature_frame(include_vix=True)
    risky = _build_risky_leg(ff5)[["date", "risky_leg", "RF"]].copy()
    risky["target_good_next"] = (risky["risky_leg"].shift(-1) > risky["RF"].shift(-1)).astype("float")

    ml_data = features.merge(risky[["date", "target_good_next"]], on="date", how="inner").sort_values("date").reset_index(drop=True)
    feature_cols = ["TERM", "DEF", "dTERM", "dDEF", "vix"]

    rows: list[dict[str, float | str | pd.Timestamp]] = []
    for i in range(len(ml_data)):
        current = ml_data.iloc[i]
        train = ml_data.iloc[:i].dropna(subset=["target_good_next"]).copy()
        if len(train) < min_train_months or train["target_good_next"].nunique() < 2:
            continue

        model = GradientBoostingClassifier(
            n_estimators=50,
            max_depth=2,
            learning_rate=0.1,
            min_samples_leaf=10,
            random_state=42,
        )
        model.fit(train[feature_cols], train["target_good_next"].astype(int))
        prob_good = float(model.predict_proba(pd.DataFrame([current[feature_cols]], columns=feature_cols))[0, 1])
        rows.append(
            {
                "date": current["date"] + pd.offsets.MonthEnd(1),
                "prob_good": prob_good,
                "regime_label": "good" if prob_good > 0.5 else "bad",
            }
        )

    if not rows:
        raise ValueError("ML robustness could not generate any predictions.")

    ml_signal = pd.DataFrame(rows)
    panel = _build_risky_leg(ff5).merge(ml_signal, on="date", how="inner")
    panel["ret"] = np.where(panel["regime_label"] == "good", panel["risky_leg"], panel["RF"])
    return panel[["date", "regime_label", "prob_good", "RMW", "CMA", "RF", "risky_leg", "ret"]]


def build_comparison_frame(main_strategy: pd.DataFrame) -> pd.DataFrame:
    comparison = main_strategy[["date", "ret", "risky_leg"]].rename(
        columns={"ret": "Timed_RMW_CMA_Cash", "risky_leg": "Static_RMW_CMA"}
    )
    comparison["Cash"] = main_strategy["RF"].to_numpy()
    return comparison


def build_robustness_summary(
    main_strategy: pd.DataFrame,
    tercile_strategy: pd.DataFrame,
    vix_strategy: pd.DataFrame,
    ml_strategy: pd.DataFrame,
    ff5: pd.DataFrame,
) -> pd.DataFrame:
    frame = (
        main_strategy[["date", "ret"]].rename(columns={"ret": "Main_Strategy"})
        .merge(tercile_strategy[["date", "ret"]].rename(columns={"ret": "Tercile_Robustness"}), on="date", how="outer")
        .merge(vix_strategy[["date", "ret"]].rename(columns={"ret": "VIX_Robustness"}), on="date", how="outer")
        .merge(ml_strategy[["date", "ret"]].rename(columns={"ret": "ML_Robustness"}), on="date", how="outer")
        .sort_values("date")
    )
    stats = descriptive_statistics(frame)
    regs = run_factor_regressions(frame, ff5)
    summary = prepare_summary_table(stats, regs)
    return summary


def run_strategy() -> RmwCmaCashArtifacts:
    ff5 = load_ff5()
    macro = build_macro_signal(threshold_mode="median", include_vix=False)
    main_strategy = build_good_state_cash_strategy(ff5, macro)

    comparison = build_comparison_frame(main_strategy)
    stats = descriptive_statistics(comparison)
    regs = run_factor_regressions(comparison, ff5)
    summary = prepare_summary_table(stats, regs)
    alpha = prepare_alpha_table(regs)
    subperiod = run_subperiod_analysis(comparison, ff5)

    tercile_strategy = run_tercile_strategy(ff5)
    vix_strategy = run_vix_strategy(ff5)
    ml_strategy = run_ml_strategy(ff5)
    robustness_summary = build_robustness_summary(main_strategy, tercile_strategy, vix_strategy, ml_strategy, ff5)

    return RmwCmaCashArtifacts(
        main_strategy=main_strategy,
        comparison=comparison,
        descriptive_stats=stats,
        regression_results=regs,
        summary_table=summary,
        alpha_table=alpha,
        subperiod_results=subperiod,
        tercile_strategy=tercile_strategy,
        vix_strategy=vix_strategy,
        ml_strategy=ml_strategy,
        robustness_summary=robustness_summary,
    )


def export_outputs(artifacts: RmwCmaCashArtifacts) -> dict[str, Path]:
    artifacts.main_strategy[["date", "ret"]].to_csv(OUTPUT_RETURNS_CSV, index=False)
    artifacts.summary_table.to_csv(OUTPUT_SUMMARY_CSV, index=False)
    artifacts.alpha_table.to_csv(OUTPUT_ALPHA_CSV, index=False)
    artifacts.robustness_summary.to_csv(OUTPUT_ROBUSTNESS_CSV, index=False)

    plot_cumulative_returns(
        artifacts.comparison,
        OUTPUT_COMPARISON_PLOT,
        title="Timed RMW+CMA / Cash vs Static RMW+CMA and Cash",
    )

    robustness_plot = (
        artifacts.main_strategy[["date", "ret"]].rename(columns={"ret": "Main_Strategy"})
        .merge(artifacts.tercile_strategy[["date", "ret"]].rename(columns={"ret": "Tercile_Robustness"}), on="date", how="outer")
        .merge(artifacts.vix_strategy[["date", "ret"]].rename(columns={"ret": "VIX_Robustness"}), on="date", how="outer")
        .merge(artifacts.ml_strategy[["date", "ret"]].rename(columns={"ret": "ML_Robustness"}), on="date", how="outer")
        .sort_values("date")
    )
    plot_cumulative_returns(
        robustness_plot,
        OUTPUT_ROBUSTNESS_PLOT,
        title="Timed RMW+CMA / Cash Robustness Checks",
    )

    return {
        "returns": OUTPUT_RETURNS_CSV,
        "summary": OUTPUT_SUMMARY_CSV,
        "alpha": OUTPUT_ALPHA_CSV,
        "robustness": OUTPUT_ROBUSTNESS_CSV,
        "comparison_plot": OUTPUT_COMPARISON_PLOT,
        "robustness_plot": OUTPUT_ROBUSTNESS_PLOT,
    }


if __name__ == "__main__":
    artifacts = run_strategy()
    paths = export_outputs(artifacts)
    print(
        f"Built {len(artifacts.main_strategy):,} monthly returns "
        f"from {artifacts.main_strategy['date'].iloc[0].date()} "
        f"to {artifacts.main_strategy['date'].iloc[-1].date()}."
    )
    for name, path in paths.items():
        print(f"{name}: {path}")
    print("\nSummary table:")
    print(artifacts.summary_table.to_string(index=False))
