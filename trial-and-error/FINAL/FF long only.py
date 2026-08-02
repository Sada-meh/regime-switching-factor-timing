from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from credit_factor_rotation_core import (
    build_macro_signal,
    descriptive_statistics,
    load_ff5,
    load_macro_feature_frame,
    plot_cumulative_returns,
    prepare_alpha_table,
    prepare_summary_table,
    run_core_strategy,
    run_factor_regressions,
    run_subperiod_analysis,
)

FINAL_DIR = PROJECT_ROOT / "FINAL"
FINAL_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_RETURNS = FINAL_DIR / "ff_long_only_main_returns.csv"
OUTPUT_SUMMARY = FINAL_DIR / "ff_long_only_summary_table.csv"
OUTPUT_ALPHA = FINAL_DIR / "ff_long_only_alpha_table.csv"
OUTPUT_SUBPERIOD = FINAL_DIR / "ff_long_only_subperiod_table.csv"
OUTPUT_ROBUSTNESS = FINAL_DIR / "ff_long_only_robustness_summary.csv"
OUTPUT_COST_SUMMARY = FINAL_DIR / "ff_long_only_cost_summary.csv"
OUTPUT_COST_ALPHA = FINAL_DIR / "ff_long_only_cost_alpha_table.csv"
OUTPUT_COMPARISON_PLOT = FINAL_DIR / "ff_long_only_comparison.png"
OUTPUT_ROBUSTNESS_PLOT = FINAL_DIR / "ff_long_only_robustness.png"
OUTPUT_COST_PLOT = FINAL_DIR / "ff_long_only_costs.png"

TRADING_COSTS = {
    "0bps": 0.0000,
    "25bps": 0.0025,
    "50bps": 0.0050,
    "75bps": 0.0075,
}


@dataclass
class FFLongOnlyArtifacts:
    ff_main: pd.DataFrame
    self_main: pd.DataFrame
    comparison: pd.DataFrame
    summary_table: pd.DataFrame
    alpha_table: pd.DataFrame
    subperiod_results: pd.DataFrame
    robustness_summary: pd.DataFrame
    cost_summary: pd.DataFrame
    cost_alpha_table: pd.DataFrame


def build_risky_baskets() -> tuple[pd.DataFrame, pd.DataFrame]:
    ff5 = load_ff5()
    ff_risky = ff5[["date", "RMW", "CMA", "RF"]].copy()
    ff_risky["risky_leg"] = 0.5 * (ff_risky["RMW"] + ff_risky["CMA"])

    core = run_core_strategy(write_csv=False)
    self_risky = core.factor_returns[["date", "R_prof", "R_inv"]].copy()
    self_risky["risky_leg"] = 0.5 * (self_risky["R_prof"] + self_risky["R_inv"])
    self_risky = self_risky.merge(ff5[["date", "RF"]], on="date", how="inner")

    return ff_risky, self_risky


def build_long_only_strategy(
    risky_frame: pd.DataFrame,
    macro_signal: pd.DataFrame,
    series_name: str,
) -> pd.DataFrame:
    signal = macro_signal.loc[macro_signal["signal_ready"], ["holding_date", "regime_label", "M_t"]].rename(
        columns={"holding_date": "date"}
    )
    panel = risky_frame.merge(signal, on="date", how="inner")
    panel = panel.dropna(subset=["risky_leg", "RF", "regime_label"]).copy()
    panel["risky_weight"] = (panel["regime_label"] == "good").astype(float)
    panel["cash_weight"] = 1.0 - panel["risky_weight"]
    panel["ret_gross"] = panel["risky_weight"] * panel["risky_leg"] + panel["cash_weight"] * panel["RF"]
    panel["series_name"] = series_name
    return panel[["date", "series_name", "regime_label", "M_t", "risky_leg", "RF", "risky_weight", "cash_weight", "ret_gross"]]


def apply_trading_costs(strategy: pd.DataFrame, prefix: str) -> pd.DataFrame:
    data = strategy.copy().sort_values("date").reset_index(drop=True)
    previous_weight = data["risky_weight"].shift(1).fillna(0.0)
    data["trade_indicator"] = (data["risky_weight"] != previous_weight).astype(float)

    outputs = [data[["date"]].assign(**{f"{prefix}_Gross": data["ret_gross"]})]
    for label, cost in TRADING_COSTS.items():
        col = f"{prefix}_{label}"
        outputs.append(data[["date"]].assign(**{col: data["ret_gross"] - cost * data["trade_indicator"]}))
    return outputs[0].merge(outputs[1], on="date").merge(outputs[2], on="date").merge(outputs[3], on="date").merge(outputs[4], on="date")


def run_ml_strategy(
    risky_frame: pd.DataFrame,
    series_name: str,
    min_train_months: int = 24,
) -> pd.DataFrame:
    try:
        from sklearn.ensemble import GradientBoostingClassifier
    except ImportError as exc:
        raise ImportError("ML robustness requires scikit-learn.") from exc

    features = load_macro_feature_frame(include_vix=True)
    risky = risky_frame[["date", "risky_leg", "RF"]].copy()
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
                "regime_label": "good" if prob_good > 0.5 else "bad",
                "prob_good": prob_good,
            }
        )

    ml_signal = pd.DataFrame(rows)
    panel = risky_frame.merge(ml_signal, on="date", how="inner")
    panel["risky_weight"] = (panel["regime_label"] == "good").astype(float)
    panel["cash_weight"] = 1.0 - panel["risky_weight"]
    panel["ret_gross"] = panel["risky_weight"] * panel["risky_leg"] + panel["cash_weight"] * panel["RF"]
    panel["series_name"] = series_name
    return panel[["date", "series_name", "regime_label", "risky_leg", "RF", "risky_weight", "cash_weight", "ret_gross"]]


def build_robustness_variants(
    risky_frame: pd.DataFrame,
    prefix: str,
) -> dict[str, pd.DataFrame]:
    return {
        f"{prefix}_Main": build_long_only_strategy(risky_frame, build_macro_signal("median", include_vix=False), f"{prefix}_Main"),
        f"{prefix}_Tercile": build_long_only_strategy(risky_frame, build_macro_signal("tercile", include_vix=False), f"{prefix}_Tercile"),
        f"{prefix}_ML": run_ml_strategy(risky_frame, f"{prefix}_ML"),
        f"{prefix}_VIX": build_long_only_strategy(risky_frame, build_macro_signal("median", include_vix=True), f"{prefix}_VIX"),
    }


def build_cost_tables(ff_main: pd.DataFrame, self_main: pd.DataFrame, ff5: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ff_costs = apply_trading_costs(ff_main, "FF_Direct")
    self_costs = apply_trading_costs(self_main, "Self_Built")
    cost_frame = ff_costs.merge(self_costs, on="date", how="outer").sort_values("date")

    stats = descriptive_statistics(cost_frame)
    regs = run_factor_regressions(cost_frame, ff5)
    return prepare_summary_table(stats, regs), prepare_alpha_table(regs)


def run_analysis() -> FFLongOnlyArtifacts:
    ff5 = load_ff5()
    ff_risky, self_risky = build_risky_baskets()

    ff_variants = build_robustness_variants(ff_risky, "FF_Direct")
    self_variants = build_robustness_variants(self_risky, "Self_Built")

    ff_main = ff_variants["FF_Direct_Main"]
    self_main = self_variants["Self_Built_Main"]

    comparison = (
        ff_main[["date", "ret_gross"]].rename(columns={"ret_gross": "FF_Direct_Main"})
        .merge(self_main[["date", "ret_gross"]].rename(columns={"ret_gross": "Self_Built_Main"}), on="date", how="outer")
        .merge(ff_risky[["date", "risky_leg"]].rename(columns={"risky_leg": "FF_Static_RMW_CMA"}), on="date", how="left")
        .merge(self_risky[["date", "risky_leg"]].rename(columns={"risky_leg": "Self_Static_Prof_Inv"}), on="date", how="left")
        .merge(ff5[["date", "RF"]].rename(columns={"RF": "Cash"}), on="date", how="left")
        .sort_values("date")
    )

    stats = descriptive_statistics(comparison)
    regs = run_factor_regressions(comparison, ff5)
    summary = prepare_summary_table(stats, regs)
    alpha = prepare_alpha_table(regs)
    subperiod = run_subperiod_analysis(comparison, ff5)

    robustness_frame = (
        ff_variants["FF_Direct_Main"][["date", "ret_gross"]].rename(columns={"ret_gross": "FF_Direct_Main"})
        .merge(ff_variants["FF_Direct_Tercile"][["date", "ret_gross"]].rename(columns={"ret_gross": "FF_Direct_Tercile"}), on="date", how="outer")
        .merge(ff_variants["FF_Direct_ML"][["date", "ret_gross"]].rename(columns={"ret_gross": "FF_Direct_ML"}), on="date", how="outer")
        .merge(ff_variants["FF_Direct_VIX"][["date", "ret_gross"]].rename(columns={"ret_gross": "FF_Direct_VIX"}), on="date", how="outer")
        .merge(self_variants["Self_Built_Main"][["date", "ret_gross"]].rename(columns={"ret_gross": "Self_Built_Main"}), on="date", how="outer")
        .merge(self_variants["Self_Built_Tercile"][["date", "ret_gross"]].rename(columns={"ret_gross": "Self_Built_Tercile"}), on="date", how="outer")
        .merge(self_variants["Self_Built_ML"][["date", "ret_gross"]].rename(columns={"ret_gross": "Self_Built_ML"}), on="date", how="outer")
        .merge(self_variants["Self_Built_VIX"][["date", "ret_gross"]].rename(columns={"ret_gross": "Self_Built_VIX"}), on="date", how="outer")
        .sort_values("date")
    )
    robustness_summary = prepare_summary_table(
        descriptive_statistics(robustness_frame),
        run_factor_regressions(robustness_frame, ff5),
    )

    cost_summary, cost_alpha = build_cost_tables(ff_main, self_main, ff5)

    return FFLongOnlyArtifacts(
        ff_main=ff_main,
        self_main=self_main,
        comparison=comparison,
        summary_table=summary,
        alpha_table=alpha,
        subperiod_results=subperiod,
        robustness_summary=robustness_summary,
        cost_summary=cost_summary,
        cost_alpha_table=cost_alpha,
    )


def export_outputs(artifacts: FFLongOnlyArtifacts) -> dict[str, Path]:
    returns = (
        artifacts.ff_main[["date", "ret_gross"]].rename(columns={"ret_gross": "FF_Direct_Main"})
        .merge(artifacts.self_main[["date", "ret_gross"]].rename(columns={"ret_gross": "Self_Built_Main"}), on="date", how="outer")
        .sort_values("date")
    )
    returns.to_csv(OUTPUT_RETURNS, index=False)
    artifacts.summary_table.to_csv(OUTPUT_SUMMARY, index=False)
    artifacts.alpha_table.to_csv(OUTPUT_ALPHA, index=False)
    artifacts.subperiod_results.to_csv(OUTPUT_SUBPERIOD, index=False)
    artifacts.robustness_summary.to_csv(OUTPUT_ROBUSTNESS, index=False)
    artifacts.cost_summary.to_csv(OUTPUT_COST_SUMMARY, index=False)
    artifacts.cost_alpha_table.to_csv(OUTPUT_COST_ALPHA, index=False)

    plot_cumulative_returns(
        artifacts.comparison,
        OUTPUT_COMPARISON_PLOT,
        title="FF Long-Only: FF Direct vs Self Built",
    )
    return {
        "returns": OUTPUT_RETURNS,
        "summary": OUTPUT_SUMMARY,
        "alpha": OUTPUT_ALPHA,
        "subperiod": OUTPUT_SUBPERIOD,
        "robustness": OUTPUT_ROBUSTNESS,
        "cost_summary": OUTPUT_COST_SUMMARY,
        "cost_alpha": OUTPUT_COST_ALPHA,
        "comparison_plot": OUTPUT_COMPARISON_PLOT,
    }


def export_plots(artifacts: FFLongOnlyArtifacts) -> None:
    ff5 = load_ff5()
    ff_risky, self_risky = build_risky_baskets()
    ff_variants = build_robustness_variants(ff_risky, "FF_Direct")
    self_variants = build_robustness_variants(self_risky, "Self_Built")

    robustness_plot = (
        ff_variants["FF_Direct_Main"][["date", "ret_gross"]].rename(columns={"ret_gross": "FF_Direct_Main"})
        .merge(ff_variants["FF_Direct_Tercile"][["date", "ret_gross"]].rename(columns={"ret_gross": "FF_Direct_Tercile"}), on="date", how="outer")
        .merge(ff_variants["FF_Direct_ML"][["date", "ret_gross"]].rename(columns={"ret_gross": "FF_Direct_ML"}), on="date", how="outer")
        .merge(ff_variants["FF_Direct_VIX"][["date", "ret_gross"]].rename(columns={"ret_gross": "FF_Direct_VIX"}), on="date", how="outer")
        .merge(self_variants["Self_Built_Main"][["date", "ret_gross"]].rename(columns={"ret_gross": "Self_Built_Main"}), on="date", how="outer")
        .merge(self_variants["Self_Built_Tercile"][["date", "ret_gross"]].rename(columns={"ret_gross": "Self_Built_Tercile"}), on="date", how="outer")
        .merge(self_variants["Self_Built_ML"][["date", "ret_gross"]].rename(columns={"ret_gross": "Self_Built_ML"}), on="date", how="outer")
        .merge(self_variants["Self_Built_VIX"][["date", "ret_gross"]].rename(columns={"ret_gross": "Self_Built_VIX"}), on="date", how="outer")
        .sort_values("date")
    )
    plot_cumulative_returns(
        robustness_plot,
        OUTPUT_ROBUSTNESS_PLOT,
        title="FF Long-Only Robustness Checks",
    )

    ff_costs = apply_trading_costs(artifacts.ff_main, "FF_Direct")
    self_costs = apply_trading_costs(artifacts.self_main, "Self_Built")
    cost_plot = ff_costs.merge(self_costs, on="date", how="outer").sort_values("date")
    plot_cumulative_returns(
        cost_plot,
        OUTPUT_COST_PLOT,
        title="FF Long-Only Trading Cost Scenarios",
    )


if __name__ == "__main__":
    artifacts = run_analysis()
    paths = export_outputs(artifacts)
    export_plots(artifacts)
    print("Saved outputs:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    print(f"  cost_plot: {OUTPUT_COST_PLOT}")
    print(f"  robustness_plot: {OUTPUT_ROBUSTNESS_PLOT}")
    print("\nMain summary:")
    print(artifacts.summary_table.to_string(index=False))
    print("\nTrading cost summary:")
    print(artifacts.cost_summary.to_string(index=False))
