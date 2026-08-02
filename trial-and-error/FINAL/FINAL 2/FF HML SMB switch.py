from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from credit_factor_rotation_core import (
    build_macro_signal,
    descriptive_statistics,
    load_crsp,
    load_ff5,
    plot_cumulative_returns,
    prepare_alpha_table,
    prepare_summary_table,
    run_core_strategy,
    run_factor_regressions,
    run_subperiod_analysis,
)
from hml_market_regime_switch_strategies import build_self_ff_size_value_factors


FINAL_DIR = PROJECT_ROOT / "FINAL" / "FINAL 2"
FINAL_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_RETURNS = FINAL_DIR / "ff_hml_smb_switch_main_returns.csv"
OUTPUT_SUMMARY = FINAL_DIR / "ff_hml_smb_switch_summary_table.csv"
OUTPUT_ALPHA = FINAL_DIR / "ff_hml_smb_switch_alpha_table.csv"
OUTPUT_SUBPERIOD = FINAL_DIR / "ff_hml_smb_switch_subperiod_table.csv"
OUTPUT_ROBUSTNESS = FINAL_DIR / "ff_hml_smb_switch_robustness_summary.csv"
OUTPUT_COST_SUMMARY = FINAL_DIR / "ff_hml_smb_switch_cost_summary.csv"
OUTPUT_COST_ALPHA = FINAL_DIR / "ff_hml_smb_switch_cost_alpha_table.csv"
OUTPUT_COMPARISON_PLOT = FINAL_DIR / "ff_hml_smb_switch_comparison.png"
OUTPUT_ROBUSTNESS_PLOT = FINAL_DIR / "ff_hml_smb_switch_robustness.png"
OUTPUT_COST_PLOT = FINAL_DIR / "ff_hml_smb_switch_costs.png"

TRADING_COSTS = {
    "0bps": 0.0000,
    "25bps": 0.0025,
    "50bps": 0.0050,
    "75bps": 0.0075,
}


@dataclass
class SwitchArtifacts:
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
    ff_risky = ff5[["date", "RMW", "CMA", "HML", "SMB"]].copy()
    ff_risky["bad_leg"] = 0.5 * (ff_risky["RMW"] + ff_risky["CMA"])
    ff_risky["good_leg"] = 0.5 * (ff_risky["HML"] + ff_risky["SMB"])

    core = run_core_strategy(write_csv=False)
    crsp = load_crsp()
    self_style = build_self_ff_size_value_factors(crsp).reset_index()
    self_risky = (
        core.factor_returns[["date", "R_prof", "R_inv"]]
        .merge(self_style, on="date", how="inner")
    )
    self_risky["bad_leg"] = 0.5 * (self_risky["R_prof"] + self_risky["R_inv"])
    self_risky["good_leg"] = 0.5 * (self_risky["R_hml_self"] + self_risky["R_smb_self"])
    return ff_risky, self_risky


def build_switch_strategy(
    risky_frame: pd.DataFrame,
    macro_signal: pd.DataFrame,
    series_name: str,
) -> pd.DataFrame:
    signal = macro_signal.loc[macro_signal["signal_ready"], ["holding_date", "regime_label", "M_t"]].rename(
        columns={"holding_date": "date"}
    )
    panel = risky_frame.merge(signal, on="date", how="inner")
    panel = panel.dropna(subset=["bad_leg", "good_leg", "regime_label"]).copy()
    panel["ret_gross"] = np.where(panel["regime_label"] == "good", panel["good_leg"], panel["bad_leg"])
    panel["series_name"] = series_name
    return panel[["date", "series_name", "regime_label", "M_t", "bad_leg", "good_leg", "ret_gross"]]


def apply_trading_costs(strategy: pd.DataFrame, prefix: str) -> pd.DataFrame:
    data = strategy.copy().sort_values("date").reset_index(drop=True)
    previous_regime = data["regime_label"].shift(1).fillna(data["regime_label"].iloc[0])
    data["trade_indicator"] = (data["regime_label"] != previous_regime).astype(float)

    outputs = [data[["date"]].assign(**{f"{prefix}_Gross": data["ret_gross"]})]
    for label, cost in TRADING_COSTS.items():
        col = f"{prefix}_{label}"
        outputs.append(data[["date"]].assign(**{col: data["ret_gross"] - cost * data["trade_indicator"]}))
    merged = outputs[0]
    for frame in outputs[1:]:
        merged = merged.merge(frame, on="date", how="inner")
    return merged


def run_ml_strategy(
    risky_frame: pd.DataFrame,
    series_name: str,
    min_train_months: int = 24,
) -> pd.DataFrame:
    try:
        from sklearn.ensemble import GradientBoostingClassifier
    except ImportError as exc:
        raise ImportError("ML robustness requires scikit-learn.") from exc

    from credit_factor_rotation_core import load_macro_feature_frame

    features = load_macro_feature_frame(include_vix=True)
    risky = risky_frame[["date", "bad_leg", "good_leg"]].copy()
    risky["target_good_next"] = (risky["good_leg"].shift(-1) > risky["bad_leg"].shift(-1)).astype("float")
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
    panel["ret_gross"] = np.where(panel["regime_label"] == "good", panel["good_leg"], panel["bad_leg"])
    panel["series_name"] = series_name
    return panel[["date", "series_name", "regime_label", "bad_leg", "good_leg", "ret_gross"]]


def build_variants(risky_frame: pd.DataFrame, prefix: str) -> dict[str, pd.DataFrame]:
    return {
        f"{prefix}_Main": build_switch_strategy(risky_frame, build_macro_signal("median", include_vix=False), f"{prefix}_Main"),
        f"{prefix}_Tercile": build_switch_strategy(risky_frame, build_macro_signal("tercile", include_vix=False), f"{prefix}_Tercile"),
        f"{prefix}_ML": run_ml_strategy(risky_frame, f"{prefix}_ML"),
        f"{prefix}_VIX": build_switch_strategy(risky_frame, build_macro_signal("median", include_vix=True), f"{prefix}_VIX"),
    }


def build_cost_tables(ff_main: pd.DataFrame, self_main: pd.DataFrame, ff5: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ff_costs = apply_trading_costs(ff_main, "FF_Direct")
    self_costs = apply_trading_costs(self_main, "Self_Built")
    cost_frame = ff_costs.merge(self_costs, on="date", how="outer").sort_values("date")
    stats = descriptive_statistics(cost_frame)
    regs = run_factor_regressions(cost_frame, ff5)
    return prepare_summary_table(stats, regs), prepare_alpha_table(regs)


def run_analysis() -> SwitchArtifacts:
    ff5 = load_ff5()
    ff_risky, self_risky = build_risky_baskets()

    ff_variants = build_variants(ff_risky, "FF_Direct")
    self_variants = build_variants(self_risky, "Self_Built")

    ff_main = ff_variants["FF_Direct_Main"]
    self_main = self_variants["Self_Built_Main"]

    comparison = (
        ff_main[["date", "ret_gross"]].rename(columns={"ret_gross": "FF_Direct_Main"})
        .merge(self_main[["date", "ret_gross"]].rename(columns={"ret_gross": "Self_Built_Main"}), on="date", how="outer")
        .merge(ff_risky[["date", "bad_leg", "good_leg"]].rename(columns={"bad_leg": "FF_Bad_Basket", "good_leg": "FF_Good_Basket"}), on="date", how="left")
        .merge(self_risky[["date", "bad_leg", "good_leg"]].rename(columns={"bad_leg": "Self_Bad_Basket", "good_leg": "Self_Good_Basket"}), on="date", how="left")
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

    return SwitchArtifacts(
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


def export_outputs(artifacts: SwitchArtifacts) -> dict[str, Path]:
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
        title="FF HML+SMB Switch: FF Direct vs Self Built",
    )

    ff5 = load_ff5()
    ff_risky, self_risky = build_risky_baskets()
    ff_variants = build_variants(ff_risky, "FF_Direct")
    self_variants = build_variants(self_risky, "Self_Built")

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
        title="FF HML+SMB Switch Robustness Checks",
    )

    cost_plot = apply_trading_costs(artifacts.ff_main, "FF_Direct").merge(
        apply_trading_costs(artifacts.self_main, "Self_Built"),
        on="date",
        how="outer",
    ).sort_values("date")
    plot_cumulative_returns(
        cost_plot,
        OUTPUT_COST_PLOT,
        title="FF HML+SMB Switch Trading Cost Scenarios",
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
        "robustness_plot": OUTPUT_ROBUSTNESS_PLOT,
        "cost_plot": OUTPUT_COST_PLOT,
    }


if __name__ == "__main__":
    artifacts = run_analysis()
    paths = export_outputs(artifacts)
    print("Saved outputs:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    print("\nMain summary:")
    print(artifacts.summary_table.to_string(index=False))
    print("\nTrading cost summary:")
    print(artifacts.cost_summary.to_string(index=False))
