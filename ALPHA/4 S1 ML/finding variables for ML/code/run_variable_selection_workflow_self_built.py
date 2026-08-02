from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
TRIAL_ERROR_ROOT = PROJECT_ROOT / "trial-and-error"
if TRIAL_ERROR_ROOT.exists() and str(TRIAL_ERROR_ROOT) not in sys.path:
    sys.path.insert(0, str(TRIAL_ERROR_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_variable_selection_workflow as ff_workflow
from ALPHA.alpha_common import ML_FEATURE_COLS, ML_MIN_OOS_MONTHS, ML_MIN_TRAIN_MONTHS, annualized_metrics, expanding_zscore, load_ff_factor_panel, load_macro_panel
from credit_factor_rotation_core import run_core_strategy
from hml_market_regime_switch_strategies import build_self_ff_size_value_factors, build_self_market_return


BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results_tables" / "self_built"
GRAPHS_DIR = BASE_DIR / "graphs_visuals" / "self_built"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)


def build_self_master_table() -> pd.DataFrame:
    master = ff_workflow.build_master_table().copy()
    source_map = {
        "z_mkt_rf_3m": "self-built market excess return",
        "z_spread_3m": "self-built cyclical and defensive sleeves",
        "z_spread_12m": "self-built cyclical and defensive sleeves",
        "z_spread_vol_12m": "self-built cyclical and defensive sleeves",
        "z_cyclical_3m": "self-built HML and SMB sleeves",
        "z_defensive_3m": "self-built profitability and investment sleeves",
    }
    concern_map = {
        "z_mkt_rf_3m": "depends on self-built market return quality",
        "z_spread_3m": "depends on self-built factor sleeve coverage",
        "z_spread_12m": "depends on self-built factor sleeve coverage",
        "z_spread_vol_12m": "depends on self-built factor sleeve coverage",
        "z_cyclical_3m": "coverage begins only when self-built HML and SMB are live",
        "z_defensive_3m": "depends on linked accounting-factor coverage",
    }
    for variable_name, data_source in source_map.items():
        mask = master["variable_name"].eq(variable_name)
        master.loc[mask, "data_source"] = data_source
        master.loc[mask, "availability_concerns"] = concern_map[variable_name]
    return master


def build_self_panel() -> pd.DataFrame:
    core = run_core_strategy(write_csv=False)
    macro = load_macro_panel().copy()
    epu = ff_workflow.load_epu().copy()
    ff = load_ff_factor_panel()[["date", "RF"]].copy()

    factor_returns = core.factor_returns[["date", "R_prof", "R_inv"]].copy()
    self_sv = build_self_ff_size_value_factors(core.crsp).reset_index()
    self_market = build_self_market_return(core.crsp).reset_index()

    panel = (
        factor_returns.merge(self_sv, on="date", how="inner")
        .merge(self_market, on="date", how="inner")
        .merge(ff, on="date", how="inner")
        .merge(
            macro[
                [
                    "date",
                    "TERM",
                    "DEF",
                    "dTERM",
                    "dDEF",
                    "VIX",
                    "z_TERM",
                    "z_DEF",
                    "z_dTERM",
                    "z_dDEF",
                    "z_VIX",
                ]
            ],
            on="date",
            how="inner",
        )
        .merge(epu[["date", "EPU", "z_EPU"]], on="date", how="left")
    )

    panel["cyclical"] = 0.5 * (panel["R_smb_self"] + panel["R_hml_self"])
    panel["defensive"] = 0.5 * (panel["R_prof"] + panel["R_inv"])
    panel["spread"] = panel["cyclical"] - panel["defensive"]
    panel["target_spread_next"] = panel["spread"].shift(-1)
    panel["target_class_next"] = (panel["target_spread_next"] > 0).astype(float)
    panel["target_date"] = panel["date"].shift(-1)
    panel["good_leg_next"] = panel["cyclical"].shift(-1)
    panel["bad_leg_next"] = panel["defensive"].shift(-1)

    market_excess = panel["R_mkt_self"] - panel["RF"]
    raw_map = {
        "mkt_rf_3m": market_excess.rolling(3, min_periods=3).sum(),
        "spread_3m": panel["spread"].rolling(3, min_periods=3).sum(),
        "spread_12m": panel["spread"].rolling(12, min_periods=12).sum(),
        "spread_vol_12m": panel["spread"].rolling(12, min_periods=12).std(ddof=1),
        "cyclical_3m": panel["cyclical"].rolling(3, min_periods=3).sum(),
        "defensive_3m": panel["defensive"].rolling(3, min_periods=3).sum(),
    }
    for name, series in raw_map.items():
        panel[name] = series
        panel[f"z_{name}"] = expanding_zscore(series)

    return panel.sort_values("date").reset_index(drop=True)


def safe_importance_diagnostics(panel: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample = panel[["date", "target_class_next"] + features].dropna().reset_index(drop=True)
    if len(sample) < ML_MIN_TRAIN_MONTHS + 12:
        empty_detail = pd.DataFrame(columns=["window_end", "variable_name", "split_importance", "perm_importance"])
        empty_summary = pd.DataFrame(
            columns=[
                "variable_name",
                "avg_split_importance",
                "split_stability_std",
                "avg_permutation_importance",
                "permutation_stability_std",
                "n_windows",
            ]
        )
        return empty_detail, empty_summary
    return ff_workflow.importance_diagnostics(panel, features)


def run_self_workflow() -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    panel = build_self_panel()
    master = build_self_master_table()
    available = [v for v in master["variable_name"] if v in panel.columns]
    screening = ff_workflow.univariate_screening(panel, available)
    nonlin = ff_workflow.nonlinearity_diagnostics(screening)
    corr, redundancy, reduced = ff_workflow.redundancy_reduction(panel, screening)
    reduced = [v for v in ML_FEATURE_COLS if v in reduced]

    oos_model, oos_sharpe = ff_workflow.fit_walkforward_gbt(panel, reduced)
    importance_detail, importance_summary = safe_importance_diagnostics(panel, reduced)
    final = ff_workflow.final_classification(master, screening, importance_summary, reduced)

    compare = oos_model.merge(
        panel[["date", "cyclical", "defensive"]].copy(),
        on="date",
        how="left",
    )
    compare["static_basket"] = 0.5 * (compare["cyclical"] + compare["defensive"])
    compare = compare.rename(columns={"ret": "screened_gbt"})
    perf = pd.DataFrame(
        [
            {"series": "screened_gbt", **annualized_metrics(compare["screened_gbt"])},
            {"series": "static_basket", **annualized_metrics(compare["static_basket"])},
            {"series": "cyclical_only", **annualized_metrics(compare["cyclical"])},
            {"series": "defensive_only", **annualized_metrics(compare["defensive"])},
        ]
    )
    sample_ok = len(oos_model) >= ML_MIN_OOS_MONTHS
    perf["sample_ok"] = sample_ok
    perf["oos_months"] = len(oos_model)
    perf["min_required_oos_months"] = ML_MIN_OOS_MONTHS

    master.to_csv(RESULTS_DIR / "candidate_variable_master.csv", index=False)
    master[["variable_name", "definition", "economic_mechanism", "literature_bucket"]].to_csv(
        RESULTS_DIR / "literature_screened_shortlist.csv",
        index=False,
    )
    screening.to_csv(RESULTS_DIR / "univariate_screening.csv", index=False)
    nonlin.to_csv(RESULTS_DIR / "nonlinearity_diagnostics.csv", index=False)
    corr.to_csv(RESULTS_DIR / "predictor_correlation_matrix.csv")
    redundancy.to_csv(RESULTS_DIR / "redundancy_summary.csv", index=False)
    importance_summary.to_csv(RESULTS_DIR / "model_importance_summary.csv", index=False)
    importance_detail.to_csv(RESULTS_DIR / "model_importance_detail.csv", index=False)
    final.to_csv(RESULTS_DIR / "final_classification.csv", index=False)
    pd.DataFrame({"variable_name": reduced}).to_csv(RESULTS_DIR / "final_proposed_predictor_set.csv", index=False)
    perf.to_csv(RESULTS_DIR / "oos_model_performance.csv", index=False)
    compare.to_csv(RESULTS_DIR / "screened_gbt_returns.csv", index=False)

    ff_workflow.plot_heatmap(corr, GRAPHS_DIR / "predictor_correlation_heatmap.png", "Self-Built Predictor Correlation Heatmap")
    ff_workflow.plot_binned_relationships(
        screening.assign(abs_extreme_gap=screening["extreme_gap"].abs()),
        GRAPHS_DIR / "binned_relationships_top_variables.png",
    )
    ff_workflow.plot_interaction_heatmap(
        panel,
        "z_TERM",
        "z_DEF",
        GRAPHS_DIR / "interaction_heatmap_term_def.png",
        "Self-Built TERM x DEF vs Next-Month Spread",
    )
    ff_workflow.plot_interaction_heatmap(
        panel,
        "z_VIX",
        "z_spread_3m",
        GRAPHS_DIR / "interaction_heatmap_vix_spread3m.png",
        "Self-Built VIX x Spread(3M) vs Next-Month Spread",
    )
    if not importance_summary.empty:
        ff_workflow.plot_heatmap(
            importance_summary.set_index("variable_name")[["avg_split_importance", "avg_permutation_importance"]],
            GRAPHS_DIR / "model_importance_heatmap.png",
            "Self-Built Average Split and Permutation Importance",
        )

    fig, ax = plt.subplots(figsize=(10, 5))
    for col in ["screened_gbt", "static_basket", "defensive", "cyclical"]:
        wealth = (1.0 + compare[col].fillna(0.0)).cumprod()
        ax.plot(compare["date"], wealth, linewidth=1.4, label=col)
    ax.set_title("Self-Built Screened GBT Model vs Simpler Baselines")
    ax.set_ylabel("Growth of $1")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(GRAPHS_DIR / "oos_cumulative_comparison.png", dpi=180)
    plt.close(fig)

    usable_sample = len(panel.dropna(subset=reduced + ["target_spread_next"]))
    summary_lines = [
        "# Self-Built Variable Selection Workflow Summary",
        "",
        "## Initial questions",
        f"1. Do we have enough data? {'Barely for a narrow workflow; not enough for a wide self-built feature set.' if usable_sample < 180 else 'Yes, but with tighter sample constraints than the FF version.'}",
        "2. Which variables can already be built? z_TERM, z_DEF, z_dTERM, z_dDEF, z_VIX, z_EPU, z_mkt_rf_3m, z_spread_3m, z_spread_12m, z_spread_vol_12m, z_cyclical_3m, z_defensive_3m",
        "3. Which variables require extra data? IP growth, factor valuation spreads, and richer crowding/liquidity proxies require extra data.",
        "4. Are any variables too redundant or weakly justified? z_cyclical_3m, z_defensive_3m, z_spread_12m, and z_EPU remain the main overlap / weak-justification candidates.",
        f"5. Is the effective sample too small for a wide feature set? The aligned self-built workflow supports about {usable_sample} usable monthly observations, so feature count should stay well below 10.",
        f"6. What is the recommended final feature count? Recommended final feature count: {len(reduced)} core variables, with at most 1-2 optional robustness variables.",
        "",
        "## Recommended final predictor set",
        "",
    ] + [f"- {v}" for v in reduced] + [
        "",
        "## Out-of-sample note",
        f"The screened GBT model produced an out-of-sample Sharpe of {oos_sharpe:.3f} on the self-built cyclical-versus-defensive timing task.",
        f"OOS months: {len(oos_model)}. Robustness threshold met: {'Yes' if sample_ok else 'No'} (minimum {ML_MIN_OOS_MONTHS} months).",
        "The self-built workflow should be interpreted cautiously because the cyclical sleeve begins much later than the FF version.",
    ]
    ff_workflow.write_summary(RESULTS_DIR / "workflow_summary.md", summary_lines)
    return compare, reduced, perf


def build_ff_returns_for_comparison() -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    ff_panel = ff_workflow.build_panel()
    ff_features = pd.read_csv(BASE_DIR / "results_tables" / "final_proposed_predictor_set.csv")["variable_name"].tolist()
    ff_oos, _ = ff_workflow.fit_walkforward_gbt(ff_panel, ff_features)
    ff_compare = ff_oos.merge(
        ff_panel[["date", "cyclical", "defensive"]].copy(),
        on="date",
        how="left",
    )
    ff_compare["static_basket"] = 0.5 * (ff_compare["cyclical"] + ff_compare["defensive"])
    ff_compare = ff_compare.rename(columns={"ret": "screened_gbt"})
    ff_perf = pd.DataFrame(
        [
            {"series": "screened_gbt", **annualized_metrics(ff_compare["screened_gbt"])},
            {"series": "static_basket", **annualized_metrics(ff_compare["static_basket"])},
            {"series": "cyclical_only", **annualized_metrics(ff_compare["cyclical"])},
            {"series": "defensive_only", **annualized_metrics(ff_compare["defensive"])},
        ]
    )
    return ff_compare, ff_features, ff_perf


def write_comparison(ff_compare: pd.DataFrame, self_compare: pd.DataFrame, ff_features: list[str], self_features: list[str]) -> None:
    comparison_dir = BASE_DIR / "results_tables"
    overlap = ff_compare[["date", "screened_gbt"]].rename(columns={"screened_gbt": "ff_screened_gbt"}).merge(
        self_compare[["date", "screened_gbt"]].rename(columns={"screened_gbt": "self_built_screened_gbt"}),
        on="date",
        how="inner",
    )
    summary_rows = [
        {"model": "FF_screened_gbt_full", **annualized_metrics(ff_compare["screened_gbt"])},
        {"model": "Self_built_screened_gbt_full", **annualized_metrics(self_compare["screened_gbt"])},
        {"model": "FF_screened_gbt_overlap", **annualized_metrics(overlap["ff_screened_gbt"])},
        {"model": "Self_built_screened_gbt_overlap", **annualized_metrics(overlap["self_built_screened_gbt"])},
    ]
    comparison = pd.DataFrame(summary_rows)
    comparison["feature_set"] = [
        ", ".join(ff_features),
        ", ".join(self_features),
        ", ".join(ff_features),
        ", ".join(self_features),
    ]
    comparison.to_csv(comparison_dir / "ff_vs_self_built_screened_gbt_comparison.csv", index=False)

    overlap.to_csv(comparison_dir / "ff_vs_self_built_overlap_returns.csv", index=False)

    summary_lines = [
        "# FF vs Self-Built Variable Selection Comparison",
        "",
        f"- FF screened model OOS months: {len(ff_compare)}",
        f"- Self-built screened model OOS months: {len(self_compare)}",
        f"- Overlap months: {len(overlap)}",
        f"- FF full-sample Sharpe: {comparison.loc[comparison['model'].eq('FF_screened_gbt_full'), 'sharpe'].iloc[0]:.3f}",
        f"- Self-built full-sample Sharpe: {comparison.loc[comparison['model'].eq('Self_built_screened_gbt_full'), 'sharpe'].iloc[0]:.3f}",
        f"- FF overlap-sample Sharpe: {comparison.loc[comparison['model'].eq('FF_screened_gbt_overlap'), 'sharpe'].iloc[0]:.3f}",
        f"- Self-built overlap-sample Sharpe: {comparison.loc[comparison['model'].eq('Self_built_screened_gbt_overlap'), 'sharpe'].iloc[0]:.3f}",
        "",
        "The self-built comparison is more fragile because the self-built HML and SMB sleeves begin much later than the FF factor series.",
    ]
    ff_workflow.write_summary(comparison_dir / "ff_vs_self_built_comparison_summary.md", summary_lines)

    fig, ax = plt.subplots(figsize=(10, 5))
    overlap = overlap.sort_values("date").reset_index(drop=True)
    ax.plot(overlap["date"], (1.0 + overlap["ff_screened_gbt"].fillna(0.0)).cumprod(), label="FF screened GBT", linewidth=1.5)
    ax.plot(
        overlap["date"],
        (1.0 + overlap["self_built_screened_gbt"].fillna(0.0)).cumprod(),
        label="Self-built screened GBT",
        linewidth=1.5,
    )
    ax.set_title("FF vs Self-Built Screened GBT on Overlapping Sample")
    ax.set_ylabel("Growth of $1")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(BASE_DIR / "graphs_visuals" / "ff_vs_self_built_overlap_comparison.png", dpi=180)
    plt.close(fig)


def main() -> None:
    self_compare, self_features, _ = run_self_workflow()
    ff_compare, ff_features, _ = build_ff_returns_for_comparison()
    write_comparison(ff_compare, self_compare, ff_features, self_features)
    print("Self-built workflow and FF comparison complete.")


if __name__ == "__main__":
    main()
