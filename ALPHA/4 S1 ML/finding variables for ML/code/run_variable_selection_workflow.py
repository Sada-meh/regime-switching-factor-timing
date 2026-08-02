from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
import statsmodels.api as sm

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
TRIAL_ERROR_ROOT = PROJECT_ROOT / "trial-and-error"
if TRIAL_ERROR_ROOT.exists() and str(TRIAL_ERROR_ROOT) not in sys.path:
    sys.path.insert(0, str(TRIAL_ERROR_ROOT))

from ALPHA.alpha_common import (
    ML_FEATURE_COLS,
    ML_MIN_OOS_MONTHS,
    ML_MIN_TRAIN_MONTHS,
    annualized_metrics,
    expanding_zscore,
    load_ff_factor_panel,
    load_macro_panel,
)
from credit_factor_rotation_core import load_fred_monthly


BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results_tables"
GRAPHS_DIR = BASE_DIR / "graphs_visuals"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
SUBPERIOD_SPLIT = pd.Timestamp("2007-12-31")
MODEL_FEATURE_COUNT_LIMIT = len(ML_FEATURE_COLS) + 2


def load_epu() -> pd.DataFrame:
    epu = load_fred_monthly(PROJECT_ROOT / "Data" / "USEPUINDXM.csv", "EPU")
    epu["z_EPU"] = expanding_zscore(epu["EPU"])
    return epu


def build_panel() -> pd.DataFrame:
    ff = load_ff_factor_panel().copy()
    macro = load_macro_panel().copy()
    epu = load_epu().copy()
    panel = ff.merge(
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
    ).merge(epu[["date", "EPU", "z_EPU"]], on="date", how="left")

    panel["cyclical"] = 0.5 * (panel["SMB"] + panel["HML"])
    panel["defensive"] = 0.5 * (panel["RMW"] + panel["CMA"])
    panel["spread"] = panel["cyclical"] - panel["defensive"]
    panel["good_leg_next"] = panel["cyclical"].shift(-1)
    panel["bad_leg_next"] = panel["defensive"].shift(-1)
    panel["target_spread_next"] = panel["spread"].shift(-1)
    panel["target_class_next"] = (panel["target_spread_next"] > 0).astype(float)
    panel["target_date"] = panel["date"].shift(-1)

    raw_map = {
        "mkt_rf_3m": panel["Mkt-RF"].rolling(3, min_periods=3).sum(),
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


def build_master_table() -> pd.DataFrame:
    rows = [
        ("z_TERM", "Expanding z-score of TERM slope", "growth expectations", "higher favors cyclical outperformance", "10Y and 3M yields", "month-end observable", "core"),
        ("z_DEF", "Expanding z-score of DEF spread", "credit conditions", "higher favors defensive outperformance", "BAA and AAA yields", "month-end observable", "core"),
        ("z_dTERM", "Expanding z-score of monthly change in TERM", "policy / rates regime", "rising slope favors cyclical rotation", "10Y and 3M yields", "month-end observable", "core"),
        ("z_dDEF", "Expanding z-score of monthly change in DEF", "funding stress", "widening credit favors defensive rotation", "BAA and AAA yields", "month-end observable", "core"),
        ("z_VIX", "Expanding z-score of VIX level", "risk aversion / market stress", "higher VIX favors defensive rotation", "FRED VIXCLS", "month-end observable", "core"),
        ("z_EPU", "Expanding z-score of economic policy uncertainty", "policy / rates regime", "higher uncertainty favors defensive rotation", "USEPUINDXM", "publication lag / revision risk", "optional"),
        ("z_mkt_rf_3m", "Expanding z-score of trailing 3M market excess return", "factor momentum / factor state", "strong recent market favors cyclical tilt", "FF5 factors", "uses only realized past returns", "core"),
        ("z_spread_3m", "Expanding z-score of trailing 3M cyclical-minus-defensive spread", "factor momentum / factor state", "positive spread momentum favors cyclical tilt", "FF5 factor sleeves", "uses only realized past returns", "core"),
        ("z_spread_12m", "Expanding z-score of trailing 12M cyclical-minus-defensive spread", "factor momentum / factor state", "persistent leadership may continue", "FF5 factor sleeves", "uses only realized past returns", "optional"),
        ("z_spread_vol_12m", "Expanding z-score of trailing 12M spread volatility", "factor state / instability", "high spread volatility may favor defensive sleeve", "FF5 factor sleeves", "uses only realized past returns", "core"),
        ("z_cyclical_3m", "Expanding z-score of trailing 3M cyclical sleeve return", "factor momentum / factor state", "strong cyclical momentum favors cyclical tilt", "FF5 factor sleeves", "uses only realized past returns", "optional"),
        ("z_defensive_3m", "Expanding z-score of trailing 3M defensive sleeve return", "factor momentum / factor state", "strong defensive momentum favors defensive tilt", "FF5 factor sleeves", "uses only realized past returns", "optional"),
        ("IP_growth_12m", "12M industrial production growth", "growth expectations", "higher growth favors cyclical tilt", "not in current files", "requires extra data", "drop"),
        ("factor_valuation_spread", "Relative valuation of cyclical vs defensive factors", "factor valuation or spread conditions", "cheap cyclical sleeve may outperform", "not in current files", "requires extra data", "drop"),
    ]
    return pd.DataFrame(rows, columns=["variable_name", "definition", "economic_mechanism", "expected_effect", "data_source", "availability_concerns", "literature_bucket"])


def univariate_screening(panel: pd.DataFrame, candidates: list[str]) -> pd.DataFrame:
    rows = []
    for var in candidates:
        sample = panel[[var, "date", "target_spread_next"]].dropna().copy()
        if len(sample) < 60:
            continue
        corr = sample[var].corr(sample["target_spread_next"])
        fit = sm.OLS(sample["target_spread_next"], sm.add_constant(sample[[var]])).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
        sample["bucket"] = pd.qcut(sample[var], 3, labels=["low", "mid", "high"], duplicates="drop")
        means = sample.groupby("bucket", observed=False)["target_spread_next"].mean()
        low = float(means.get("low", np.nan))
        mid = float(means.get("mid", np.nan))
        high = float(means.get("high", np.nan))
        pre = sample.loc[sample["date"] <= SUBPERIOD_SPLIT]
        post = sample.loc[sample["date"] > SUBPERIOD_SPLIT]
        pre_corr = pre[var].corr(pre["target_spread_next"]) if len(pre) > 24 else np.nan
        post_corr = post[var].corr(post["target_spread_next"]) if len(post) > 24 else np.nan
        sign_consistent = (np.sign(corr) == np.sign(pre_corr) == np.sign(post_corr)) if pd.notna(pre_corr) and pd.notna(post_corr) else False
        extreme_gap = high - low
        if (abs(corr) >= 0.05 or fit.pvalues[var] < 0.10) and abs(extreme_gap) >= 0.002 and sign_consistent:
            decision = "keep"
        elif abs(corr) >= 0.02 or abs(extreme_gap) >= 0.001:
            decision = "review"
        else:
            decision = "drop"
        strength = "strong" if abs(corr) >= 0.08 else "moderate" if abs(corr) >= 0.04 else "weak"
        rows.append(
            {
                "variable_name": var,
                "corr_next_spread": corr,
                "reg_beta": float(fit.params[var]),
                "reg_t": float(fit.tvalues[var]),
                "reg_p": float(fit.pvalues[var]),
                "low_state_mean": low,
                "mid_state_mean": mid,
                "high_state_mean": high,
                "extreme_gap": extreme_gap,
                "pre_corr": pre_corr,
                "post_corr": post_corr,
                "sign_of_relationship": "positive" if corr > 0 else "negative",
                "rough_strength": strength,
                "stable_across_subperiods": sign_consistent,
                "preliminary_decision": decision,
            }
        )
    return pd.DataFrame(rows).sort_values(["preliminary_decision", "corr_next_spread"], ascending=[True, False])


def nonlinearity_diagnostics(screening: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in screening.iterrows():
        low, mid, high = row["low_state_mean"], row["mid_state_mean"], row["high_state_mean"]
        monotonic = (low <= mid <= high) or (low >= mid >= high)
        if monotonic and abs(row["extreme_gap"]) >= 0.002:
            label = "approximately linear"
        elif abs(row["extreme_gap"]) >= 0.002:
            label = "nonlinear but useful"
        else:
            label = "noisy / patternless"
        rows.append({"variable_name": row["variable_name"], "nonlinearity_class": label, "monotonic": monotonic, "extreme_gap": row["extreme_gap"]})
    return pd.DataFrame(rows)


def redundancy_reduction(panel: pd.DataFrame, screening: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    shortlisted = screening.loc[screening["preliminary_decision"] != "drop", "variable_name"].tolist()
    corr = panel[shortlisted].corr()
    rows = []
    final_drop = set()
    manual_rules = {
        "z_cyclical_3m": "overlaps with z_spread_3m and is less directly tied to relative sleeve rotation",
        "z_defensive_3m": "overlaps with z_spread_3m and is less directly tied to relative sleeve rotation",
        "z_spread_12m": "overlaps with z_spread_3m; longer horizon is useful only as optional context",
        "z_EPU": "stress / uncertainty proxy overlaps with z_VIX and z_DEF and has publication-lag concerns",
    }
    for i, col_i in enumerate(shortlisted):
        for col_j in shortlisted[i + 1 :]:
            rho = corr.loc[col_i, col_j]
            if abs(rho) >= 0.65:
                rows.append({"var_1": col_i, "var_2": col_j, "abs_corr": abs(rho), "action": "review overlap"})
    for var, reason in manual_rules.items():
        if var in shortlisted:
            final_drop.add(var)
            rows.append({"var_1": var, "var_2": "", "abs_corr": np.nan, "action": reason})
    reduced = [v for v in shortlisted if v not in final_drop]
    return corr, pd.DataFrame(rows), reduced


def fit_walkforward_gbt(panel: pd.DataFrame, features: list[str], min_train: int = ML_MIN_TRAIN_MONTHS) -> tuple[pd.DataFrame, float]:
    sample = panel[["date", "target_date", "good_leg_next", "bad_leg_next", "target_class_next"] + features].dropna().reset_index(drop=True)
    rows = []
    for idx in range(min_train, len(sample)):
        train = sample.iloc[:idx].copy()
        if train["target_class_next"].nunique() < 2:
            continue
        model = GradientBoostingClassifier(
            n_estimators=50,
            max_depth=2,
            learning_rate=0.1,
            min_samples_leaf=10,
            random_state=42,
        )
        model.fit(train[features], train["target_class_next"].astype(int))
        current = sample.iloc[[idx]]
        prob = float(model.predict_proba(current[features])[0, 1])
        pred = 1 if prob > 0.5 else 0
        realized = float(current["good_leg_next"].iloc[0] if pred == 1 else current["bad_leg_next"].iloc[0])
        rows.append({"date": current["target_date"].iloc[0], "ret": realized, "p_good": prob, "pred_class": pred})
    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    out["switch"] = out["pred_class"].ne(out["pred_class"].shift(1)).astype(float)
    sharpe = annualized_metrics(out["ret"])["sharpe"]
    return out, sharpe


def importance_diagnostics(panel: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    sample = panel[["date", "target_class_next"] + features].dropna().reset_index(drop=True)
    rows = []
    for end in range(ML_MIN_TRAIN_MONTHS, len(sample) - 12, 12):
        train = sample.iloc[:end].copy()
        test = sample.iloc[end : end + 12].copy()
        if train["target_class_next"].nunique() < 2 or len(test) < 6:
            continue
        model = GradientBoostingClassifier(
            n_estimators=50,
            max_depth=2,
            learning_rate=0.1,
            min_samples_leaf=10,
            random_state=42,
        )
        model.fit(train[features], train["target_class_next"].astype(int))
        perm = permutation_importance(model, test[features], test["target_class_next"].astype(int), n_repeats=10, random_state=42, scoring="accuracy")
        for i, feat in enumerate(features):
            rows.append(
                {
                    "window_end": str(test["date"].iloc[-1].date()),
                    "variable_name": feat,
                    "split_importance": float(model.feature_importances_[i]),
                    "perm_importance": float(perm.importances_mean[i]),
                }
            )
    detail = pd.DataFrame(rows)
    if detail.empty:
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
    summary = detail.groupby("variable_name").agg(
        avg_split_importance=("split_importance", "mean"),
        split_stability_std=("split_importance", "std"),
        avg_permutation_importance=("perm_importance", "mean"),
        permutation_stability_std=("perm_importance", "std"),
        n_windows=("split_importance", "size"),
    ).reset_index()
    return detail.merge(summary, on="variable_name", how="left"), summary.sort_values("avg_permutation_importance", ascending=False)


def final_classification(master: pd.DataFrame, screening: pd.DataFrame, importance: pd.DataFrame, reduced: list[str]) -> pd.DataFrame:
    frame = master.merge(screening[["variable_name", "preliminary_decision", "stable_across_subperiods"]], on="variable_name", how="left")
    frame = frame.merge(importance[["variable_name", "avg_split_importance", "avg_permutation_importance"]], on="variable_name", how="left")
    final_labels = []
    for _, row in frame.iterrows():
        var = row["variable_name"]
        if var in reduced and row["literature_bucket"] == "core" and row["preliminary_decision"] in {"keep", "review"}:
            label = "core"
        elif var in reduced or row["literature_bucket"] == "optional":
            label = "optional"
        else:
            label = "drop"
        final_labels.append(label)
    frame["final_bucket"] = final_labels
    return frame


def plot_heatmap(matrix: pd.DataFrame, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=90)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_binned_relationships(screening: pd.DataFrame, path: Path) -> None:
    top = screening.sort_values("abs_extreme_gap" if "abs_extreme_gap" in screening.columns else "extreme_gap", ascending=False).head(6).copy()
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, (_, row) in zip(axes.flatten(), top.iterrows()):
        vals = [row["low_state_mean"], row["mid_state_mean"], row["high_state_mean"]]
        ax.bar(["Low", "Mid", "High"], vals, color=["#457b9d", "#adb5bd", "#e76f51"])
        ax.set_title(row["variable_name"])
        ax.axhline(0.0, color="black", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_interaction_heatmap(panel: pd.DataFrame, x: str, y: str, path: Path, title: str) -> None:
    sample = panel[[x, y, "target_spread_next"]].dropna().copy()
    sample["x_bin"] = pd.qcut(sample[x], 3, labels=["low", "mid", "high"], duplicates="drop")
    sample["y_bin"] = pd.qcut(sample[y], 3, labels=["low", "mid", "high"], duplicates="drop")
    pivot = sample.pivot_table(index="y_bin", columns="x_bin", values="target_spread_next", aggfunc="mean", observed=False)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(pivot.values, cmap="RdYlBu_r")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_summary(markdown_path: Path, summary_lines: list[str]) -> None:
    markdown_path.write_text("\n".join(summary_lines), encoding="utf-8")


def main() -> None:
    panel = build_panel()
    master = build_master_table()
    available = [v for v in master["variable_name"] if v in panel.columns]
    screening = univariate_screening(panel, available)
    nonlin = nonlinearity_diagnostics(screening)
    corr, redundancy, reduced = redundancy_reduction(panel, screening)
    reduced = [v for v in ML_FEATURE_COLS if v in reduced]
    oos_model, oos_sharpe = fit_walkforward_gbt(panel, reduced)
    importance_detail, importance_summary = importance_diagnostics(panel, reduced)
    final = final_classification(master, screening, importance_summary, reduced)

    ff = load_ff_factor_panel().copy()
    ff["cyclical"] = 0.5 * (ff["SMB"] + ff["HML"])
    ff["defensive"] = 0.5 * (ff["RMW"] + ff["CMA"])
    ff["static_basket"] = 0.25 * (ff["SMB"] + ff["HML"] + ff["RMW"] + ff["CMA"])
    compare = oos_model.merge(ff[["date", "cyclical", "defensive", "static_basket"]], on="date", how="left")
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
    master[["variable_name", "definition", "economic_mechanism", "literature_bucket"]].to_csv(RESULTS_DIR / "literature_screened_shortlist.csv", index=False)
    screening.to_csv(RESULTS_DIR / "univariate_screening.csv", index=False)
    nonlin.to_csv(RESULTS_DIR / "nonlinearity_diagnostics.csv", index=False)
    corr.to_csv(RESULTS_DIR / "predictor_correlation_matrix.csv")
    redundancy.to_csv(RESULTS_DIR / "redundancy_summary.csv", index=False)
    importance_summary.to_csv(RESULTS_DIR / "model_importance_summary.csv", index=False)
    importance_detail.to_csv(RESULTS_DIR / "model_importance_detail.csv", index=False)
    final.to_csv(RESULTS_DIR / "final_classification.csv", index=False)
    pd.DataFrame({"variable_name": reduced}).to_csv(RESULTS_DIR / "final_proposed_predictor_set.csv", index=False)
    perf.to_csv(RESULTS_DIR / "oos_model_performance.csv", index=False)

    plot_heatmap(corr, GRAPHS_DIR / "predictor_correlation_heatmap.png", "Predictor Correlation Heatmap")
    plot_binned_relationships(screening.assign(abs_extreme_gap=screening["extreme_gap"].abs()), GRAPHS_DIR / "binned_relationships_top_variables.png")
    plot_interaction_heatmap(panel, "z_TERM", "z_DEF", GRAPHS_DIR / "interaction_heatmap_term_def.png", "TERM x DEF vs Next-Month Spread")
    plot_interaction_heatmap(panel, "z_VIX", "z_spread_3m", GRAPHS_DIR / "interaction_heatmap_vix_spread3m.png", "VIX x Spread(3M) vs Next-Month Spread")
    plot_heatmap(
        importance_summary.set_index("variable_name")[["avg_split_importance", "avg_permutation_importance"]],
        GRAPHS_DIR / "model_importance_heatmap.png",
        "Average Split and Permutation Importance",
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    for col in ["screened_gbt", "static_basket", "defensive", "cyclical"]:
        wealth = (1.0 + compare[col].fillna(0.0)).cumprod()
        ax.plot(compare["date"], wealth, linewidth=1.4, label=col)
    ax.set_title("Screened GBT Model vs Simpler Baselines")
    ax.set_ylabel("Growth of $1")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(GRAPHS_DIR / "oos_cumulative_comparison.png", dpi=180)
    plt.close(fig)

    enough_data = f"Yes for a focused {len(ML_FEATURE_COLS)}-feature FF-based workflow; no for a wide or self-built-heavy feature set."
    buildable = ", ".join(available)
    extra_needed = "IP growth, factor valuation spreads, and richer crowding/liquidity proxies require extra data."
    weak_redundant = "z_cyclical_3m, z_defensive_3m, z_spread_12m, and z_EPU are the main overlap / weak-justification candidates."
    sample_risk = f"The aligned FF workflow supports about {len(panel.dropna(subset=reduced + ['target_spread_next']))} usable monthly observations, so feature count should stay below {MODEL_FEATURE_COUNT_LIMIT}."
    rec_count = f"Recommended final feature count: {len(reduced)} core variables, with at most 1 optional robustness variable."
    summary = [
        "# Variable Selection Workflow Summary",
        "",
        "## Initial questions",
        f"1. Do we have enough data? {enough_data}",
        f"2. Which variables can already be built? {buildable}",
        f"3. Which variables require extra data? {extra_needed}",
        f"4. Are any variables too redundant or weakly justified? {weak_redundant}",
        f"5. Is the effective sample too small for a wide feature set? {sample_risk}",
        f"6. What is the recommended final feature count? {rec_count}",
        "",
        "## Recommended final predictor set",
        "",
    ] + [f"- {v}" for v in reduced] + [
        "",
        "## Out-of-sample note",
        f"The screened GBT model produced an out-of-sample Sharpe of {oos_sharpe:.3f} on the cyclical-versus-defensive timing task.",
        f"OOS months: {len(oos_model)}. Robustness threshold met: {'Yes' if sample_ok else 'No'} (minimum {ML_MIN_OOS_MONTHS} months).",
    ]
    write_summary(RESULTS_DIR / "workflow_summary.md", summary)
    print("Workflow complete.")


if __name__ == "__main__":
    main()
