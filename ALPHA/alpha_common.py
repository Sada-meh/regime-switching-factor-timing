from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd
from scipy.stats import jarque_bera, t as student_t
import statsmodels.api as sm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
TRIAL_ERROR_ROOT = PROJECT_ROOT / "trial-and-error"
if TRIAL_ERROR_ROOT.exists() and str(TRIAL_ERROR_ROOT) not in sys.path:
    sys.path.insert(0, str(TRIAL_ERROR_ROOT))
TRIAL_DIR = PROJECT_ROOT / "trial-and-error"
if str(TRIAL_DIR) not in sys.path:
    sys.path.insert(0, str(TRIAL_DIR))

import credit_factor_rotation_core as cf_core
import hml_market_regime_switch_strategies as hmrs

cf_core.PROJECT_ROOT = PROJECT_ROOT
cf_core.DATA_DIR = PROJECT_ROOT / "Data"
hmrs.PROJECT_ROOT = PROJECT_ROOT

from credit_factor_rotation_core import (
    compute_max_drawdown,
    load_crsp,
    load_ff5,
    load_fred_monthly,
    load_reuters_monthly,
    run_core_strategy,
)
from hml_market_regime_switch_strategies import build_self_ff_size_value_factors, build_self_market_return


SAMPLE_START = pd.Timestamp("1990-01-31")
SAMPLE_END = pd.Timestamp("2024-09-30")
BURN_IN_MONTHS = 24
SUBPERIOD_SPLIT = pd.Timestamp("2007-12-31")
TRADING_COSTS_BPS = [0, 25, 50, 75]
LAMBDA_GRID = [1, 3, 10, 30, 100, 300, 1000, 10000, 100000]
ML_FEATURE_COLS = [
    "z_mkt_rf_3m",
    "z_TERM",
    "z_dDEF",
    "z_dTERM",
    "z_spread_3m",
]
ML_OPTIONAL_FEATURE_COLS = ["z_spread_vol_12m", "z_DEF", "z_VIX"]
ML_MIN_TRAIN_MONTHS = 120
ML_MIN_OOS_MONTHS = 60


@dataclass
class OutputDirs:
    code: Path
    results: Path
    graphs: Path


@dataclass
class MLBacktestArtifacts:
    backtest: pd.DataFrame
    importances: pd.DataFrame
    sample_status: pd.DataFrame


def ensure_dirs(base: Path) -> OutputDirs:
    code = base / "code"
    results = base / "results_tables"
    graphs = base / "graphs_visuals"
    code.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    graphs.mkdir(parents=True, exist_ok=True)
    return OutputDirs(code=code, results=results, graphs=graphs)


def annualized_metrics(returns: pd.Series) -> dict[str, float]:
    ret = pd.Series(returns).dropna()
    if ret.empty:
        return {
            "n_months": 0,
            "mean_ann": np.nan,
            "vol_ann": np.nan,
            "sharpe": np.nan,
            "t_stat": np.nan,
            "p_val": np.nan,
            "max_dd": np.nan,
        }
    mean_m = float(ret.mean())
    vol_m = float(ret.std(ddof=1))
    has_vol = pd.notna(vol_m) and vol_m > 0
    t_stat = mean_m / (vol_m / np.sqrt(len(ret))) if has_vol else np.nan
    p_val = 2 * (1 - student_t.cdf(abs(t_stat), df=len(ret) - 1)) if len(ret) > 1 and pd.notna(t_stat) else np.nan
    return {
        "n_months": int(len(ret)),
        "mean_ann": mean_m * 12,
        "vol_ann": vol_m * np.sqrt(12) if pd.notna(vol_m) else np.nan,
        "sharpe": (mean_m / vol_m) * np.sqrt(12) if has_vol else np.nan,
        "t_stat": t_stat,
        "p_val": p_val,
        "max_dd": compute_max_drawdown(ret),
    }


def expanding_zscore(series: pd.Series) -> pd.Series:
    mean = series.expanding().mean()
    std = series.expanding().std(ddof=1)
    return (series - mean) / std


@lru_cache(maxsize=1)
def load_macro_panel() -> pd.DataFrame:
    y10 = load_reuters_monthly(PROJECT_ROOT / "Data" / "10Y monthly US(Table Data).csv", "Yield_10Y")
    y3m = load_reuters_monthly(PROJECT_ROOT / "Data" / "US 3M monthly(Table Data).csv", "Yield_3M")
    aaa = load_fred_monthly(PROJECT_ROOT / "Data" / "DAAA.csv", "DAAA")
    baa = load_fred_monthly(PROJECT_ROOT / "Data" / "DBAA.csv", "DBAA")
    vix = load_fred_monthly(PROJECT_ROOT / "Data" / "VIXCLS (1).csv", "VIX")

    panel = y10.merge(y3m, on="date", how="inner").merge(aaa, on="date", how="inner").merge(baa, on="date", how="inner").merge(vix, on="date", how="left")
    panel = panel.loc[panel["date"].between(SAMPLE_START, SAMPLE_END)].sort_values("date").reset_index(drop=True)
    panel["TERM"] = panel["Yield_10Y"] - panel["Yield_3M"]
    panel["DEF"] = panel["DBAA"] - panel["DAAA"]
    panel["dTERM"] = panel["TERM"].diff()
    panel["dDEF"] = panel["DEF"].diff()
    panel["z_TERM"] = expanding_zscore(panel["TERM"])
    panel["z_DEF"] = expanding_zscore(panel["DEF"])
    panel["z_dTERM"] = expanding_zscore(panel["dTERM"])
    panel["z_dDEF"] = expanding_zscore(panel["dDEF"])
    panel["z_VIX"] = expanding_zscore(panel["VIX"])
    panel["M_t"] = panel["z_TERM"] - panel["z_DEF"]
    panel["M_vix"] = panel["z_TERM"] - panel["z_DEF"] - panel["z_VIX"]
    panel["exp_median_M"] = panel["M_t"].expanding().median()
    panel["exp_q33_M"] = panel["M_t"].expanding().quantile(1 / 3)
    panel["exp_q67_M"] = panel["M_t"].expanding().quantile(2 / 3)
    panel["exp_median_M_vix"] = panel["M_vix"].expanding().median()
    panel["signal_ready"] = np.arange(len(panel)) >= BURN_IN_MONTHS
    panel["holding_date"] = panel["date"].shift(-1)
    return panel


def macro_validation_table() -> pd.DataFrame:
    macro = load_macro_panel()
    checks = [
        {"check": "Total months", "expected": 417, "actual": int(len(macro))},
        {"check": "Date range", "expected": "1990-01 to 2024-09", "actual": f"{macro['date'].min():%Y-%m} to {macro['date'].max():%Y-%m}"},
        {"check": "NaN count", "expected": 0, "actual": int(macro[["TERM", "DEF"]].isna().sum().sum())},
        {"check": "Mean Mkt-RF", "expected": "~0.007", "actual": np.nan},
        {"check": "Mean TERM", "expected": "~1.5", "actual": round(float(macro["TERM"].mean()), 4)},
        {"check": "Mean DEF", "expected": "~1.0", "actual": round(float(macro["DEF"].mean()), 4)},
        {"check": "DEF always positive", "expected": "Yes", "actual": bool((macro["DEF"] > 0).all())},
        {"check": "Correlation(TERM, DEF)", "expected": "Slightly negative", "actual": round(float(macro["TERM"].corr(macro["DEF"])), 4)},
    ]
    ff5 = load_ff5()
    checks[3]["actual"] = round(float(ff5["Mkt-RF"].mean()), 4)
    return pd.DataFrame(checks)


def build_classifier_frame(use_vix_signal: bool = False) -> pd.DataFrame:
    macro = load_macro_panel().copy()
    signal_col = "M_vix" if use_vix_signal else "M_t"
    median_col = "exp_median_M_vix" if use_vix_signal else "exp_median_M"
    macro["regime_med"] = np.where(macro[signal_col] > macro[median_col], "good", "bad")
    macro["regime_tercile"] = np.select(
        [
            macro["M_t"] > macro["exp_q67_M"],
            macro["M_t"] < macro["exp_q33_M"],
        ],
        ["good", "bad"],
        default="neutral",
    )
    return macro


def compute_spell_lengths(regimes: pd.Series) -> list[tuple[str, int]]:
    spells: list[tuple[str, int]] = []
    current = None
    length = 0
    for value in regimes.dropna():
        if value == current:
            length += 1
        else:
            if current is not None:
                spells.append((str(current), length))
            current = value
            length = 1
    if current is not None:
        spells.append((str(current), length))
    return spells


def regime_diagnostics(signal_frame: pd.DataFrame, regime_col: str = "regime_med") -> pd.DataFrame:
    live = signal_frame.loc[signal_frame["signal_ready"] & signal_frame["holding_date"].notna(), ["holding_date", regime_col, "M_t", "exp_median_M"]].copy()
    live = live.rename(columns={"holding_date": "date", regime_col: "regime"})
    counts = live["regime"].value_counts()
    spells = compute_spell_lengths(live["regime"])
    good_spells = [length for regime, length in spells if regime == "good"]
    bad_spells = [length for regime, length in spells if regime == "bad"]
    switches = int(live["regime"].ne(live["regime"].shift(1)).sum() - 1)
    years = max((live["date"].max() - live["date"].min()).days / 365.25, 1)
    return pd.DataFrame(
        [
            {
                "n_good": int(counts.get("good", 0)),
                "n_bad": int(counts.get("bad", 0)),
                "pct_good": float(counts.get("good", 0) / len(live)) if len(live) else np.nan,
                "pct_bad": float(counts.get("bad", 0) / len(live)) if len(live) else np.nan,
                "avg_good_spell": float(np.mean(good_spells)) if good_spells else np.nan,
                "avg_bad_spell": float(np.mean(bad_spells)) if bad_spells else np.nan,
                "total_switches": switches,
                "switches_per_year": switches / years,
            }
        ]
    )


@lru_cache(maxsize=1)
def load_ff_factor_panel() -> pd.DataFrame:
    ff5 = load_ff5()[["date", "Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]].copy()
    return ff5.loc[ff5["date"].between(SAMPLE_START, SAMPLE_END)].reset_index(drop=True)


@lru_cache(maxsize=1)
def load_self_built_factor_panel() -> pd.DataFrame:
    core = run_core_strategy(write_csv=False)
    crsp = load_crsp()
    self_style = build_self_ff_size_value_factors(crsp).reset_index()
    self_market = build_self_market_return(crsp).reset_index()
    rf = load_ff_factor_panel()[["date", "RF"]].copy()
    panel = core.factor_returns[["date", "R_prof", "R_inv"]].copy()
    panel = panel.merge(self_style, on="date", how="inner").merge(self_market, on="date", how="inner").merge(rf, on="date", how="inner")
    panel = panel.rename(
        columns={
            "R_prof": "RMW",
            "R_inv": "CMA",
            "R_hml_self": "HML",
            "R_smb_self": "SMB",
        }
    )
    panel["Mkt-RF"] = panel["R_mkt_self"] - panel["RF"]
    return panel.sort_values("date").reset_index(drop=True)


def get_factor_panel(version: str) -> pd.DataFrame:
    if version == "FF":
        return load_ff_factor_panel().copy()
    if version == "Self-built":
        return load_self_built_factor_panel().copy()
    raise ValueError(f"Unknown version: {version}")


def descriptive_stats_table(returns_df: pd.DataFrame, turnover: dict[str, pd.Series] | None = None) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    for col in [c for c in returns_df.columns if c != "date"]:
        row = {"strategy": col}
        row.update(annualized_metrics(returns_df[col]))
        row["turnover"] = float(turnover[col].mean()) if turnover is not None and col in turnover else np.nan
        row["annual_switches"] = float((turnover[col].sum() / max(len(returns_df) / 12, 1))) if turnover is not None and col in turnover else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def alpha_regression_tables(
    returns_df: pd.DataFrame,
    ff5: pd.DataFrame,
    *,
    subtract_rf: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = returns_df.merge(ff5[["date", "RF", "Mkt-RF", "SMB", "HML", "RMW", "CMA"]], on="date", how="inner")
    alpha_rows: list[dict[str, float | str]] = []
    diag_rows: list[dict[str, float | str]] = []
    models = {
        "CAPM": ["Mkt-RF"],
        "FF3": ["Mkt-RF", "SMB", "HML"],
        "FF5": ["Mkt-RF", "SMB", "HML", "RMW", "CMA"],
    }

    for strategy in [c for c in returns_df.columns if c != "date"]:
        sample = merged[["date", strategy, "RF", "Mkt-RF", "SMB", "HML", "RMW", "CMA"]].dropna().copy()
        if sample.empty:
            continue
        sample["R_target"] = sample[strategy] - sample["RF"] if subtract_rf else sample[strategy]

        for model_name, regressors in models.items():
            if len(sample) <= len(regressors):
                continue
            X = sm.add_constant(sample[regressors])
            y = sample["R_target"]
            fit = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
            ols_plain = sm.OLS(y, X).fit()

            alpha_row: dict[str, float | str] = {
                "strategy": strategy,
                "model": model_name,
                "return_basis": "excess" if subtract_rf else "raw",
                "alpha": float(fit.params.get("const", np.nan)),
                "alpha_t": float(fit.tvalues.get("const", np.nan)),
                "alpha_p": float(fit.pvalues.get("const", np.nan)),
                "R2": float(fit.rsquared),
                "adj_R2": float(ols_plain.rsquared_adj),
            }
            for reg in ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]:
                alpha_row[f"beta_{reg.replace('-', '')}"] = float(fit.params.get(reg, np.nan))
            alpha_rows.append(alpha_row)

            bp_stat, bp_p, _, _ = het_breuschpagan(ols_plain.resid, ols_plain.model.exog)
            jb_stat, jb_p = jarque_bera(ols_plain.resid)
            if len(regressors) > 1:
                vif_values = [variance_inflation_factor(sample[regressors].to_numpy(), i) for i in range(len(regressors))]
                max_vif = float(np.nanmax(vif_values))
            else:
                max_vif = np.nan
            diag_rows.append(
                {
                    "strategy": strategy,
                    "model": model_name,
                    "DW": float(durbin_watson(ols_plain.resid)),
                    "BP_p": float(bp_p),
                    "JB_p": float(jb_p),
                    "max_VIF": max_vif,
                    "adj_R2": float(ols_plain.rsquared_adj),
                }
            )

    return pd.DataFrame(alpha_rows), pd.DataFrame(diag_rows)


def get_ff5_alpha_and_t(alpha_df: pd.DataFrame) -> tuple[float, float]:
    if alpha_df.empty or "model" not in alpha_df.columns:
        return np.nan, np.nan
    ff5_row = alpha_df.loc[alpha_df["model"] == "FF5"]
    if ff5_row.empty:
        return np.nan, np.nan
    return float(ff5_row["alpha"].iloc[0]), float(ff5_row["alpha_t"].iloc[0])


def ff5_alpha_only(
    returns: pd.Series,
    ff5: pd.DataFrame,
    dates: pd.Series,
    *,
    subtract_rf: bool = False,
) -> float:
    frame = pd.DataFrame({"date": dates, "ret": returns}).dropna()
    merged = frame.merge(ff5[["date", "RF", "Mkt-RF", "SMB", "HML", "RMW", "CMA"]], on="date", how="inner")
    if len(merged) <= 5:
        return np.nan
    y = merged["ret"] - merged["RF"] if subtract_rf else merged["ret"]
    X = sm.add_constant(merged[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]])
    fit = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
    return float(fit.params.get("const", np.nan))


def breakeven_cost_bps(
    base_returns: pd.Series,
    cost_multiplier: pd.Series,
    ff5: pd.DataFrame,
    dates: pd.Series,
    *,
    subtract_rf: bool = False,
) -> float:
    low, high = 0.0, 500.0
    target = ff5_alpha_only(base_returns, ff5, dates, subtract_rf=subtract_rf)
    if pd.isna(target):
        return np.nan
    for _ in range(20):
        mid = 0.5 * (low + high)
        adjusted = base_returns - (mid / 10000.0) * cost_multiplier
        alpha_mid = ff5_alpha_only(adjusted, ff5, dates, subtract_rf=subtract_rf)
        if pd.isna(alpha_mid):
            return np.nan
        if alpha_mid > 0:
            low = mid
        else:
            high = mid
    return round(high, 2)


def transaction_cost_table(
    strategy_name: str,
    dates: pd.Series,
    base_returns: pd.Series,
    turnover_cost_multiplier: pd.Series,
    ff5: pd.DataFrame,
    *,
    subtract_rf: bool = False,
) -> pd.DataFrame:
    rows = []
    breakeven = breakeven_cost_bps(base_returns, turnover_cost_multiplier, ff5, dates, subtract_rf=subtract_rf)
    for cost_bps in TRADING_COSTS_BPS:
        adjusted = base_returns - (cost_bps / 10000.0) * turnover_cost_multiplier
        perf = annualized_metrics(adjusted)
        alpha = ff5_alpha_only(adjusted, ff5, dates, subtract_rf=subtract_rf)
        rows.append(
            {
                "strategy": strategy_name,
                "cost_bps": cost_bps,
                "sharpe": perf["sharpe"],
                "ff5_alpha": alpha,
                "breakeven_bps": breakeven,
            }
        )
    return pd.DataFrame(rows)


def plot_cumulative(frame: pd.DataFrame, columns: list[str], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    for col in columns:
        wealth = (1.0 + frame[col].fillna(0.0)).cumprod()
        ax.plot(frame["date"], wealth, linewidth=1.5, label=col)
    ax.set_title(title)
    ax.set_ylabel("Growth of $1")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_regime_timeline(signal_frame: pd.DataFrame, regime_col: str, path: Path, title: str) -> None:
    live = signal_frame.loc[signal_frame["signal_ready"] & signal_frame["holding_date"].notna(), ["holding_date", regime_col, "M_t", "exp_median_M"]].copy()
    live = live.rename(columns={"holding_date": "date", regime_col: "regime"})
    colors = live["regime"].map({"good": "#2a9d8f", "bad": "#e76f51", "neutral": "#adb5bd"}).fillna("#adb5bd")
    fig, ax1 = plt.subplots(figsize=(13, 4))
    ax1.bar(live["date"], np.ones(len(live)), color=colors, width=25, alpha=0.35)
    ax1.set_yticks([])
    ax2 = ax1.twinx()
    ax2.plot(live["date"], live["M_t"], color="#1d3557", linewidth=1.2, label="M(t)")
    ax2.plot(live["date"], live["exp_median_M"], color="#6c757d", linewidth=1.0, linestyle="--", label="Expanding median")
    ax2.set_title(title)
    ax2.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_barh(series: pd.Series, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    series.sort_values().plot(kind="barh", ax=ax, color="#457b9d")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_lines(frame: pd.DataFrame, cols: list[str], path: Path, title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    for col in cols:
        ax.plot(frame["date"], frame[col], linewidth=1.4, label=col)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _binary_classification_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    truth = pd.Series(y_true).dropna().astype(int)
    pred = pd.Series(y_pred).reindex(truth.index).astype(int)
    if truth.empty:
        return {
            "accuracy": np.nan,
            "balanced_accuracy": np.nan,
            "tn": np.nan,
            "fp": np.nan,
            "fn": np.nan,
            "tp": np.nan,
        }
    cm = confusion_matrix(truth, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn) if (tp + fn) else np.nan
    tnr = tn / (tn + fp) if (tn + fp) else np.nan
    return {
        "accuracy": float((tn + tp) / cm.sum()) if cm.sum() else np.nan,
        "balanced_accuracy": float(np.nanmean([tpr, tnr])),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
    }


def _placeholder_csv(path: Path, columns: list[str]) -> None:
    pd.DataFrame(columns=columns).to_csv(path, index=False)


def build_centerpiece_frames(version: str) -> dict[str, pd.DataFrame]:
    factor_panel = get_factor_panel(version)
    ff_eval = load_ff_factor_panel()
    if "Mkt-RF" not in factor_panel.columns:
        factor_panel = factor_panel.merge(ff_eval[["date", "Mkt-RF"]], on="date", how="left")
    if "RF" not in factor_panel.columns:
        factor_panel = factor_panel.merge(ff_eval[["date", "RF"]], on="date", how="left")
    factor_panel["good_leg"] = 0.5 * (factor_panel["HML"] + factor_panel["SMB"])
    factor_panel["bad_leg"] = 0.5 * (factor_panel["RMW"] + factor_panel["CMA"])
    factor_panel["static_basket"] = 0.25 * (factor_panel["HML"] + factor_panel["SMB"] + factor_panel["RMW"] + factor_panel["CMA"])
    factor_panel["good_only"] = factor_panel["good_leg"]
    factor_panel["bad_only"] = factor_panel["bad_leg"]
    factor_panel["mkt_rf"] = factor_panel["Mkt-RF"]

    macro = build_classifier_frame(use_vix_signal=False)
    macro_vix = build_classifier_frame(use_vix_signal=True)

    med = factor_panel.merge(
        macro.loc[macro["signal_ready"], ["holding_date", "regime_med"]].rename(columns={"holding_date": "date"}),
        on="date",
        how="inner",
    )
    med["ret"] = np.where(med["regime_med"] == "good", med["good_leg"], med["bad_leg"])
    med["switch"] = med["regime_med"].ne(med["regime_med"].shift(1)).astype(float)

    tercile = factor_panel.merge(
        macro.loc[macro["signal_ready"], ["holding_date", "regime_tercile"]].rename(columns={"holding_date": "date"}),
        on="date",
        how="inner",
    )
    tercile["ret"] = np.select(
        [
            tercile["regime_tercile"] == "good",
            tercile["regime_tercile"] == "bad",
        ],
        [tercile["good_leg"], tercile["bad_leg"]],
        default=tercile["static_basket"],
    )
    tercile["switch"] = tercile["regime_tercile"].ne(tercile["regime_tercile"].shift(1)).astype(float)

    vix = factor_panel.merge(
        macro_vix.loc[macro_vix["signal_ready"], ["holding_date", "regime_med"]].rename(columns={"holding_date": "date", "regime_med": "regime_vix"}),
        on="date",
        how="inner",
    )
    vix["ret"] = np.where(vix["regime_vix"] == "good", vix["good_leg"], vix["bad_leg"])
    vix["switch"] = vix["regime_vix"].ne(vix["regime_vix"].shift(1)).astype(float)

    return {"baseline": med, "tercile": tercile, "vix": vix, "macro": macro, "factor_panel": factor_panel}


def evaluate_centerpiece(version: str, base_dir: Path) -> None:
    dirs = ensure_dirs(base_dir)
    frames = build_centerpiece_frames(version)
    baseline = frames["baseline"]
    tercile = frames["tercile"]
    vix = frames["vix"]
    macro = frames["macro"]
    ff_eval = load_ff_factor_panel()

    strategy_returns = baseline[["date", "ret"]].rename(columns={"ret": "strategy_1A_med"})
    benchmarks = baseline[["date", "static_basket", "good_only", "bad_only", "mkt_rf"]].rename(
        columns={
            "static_basket": "static_basket",
            "good_only": "good_only",
            "bad_only": "bad_only",
            "mkt_rf": "mkt_rf",
        }
    )
    all_returns = strategy_returns.merge(benchmarks, on="date", how="left")

    desc = descriptive_stats_table(all_returns, turnover={"strategy_1A_med": baseline["switch"]})
    alpha, diag = alpha_regression_tables(all_returns.rename(columns={"mkt_rf": "mkt_rf_benchmark"}), ff_eval)
    cost = transaction_cost_table("strategy_1A_med", baseline["date"], baseline["ret"], baseline["switch"], ff_eval)
    subperiod_rows = []
    for label, mask in [("pre_2008", strategy_returns["date"] <= SUBPERIOD_SPLIT), ("post_2007", strategy_returns["date"] > SUBPERIOD_SPLIT)]:
        ret_sub = strategy_returns.loc[mask].copy()
        perf = annualized_metrics(ret_sub["strategy_1A_med"])
        ff5_alpha = ff5_alpha_only(ret_sub["strategy_1A_med"], ff_eval, ret_sub["date"])
        alpha_rows, _ = alpha_regression_tables(ret_sub, ff_eval)
        t_val = alpha_rows.loc[alpha_rows["model"] == "FF5", "alpha_t"]
        subperiod_rows.append(
            {
                "strategy": "strategy_1A_med",
                "period": label,
                "sharpe": perf["sharpe"],
                "ff5_alpha": ff5_alpha,
                "ff5_alpha_t": float(t_val.iloc[0]) if len(t_val) else np.nan,
            }
        )
    robustness = pd.DataFrame(
        [
            {
                "variant": "1A-Med (baseline)",
                "sharpe": annualized_metrics(baseline["ret"])["sharpe"],
                "ff5_alpha": ff5_alpha_only(baseline["ret"], ff_eval, baseline["date"]),
                "ff5_alpha_t": float(alpha_regression_tables(strategy_returns, ff_eval)[0].loc[lambda x: x["model"] == "FF5", "alpha_t"].iloc[0]),
            },
            {
                "variant": "Tercile classifier",
                "sharpe": annualized_metrics(tercile["ret"])["sharpe"],
                "ff5_alpha": ff5_alpha_only(tercile["ret"], ff_eval, tercile["date"]),
                "ff5_alpha_t": float(alpha_regression_tables(tercile[["date", "ret"]].rename(columns={"ret": "strategy"}), ff_eval)[0].loc[lambda x: x["model"] == "FF5", "alpha_t"].iloc[0]),
            },
            {
                "variant": "VIX-augmented signal",
                "sharpe": annualized_metrics(vix["ret"])["sharpe"],
                "ff5_alpha": ff5_alpha_only(vix["ret"], ff_eval, vix["date"]),
                "ff5_alpha_t": float(alpha_regression_tables(vix[["date", "ret"]].rename(columns={"ret": "strategy"}), ff_eval)[0].loc[lambda x: x["model"] == "FF5", "alpha_t"].iloc[0]),
            },
        ]
    )
    regime_diag = regime_diagnostics(macro, regime_col="regime_med")

    macro_validation_table().to_csv(dirs.results / "validation_summary.csv", index=False)
    strategy_returns.rename(columns={"strategy_1A_med": "ret"}).to_csv(dirs.results / "strategy_1A_med_returns.csv", index=False)
    benchmarks.to_csv(dirs.results / "benchmarks_returns.csv", index=False)
    desc.to_csv(dirs.results / "descriptive_stats.csv", index=False)
    alpha.to_csv(dirs.results / "alpha_regressions.csv", index=False)
    diag.to_csv(dirs.results / "regression_diagnostics.csv", index=False)
    cost.to_csv(dirs.results / "transaction_costs.csv", index=False)
    pd.DataFrame(subperiod_rows).to_csv(dirs.results / "robustness_subperiod.csv", index=False)
    robustness.to_csv(dirs.results / "robustness_variants.csv", index=False)
    regime_diag.to_csv(dirs.results / "regime_diagnostics.csv", index=False)

    plot_cumulative(all_returns, ["strategy_1A_med", "static_basket", "mkt_rf"], dirs.graphs / "cumulative_return_comparison.png", f"1A Centerpiece {version}: Strategy vs Static Basket vs Mkt-RF")
    plot_regime_timeline(macro, "regime_med", dirs.graphs / "regime_timeline.png", f"1A Centerpiece {version}: Regime Timeline")
    plot_cumulative(pd.concat([baseline[["date", "ret"]].rename(columns={"ret": "baseline"}), tercile[["ret"]].rename(columns={"ret": "tercile"}), vix[["ret"]].rename(columns={"ret": "vix"})], axis=1), ["baseline", "tercile", "vix"], dirs.graphs / "robustness_comparison.png", f"1A Centerpiece {version}: Robustness Comparison")


def build_ml_backtest(version: str) -> MLBacktestArtifacts:
    factor_panel = get_factor_panel(version)
    macro = build_classifier_frame(use_vix_signal=True)
    frame = factor_panel.merge(
        macro[["date", "z_TERM", "z_DEF", "z_dTERM", "z_dDEF", "z_VIX", "signal_ready", "holding_date", "regime_med"]],
        on="date",
        how="inner",
    )
    frame["good_leg"] = 0.5 * (frame["HML"] + frame["SMB"])
    frame["bad_leg"] = 0.5 * (frame["RMW"] + frame["CMA"])
    frame["spread"] = frame["good_leg"] - frame["bad_leg"]
    frame["mkt_rf_3m"] = frame["Mkt-RF"].rolling(3, min_periods=3).sum()
    frame["spread_3m"] = frame["spread"].rolling(3, min_periods=3).sum()
    frame["spread_vol_12m"] = frame["spread"].rolling(12, min_periods=12).std(ddof=1)
    frame["z_mkt_rf_3m"] = expanding_zscore(frame["mkt_rf_3m"])
    frame["z_spread_3m"] = expanding_zscore(frame["spread_3m"])
    frame["z_spread_vol_12m"] = expanding_zscore(frame["spread_vol_12m"])
    frame["good_leg_next"] = frame["good_leg"].shift(-1)
    frame["bad_leg_next"] = frame["bad_leg"].shift(-1)
    frame["target_spread_next"] = frame["good_leg_next"] - frame["bad_leg_next"]
    frame["target"] = np.where(frame["target_spread_next"].notna(), (frame["target_spread_next"] > 0).astype(float), np.nan)
    frame["target_date"] = frame["date"].shift(-1)

    feature_cols = ML_FEATURE_COLS
    rows = []
    importances = []
    sample = frame[["date", "target_date", "good_leg_next", "bad_leg_next", "regime_med", "target"] + feature_cols].dropna().reset_index(drop=True)
    for idx in range(ML_MIN_TRAIN_MONTHS, len(sample)):
        train = sample.iloc[:idx].copy()
        if len(train) < ML_MIN_TRAIN_MONTHS or train["target"].nunique() < 2:
            continue
        model = GradientBoostingClassifier(
            n_estimators=50,
            max_depth=2,
            learning_rate=0.1,
            min_samples_leaf=10,
            random_state=42,
        )
        model.fit(train[feature_cols], train["target"].astype(int))
        current = sample.iloc[[idx]].copy()
        prob_good = float(model.predict_proba(current[feature_cols])[0, 1])
        pred = 1 if prob_good > 0.5 else 0
        majority_prob = float(train["target"].mean())
        majority_pred = 1 if majority_prob >= 0.5 else 0
        rows.append(
            {
                "signal_date": current["date"].iloc[0],
                "date": current["target_date"].iloc[0],
                "p_good": prob_good,
                "pred_regime": "good" if pred == 1 else "bad",
                "median_regime": current["regime_med"].iloc[0],
                "realized_target": int(current["target"].iloc[0]),
                "majority_pred": majority_pred,
                "majority_prob": majority_prob,
                "good_leg_next": float(current["good_leg_next"].iloc[0]),
                "bad_leg_next": float(current["bad_leg_next"].iloc[0]),
                "ret": float(current["good_leg_next"].iloc[0] if pred == 1 else current["bad_leg_next"].iloc[0]),
                "switch": np.nan,
            }
        )
        importances.append(model.feature_importances_)
    backtest = pd.DataFrame(rows)
    if not backtest.empty:
        backtest["switch"] = backtest["pred_regime"].ne(backtest["pred_regime"].shift(1)).astype(float)
    else:
        backtest = pd.DataFrame(
            columns=[
                "signal_date",
                "date",
                "p_good",
                "pred_regime",
                "median_regime",
                "realized_target",
                "majority_pred",
                "majority_prob",
                "good_leg_next",
                "bad_leg_next",
                "ret",
                "switch",
            ]
        )
    importances_df = pd.DataFrame(importances, columns=feature_cols) if importances else pd.DataFrame(columns=feature_cols)
    sample_status = pd.DataFrame(
        [
            {
                "version": version,
                "feature_count": len(feature_cols),
                "features": ", ".join(feature_cols),
                "usable_months": int(len(sample)),
                "min_train_months": ML_MIN_TRAIN_MONTHS,
                "oos_months": int(len(backtest)),
                "min_required_oos_months": ML_MIN_OOS_MONTHS,
                "sample_ok": bool(len(backtest) >= ML_MIN_OOS_MONTHS),
            }
        ]
    )
    return MLBacktestArtifacts(backtest=backtest, importances=importances_df, sample_status=sample_status)


def evaluate_ml_strategy(version: str, base_dir: Path) -> None:
    dirs = ensure_dirs(base_dir)
    artifacts = build_ml_backtest(version)
    backtest = artifacts.backtest
    importances_df = artifacts.importances
    sample_status = artifacts.sample_status.copy()
    sample_ok = bool(sample_status["sample_ok"].iloc[0]) if not sample_status.empty else False
    ff_eval = load_ff_factor_panel()
    factor_panel = get_factor_panel(version)
    factor_panel["good_leg"] = 0.5 * (factor_panel["HML"] + factor_panel["SMB"])
    factor_panel["bad_leg"] = 0.5 * (factor_panel["RMW"] + factor_panel["CMA"])
    static = 0.25 * (factor_panel["HML"] + factor_panel["SMB"] + factor_panel["RMW"] + factor_panel["CMA"])
    comparison = backtest.merge(
        factor_panel[["date", "good_leg", "bad_leg"]].rename(columns={"good_leg": "good_only", "bad_leg": "bad_only"}),
        on="date",
        how="left",
    ).merge(pd.DataFrame({"date": factor_panel["date"], "static_basket": static}), on="date", how="left").merge(
        ff_eval[["date", "Mkt-RF"]].rename(columns={"Mkt-RF": "mkt_rf"}),
        on="date",
        how="left",
    )
    returns = comparison[["date", "ret", "static_basket", "good_only", "bad_only", "mkt_rf"]].rename(columns={"ret": "strategy_4_S1"})
    desc = descriptive_stats_table(returns, turnover={"strategy_4_S1": backtest["switch"]})
    desc["sample_ok"] = sample_ok
    desc["min_train_months"] = ML_MIN_TRAIN_MONTHS
    desc["min_required_oos_months"] = ML_MIN_OOS_MONTHS

    realized = backtest["realized_target"].astype(int) if not backtest.empty else pd.Series(dtype=int)
    pred = (backtest["pred_regime"] == "good").astype(int) if not backtest.empty else pd.Series(dtype=int)
    majority_pred = backtest["majority_pred"].astype(int) if not backtest.empty else pd.Series(dtype=int)
    model_metrics = _binary_classification_metrics(realized, pred)
    majority_metrics = _binary_classification_metrics(realized, majority_pred)
    brier_score = float(np.mean((backtest["p_good"] - realized) ** 2)) if not backtest.empty else np.nan
    majority_brier_score = float(np.mean((backtest["majority_prob"] - realized) ** 2)) if not backtest.empty else np.nan
    agreement_rate = float((backtest["pred_regime"] == backtest["median_regime"]).mean()) if not backtest.empty else np.nan
    importance_mean = importances_df.mean(axis=0).sort_values(ascending=False) if not importances_df.empty else pd.Series(dtype=float)

    gbt_diag = pd.DataFrame(
        [
            {
                "feature": feature,
                "importance": float(val),
                "agreement_rate": agreement_rate,
                "oos_accuracy": model_metrics["accuracy"],
                "majority_accuracy": majority_metrics["accuracy"],
                "oos_balanced_accuracy": model_metrics["balanced_accuracy"],
                "majority_balanced_accuracy": majority_metrics["balanced_accuracy"],
                "brier_score": brier_score,
                "majority_brier_score": majority_brier_score,
                "confusion_tn": model_metrics["tn"],
                "confusion_fp": model_metrics["fp"],
                "confusion_fn": model_metrics["fn"],
                "confusion_tp": model_metrics["tp"],
                "sample_ok": sample_ok,
            }
            for feature, val in importance_mean.items()
        ]
    )
    if gbt_diag.empty:
        gbt_diag = pd.DataFrame(
            [
                {
                    "feature": pd.NA,
                    "importance": np.nan,
                    "agreement_rate": agreement_rate,
                    "oos_accuracy": model_metrics["accuracy"],
                    "majority_accuracy": majority_metrics["accuracy"],
                    "oos_balanced_accuracy": model_metrics["balanced_accuracy"],
                    "majority_balanced_accuracy": majority_metrics["balanced_accuracy"],
                    "brier_score": brier_score,
                    "majority_brier_score": majority_brier_score,
                    "confusion_tn": model_metrics["tn"],
                    "confusion_fp": model_metrics["fp"],
                    "confusion_fn": model_metrics["fn"],
                    "confusion_tp": model_metrics["tp"],
                    "sample_ok": sample_ok,
                }
            ]
        )

    desc_map = desc.set_index("strategy") if not desc.empty else pd.DataFrame()
    benchmark_table = pd.DataFrame(
        [
            {
                "sample_ok": sample_ok,
                "oos_months": int(len(backtest)),
                "feature_count": len(ML_FEATURE_COLS),
                "accuracy": model_metrics["accuracy"],
                "majority_accuracy": majority_metrics["accuracy"],
                "balanced_accuracy": model_metrics["balanced_accuracy"],
                "majority_balanced_accuracy": majority_metrics["balanced_accuracy"],
                "brier_score": brier_score,
                "majority_brier_score": majority_brier_score,
                "strategy_sharpe": float(desc_map.loc["strategy_4_S1", "sharpe"]) if "strategy_4_S1" in desc_map.index else np.nan,
                "static_basket_sharpe": float(desc_map.loc["static_basket", "sharpe"]) if "static_basket" in desc_map.index else np.nan,
                "bad_only_sharpe": float(desc_map.loc["bad_only", "sharpe"]) if "bad_only" in desc_map.index else np.nan,
                "delta_sharpe_vs_static": float(desc_map.loc["strategy_4_S1", "sharpe"] - desc_map.loc["static_basket", "sharpe"])
                if {"strategy_4_S1", "static_basket"}.issubset(desc_map.index)
                else np.nan,
                "delta_sharpe_vs_bad_only": float(desc_map.loc["strategy_4_S1", "sharpe"] - desc_map.loc["bad_only", "sharpe"])
                if {"strategy_4_S1", "bad_only"}.issubset(desc_map.index)
                else np.nan,
                "delta_mean_ann_vs_static": float(desc_map.loc["strategy_4_S1", "mean_ann"] - desc_map.loc["static_basket", "mean_ann"])
                if {"strategy_4_S1", "static_basket"}.issubset(desc_map.index)
                else np.nan,
                "delta_mean_ann_vs_bad_only": float(desc_map.loc["strategy_4_S1", "mean_ann"] - desc_map.loc["bad_only", "mean_ann"])
                if {"strategy_4_S1", "bad_only"}.issubset(desc_map.index)
                else np.nan,
            }
        ]
    )

    backtest[["date", "ret"]].to_csv(dirs.results / "strategy_4_S1_returns.csv", index=False)
    desc.to_csv(dirs.results / "descriptive_stats.csv", index=False)
    gbt_diag.to_csv(dirs.results / "gbt_diagnostics.csv", index=False)
    comparison[["date", "static_basket", "good_only", "bad_only", "mkt_rf"]].to_csv(dirs.results / "benchmarks_returns.csv", index=False)
    sample_status.to_csv(dirs.results / "ml_sample_status.csv", index=False)
    benchmark_table.to_csv(dirs.results / "benchmark_comparison.csv", index=False)

    if not sample_ok:
        _placeholder_csv(
            dirs.results / "alpha_regressions.csv",
            ["strategy", "model", "return_basis", "alpha", "alpha_t", "alpha_p", "R2", "adj_R2", "beta_MktRF", "beta_SMB", "beta_HML", "beta_RMW", "beta_CMA"],
        )
        _placeholder_csv(
            dirs.results / "regression_diagnostics.csv",
            ["strategy", "model", "DW", "BP_p", "JB_p", "max_VIF", "adj_R2"],
        )
        _placeholder_csv(
            dirs.results / "transaction_costs.csv",
            ["strategy", "cost_bps", "sharpe", "ff5_alpha", "breakeven_bps"],
        )
        _placeholder_csv(
            dirs.results / "robustness_subperiod.csv",
            ["strategy", "period", "sharpe", "ff5_alpha", "ff5_alpha_t"],
        )
        return

    alpha, diag = alpha_regression_tables(returns.rename(columns={"mkt_rf": "mkt_rf_benchmark"}), ff_eval)
    alpha["sample_ok"] = sample_ok
    cost = transaction_cost_table("strategy_4_S1", backtest["date"], backtest["ret"], backtest["switch"], ff_eval)
    cost["sample_ok"] = sample_ok

    subperiod_rows = []
    strat_only = returns[["date", "strategy_4_S1"]]
    for label, mask in [("pre_2008", strat_only["date"] <= SUBPERIOD_SPLIT), ("post_2007", strat_only["date"] > SUBPERIOD_SPLIT)]:
        ret_sub = strat_only.loc[mask].copy()
        perf = annualized_metrics(ret_sub["strategy_4_S1"])
        alpha_rows, _ = alpha_regression_tables(ret_sub, ff_eval)
        ff5_alpha, ff5_t = get_ff5_alpha_and_t(alpha_rows)
        subperiod_rows.append(
            {
                "strategy": "strategy_4_S1",
                "period": label,
                "sharpe": perf["sharpe"],
                "ff5_alpha": ff5_alpha,
                "ff5_alpha_t": ff5_t,
                "sample_ok": sample_ok,
            }
        )

    alpha.to_csv(dirs.results / "alpha_regressions.csv", index=False)
    diag.to_csv(dirs.results / "regression_diagnostics.csv", index=False)
    cost.to_csv(dirs.results / "transaction_costs.csv", index=False)
    pd.DataFrame(subperiod_rows).to_csv(dirs.results / "robustness_subperiod.csv", index=False)

    plot_cumulative(returns, ["strategy_4_S1", "static_basket", "mkt_rf"], dirs.graphs / "cumulative_return_comparison.png", f"4 S1 ML {version}: Strategy vs Static Basket vs Mkt-RF")
    if not importance_mean.empty:
        plot_barh(importance_mean, dirs.graphs / "gbt_feature_importances.png", f"4 S1 ML {version}: GBT Feature Importances")
    plot_lines(backtest[["date", "p_good"]], ["p_good"], dirs.graphs / "gbt_probability_timeseries.png", f"4 S1 ML {version}: Predicted Good-Regime Probability", "Probability")


def _solve_regularized_weights(mu: np.ndarray, sigma: np.ndarray, lam: float, w0: np.ndarray) -> np.ndarray:
    mat = sigma + lam * np.eye(len(w0))
    rhs = mu + lam * w0
    return np.linalg.solve(mat, rhs)


def _dynamic_month_return(weight_vector: np.ndarray, z_term: float, z_def: float, returns_row: pd.Series) -> tuple[float, np.ndarray]:
    raw = weight_vector[:3]
    z_term_w = weight_vector[3:6]
    z_def_w = weight_vector[6:9]
    factor_weights = raw + z_term_w * z_term + z_def_w * z_def
    gross = np.abs(factor_weights).sum()
    if gross == 0 or pd.isna(gross):
        normalized = np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)
    else:
        normalized = factor_weights / gross
    realized = float(np.dot(normalized, returns_row[["RMW", "CMA", "HML"]].to_numpy(dtype=float)))
    return realized, normalized


def build_dynamic_panel(version: str) -> pd.DataFrame:
    factor_panel = get_factor_panel(version)
    macro = build_classifier_frame(use_vix_signal=False)
    panel = factor_panel.merge(macro[["date", "z_TERM", "z_DEF", "holding_date"]], on="date", how="inner")
    panel = panel.sort_values("date").reset_index(drop=True)
    panel["z_TERM_lag"] = panel["z_TERM"].shift(1)
    panel["z_DEF_lag"] = panel["z_DEF"].shift(1)
    panel = panel.dropna(subset=["RMW", "CMA", "HML", "z_TERM_lag", "z_DEF_lag"]).reset_index(drop=True)
    for factor in ["RMW", "CMA", "HML"]:
        panel[f"{factor}_x_zTERM"] = panel[factor] * panel["z_TERM_lag"]
        panel[f"{factor}_x_zDEF"] = panel[factor] * panel["z_DEF_lag"]
    return panel


def dynamic_timing_backtest(version: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = build_dynamic_panel(version)
    series_cols = ["RMW", "CMA", "HML", "RMW_x_zTERM", "CMA_x_zTERM", "HML_x_zTERM", "RMW_x_zDEF", "CMA_x_zDEF", "HML_x_zDEF"]
    w0 = np.array([1 / 3, 1 / 3, 1 / 3, 0, 0, 0, 0, 0, 0], dtype=float)

    returns_rows = []
    lambda_rows = []
    test_start = 120
    while test_start < len(panel):
        validation_start = test_start - 24
        estimate = panel.iloc[:validation_start].copy()
        validation = panel.iloc[validation_start:test_start].copy()
        test_end = min(test_start + 12, len(panel))
        test = panel.iloc[test_start:test_end].copy()
        if len(estimate) < 60 or validation.empty or test.empty:
            break

        mu_est = estimate[series_cols].mean().to_numpy(dtype=float)
        sigma_est = estimate[series_cols].cov().to_numpy(dtype=float)
        best_lambda = None
        best_sharpe = -np.inf
        for lam in LAMBDA_GRID:
            weights_9 = _solve_regularized_weights(mu_est, sigma_est, lam, w0)
            validation_returns = []
            for _, row in validation.iterrows():
                realized, _ = _dynamic_month_return(weights_9, float(row["z_TERM_lag"]), float(row["z_DEF_lag"]), row)
                validation_returns.append(realized)
            sharpe = annualized_metrics(pd.Series(validation_returns))["sharpe"]
            if pd.notna(sharpe) and sharpe > best_sharpe:
                best_sharpe = sharpe
                best_lambda = lam
        if best_lambda is None:
            best_lambda = max(LAMBDA_GRID)

        full_train = panel.iloc[:test_start].copy()
        mu_full = full_train[series_cols].mean().to_numpy(dtype=float)
        sigma_full = full_train[series_cols].cov().to_numpy(dtype=float)
        best_weights_9 = _solve_regularized_weights(mu_full, sigma_full, best_lambda, w0)

        prev_weights = None
        for _, row in test.iterrows():
            realized, weights3 = _dynamic_month_return(best_weights_9, float(row["z_TERM_lag"]), float(row["z_DEF_lag"]), row)
            turnover = 0.0 if prev_weights is None else float(np.abs(weights3 - prev_weights).sum())
            returns_rows.append(
                {
                    "date": row["date"],
                    "ret": realized,
                    "W_RMW": weights3[0],
                    "W_CMA": weights3[1],
                    "W_HML": weights3[2],
                    "turnover": turnover,
                    "selected_lambda": best_lambda,
                }
            )
            prev_weights = weights3
        lambda_rows.append({"rebalance_date": test["date"].iloc[0], "selected_lambda": best_lambda, "validation_sharpe": best_sharpe})
        test_start += 12
    return pd.DataFrame(returns_rows), pd.DataFrame(lambda_rows)


def evaluate_dynamic_timing(version: str, base_dir: Path) -> None:
    dirs = ensure_dirs(base_dir)
    returns, lambda_path = dynamic_timing_backtest(version)
    ff_eval = load_ff_factor_panel()
    factor_panel = get_factor_panel(version).copy()
    factor_panel["equal_weight"] = (factor_panel["RMW"] + factor_panel["CMA"] + factor_panel["HML"]) / 3
    centerpiece = build_centerpiece_frames(version)["baseline"][["date", "ret"]].rename(columns={"ret": "strategy_1A_med"})

    comparison = returns[["date", "ret"]].rename(columns={"ret": "strategy_3A_T1"})
    comparison = comparison.merge(factor_panel[["date", "equal_weight"]], on="date", how="left").merge(centerpiece, on="date", how="left")

    desc = descriptive_stats_table(comparison, turnover={"strategy_3A_T1": returns["turnover"]})
    alpha, diag = alpha_regression_tables(comparison, ff_eval)
    cost = transaction_cost_table("strategy_3A_T1", returns["date"], returns["ret"], returns["turnover"], ff_eval)

    subperiod_rows = []
    strat_only = comparison[["date", "strategy_3A_T1"]]
    for label, mask in [("pre_2008", strat_only["date"] <= SUBPERIOD_SPLIT), ("post_2007", strat_only["date"] > SUBPERIOD_SPLIT)]:
        ret_sub = strat_only.loc[mask].copy()
        perf = annualized_metrics(ret_sub["strategy_3A_T1"])
        alpha_rows, _ = alpha_regression_tables(ret_sub, ff_eval)
        ff5_alpha, ff5_t = get_ff5_alpha_and_t(alpha_rows)
        subperiod_rows.append(
            {
                "strategy": "strategy_3A_T1",
                "period": label,
                "sharpe": perf["sharpe"],
                "ff5_alpha": ff5_alpha,
                "ff5_alpha_t": ff5_t,
            }
        )

    rolling = comparison[["date", "strategy_3A_T1", "equal_weight"]].copy()
    for col in ["strategy_3A_T1", "equal_weight"]:
        rolling[f"{col}_rolling36_sharpe"] = rolling[col].rolling(36).mean() / rolling[col].rolling(36).std(ddof=1) * np.sqrt(12)

    returns[["date", "ret"]].rename(columns={"ret": "strategy_3A_T1_returns"}).to_csv(dirs.results / "strategy_3A_T1_returns.csv", index=False)
    comparison.to_csv(dirs.results / "benchmarks_returns.csv", index=False)
    desc.to_csv(dirs.results / "descriptive_stats.csv", index=False)
    alpha.to_csv(dirs.results / "alpha_regressions.csv", index=False)
    diag.to_csv(dirs.results / "regression_diagnostics.csv", index=False)
    cost.to_csv(dirs.results / "transaction_costs.csv", index=False)
    pd.DataFrame(subperiod_rows).to_csv(dirs.results / "robustness_subperiod.csv", index=False)
    lambda_path.to_csv(dirs.results / "lambda_path.csv", index=False)
    returns[["date", "W_RMW", "W_CMA", "W_HML", "turnover", "selected_lambda"]].to_csv(dirs.results / "weight_evolution.csv", index=False)

    plot_cumulative(comparison, ["strategy_3A_T1", "equal_weight", "strategy_1A_med"], dirs.graphs / "cumulative_return_comparison.png", f"3A T1 Dynamic Timing {version}: Strategy vs Benchmarks")
    plot_lines(returns[["date", "W_RMW", "W_CMA", "W_HML"]], ["W_RMW", "W_CMA", "W_HML"], dirs.graphs / "weight_evolution.png", f"3A T1 Dynamic Timing {version}: Weight Evolution", "Weight")
    plot_lines(lambda_path.rename(columns={"rebalance_date": "date"})[["date", "selected_lambda"]], ["selected_lambda"], dirs.graphs / "lambda_selection_path.png", f"3A T1 Dynamic Timing {version}: Lambda Path", "Lambda")
    plot_lines(rolling[["date", "strategy_3A_T1_rolling36_sharpe", "equal_weight_rolling36_sharpe"]], ["strategy_3A_T1_rolling36_sharpe", "equal_weight_rolling36_sharpe"], dirs.graphs / "rolling_sharpe_comparison.png", f"3A T1 Dynamic Timing {version}: Rolling 36M Sharpe", "Sharpe")


@dataclass(frozen=True)
class DynamicTimingConfig:
    validation_months: int = 24
    validation_aggregation: str = "single"
    z_clip: float | None = None
    factor_cap: float | None = None
    interaction_gamma: float = 1.0
    cov_shrink_rho: float = 0.0
    lambda_selection_metric: str = "gross_sharpe"
    predictors: tuple[str, ...] = ("z_TERM", "z_DEF")


@dataclass
class DynamicRunArtifacts:
    config: DynamicTimingConfig
    returns: pd.DataFrame
    lambda_path: pd.DataFrame
    comparison: pd.DataFrame
    descriptive: pd.DataFrame
    alpha: pd.DataFrame
    diagnostics: pd.DataFrame
    cost: pd.DataFrame
    subperiod: pd.DataFrame
    rolling: pd.DataFrame
    metrics: dict[str, float]


def _dynamic_factor_names() -> list[str]:
    return ["RMW", "CMA", "HML"]


def _dynamic_interaction_suffix(predictor: str) -> str:
    return predictor.replace("z_", "")


def _dynamic_series_cols(config: DynamicTimingConfig) -> list[str]:
    cols = _dynamic_factor_names().copy()
    for predictor in config.predictors:
        suffix = _dynamic_interaction_suffix(predictor)
        cols.extend([f"{factor}_x_{suffix}" for factor in _dynamic_factor_names()])
    return cols


def _dynamic_w0(config: DynamicTimingConfig) -> np.ndarray:
    return np.array([1 / 3, 1 / 3, 1 / 3] + [0.0] * (3 * len(config.predictors)), dtype=float)


def _apply_covariance_shrinkage(sigma: np.ndarray, rho: float) -> np.ndarray:
    if rho <= 0:
        return sigma
    diag = np.diag(np.diag(sigma))
    return (1 - rho) * sigma + rho * diag


def _map_dynamic_factor_weights(
    weight_vector: np.ndarray,
    predictor_values: np.ndarray,
    config: DynamicTimingConfig,
) -> tuple[np.ndarray, bool]:
    raw = weight_vector[:3]
    interactions = weight_vector[3:].reshape(len(config.predictors), 3) if len(config.predictors) else np.zeros((0, 3))
    factor_weights = raw + config.interaction_gamma * np.sum(interactions * predictor_values[:, None], axis=0)
    clip_hit = False
    if config.factor_cap is not None:
        clipped = np.clip(factor_weights, -config.factor_cap, config.factor_cap)
        clip_hit = bool(np.any(np.abs(clipped - factor_weights) > 1e-12))
        factor_weights = clipped
    gross = np.abs(factor_weights).sum()
    if gross == 0 or pd.isna(gross):
        normalized = np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)
    else:
        normalized = factor_weights / gross
    return normalized, clip_hit


def _simulate_dynamic_period(
    weight_vector: np.ndarray,
    period: pd.DataFrame,
    config: DynamicTimingConfig,
    prev_weights: np.ndarray | None = None,
) -> tuple[pd.DataFrame, np.ndarray | None]:
    rows: list[dict[str, float | pd.Timestamp]] = []
    last_weights = prev_weights
    lag_cols = [f"{pred}_lag" for pred in config.predictors]
    for _, row in period.iterrows():
        predictor_values = row[lag_cols].to_numpy(dtype=float)
        weights3, clip_hit = _map_dynamic_factor_weights(weight_vector, predictor_values, config)
        turnover = 0.0 if last_weights is None else float(np.abs(weights3 - last_weights).sum())
        realized = float(np.dot(weights3, row[_dynamic_factor_names()].to_numpy(dtype=float)))
        rows.append(
            {
                "date": row["date"],
                "ret": realized,
                "W_RMW": weights3[0],
                "W_CMA": weights3[1],
                "W_HML": weights3[2],
                "turnover": turnover,
                "clip_hit": float(clip_hit),
            }
        )
        last_weights = weights3
    return pd.DataFrame(rows), last_weights


def build_dynamic_panel_upgrade(version: str, config: DynamicTimingConfig) -> pd.DataFrame:
    factor_panel = get_factor_panel(version)
    macro = build_classifier_frame(use_vix_signal=False)
    macro_cols = ["date", *config.predictors, "holding_date"]
    panel = factor_panel.merge(macro[macro_cols], on="date", how="inner")
    panel = panel.sort_values("date").reset_index(drop=True)
    lag_cols = []
    for predictor in config.predictors:
        lag_col = f"{predictor}_lag"
        lag_cols.append(lag_col)
        panel[lag_col] = panel[predictor].shift(1)
        if config.z_clip is not None:
            panel[lag_col] = panel[lag_col].clip(lower=-config.z_clip, upper=config.z_clip)
    panel = panel.dropna(subset=[*_dynamic_factor_names(), *lag_cols]).reset_index(drop=True)
    for predictor in config.predictors:
        suffix = _dynamic_interaction_suffix(predictor)
        lag_col = f"{predictor}_lag"
        for factor in _dynamic_factor_names():
            panel[f"{factor}_x_{suffix}"] = panel[factor] * panel[lag_col]
    return panel


def _collect_validation_folds(
    panel: pd.DataFrame,
    test_start: int,
    config: DynamicTimingConfig,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    offsets = [0] if config.validation_aggregation == "single" else [0, 12, 24]
    folds: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for offset in offsets:
        fold_test_start = test_start - offset
        validation_start = fold_test_start - config.validation_months
        if validation_start <= 0:
            continue
        estimate = panel.iloc[:validation_start].copy()
        validation = panel.iloc[validation_start:fold_test_start].copy()
        if len(estimate) < 60 or validation.empty:
            continue
        folds.append((estimate, validation))
    return folds


def _dynamic_validation_score(
    lam: float,
    folds: list[tuple[pd.DataFrame, pd.DataFrame]],
    config: DynamicTimingConfig,
    series_cols: list[str],
    w0: np.ndarray,
) -> tuple[float, float]:
    score_values: list[float] = []
    turnover_values: list[float] = []
    for estimate, validation in folds:
        mu_est = estimate[series_cols].mean().to_numpy(dtype=float)
        sigma_est = estimate[series_cols].cov().to_numpy(dtype=float)
        sigma_est = _apply_covariance_shrinkage(sigma_est, config.cov_shrink_rho)
        weights = _solve_regularized_weights(mu_est, sigma_est, lam, w0)
        simulated, _ = _simulate_dynamic_period(weights, validation, config, prev_weights=None)
        if simulated.empty:
            continue
        gross_sharpe = annualized_metrics(simulated["ret"])["sharpe"]
        net25_sharpe = annualized_metrics(simulated["ret"] - 0.0025 * simulated["turnover"])["sharpe"]
        turnover_mean = float(simulated["turnover"].mean())
        score_values.append(net25_sharpe if config.lambda_selection_metric == "net25_turnover" else gross_sharpe)
        turnover_values.append(turnover_mean)
    if not score_values:
        return np.nan, np.nan
    if config.validation_aggregation == "median3":
        return float(np.nanmedian(score_values)), float(np.nanmedian(turnover_values))
    return float(score_values[0]), float(turnover_values[0])


def _select_dynamic_lambda(
    panel: pd.DataFrame,
    test_start: int,
    config: DynamicTimingConfig,
    series_cols: list[str],
    w0: np.ndarray,
) -> tuple[float, float, int]:
    folds = _collect_validation_folds(panel, test_start, config)
    if not folds:
        return max(LAMBDA_GRID), np.nan, 0

    best_lambda = max(LAMBDA_GRID)
    best_score = -np.inf
    best_turnover = np.inf
    for lam in LAMBDA_GRID:
        score, turnover = _dynamic_validation_score(lam, folds, config, series_cols, w0)
        if pd.isna(score):
            continue
        better_score = score > best_score + 1e-12
        better_tie = (
            config.lambda_selection_metric == "net25_turnover"
            and abs(score - best_score) <= 1e-12
            and turnover < best_turnover - 1e-12
        )
        if better_score or better_tie:
            best_lambda = lam
            best_score = score
            best_turnover = turnover
    return best_lambda, best_score, len(folds)


def dynamic_timing_backtest_upgrade(
    version: str,
    config: DynamicTimingConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = build_dynamic_panel_upgrade(version, config)
    series_cols = _dynamic_series_cols(config)
    w0 = _dynamic_w0(config)
    returns_rows: list[dict[str, float | pd.Timestamp]] = []
    lambda_rows: list[dict[str, float | str | int]] = []
    test_start = 120
    prev_weights: np.ndarray | None = None
    while test_start < len(panel):
        test_end = min(test_start + 12, len(panel))
        test = panel.iloc[test_start:test_end].copy()
        if test.empty:
            break
        best_lambda, best_score, n_folds = _select_dynamic_lambda(panel, test_start, config, series_cols, w0)
        full_train = panel.iloc[:test_start].copy()
        if len(full_train) < 60:
            break
        mu_full = full_train[series_cols].mean().to_numpy(dtype=float)
        sigma_full = full_train[series_cols].cov().to_numpy(dtype=float)
        sigma_full = _apply_covariance_shrinkage(sigma_full, config.cov_shrink_rho)
        weight_vector = _solve_regularized_weights(mu_full, sigma_full, best_lambda, w0)
        simulated, prev_weights = _simulate_dynamic_period(weight_vector, test, config, prev_weights=prev_weights)
        if simulated.empty:
            break
        simulated["selected_lambda"] = best_lambda
        returns_rows.extend(simulated.to_dict("records"))
        lambda_rows.append(
            {
                "rebalance_date": test["date"].iloc[0],
                "selected_lambda": best_lambda,
                "validation_score": best_score,
                "n_validation_folds": n_folds,
                "validation_metric": config.lambda_selection_metric,
                "validation_aggregation": config.validation_aggregation,
            }
        )
        test_start += 12
    return pd.DataFrame(returns_rows), pd.DataFrame(lambda_rows)


def _dynamic_upgrade_metrics(
    returns: pd.DataFrame,
    lambda_path: pd.DataFrame,
    cost: pd.DataFrame,
    alpha: pd.DataFrame,
) -> dict[str, float]:
    strat_metrics = annualized_metrics(returns["ret"])
    ff5_row = alpha.loc[(alpha["strategy"] == "strategy_3A_T1") & (alpha["model"] == "FF5")]
    ff5_alpha = float(ff5_row["alpha"].iloc[0]) if not ff5_row.empty else np.nan
    ff5_alpha_t = float(ff5_row["alpha_t"].iloc[0]) if not ff5_row.empty else np.nan
    cost25 = cost.loc[cost["cost_bps"] == 25, "sharpe"]
    cost50 = cost.loc[cost["cost_bps"] == 50, "sharpe"]
    weights = returns[["W_RMW", "W_CMA", "W_HML"]].copy()
    abs_delta = weights.diff().abs()
    lambda_share = lambda_path["selected_lambda"] if not lambda_path.empty else pd.Series(dtype=float)
    return {
        "gross_sharpe": float(strat_metrics["sharpe"]),
        "net_sharpe_25bps": float(cost25.iloc[0]) if not cost25.empty else np.nan,
        "net_sharpe_50bps": float(cost50.iloc[0]) if not cost50.empty else np.nan,
        "mean_ann_return": float(strat_metrics["mean_ann"]),
        "vol_ann": float(strat_metrics["vol_ann"]),
        "ff5_alpha_ann": ff5_alpha * 12 if pd.notna(ff5_alpha) else np.nan,
        "ff5_alpha_t": ff5_alpha_t,
        "max_drawdown": float(strat_metrics["max_dd"]),
        "turnover": float(returns["turnover"].mean()) if len(returns) else np.nan,
        "lambda_extreme_share": float(lambda_share.isin([1, 3, 1000, 10000, 100000]).mean()) if len(lambda_share) else np.nan,
        "lambda_low_share": float(lambda_share.isin([1, 3]).mean()) if len(lambda_share) else np.nan,
        "lambda_high_share": float(lambda_share.isin([1000, 10000, 100000]).mean()) if len(lambda_share) else np.nan,
        "weight_instability": float(returns["turnover"].mean()) if len(returns) else np.nan,
        "avg_abs_delta_weight": float(abs_delta.mean(axis=1, skipna=True).mean()) if len(abs_delta) else np.nan,
        "max_abs_weight": float(weights.abs().to_numpy().max()) if len(weights) else np.nan,
        "clip_rate": float(returns["clip_hit"].mean()) if "clip_hit" in returns.columns and len(returns) else 0.0,
        "live_months": int(len(returns)),
    }


def _run_dynamic_strategy(version: str, config: DynamicTimingConfig) -> DynamicRunArtifacts:
    returns, lambda_path = dynamic_timing_backtest_upgrade(version, config=config)
    ff_eval = load_ff_factor_panel()
    factor_panel = get_factor_panel(version).copy()
    factor_panel["equal_weight"] = (factor_panel["RMW"] + factor_panel["CMA"] + factor_panel["HML"]) / 3
    centerpiece = build_centerpiece_frames(version)["baseline"][["date", "ret"]].rename(columns={"ret": "strategy_1A_med"})
    comparison = returns[["date", "ret"]].rename(columns={"ret": "strategy_3A_T1"})
    comparison = comparison.merge(factor_panel[["date", "equal_weight"]], on="date", how="left").merge(centerpiece, on="date", how="left")
    desc = descriptive_stats_table(comparison, turnover={"strategy_3A_T1": returns["turnover"]})
    alpha, diag = alpha_regression_tables(comparison, ff_eval)
    cost = transaction_cost_table("strategy_3A_T1", returns["date"], returns["ret"], returns["turnover"], ff_eval)

    subperiod_rows = []
    strat_only = comparison[["date", "strategy_3A_T1"]]
    for label, mask in [("pre_2008", strat_only["date"] <= SUBPERIOD_SPLIT), ("post_2007", strat_only["date"] > SUBPERIOD_SPLIT)]:
        ret_sub = strat_only.loc[mask].copy()
        perf = annualized_metrics(ret_sub["strategy_3A_T1"])
        alpha_rows, _ = alpha_regression_tables(ret_sub, ff_eval)
        ff5_alpha, ff5_t = get_ff5_alpha_and_t(alpha_rows)
        subperiod_rows.append(
            {
                "strategy": "strategy_3A_T1",
                "period": label,
                "sharpe": perf["sharpe"],
                "ff5_alpha": ff5_alpha,
                "ff5_alpha_t": ff5_t,
            }
        )
    subperiod = pd.DataFrame(subperiod_rows)

    rolling = comparison[["date", "strategy_3A_T1", "equal_weight"]].copy()
    for col in ["strategy_3A_T1", "equal_weight"]:
        rolling[f"{col}_rolling36_sharpe"] = rolling[col].rolling(36).mean() / rolling[col].rolling(36).std(ddof=1) * np.sqrt(12)

    metrics = _dynamic_upgrade_metrics(returns, lambda_path, cost, alpha)
    return DynamicRunArtifacts(
        config=config,
        returns=returns,
        lambda_path=lambda_path,
        comparison=comparison,
        descriptive=desc,
        alpha=alpha,
        diagnostics=diag,
        cost=cost,
        subperiod=subperiod,
        rolling=rolling,
        metrics=metrics,
    )


def _compare_dynamic_metrics(old: dict[str, float], new: dict[str, float]) -> tuple[bool, str]:
    if pd.isna(new["net_sharpe_25bps"]) or pd.isna(old["net_sharpe_25bps"]):
        return False, "rejected: missing net_sharpe_25"
    if new["net_sharpe_25bps"] <= old["net_sharpe_25bps"] + 1e-12:
        return False, "rejected: net_sharpe_25 worse"

    secondary_checks = [
        pd.notna(new["net_sharpe_50bps"]) and pd.notna(old["net_sharpe_50bps"]) and new["net_sharpe_50bps"] > old["net_sharpe_50bps"] + 1e-12,
        pd.notna(new["ff5_alpha_ann"]) and pd.notna(old["ff5_alpha_ann"]) and new["ff5_alpha_ann"] > old["ff5_alpha_ann"] + 1e-12,
        pd.notna(new["lambda_extreme_share"]) and pd.notna(old["lambda_extreme_share"]) and new["lambda_extreme_share"] < old["lambda_extreme_share"] - 1e-12,
        pd.notna(new["weight_instability"]) and pd.notna(old["weight_instability"]) and new["weight_instability"] < old["weight_instability"] - 1e-12,
        pd.notna(new["turnover"]) and pd.notna(old["turnover"]) and new["turnover"] < old["turnover"] - 1e-12,
    ]
    if not any(secondary_checks):
        return False, "rejected: no secondary improvement"

    old_dd = abs(old["max_drawdown"]) if pd.notna(old["max_drawdown"]) else np.nan
    new_dd = abs(new["max_drawdown"]) if pd.notna(new["max_drawdown"]) else np.nan
    if pd.notna(old_dd) and pd.notna(new_dd) and new_dd > old_dd * 1.10 + 1e-12:
        return False, "rejected: max drawdown worsened beyond guardrail"
    if pd.notna(old["turnover"]) and pd.notna(new["turnover"]) and new["turnover"] > old["turnover"] * 1.15 + 1e-12:
        return False, "rejected: turnover worsened beyond guardrail"
    if pd.notna(old["lambda_extreme_share"]) and pd.notna(new["lambda_extreme_share"]) and new["lambda_extreme_share"] > old["lambda_extreme_share"] + 0.10 + 1e-12:
        return False, "rejected: lambda_extreme_share worsened beyond guardrail"
    if pd.notna(old["weight_instability"]) and pd.notna(new["weight_instability"]) and new["weight_instability"] > old["weight_instability"] * 1.15 + 1e-12:
        return False, "rejected: weight_instability worsened beyond guardrail"
    if int(new["live_months"]) < int(old["live_months"]):
        return False, "rejected: live sample shrank"
    return True, "accepted: net_sharpe_25 improved"


def _dynamic_candidate_queue() -> list[tuple[str, callable]]:
    return [
        ("candidate_1_validation_36m", lambda cfg: replace(cfg, validation_months=36)),
        ("candidate_2_validation_48m", lambda cfg: replace(cfg, validation_months=48)),
        ("candidate_3_lambda_smoothing", lambda cfg: replace(cfg, validation_aggregation="median3")),
        ("candidate_4_z_clip_3", lambda cfg: replace(cfg, z_clip=3.0)),
        ("candidate_5_factor_cap_60", lambda cfg: replace(cfg, factor_cap=0.60)),
        ("candidate_6_gamma_050", lambda cfg: replace(cfg, interaction_gamma=0.50)),
        ("candidate_7_gamma_025", lambda cfg: replace(cfg, interaction_gamma=0.25)),
        ("candidate_8_cov_shrink_025", lambda cfg: replace(cfg, cov_shrink_rho=0.25)),
        ("candidate_9_cov_shrink_050", lambda cfg: replace(cfg, cov_shrink_rho=0.50)),
        ("candidate_10_lambda_net25", lambda cfg: replace(cfg, lambda_selection_metric="net25_turnover")),
        ("candidate_11_add_vix", lambda cfg: replace(cfg, predictors=("z_TERM", "z_DEF", "z_VIX"))),
        ("candidate_12_vix_plus_clip", lambda cfg: replace(cfg, predictors=("z_TERM", "z_DEF", "z_VIX"), z_clip=3.0)),
    ]


def _dynamic_version_slug(label: str) -> str:
    return label.replace(" ", "_").replace("/", "_").replace(":", "_").replace("-", "_").replace(".", "_")


def _save_dynamic_snapshot(run: DynamicRunArtifacts, results_dir: Path, graphs_dir: Path, label: str) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)
    slug = _dynamic_version_slug(label)
    pd.DataFrame([run.metrics]).to_csv(results_dir / f"{slug}_metrics.csv", index=False)
    run.returns.to_csv(results_dir / f"{slug}_returns.csv", index=False)
    run.lambda_path.to_csv(results_dir / f"{slug}_lambda_path.csv", index=False)
    run.returns[["date", "W_RMW", "W_CMA", "W_HML", "turnover", "selected_lambda", "clip_hit"]].to_csv(results_dir / f"{slug}_weights.csv", index=False)
    plot_cumulative(run.comparison, ["strategy_3A_T1", "equal_weight", "strategy_1A_med"], graphs_dir / f"{slug}_cumulative_return_comparison.png", f"3A T1 Dynamic Timing: {label}")
    plot_lines(run.returns[["date", "W_RMW", "W_CMA", "W_HML"]], ["W_RMW", "W_CMA", "W_HML"], graphs_dir / f"{slug}_weight_evolution.png", f"3A T1 Dynamic Timing: {label} Weights", "Weight")
    if not run.lambda_path.empty:
        plot_lines(run.lambda_path.rename(columns={"rebalance_date": "date"})[["date", "selected_lambda"]], ["selected_lambda"], graphs_dir / f"{slug}_lambda_selection_path.png", f"3A T1 Dynamic Timing: {label} Lambda Path", "Lambda")
    plot_lines(run.rolling[["date", "strategy_3A_T1_rolling36_sharpe", "equal_weight_rolling36_sharpe"]], ["strategy_3A_T1_rolling36_sharpe", "equal_weight_rolling36_sharpe"], graphs_dir / f"{slug}_rolling_sharpe_comparison.png", f"3A T1 Dynamic Timing: {label} Rolling 36M Sharpe", "Sharpe")


def _write_dynamic_final_outputs(run: DynamicRunArtifacts, dirs: OutputDirs) -> None:
    run.returns[["date", "ret"]].rename(columns={"ret": "strategy_3A_T1_returns"}).to_csv(dirs.results / "strategy_3A_T1_returns.csv", index=False)
    run.comparison.to_csv(dirs.results / "benchmarks_returns.csv", index=False)
    run.descriptive.to_csv(dirs.results / "descriptive_stats.csv", index=False)
    run.alpha.to_csv(dirs.results / "alpha_regressions.csv", index=False)
    run.diagnostics.to_csv(dirs.results / "regression_diagnostics.csv", index=False)
    run.cost.to_csv(dirs.results / "transaction_costs.csv", index=False)
    run.subperiod.to_csv(dirs.results / "robustness_subperiod.csv", index=False)
    run.lambda_path.to_csv(dirs.results / "lambda_path.csv", index=False)
    run.returns[["date", "W_RMW", "W_CMA", "W_HML", "turnover", "selected_lambda", "clip_hit"]].to_csv(dirs.results / "weight_evolution.csv", index=False)
    pd.DataFrame([run.metrics]).to_csv(dirs.results / "final_accepted_metrics.csv", index=False)
    plot_cumulative(run.comparison, ["strategy_3A_T1", "equal_weight", "strategy_1A_med"], dirs.graphs / "cumulative_return_comparison.png", "3A T1 Dynamic Timing: Final Accepted vs Benchmarks")
    plot_lines(run.returns[["date", "W_RMW", "W_CMA", "W_HML"]], ["W_RMW", "W_CMA", "W_HML"], dirs.graphs / "weight_evolution.png", "3A T1 Dynamic Timing: Final Accepted Weight Evolution", "Weight")
    if not run.lambda_path.empty:
        plot_lines(run.lambda_path.rename(columns={"rebalance_date": "date"})[["date", "selected_lambda"]], ["selected_lambda"], dirs.graphs / "lambda_selection_path.png", "3A T1 Dynamic Timing: Final Accepted Lambda Path", "Lambda")
    plot_lines(run.rolling[["date", "strategy_3A_T1_rolling36_sharpe", "equal_weight_rolling36_sharpe"]], ["strategy_3A_T1_rolling36_sharpe", "equal_weight_rolling36_sharpe"], dirs.graphs / "rolling_sharpe_comparison.png", "3A T1 Dynamic Timing: Final Accepted Rolling 36M Sharpe", "Sharpe")


def evaluate_dynamic_timing(version: str, base_dir: Path) -> None:
    dirs = ensure_dirs(base_dir)
    snapshot_results = dirs.results / "upgrade_snapshots"
    snapshot_graphs = dirs.graphs / "upgrade_snapshots"
    baseline_config = DynamicTimingConfig()
    baseline_run = _run_dynamic_strategy(version, baseline_config)
    accepted_run = baseline_run
    accepted_config = baseline_config
    accepted_label = "baseline"
    accepted_changes: list[str] = []
    rejected_changes: list[str] = []
    log_rows: list[dict[str, float | str | int]] = []
    iteration_id = 0
    pass_number = 1
    trivial_accept_streak = 0
    _save_dynamic_snapshot(baseline_run, snapshot_results, snapshot_graphs, "baseline_accepted")

    while True:
        pass_accepts = 0
        for candidate_name, apply_change in _dynamic_candidate_queue():
            iteration_id += 1
            if candidate_name == "candidate_12_vix_plus_clip" and "z_VIX" not in accepted_config.predictors:
                candidate_run = accepted_run
                accepted = False
                reason = "rejected: candidate 11 not accepted previously"
            else:
                candidate_config = apply_change(accepted_config)
                if candidate_config == accepted_config:
                    candidate_run = accepted_run
                    accepted = False
                    reason = "rejected: no change from accepted version"
                else:
                    candidate_run = _run_dynamic_strategy(version, candidate_config)
                    accepted, reason = _compare_dynamic_metrics(accepted_run.metrics, candidate_run.metrics)

            _save_dynamic_snapshot(accepted_run, snapshot_results, snapshot_graphs, f"iter_{iteration_id:03d}_accepted_current")
            _save_dynamic_snapshot(candidate_run, snapshot_results, snapshot_graphs, f"iter_{iteration_id:03d}_{candidate_name}")

            old_metrics = accepted_run.metrics
            new_metrics = candidate_run.metrics
            log_rows.append(
                {
                    "iteration_id": iteration_id,
                    "pass_number": pass_number,
                    "candidate_name": candidate_name,
                    "parent_version": accepted_label,
                    "candidate_version": f"{accepted_label}__{candidate_name}",
                    "accepted": "YES" if accepted else "NO",
                    "reason": reason,
                    "gross_sharpe_old": old_metrics["gross_sharpe"],
                    "gross_sharpe_new": new_metrics["gross_sharpe"],
                    "net_sharpe_25_old": old_metrics["net_sharpe_25bps"],
                    "net_sharpe_25_new": new_metrics["net_sharpe_25bps"],
                    "net_sharpe_50_old": old_metrics["net_sharpe_50bps"],
                    "net_sharpe_50_new": new_metrics["net_sharpe_50bps"],
                    "ff5_alpha_old": old_metrics["ff5_alpha_ann"],
                    "ff5_alpha_new": new_metrics["ff5_alpha_ann"],
                    "max_dd_old": old_metrics["max_drawdown"],
                    "max_dd_new": new_metrics["max_drawdown"],
                    "turnover_old": old_metrics["turnover"],
                    "turnover_new": new_metrics["turnover"],
                    "lambda_extreme_share_old": old_metrics["lambda_extreme_share"],
                    "lambda_extreme_share_new": new_metrics["lambda_extreme_share"],
                    "weight_instability_old": old_metrics["weight_instability"],
                    "weight_instability_new": new_metrics["weight_instability"],
                    "live_months_old": old_metrics["live_months"],
                    "live_months_new": new_metrics["live_months"],
                }
            )
            pd.DataFrame(log_rows).to_csv(dirs.results / "strategy_3A_T1_upgrade_log.csv", index=False)

            if accepted:
                gain = float(candidate_run.metrics["net_sharpe_25bps"] - old_metrics["net_sharpe_25bps"])
                accepted_run = candidate_run
                accepted_config = candidate_run.config
                accepted_label = f"{accepted_label}__{candidate_name}"
                accepted_changes.append(candidate_name)
                pass_accepts += 1
                trivial_accept_streak = trivial_accept_streak + 1 if gain < 0.01 else 0
                _save_dynamic_snapshot(accepted_run, snapshot_results, snapshot_graphs, f"{accepted_label}_accepted")
                if trivial_accept_streak > 3:
                    rejected_changes.append("stopped: more than 3 consecutive accepted changes produced trivial improvements")
                    break
            else:
                rejected_changes.append(f"{candidate_name}: {reason}")
        if trivial_accept_streak > 3 or pass_accepts == 0:
            break
        pass_number += 1

    final_lambda = accepted_run.lambda_path["selected_lambda"].value_counts(normalize=True).rename_axis("selected_lambda").reset_index(name="share") if not accepted_run.lambda_path.empty else pd.DataFrame(columns=["selected_lambda", "share"])
    final_lambda.to_csv(dirs.results / "final_lambda_distribution.csv", index=False)
    pd.DataFrame([baseline_run.metrics]).to_csv(dirs.results / "baseline_metrics.csv", index=False)
    final_yes = accepted_run.metrics["net_sharpe_25bps"] > baseline_run.metrics["net_sharpe_25bps"] + 1e-12
    report_lines = [
        "# Strategy 3A-T1 Upgrade Loop Final Report",
        "",
        "## Baseline Metrics",
        f"- gross_sharpe: {baseline_run.metrics['gross_sharpe']:.4f}",
        f"- net_sharpe_25bps: {baseline_run.metrics['net_sharpe_25bps']:.4f}",
        f"- net_sharpe_50bps: {baseline_run.metrics['net_sharpe_50bps']:.4f}",
        f"- ff5_alpha_ann: {baseline_run.metrics['ff5_alpha_ann']:.6f}",
        f"- max_drawdown: {baseline_run.metrics['max_drawdown']:.4f}",
        "",
        "## Final Accepted Metrics",
        f"- gross_sharpe: {accepted_run.metrics['gross_sharpe']:.4f}",
        f"- net_sharpe_25bps: {accepted_run.metrics['net_sharpe_25bps']:.4f}",
        f"- net_sharpe_50bps: {accepted_run.metrics['net_sharpe_50bps']:.4f}",
        f"- ff5_alpha_ann: {accepted_run.metrics['ff5_alpha_ann']:.6f}",
        f"- ff5_alpha_t: {accepted_run.metrics['ff5_alpha_t']:.4f}",
        f"- max_drawdown: {accepted_run.metrics['max_drawdown']:.4f}",
        "",
        "## Accepted Changes",
    ]
    report_lines.extend([f"- {name}" for name in accepted_changes] if accepted_changes else ["- none"])
    report_lines.extend(["", "## Rejected Changes"])
    report_lines.extend([f"- {text}" for text in rejected_changes] if rejected_changes else ["- none"])
    report_lines.extend(["", "## Final Lambda Distribution"])
    report_lines.extend([f"- lambda {int(row['selected_lambda'])}: {row['share']:.2%}" for _, row in final_lambda.iterrows()] if not final_lambda.empty else ["- none"])
    report_lines.extend(
        [
            "",
            "## Final Turnover and Weight Instability",
            f"- turnover: {accepted_run.metrics['turnover']:.6f}",
            f"- weight_instability: {accepted_run.metrics['weight_instability']:.6f}",
            f"- avg_abs_delta_weight: {accepted_run.metrics['avg_abs_delta_weight']:.6f}",
            f"- max_abs_weight: {accepted_run.metrics['max_abs_weight']:.6f}",
            f"- clip_rate: {accepted_run.metrics['clip_rate']:.6f}",
            "",
            "## VIX Needed",
            f"- {'YES' if 'z_VIX' in accepted_config.predictors else 'NO'}",
            "",
            "## Final Verdict",
            f"- {'YES' if final_yes else 'NO'}",
        ]
    )
    (dirs.results / "strategy_3A_T1_final_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    (dirs.results / "strategy_3A_T1_final_label.txt").write_text("Strategy_3A_T1_final_accepted", encoding="utf-8")
    _write_dynamic_final_outputs(accepted_run, dirs)
