from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from pandas.tseries.offsets import MonthEnd
import statsmodels.api as sm


PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent
DATA_DIR = REPO_ROOT / "Data"


LICENSED_DATA_NOTE = (
    "\n\nThis file is not distributed with the repository because it is "
    "subject to a WRDS / LSEG subscription licence.\n"
    "See Data/README.md for instructions on rebuilding it from your own "
    "institutional access.\n"
)


def require_data(path: Path) -> Path:
    """Return `path`, or raise a clear error if the licensed extract is absent."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Required data file not found: {path}{LICENSED_DATA_NOTE}")
    return path

SAMPLE_START = pd.Timestamp("1990-01-31")
SAMPLE_END = pd.Timestamp("2024-09-30")
OUTPUT_CSV = PROJECT_ROOT / "credit_factor_rotation_returns.csv"
OUTPUT_PLOT = PROJECT_ROOT / "credit_factor_rotation_regime_returns.png"
OUTPUT_BENCHMARKS_CSV = PROJECT_ROOT / "credit_factor_rotation_benchmarks.csv"
OUTPUT_SUMMARY_TABLE_CSV = PROJECT_ROOT / "credit_factor_rotation_summary_table.csv"
OUTPUT_ALPHA_TABLE_CSV = PROJECT_ROOT / "credit_factor_rotation_alpha_table.csv"
OUTPUT_ROBUSTNESS_TABLE_CSV = PROJECT_ROOT / "credit_factor_rotation_robustness_summary.csv"
OUTPUT_ML_SUMMARY_CSV = PROJECT_ROOT / "credit_factor_rotation_ml_summary.csv"
OUTPUT_ML_FEATURE_IMPORTANCE_CSV = PROJECT_ROOT / "credit_factor_rotation_ml_feature_importance.csv"
OUTPUT_CUMULATIVE_PLOT = PROJECT_ROOT / "credit_factor_rotation_cumulative_comparison.png"
OUTPUT_ROBUSTNESS_PLOT = PROJECT_ROOT / "credit_factor_rotation_robustness_comparison.png"
OUTPUT_FF_GOOD_CASH_PLOT = PROJECT_ROOT / "credit_factor_rotation_ff_good_cash_comparison.png"
OUTPUT_ML_IMPORTANCE_PLOT = PROJECT_ROOT / "credit_factor_rotation_ml_feature_importance.png"

BAD_REGIME_WEIGHTS = {"w_LV": 0.40, "w_P": 0.30, "w_I": 0.30}
GOOD_REGIME_WEIGHTS = {"w_LV": -0.40, "w_P": 0.30, "w_I": -0.30}
MAX_SIGNAL_FORMATION_LAG_MONTHS = 6


@dataclass
class StrategyArtifacts:
    crsp: pd.DataFrame
    fundamentals: pd.DataFrame
    linked_signals: pd.DataFrame
    factor_returns: pd.DataFrame
    factor_validation: pd.DataFrame
    macro_signal: pd.DataFrame
    strategy_panel: pd.DataFrame
    strategy_returns: pd.DataFrame


@dataclass
class FullStrategyArtifacts:
    core: StrategyArtifacts
    ff5: pd.DataFrame
    benchmarks: pd.DataFrame
    descriptive_stats: pd.DataFrame
    regression_results: pd.DataFrame
    subperiod_results: pd.DataFrame
    tercile_strategy: pd.DataFrame
    vix_strategy: pd.DataFrame
    ff_good_cash_strategy: pd.DataFrame
    ml_strategy: pd.DataFrame
    summary_table: pd.DataFrame
    alpha_table: pd.DataFrame
    robustness_summary: pd.DataFrame
    ml_summary: pd.DataFrame
    ml_feature_importance: pd.DataFrame


def _to_month_end(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce") + MonthEnd(0)


def _weighted_quintile_returns(panel: pd.DataFrame) -> pd.DataFrame:
    data = panel.copy()
    valid = data["ret"].notna()
    data["weight_valid"] = np.where(valid, data["weight"], 0.0)
    data["weighted_ret"] = np.where(valid, data["ret"] * data["weight"], 0.0)

    grouped = (
        data.groupby(["date", "quintile"], sort=True)[["weight_valid", "weighted_ret"]]
        .sum()
        .reset_index()
    )
    grouped["vwret"] = grouped["weighted_ret"] / grouped["weight_valid"]

    pivot = grouped.pivot(index="date", columns="quintile", values="vwret").sort_index()
    pivot.columns = [f"Q{int(col)}" for col in pivot.columns]
    return pivot


def _assign_quintiles(universe: pd.DataFrame, signal_col: str, date_col: str) -> pd.DataFrame:
    breakpoints = (
        universe.loc[universe["exchcd"] == 1]
        .groupby(date_col)[signal_col]
        .quantile([0.2, 0.4, 0.6, 0.8])
        .unstack()
        .rename(columns={0.2: "q20", 0.4: "q40", 0.6: "q60", 0.8: "q80"})
    )

    ranked = universe.merge(
        breakpoints,
        left_on=date_col,
        right_index=True,
        how="inner",
    )

    ranked["quintile"] = np.select(
        [
            ranked[signal_col] <= ranked["q20"],
            ranked[signal_col] <= ranked["q40"],
            ranked[signal_col] <= ranked["q60"],
            ranked[signal_col] <= ranked["q80"],
            ranked[signal_col] > ranked["q80"],
        ],
        [1, 2, 3, 4, 5],
        default=np.nan,
    )

    ranked = ranked.dropna(subset=["quintile"]).copy()
    ranked["quintile"] = ranked["quintile"].astype("int8")
    return ranked


def load_crsp(path: Path | None = None) -> pd.DataFrame:
    path = path or DATA_DIR / "crsp_clean_filtered(in).csv"
    require_data(path)
    crsp = pd.read_csv(
        path,
        usecols=["permno", "date", "shrcd", "exchcd", "siccd", "ret", "me"],
        dtype={
            "permno": "int32",
            "shrcd": "string",
            "exchcd": "string",
            "siccd": "string",
            "ret": "string",
            "me": "string",
        },
        low_memory=False,
    )

    crsp["date"] = _to_month_end(crsp["date"])
    for col in ["shrcd", "exchcd", "siccd", "ret", "me"]:
        crsp[col] = pd.to_numeric(crsp[col], errors="coerce")

    crsp = crsp.loc[
        crsp["date"].between(SAMPLE_START, pd.Timestamp("2024-12-31"))
        & crsp["shrcd"].isin([10, 11])
        & crsp["exchcd"].isin([1, 3])
    ].copy()

    crsp = crsp.sort_values(["permno", "date"]).reset_index(drop=True)

    valid_ret = crsp["ret"].notna().astype("int8")
    crsp["ret_history_24"] = (
        valid_ret.groupby(crsp["permno"])
        .rolling(24, min_periods=24)
        .sum()
        .reset_index(level=0, drop=True)
        .eq(24)
    )
    crsp["lowvol_signal"] = (
        crsp.groupby("permno")["ret"]
        .rolling(12, min_periods=12)
        .std(ddof=1)
        .reset_index(level=0, drop=True)
    )

    return crsp


def load_fundamentals(path: Path | None = None) -> pd.DataFrame:
    path = path or DATA_DIR / "QMJ_data.csv"
    require_data(path)
    funda = pd.read_csv(
        path,
        usecols=[
            "gvkey",
            "datadate",
            "fyear",
            "sale",
            "cogs",
            "at",
            "indfmt",
            "datafmt",
            "consol",
        ],
        dtype={"gvkey": "string"},
        low_memory=False,
    )

    funda["datadate"] = pd.to_datetime(funda["datadate"], errors="coerce")
    for col in ["sale", "cogs", "at"]:
        funda[col] = pd.to_numeric(funda[col], errors="coerce")

    funda = funda.loc[
        (funda["indfmt"] == "INDL")
        & (funda["datafmt"] == "STD")
        & (funda["consol"] == "C")
        & funda["datadate"].notna()
    ].copy()

    funda = funda.sort_values(["gvkey", "datadate"]).reset_index(drop=True)
    funda["gpoa"] = (funda["sale"] - funda["cogs"]) / funda["at"]
    funda["at_lag"] = funda.groupby("gvkey")["at"].shift(1)
    funda["inv"] = (funda["at"] - funda["at_lag"]) / funda["at_lag"]
    return funda[["gvkey", "datadate", "fyear", "gpoa", "inv"]]


def load_ccm(path: Path | None = None) -> pd.DataFrame:
    path = path or DATA_DIR / "crsp_a_ccm.csv"
    require_data(path)
    ccm = pd.read_csv(
        path,
        usecols=["gvkey", "LINKPRIM", "LINKTYPE", "LPERMNO", "LINKDT", "LINKENDDT"],
        dtype={"gvkey": "string", "LINKPRIM": "string", "LINKTYPE": "string", "LPERMNO": "Int64"},
        low_memory=False,
    )

    ccm = ccm.loc[
        ccm["LINKTYPE"].isin(["LU", "LC"]) & ccm["LINKPRIM"].isin(["P", "C"])
    ].copy()

    ccm["LINKDT"] = pd.to_datetime(ccm["LINKDT"], errors="coerce", format="mixed", dayfirst=True)
    ccm["LINKENDDT"] = pd.to_datetime(
        ccm["LINKENDDT"].replace({"E": None}),
        errors="coerce",
        format="mixed",
        dayfirst=True,
    ).fillna(pd.Timestamp("2100-12-31"))

    ccm = ccm.rename(
        columns={
            "LINKDT": "linkdt",
            "LINKENDDT": "linkenddt",
            "LPERMNO": "permno",
        }
    )
    ccm["permno"] = ccm["permno"].astype("int32")
    return ccm[["gvkey", "permno", "linkdt", "linkenddt"]]


def link_fundamentals_to_crsp(
    funda: pd.DataFrame,
    ccm: pd.DataFrame,
    crsp: pd.DataFrame,
    max_months_after_availability: int = MAX_SIGNAL_FORMATION_LAG_MONTHS,
) -> pd.DataFrame:
    linked = funda.merge(ccm, on="gvkey", how="inner")
    linked = linked.loc[
        linked["datadate"].between(linked["linkdt"], linked["linkenddt"])
    ].copy()

    crsp_months = (
        crsp[["permno", "date"]]
        .drop_duplicates()
        .sort_values(["permno", "date"])
        .groupby("permno")["date"]
        .apply(lambda s: s.to_numpy(dtype="datetime64[ns]"))
        .to_dict()
    )

    matched_groups = []
    for permno, group in linked.sort_values(["permno", "datadate"]).groupby("permno", sort=False):
        group = group.copy()
        dates = crsp_months.get(int(permno))
        if dates is None or len(dates) == 0:
            group["formation_date"] = pd.NaT
        else:
            availability = group["datadate"].to_numpy(dtype="datetime64[ns]")
            match_idx = np.searchsorted(dates, availability, side="left")
            matched = np.full(len(group), np.datetime64("NaT"), dtype="datetime64[ns]")
            valid = match_idx < len(dates)
            matched[valid] = dates[match_idx[valid]]
            group["formation_date"] = pd.to_datetime(matched)
        matched_groups.append(group)

    linked = pd.concat(matched_groups, ignore_index=True)

    max_formation_date = linked["datadate"] + MonthEnd(max_months_after_availability)
    linked = linked.loc[
        linked["formation_date"].notna() & linked["formation_date"].le(max_formation_date)
    ].copy()

    linked = linked.sort_values(["permno", "formation_date", "datadate"])
    linked = linked.drop_duplicates(["permno", "formation_date"], keep="last")
    return linked[["permno", "formation_date", "gpoa", "inv"]]


def build_annual_factor_returns(
    crsp: pd.DataFrame,
    linked_signals: pd.DataFrame,
    signal_col: str,
    factor_name: str,
    long_high_signal: bool,
    exclude_financials: bool,
) -> pd.Series:
    signal_updates = linked_signals[["permno", "formation_date", signal_col]].dropna().copy()
    signal_updates = signal_updates.sort_values(["permno", "formation_date"])
    signal_updates = signal_updates.drop_duplicates(["permno", "formation_date"], keep="last")
    signal_updates = signal_updates.rename(columns={"formation_date": "signal_date"})

    monthly_universe = crsp.loc[
        :,
        ["permno", "date", "exchcd", "siccd", "me", "ret_history_24"],
    ].copy()
    monthly_universe = monthly_universe.merge(
        signal_updates,
        left_on=["permno", "date"],
        right_on=["permno", "signal_date"],
        how="left",
    )
    monthly_universe[signal_col] = monthly_universe.groupby("permno")[signal_col].ffill()

    universe = monthly_universe.loc[
        monthly_universe["ret_history_24"] & monthly_universe["me"].gt(0) & monthly_universe[signal_col].notna()
    ].copy()

    if exclude_financials:
        universe = universe.loc[~universe["siccd"].between(6000, 6999, inclusive="both")].copy()

    universe = _assign_quintiles(universe, signal_col=signal_col, date_col="date")
    universe["weight"] = universe["me"] / universe.groupby(["date", "quintile"])["me"].transform("sum")
    universe = universe.rename(columns={"date": "formation_date"})

    monthly = crsp.loc[:, ["permno", "date", "ret"]].copy()
    monthly["formation_date"] = monthly["date"] - MonthEnd(1)

    panel = monthly.merge(
        universe[["permno", "formation_date", "quintile", "weight"]],
        on=["permno", "formation_date"],
        how="inner",
    )

    quintile_returns = _weighted_quintile_returns(panel)
    long_leg = "Q5" if long_high_signal else "Q1"
    short_leg = "Q1" if long_high_signal else "Q5"
    factor = (quintile_returns[long_leg] - quintile_returns[short_leg]).rename(factor_name)
    return factor


def build_lowvol_factor_returns(crsp: pd.DataFrame) -> pd.Series:
    universe = crsp.loc[
        crsp["ret_history_24"] & crsp["lowvol_signal"].notna() & crsp["me"].gt(0),
        ["permno", "date", "exchcd", "me", "lowvol_signal"],
    ].copy()
    universe = universe.rename(columns={"date": "formation_date"})
    universe = _assign_quintiles(universe, signal_col="lowvol_signal", date_col="formation_date")
    universe["weight"] = universe["me"] / universe.groupby(["formation_date", "quintile"])["me"].transform("sum")

    monthly = crsp.loc[:, ["permno", "date", "ret"]].copy()
    monthly["formation_date"] = monthly["date"] - MonthEnd(1)

    panel = monthly.merge(
        universe[["permno", "formation_date", "quintile", "weight"]],
        on=["permno", "formation_date"],
        how="inner",
    )

    quintile_returns = _weighted_quintile_returns(panel)
    return (quintile_returns["Q1"] - quintile_returns["Q5"]).rename("R_lowvol")


def load_reuters_monthly(path: Path, value_name: str) -> pd.DataFrame:
    data = pd.read_csv(path, skiprows=[1])
    data = data.rename(columns={data.columns[0]: "raw_date", data.columns[1]: value_name})
    data["date"] = pd.to_datetime(data["raw_date"], dayfirst=True, errors="coerce") + MonthEnd(0)
    data[value_name] = pd.to_numeric(data[value_name], errors="coerce")
    data = data[["date", value_name]].dropna().drop_duplicates("date").sort_values("date")
    return data.reset_index(drop=True)


def load_fred_monthly(path: Path, value_name: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    data = data.rename(columns={data.columns[0]: "raw_date", data.columns[1]: value_name})
    data["date"] = pd.to_datetime(data["raw_date"], errors="coerce") + MonthEnd(0)
    data[value_name] = pd.to_numeric(data[value_name], errors="coerce")
    data = data[["date", value_name]].dropna().drop_duplicates("date").sort_values("date")
    return data.reset_index(drop=True)


def load_macro_feature_frame(include_vix: bool = True) -> pd.DataFrame:
    y10 = load_reuters_monthly(DATA_DIR / "10Y monthly US(Table Data).csv", "yield_10y")
    y3m = load_reuters_monthly(DATA_DIR / "US 3M monthly(Table Data).csv", "yield_3m")
    aaa = load_fred_monthly(DATA_DIR / "DAAA.csv", "yield_aaa")
    baa = load_fred_monthly(DATA_DIR / "DBAA.csv", "yield_baa")
    macro = y10.merge(y3m, on="date", how="inner").merge(aaa, on="date", how="inner").merge(baa, on="date", how="inner")

    if include_vix:
        vix = load_fred_monthly(DATA_DIR / "VIXCLS (1).csv", "vix")
        macro = macro.merge(vix, on="date", how="inner")

    macro = macro.sort_values("date").reset_index(drop=True)
    macro = macro.loc[macro["date"].between(SAMPLE_START, SAMPLE_END)].copy()
    macro["TERM"] = macro["yield_10y"] - macro["yield_3m"]
    macro["DEF"] = macro["yield_baa"] - macro["yield_aaa"]
    macro["dTERM"] = macro["TERM"].diff()
    macro["dDEF"] = macro["DEF"].diff()

    keep_cols = ["date", "TERM", "DEF", "dTERM", "dDEF"]
    if include_vix:
        keep_cols.append("vix")
    macro = macro[keep_cols].dropna().reset_index(drop=True)
    return macro


def build_macro_signal(
    threshold_mode: str = "median",
    include_vix: bool = False,
) -> pd.DataFrame:
    macro = load_macro_feature_frame(include_vix=include_vix)
    keep_cols = ["date", "TERM", "DEF"]
    if include_vix:
        keep_cols.append("vix")
    macro = macro[keep_cols].copy()

    macro["z_TERM"] = (macro["TERM"] - macro["TERM"].expanding().mean()) / macro["TERM"].expanding().std(ddof=1)
    macro["z_DEF"] = (macro["DEF"] - macro["DEF"].expanding().mean()) / macro["DEF"].expanding().std(ddof=1)
    if include_vix:
        macro["z_VIX"] = (macro["vix"] - macro["vix"].expanding().mean()) / macro["vix"].expanding().std(ddof=1)
        macro["M_t"] = macro["z_TERM"] - macro["z_DEF"] - macro["z_VIX"]
    else:
        macro["M_t"] = macro["z_TERM"] - macro["z_DEF"]
    macro["expanding_median"] = macro["M_t"].expanding().median()

    macro["signal_ready"] = np.arange(len(macro)) >= 24
    if threshold_mode == "median":
        macro["regime_label"] = np.where(
            macro["signal_ready"],
            np.where(macro["M_t"] > macro["expanding_median"], "good", "bad"),
            pd.NA,
        )
    elif threshold_mode == "tercile":
        q33 = macro["M_t"].expanding().quantile(1 / 3)
        q67 = macro["M_t"].expanding().quantile(2 / 3)
        macro["regime_label"] = np.where(
            macro["signal_ready"],
            np.where(
                macro["M_t"] > q67,
                "good",
                np.where(macro["M_t"] < q33, "bad", "neutral"),
            ),
            pd.NA,
        )
        macro["expanding_q33"] = q33
        macro["expanding_q67"] = q67
    else:
        raise ValueError("threshold_mode must be 'median' or 'tercile'.")
    macro["holding_date"] = macro["date"] + MonthEnd(1)

    return macro


def load_ff5(path: Path | None = None) -> pd.DataFrame:
    path = path or DATA_DIR / "F-F_Research_Data_5_Factors_2x3_CSV" / "F-F_Research_Data_5_Factors_2x3.csv"
    text = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    header_idx = next(i for i, line in enumerate(text) if line.startswith(",Mkt-RF"))
    data_lines = [text[header_idx]]
    data_lines.extend(line for line in text[header_idx + 1 :] if re.match(r"^\d{6},", line))

    ff5 = pd.read_csv(StringIO("\n".join(data_lines)))
    ff5 = ff5.rename(columns={ff5.columns[0]: "yyyymm"})
    ff5["date"] = pd.to_datetime(ff5["yyyymm"].astype(str), format="%Y%m") + MonthEnd(0)

    factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
    for col in factor_cols:
        ff5[col] = pd.to_numeric(ff5[col], errors="coerce") / 100.0

    ff5 = ff5[["date"] + factor_cols].dropna().sort_values("date")
    ff5 = ff5.loc[ff5["date"].between(SAMPLE_START, SAMPLE_END)].reset_index(drop=True)
    return ff5


def load_qmj_returns(path: Path | None = None) -> pd.DataFrame:
    path = path or DATA_DIR / "qmj_returns.csv"
    qmj = pd.read_csv(path)
    qmj["date"] = pd.to_datetime(qmj["date"], errors="coerce") + MonthEnd(0)
    qmj["ret"] = pd.to_numeric(qmj["ret"], errors="coerce")
    qmj = qmj.dropna().sort_values("date")
    qmj = qmj.loc[qmj["date"].between(SAMPLE_START, SAMPLE_END)].reset_index(drop=True)
    return qmj


def build_factor_validation(factor_returns: pd.DataFrame, ff5: pd.DataFrame) -> pd.DataFrame:
    merged = factor_returns.merge(ff5[["date", "RMW", "CMA"]], on="date", how="inner")

    diagnostics = []
    for series_name, benchmark_name in [("R_prof", "RMW"), ("R_inv", "CMA")]:
        sample = merged[[series_name, benchmark_name]].dropna()
        diagnostics.append(
            {
                "factor": series_name,
                "benchmark": benchmark_name,
                "correlation": sample[series_name].corr(sample[benchmark_name]),
                "mean": sample[series_name].mean(),
                "std": sample[series_name].std(ddof=1),
                "sharpe": sample[series_name].mean() / sample[series_name].std(ddof=1),
                "n_months": int(sample.shape[0]),
            }
        )

    return pd.DataFrame(diagnostics)


def build_benchmarks(
    strategy_panel: pd.DataFrame,
    ff5: pd.DataFrame,
    qmj_returns: pd.DataFrame,
) -> pd.DataFrame:
    benchmarks = strategy_panel[["date", "R_strategy", "R_lowvol", "R_prof", "R_inv"]].copy()
    benchmarks["Benchmark_A"] = (benchmarks["R_lowvol"] + benchmarks["R_prof"] + benchmarks["R_inv"]) / 3.0
    benchmarks = benchmarks.merge(ff5[["date", "RMW", "CMA"]], on="date", how="left")
    benchmarks["Benchmark_B"] = 0.5 * benchmarks["RMW"] + 0.5 * benchmarks["CMA"]
    benchmarks = benchmarks.merge(
        qmj_returns.rename(columns={"ret": "Benchmark_D"}),
        on="date",
        how="left",
    )
    return benchmarks[["date", "R_strategy", "Benchmark_A", "Benchmark_B", "Benchmark_D"]]


def compute_max_drawdown(returns: pd.Series) -> float:
    wealth = (1.0 + returns).cumprod()
    running_peak = wealth.cummax()
    drawdown = wealth / running_peak - 1.0
    return float(drawdown.min())


def compute_turnover_from_weights(weights: pd.DataFrame) -> pd.Series:
    changes = weights[["w_LV", "w_P", "w_I"]].diff().abs().sum(axis=1).fillna(0.0)
    return pd.Series(changes.to_numpy(), index=weights["date"], name="turnover")


def descriptive_statistics(
    returns_df: pd.DataFrame,
    turnover: pd.Series | None = None,
) -> pd.DataFrame:
    rows = []
    monthly_turnover = turnover.reindex(returns_df["date"]).fillna(0.0) if turnover is not None else None

    for column in [col for col in returns_df.columns if col != "date"]:
        sample = returns_df[["date", column]].dropna().copy()
        ret = sample[column]
        if ret.empty:
            rows.append(
                {
                    "series": column,
                    "n_months": 0,
                    "mean_annual": np.nan,
                    "vol_annual": np.nan,
                    "sharpe_annual": np.nan,
                    "t_stat": np.nan,
                    "p_value": np.nan,
                    "max_drawdown": np.nan,
                    "avg_monthly_turnover": np.nan,
                }
            )
            continue
        mean_m = ret.mean()
        vol_m = ret.std(ddof=1)
        has_vol = pd.notna(vol_m) and vol_m > 0
        t_stat = mean_m / (vol_m / np.sqrt(len(ret))) if has_vol else np.nan
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(ret) - 1)) if len(ret) > 1 and pd.notna(t_stat) else np.nan
        row = {
            "series": column,
            "n_months": int(len(ret)),
            "mean_annual": mean_m * 12,
            "vol_annual": vol_m * np.sqrt(12),
            "sharpe_annual": (mean_m / vol_m) * np.sqrt(12) if has_vol else np.nan,
            "t_stat": t_stat,
            "p_value": p_value,
            "max_drawdown": compute_max_drawdown(ret),
            "turnover_mean_monthly": 0.0,
        }
        if monthly_turnover is not None and column in {"R_strategy", "Strategy"}:
            row["turnover_mean_monthly"] = float(monthly_turnover.loc[sample["date"]].mean())
        rows.append(row)

    return pd.DataFrame(rows)


def _display_name_map() -> dict[str, str]:
    return {
        "Strategy": "Main Strategy",
        "Benchmark_A": "Static Equal-Weight Basket",
        "Benchmark_B": "Ken French RMW+CMA",
        "Benchmark_D": "QMJ Benchmark",
        "Main_Strategy": "Main Strategy",
        "Tercile_Robustness": "Tercile Regime Rule",
        "VIX_Robustness": "VIX-Augmented Signal",
        "FF_Good_Cash_Robustness": "FF RMW+CMA / Cash",
        "ML_Robustness": "ML Classifier",
    }


def prepare_summary_table(
    descriptive_stats: pd.DataFrame,
    regression_results: pd.DataFrame,
) -> pd.DataFrame:
    ff5_alpha = (
        regression_results.loc[regression_results["model"] == "FF5", ["series", "alpha", "alpha_t", "alpha_p", "r_squared"]]
        .rename(
            columns={
                "alpha": "ff5_alpha",
                "alpha_t": "ff5_alpha_t",
                "alpha_p": "ff5_alpha_p",
                "r_squared": "ff5_r_squared",
            }
        )
    )

    summary = descriptive_stats.merge(ff5_alpha, on="series", how="left")
    summary["series"] = summary["series"].replace(_display_name_map())
    summary = summary[
        [
            "series",
            "n_months",
            "mean_annual",
            "vol_annual",
            "sharpe_annual",
            "t_stat",
            "p_value",
            "max_drawdown",
            "turnover_mean_monthly",
            "ff5_alpha",
            "ff5_alpha_t",
            "ff5_alpha_p",
            "ff5_r_squared",
        ]
    ].sort_values("series")

    numeric_cols = [col for col in summary.columns if col != "series"]
    summary[numeric_cols] = summary[numeric_cols].round(4)
    return summary.reset_index(drop=True)


def prepare_alpha_table(regression_results: pd.DataFrame) -> pd.DataFrame:
    alpha_table = regression_results.pivot(
        index="series",
        columns="model",
        values=["alpha", "alpha_t", "alpha_p", "r_squared"],
    )
    alpha_table.columns = [f"{model.lower()}_{metric}" for metric, model in alpha_table.columns]
    alpha_table = alpha_table.reset_index()
    alpha_table["series"] = alpha_table["series"].replace(_display_name_map())

    numeric_cols = [col for col in alpha_table.columns if col != "series"]
    alpha_table[numeric_cols] = alpha_table[numeric_cols].round(4)
    return alpha_table.sort_values("series").reset_index(drop=True)


def run_factor_regressions(
    returns_df: pd.DataFrame,
    ff5: pd.DataFrame,
    nw_lags: int = 6,
) -> pd.DataFrame:
    merged = returns_df.merge(ff5, on="date", how="inner")
    models = {
        "CAPM": ["Mkt-RF"],
        "FF3": ["Mkt-RF", "SMB", "HML"],
        "FF5": ["Mkt-RF", "SMB", "HML", "RMW", "CMA"],
    }
    rows = []

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
            result = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags})

            row = {
                "series": series,
                "model": model_name,
                "n_obs": int(result.nobs),
                "alpha": result.params.get("const", np.nan),
                "alpha_t": result.tvalues.get("const", np.nan),
                "alpha_p": result.pvalues.get("const", np.nan),
                "r_squared": result.rsquared,
            }
            for reg in ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]:
                row[f"beta_{reg}"] = result.params.get(reg, np.nan)
                row[f"t_{reg}"] = result.tvalues.get(reg, np.nan)
            rows.append(row)

    return pd.DataFrame(rows)


def run_subperiod_analysis(
    returns_df: pd.DataFrame,
    ff5: pd.DataFrame,
    split_date: str = "2007-12-31",
) -> pd.DataFrame:
    split_ts = pd.Timestamp(split_date)
    periods = {
        "pre_2008": returns_df["date"] <= split_ts,
        "post_2007": returns_df["date"] > split_ts,
    }
    outputs = []

    for label, mask in periods.items():
        sub_returns = returns_df.loc[mask].copy()
        sub_ff5 = ff5.loc[ff5["date"].isin(sub_returns["date"])].copy()
        stats_df = descriptive_statistics(sub_returns)
        regs_df = run_factor_regressions(sub_returns, sub_ff5)
        stats_df["period"] = label
        regs_df["period"] = label
        outputs.append(
            stats_df.merge(
                regs_df[["series", "model", "alpha", "alpha_t", "alpha_p", "r_squared", "period"]],
                on=["series", "period"],
                how="left",
            )
        )

    return pd.concat(outputs, ignore_index=True)


def build_robustness_summary(
    main_strategy: pd.DataFrame,
    tercile_strategy: pd.DataFrame,
    vix_strategy: pd.DataFrame,
    ff_good_cash_strategy: pd.DataFrame,
    ml_strategy: pd.DataFrame,
    ff5: pd.DataFrame,
) -> pd.DataFrame:
    combined = (
        main_strategy.rename(columns={"ret": "Main_Strategy"})
        .merge(tercile_strategy[["date", "ret"]].rename(columns={"ret": "Tercile_Robustness"}), on="date", how="outer")
        .merge(vix_strategy[["date", "ret"]].rename(columns={"ret": "VIX_Robustness"}), on="date", how="outer")
        .merge(
            ff_good_cash_strategy[["date", "ret"]].rename(columns={"ret": "FF_Good_Cash_Robustness"}),
            on="date",
            how="outer",
        )
        .merge(ml_strategy[["date", "ret"]].rename(columns={"ret": "ML_Robustness"}), on="date", how="outer")
        .sort_values("date")
    )

    stats_df = descriptive_statistics(combined)
    regs_df = run_factor_regressions(combined, ff5)
    summary = prepare_summary_table(stats_df, regs_df)
    return summary


def build_strategy_panel(
    factor_returns: pd.DataFrame,
    macro_signal: pd.DataFrame,
) -> pd.DataFrame:
    weights = macro_signal.loc[
        macro_signal["signal_ready"],
        ["holding_date", "regime_label", "M_t"],
    ].rename(columns={"holding_date": "date"})

    panel = factor_returns.merge(weights, on="date", how="inner")
    panel = panel.dropna(subset=["R_lowvol", "R_prof", "R_inv", "regime_label"]).copy()

    panel["w_LV"] = np.select(
        [
            panel["regime_label"] == "good",
            panel["regime_label"] == "neutral",
        ],
        [
            GOOD_REGIME_WEIGHTS["w_LV"],
            1.0 / 3.0,
        ],
        default=BAD_REGIME_WEIGHTS["w_LV"],
    )
    panel["w_P"] = np.select(
        [
            panel["regime_label"] == "good",
            panel["regime_label"] == "neutral",
        ],
        [
            GOOD_REGIME_WEIGHTS["w_P"],
            1.0 / 3.0,
        ],
        default=BAD_REGIME_WEIGHTS["w_P"],
    )
    panel["w_I"] = np.select(
        [
            panel["regime_label"] == "good",
            panel["regime_label"] == "neutral",
        ],
        [
            GOOD_REGIME_WEIGHTS["w_I"],
            1.0 / 3.0,
        ],
        default=BAD_REGIME_WEIGHTS["w_I"],
    )
    panel["R_strategy"] = (
        panel["w_LV"] * panel["R_lowvol"]
        + panel["w_P"] * panel["R_prof"]
        + panel["w_I"] * panel["R_inv"]
    )

    panel = panel.loc[panel["date"].between(SAMPLE_START, SAMPLE_END)].sort_values("date").reset_index(drop=True)
    return panel


def validate_strategy_returns(strategy_returns: pd.DataFrame) -> None:
    if strategy_returns["ret"].isna().any():
        raise ValueError("Strategy return series contains NaN values.")
    if np.isinf(strategy_returns["ret"]).any():
        raise ValueError("Strategy return series contains infinite values.")

    monthly_index = pd.period_range(
        strategy_returns["date"].min().to_period("M"),
        strategy_returns["date"].max().to_period("M"),
        freq="M",
    )
    actual_index = strategy_returns["date"].dt.to_period("M")
    if len(monthly_index) != actual_index.nunique():
        raise ValueError("Strategy return series has missing months.")
    if (strategy_returns["ret"].abs() > 0.5).any():
        raise ValueError("Strategy return series contains values outside the expected range [-0.5, 0.5].")


def plot_strategy_returns_by_regime(
    strategy_panel: pd.DataFrame,
    output_path: Path | None = None,
) -> Path:
    output_path = output_path or OUTPUT_PLOT

    plot_data = strategy_panel[["date", "regime_label", "R_strategy"]].copy()
    colors = np.where(plot_data["regime_label"] == "good", "#2e8b57", "#c0392b")

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(plot_data["date"], plot_data["R_strategy"], color=colors, width=25, linewidth=0)
    ax.axhline(0, color="black", linewidth=1, alpha=0.7)
    ax.set_title("Monthly Strategy Returns by Credit Regime")
    ax.set_xlabel("Date")
    ax.set_ylabel("Strategy Return")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#2e8b57", label="Good regime"),
        plt.Rectangle((0, 0), 1, 1, color="#c0392b", label="Bad regime"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_cumulative_returns(
    returns_df: pd.DataFrame,
    output_path: Path,
    title: str,
    ylabel: str = "Growth of $1",
) -> Path:
    fig, ax = plt.subplots(figsize=(14, 7))

    for column in [col for col in returns_df.columns if col != "date"]:
        sample = returns_df[["date", column]].dropna().copy()
        if sample.empty:
            continue
        sample["cum_growth"] = (1.0 + sample[column]).cumprod()
        label = _display_name_map().get(column, column.replace("_", " "))
        ax.plot(sample["date"], sample["cum_growth"], linewidth=2, label=label)

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, ncol=2)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def summarize_ml_robustness(
    ml_strategy: pd.DataFrame,
    main_strategy_panel: pd.DataFrame,
    ff5: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    aligned = ml_strategy.merge(
        main_strategy_panel[["date", "regime_label"]].rename(columns={"regime_label": "main_regime"}),
        on="date",
        how="inner",
    )
    agreement_rate = float((aligned["regime_label"] == aligned["main_regime"]).mean())

    ml_returns = ml_strategy[["date", "ret"]].rename(columns={"ret": "ML_Robustness"})
    ml_stats = descriptive_statistics(ml_returns)
    ml_regs = run_factor_regressions(ml_returns, ff5)
    ff5_row = ml_regs.loc[ml_regs["model"] == "FF5"].iloc[0]

    summary = pd.DataFrame(
        [
            {
                "metric": "agreement_rate_with_main_classifier",
                "value": agreement_rate,
            },
            {
                "metric": "mean_prob_good",
                "value": float(ml_strategy["prob_good"].mean()),
            },
            {
                "metric": "share_good_predictions",
                "value": float((ml_strategy["regime_label"] == "good").mean()),
            },
            {
                "metric": "share_bad_predictions",
                "value": float((ml_strategy["regime_label"] == "bad").mean()),
            },
            {
                "metric": "annual_mean_return",
                "value": float(ml_stats["mean_annual"].iloc[0]),
            },
            {
                "metric": "annual_volatility",
                "value": float(ml_stats["vol_annual"].iloc[0]),
            },
            {
                "metric": "annual_sharpe",
                "value": float(ml_stats["sharpe_annual"].iloc[0]),
            },
            {
                "metric": "ff5_alpha",
                "value": float(ff5_row["alpha"]),
            },
            {
                "metric": "ff5_alpha_t",
                "value": float(ff5_row["alpha_t"]),
            },
            {
                "metric": "ff5_alpha_p",
                "value": float(ff5_row["alpha_p"]),
            },
        ]
    )
    summary["value"] = summary["value"].round(4)

    feature_importance = (
        ml_strategy[
            [
                "feature_importance_TERM",
                "feature_importance_DEF",
                "feature_importance_dTERM",
                "feature_importance_dDEF",
                "feature_importance_vix",
            ]
        ]
        .mean()
        .rename(
            {
                "feature_importance_TERM": "TERM",
                "feature_importance_DEF": "DEF",
                "feature_importance_dTERM": "Delta TERM",
                "feature_importance_dDEF": "Delta DEF",
                "feature_importance_vix": "VIX",
            }
        )
        .reset_index()
        .rename(columns={"index": "feature", 0: "mean_importance"})
        .sort_values("mean_importance", ascending=False)
        .reset_index(drop=True)
    )
    feature_importance["mean_importance"] = feature_importance["mean_importance"].round(4)

    return summary, feature_importance


def plot_ml_feature_importance(
    feature_importance: pd.DataFrame,
    output_path: Path | None = None,
) -> Path:
    output_path = output_path or OUTPUT_ML_IMPORTANCE_PLOT
    data = feature_importance.sort_values("mean_importance", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(data["feature"], data["mean_importance"], color="#1f77b4")
    ax.set_title("ML Robustness Feature Importances")
    ax.set_xlabel("Average Feature Importance")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def export_report_outputs(artifacts: FullStrategyArtifacts) -> dict[str, Path]:
    paths = {
        "benchmarks": OUTPUT_BENCHMARKS_CSV,
        "summary_table": OUTPUT_SUMMARY_TABLE_CSV,
        "alpha_table": OUTPUT_ALPHA_TABLE_CSV,
        "robustness_summary": OUTPUT_ROBUSTNESS_TABLE_CSV,
        "ml_summary": OUTPUT_ML_SUMMARY_CSV,
        "ml_feature_importance": OUTPUT_ML_FEATURE_IMPORTANCE_CSV,
    }

    artifacts.benchmarks.to_csv(paths["benchmarks"], index=False)
    artifacts.summary_table.to_csv(paths["summary_table"], index=False)
    artifacts.alpha_table.to_csv(paths["alpha_table"], index=False)
    artifacts.robustness_summary.to_csv(paths["robustness_summary"], index=False)
    artifacts.ml_summary.to_csv(paths["ml_summary"], index=False)
    artifacts.ml_feature_importance.to_csv(paths["ml_feature_importance"], index=False)

    plot_cumulative_returns(
        artifacts.benchmarks.rename(columns={"R_strategy": "Strategy"}),
        OUTPUT_CUMULATIVE_PLOT,
        title="Cumulative Returns: Strategy vs Benchmarks",
    )
    ff_good_cash_plot_frame = (
        artifacts.core.strategy_returns.assign(date=pd.to_datetime(artifacts.core.strategy_returns["date"]))
        .rename(columns={"ret": "Main_Strategy"})
        .merge(
            artifacts.ff_good_cash_strategy[["date", "ret"]].rename(columns={"ret": "FF_Good_Cash_Robustness"}),
            on="date",
            how="inner",
        )
        .sort_values("date")
    )
    plot_cumulative_returns(
        ff_good_cash_plot_frame,
        OUTPUT_FF_GOOD_CASH_PLOT,
        title="Cumulative Returns: Main Strategy vs FF RMW+CMA / Cash",
    )
    robustness_plot_frame = (
        artifacts.core.strategy_returns.assign(date=pd.to_datetime(artifacts.core.strategy_returns["date"]))
        .rename(columns={"ret": "Main_Strategy"})
        .merge(artifacts.tercile_strategy[["date", "ret"]].rename(columns={"ret": "Tercile_Robustness"}), on="date", how="outer")
        .merge(artifacts.vix_strategy[["date", "ret"]].rename(columns={"ret": "VIX_Robustness"}), on="date", how="outer")
        .merge(
            artifacts.ff_good_cash_strategy[["date", "ret"]].rename(columns={"ret": "FF_Good_Cash_Robustness"}),
            on="date",
            how="outer",
        )
        .merge(artifacts.ml_strategy[["date", "ret"]].rename(columns={"ret": "ML_Robustness"}), on="date", how="outer")
        .sort_values("date")
    )
    plot_cumulative_returns(
        robustness_plot_frame,
        OUTPUT_ROBUSTNESS_PLOT,
        title="Cumulative Returns: Main Strategy vs Robustness Variants",
    )
    plot_ml_feature_importance(artifacts.ml_feature_importance, OUTPUT_ML_IMPORTANCE_PLOT)

    paths["cumulative_plot"] = OUTPUT_CUMULATIVE_PLOT
    paths["ff_good_cash_plot"] = OUTPUT_FF_GOOD_CASH_PLOT
    paths["robustness_plot"] = OUTPUT_ROBUSTNESS_PLOT
    paths["ml_importance_plot"] = OUTPUT_ML_IMPORTANCE_PLOT
    return paths


def run_tercile_strategy(factor_returns: pd.DataFrame) -> pd.DataFrame:
    macro_tercile = build_macro_signal(threshold_mode="tercile", include_vix=False)
    tercile_panel = build_strategy_panel(factor_returns=factor_returns, macro_signal=macro_tercile)
    return tercile_panel[["date", "regime_label", "R_strategy"]].rename(columns={"R_strategy": "ret"})


def run_vix_strategy(factor_returns: pd.DataFrame) -> pd.DataFrame:
    macro_vix = build_macro_signal(threshold_mode="median", include_vix=True)
    vix_panel = build_strategy_panel(factor_returns=factor_returns, macro_signal=macro_vix)
    return vix_panel[["date", "regime_label", "R_strategy"]].rename(columns={"R_strategy": "ret"})


def run_ff_good_cash_strategy(
    ff5: pd.DataFrame,
    macro_signal: pd.DataFrame | None = None,
    sample_dates: pd.Series | pd.Index | None = None,
) -> pd.DataFrame:
    macro_signal = build_macro_signal() if macro_signal is None else macro_signal
    signal = macro_signal.loc[macro_signal["signal_ready"], ["holding_date", "regime_label", "M_t"]].rename(
        columns={"holding_date": "date"}
    )

    panel = ff5[["date", "RMW", "CMA", "RF"]].merge(signal, on="date", how="inner")
    panel = panel.dropna(subset=["RMW", "CMA", "RF", "regime_label"]).copy()
    panel["risky_leg"] = 0.5 * (panel["RMW"] + panel["CMA"])
    panel["ret"] = np.where(panel["regime_label"] == "good", panel["risky_leg"], panel["RF"])
    if sample_dates is not None:
        sample_index = pd.Index(pd.to_datetime(sample_dates)).dropna().unique()
        panel = panel.loc[panel["date"].isin(sample_index)].copy()
    panel = panel.loc[panel["date"].between(SAMPLE_START, SAMPLE_END)].sort_values("date").reset_index(drop=True)

    return panel[["date", "regime_label", "M_t", "RMW", "CMA", "RF", "risky_leg", "ret"]]


def run_ml_robustness(
    factor_returns: pd.DataFrame,
    min_train_months: int = 24,
) -> pd.DataFrame:
    try:
        from sklearn.ensemble import GradientBoostingClassifier
    except ImportError as exc:
        raise ImportError(
            "ML robustness requires scikit-learn, which is not installed in this environment."
        ) from exc

    macro_features = load_macro_feature_frame(include_vix=True)
    basket = factor_returns[["date", "R_lowvol", "R_prof", "R_inv"]].copy()
    basket["eq_basket_ret"] = (basket["R_lowvol"] + basket["R_prof"] + basket["R_inv"]) / 3.0
    basket["target_positive_next"] = (basket["eq_basket_ret"].shift(-1) > 0).astype("float")

    ml_data = macro_features.merge(
        basket[["date", "target_positive_next"]],
        on="date",
        how="inner",
    ).sort_values("date").reset_index(drop=True)

    feature_cols = ["TERM", "DEF", "dTERM", "dDEF", "vix"]
    results = []

    for i in range(len(ml_data)):
        current = ml_data.iloc[i]
        train = ml_data.iloc[:i].dropna(subset=["target_positive_next"]).copy()

        if len(train) < min_train_months or train["target_positive_next"].nunique() < 2:
            continue

        model = GradientBoostingClassifier(
            n_estimators=50,
            max_depth=2,
            learning_rate=0.1,
            min_samples_leaf=10,
            random_state=42,
        )
        model.fit(train[feature_cols], train["target_positive_next"].astype(int))
        prob_good = float(model.predict_proba(pd.DataFrame([current[feature_cols]], columns=feature_cols))[0, 1])
        predicted_regime = "good" if prob_good > 0.5 else "bad"

        results.append(
            {
                "date": current["date"] + MonthEnd(1),
                "feature_date": current["date"],
                "prob_good": prob_good,
                "regime_label": predicted_regime,
                "feature_importance_TERM": float(model.feature_importances_[0]),
                "feature_importance_DEF": float(model.feature_importances_[1]),
                "feature_importance_dTERM": float(model.feature_importances_[2]),
                "feature_importance_dDEF": float(model.feature_importances_[3]),
                "feature_importance_vix": float(model.feature_importances_[4]),
            }
        )

    if not results:
        raise ValueError("ML robustness could not generate any out-of-sample predictions.")

    ml_signal = pd.DataFrame(results)
    ml_panel = factor_returns.merge(
        ml_signal[
            [
                "date",
                "regime_label",
                "prob_good",
                "feature_importance_TERM",
                "feature_importance_DEF",
                "feature_importance_dTERM",
                "feature_importance_dDEF",
                "feature_importance_vix",
            ]
        ],
        on="date",
        how="inner",
    ).dropna(subset=["R_lowvol", "R_prof", "R_inv"])

    ml_panel["w_LV"] = np.where(ml_panel["regime_label"] == "good", GOOD_REGIME_WEIGHTS["w_LV"], BAD_REGIME_WEIGHTS["w_LV"])
    ml_panel["w_P"] = np.where(ml_panel["regime_label"] == "good", GOOD_REGIME_WEIGHTS["w_P"], BAD_REGIME_WEIGHTS["w_P"])
    ml_panel["w_I"] = np.where(ml_panel["regime_label"] == "good", GOOD_REGIME_WEIGHTS["w_I"], BAD_REGIME_WEIGHTS["w_I"])
    ml_panel["R_strategy"] = (
        ml_panel["w_LV"] * ml_panel["R_lowvol"]
        + ml_panel["w_P"] * ml_panel["R_prof"]
        + ml_panel["w_I"] * ml_panel["R_inv"]
    )

    return ml_panel[
        [
            "date",
            "regime_label",
            "prob_good",
            "feature_importance_TERM",
            "feature_importance_DEF",
            "feature_importance_dTERM",
            "feature_importance_dDEF",
            "feature_importance_vix",
            "w_LV",
            "w_P",
            "w_I",
            "R_strategy",
        ]
    ].rename(
        columns={"R_strategy": "ret"}
    )


def run_core_strategy(write_csv: bool = True) -> StrategyArtifacts:
    crsp = load_crsp()
    funda = load_fundamentals()
    ccm = load_ccm()
    linked = link_fundamentals_to_crsp(funda, ccm, crsp)

    profitability = build_annual_factor_returns(
        crsp=crsp,
        linked_signals=linked,
        signal_col="gpoa",
        factor_name="R_prof",
        long_high_signal=True,
        exclude_financials=True,
    )
    investment = build_annual_factor_returns(
        crsp=crsp,
        linked_signals=linked,
        signal_col="inv",
        factor_name="R_inv",
        long_high_signal=False,
        exclude_financials=True,
    )
    lowvol = build_lowvol_factor_returns(crsp)

    factor_returns = pd.concat([lowvol, profitability, investment], axis=1).reset_index()
    factor_returns = factor_returns.rename(columns={"index": "date"})
    factor_returns = factor_returns.loc[factor_returns["date"].between(SAMPLE_START, SAMPLE_END)].copy()

    macro_signal = build_macro_signal()
    strategy_panel = build_strategy_panel(factor_returns=factor_returns, macro_signal=macro_signal)

    strategy_returns = strategy_panel[["date", "R_strategy"]].rename(columns={"R_strategy": "ret"}).copy()
    strategy_returns["date"] = strategy_returns["date"].dt.strftime("%Y-%m-%d")
    validate_strategy_returns(
        strategy_returns.assign(date=pd.to_datetime(strategy_returns["date"], errors="coerce"))
    )

    ff5 = load_ff5()
    factor_validation = build_factor_validation(factor_returns=factor_returns, ff5=ff5)

    if write_csv:
        strategy_returns.to_csv(OUTPUT_CSV, index=False)

    return StrategyArtifacts(
        crsp=crsp,
        fundamentals=funda,
        linked_signals=linked,
        factor_returns=factor_returns,
        factor_validation=factor_validation,
        macro_signal=macro_signal,
        strategy_panel=strategy_panel,
        strategy_returns=strategy_returns,
    )


def run_full_strategy(write_csv: bool = True) -> FullStrategyArtifacts:
    core = run_core_strategy(write_csv=write_csv)
    ff5 = load_ff5()
    qmj = load_qmj_returns()
    benchmarks = build_benchmarks(core.strategy_panel, ff5, qmj)
    weights_turnover = compute_turnover_from_weights(core.strategy_panel[["date", "w_LV", "w_P", "w_I"]])
    descriptive = descriptive_statistics(benchmarks.rename(columns={"R_strategy": "Strategy"}), turnover=weights_turnover)
    regressions = run_factor_regressions(benchmarks.rename(columns={"R_strategy": "Strategy"}), ff5)
    summary_table = prepare_summary_table(descriptive, regressions)
    alpha_table = prepare_alpha_table(regressions)
    subperiod = run_subperiod_analysis(benchmarks.rename(columns={"R_strategy": "Strategy"}), ff5)
    tercile_strategy = run_tercile_strategy(core.factor_returns)
    vix_strategy = run_vix_strategy(core.factor_returns)
    ff_good_cash_strategy = run_ff_good_cash_strategy(
        ff5=ff5,
        macro_signal=core.macro_signal,
        sample_dates=core.strategy_panel["date"],
    )
    ml_strategy = run_ml_robustness(core.factor_returns)
    robustness_summary = build_robustness_summary(
        main_strategy=core.strategy_returns.assign(date=pd.to_datetime(core.strategy_returns["date"], errors="coerce")),
        tercile_strategy=tercile_strategy,
        vix_strategy=vix_strategy,
        ff_good_cash_strategy=ff_good_cash_strategy,
        ml_strategy=ml_strategy,
        ff5=ff5,
    )
    ml_summary, ml_feature_importance = summarize_ml_robustness(
        ml_strategy=ml_strategy,
        main_strategy_panel=core.strategy_panel,
        ff5=ff5,
    )

    return FullStrategyArtifacts(
        core=core,
        ff5=ff5,
        benchmarks=benchmarks,
        descriptive_stats=descriptive,
        regression_results=regressions,
        subperiod_results=subperiod,
        tercile_strategy=tercile_strategy,
        vix_strategy=vix_strategy,
        ff_good_cash_strategy=ff_good_cash_strategy,
        ml_strategy=ml_strategy,
        summary_table=summary_table,
        alpha_table=alpha_table,
        robustness_summary=robustness_summary,
        ml_summary=ml_summary,
        ml_feature_importance=ml_feature_importance,
    )


if __name__ == "__main__":
    artifacts = run_full_strategy(write_csv=True)
    plot_path = plot_strategy_returns_by_regime(artifacts.core.strategy_panel)
    report_paths = export_report_outputs(artifacts)
    print(
        f"Built {len(artifacts.core.strategy_returns):,} monthly strategy returns "
        f"from {artifacts.core.strategy_returns['date'].iloc[0]} "
        f"to {artifacts.core.strategy_returns['date'].iloc[-1]}."
    )
    print(f"Saved regime chart to {plot_path}.")
    print("Saved report outputs:")
    for name, path in report_paths.items():
        print(f"  {name}: {path}")
    print("\nDescriptive statistics:")
    print(artifacts.descriptive_stats.to_string(index=False))
