from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd

from credit_factor_rotation_core import (
    MAX_SIGNAL_FORMATION_LAG_MONTHS,
    build_macro_signal,
    descriptive_statistics,
    load_ccm,
    load_crsp,
    load_ff5,
    plot_cumulative_returns,
    prepare_alpha_table,
    prepare_summary_table,
    run_factor_regressions,
    run_subperiod_analysis,
    run_core_strategy,
)


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_RETURNS_CSV = PROJECT_ROOT / "hml_market_regime_switch_returns.csv"
OUTPUT_SUMMARY_CSV = PROJECT_ROOT / "hml_market_regime_switch_summary_table.csv"
OUTPUT_ALPHA_CSV = PROJECT_ROOT / "hml_market_regime_switch_alpha_table.csv"
OUTPUT_ROBUSTNESS_CSV = PROJECT_ROOT / "hml_market_regime_switch_robustness_summary.csv"
OUTPUT_COMPARISON_PLOT = PROJECT_ROOT / "hml_market_regime_switch_comparison.png"
OUTPUT_ROBUSTNESS_PLOT = PROJECT_ROOT / "hml_market_regime_switch_robustness.png"


@dataclass
class HmlMarketSwitchArtifacts:
    ff_main: pd.DataFrame
    self_main: pd.DataFrame
    comparison: pd.DataFrame
    descriptive_stats: pd.DataFrame
    regression_results: pd.DataFrame
    summary_table: pd.DataFrame
    alpha_table: pd.DataFrame
    subperiod_results: pd.DataFrame
    ff_tercile: pd.DataFrame
    ff_vix: pd.DataFrame
    self_tercile: pd.DataFrame
    self_vix: pd.DataFrame
    robustness_summary: pd.DataFrame


def load_value_fundamentals() -> pd.DataFrame:
    path = Path(__file__).resolve().parent.parent / "Data" / "QMJ_data.csv"
    funda = pd.read_csv(
        path,
        usecols=["gvkey", "datadate", "fyear", "ceq", "indfmt", "datafmt", "consol"],
        dtype={"gvkey": "string"},
        low_memory=False,
    )
    funda["datadate"] = pd.to_datetime(funda["datadate"], errors="coerce")
    funda["ceq"] = pd.to_numeric(funda["ceq"], errors="coerce")
    funda = funda.loc[
        (funda["indfmt"] == "INDL")
        & (funda["datafmt"] == "STD")
        & (funda["consol"] == "C")
        & funda["datadate"].notna()
        & funda["ceq"].gt(0)
    ].copy()
    funda = funda.sort_values(["gvkey", "datadate"]).reset_index(drop=True)
    funda["book_equity"] = funda["ceq"]
    funda["availability_date"] = funda["datadate"] + pd.DateOffset(months=6) + MonthEnd(0)
    funda["formation_year"] = np.where(
        funda["availability_date"].dt.month.le(6),
        funda["availability_date"].dt.year,
        funda["availability_date"].dt.year + 1,
    ).astype("int32")
    funda["formation_date"] = pd.to_datetime(
        {
            "year": funda["formation_year"],
            "month": 6,
            "day": 30,
        }
    )
    return funda[["gvkey", "datadate", "fyear", "book_equity", "formation_year", "formation_date"]]


def link_ff_style_book_equity(
    funda: pd.DataFrame,
    ccm: pd.DataFrame,
) -> pd.DataFrame:
    linked = funda.merge(ccm, on="gvkey", how="inner")
    linked = linked.loc[linked["datadate"].between(linked["linkdt"], linked["linkenddt"])].copy()
    linked = linked.sort_values(["permno", "formation_year", "datadate"])
    linked = linked.drop_duplicates(["permno", "formation_year"], keep="last")
    return linked[["permno", "formation_year", "formation_date", "book_equity"]]


def _assign_ff_buckets(universe: pd.DataFrame) -> pd.DataFrame:
    nyse = universe.loc[(universe["exchcd"] == 1) & universe["bm"].gt(0)].copy()
    size_break = nyse.groupby("formation_date")["size_me"].median().rename("size_median")
    bm_breaks = (
        nyse.groupby("formation_date")["bm"]
        .quantile([0.3, 0.7])
        .unstack()
        .rename(columns={0.3: "bm30", 0.7: "bm70"})
    )

    ranked = (
        universe.merge(size_break, left_on="formation_date", right_index=True, how="inner")
        .merge(bm_breaks, left_on="formation_date", right_index=True, how="inner")
    )
    ranked = ranked.loc[ranked["bm"].gt(0) & ranked["size_me"].gt(0)].copy()
    ranked["size_bucket"] = np.where(ranked["size_me"] <= ranked["size_median"], "S", "B")
    ranked["bm_bucket"] = np.select(
        [
            ranked["bm"] <= ranked["bm30"],
            ranked["bm"] <= ranked["bm70"],
            ranked["bm"] > ranked["bm70"],
        ],
        ["L", "N", "H"],
        default=None,
    )
    ranked = ranked.loc[ranked["bm_bucket"].notna()].copy()
    ranked["portfolio"] = ranked["size_bucket"] + ranked["bm_bucket"]
    return ranked


def build_self_ff_size_value_factors(crsp: pd.DataFrame) -> pd.DataFrame:
    funda = load_value_fundamentals()
    ccm = load_ccm()
    linked = link_ff_style_book_equity(funda, ccm)

    june_snapshot = crsp.loc[
        crsp["date"].dt.month.eq(6),
        ["permno", "date", "exchcd", "me", "ret_history_24"],
    ].copy()
    june_snapshot["formation_year"] = june_snapshot["date"].dt.year.astype("int32")
    june_snapshot = june_snapshot.rename(columns={"date": "formation_date", "me": "size_me"})

    december_snapshot = crsp.loc[
        crsp["date"].dt.month.eq(12),
        ["permno", "date", "me"],
    ].copy()
    december_snapshot["formation_year"] = (december_snapshot["date"].dt.year + 1).astype("int32")
    december_snapshot = december_snapshot.rename(columns={"me": "dec_me"})

    universe = (
        linked.merge(june_snapshot, on=["permno", "formation_year", "formation_date"], how="inner")
        .merge(december_snapshot[["permno", "formation_year", "dec_me"]], on=["permno", "formation_year"], how="inner")
    )
    universe["bm"] = universe["book_equity"] / universe["dec_me"]
    universe = universe.loc[
        universe["ret_history_24"] & universe["size_me"].gt(0) & universe["dec_me"].gt(0) & universe["bm"].gt(0)
    ].copy()
    universe = _assign_ff_buckets(universe)

    monthly = crsp[["permno", "date", "ret", "me"]].copy().sort_values(["permno", "date"])
    monthly["me_lag"] = monthly.groupby("permno")["me"].shift(1)
    monthly["formation_year"] = np.where(
        monthly["date"].dt.month.ge(7),
        monthly["date"].dt.year,
        monthly["date"].dt.year - 1,
    ).astype("int32")
    panel = monthly.merge(
        universe[["permno", "formation_year", "portfolio"]],
        on=["permno", "formation_year"],
        how="inner",
    )
    panel = panel.loc[
        panel["ret"].notna()
        & panel["me_lag"].gt(0)
        & panel["date"].between(panel["formation_year"].map(lambda y: pd.Timestamp(year=int(y), month=7, day=31)),
                                panel["formation_year"].map(lambda y: pd.Timestamp(year=int(y) + 1, month=6, day=30)))
    ].copy()

    grouped = panel.groupby(["date", "portfolio"])[["ret", "me_lag"]].apply(
        lambda g: np.average(g["ret"], weights=g["me_lag"])
    )
    portfolio_returns = grouped.unstack("portfolio").sort_index()
    required = ["SL", "SN", "SH", "BL", "BN", "BH"]
    portfolio_returns = portfolio_returns.reindex(columns=required)

    factors = pd.DataFrame(index=portfolio_returns.index)
    factors["R_hml_self"] = 0.5 * (portfolio_returns["SH"] + portfolio_returns["BH"]) - 0.5 * (
        portfolio_returns["SL"] + portfolio_returns["BL"]
    )
    factors["R_smb_self"] = (portfolio_returns["SL"] + portfolio_returns["SN"] + portfolio_returns["SH"]) / 3 - (
        portfolio_returns["BL"] + portfolio_returns["BN"] + portfolio_returns["BH"]
    ) / 3
    return factors.sort_index()


def build_self_hml_factor(crsp: pd.DataFrame) -> pd.Series:
    return build_self_ff_size_value_factors(crsp)["R_hml_self"]


def build_self_smb_factor(crsp: pd.DataFrame) -> pd.Series:
    return build_self_ff_size_value_factors(crsp)["R_smb_self"]


def build_self_market_return(crsp: pd.DataFrame) -> pd.Series:
    market = crsp[["permno", "date", "ret", "me"]].copy()
    market = market.sort_values(["permno", "date"])
    market["me_lag"] = market.groupby("permno")["me"].shift(1)
    market = market.loc[market["ret"].notna() & market["me_lag"].gt(0)].copy()
    market["weighted_ret"] = market["ret"] * market["me_lag"]

    grouped = market.groupby("date")[["weighted_ret", "me_lag"]].sum()
    grouped["R_mkt_self"] = grouped["weighted_ret"] / grouped["me_lag"]
    return grouped["R_mkt_self"].sort_index()


def build_ff_strategy(ff5: pd.DataFrame, macro_signal: pd.DataFrame) -> pd.DataFrame:
    panel = ff5[["date", "RMW", "CMA", "HML", "Mkt-RF", "RF"]].copy()
    panel["market_total"] = panel["Mkt-RF"] + panel["RF"]
    panel["bad_leg"] = 0.5 * (panel["RMW"] + panel["CMA"])
    panel["good_leg"] = 0.5 * (panel["HML"] + panel["market_total"])

    signal = macro_signal.loc[macro_signal["signal_ready"], ["holding_date", "regime_label", "M_t"]].rename(
        columns={"holding_date": "date"}
    )
    panel = panel.merge(signal, on="date", how="inner")
    panel["ret"] = np.where(panel["regime_label"] == "good", panel["good_leg"], panel["bad_leg"])
    return panel[["date", "regime_label", "M_t", "bad_leg", "good_leg", "ret"]]


def build_self_strategy(
    factor_returns: pd.DataFrame,
    self_hml: pd.Series,
    self_market: pd.Series,
    macro_signal: pd.DataFrame,
) -> pd.DataFrame:
    panel = factor_returns.merge(self_hml.reset_index(), on="date", how="inner").merge(
        self_market.reset_index(), on="date", how="inner"
    )
    panel["bad_leg"] = 0.5 * (panel["R_prof"] + panel["R_inv"])
    panel["good_leg"] = 0.5 * (panel["R_hml_self"] + panel["R_mkt_self"])

    signal = macro_signal.loc[macro_signal["signal_ready"], ["holding_date", "regime_label", "M_t"]].rename(
        columns={"holding_date": "date"}
    )
    panel = panel.merge(signal, on="date", how="inner")
    panel["ret"] = np.where(panel["regime_label"] == "good", panel["good_leg"], panel["bad_leg"])
    return panel[
        ["date", "regime_label", "M_t", "R_prof", "R_inv", "R_hml_self", "R_mkt_self", "bad_leg", "good_leg", "ret"]
    ]


def build_comparison_frame(ff_main: pd.DataFrame, self_main: pd.DataFrame) -> pd.DataFrame:
    comparison = ff_main[["date", "ret", "bad_leg", "good_leg"]].rename(
        columns={
            "ret": "FF_Direct_Main",
            "bad_leg": "FF_Bad_Basket",
            "good_leg": "FF_Good_Basket",
        }
    )
    comparison = comparison.merge(
        self_main[["date", "ret", "bad_leg", "good_leg"]].rename(
            columns={
                "ret": "Self_Built_Main",
                "bad_leg": "Self_Bad_Basket",
                "good_leg": "Self_Good_Basket",
            }
        ),
        on="date",
        how="outer",
    ).sort_values("date")
    return comparison


def build_robustness_summary(
    ff_main: pd.DataFrame,
    ff_tercile: pd.DataFrame,
    ff_vix: pd.DataFrame,
    self_main: pd.DataFrame,
    self_tercile: pd.DataFrame,
    self_vix: pd.DataFrame,
    ff5: pd.DataFrame,
) -> pd.DataFrame:
    frame = (
        ff_main[["date", "ret"]].rename(columns={"ret": "FF_Direct_Main"})
        .merge(ff_tercile[["date", "ret"]].rename(columns={"ret": "FF_Direct_Tercile"}), on="date", how="outer")
        .merge(ff_vix[["date", "ret"]].rename(columns={"ret": "FF_Direct_VIX"}), on="date", how="outer")
        .merge(self_main[["date", "ret"]].rename(columns={"ret": "Self_Built_Main"}), on="date", how="outer")
        .merge(self_tercile[["date", "ret"]].rename(columns={"ret": "Self_Built_Tercile"}), on="date", how="outer")
        .merge(self_vix[["date", "ret"]].rename(columns={"ret": "Self_Built_VIX"}), on="date", how="outer")
        .sort_values("date")
    )
    stats = descriptive_statistics(frame)
    regs = run_factor_regressions(frame, ff5)
    return prepare_summary_table(stats, regs)


def run_strategy_family() -> HmlMarketSwitchArtifacts:
    ff5 = load_ff5()
    macro_main = build_macro_signal(threshold_mode="median", include_vix=False)
    macro_tercile = build_macro_signal(threshold_mode="tercile", include_vix=False)
    macro_vix = build_macro_signal(threshold_mode="median", include_vix=True)

    core = run_core_strategy(write_csv=False)
    crsp = load_crsp()
    self_hml = build_self_hml_factor(crsp)
    self_market = build_self_market_return(crsp)

    ff_main = build_ff_strategy(ff5, macro_main)
    ff_tercile = build_ff_strategy(ff5, macro_tercile)
    ff_vix = build_ff_strategy(ff5, macro_vix)

    self_main = build_self_strategy(core.factor_returns, self_hml, self_market, macro_main)
    self_tercile = build_self_strategy(core.factor_returns, self_hml, self_market, macro_tercile)
    self_vix = build_self_strategy(core.factor_returns, self_hml, self_market, macro_vix)

    comparison = build_comparison_frame(ff_main, self_main)
    stats = descriptive_statistics(comparison)
    regs = run_factor_regressions(comparison, ff5)
    summary = prepare_summary_table(stats, regs)
    alpha = prepare_alpha_table(regs)
    subperiod = run_subperiod_analysis(comparison, ff5)
    robustness = build_robustness_summary(ff_main, ff_tercile, ff_vix, self_main, self_tercile, self_vix, ff5)

    return HmlMarketSwitchArtifacts(
        ff_main=ff_main,
        self_main=self_main,
        comparison=comparison,
        descriptive_stats=stats,
        regression_results=regs,
        summary_table=summary,
        alpha_table=alpha,
        subperiod_results=subperiod,
        ff_tercile=ff_tercile,
        ff_vix=ff_vix,
        self_tercile=self_tercile,
        self_vix=self_vix,
        robustness_summary=robustness,
    )


def export_outputs(artifacts: HmlMarketSwitchArtifacts) -> dict[str, Path]:
    returns = (
        artifacts.ff_main[["date", "ret"]].rename(columns={"ret": "FF_Direct_Main"})
        .merge(artifacts.self_main[["date", "ret"]].rename(columns={"ret": "Self_Built_Main"}), on="date", how="outer")
        .sort_values("date")
    )
    returns.to_csv(OUTPUT_RETURNS_CSV, index=False)
    artifacts.summary_table.to_csv(OUTPUT_SUMMARY_CSV, index=False)
    artifacts.alpha_table.to_csv(OUTPUT_ALPHA_CSV, index=False)
    artifacts.robustness_summary.to_csv(OUTPUT_ROBUSTNESS_CSV, index=False)

    plot_cumulative_returns(
        artifacts.comparison,
        OUTPUT_COMPARISON_PLOT,
        title="Regime Switch Comparison: FF Direct vs Self Built",
    )
    robustness_plot = (
        artifacts.ff_main[["date", "ret"]].rename(columns={"ret": "FF_Direct_Main"})
        .merge(artifacts.ff_tercile[["date", "ret"]].rename(columns={"ret": "FF_Direct_Tercile"}), on="date", how="outer")
        .merge(artifacts.ff_vix[["date", "ret"]].rename(columns={"ret": "FF_Direct_VIX"}), on="date", how="outer")
        .merge(artifacts.self_main[["date", "ret"]].rename(columns={"ret": "Self_Built_Main"}), on="date", how="outer")
        .merge(artifacts.self_tercile[["date", "ret"]].rename(columns={"ret": "Self_Built_Tercile"}), on="date", how="outer")
        .merge(artifacts.self_vix[["date", "ret"]].rename(columns={"ret": "Self_Built_VIX"}), on="date", how="outer")
        .sort_values("date")
    )
    plot_cumulative_returns(
        robustness_plot,
        OUTPUT_ROBUSTNESS_PLOT,
        title="Regime Switch Robustness: FF Direct and Self Built",
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
    artifacts = run_strategy_family()
    paths = export_outputs(artifacts)
    print(
        f"Built FF-direct sample with {len(artifacts.ff_main):,} months "
        f"and self-built sample with {len(artifacts.self_main):,} months."
    )
    for name, path in paths.items():
        print(f"{name}: {path}")
    print("\nSummary table:")
    print(artifacts.summary_table.to_string(index=False))
