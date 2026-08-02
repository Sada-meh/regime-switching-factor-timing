# Data

## What is included

These files are public and ship with the repository:

| File | Source | Notes |
|---|---|---|
| `F-F_Research_Data_5_Factors_2x3_CSV/` | Kenneth R. French Data Library | Monthly FF5 factors + RF |
| `DAAA.csv` | FRED `DAAA` | Moody's Aaa corporate yield |
| `DBAA.csv` | FRED `DBAA` | Moody's Baa corporate yield |
| `VIXCLS (1).csv` | FRED `VIXCLS` | CBOE VIX |
| `USEPUINDXM.csv` | FRED `USEPUINDXM` | Baker-Bloom-Davis Economic Policy Uncertainty |
| `qmj_returns.csv` | Derived monthly return series | Aggregated output, not raw vendor data |

## What is NOT included, and why

The following inputs are **subscription-licensed** and cannot be redistributed.
They are excluded from this repository and blocked in `.gitignore`:

| Missing file | Source | How to rebuild |
|---|---|---|
| `crsp_clean_filtered(in).csv` | CRSP Monthly Stock File (via WRDS) | Query `crsp.msf` joined to `crsp.msenames`; keep `permno, date, shrcd, exchcd, siccd, ticker, comnam, prc, vol, ret, shrout`; compute `me = |prc| * shrout`; filter `shrcd in (10,11)` and `exchcd in (1,2,3)`; sample 1990-01 to 2024-09 |
| `crsp_a_ccm.csv` | CRSP/Compustat Merged Linking Table (via WRDS) | Export `crsp.ccmxpf_linktable` with `gvkey, tic, LINKPRIM, LIID, LINKTYPE, LPERMNO, LPERMCO, LINKDT, LINKENDDT` |
| `QMJ_data.csv` | Compustat North America Fundamentals Annual (via WRDS) | Export `comp.funda` with `gvkey, datadate, fyear, indfmt, datafmt, consol, curcd, costat, tic, act, at, ceq, che, dlc, dltt, lct, ppegt, cogs, dp, ib, ni, oibdp, sale, capx, oancf` |
| `10Y monthly US(Table Data).csv` | LSEG/Refinitiv `US10YT=RR` | Substitute FRED `DGS10` (monthly average), or re-export from Datastream |
| `US 3M monthly(Table Data).csv` | LSEG/Refinitiv `US3MT=RR` | Substitute FRED `DTB3` (monthly average), or re-export from Datastream |

Place the rebuilt files in this directory using the exact filenames above.

Any loader that needs a missing file raises a `FileNotFoundError` pointing
back to this page, rather than failing with an opaque traceback.

## What still runs without them

The public files are enough to load the Fama-French factor panel and the
macro signal inputs. The Markov regime model in
`trial-and-error/Markov/markov_factor_timing.py` is built on FF5 factors
only, so it runs end-to-end on the included data. The CRSP-based
self-built factor strategies do not.
