#!/usr/bin/env python3
"""Backfill point-in-time historical beta values for stock CSV files."""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import pandas as pd
import yfinance as yf


BASE_DIR = Path("/srv/data")
STOCKS_DIR = BASE_DIR / "stocks"

BETA_COLUMN = "historical_beta"
BENCHMARK_TICKER = "^GSPC"
BETA_WINDOW_DAYS = 252
BETA_MIN_PERIODS = 126
BETA_LOOKBACK_DAYS = 365 * 2

STOCK_FILE_RE = re.compile(r"^(?P<ticker>.+)_stock_data\.csv$")


def stock_ticker(path: Path) -> str | None:
    match = STOCK_FILE_RE.match(path.name)
    if not match:
        return None
    return match.group("ticker")


def is_blank(value: object) -> bool:
    if pd.isna(value):
        return True
    return str(value) == ""


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    df.to_csv(temp_path, index=False)
    temp_path.replace(path)


@lru_cache(maxsize=2048)
def download_adjusted_close(ticker: str, start: date, end: date) -> pd.Series:
    downloaded = yf.download(
        ticker,
        start=start.isoformat(),
        end=end.isoformat(),
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if downloaded.empty:
        return pd.Series(dtype="float64", name=ticker)

    if isinstance(downloaded.columns, pd.MultiIndex):
        if "Adj Close" in downloaded.columns.get_level_values(0):
            close = downloaded["Adj Close"]
        else:
            close = downloaded["Close"]

        if isinstance(close, pd.DataFrame):
            close = close[ticker] if ticker in close.columns else close.iloc[:, 0]
    else:
        column = "Adj Close" if "Adj Close" in downloaded.columns else "Close"
        close = downloaded[column]

    close.index = pd.to_datetime(close.index).normalize()
    return pd.to_numeric(close, errors="coerce").dropna().rename(ticker)


def calculate_historical_beta(
    ticker: str,
    dates: pd.Series,
    *,
    benchmark_ticker: str = BENCHMARK_TICKER,
    window: int = BETA_WINDOW_DAYS,
    min_periods: int = BETA_MIN_PERIODS,
) -> pd.Series:
    clean_dates = pd.to_datetime(dates, errors="coerce").dropna()
    if clean_dates.empty:
        return pd.Series(dtype="float64")

    start = clean_dates.min().date() - timedelta(days=BETA_LOOKBACK_DAYS)
    end = clean_dates.max().date() + timedelta(days=1)

    stock_close = download_adjusted_close(ticker, start, end)
    benchmark_close = download_adjusted_close(benchmark_ticker, start, end)
    if stock_close.empty or benchmark_close.empty:
        return pd.Series(dtype="float64")

    prices = pd.concat([stock_close, benchmark_close], axis=1, join="inner")
    returns = prices.pct_change(fill_method=None).dropna()
    if returns.empty:
        return pd.Series(dtype="float64")

    covariance = returns[ticker].rolling(window, min_periods=min_periods).cov(
        returns[benchmark_ticker]
    )
    variance = returns[benchmark_ticker].rolling(window, min_periods=min_periods).var()
    beta = covariance / variance
    beta.index = pd.to_datetime(beta.index).strftime("%Y-%m-%d")
    return beta.dropna()


def add_historical_beta(
    df: pd.DataFrame, ticker: str, *, overwrite: bool = False
) -> tuple[pd.DataFrame, int]:
    if "Date" not in df.columns:
        raise ValueError("CSV is missing required Date column")

    updated = df.copy()
    updated["Date"] = pd.to_datetime(updated["Date"], errors="coerce").dt.strftime(
        "%Y-%m-%d"
    )
    updated = updated.dropna(subset=["Date"])

    if BETA_COLUMN not in updated.columns:
        updated[BETA_COLUMN] = ""
    updated[BETA_COLUMN] = updated[BETA_COLUMN].astype("object")

    target_mask = pd.Series(overwrite, index=updated.index)
    if not overwrite:
        target_mask = updated[BETA_COLUMN].map(is_blank)

    target_dates = updated.loc[target_mask, "Date"]
    if target_dates.empty:
        return updated, 0

    beta_by_date = calculate_historical_beta(ticker, target_dates)
    if beta_by_date.empty:
        return updated, 0

    before = updated[BETA_COLUMN].copy()
    mapped = updated.loc[target_mask, "Date"].map(beta_by_date)
    mapped = mapped.dropna()
    updated.loc[mapped.index, BETA_COLUMN] = mapped

    changed = int((updated[BETA_COLUMN].astype(str) != before.astype(str)).sum())
    return updated, changed


def backfill_historical_beta_file(path: Path, *, overwrite: bool = False) -> int:
    ticker = stock_ticker(path)
    if ticker is None:
        return 0

    original = pd.read_csv(path)
    updated, changed = add_historical_beta(original, ticker, overwrite=overwrite)
    if changed or BETA_COLUMN not in original.columns:
        atomic_write_csv(updated, path)
    logging.info("%s: filled %s historical beta value(s)", ticker, changed)
    return changed


def backfill_historical_beta(
    stocks_dir: Path = STOCKS_DIR, *, overwrite: bool = False
) -> None:
    for path in sorted(stocks_dir.glob("*_stock_data.csv")):
        try:
            backfill_historical_beta_file(path, overwrite=overwrite)
        except Exception:
            logging.exception("Failed to backfill historical beta for %s", path)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add or fill point-in-time historical beta values in stock CSVs."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Specific stock CSV files to backfill. Defaults to all stock CSVs.",
    )
    parser.add_argument(
        "--stocks-dir",
        type=Path,
        default=STOCKS_DIR,
        help=f"Directory containing *_stock_data.csv files. Default: {STOCKS_DIR}",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recalculate beta values even when historical_beta already has data.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = parse_args(sys.argv[1:] if argv is None else argv)

    paths = args.paths or sorted(args.stocks_dir.glob("*_stock_data.csv"))
    for path in paths:
        try:
            backfill_historical_beta_file(path, overwrite=args.overwrite)
        except Exception:
            logging.exception("Failed to backfill historical beta for %s", path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
