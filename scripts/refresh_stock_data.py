#!/usr/bin/env python3
"""Append missing yfinance and FRED rows to local CSV data files."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import urllib.parse
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

from historical_beta import BETA_COLUMN, add_historical_beta


BASE_DIR = Path("/srv/data")
STOCKS_DIR = BASE_DIR / "stocks"
RISK_FREE_DIR = BASE_DIR / "risk_free_rate"
ENV_FILE = Path("/home/joflay/stocks_update/.env")

FRED_KEY_NAME = "Fred_Key"
FRED_SERIES_ID = "DGS3MO"
FRED_START_DATE = date(2010, 1, 1)

STOCK_FILE_RE = re.compile(r"^(?P<ticker>.+)_stock_data\.csv$")
YFINANCE_UNIVERSE = "yfinance-raw-split-safe"
STOCK_RECENT_BACKFILL_DAYS = 14
YFINANCE_STOCK_COLUMNS = [
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
    "Dividends",
    "Stock Splits",
]
LEGACY_CLOSE_COLUMN = "TR.CLOSEPRICE(Adjusted=0)"
REFRESHABLE_STOCK_COLUMNS = set(YFINANCE_STOCK_COLUMNS[1:])


def stock_ticker(path: Path) -> str | None:
    match = STOCK_FILE_RE.match(path.name)
    if not match:
        return None
    return match.group("ticker")


def clean_dates(df: pd.DataFrame) -> pd.DataFrame:
    if "Date" not in df.columns:
        raise ValueError("CSV is missing required Date column")

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return df.dropna(subset=["Date"])


def next_missing_date(existing: pd.DataFrame, default_start: date) -> date:
    dates = pd.to_datetime(existing["Date"], errors="coerce").dropna()
    if dates.empty:
        return default_start
    return dates.max().date() + timedelta(days=1)


def drop_unclosed_rows(existing: pd.DataFrame, today: date) -> pd.DataFrame:
    """Remove rows for today or later because daily bars are not final yet."""
    dates = pd.to_datetime(existing["Date"], errors="coerce")
    return existing[dates.dt.date < today].reset_index(drop=True)


def stock_download_start(
    existing: pd.DataFrame,
    default_start: date,
    today: date,
    *,
    full_history: bool = False,
) -> date:
    if full_history and not existing.empty:
        dates = pd.to_datetime(existing["Date"], errors="coerce").dropna()
        if not dates.empty:
            return dates.min().date()

    next_date = next_missing_date(existing, default_start)
    if existing.empty:
        return next_date

    backfill_start = today - timedelta(days=STOCK_RECENT_BACKFILL_DAYS)
    return max(default_start, min(next_date, backfill_start))


def load_env_value(key: str) -> str | None:
    if key in os.environ:
        return os.environ[key]

    if not ENV_FILE.exists():
        return None

    for raw_line in ENV_FILE.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        name, value = line.split("=", 1)
        if name.strip() != key:
            continue

        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            return value[1:-1]
        return value

    return None


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    df.to_csv(temp_path, index=False)
    temp_path.replace(path)


def flatten_yfinance(downloaded: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if downloaded.empty:
        return downloaded

    if isinstance(downloaded.columns, pd.MultiIndex):
        if ticker in downloaded.columns.get_level_values(-1):
            downloaded = downloaded.xs(ticker, axis=1, level=-1)
        else:
            downloaded.columns = [
                "_".join(str(part) for part in column if part)
                for column in downloaded.columns
            ]

    downloaded = downloaded.reset_index()
    if "Date" not in downloaded.columns and "Datetime" in downloaded.columns:
        downloaded = downloaded.rename(columns={"Datetime": "Date"})

    downloaded["Date"] = pd.to_datetime(downloaded["Date"]).dt.strftime("%Y-%m-%d")
    return downloaded


def yfinance_ticker(ticker: str) -> str:
    """Convert dot-class tickers such as BRK.B to yfinance's BRK-B form."""
    return ticker.replace(".", "-")


def last_non_blank(existing: pd.DataFrame, column: str, default: object = "") -> object:
    if column not in existing.columns:
        return default

    values = existing[column].dropna()
    values = values[values.astype(str) != ""]
    if values.empty:
        return default
    return values.iloc[-1]


def ensure_stock_schema(existing: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Add the complete yfinance schema while retaining all legacy columns."""
    updated = existing.copy()
    legacy_close = updated.get(
        LEGACY_CLOSE_COLUMN,
        pd.Series(index=updated.index, dtype="float64"),
    )
    close = updated.get("Close", legacy_close)
    close = close.where(close.notna(), legacy_close)

    defaults: dict[str, object] = {
        "Date": pd.NA,
        "Open": pd.NA,
        "High": pd.NA,
        "Low": pd.NA,
        "Close": close,
        "Adj Close": close,
        "Volume": pd.NA,
        "Dividends": pd.NA,
        "Stock Splits": pd.NA,
    }
    for column, default in defaults.items():
        if column not in updated.columns:
            updated[column] = default

    updated["Close"] = close
    updated = updated.drop(columns=[LEGACY_CLOSE_COLUMN], errors="ignore")

    if "ticker" in updated.columns:
        updated["ticker"] = updated["ticker"].fillna(ticker)

    legacy_columns = [
        column for column in updated.columns if column not in YFINANCE_STOCK_COLUMNS
    ]
    return updated[YFINANCE_STOCK_COLUMNS + legacy_columns]


def needs_stock_history_backfill(existing: pd.DataFrame) -> bool:
    """Return whether canonical yfinance fields need a full-history download."""
    for column in YFINANCE_STOCK_COLUMNS[1:]:
        if column not in existing.columns:
            return True
        if pd.to_numeric(existing[column], errors="coerce").isna().any():
            return True
    return False


def split_adjust_yfinance_prices(downloaded: pd.DataFrame) -> pd.DataFrame:
    """Match the split-safe OHLC transformation used by the options pipeline."""
    adjusted = downloaded.copy()
    splits = pd.to_numeric(
        adjusted.get("Stock Splits", pd.Series(0.0, index=adjusted.index)),
        errors="coerce",
    ).fillna(0.0)
    split_multiplier = splits.where(splits > 0.0, 1.0)
    future_split_factor = (
        split_multiplier.iloc[::-1].cumprod().iloc[::-1] / split_multiplier
    )

    for column in ["Open", "High", "Low"]:
        if column in adjusted.columns:
            adjusted[column] = (
                pd.to_numeric(adjusted[column], errors="coerce") * future_split_factor
            )

    return adjusted


def format_stock_rows(
    downloaded: pd.DataFrame, existing: pd.DataFrame, ticker: str
) -> pd.DataFrame:
    existing = ensure_stock_schema(existing, ticker)
    columns = list(existing.columns)
    rows = pd.DataFrame(index=downloaded.index)

    downloaded = split_adjust_yfinance_prices(downloaded)
    close = downloaded.get("Close", pd.Series(index=downloaded.index, dtype="float64"))
    adj_close = downloaded.get(
        "Adj Close", pd.Series(index=downloaded.index, dtype="float64")
    )

    for column in columns:
        if column == "Date":
            rows[column] = downloaded["Date"]
        elif column == "Adj Close":
            rows[column] = adj_close.fillna(close)
        elif column == "dividend_yield":
            rows[column] = ""
        elif column == "ticker":
            rows[column] = ticker
        elif column == "lseg_universe":
            rows[column] = (
                last_non_blank(existing, "lseg_universe", YFINANCE_UNIVERSE)
                or YFINANCE_UNIVERSE
            )
        elif column == "lseg_ric":
            rows[column] = last_non_blank(existing, "lseg_ric", "")
        elif column == "Stock Splits":
            rows[column] = downloaded.get("Stock Splits", 0.0)
        elif column == "Dividends":
            rows[column] = downloaded.get("Dividends", 0.0)
        elif column == "Close":
            rows[column] = close
        elif column == "stock_beta":
            rows[column] = last_non_blank(existing, column, "")
        elif column == BETA_COLUMN:
            rows[column] = ""
        elif column in downloaded.columns:
            rows[column] = downloaded[column]
        else:
            rows[column] = last_non_blank(existing, column, "")

    return rows[columns]


def add_dividend_yield(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate trailing-365-day dividend yield as the source pipeline does."""
    if df.empty or "Dividends" not in df.columns:
        return df

    updated = df.copy()
    dates = pd.to_datetime(updated["Date"], errors="coerce")
    dividends = pd.to_numeric(updated["Dividends"], errors="coerce").fillna(0.0)
    prices = pd.to_numeric(updated["Close"], errors="coerce")

    trailing_dividends = (
        pd.Series(dividends.to_numpy(), index=dates)
        .rolling("365D", min_periods=1)
        .sum()
        .to_numpy()
    )
    dividend_yield = pd.Series(trailing_dividends, index=updated.index) / prices
    updated["dividend_yield"] = (
        pd.to_numeric(dividend_yield, errors="coerce")
        .replace([float("inf"), float("-inf")], 0.0)
        .fillna(0.0)
        .clip(lower=0.0)
    )
    return updated


def is_blank(value: object) -> bool:
    if pd.isna(value):
        return True
    return str(value) == ""


def merge_stock_rows(
    existing: pd.DataFrame,
    new_rows: pd.DataFrame,
    *,
    overwrite_columns: set[str] | None = None,
) -> tuple[pd.DataFrame, int, int]:
    existing_dates = set(existing["Date"].astype(str))
    append_rows = new_rows[~new_rows["Date"].astype(str).isin(existing_dates)]

    updated = existing.copy()
    filled = 0
    if not new_rows.empty:
        row_indexes_by_date = {
            date_value: index
            for index, date_value in updated["Date"].astype(str).items()
        }
        for _, row in new_rows.iterrows():
            row_index = row_indexes_by_date.get(str(row["Date"]))
            if row_index is None:
                continue

            for column in updated.columns:
                if column == "Date" or column not in row:
                    continue

                value = row[column]
                should_overwrite = (
                    overwrite_columns is not None and column in overwrite_columns
                )
                if (
                    (is_blank(updated.at[row_index, column]) or should_overwrite)
                    and not is_blank(value)
                ):
                    updated.at[row_index, column] = value
                    filled += 1

    if not append_rows.empty:
        updated = pd.concat([updated, append_rows], ignore_index=True)

    updated = updated.sort_values("Date").reset_index(drop=True)
    return updated, len(append_rows), filled


def append_new_rows(existing: pd.DataFrame, new_rows: pd.DataFrame) -> pd.DataFrame:
    updated, _, _ = merge_stock_rows(existing, new_rows)
    return updated


def refresh_stock_file(path: Path, *, full_overwrite: bool = False) -> None:
    ticker = stock_ticker(path)
    if ticker is None:
        return

    original = clean_dates(pd.read_csv(path))
    needs_history_backfill = needs_stock_history_backfill(original)
    today = date.today()
    existing = ensure_stock_schema(drop_unclosed_rows(original, today), ticker)
    schema_changed = list(existing.columns) != list(original.columns)
    start = stock_download_start(
        existing,
        today - timedelta(days=365 * 5),
        today,
        full_history=full_overwrite or needs_history_backfill,
    )
    end = today

    if start >= end:
        existing, beta_filled = add_historical_beta(existing, ticker)
        if len(existing) != len(original) or schema_changed:
            atomic_write_csv(existing, path)
            logging.info(
                "%s: normalized schema and removed %s unclosed row(s)",
                ticker,
                len(original) - len(existing),
            )
        elif beta_filled:
            atomic_write_csv(existing, path)
        logging.info("%s: already current", ticker)
        return

    yf_ticker = yfinance_ticker(ticker)
    downloaded = yf.download(
        yf_ticker,
        start=start.isoformat(),
        end=end.isoformat(),
        actions=True,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    downloaded = flatten_yfinance(downloaded, yf_ticker)
    if downloaded.empty:
        existing, beta_filled = add_historical_beta(existing, ticker)
        if len(existing) != len(original) or schema_changed:
            atomic_write_csv(existing, path)
            logging.info(
                "%s: normalized schema and removed %s unclosed row(s)",
                ticker,
                len(original) - len(existing),
            )
        elif beta_filled:
            atomic_write_csv(existing, path)
        logging.info(
            "%s: no new yfinance rows, filled %s beta value(s)", ticker, beta_filled
        )
        return

    new_rows = format_stock_rows(downloaded, existing, ticker)
    updated, added, filled = merge_stock_rows(
        existing,
        new_rows,
        overwrite_columns=REFRESHABLE_STOCK_COLUMNS,
    )
    updated = add_dividend_yield(updated)
    updated, beta_filled = add_historical_beta(updated, ticker)

    removed = len(original) - len(existing)
    if added or filled or removed or beta_filled or schema_changed:
        atomic_write_csv(updated, path)
    logging.info(
        (
            "%s: appended %s row(s), refreshed %s value(s), "
            "filled %s beta value(s), removed %s unclosed row(s)"
        ),
        ticker,
        added,
        filled,
        beta_filled,
        removed,
    )


def refresh_stocks(*, full_overwrite: bool = False) -> None:
    for path in sorted(STOCKS_DIR.glob("*_stock_data.csv")):
        try:
            refresh_stock_file(path, full_overwrite=full_overwrite)
        except Exception:
            logging.exception("Failed to refresh %s", path)


def fetch_fred_rows(api_key: str, start: date) -> pd.DataFrame:
    params = urllib.parse.urlencode(
        {
            "series_id": FRED_SERIES_ID,
            "api_key": api_key,
            "file_type": "json",
            "observation_start": start.isoformat(),
        }
    )
    url = f"https://api.stlouisfed.org/fred/series/observations?{params}"

    with urllib.request.urlopen(url, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))

    if "error_message" in payload:
        raise RuntimeError(payload["error_message"])

    rows = pd.DataFrame(payload.get("observations", []))
    if rows.empty:
        return pd.DataFrame(
            columns=[
                "Date",
                "risk_free_rate_percent",
                "risk_free_rate",
                "series_id",
                "source",
            ]
        )

    values = pd.to_numeric(rows["value"], errors="coerce")
    result = pd.DataFrame(
        {
            "Date": pd.to_datetime(rows["date"]).dt.strftime("%Y-%m-%d"),
            "risk_free_rate_percent": values,
            "risk_free_rate": values / 100.0,
            "series_id": FRED_SERIES_ID,
            "source": "fred",
        }
    )
    return result.dropna(subset=["risk_free_rate_percent"])


def refresh_risk_free_rate() -> None:
    api_key = load_env_value(FRED_KEY_NAME)
    if not api_key:
        raise RuntimeError(f"Missing {FRED_KEY_NAME} in environment or {ENV_FILE}")

    path = RISK_FREE_DIR / f"{FRED_SERIES_ID}_risk_free_rate.csv"
    if path.exists():
        existing = clean_dates(pd.read_csv(path))
        start = next_missing_date(existing, FRED_START_DATE)
    else:
        existing = pd.DataFrame(
            columns=[
                "Date",
                "risk_free_rate_percent",
                "risk_free_rate",
                "series_id",
                "source",
            ]
        )
        start = FRED_START_DATE

    new_rows = fetch_fred_rows(api_key, start)
    updated = append_new_rows(existing, new_rows)
    added = len(updated) - len(existing)

    if added:
        atomic_write_csv(updated, path)
    logging.info("%s: appended %s risk-free-rate row(s)", FRED_SERIES_ID, added)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh stock histories from yfinance and risk-free rates from FRED."
    )
    parser.add_argument(
        "--full-overwrite",
        action="store_true",
        help=(
            "Re-download every stock file from its earliest retained date and "
            "replace yfinance price/action fields while preserving beta and metadata."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = parse_args(sys.argv[1:] if argv is None else argv)

    refresh_stocks(full_overwrite=args.full_overwrite)
    refresh_risk_free_rate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
