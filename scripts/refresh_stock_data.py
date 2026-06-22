#!/usr/bin/env python3
"""Append missing yfinance and FRED rows to local CSV data files."""

from __future__ import annotations

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


def stock_download_start(existing: pd.DataFrame, default_start: date, today: date) -> date:
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


def last_non_blank(existing: pd.DataFrame, column: str, default: object = "") -> object:
    if column not in existing.columns:
        return default

    values = existing[column].dropna()
    values = values[values.astype(str) != ""]
    if values.empty:
        return default
    return values.iloc[-1]


def format_stock_rows(
    downloaded: pd.DataFrame, existing: pd.DataFrame, ticker: str
) -> pd.DataFrame:
    columns = list(existing.columns)
    rows = pd.DataFrame(index=downloaded.index)

    close = downloaded.get("Close", pd.Series(index=downloaded.index, dtype="float64"))
    adj_close = downloaded.get(
        "Adj Close", pd.Series(index=downloaded.index, dtype="float64")
    )

    for column in columns:
        if column == "Date":
            rows[column] = downloaded["Date"]
        elif column == "TR.CLOSEPRICE(Adjusted=0)":
            rows[column] = close
        elif column == "Adj Close":
            rows[column] = adj_close.fillna(close)
        elif column == "dividend_yield":
            rows[column] = 0.0
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
        elif column in ("stock_beta", "historical_beta"):
            rows[column] = last_non_blank(existing, column, "")
        elif column in downloaded.columns:
            rows[column] = downloaded[column]
        else:
            rows[column] = last_non_blank(existing, column, "")

    return rows[columns]


def is_blank(value: object) -> bool:
    if pd.isna(value):
        return True
    return str(value) == ""


def merge_stock_rows(
    existing: pd.DataFrame, new_rows: pd.DataFrame
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
                if is_blank(updated.at[row_index, column]) and not is_blank(value):
                    updated.at[row_index, column] = value
                    filled += 1

    if not append_rows.empty:
        updated = pd.concat([updated, append_rows], ignore_index=True)

    updated = updated.sort_values("Date").reset_index(drop=True)
    return updated, len(append_rows), filled


def append_new_rows(existing: pd.DataFrame, new_rows: pd.DataFrame) -> pd.DataFrame:
    updated, _, _ = merge_stock_rows(existing, new_rows)
    return updated


def refresh_stock_file(path: Path) -> None:
    ticker = stock_ticker(path)
    if ticker is None:
        return

    original = clean_dates(pd.read_csv(path))
    today = date.today()
    existing = drop_unclosed_rows(original, today)
    start = stock_download_start(existing, today - timedelta(days=365 * 5), today)
    end = today

    if start >= end:
        if len(existing) != len(original):
            atomic_write_csv(existing, path)
            logging.info(
                "%s: removed %s unclosed row(s)", ticker, len(original) - len(existing)
            )
        logging.info("%s: already current", ticker)
        return

    downloaded = yf.download(
        ticker,
        start=start.isoformat(),
        end=end.isoformat(),
        actions=True,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    downloaded = flatten_yfinance(downloaded, ticker)
    if downloaded.empty:
        if len(existing) != len(original):
            atomic_write_csv(existing, path)
            logging.info(
                "%s: removed %s unclosed row(s)", ticker, len(original) - len(existing)
            )
        logging.info("%s: no new yfinance rows", ticker)
        return

    new_rows = format_stock_rows(downloaded, existing, ticker)
    updated, added, filled = merge_stock_rows(existing, new_rows)

    removed = len(original) - len(existing)
    if added or filled or removed:
        atomic_write_csv(updated, path)
    logging.info(
        "%s: appended %s row(s), filled %s blank value(s), removed %s unclosed row(s)",
        ticker,
        added,
        filled,
        removed,
    )


def refresh_stocks() -> None:
    for path in sorted(STOCKS_DIR.glob("*_stock_data.csv")):
        try:
            refresh_stock_file(path)
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


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    refresh_stocks()
    refresh_risk_free_rate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
