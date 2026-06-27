from __future__ import annotations

import sys
import unittest
from datetime import date
from pathlib import Path

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from refresh_stock_data import (  # noqa: E402
    YFINANCE_STOCK_COLUMNS,
    add_dividend_yield,
    ensure_stock_schema,
    format_stock_rows,
    merge_stock_rows,
    needs_stock_history_backfill,
    parse_args,
    stock_download_start,
    yfinance_ticker,
)


class StockSchemaTests(unittest.TestCase):
    def test_schema_adds_yfinance_columns_and_preserves_beta_and_metadata(self) -> None:
        existing = pd.DataFrame(
            {
                "Date": ["2026-01-02"],
                "TR.CLOSEPRICE(Adjusted=0)": [100.0],
                "historical_beta": [1.25],
                "custom": ["keep"],
            }
        )

        normalized = ensure_stock_schema(existing, "TEST")

        self.assertEqual(list(normalized.columns[:9]), YFINANCE_STOCK_COLUMNS)
        self.assertEqual(normalized.loc[0, "Close"], 100.0)
        self.assertEqual(normalized.loc[0, "Adj Close"], 100.0)
        self.assertEqual(normalized.loc[0, "historical_beta"], 1.25)
        self.assertEqual(normalized.loc[0, "custom"], "keep")
        self.assertNotIn("TR.CLOSEPRICE(Adjusted=0)", normalized.columns)
        self.assertTrue(pd.isna(normalized.loc[0, "Dividends"]))
        self.assertTrue(pd.isna(normalized.loc[0, "Stock Splits"]))

    def test_downloaded_actions_fill_existing_rows_without_replacing_beta(self) -> None:
        existing = ensure_stock_schema(
            pd.DataFrame(
                {
                    "Date": ["2026-01-02"],
                    "TR.CLOSEPRICE(Adjusted=0)": [100.0],
                    "historical_beta": [1.25],
                }
            ),
            "TEST",
        )
        downloaded = pd.DataFrame(
            {
                "Date": ["2026-01-02"],
                "Open": [99.0],
                "High": [102.0],
                "Low": [98.0],
                "Close": [101.0],
                "Adj Close": [100.5],
                "Volume": [1000],
                "Dividends": [0.5],
                "Stock Splits": [0.0],
            }
        )

        rows = format_stock_rows(downloaded, existing, "TEST")
        merged, added, _ = merge_stock_rows(existing, rows)

        self.assertEqual(added, 0)
        self.assertEqual(merged.loc[0, "Dividends"], 0.5)
        self.assertEqual(merged.loc[0, "Stock Splits"], 0.0)
        self.assertEqual(merged.loc[0, "historical_beta"], 1.25)

    def test_dividend_yield_uses_trailing_year_of_dividends(self) -> None:
        frame = pd.DataFrame(
            {
                "Date": ["2025-01-02", "2025-04-02", "2026-02-01"],
                "Dividends": [1.0, 1.0, 2.0],
                "Close": [100.0, 100.0, 200.0],
            }
        )

        updated = add_dividend_yield(frame)

        self.assertAlmostEqual(updated.loc[0, "dividend_yield"], 0.01)
        self.assertAlmostEqual(updated.loc[1, "dividend_yield"], 0.02)
        self.assertAlmostEqual(updated.loc[2, "dividend_yield"], 0.015)

    def test_yfinance_class_ticker_conversion(self) -> None:
        self.assertEqual(yfinance_ticker("BRK.B"), "BRK-B")

    def test_blank_action_column_still_requires_historical_backfill(self) -> None:
        frame = pd.DataFrame(
            {
                "Dividends": [0.0, pd.NA],
                "Stock Splits": [0.0, 0.0],
            }
        )

        self.assertTrue(needs_stock_history_backfill(frame))

    def test_blank_close_also_requires_historical_backfill(self) -> None:
        frame = pd.DataFrame(
            {
                column: [1.0, 1.0]
                for column in YFINANCE_STOCK_COLUMNS
                if column != "Date"
            }
        )
        frame.loc[0, "Close"] = pd.NA

        self.assertTrue(needs_stock_history_backfill(frame))

    def test_full_history_download_starts_at_earliest_retained_date(self) -> None:
        frame = pd.DataFrame({"Date": ["2024-04-01", "2026-06-26"]})

        start = stock_download_start(
            frame,
            date(2021, 6, 27),
            date(2026, 6, 27),
            full_history=True,
        )

        self.assertEqual(start, date(2024, 4, 1))

    def test_full_overwrite_flag(self) -> None:
        self.assertTrue(parse_args(["--full-overwrite"]).full_overwrite)


if __name__ == "__main__":
    unittest.main()
