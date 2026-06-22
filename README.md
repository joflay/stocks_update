# Stocks Update

Refresh local stock CSVs from yfinance and the 3-month Treasury risk-free-rate CSV from FRED.

## Setup

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Add your FRED API key to `.env`:

```bash
Fred_Key=your_fred_api_key
```

## Run

From the project directory, run:

```bash
.venv/bin/python scripts/refresh_stock_data.py
```

The script reads and writes data under `/srv/data`:

- `/srv/data/stocks/*_stock_data.csv`
- `/srv/data/risk_free_rate/DGS3MO_risk_free_rate.csv`

## Cron

An example cron entry is available in `cron.example`. It runs the refresh every morning at 07:15 UTC and appends logs to `refresh_stock_data.log`.
