#!/bin/bash
# Quick start script for generating common market data scenarios

set -e

echo "================================================================"
echo "Synthetic Market Data Generator - Quick Start"
echo "================================================================"
echo ""
echo "This script will generate common datasets for backtesting."
echo "All data will be saved to the 'data' directory."
echo ""

# Create data directory
mkdir -p data

echo "Generating datasets..."
echo ""

# 1. Major tech stocks - 1 year of daily data
echo "1. Generating 1 year of daily data for major tech stocks..."
python generate_market_data.py \
  --symbols AAPL MSFT GOOGL AMZN META TSLA NVDA \
  --timeframe 1Day \
  --days 365 \
  --format parquet csv

echo ""

# 2. SPY index - 2 years of daily data for long-term backtesting
echo "2. Generating 2 years of daily data for SPY..."
python generate_market_data.py \
  --symbols SPY \
  --timeframe 1Day \
  --days 730 \
  --format parquet csv

echo ""

# 3. Intraday data for day trading backtests - 60 days of 1-minute data
echo "3. Generating 60 days of 1-minute data for day trading (SPY, QQQ)..."
python generate_market_data.py \
  --symbols SPY QQQ \
  --timeframe 1Min \
  --days 60 \
  --include-vwap \
  --format parquet

echo ""

# 4. Multiple timeframes for the same symbol
echo "4. Generating multiple timeframes for AAPL..."
for timeframe in 1Min 5Min 15Min 1Hour 1Day; do
  echo "   - $timeframe"
  python generate_market_data.py \
    --symbols AAPL \
    --timeframe $timeframe \
    --days 90 \
    --format parquet
done

echo ""
echo "================================================================"
echo "✓ Quick start complete!"
echo "================================================================"
echo ""
echo "Generated datasets:"
echo "  - 7 tech stocks (1 year daily)"
echo "  - SPY (2 years daily)"
echo "  - SPY, QQQ (60 days 1-minute with VWAP)"
echo "  - AAPL (90 days all timeframes)"
echo ""
echo "Data location: ./data/parquet/{SYMBOL}/{TIMEFRAME}.parquet"
echo ""
echo "To verify the data:"
echo "  python verify_data.py"
echo ""
echo "To see usage examples:"
echo "  python examples.py"
echo "================================================================"
