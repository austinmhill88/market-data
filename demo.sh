#!/bin/bash
# Demonstration of the Synthetic Market Data Generator

echo "=========================================="
echo "Synthetic Market Data Generator Demo"
echo "=========================================="
echo ""

echo "1. Generating 1 year of daily data for AAPL..."
python generate_market_data.py --symbols AAPL --timeframe 1Day --days 365
echo ""

echo "2. Showing the data was created..."
ls -lh data/parquet/AAPL/
echo ""

echo "3. Validating the generated data..."
python verify_data.py
echo ""

echo "4. Demonstrating randomization - running again..."
python generate_market_data.py --symbols AAPL --timeframe 1Day --days 365
echo ""

echo "5. Reading the data with Python..."
python << 'PYTHON'
import pandas as pd

df = pd.read_parquet('data/parquet/AAPL/1Day.parquet')
print("\nData shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nLast 5 rows:")
print(df.tail())
print("\nPrice statistics:")
print(df[['open', 'high', 'low', 'close']].describe())
print("\nData is ready for backtesting!")
PYTHON

echo ""
echo "=========================================="
echo "Demo Complete!"
echo "=========================================="
