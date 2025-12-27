#!/usr/bin/env python3
"""
Verify generated market data meets all specifications.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path


def verify_data(filepath: Path):
    """Verify a generated data file meets all specs."""
    print(f"\n{'='*60}")
    print(f"Verifying: {filepath}")
    print(f"{'='*60}")
    
    # Read the data
    if filepath.suffix == '.parquet':
        df = pd.read_parquet(filepath)
    else:
        df = pd.read_csv(filepath, index_col='ts', parse_dates=['ts'])
    
    print(f"\nShape: {df.shape}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print(f"Number of bars: {len(df)}")
    
    # Show sample data
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nLast 5 rows:")
    print(df.tail())
    
    # Show statistics
    print("\nPrice statistics:")
    print(df[['open', 'high', 'low', 'close']].describe())
    
    print("\nVolume statistics:")
    print(df['volume'].describe())
    
    # Validation checks
    print("\n" + "="*60)
    print("VALIDATION CHECKS")
    print("="*60)
    
    checks_passed = 0
    checks_total = 0
    
    # Check 1: Index is DateTimeIndex
    checks_total += 1
    if isinstance(df.index, pd.DatetimeIndex):
        print("✓ Index is DateTimeIndex")
        checks_passed += 1
    else:
        print("✗ Index is NOT DateTimeIndex")
    
    # Check 2: UTC timezone
    checks_total += 1
    if df.index.tz is not None and str(df.index.tz) == 'UTC':
        print("✓ Index is in UTC timezone")
        checks_passed += 1
    else:
        print(f"✗ Index timezone is {df.index.tz}, expected UTC")
    
    # Check 3: Sorted ascending
    checks_total += 1
    if df.index.is_monotonic_increasing:
        print("✓ Index is sorted in ascending order")
        checks_passed += 1
    else:
        print("✗ Index is NOT sorted")
    
    # Check 4: No duplicates
    checks_total += 1
    if not df.index.duplicated().any():
        print("✓ No duplicate timestamps")
        checks_passed += 1
    else:
        print(f"✗ Found {df.index.duplicated().sum()} duplicate timestamps")
    
    # Check 5: Required columns
    checks_total += 1
    required_cols = ['symbol', 'open', 'high', 'low', 'close', 'volume']
    if all(col in df.columns for col in required_cols):
        print(f"✓ All required columns present: {required_cols}")
        checks_passed += 1
    else:
        missing = [col for col in required_cols if col not in df.columns]
        print(f"✗ Missing columns: {missing}")
    
    # Check 6: Finite values
    checks_total += 1
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    all_finite = all(df[col].apply(lambda x: pd.notnull(x) and pd.api.types.is_number(x)).all() for col in numeric_cols if col in df.columns)
    if all_finite:
        print("✓ All numeric values are finite (no NaN/Inf)")
        checks_passed += 1
    else:
        print("✗ Found NaN or Inf in numeric columns")
    
    # Check 7: High >= Low
    checks_total += 1
    if (df['high'] >= df['low']).all():
        print("✓ high >= low for all bars")
        checks_passed += 1
    else:
        violations = (df['high'] < df['low']).sum()
        print(f"✗ Found {violations} bars where high < low")
    
    # Check 8: Open in [low, high]
    checks_total += 1
    violations_open = ~((df['open'] >= df['low']) & (df['open'] <= df['high']))
    if not violations_open.any():
        print("✓ open in [low, high] for all bars")
        checks_passed += 1
    else:
        print(f"✗ Found {violations_open.sum()} bars where open not in [low, high]")
    
    # Check 9: Close in [low, high]
    checks_total += 1
    violations_close = ~((df['close'] >= df['low']) & (df['close'] <= df['high']))
    if not violations_close.any():
        print("✓ close in [low, high] for all bars")
        checks_passed += 1
    else:
        print(f"✗ Found {violations_close.sum()} bars where close not in [low, high]")
    
    # Check 10: Positive prices
    checks_total += 1
    if (df['low'] > 0).all() and (df['open'] > 0).all() and (df['close'] > 0).all():
        print("✓ All prices are positive")
        checks_passed += 1
    else:
        print("✗ Found non-positive prices")
    
    # Check 11: Volume >= 0
    checks_total += 1
    if (df['volume'] >= 0).all():
        print("✓ Volume >= 0 for all bars")
        checks_passed += 1
    else:
        print("✗ Found negative volumes")
    
    # Check 12: Minimum bars
    checks_total += 1
    if len(df) >= 50:
        print(f"✓ Sufficient history: {len(df)} bars >= 50 minimum")
        checks_passed += 1
    else:
        print(f"✗ Insufficient history: {len(df)} bars < 50 minimum")
    
    # Check optional columns
    if 'vwap' in df.columns:
        checks_total += 1
        if not (df['vwap'] >= 0).all():
            print("✗ vwap < 0 for some bars")
        elif not np.isfinite(df['vwap']).all():
            print("✗ vwap contains non-finite values")
        elif not ((df['vwap'] >= df['low']) & (df['vwap'] <= df['high'])).all():
            print("✗ vwap not in [low, high] for some bars")
        else:
            print("✓ vwap >= 0, finite, and in [low, high] for all bars")
            checks_passed += 1
    
    if 'trade_count' in df.columns:
        checks_total += 1
        # Check integer type
        if not pd.api.types.is_integer_dtype(df['trade_count']):
            print("✗ trade_count is not integer type")
        elif not (df['trade_count'] >= 0).all():
            print("✗ trade_count < 0 for some bars")
        else:
            print("✓ trade_count is integer type and >= 0 for all bars")
            checks_passed += 1
    
    # Check volume integer type
    checks_total += 1
    if not pd.api.types.is_integer_dtype(df['volume']):
        print("✗ volume is not integer type")
    else:
        print("✓ volume is integer type")
        checks_passed += 1
    
    # Summary
    print("\n" + "="*60)
    print(f"VALIDATION SUMMARY: {checks_passed}/{checks_total} checks passed")
    print("="*60)
    
    return checks_passed == checks_total


if __name__ == '__main__':
    data_dir = Path('data/parquet')
    
    if not data_dir.exists():
        print(f"Error: {data_dir} does not exist")
        sys.exit(1)
    
    # Find all parquet files
    parquet_files = list(data_dir.glob('**/*.parquet'))
    
    if not parquet_files:
        print(f"No parquet files found in {data_dir}")
        sys.exit(1)
    
    print(f"Found {len(parquet_files)} data files to verify")
    
    all_passed = True
    for filepath in parquet_files:
        passed = verify_data(filepath)
        if not passed:
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL FILES PASSED VALIDATION")
    else:
        print("✗ SOME FILES FAILED VALIDATION")
    print("="*60)
    
    sys.exit(0 if all_passed else 1)
