#!/usr/bin/env python3
"""
Test script to verify the refinements to the market data generator.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from generate_market_data import MarketDataGenerator
import datetime


def test_utc_enforcement():
    """Test that timestamps are explicitly stored as UTC."""
    print("\n" + "="*60)
    print("TEST 1: UTC Timestamp Enforcement")
    print("="*60)
    
    generator = MarketDataGenerator(seed=42)
    df = generator.generate_data(
        symbol='UTC_TEST',
        start_date=datetime.date(2024, 1, 1),
        end_date=datetime.date(2024, 12, 31),
        timeframe='1Day'
    )
    
    # Check index is DateTimeIndex with UTC
    assert isinstance(df.index, pd.DatetimeIndex), "Index must be DateTimeIndex"
    assert df.index.tz is not None, "Index must have timezone"
    assert str(df.index.tz) == 'UTC', "Index must be in UTC timezone"
    
    # Save and reload to verify UTC is preserved
    output_dir = Path('/tmp/test_data')
    generator.save_to_parquet(df, output_dir, 'UTC_TEST', '1Day')
    
    df_loaded = pd.read_parquet(output_dir / 'parquet' / 'UTC_TEST' / '1Day.parquet')
    assert isinstance(df_loaded.index, pd.DatetimeIndex), "Loaded index must be DateTimeIndex"
    assert str(df_loaded.index.tz) == 'UTC', "Loaded index must preserve UTC timezone"
    
    print("✓ Timestamps are explicitly stored as UTC")
    print("✓ DateTimeIndex is preserved after save/load")
    print("✓ UTC timezone is preserved in Parquet files")
    

def test_validation_before_save():
    """Test that validation occurs before saving and fails fast."""
    print("\n" + "="*60)
    print("TEST 2: Fail-Fast Validation Before Save")
    print("="*60)
    
    generator = MarketDataGenerator(seed=42)
    df = generator.generate_data(
        symbol='VALID_TEST',
        start_date=datetime.date(2024, 1, 1),
        end_date=datetime.date(2024, 12, 31),
        timeframe='1Day'
    )
    
    # Verify all data is valid
    assert np.isfinite(df[['open', 'high', 'low', 'close']].values).all(), "All prices must be finite"
    assert (df['high'] >= df['low']).all(), "High must be >= Low"
    assert ((df['open'] >= df['low']) & (df['open'] <= df['high'])).all(), "Open must be in [low, high]"
    assert ((df['close'] >= df['low']) & (df['close'] <= df['high'])).all(), "Close must be in [low, high]"
    assert (df[['open', 'high', 'low', 'close']] > 0).all().all(), "All prices must be positive"
    
    # Test that save succeeds with valid data
    output_dir = Path('/tmp/test_data')
    try:
        generator.save_to_parquet(df, output_dir, 'VALID_TEST', '1Day')
        print("✓ Valid data passes pre-save validation")
    except ValueError as e:
        raise AssertionError(f"Valid data should not fail validation: {e}")
    
    # Test that invalid data fails fast
    df_invalid = df.copy()
    df_invalid.loc[df_invalid.index[0], 'high'] = df_invalid.loc[df_invalid.index[0], 'low'] - 1  # Make high < low
    
    try:
        generator.save_to_parquet(df_invalid, output_dir, 'INVALID_TEST', '1Day')
        raise AssertionError("Invalid data should fail validation")
    except ValueError as e:
        if "high < low" in str(e):
            print("✓ Invalid data fails fast before save with detailed error")
        else:
            raise AssertionError(f"Wrong error message: {e}")


def test_market_calendar():
    """Test that 1Min data respects market calendar (no weekends, NYSE hours)."""
    print("\n" + "="*60)
    print("TEST 3: Market Calendar for 1Min Data")
    print("="*60)
    
    generator = MarketDataGenerator(seed=42)
    
    # Generate 10 days of 1Min data (should span ~2 weeks with weekends excluded)
    df = generator.generate_data(
        symbol='CALENDAR_TEST',
        start_date=datetime.date(2024, 12, 16),  # Monday
        end_date=datetime.date(2024, 12, 27),    # Friday
        timeframe='1Min'
    )
    
    # Check no weekends (Saturday=5, Sunday=6)
    weekdays = df.index.dayofweek.unique()
    assert not any(d >= 5 for d in weekdays), "No weekend days should be present"
    print(f"✓ No weekends in data (weekdays present: {sorted(weekdays)})")
    
    # Check NYSE hours (09:30-16:00 EST = 14:30-21:00 UTC)
    # First bar should be at 14:30 UTC, last bar at 20:59 UTC
    for date in df.index.date:
        day_data = df[df.index.date == date]
        first_bar = day_data.index[0]
        last_bar = day_data.index[-1]
        
        assert first_bar.hour == 14 and first_bar.minute == 30, f"First bar must be at 14:30 UTC (09:30 EST)"
        assert last_bar.hour == 20 and last_bar.minute == 59, f"Last bar must be at 20:59 UTC (15:59 EST)"
        
        # Check we have 390 bars per day (6.5 hours * 60 minutes)
        assert len(day_data) == 390, f"Should have 390 1-minute bars per day, got {len(day_data)}"
    
    print("✓ Market hours are correct (09:30-16:00 EST / 14:30-21:00 UTC)")
    print("✓ Each trading day has 390 bars (6.5 hours)")


def test_regime_switching_and_clustering():
    """Test that volatility regime switching and clustering are present."""
    print("\n" + "="*60)
    print("TEST 4: Regime Switching and Volatility Clustering")
    print("="*60)
    
    generator = MarketDataGenerator(seed=42)
    
    # Generate enough data to see regime switching
    df = generator.generate_data(
        symbol='REGIME_TEST',
        start_date=datetime.date(2022, 1, 1),
        end_date=datetime.date(2024, 12, 31),
        timeframe='1Day',
        volatility=0.30  # Base volatility
    )
    
    # Calculate rolling volatility to detect regime switching
    returns = df['close'].pct_change().dropna()
    rolling_vol = returns.rolling(window=20).std() * np.sqrt(252)  # Annualized 20-day vol
    
    # Check that volatility varies over time (evidence of regime switching)
    vol_std = rolling_vol.std()
    print(f"✓ Rolling volatility shows variation (std={vol_std:.4f})")
    assert vol_std > 0.01, "Volatility should vary over time (regime switching)"
    
    # Check for volatility clustering (autocorrelation in squared returns)
    squared_returns = returns ** 2
    autocorr = squared_returns.autocorr(lag=1)
    print(f"✓ Squared returns show autocorrelation (ρ={autocorr:.4f})")
    assert autocorr > 0, "Squared returns should be positively autocorrelated (volatility clustering)"
    
    # Show volatility regime changes
    high_vol_periods = rolling_vol > rolling_vol.quantile(0.75)
    low_vol_periods = rolling_vol < rolling_vol.quantile(0.25)
    
    print(f"✓ High volatility periods: {high_vol_periods.sum()} days")
    print(f"✓ Low volatility periods: {low_vol_periods.sum()} days")
    print(f"✓ Volatility range: {rolling_vol.min():.2%} - {rolling_vol.max():.2%}")


def test_file_convention():
    """Test that files follow the documented convention."""
    print("\n" + "="*60)
    print("TEST 5: File Convention")
    print("="*60)
    
    generator = MarketDataGenerator(seed=42)
    df = generator.generate_data(
        symbol='CONVENTION_TEST',
        start_date=datetime.date(2024, 1, 1),
        end_date=datetime.date(2024, 12, 31),
        timeframe='1Day'
    )
    
    output_dir = Path('/tmp/test_data')
    generator.save_to_parquet(df, output_dir, 'CONVENTION_TEST', '1Day')
    
    # Check file path follows convention
    expected_path = output_dir / 'parquet' / 'CONVENTION_TEST' / '1Day.parquet'
    assert expected_path.exists(), f"File should exist at {expected_path}"
    print(f"✓ File saved at expected path: {expected_path}")
    
    # Test with different timeframe
    df_min = generator.generate_data(
        symbol='CONVENTION_TEST',
        start_date=datetime.date(2024, 12, 26),
        end_date=datetime.date(2024, 12, 27),
        timeframe='1Min'
    )
    generator.save_to_parquet(df_min, output_dir, 'CONVENTION_TEST', '1Min')
    
    expected_path_min = output_dir / 'parquet' / 'CONVENTION_TEST' / '1Min.parquet'
    assert expected_path_min.exists(), f"File should exist at {expected_path_min}"
    print(f"✓ Multiple timeframes follow convention: {expected_path_min}")
    
    print("✓ Files follow data/parquet/{SYMBOL}/{TIMEFRAME}.parquet convention")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("MARKET DATA GENERATOR REFINEMENT TESTS")
    print("="*60)
    
    # Run all tests
    test_utc_enforcement()
    test_validation_before_save()
    test_market_calendar()
    test_regime_switching_and_clustering()
    test_file_convention()
    
    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED")
    print("="*60)
    print("\nAll refinements have been successfully implemented:")
    print("  1. UTC timestamps are explicitly enforced")
    print("  2. Validation occurs before save with fail-fast behavior")
    print("  3. Market calendar gaps are correct (no weekends, NYSE hours)")
    print("  4. Regime switching and volatility clustering are implemented")
    print("  5. File convention is clearly documented and followed")
    print("="*60 + "\n")
