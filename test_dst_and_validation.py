#!/usr/bin/env python3
"""
Test script to verify DST handling, holiday exclusion, and strengthened validation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from generate_market_data import MarketDataGenerator
import datetime


def test_dst_handling():
    """Test that DST transitions are properly handled."""
    print("\n" + "="*60)
    print("TEST 1: DST Handling")
    print("="*60)
    
    generator = MarketDataGenerator(seed=42)
    
    # Generate data around DST transition (March 10, 2024)
    df = generator.generate_data(
        symbol='DST_TEST',
        start_date=datetime.date(2024, 3, 8),   # Friday before DST
        end_date=datetime.date(2024, 3, 11),    # Monday after DST
        timeframe='1Min'
    )
    
    # Get unique trading days
    unique_dates = sorted(set(df.index.date))
    
    # Check March 8 (before DST) - should be 14:30-20:59 UTC (09:30-16:00 EST)
    march_8_data = df[df.index.date == datetime.date(2024, 3, 8)]
    first_bar_before = march_8_data.index[0]
    last_bar_before = march_8_data.index[-1]
    
    assert first_bar_before.hour == 14 and first_bar_before.minute == 30, \
        f"Before DST: First bar should be 14:30 UTC, got {first_bar_before.hour}:{first_bar_before.minute}"
    assert last_bar_before.hour == 20 and last_bar_before.minute == 59, \
        f"Before DST: Last bar should be 20:59 UTC, got {last_bar_before.hour}:{last_bar_before.minute}"
    
    print(f"✓ Before DST (March 8): {first_bar_before.strftime('%H:%M UTC')} to {last_bar_before.strftime('%H:%M UTC')}")
    
    # Check March 11 (after DST) - should be 13:30-19:59 UTC (09:30-16:00 EDT)
    march_11_data = df[df.index.date == datetime.date(2024, 3, 11)]
    first_bar_after = march_11_data.index[0]
    last_bar_after = march_11_data.index[-1]
    
    assert first_bar_after.hour == 13 and first_bar_after.minute == 30, \
        f"After DST: First bar should be 13:30 UTC, got {first_bar_after.hour}:{first_bar_after.minute}"
    assert last_bar_after.hour == 19 and last_bar_after.minute == 59, \
        f"After DST: Last bar should be 19:59 UTC, got {last_bar_after.hour}:{last_bar_after.minute}"
    
    print(f"✓ After DST (March 11): {first_bar_after.strftime('%H:%M UTC')} to {last_bar_after.strftime('%H:%M UTC')}")
    
    # Both days should have 390 bars
    assert len(march_8_data) == 390, f"Should have 390 bars, got {len(march_8_data)}"
    assert len(march_11_data) == 390, f"Should have 390 bars, got {len(march_11_data)}"
    
    print("✓ Both days have 390 bars (6.5 hours * 60 minutes)")
    print("✓ DST transition handled correctly")


def test_holiday_handling():
    """Test that holidays are properly excluded."""
    print("\n" + "="*60)
    print("TEST 2: Holiday Handling")
    print("="*60)
    
    generator = MarketDataGenerator(seed=42)
    
    # Generate full year to check holidays
    df = generator.generate_data(
        symbol='HOLIDAY_TEST',
        start_date=datetime.date(2024, 1, 1),
        end_date=datetime.date(2024, 12, 31),
        timeframe='1Day'
    )
    
    dates_list = [d.date() for d in df.index]
    
    # Check major US market holidays are excluded
    holidays_to_check = [
        (datetime.date(2024, 1, 1), "New Year's Day"),
        (datetime.date(2024, 7, 4), "Independence Day"),
        (datetime.date(2024, 11, 28), "Thanksgiving"),
        (datetime.date(2024, 12, 25), "Christmas"),
    ]
    
    for holiday, name in holidays_to_check:
        assert holiday not in dates_list, f"{name} ({holiday}) should be excluded"
        print(f"✓ {name} ({holiday}) excluded")
    
    # Check that regular weekdays are included
    regular_weekday = datetime.date(2024, 1, 2)  # Tuesday
    assert regular_weekday in dates_list, "Regular weekday should be included"
    print(f"✓ Regular weekdays included")
    
    print(f"✓ Generated {len(df)} trading days (excludes holidays and weekends)")


def test_strengthened_vwap_validation():
    """Test strengthened validation for VWAP."""
    print("\n" + "="*60)
    print("TEST 3: Strengthened VWAP Validation")
    print("="*60)
    
    generator = MarketDataGenerator(seed=42)
    
    # Generate data with VWAP
    df = generator.generate_data(
        symbol='VWAP_TEST',
        start_date=datetime.date(2024, 1, 1),
        end_date=datetime.date(2024, 3, 31),
        timeframe='1Day',
        include_vwap=True
    )
    
    # Check VWAP is finite
    assert np.isfinite(df['vwap']).all(), "VWAP must be finite"
    print("✓ VWAP contains only finite values")
    
    # Check VWAP >= 0
    assert (df['vwap'] >= 0).all(), "VWAP must be >= 0"
    print("✓ VWAP >= 0 for all bars")
    
    # Check VWAP is within [low, high]
    assert ((df['vwap'] >= df['low']) & (df['vwap'] <= df['high'])).all(), \
        "VWAP must be within [low, high]"
    print("✓ VWAP within [low, high] for all bars")
    
    print("✓ All VWAP validation checks passed")


def test_integer_dtype_enforcement():
    """Test that volume and trade_count are integer types."""
    print("\n" + "="*60)
    print("TEST 4: Integer Dtype Enforcement")
    print("="*60)
    
    generator = MarketDataGenerator(seed=42)
    
    # Generate data with trade_count
    df = generator.generate_data(
        symbol='DTYPE_TEST',
        start_date=datetime.date(2024, 1, 1),
        end_date=datetime.date(2024, 3, 31),
        timeframe='1Day',
        include_trade_count=True
    )
    
    # Check volume is integer type
    assert df['volume'].dtype in [np.int32, np.int64], \
        f"volume must be integer type, got {df['volume'].dtype}"
    print(f"✓ volume has integer dtype: {df['volume'].dtype}")
    
    # Check trade_count is integer type
    assert df['trade_count'].dtype in [np.int32, np.int64], \
        f"trade_count must be integer type, got {df['trade_count'].dtype}"
    print(f"✓ trade_count has integer dtype: {df['trade_count'].dtype}")
    
    # Check both are >= 0
    assert (df['volume'] >= 0).all(), "volume must be >= 0"
    assert (df['trade_count'] >= 0).all(), "trade_count must be >= 0"
    print("✓ Both volume and trade_count are >= 0")
    
    print("✓ Integer dtype enforcement successful")


def test_timeframes_documentation():
    """Test that TIMEFRAMES mapping is properly documented."""
    print("\n" + "="*60)
    print("TEST 5: TIMEFRAMES Documentation")
    print("="*60)
    
    generator = MarketDataGenerator(seed=42)
    
    # Check that 1Day value exists but is documented
    assert "1Day" in generator.TIMEFRAMES, "1Day should be in TIMEFRAMES"
    assert generator.TIMEFRAMES["1Day"] == 390, "1Day value should be 390"
    print("✓ TIMEFRAMES includes '1Day': 390")
    
    # Verify that daily data doesn't use the 390 value for timing
    # by checking that daily bars are at midnight UTC, not spaced by 390 minutes
    df = generator.generate_data(
        symbol='TIMEFRAME_TEST',
        start_date=datetime.date(2024, 1, 1),
        end_date=datetime.date(2024, 3, 31),
        timeframe='1Day'
    )
    
    # All daily bars should be at midnight UTC
    for ts in df.index:
        assert ts.hour == 0 and ts.minute == 0, \
            f"Daily bars should be at midnight, got {ts.hour}:{ts.minute}"
    
    print("✓ Daily bars are at midnight UTC (not using 390-minute intervals)")
    print("✓ TIMEFRAMES '1Day' value is for reference only")


def test_market_calendar_integration():
    """Test that market calendar is properly integrated."""
    print("\n" + "="*60)
    print("TEST 6: Market Calendar Integration")
    print("="*60)
    
    generator = MarketDataGenerator(seed=42)
    
    # Check that generator has market calendar
    assert hasattr(generator, 'nyse_cal'), "Generator should have NYSE calendar"
    print("✓ Generator initialized with NYSE calendar")
    
    # Generate intraday data and verify no weekends
    df = generator.generate_data(
        symbol='CALENDAR_TEST',
        start_date=datetime.date(2024, 1, 1),
        end_date=datetime.date(2024, 1, 31),
        timeframe='1Hour'
    )
    
    # Check no weekend days (Saturday=5, Sunday=6)
    weekdays = df.index.dayofweek.unique()
    assert not any(d >= 5 for d in weekdays), "No weekend days should be present"
    print(f"✓ No weekends in data (weekdays: {sorted(weekdays)})")
    
    print("✓ Market calendar integration successful")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("DST HANDLING AND VALIDATION TESTS")
    print("="*60)
    
    # Run all tests
    test_dst_handling()
    test_holiday_handling()
    test_strengthened_vwap_validation()
    test_integer_dtype_enforcement()
    test_timeframes_documentation()
    test_market_calendar_integration()
    
    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED")
    print("="*60)
    print("\nAll refinements successfully implemented:")
    print("  1. DST transitions handled correctly via pandas_market_calendars")
    print("  2. Holidays automatically excluded via market calendar")
    print("  3. VWAP validation strengthened (finite, >= 0, within [low, high])")
    print("  4. Integer dtype enforced for volume and trade_count")
    print("  5. TIMEFRAMES '1Day' value documented as reference only")
    print("  6. Market calendar properly integrated for NYSE trading hours")
    print("="*60 + "\n")
