#!/usr/bin/env python3
"""
Example usage of the synthetic market data generator.
Demonstrates various use cases and data analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from generate_market_data import MarketDataGenerator
import datetime


def example_basic_usage():
    """Example 1: Basic usage with default random parameters."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Usage with Random Parameters")
    print("="*60)
    
    generator = MarketDataGenerator()
    
    # Generate 1 year of daily data
    df = generator.generate_data(
        symbol='AAPL',
        start_date=datetime.date(2024, 1, 1),
        end_date=datetime.date(2024, 12, 31),
        timeframe='1Day'
    )
    
    print("\nGenerated data shape:", df.shape)
    print("\nFirst few rows:")
    print(df.head())
    
    # Calculate some basic statistics
    print("\nPrice range:")
    print(f"  Lowest: ${df['low'].min():.2f}")
    print(f"  Highest: ${df['high'].max():.2f}")
    print(f"  Average close: ${df['close'].mean():.2f}")
    
    # Save to file
    output_dir = Path('examples')
    generator.save_to_parquet(df, output_dir, 'AAPL', '1Day')
    
    return df


def example_custom_parameters():
    """Example 2: Generate data with custom parameters."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Custom Parameters")
    print("="*60)
    
    generator = MarketDataGenerator(seed=42)  # Fixed seed for reproducibility
    
    # Generate data with specific characteristics
    df = generator.generate_data(
        symbol='VOLATILE',
        start_date=datetime.date(2024, 1, 1),
        end_date=datetime.date(2024, 12, 31),
        timeframe='1Day',
        start_price=100.0,
        volatility=0.50,  # 50% annualized volatility (very volatile)
        drift=0.10,        # 10% annualized return
        avg_volume=1000000
    )
    
    print("\nGenerated highly volatile stock data")
    print("Final price:", f"${df['close'].iloc[-1]:.2f}")
    print("Return:", f"{(df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100:.1f}%")
    
    return df


def example_intraday_data():
    """Example 3: Generate intraday data for backtesting."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Intraday 5-Minute Data")
    print("="*60)
    
    generator = MarketDataGenerator()
    
    # Generate 30 days of 5-minute data
    df = generator.generate_data(
        symbol='SPY',
        start_date=datetime.date(2024, 11, 1),
        end_date=datetime.date(2024, 11, 30),
        timeframe='5Min',
        include_vwap=True
    )
    
    print(f"\nGenerated {len(df)} 5-minute bars")
    print(f"Trading days covered: ~{len(df) / 78:.0f} days")  # ~78 5-min bars per day
    
    # Show intraday pattern
    print("\nSample of intraday data:")
    print(df.head(10))
    
    return df


def example_multiple_symbols():
    """Example 4: Generate data for portfolio backtesting."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Multiple Symbols for Portfolio")
    print("="*60)
    
    generator = MarketDataGenerator()
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    
    portfolio_data = {}
    
    for symbol in symbols:
        df = generator.generate_data(
            symbol=symbol,
            start_date=datetime.date(2024, 1, 1),
            end_date=datetime.date(2024, 12, 31),
            timeframe='1Day'
        )
        portfolio_data[symbol] = df
    
    # Create a combined price DataFrame
    combined_closes = pd.DataFrame({
        symbol: data['close'] for symbol, data in portfolio_data.items()
    })
    
    print("\nCombined closing prices:")
    print(combined_closes.head())
    
    # Calculate correlation matrix
    print("\nPrice correlation matrix:")
    print(combined_closes.corr().round(2))
    
    return portfolio_data


def example_calculate_indicators():
    """Example 5: Calculate technical indicators on generated data."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Technical Indicators")
    print("="*60)
    
    generator = MarketDataGenerator()
    
    # Generate data with enough history for indicators
    df = generator.generate_data(
        symbol='AAPL',
        start_date=datetime.date(2024, 1, 1),
        end_date=datetime.date(2024, 12, 31),
        timeframe='1Day'
    )
    
    # Calculate some common indicators
    
    # Simple Moving Averages
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    
    # Exponential Moving Averages
    df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    # Average True Range (ATR)
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR_14'] = true_range.rolling(window=14).mean()
    
    # Returns
    df['daily_return'] = df['close'].pct_change()
    
    # Volatility (20-day rolling)
    df['volatility_20'] = df['daily_return'].rolling(window=20).std() * np.sqrt(252)  # Annualized
    
    print("\nData with indicators:")
    print(df[['close', 'SMA_20', 'SMA_50', 'EMA_20', 'ATR_14', 'volatility_20']].tail(10))
    
    # Check for valid indicators (no NaN in recent data after warmup)
    recent_data = df.iloc[-30:]
    valid_indicators = recent_data[['SMA_50', 'EMA_50', 'ATR_14']].notna().all().all()
    print(f"\n✓ All indicators valid in recent data: {valid_indicators}")
    
    return df


def example_backtest_scenario():
    """Example 6: Simple backtest scenario setup."""
    print("\n" + "="*60)
    print("EXAMPLE 6: Simple Backtest Setup")
    print("="*60)
    
    generator = MarketDataGenerator()
    
    # Generate training and testing data
    train_df = generator.generate_data(
        symbol='BACKTEST',
        start_date=datetime.date(2023, 1, 1),
        end_date=datetime.date(2023, 12, 31),
        timeframe='1Day'
    )
    
    test_df = generator.generate_data(
        symbol='BACKTEST',
        start_date=datetime.date(2024, 1, 1),
        end_date=datetime.date(2024, 12, 31),
        timeframe='1Day'
    )
    
    print(f"\nTraining data: {len(train_df)} bars")
    print(f"Testing data: {len(test_df)} bars")
    
    # Simple strategy example: Golden Cross
    train_df['SMA_50'] = train_df['close'].rolling(window=50).mean()
    train_df['SMA_200'] = train_df['close'].rolling(window=200).mean()
    
    # Signal: 1 when SMA_50 > SMA_200 (bullish), -1 otherwise
    train_df['signal'] = np.where(train_df['SMA_50'] > train_df['SMA_200'], 1, -1)
    
    # Calculate strategy returns
    train_df['market_return'] = train_df['close'].pct_change()
    train_df['strategy_return'] = train_df['signal'].shift(1) * train_df['market_return']
    
    # Performance metrics (excluding warmup period)
    valid_data = train_df.dropna()
    cumulative_market = (1 + valid_data['market_return']).cumprod()
    cumulative_strategy = (1 + valid_data['strategy_return']).cumprod()
    
    print(f"\nBuy & Hold Return: {(cumulative_market.iloc[-1] - 1) * 100:.2f}%")
    print(f"Strategy Return: {(cumulative_strategy.iloc[-1] - 1) * 100:.2f}%")
    
    return train_df, test_df


if __name__ == '__main__':
    print("\n" + "="*60)
    print("SYNTHETIC MARKET DATA GENERATOR - USAGE EXAMPLES")
    print("="*60)
    
    # Run all examples
    example_basic_usage()
    example_custom_parameters()
    example_intraday_data()
    example_multiple_symbols()
    example_calculate_indicators()
    example_backtest_scenario()
    
    print("\n" + "="*60)
    print("✓ All examples completed successfully!")
    print("="*60)
    print("\nCheck the 'examples' directory for saved data files.")
    print("You can load them with: pd.read_parquet('examples/parquet/AAPL/1Day.parquet')")
    print("="*60 + "\n")
