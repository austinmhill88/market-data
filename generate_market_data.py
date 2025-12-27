#!/usr/bin/env python3
"""
Synthetic Market Data Generator

Generates realistic synthetic market data for backtesting trading strategies.
Implements all specifications from 'SPECS FOR SYNTHETIC MARKET DATA' document.
"""

import argparse
import datetime
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import sys


class MarketDataGenerator:
    """
    Generate synthetic OHLCV market data with proper validation.
    
    Features:
    - Randomized price movements using geometric Brownian motion
    - Configurable volatility and drift
    - Multiple timeframe support (1Min, 5Min, 15Min, 1Hour, 1Day)
    - NYSE trading hours for intraday data
    - Business days only
    - Full validation per specs
    """
    
    TIMEFRAMES = {
        "1Min": 1,
        "5Min": 5,
        "15Min": 15,
        "1Hour": 60,
        "1Day": 390  # NYSE regular hours: 6.5 hours = 390 minutes
    }
    
    # Scaling factors for price generation
    GAP_SCALE_FACTOR = 0.01  # 1% scaling for overnight gaps
    RANGE_SCALE_FACTOR = 0.02  # 2% scaling for intrabar range
    
    # NYSE regular trading hours (09:30 - 16:00 EST)
    MARKET_OPEN = datetime.time(14, 30)  # UTC time (EST is UTC-5, so 09:30 EST = 14:30 UTC)
    MARKET_CLOSE = datetime.time(21, 0)  # UTC time (16:00 EST = 21:00 UTC)
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the generator.
        
        Args:
            seed: Random seed for reproducibility. If None, uses a random seed.
        """
        if seed is None:
            # Use current time in microseconds for truly random seed
            seed = int(datetime.datetime.now().timestamp() * 1000000) % (2**32)
        
        self.rng = np.random.default_rng(seed)
        print(f"Initialized generator with seed: {seed}")
    
    def _generate_trading_timestamps(
        self,
        start_date: datetime.date,
        end_date: datetime.date,
        timeframe: str
    ) -> pd.DatetimeIndex:
        """
        Generate valid trading timestamps for the given timeframe.
        
        Args:
            start_date: Start date for data generation
            end_date: End date for data generation
            timeframe: One of TIMEFRAMES keys
            
        Returns:
            DatetimeIndex with valid trading timestamps in UTC
        """
        if timeframe == "1Day":
            # Generate daily bars (one per business day)
            dates = pd.bdate_range(start=start_date, end=end_date, freq='B')
            # Set to midnight UTC
            return pd.DatetimeIndex([pd.Timestamp(d, tz='UTC') for d in dates])
        
        # Generate intraday timestamps
        timestamps = []
        current_date = start_date
        
        while current_date <= end_date:
            # Skip weekends
            if current_date.weekday() < 5:  # Monday = 0, Friday = 4
                # Generate timestamps for this trading day
                day_start = pd.Timestamp(
                    year=current_date.year,
                    month=current_date.month,
                    day=current_date.day,
                    hour=self.MARKET_OPEN.hour,
                    minute=self.MARKET_OPEN.minute,
                    tz='UTC'
                )
                day_end = pd.Timestamp(
                    year=current_date.year,
                    month=current_date.month,
                    day=current_date.day,
                    hour=self.MARKET_CLOSE.hour,
                    minute=self.MARKET_CLOSE.minute,
                    tz='UTC'
                )
                
                # Generate bars for this day
                minutes = self.TIMEFRAMES[timeframe]
                current_ts = day_start
                
                while current_ts < day_end:
                    timestamps.append(current_ts)
                    current_ts += pd.Timedelta(minutes=minutes)
            
            current_date += datetime.timedelta(days=1)
        
        return pd.DatetimeIndex(timestamps)
    
    def _generate_price_series(
        self,
        n_bars: int,
        start_price: float,
        volatility: float,
        drift: float
    ) -> np.ndarray:
        """
        Generate a price series using geometric Brownian motion.
        
        Args:
            n_bars: Number of bars to generate
            start_price: Starting price
            volatility: Volatility parameter (annualized)
            drift: Drift parameter (annualized return)
            
        Returns:
            Array of prices
        """
        # Geometric Brownian Motion: dS = μS dt + σS dW
        dt = 1.0 / 252  # Daily time step (252 trading days per year)
        
        # Generate random returns
        returns = self.rng.normal(
            drift * dt,
            volatility * np.sqrt(dt),
            n_bars
        )
        
        # Convert to prices
        price_series = start_price * np.exp(np.cumsum(returns))
        
        return price_series
    
    def _generate_ohlc_from_close(
        self,
        close_prices: np.ndarray,
        volatility: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate OHLC data from close prices.
        
        Args:
            close_prices: Array of close prices
            volatility: Annualized volatility (0.0-1.0, e.g., 0.20 for 20%)
            
        Returns:
            Tuple of (open, high, low, close) arrays
        """
        n_bars = len(close_prices)
        
        # Generate open prices with some randomness
        open_prices = np.zeros(n_bars)
        open_prices[0] = close_prices[0]
        
        for i in range(1, n_bars):
            # Open is close of previous bar with small gap
            gap = self.rng.normal(0, volatility * close_prices[i-1] * self.GAP_SCALE_FACTOR)
            open_prices[i] = close_prices[i-1] + gap
        
        # Generate high and low with realistic intrabar movement
        high_prices = np.zeros(n_bars)
        low_prices = np.zeros(n_bars)
        
        for i in range(n_bars):
            # Determine range based on volatility
            range_pct = abs(self.rng.normal(0, volatility * self.RANGE_SCALE_FACTOR))
            bar_range = close_prices[i] * range_pct
            
            # High and low should encompass open and close
            max_oc = max(open_prices[i], close_prices[i])
            min_oc = min(open_prices[i], close_prices[i])
            
            # Add some randomness to high/low beyond open/close
            high_extension = self.rng.uniform(0, bar_range)
            low_extension = self.rng.uniform(0, bar_range)
            
            high_prices[i] = max_oc + high_extension
            low_prices[i] = min_oc - low_extension
            
            # Ensure low is positive
            if low_prices[i] <= 0:
                low_prices[i] = min_oc * 0.95
        
        return open_prices, high_prices, low_prices, close_prices
    
    def _generate_volume(
        self,
        n_bars: int,
        avg_volume: int,
        volatility: float
    ) -> np.ndarray:
        """
        Generate realistic volume data.
        
        Args:
            n_bars: Number of bars to generate
            avg_volume: Average volume per bar
            volatility: Volatility for volume variation
            
        Returns:
            Array of integer volumes
        """
        # Use log-normal distribution for realistic volume patterns
        volumes = self.rng.lognormal(
            mean=np.log(avg_volume),
            sigma=volatility,
            size=n_bars
        )
        
        # Convert to integers and ensure >= 0
        volumes = np.maximum(volumes.astype(np.int64), 0)
        
        return volumes
    
    def _validate_dataframe(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate the generated data against all specs.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []
        
        # Check index is DateTimeIndex (UTC), sorted, no duplicates
        if not isinstance(df.index, pd.DatetimeIndex):
            errors.append("Index is not DateTimeIndex")
        elif df.index.tz is None or str(df.index.tz) != 'UTC':
            errors.append("Index is not in UTC timezone")
        
        if not df.index.is_monotonic_increasing:
            errors.append("Index is not sorted in ascending order")
        
        if df.index.duplicated().any():
            errors.append("Index contains duplicate timestamps")
        
        # Check required columns
        required_cols = ['symbol', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
            return False, errors
        
        # Check for finite values (no NaN/Inf)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if not np.isfinite(df[col]).all():
                errors.append(f"Column '{col}' contains NaN or Inf values")
        
        # Check OHLC bounds
        if not (df['high'] >= df['low']).all():
            errors.append("high < low in some bars")
        
        if not ((df['open'] >= df['low']) & (df['open'] <= df['high'])).all():
            errors.append("open not in [low, high] for some bars")
        
        if not ((df['close'] >= df['low']) & (df['close'] <= df['high'])).all():
            errors.append("close not in [low, high] for some bars")
        
        # Check positive prices
        if not (df['low'] > 0).all():
            errors.append("low <= 0 for some bars")
        
        if not (df['open'] > 0).all():
            errors.append("open <= 0 for some bars")
        
        if not (df['close'] > 0).all():
            errors.append("close <= 0 for some bars")
        
        # Check volume is integer and >= 0
        if df['volume'].dtype not in [np.int32, np.int64]:
            errors.append("volume is not integer type")
        
        if not (df['volume'] >= 0).all():
            errors.append("volume < 0 for some bars")
        
        # Check optional columns if present
        if 'vwap' in df.columns:
            if not np.isfinite(df['vwap']).all():
                errors.append("vwap contains NaN or Inf values")
            if not ((df['vwap'] >= df['low']) & (df['vwap'] <= df['high'])).all():
                errors.append("vwap not in [low, high] for some bars")
        
        if 'trade_count' in df.columns:
            if not (df['trade_count'] >= 0).all():
                errors.append("trade_count < 0 for some bars")
        
        # Check minimum history requirements
        if len(df) < 50:
            errors.append(f"Insufficient history: {len(df)} bars < 50 minimum")
        
        return len(errors) == 0, errors
    
    def generate_data(
        self,
        symbol: str,
        start_date: datetime.date,
        end_date: datetime.date,
        timeframe: str = "1Day",
        start_price: Optional[float] = None,
        volatility: Optional[float] = None,
        drift: Optional[float] = None,
        avg_volume: Optional[int] = None,
        include_vwap: bool = False,
        include_trade_count: bool = False
    ) -> pd.DataFrame:
        """
        Generate synthetic market data for a symbol.
        
        Args:
            symbol: Stock symbol (uppercase ticker)
            start_date: Start date for data generation
            end_date: End date for data generation
            timeframe: Timeframe (1Min, 5Min, 15Min, 1Hour, 1Day)
            start_price: Starting price in dollars (randomized if None)
            volatility: Annualized volatility (0.0-1.0, e.g., 0.20 for 20%, randomized if None)
            drift: Annualized expected return (-1.0 to 1.0, e.g., 0.08 for 8%, randomized if None)
            avg_volume: Average volume per bar in shares (randomized if None)
            include_vwap: Include VWAP column
            include_trade_count: Include trade_count column
            
        Returns:
            DataFrame with synthetic market data
        """
        if timeframe not in self.TIMEFRAMES:
            raise ValueError(f"Invalid timeframe: {timeframe}. Must be one of {list(self.TIMEFRAMES.keys())}")
        
        # Randomize parameters if not provided
        if start_price is None:
            # Random price between $10 and $500
            start_price = self.rng.uniform(10.0, 500.0)
        
        if volatility is None:
            # Random volatility between 15% and 60% annualized
            volatility = self.rng.uniform(0.15, 0.60)
        
        if drift is None:
            # Random drift between -10% and +20% annualized
            drift = self.rng.uniform(-0.10, 0.20)
        
        if avg_volume is None:
            # Random average volume based on price (inverse relationship common in markets)
            if start_price < 50:
                avg_volume = int(self.rng.uniform(500000, 5000000))
            elif start_price < 200:
                avg_volume = int(self.rng.uniform(200000, 2000000))
            else:
                avg_volume = int(self.rng.uniform(50000, 1000000))
        
        print(f"\nGenerating data for {symbol}:")
        print(f"  Start Price: ${start_price:.2f}")
        print(f"  Volatility: {volatility*100:.1f}%")
        print(f"  Drift: {drift*100:.1f}%")
        print(f"  Avg Volume: {avg_volume:,}")
        
        # Generate timestamps
        timestamps = self._generate_trading_timestamps(start_date, end_date, timeframe)
        n_bars = len(timestamps)
        
        print(f"  Generating {n_bars} bars...")
        
        if n_bars < 50:
            print(f"  WARNING: Only {n_bars} bars generated. Minimum recommended is 50.")
        
        # Generate price series
        close_prices = self._generate_price_series(n_bars, start_price, volatility, drift)
        
        # Generate OHLC from closes
        open_prices, high_prices, low_prices, close_prices = self._generate_ohlc_from_close(
            close_prices, volatility
        )
        
        # Generate volume
        volumes = self._generate_volume(n_bars, avg_volume, volatility * 0.5)
        
        # Create DataFrame
        data = {
            'symbol': [symbol.upper()] * n_bars,
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volumes
        }
        
        # Add optional columns
        if include_vwap:
            # VWAP (Volume Weighted Average Price) = sum(price * volume) / sum(volume)
            # For simplification, we approximate VWAP as volume-weighted typical price
            typical_prices = (high_prices + low_prices + close_prices) / 3.0
            # In reality, VWAP would need tick-by-tick data; this is a bar-level approximation
            data['vwap'] = typical_prices
        
        if include_trade_count:
            # Trade count roughly proportional to volume
            data['trade_count'] = (volumes / 100).astype(np.int64)
        
        df = pd.DataFrame(data, index=timestamps)
        df.index.name = 'ts'
        
        # Validate the data
        is_valid, errors = self._validate_dataframe(df)
        
        if not is_valid:
            print(f"  ERROR: Generated data failed validation:")
            for error in errors:
                print(f"    - {error}")
            raise ValueError("Generated data failed validation")
        
        print(f"  ✓ Generated {len(df)} valid bars")
        print(f"  Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
        print(f"  Final close: ${df['close'].iloc[-1]:.2f}")
        
        return df
    
    def save_to_parquet(
        self,
        df: pd.DataFrame,
        output_dir: Path,
        symbol: str,
        timeframe: str
    ):
        """
        Save DataFrame to Parquet file following convention.
        
        Args:
            df: DataFrame to save
            output_dir: Base output directory
            symbol: Stock symbol
            timeframe: Timeframe string
        """
        # Create directory structure: data/parquet/{SYMBOL}/{TIMEFRAME}.parquet
        symbol_dir = output_dir / "parquet" / symbol.upper()
        symbol_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = symbol_dir / f"{timeframe}.parquet"
        df.to_parquet(output_file, engine='pyarrow', compression='snappy')
        
        print(f"  Saved to: {output_file}")
    
    def save_to_csv(
        self,
        df: pd.DataFrame,
        output_dir: Path,
        symbol: str,
        timeframe: str
    ):
        """
        Save DataFrame to CSV file.
        
        Args:
            df: DataFrame to save
            output_dir: Base output directory
            symbol: Stock symbol
            timeframe: Timeframe string
        """
        # Create directory structure: data/csv/{SYMBOL}/{TIMEFRAME}.csv
        symbol_dir = output_dir / "csv" / symbol.upper()
        symbol_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = symbol_dir / f"{timeframe}.csv"
        df.to_csv(output_file)
        
        print(f"  Saved to: {output_file}")


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic market data for backtesting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 1 year of daily data for AAPL
  %(prog)s --symbols AAPL --timeframe 1Day --days 365
  
  # Generate 60 days of 1-minute data for multiple symbols
  %(prog)s --symbols AAPL MSFT GOOGL --timeframe 1Min --days 60
  
  # Generate data with custom parameters
  %(prog)s --symbols SPY --timeframe 1Day --days 365 --start-price 450 --volatility 0.18
  
  # Export to both Parquet and CSV
  %(prog)s --symbols AMZN --timeframe 1Hour --days 90 --format parquet csv
        """
    )
    
    parser.add_argument(
        '--symbols',
        nargs='+',
        required=True,
        help='Stock symbols to generate data for (e.g., AAPL MSFT GOOGL)'
    )
    
    parser.add_argument(
        '--timeframe',
        choices=list(MarketDataGenerator.TIMEFRAMES.keys()),
        default='1Day',
        help='Timeframe for bars (default: 1Day)'
    )
    
    parser.add_argument(
        '--days',
        type=int,
        default=365,
        help='Number of calendar days of data to generate (default: 365)'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date (YYYY-MM-DD). If not provided, uses --days from today.'
    )
    
    parser.add_argument(
        '--start-price',
        type=float,
        help='Starting price (randomized if not provided)'
    )
    
    parser.add_argument(
        '--volatility',
        type=float,
        help='Annualized volatility (0.0-1.0, randomized if not provided)'
    )
    
    parser.add_argument(
        '--drift',
        type=float,
        help='Annualized drift/return (-1.0 to 1.0, randomized if not provided)'
    )
    
    parser.add_argument(
        '--avg-volume',
        type=int,
        help='Average volume per bar (randomized if not provided)'
    )
    
    parser.add_argument(
        '--include-vwap',
        action='store_true',
        help='Include VWAP column in output'
    )
    
    parser.add_argument(
        '--include-trade-count',
        action='store_true',
        help='Include trade_count column in output'
    )
    
    parser.add_argument(
        '--format',
        nargs='+',
        choices=['parquet', 'csv'],
        default=['parquet'],
        help='Output format(s) (default: parquet)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data',
        help='Output directory for generated data (default: data)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed for reproducibility (randomized if not provided)'
    )
    
    args = parser.parse_args()
    
    # Determine date range
    if args.start_date:
        start_date = datetime.datetime.strptime(args.start_date, '%Y-%m-%d').date()
        end_date = start_date + datetime.timedelta(days=args.days)
    else:
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=args.days)
    
    # Create generator
    generator = MarketDataGenerator(seed=args.seed)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Synthetic Market Data Generator")
    print(f"{'='*60}")
    print(f"Date Range: {start_date} to {end_date}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Symbols: {', '.join(args.symbols)}")
    print(f"Output Formats: {', '.join(args.format)}")
    print(f"{'='*60}")
    
    # Generate data for each symbol
    for symbol in args.symbols:
        try:
            df = generator.generate_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                timeframe=args.timeframe,
                start_price=args.start_price,
                volatility=args.volatility,
                drift=args.drift,
                avg_volume=args.avg_volume,
                include_vwap=args.include_vwap,
                include_trade_count=args.include_trade_count
            )
            
            # Save in requested formats
            if 'parquet' in args.format:
                generator.save_to_parquet(df, output_dir, symbol, args.timeframe)
            
            if 'csv' in args.format:
                generator.save_to_csv(df, output_dir, symbol, args.timeframe)
            
            print()
        
        except Exception as e:
            print(f"\nERROR generating data for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    print(f"{'='*60}")
    print(f"✓ Successfully generated data for {len(args.symbols)} symbol(s)")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
