# Synthetic Market Data Generator

A Python application for generating realistic synthetic market data for backtesting trading strategies. The generator produces OHLCV (Open, High, Low, Close, Volume) data that conforms to all specifications outlined in the `SPECS FOR SYNTHETIC MARKET DATA` document.

## Features

- ✅ **Randomized Data Generation**: Each run produces different, non-predictable data using randomized parameters
- ✅ **Multiple Timeframes**: Support for 1Min, 5Min, 15Min, 1Hour, and 1Day bars
- ✅ **Realistic Price Movement**: Uses geometric Brownian motion with regime switching and volatility clustering for natural-looking price walks
- ✅ **NYSE Trading Hours**: Respects market hours (09:30-16:00 EST) for intraday data
- ✅ **Business Days Only**: Excludes weekends and generates only valid trading days
- ✅ **Full Validation**: Ensures all data meets specifications before output with fail-fast validation
- ✅ **UTC Timestamps**: All timestamps explicitly stored as UTC with DateTimeIndex
- ✅ **Multiple Export Formats**: Supports both Parquet and CSV export
- ✅ **Configurable Parameters**: Customize starting price, volatility, drift, and volume
- ✅ **Optional Columns**: Include VWAP and trade count data

## Installation

1. Clone the repository:
```bash
git clone https://github.com/austinmhill88/market-data.git
cd market-data
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

Generate 1 year of daily data for a single symbol:

```bash
python generate_market_data.py --symbols AAPL --timeframe 1Day --days 365
```

This will create a Parquet file at `data/parquet/AAPL/1Day.parquet` with fully randomized parameters.

## Usage Examples

### Generate data for multiple symbols

```bash
python generate_market_data.py --symbols AAPL MSFT GOOGL AMZN --timeframe 1Day --days 365
```

### Generate intraday 1-minute data

```bash
python generate_market_data.py --symbols SPY --timeframe 1Min --days 60
```

### Generate data with custom starting price and volatility

```bash
python generate_market_data.py --symbols TSLA --timeframe 1Day --days 365 \
  --start-price 250.0 --volatility 0.50
```

### Export to both Parquet and CSV

```bash
python generate_market_data.py --symbols AAPL --timeframe 1Hour --days 90 \
  --format parquet csv
```

### Include optional VWAP and trade count columns

```bash
python generate_market_data.py --symbols AAPL --timeframe 1Min --days 30 \
  --include-vwap --include-trade-count
```

### Use a specific random seed for reproducibility

```bash
python generate_market_data.py --symbols AAPL --timeframe 1Day --days 365 \
  --seed 12345
```

### Specify custom date range

```bash
python generate_market_data.py --symbols AAPL --timeframe 1Day \
  --start-date 2024-01-01 --days 365
```

## Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--symbols` | Stock symbols to generate (space-separated) | Required |
| `--timeframe` | Bar timeframe: 1Min, 5Min, 15Min, 1Hour, 1Day | 1Day |
| `--days` | Number of calendar days to generate | 365 |
| `--start-date` | Start date (YYYY-MM-DD) | Today minus --days |
| `--start-price` | Starting price | Random (10-500) |
| `--volatility` | Annualized volatility (0.0-1.0) | Random (0.15-0.60) |
| `--drift` | Annualized return (-1.0 to 1.0) | Random (-0.10-0.20) |
| `--avg-volume` | Average volume per bar | Random (price-dependent) |
| `--include-vwap` | Include VWAP column | False |
| `--include-trade-count` | Include trade_count column | False |
| `--format` | Output formats: parquet, csv | parquet |
| `--output-dir` | Output directory | data |
| `--seed` | Random seed for reproducibility | Random |

## Data Specifications

The generated data conforms to all requirements in the `SPECS FOR SYNTHETIC MARKET DATA` document:

### Schema

**Required Columns:**
- `ts` (index): DateTimeIndex in UTC, sorted, no duplicates
- `symbol`: String (uppercase ticker)
- `open`: Float > 0
- `high`: Float ≥ low
- `low`: Float > 0
- `close`: Float > 0, low ≤ close ≤ high
- `volume`: Integer ≥ 0

**Optional Columns:**
- `vwap`: Float ≥ 0, within [low, high]
- `trade_count`: Integer ≥ 0

### Validation Rules

All generated data is validated to ensure:
- ✅ Timestamps are in UTC and monotonically increasing
- ✅ No duplicate timestamps
- ✅ All numeric values are finite (no NaN/Inf)
- ✅ OHLC consistency: high ≥ low, open/close within [low, high]
- ✅ All prices are positive
- ✅ Volume is non-negative integer
- ✅ Minimum 50 bars for indicator calculations

### Trading Calendar

- **Intraday data** (1Min, 5Min, 15Min, 1Hour): NYSE regular hours only (09:30-16:00 EST)
- **Daily data** (1Day): One bar per business day (weekdays only)
- **Timezone**: All timestamps in UTC

## Randomization Strategy

The generator ensures non-predictable data through:

1. **Random Seed**: Uses microsecond-precision timestamp by default for unique seeds
2. **Randomized Parameters**: If not specified, all parameters are randomized:
   - Starting price: $10 - $500
   - Volatility: 15% - 60% annualized
   - Drift: -10% to +20% annualized
   - Volume: Price-dependent ranges
3. **Geometric Brownian Motion**: Natural price evolution with random walk
4. **Regime Switching**: Volatility switches between low, medium, and high volatility regimes to simulate market conditions (calm, normal, stressed)
5. **Volatility Clustering**: GARCH-like volatility clustering where periods of high volatility tend to cluster together, creating realistic market behavior
6. **Stochastic Volume**: Log-normal distribution for realistic volume patterns

## Output Structure

The generator follows a consistent file convention for organizing market data:

### File Convention

**Standard path format:** `data/parquet/{SYMBOL}/{TIMEFRAME}.parquet`

This is the expected format for local file feeds and ensures consistent data organization across the application.

### Parquet Format (Default)
```
data/
└── parquet/
    ├── AAPL/
    │   ├── 1Min.parquet
    │   ├── 5Min.parquet
    │   ├── 1Hour.parquet
    │   ├── 1Day.parquet
    │   └── ...
    ├── MSFT/
    │   └── 1Day.parquet
    └── ...
```

**Example paths:**
- `data/parquet/AAPL/1Day.parquet` - Daily data for AAPL
- `data/parquet/SPY/1Min.parquet` - 1-minute intraday data for SPY
- `data/parquet/MSFT/1Hour.parquet` - Hourly data for MSFT

### CSV Format
```
data/
└── csv/
    ├── AAPL/
    │   └── 1Day.csv
    └── ...
```

**Note:** All timestamps in output files are stored in UTC timezone with the index as a DateTimeIndex. This ensures compatibility with pandas and consistent time handling across different systems.

## Example Output

```
============================================================
Synthetic Market Data Generator
============================================================
Date Range: 2024-01-01 to 2025-01-01
Timeframe: 1Day
Symbols: AAPL
Output Formats: parquet
============================================================
Initialized generator with seed: 1735262112650000

Generating data for AAPL:
  Start Price: $168.42
  Volatility: 28.3%
  Drift: 8.5%
  Avg Volume: 1,245,678
  Generating 253 bars...
  ✓ Generated 253 valid bars
  Price range: $142.18 - $201.35
  Final close: $195.47
  Saved to: data/parquet/AAPL/1Day.parquet

============================================================
✓ Successfully generated data for 1 symbol(s)
============================================================
```

## Reading Generated Data

### Python with Pandas

```python
import pandas as pd

# Read Parquet file
df = pd.read_parquet('data/parquet/AAPL/1Day.parquet')

# Read CSV file
df = pd.read_csv('data/csv/AAPL/1Day.csv', index_col='ts', parse_dates=['ts'])

# View data
print(df.head())
print(f"Shape: {df.shape}")
print(f"Date range: {df.index[0]} to {df.index[-1]}")
```

### Calculate Returns

```python
# Calculate daily returns
df['returns'] = df['close'].pct_change()

# Calculate indicators (example: 20-day SMA)
df['sma_20'] = df['close'].rolling(window=20).mean()
```

## Validation

The generator includes comprehensive validation to ensure data quality:

```python
from generate_market_data import MarketDataGenerator

# Create generator
generator = MarketDataGenerator()

# Generate and validate
df = generator.generate_data(
    symbol='AAPL',
    start_date=datetime.date(2024, 1, 1),
    end_date=datetime.date(2025, 1, 1),
    timeframe='1Day'
)

# Validation is automatic - will raise error if data is invalid
```

## Minimum Requirements

The generator ensures sufficient data for common trading indicators:

- **Minimum 50 bars** required (supports EMA calculations)
- **Recommended 60+ trading days** for 1Min data (for robust feature generation)
- **Recommended 60+ bars** for daily data (1-3 years ideal for backtesting)

## License

This project is open source and available for use in backtesting and educational purposes.

## Support

For issues or questions, please open an issue on the GitHub repository.
