# ─────────────────────────────────────────────────────────────────────────────
# CORRECTED PATTERN GENERATOR (Assertions Fixed)
# ─────────────────────────────────────────────────────────────────────────────

def generate_pattern(pattern_type):
    """Generate patterns with STRICT adherence to WR Trading PDF rules"""
    df = generate_base_candles(20)
    idx = -2  # Pattern forms at 2nd to last candle
    
    body_size = random.uniform(0.3, 0.8)
    base_price = df.iloc[idx]['open']
    
    if pattern_type == "Hammer":
        # FIXED: Use 1.09 instead of 1.1 to ensure < 10% wick
        df.at[df.index[idx], 'open'] = base_price
        df.at[df.index[idx], 'close'] = base_price + body_size  # Green body
        df.at[df.index[idx], 'high'] = base_price + body_size * 1.09  # <10% upper wick
        df.at[df.index[idx], 'low'] = base_price - (body_size * 3)  # Long lower wick
        # Validate: upper wick must be < 10% of body size
        assert df.iloc[idx]['high'] - max(df.iloc[idx]['open'], df.iloc[idx]['close']) < body_size * 0.1
    
    elif pattern_type == "Shooting Star":
        # FIXED: Use 0.09 instead of 0.1 to ensure < 10% wick
        df.at[df.index[idx], 'open'] = base_price + body_size
        df.at[df.index[idx], 'close'] = base_price  # Red body
        df.at[df.index[idx], 'high'] = base_price + (body_size * 3)  # Long upper wick
        df.at[df.index[idx], 'low'] = base_price - body_size * 0.09  # <10% lower wick
        # Validate: lower wick must be < 10% of body size
        assert min(df.iloc[idx]['open'], df.iloc[idx]['close']) - df.iloc[idx]['low'] < body_size * 0.1
    
    elif pattern_type == "Long-legged Doji":
        # CRITICAL: Open ≈ Close (difference < 2% of total range)
        df.at[df.index[idx], 'open'] = base_price
        df.at[df.index[idx], 'close'] = base_price + random.uniform(-0.02, 0.02)
        df.at[df.index[idx], 'high'] = base_price + body_size * 2.5
        df.at[df.index[idx], 'low'] = base_price - body_size * 2.5
        # Validate Doji: open/close difference < 2% of total range
        total_range = df.iloc[idx]['high'] - df.iloc[idx]['low']
        assert abs(df.iloc[idx]['open'] - df.iloc[idx]['close']) < total_range * 0.02
    
    elif pattern_type == "Bullish Engulfing":
        prev_idx = idx - 1
        # Previous: small red body
        df.at[df.index[prev_idx], 'open'] = base_price + body_size * 0.3
        df.at[df.index[prev_idx], 'close'] = base_price - body_size * 0.3
        # Current: big green engulfs previous body
        df.at[df.index[idx], 'open'] = base_price - body_size * 0.6
        df.at[df.index[idx], 'close'] = base_price + body_size * 1.4
        df.at[df.index[idx], 'high'] = base_price + body_size * 1.5
        df.at[df.index[idx], 'low'] = base_price - body_size * 0.7
    
    elif pattern_type == "Bearish Engulfing":
        prev_idx = idx - 1
        # Previous: small green body
        df.at[df.index[prev_idx], 'open'] = base_price - body_size * 0.3
        df.at[df.index[prev_idx], 'close'] = base_price + body_size * 0.3
        # Current: big red engulfs previous body
        df.at[df.index[idx], 'open'] = base_price + body_size * 0.6
        df.at[df.index[idx], 'close'] = base_price - body_size * 1.4
        df.at[df.index[idx], 'high'] = base_price + body_size * 0.7
        df.at[df.index[idx], 'low'] = base_price - body_size * 1.5
    
    return df, pattern_type

def generate_base_candles(n=20):
    """Generate realistic candlestick data"""
    np.random.seed(int(time.time()) + random.randint(0, 1000))
    start_price = random.uniform(80, 120)
    volatility = random.uniform(0.008, 0.025)
    
    df = pd.DataFrame()
    df['open'] = [start_price + np.random.normal(0, volatility*start_price) for _ in range(n)]
    df['close'] = df['open'] + np.random.normal(0, volatility*start_price*0.6, n)
    df['high'] = df[['open', 'close']].max(axis=1) + np.random.uniform(0, volatility*start_price*0.7, n)
    df['low'] = df[['open', 'close']].min(axis=1) - np.random.uniform(0, volatility*start_price*0.7, n)
    df['volume'] = np.random.randint(200000, 800000, n)
    df['date'] = [datetime.now() - timedelta(minutes=5*i) for i in range(n)][::-1]
    
    # Ensure OHLC consistency
    df['high'] = df[['open', 'close', 'high']].max(axis=1)
    df['low'] = df[['open', 'close', 'low']].min(axis=1)
    
    return df
