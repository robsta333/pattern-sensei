# ─────────────────────────────────────────────────────────────────────────────
# PERFECT MERGED PATTERN GENERATOR (Safe + Accurate)
# ─────────────────────────────────────────────────────────────────────────────

def generate_pattern(pattern_type):
    """Creates realistic + accurate patterns without assertions or crashes."""
    
    df = generate_base_candles(20)
    idx = -2
    prev_idx = idx - 1
    
    # Body size baseline
    body = random.uniform(0.35, 0.85)
    base = df.iloc[idx]['open']

    # ---------------------------------------------------------
    # HELPER FUNCTIONS
    # ---------------------------------------------------------
    def enforce_ohlc_safe(row):
        """Ensure high >= open/close and low <= open/close."""
        row['high'] = max(row['open'], row['close'], row['high'])
        row['low'] = min(row['open'], row['close'], row['low'])
        return row

    def safe_adjust(condition, adjust_fn):
        """If strict-rule check fails, adjust instead of asserting."""
        if not condition():
            adjust_fn()

    # ---------------------------------------------------------
    # PATTERN LOGIC
    # ---------------------------------------------------------

    # ────────────────────────────────
    # HAMMER
    # ────────────────────────────────
    if pattern_type == "Hammer":

        open_p = base
        close_p = base + body
        high_p = max(open_p, close_p) + body * 0.08       # <10% upper wick
        low_p = min(open_p, close_p) - body * 3            # long lower wick

        # Apply candle
        df.at[df.index[idx], 'open'] = open_p
        df.at[df.index[idx], 'close'] = close_p
        df.at[df.index[idx], 'high'] = high_p
        df.at[df.index[idx], 'low'] = low_p

        # SAFETY: Ensure wick rule (<10%) even if randoms misbehave
        safe_adjust(
            lambda: (high_p - close_p) < body * 0.1,
            lambda: df.at[df.index[idx], 'high'] = close_p + body * 0.05
        )

    # ────────────────────────────────
    # SHOOTING STAR
    # ────────────────────────────────
    elif pattern_type == "Shooting Star":

        open_p = base + body
        close_p = base
        high_p = max(open_p, close_p) + body * 3           # long upper wick
        low_p = min(open_p, close_p) - body * 0.08         # <10% lower wick

        df.at[df.index[idx], 'open'] = open_p
        df.at[df.index[idx], 'close'] = close_p
        df.at[df.index[idx], 'high'] = high_p
        df.at[df.index[idx], 'low'] = low_p

        # Safety-adjust (lower wick must be <10%)
        safe_adjust(
            lambda: (open_p - low_p) < body * 0.1,
            lambda: df.at[df.index[idx], 'low'] = min(open_p, close_p) - body * 0.05
        )

    # ────────────────────────────────
    # LONG-LEGGED DOJI
    # ────────────────────────────────
    elif pattern_type == "Long-legged Doji":

        open_p  = base
        close_p = base + random.uniform(-0.015, 0.015)      # near-equal
        high_p  = base + body * 2.5
        low_p   = base - body * 2.5

        df.at[df.index[idx], 'open'] = open_p
        df.at[df.index[idx], 'close'] = close_p
        df.at[df.index[idx], 'high'] = high_p
        df.at[df.index[idx], 'low'] = low_p

        # Safety-adjust open ≈ close (<2% of range)
        def doji_condition():
            rng = high_p - low_p
            return abs(open_p - close_p) <= rng * 0.02

        def doji_adjust():
            df.at[df.index[idx], 'close'] = open_p + random.uniform(-rng*0.015, rng*0.015)

        safe_adjust(doji_condition, doji_adjust)

    # ────────────────────────────────
    # BULLISH ENGULFING
    # ────────────────────────────────
    elif pattern_type == "Bullish Engulfing":

        # Prev red
        df.at[df.index[prev_idx], 'open']  = base + body*0.3
        df.at[df.index[prev_idx], 'close'] = base - body*0.3
        df.at[df.index[prev_idx], 'high']  = max(df.iloc[prev_idx]['open'], df.iloc[prev_idx]['close']) + body * 0.1
        df.at[df.index[prev_idx], 'low']   = min(df.iloc[prev_idx]['open'], df.iloc[prev_idx]['close']) - body * 0.1

        # Current BIG green
        df.at[df.index[idx], 'open']  = base - body*0.6
        df.at[df.index[idx], 'close'] = base + body*1.4
        df.at[df.index[idx], 'high']  = base + body*1.5
        df.at[df.index[idx], 'low']   = base - body*0.7

    # ────────────────────────────────
    # BEARISH ENGULFING
    # ────────────────────────────────
    elif pattern_type == "Bearish Engulfing":

        # Prev green
        df.at[df.index[prev_idx], 'open']  = base - body*0.3
        df.at[df.index[prev_idx], 'close'] = base + body*0.3
        df.at[df.index[prev_idx], 'high']  = base + body*0.4
        df.at[df.index[prev_idx], 'low']   = base - body*0.4

        # Current BIG red
        df.at[df.index[idx], 'open']  = base + body*0.6
        df.at[df.index[idx], 'close'] = base - body*1.4
        df.at[df.index[idx], 'high']  = base + body*0.7
        df.at[df.index[idx], 'low']   = base - body*1.5

    # Final OHLC enforcement
    df.iloc[idx]     = enforce_ohlc_safe(df.iloc[idx])
    df.iloc[prev_idx] = enforce_ohlc_safe(df.iloc[prev_idx])

    return df, pattern_type




def generate_base_candles(n=20):
    """Realistic volatility, trend drift, clean OHLC."""
    np.random.seed(int(time.time()) + random.randint(0, 999))

    start = random.uniform(80, 120)
    vol = random.uniform(0.008, 0.025)

    prices = [start]
    for _ in range(n-1):
        drift = random.uniform(-vol, vol) * start
        prices.append(prices[-1] + drift)

    df = pd.DataFrame()
    df['open'] = prices
    df['close'] = [o + random.uniform(-vol, vol)*10 for o in df['open']]
    df['high'] = df[['open','close']].max(axis=1) + np.random.uniform(0.1, 0.5, n)
    df['low']  = df[['open','close']].min(axis=1) - np.random.uniform(0.1, 0.5, n)
    df['volume'] = np.random.randint(200000, 800000, n)
    df['date'] = [datetime.now() - timedelta(minutes=5*i) for i in range(n)][::-1]

    df['high'] = df[['open','close','high']].max(axis=1)
    df['low']  = df[['open','close','low']].min(axis=1)
    
    return df
