import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import random

# ─────────────────────────────────────────────────────────────────────────────
# PATTERN DEFINITIONS & GENERATION
# ─────────────────────────────────────────────────────────────────────────────

PATTERNS = {
    "Hammer": {
        "desc": "Small body, long lower wick (2x+ body), bullish",
        "bias": "Bullish",
        "stop_loss": "Below wick low",
        "generator": lambda: generate_hammer()
    },
    "Shooting Star": {
        "desc": "Small body, long upper wick (2x+ body), bearish",
        "bias": "Bearish", 
        "stop_loss": "Above wick high",
        "generator": lambda: generate_shooting_star()
    },
    "Doji": {
        "desc": "Open ≈ Close, indecision",
        "bias": "Neutral",
        "stop_loss": "Beyond wicks",
        "generator": lambda: generate_doji()
    },
    "Bullish Engulfing": {
        "desc": "Small red candle fully covered by large green candle",
        "bias": "Bullish",
        "stop_loss": "Below green candle low",
        "generator": lambda: generate_engulfing("bullish")
    },
    "Bearish Engulfing": {
        "desc": "Small green candle fully covered by large red candle", 
        "bias": "Bearish",
        "stop_loss": "Above red candle high",
        "generator": lambda: generate_engulfing("bearish")
    }
}

def generate_base_candles(n=20):
    """Generate realistic OHLCV data"""
    np.random.seed(int(time.time()) + random.randint(0, 1000))
    start_price = random.uniform(50, 150)
    volatility = random.uniform(0.01, 0.03)
    
    df = pd.DataFrame()
    df['open'] = [start_price + np.random.normal(0, volatility*start_price) for _ in range(n)]
    df['close'] = df['open'] + np.random.normal(0, volatility*start_price*0.5, n)
    df['high'] = df[['open', 'close']].max(axis=1) + np.random.uniform(0, volatility*start_price*0.8, n)
    df['low'] = df[['open', 'close']].min(axis=1) - np.random.uniform(0, volatility*start_price*0.8, n)
    df['volume'] = np.random.randint(100000, 1000000, n)
    df['date'] = [datetime.now() - timedelta(minutes=5*i) for i in range(n)][::-1]
    
    return df

def generate_hammer():
    df = generate_base_candles()
    idx = -3  # Pattern at end
    body = abs(df.iloc[idx]['open'] - df.iloc[idx]['close'])
    wick_size = body * random.uniform(2.5, 4)
    df.at[df.index[idx], 'low'] = min(df.iloc[idx]['open'], df.iloc[idx]['close']) - wick_size
    df.at[df.index[idx], 'high'] = max(df.iloc[idx]['open'], df.iloc[idx]['close']) + body*0.3
    return df, "Hammer"

def generate_shooting_star():
    df = generate_base_candles()
    idx = -3
    body = abs(df.iloc[idx]['open'] - df.iloc[idx]['close'])
    wick_size = body * random.uniform(2.5, 4)
    df.at[df.index[idx], 'high'] = max(df.iloc[idx]['open'], df.iloc[idx]['close']) + wick_size
    df.at[df.index[idx], 'low'] = min(df.iloc[idx]['open'], df.iloc[idx]['close']) - body*0.3
    return df, "Shooting Star"

def generate_doji():
    df = generate_base_candles()
    idx = -3
    body_range = df.iloc[idx]['open'] * 0.002
    df.at[df.index[idx], 'close'] = df.iloc[idx]['open'] + random.uniform(-body_range, body_range)
    df.at[df.index[idx], 'high'] = df.iloc[idx]['open'] + df.iloc[idx]['open']*0.01
    df.at[df.index[idx], 'low'] = df.iloc[idx]['open'] - df.iloc[idx]['open']*0.01
    return df, "Doji"

def generate_engulfing(bias):
    df = generate_base_candles()
    idx = -3
    
    if bias == "bullish":
        df.at[df.index[idx-1], 'close'] = df.iloc[idx-1]['open'] - 0.5  # Red candle
        df.at[df.index[idx], 'open'] = df.iloc[idx-1]['close'] - 0.3
        df.at[df.index[idx], 'close'] = df.iloc[idx-1]['open'] + 1.5  # Green engulfs
    else:
        df.at[df.index[idx-1], 'close'] = df.iloc[idx-1]['open'] + 0.5  # Green candle
        df.at[df.index[idx], 'open'] = df.iloc[idx-1]['close'] + 0.3
        df.at[df.index[idx], 'close'] = df.iloc[idx-1]['open'] - 1.5  # Red engulfs
        
    df.at[df.index[idx], 'high'] = max(df.iloc[idx]['open'], df.iloc[idx]['close']) + 0.2
    df.at[df.index[idx], 'low'] = min(df.iloc[idx]['open'], df.iloc[idx]['close']) - 0.2
    
    pattern = "Bullish Engulfing" if bias == "bullish" else "Bearish Engulfing"
    return df, pattern

# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT UI & GAME LOGIC
# ─────────────────────────────────────────────────────────────────────────────

def initialize_session():
    if 'score' not in st.session_state:
        st.session_state.score = 0
        st.session_state.streak = 0
        st.session_state.account' = 10000
        st.session_state.phase = 1
        st.session_state.total_trades = 0
        st.session_state.winning_trades = 0
        st.session_state.current_chart' = None
        st.session_state.correct_pattern' = None
        st.session_state.start_time' = None
        st.session_state.game_active' = False
        st.session_state.time_limit' = 15

def draw_chart(df):
    fig = go.Figure(data=go.Candlestick(
        x=df['date'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        increasing_line_color='#26d367',
        decreasing_line_color='#ff4757',
        increasing_fillcolor='#26d367',
        decreasing_fillcolor='#ff4757'
    ))
    
    fig.update_layout(
        title="CHART ACTION - IDENTIFY THE PATTERN",
        title_font_color="#ff6b6b",
        title_font_size=24,
        xaxis_rangeslider_visible=False,
        template='plotly_dark',
        height=500,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor='#1a1a2e',
        plot_bgcolor='#16213e'
    )
    
    # Highlight last 3 candles (where pattern appears)
    last_date = df['date'].iloc[-3]
    fig.add_vrect(
        x0=last_date, x1=df['date'].iloc[-1],
        fillcolor="rgba(255,107,107,0.1)",
        layer="below",
        line_width=0,
        annotation_text="FOCUS ZONE",
        annotation_position="top left"
    )
    
    return fig

def calculate_score(pattern_correct, time_remaining, risk_ratio, sl_correct):
    base = 100
    pattern_mult = 1.0 if pattern_correct else 0
    time_bonus = 0.1 * time_remaining
    risk_mult = min(risk_ratio * 0.5, 2.0)  # 1:2 R:R = 1.0, 1:3 = 1.5
    sl_mult = 1.0 if sl_correct else 0.7
    
    return int(base * pattern_mult * (1 + time_bonus/10) * risk_mult * sl_mult)

def main():
    st.set_page_config(page_title="Pattern Sensei", layout="wide", initial_sidebar_state="collapsed")
    
    # Arcade-style CSS
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700&display=swap');
    body { background-color: #1a1a2e; color: white; }
    .stButton>button { 
        font-size: 20px; 
        font-weight: bold; 
        background: linear-gradient(90deg, #ff6b6b 0%, #feca57 100%);
        color: black;
        border-radius: 10px;
        padding: 15px 30px;
        font-family: 'Orbitron', sans-serif;
    }
    .scoreboard {
        font-family: 'Orbitron', sans-serif;
        font-size: 28px;
        color: #feca57;
        text-align: center;
        padding: 20px;
        background: rgba(255,255,255,0.1);
        border-radius: 15px;
    }
    .timer {
        font-size: 48px;
        color: #ff6b6b;
        font-weight: bold;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)
    
    initialize_session()
    
    # Header
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        st.markdown(f'<div class="scoreboard">💰 ACCOUNT<br>${st.session_state.account:,.2f}</div>', unsafe_allow_html=True)
    with col2:
        st.markdown("<h1 style='text-align: center; color: #ff6b6b; font-family: Orbitron;'>PATTERN SENSEI</h1>", unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="scoreboard">⚡ SCORE<br>{st.session_state.score}</div>', unsafe_allow_html=True)
    
    # Phase indicator
    st.markdown(f"<h3 style='text-align: center; color: #48dbfb;'>Phase {st.session_state.phase} - {['Initiate', 'Apprentice', 'Journeyman'][st.session_state.phase-1].upper()}</h3>", unsafe_allow_html=True)
    
    # Game area
    if not st.session_state.game_active:
        if st.button("🎯 START TRADE", key="start"):
            # Generate new chart
            pattern_name = random.choice(list(PATTERNS.keys()))
            df, pattern = PATTERNS[pattern_name]['generator']()
            
            st.session_state.current_chart = df
            st.session_state.correct_pattern = pattern
            st.session_state.start_time = time.time()
            st.session_state.game_active = True
            st.rerun()
    
    else:
        # Show timer
        elapsed = time.time() - st.session_state.start_time
        remaining = max(0, st.session_state.time_limit - elapsed)
        
        if remaining > 0:
            st.markdown(f'<div class="timer">{remaining:.1f}s</div>', unsafe_allow_html=True)
            
            # Draw chart
            fig = draw_chart(st.session_state.current_chart)
            st.plotly_chart(fig, use_container_width=True, key="chart")
            
            # Pattern selection
            st.markdown("### 🎯 SELECT PATTERN")
            cols = st.columns(3)
            pattern_options = list(PATTERNS.keys())
            for i, pattern in enumerate(pattern_options):
                with cols[i % 3]:
                    if st.button(pattern, key=f"pat_{pattern}"):
                        handle_submission(pattern, remaining)
            
            # Risk management controls
            st.markdown("### ⚔️ RISK MANAGEMENT")
            col_r1, col_r2 = st.columns(2)
            with col_r1:
                risk_ratio = st.slider("Risk:Reward Ratio", 1.0, 3.0, 1.5, 0.1)
            with col_r2:
                sl_placement = st.radio("Stop Loss Placement", ["Correct", "Too Tight", "Too Loose"])
            
        else:
            st.error("⏰ TIME'S UP! Pattern was: " + st.session_state.correct_pattern)
            st.session_state.streak = 0
            st.session_state.game_active = False
            if st.button("NEXT CHART"):
                st.rerun()

def handle_submission(selected_pattern, time_remaining):
    correct = selected_pattern == st.session_state.correct_pattern
    st.session_state.total_trades += 1
    
    if correct:
        st.session_state.winning_trades += 1
        st.session_state.streak += 1
        
        # Calculate score (simplified for Phase 1)
        points = calculate_score(True, time_remaining, 1.5, True)
        st.session_state.score += points
        
        # Update account (2% risk, 1:2 R:R)
        profit = st.session_state.account * 0.02 * 2
        st.session_state.account += profit
        
        st.success(f"✅ CORRECT! +${profit:,.2f} | +{points} pts | Streak: {st.session_state.streak}")
    else:
        st.session_state.streak = 0
        loss = st.session_state.account * 0.02
        st.session_state.account -= loss
        st.error(f"❌ WRONG! -${loss:,.2f} | Pattern was: {st.session_state.correct_pattern}")
    
    st.session_state.game_active = False
    time.sleep(2)
    st.rerun()

if __name__ == "__main__":
    main()