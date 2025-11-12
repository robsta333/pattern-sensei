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

# Use explicit lists instead of list comprehensions to avoid syntax errors
@st.cache_data
def create_perfect_examples():
    """Generate perfect pattern examples ONCE and cache them"""
    examples = {}
    now = datetime.now()
    
    # Explicit dates - no list comprehensions that break on copy
    dates = [now - timedelta(minutes=4), now - timedelta(minutes=3), 
             now - timedelta(minutes=2), now - timedelta(minutes=1)]
    
    # Hammer
    examples['Hammer'] = pd.DataFrame({
        'date': dates,
        'open': [100, 100, 99.8, 100],
        'high': [101, 101, 100.1, 100.3],
        'low': [99.5, 96.0, 99.9, 99.9],
        'close': [100.3, 100.8, 99.9, 100.2]
    })
    
    # Shooting Star
    examples['Shooting Star'] = pd.DataFrame({
        'date': dates,
        'open': [100, 100, 100.5, 100.2],
        'high': [101, 104.5, 101.2, 100.8],
        'low': [99.5, 99.8, 100.3, 100.0],
        'close': [100.3, 100.1, 100.4, 100.3]
    })
    
    # Doji
    examples['Doji'] = pd.DataFrame({
        'date': dates,
        'open': [100, 100, 100.5, 100.2],
        'high': [101, 101.5, 101.2, 100.8],
        'low': [99.5, 98.5, 99.8, 100.0],
        'close': [100.3, 100.01, 100.4, 100.3]
    })
    
    # Bullish Engulfing
    examples['Bullish Engulfing'] = pd.DataFrame({
        'date': dates,
        'open': [100, 100.5, 100.5, 99.5],
        'high': [100.8, 101, 100.6, 101.5],
        'low': [99.5, 100.2, 100.2, 99.0],
        'close': [100.5, 100.2, 100.2, 101.2]
    })
    
    # Bearish Engulfing
    examples['Bearish Engulfing'] = pd.DataFrame({
        'date': dates,
        'open': [100, 99.5, 99.8, 100.4],
        'high': [100.8, 100, 100.5, 100.5],
        'low': [99.5, 99.0, 99.5, 99.2],
        'close': [100.5, 100.4, 100.4, 99.5]
    })
    
    return examples

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
    idx = -3
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
        df.at[df.index[idx-1], 'close'] = df.iloc[idx-1]['open'] - 0.5
        df.at[df.index[idx], 'open'] = df.iloc[idx-1]['close'] - 0.3
        df.at[df.index[idx], 'close'] = df.iloc[idx-1]['open'] + 1.5
    else:
        df.at[df.index[idx-1], 'close'] = df.iloc[idx-1]['open'] + 0.5
        df.at[df.index[idx], 'open'] = df.iloc[idx-1]['close'] + 0.3
        df.at[df.index[idx], 'close'] = df.iloc[idx-1]['open'] - 1.5
        
    df.at[df.index[idx], 'high'] = max(df.iloc[idx]['open'], df.iloc[idx]['close']) + 0.2
    df.at[df.index[idx], 'low'] = min(df.iloc[idx]['open'], df.iloc[idx]['close']) - 0.2
    
    pattern = "Bullish Engulfing" if bias == "bullish" else "Bearish Engulfing"
    return df, pattern

def initialize_session():
    if 'score' not in st.session_state:
        st.session_state.score = 0
        st.session_state.streak = 0
        st.session_state.account = 10000
        st.session_state.phase = 1
        st.session_state.total_trades = 0
        st.session_state.winning_trades = 0
        st.session_state.current_chart = None
        st.session_state.correct_pattern = None
        st.session_state.start_time = None
        st.session_state.game_active = False
        st.session_state.time_limit = 15

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
    risk_mult = min(risk_ratio * 0.5, 2.0)
    sl_mult = 1.0 if sl_correct else 0.7
    
    return int(base * pattern_mult * (1 + time_bonus/10) * risk_mult * sl_mult)

def handle_submission(selected_pattern, time_remaining):
    correct = selected_pattern == st.session_state.correct_pattern
    st.session_state.total_trades += 1
    
    if correct:
        st.session_state.winning_trades += 1
        st.session_state.streak += 1
        
        points = calculate_score(True, time_remaining, 1.5, True)
        st.session_state.score += points
        
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

def main():
    st.set_page_config(page_title="Pattern Sensei", layout="wide")
    
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
    
    # ──────────────────────────────────────────────────────────────
    # COLLAPSIBLE CHEAT SHEET (CACHED)
    # ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 📖 PATTERN CHEAT SHEET")
        
        examples = create_perfect_examples()
        
        with st.expander("CLICK TO EXPAND", expanded=False):
            st.warning("⚠️ **Using cheat sheet reduces next score by 30%**")
            
            for pattern_name in PATTERNS.keys():
                st.markdown(f"---")
                st.markdown(f"#### **{pattern_name.upper()}**")
                
                fig = go.Figure(data=go.Candlestick(
                    x=examples[pattern_name]['date'],
                    open=examples[pattern_name]['open'],
                    high=examples[pattern_name]['high'],
                    low=examples[pattern_name]['low'],
                    close=examples[pattern_name]['close'],
                    increasing_line_color='#26d367',
                    decreasing_line_color='#ff4757'
                ))
                
                fig.update_layout(
                    height=120,
                    margin=dict(l=10, r=10, t=10, b=10),
                    xaxis_visible=False,
                    yaxis_visible=False,
                    template='plotly_dark',
                    paper_bgcolor='#1a1a2e',
                    plot_bgcolor='#16213e'
                )
                
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                
                st.markdown(f"""
                **Visual:** {PATTERNS[pattern_name]['desc']}
