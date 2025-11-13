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
        "desc": "Small body, LONG lower wick (3x+ body), little upper wick",
        "bias": "Bullish",
        "stop_loss": "Below wick low - 1% buffer",
        "generator": lambda: generate_hammer()
    },
    "Shooting Star": {
        "desc": "Small body, LONG upper wick (3x+ body), little lower wick", 
        "bias": "Bearish",
        "stop_loss": "Above wick high + 1% buffer",
        "generator": lambda: generate_shooting_star()
    },
    "Doji": {
        "desc": "Open ≈ Close (within 0.5%), long wicks both sides",
        "bias": "Neutral/Indecision",
        "stop_loss": "Beyond extremes",
        "generator": lambda: generate_doji()
    },
    "Bullish Engulfing": {
        "desc": "Small RED candle fully covered by larger GREEN candle",
        "bias": "Bullish",
        "stop_loss": "Below engulfing candle low",
        "generator": lambda: generate_engulfing("bullish")
    },
    "Bearish Engulfing": {
        "desc": "Small GREEN candle fully covered by larger RED candle",
        "bias": "Bearish",
        "stop_loss": "Above engulfing candle high",
        "generator": lambda: generate_engulfing("bearish")
    }
}

@st.cache_data
def create_perfect_examples():
    """Generate textbook-perfect examples - EXAGGERATED for learning"""
    examples = {}
    now = datetime.now()
    
    dates = [now - timedelta(minutes=2), now - timedelta(minutes=1), now]
    
    # HAMMER: Clear bullish
    examples['Hammer'] = pd.DataFrame({
        'date': dates, 'open': [100.0, 100.5, 100.8], 'high': [100.5, 100.6, 101.0],
        'low': [99.5, 96.0, 100.2], 'close': [100.2, 100.4, 100.9]
    })
    
    # SHOOTING STAR: Clear bearish
    examples['Shooting Star'] = pd.DataFrame({
        'date': dates, 'open': [102.0, 102.2, 101.5], 'high': [102.5, 106.5, 101.8],
        'low': [101.8, 102.0, 101.2], 'close': [102.3, 102.1, 101.3]
    })
    
    # DOJI: Perfect cross
    examples['Doji'] = pd.DataFrame({
        'date': dates, 'open': [100.5, 100.5, 100.4], 'high': [101.5, 102.0, 100.8],
        'low': [99.5, 99.0, 100.0], 'close': [100.3, 100.5, 100.6]
    })
    
    # BULLISH ENGULFING
    examples['Bullish Engulfing'] = pd.DataFrame({
        'date': dates, 'open': [99.0, 100.2, 99.8], 'high': [100.0, 100.3, 101.5],
        'low': [98.5, 99.8, 99.5], 'close': [100.1, 99.9, 101.2]
    })
    
    # BEARISH ENGULFING
    examples['Bearish Engulfing'] = pd.DataFrame({
        'date': dates, 'open': [101.0, 100.8, 101.2], 'high': [101.5, 101.0, 101.5],
        'low': [100.5, 100.5, 99.5], 'close': [100.9, 101.0, 99.8]
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
    """EXAGGERATED hammer - unmistakable"""
    df = generate_base_candles()
    idx = -2  # Pattern at center position
    
    base_price = df.iloc[idx]['open']
    body_size = 0.2  # Tiny body
    wick_size = body_size * random.uniform(5.0, 7.0)  # VERY long wick
    
    df.at[df.index[idx], 'open'] = base_price
    df.at[df.index[idx], 'close'] = base_price + body_size  # Small green body
    df.at[df.index[idx], 'low'] = base_price - wick_size  # Dramatic lower wick
    df.at[df.index[idx], 'high'] = base_price + (body_size * 0.3)  # Minimal upper wick
    
    return df, "Hammer"

def generate_shooting_star():
    """EXAGGERATED shooting star - unmistakable"""
    df = generate_base_candles()
    idx = -2
    
    base_price = df.iloc[idx]['open']
    body_size = 0.2
    wick_size = body_size * random.uniform(5.0, 7.0)
    
    df.at[df.index[idx], 'open'] = base_price + body_size
    df.at[df.index[idx], 'close'] = base_price  # Small red body
    df.at[df.index[idx], 'high'] = base_price + wick_size  # Dramatic upper wick
    df.at[df.index[idx], 'low'] = base_price - (body_size * 0.3)  # Minimal lower wick
    
    return df, "Shooting Star"

def generate_doji():
    """EXAGGERATED doji - perfect cross"""
    df = generate_base_candles()
    idx = -2
    
    base_price = df.iloc[idx]['open']
    df.at[df.index[idx], 'open'] = base_price
    df.at[df.index[idx], 'close'] = base_price + random.uniform(-0.03, 0.03)  # Open ≈ Close
    df.at[df.index[idx], 'high'] = base_price + 1.8  # Long upper wick
    df.at[df.index[idx], 'low'] = base_price - 1.8   # Long lower wick
    
    return df, "Doji"

def generate_engulfing(bias):
    """EXAGGERATED engulfing - tiny candle vs huge candle"""
    df = generate_base_candles()
    idx = -2
    
    base_price = df.iloc[idx-1]['open']
    
    if bias == "bullish":
        # Small red candle
        df.at[df.index[idx-1], 'open'] = base_price + 0.3
        df.at[df.index[idx-1], 'close'] = base_price - 0.3
        df.at[df.index[idx-1], 'high'] = base_price + 0.4
        df.at[df.index[idx-1], 'low'] = base_price - 0.4
        
        # Massive green candle that engulfs
        df.at[df.index[idx], 'open'] = base_price - 0.5
        df.at[df.index[idx], 'close'] = base_price + 1.2
        df.at[df.index[idx], 'high'] = base_price + 1.5
        df.at[df.index[idx], 'low'] = base_price - 0.6
        
    else:
        # Small green candle
        df.at[df.index[idx-1], 'open'] = base_price - 0.3
        df.at[df.index[idx-1], 'close'] = base_price + 0.3
        df.at[df.index[idx-1], 'high'] = base_price + 0.4
        df.at[df.index[idx-1], 'low'] = base_price - 0.4
        
        # Massive red candle that engulfs
        df.at[df.index[idx], 'open'] = base_price + 0.5
        df.at[df.index[idx], 'close'] = base_price - 1.2
        df.at[df.index[idx], 'high'] = base_price + 0.6
        df.at[df.index[idx], 'low'] = base_price - 1.5
    
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
        x=df['date'], open=df['open'], high=df['high'], low=df['low'], close=df['close'],
        increasing_line_color='#26d367', decreasing_line_color='#ff4757',
        increasing_fillcolor='#26d367', decreasing_fillcolor='#ff4757'
    ))
    
    fig.update_layout(
        title="CHART ACTION - IDENTIFY THE PATTERN (Focus on highlighted candle)",
        title_font_color="#ff6b6b", title_font_size=24,
        xaxis_rangeslider_visible=False, template='plotly_dark', height=500,
        margin=dict(l=20, r=20, t=80, b=20), paper_bgcolor='#1a1a2e', plot_bgcolor='#16213e'
    )
    
    # Highlight the pattern candle (center of focus zone)
    pattern_date = df['date'].iloc[-2]
    fig.add_vrect(
        x0=pattern_date, x1=pattern_date,
        fillcolor="rgba(255,255,0,0.25)", layer="below", line_width=3,
        line_color="yellow", annotation_text="PATTERN CANDLE", 
        annotation_position="top"
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
    
    st.markdown("""<style>@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700&display=swap');body{background-color:#1a1a2e;color:white;}.stButton>button{font-size:20px;font-weight:bold;background:linear-gradient(90deg,#ff6b6b 0%,#feca57 100%);color:black;border-radius:10px;padding:15px 30px;font-family:'Orbitron',sans-serif;}.scoreboard{font-family:'Orbitron',sans-serif;font-size:28px;color:#feca57;text-align:center;padding:20px;background:rgba(255,255,255,0.1);border-radius:15px;}.timer{font-size:48px;color:#ff6b6b;font-weight:bold;text-align:center;}</style>""", unsafe_allow_html=True)
    
    initialize_session()
    
    # CHEAT SHEET
    with st.sidebar:
        st.markdown("## 📖 PATTERN CHEAT SHEET")
        examples = create_perfect_examples()
        with st.expander("CLICK TO EXPAND", expanded=False):
            st.warning("⚠️ Using cheat sheet = -30% score penalty")
            
            for pattern_name in PATTERNS.keys():
                st.markdown("---")
                
                bias_color = "#26d367" if PATTERNS[pattern_name]['bias'] == "Bullish" else "#ff4757"
                if PATTERNS[pattern_name]['bias'] == "Neutral":
                    bias_color = "#feca57"
                
                st.markdown(f"<h4 style='color:{bias_color}'>{pattern_name.upper()}</h4>", unsafe_allow_html=True)
                
                fig = go.Figure(data=go.Candlestick(
                    x=examples[pattern_name]['date'],
                    open=examples[pattern_name]['open'],
                    high=examples[pattern_name]['high'],
                    low=examples[pattern_name]['low'],
                    close=examples[pattern_name]['close'],
                    increasing_line_color='#26d367', decreasing_line_color='#ff4757'
                ))
                
                pattern_date = examples[pattern_name]['date'].iloc[1]
                fig.add_vrect(
                    x0=pattern_date, x1=pattern_date,
                    fillcolor="rgba(255,255,0,0.3)", line_width=3, line_color="yellow"
                )
                
                fig.add_annotation(
                    x=pattern_date, y=examples[pattern_name]['high'].iloc[1] + 0.8,
                    text="PATTERN", showarrow=True, arrowhead=2, arrowcolor="yellow",
                    font=dict(color="yellow", size=14)
                )
                
                fig.update_layout(height=120, margin=dict(l=10, r=10, t=10, b=10), 
                                 xaxis_visible=False, yaxis_visible=False,
                                 template='plotly_dark', paper_bgcolor='#1a1a2e', plot_bgcolor='#16213e')
                
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                
                st.markdown(f"**Visual:** {PATTERNS[pattern_name]['desc']}")
                st.markdown(f"**Bias:** <span style='color:{bias_color}'>{PATTERNS[pattern_name]['bias']}</span>", unsafe_allow_html=True)
                st.markdown(f"**Stop:** {PATTERNS[pattern_name]['stop_loss']}")
    
    # MAIN GAME
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        st.markdown(f'<div class="scoreboard">💰 ACCOUNT<br>${st.session_state.account:,.2f}</div>', unsafe_allow_html=True)
    with col2:
        st.markdown("""<h1 style='text-align:center;color:#ff6b6b;font-family:Orbitron'>PATTERN SENSEI</h1>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="scoreboard">⚡ SCORE<br>{st.session_state.score}</div>', unsafe_allow_html=True)
    
    st.markdown(f"<h3 style='text-align:center;color:#48dbfb'>Phase {st.session_state.phase} - {['Initiate','Apprentice','Journeyman'][st.session_state.phase-1].upper()}</h3>", unsafe_allow_html=True)
    
    if not st.session_state.game_active:
        if st.button("🎯 START TRADE", key="start", use_container_width=True):
            pattern_name = random.choice(list(PATTERNS.keys()))
            df, pattern = PATTERNS[pattern_name]['generator']()
            st.session_state.current_chart = df
            st.session_state.correct_pattern = pattern
            st.session_state.start_time = time.time()
            st.session_state.game_active = True
            st.rerun()
    else:
        elapsed = time.time() - st.session_state.start_time
        remaining = max(0, st.session_state.time_limit - elapsed)
        if remaining > 0:
            st.markdown(f'<div class="timer">{remaining:.1f}s</div>', unsafe_allow_html=True)
            fig = draw_chart(st.session_state.current_chart)
            st.plotly_chart(fig, use_container_width=True, key="chart")
            st.markdown("### 🎯 SELECT PATTERN")
            cols = st.columns(3)
            pattern_options = list(PATTERNS.keys())
            for i, pattern in enumerate(pattern_options):
                with cols[i % 3]:
                    if st.button(pattern, key=f"pat_{pattern}", use_container_width=True):
                        handle_submission(pattern, remaining)
        else:
            st.error("⏰ TIME'S UP! Pattern was: " + st.session_state.correct_pattern)
            st.session_state.streak = 0
            st.session_state.game_active = False
            if st.button("NEXT CHART", use_container_width=True):
                st.rerun()

if __name__ == "__main__":
    main()
