import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import random

# ─────────────────────────────────────────────────────────────────────────────
# PATTERN DEFINITIONS - CORRECT & UNAMBIGUOUS
# ─────────────────────────────────────────────────────────────────────────────

PATTERNS = {
    "Hammer": {
        "desc": "Small GREEN body at TOP, LONG lower wick (5x+ body), tiny upper wick",
        "bias": "Bullish",
        "stop_loss": "Below wick low",
        "generator": lambda: generate_pattern("Hammer")
    },
    "Shooting Star": {
        "desc": "Small RED body at BOTTOM, LONG upper wick (5x+ body), tiny lower wick", 
        "bias": "Bearish",
        "stop_loss": "Above wick high",
        "generator": lambda: generate_pattern("Shooting Star")
    },
    "Doji": {
        "desc": "Open ≈ Close (virtually same price), long wicks BOTH directions",
        "bias": "Neutral/Indecision",
        "stop_loss": "Beyond both wicks",
        "generator": lambda: generate_pattern("Doji")
    },
    "Bullish Engulfing": {
        "desc": "GREEN candle body COMPLETELY covers previous RED candle body",
        "bias": "Bullish",
        "stop_loss": "Below green candle low",
        "generator": lambda: generate_pattern("Bullish Engulfing")
    },
    "Bearish Engulfing": {
        "desc": "RED candle body COMPLETELY covers previous GREEN candle body",
        "bias": "Bearish",
        "stop_loss": "Above red candle high",
        "generator": lambda: generate_pattern("Bearish Engulfing")
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# SINGLE PATTERN GENERATOR FOR BOTH CHEAT SHEET & TEST
# ─────────────────────────────────────────────────────────────────────────────

def generate_pattern(pattern_type):
    """EXACT SAME LOGIC for cheat sheet and test - no inconsistencies"""
    df = generate_base_candles(20)
    idx = -2  # Pattern is ALWAYS the 2nd to last candle (center of focus)
    
    base_price = df.iloc[idx]['open']
    
    if pattern_type == "Hammer":
        # Hammer: green body at top, huge lower wick
        body_size = 0.2
        df.at[df.index[idx], 'open'] = base_price
        df.at[df.index[idx], 'close'] = base_price + body_size
        df.at[df.index[idx], 'low'] = base_price - (body_size * 6)  # Huge lower wick
        df.at[df.index[idx], 'high'] = base_price + (body_size * 0.2)  # Tiny upper wick
    
    elif pattern_type == "Shooting Star":
        # Shooting star: red body at bottom, huge upper wick
        body_size = 0.2
        df.at[df.index[idx], 'open'] = base_price + body_size
        df.at[df.index[idx], 'close'] = base_price
        df.at[df.index[idx], 'high'] = base_price + (body_size * 6)  # Huge upper wick
        df.at[df.index[idx], 'low'] = base_price - (body_size * 0.2)  # Tiny lower wick
    
    elif pattern_type == "Doji":
        # Doji: open = close, long wicks both sides
        df.at[df.index[idx], 'open'] = base_price
        df.at[df.index[idx], 'close'] = base_price + random.uniform(-0.05, 0.05)
        df.at[df.index[idx], 'high'] = base_price + 2.0
        df.at[df.index[idx], 'low'] = base_price - 2.0
    
    elif pattern_type == "Bullish Engulfing":
        # Previous candle: small red
        df.at[df.index[idx-1], 'open'] = base_price + 0.3
        df.at[df.index[idx-1], 'close'] = base_price - 0.3
        df.at[df.index[idx-1], 'high'] = base_price + 0.4
        df.at[df.index[idx-1], 'low'] = base_price - 0.4
        
        # Current candle: big green that engulfs
        df.at[df.index[idx], 'open'] = base_price - 0.5
        df.at[df.index[idx], 'close'] = base_price + 1.0
        df.at[df.index[idx], 'high'] = base_price + 1.2
        df.at[df.index[idx], 'low'] = base_price - 0.6
    
    elif pattern_type == "Bearish Engulfing":
        # Previous candle: small green
        df.at[df.index[idx-1], 'open'] = base_price - 0.3
        df.at[df.index[idx-1], 'close'] = base_price + 0.3
        df.at[df.index[idx-1], 'high'] = base_price + 0.4
        df.at[df.index[idx-1], 'low'] = base_price - 0.4
        
        # Current candle: big red that engulfs
        df.at[df.index[idx], 'open'] = base_price + 0.5
        df.at[df.index[idx], 'close'] = base_price - 1.0
        df.at[df.index[idx], 'high'] = base_price + 0.6
        df.at[df.index[idx], 'low'] = base_price - 1.2
    
    return df, pattern_type

def generate_base_candles(n=20):
    """Generate random OHLC data"""
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

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────

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

# ─────────────────────────────────────────────────────────────────────────────
# CHART DRAWING
# ─────────────────────────────────────────────────────────────────────────────

def draw_chart(df, is_cheat_sheet=False):
    fig = go.Figure(data=go.Candlestick(
        x=df['date'], open=df['open'], high=df['high'], low=df['low'], close=df['close'],
        increasing_line_color='#26d367', decreasing_line_color='#ff4757',
        increasing_fillcolor='#26d367', decreasing_fillcolor='#ff4757'
    ))
    
    if is_cheat_sheet:
        fig.update_layout(height=120, margin=dict(l=10, r=10, t=10, b=10),
                         xaxis_visible=False, yaxis_visible=False)
    else:
        fig.update_layout(
            title="CHART ACTION - <span style='color:yellow'>YELLOW LINE</span> marks pattern candle",
            title_font_color="#ff6b6b", title_font_size=24,
            xaxis_rangeslider_visible=False, height=500,
            margin=dict(l=20, r=20, t=80, b=20)
        )
    
    fig.update_layout(
        template='plotly_dark', paper_bgcolor='#1a1a2e', plot_bgcolor='#16213e'
    )
    
    # Yellow line at pattern candle
    pattern_date = df['date'].iloc[-2]
    fig.add_vline(x=pattern_date, line_width=3, line_color="yellow", opacity=0.7)
    
    return fig

def calculate_score(pattern_correct, time_remaining):
    base = 100
    pattern_mult = 1.0 if pattern_correct else 0
    time_bonus = 0.1 * time_remaining
    return int(base * pattern_mult * (1 + time_bonus/10))

def handle_submission(selected_pattern, time_remaining):
    correct = selected_pattern == st.session_state.correct_pattern
    st.session_state.total_trades += 1
    
    if correct:
        st.session_state.winning_trades += 1
        st.session_state.streak += 1
        
        points = calculate_score(True, time_remaining)
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

# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="Pattern Sensei", layout="wide")
    st.markdown("""<style>@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700&display=swap');body{background-color:#1a1a2e;color:white;}.stButton>button{font-size:20px;font-weight:bold;background:linear-gradient(90deg,#ff6b6b 0%,#feca57 100%);color:black;border-radius:10px;padding:15px 30px;font-family:'Orbitron',sans-serif;}.scoreboard{font-family:'Orbitron',sans-serif;font-size:28px;color:#feca57;text-align:center;padding:20px;background:rgba(255,255,255,0.1);border-radius:15px;}.timer{font-size:48px;color:#ff6b6b;font-weight:bold;text-align:center;}</style>""", unsafe_allow_html=True)
    
    initialize_session()
    
    # CHEAT SHEET
    with st.sidebar:
        st.markdown("## 📖 PATTERN CHEAT SHEET")
        
        with st.expander("CLICK TO VIEW PERFECT EXAMPLES", expanded=False):
            for pattern_name in PATTERNS.keys():
                st.divider()
                
                bias_color = "#26d367" if PATTERNS[pattern_name]['bias'] == "Bullish" else "#ff4757"
                if PATTERNS[pattern_name]['bias'] == "Neutral":
                    bias_color = "#feca57"
                
                st.markdown(f"<h4 style='color:{bias_color}'>{pattern_name.upper()}</h4>", unsafe_allow_html=True)
                
                # Generate using SAME function as test
                df, _ = generate_pattern(pattern_name)
                df = df.iloc[-5:]  # Show last 5 candles
                
                # Draw mini chart
                fig = draw_chart(df, is_cheat_sheet=True)
                
                # Add pattern label
                pattern_date = df['date'].iloc[-2]
                fig.add_annotation(
                    x=pattern_date, y=df['high'].iloc[-2] + 1,
                    text="PATTERN", showarrow=True, arrowhead=2, arrowcolor="yellow",
                    font=dict(color="yellow", size=14)
                )
                
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                
                # Show OHLC values of pattern candle for absolute clarity
                pattern_candle = df.iloc[-2]
                st.markdown(f"**Pattern Candle OHLC:**")
                st.code(f"O: {pattern_candle['open']:.2f} | H: {pattern_candle['high']:.2f} | L: {pattern_candle['low']:.2f} | C: {pattern_candle['close']:.2f}")
                
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
            df, pattern = generate_pattern(pattern_name)  # Use same generator
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
