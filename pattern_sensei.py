import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import random

# ─────────────────────────────────────────────────────────────────────────────
# CORRECTED PATTERN DEFINITIONS (Based on WR Trading PDF)
# ─────────────────────────────────────────────────────────────────────────────

PATTERNS = {
    "Hammer": {
        "desc": "Small GREEN/RED body at TOP, LONG LOWER wick (2x+ body), little/no upper wick",
        "bias": "Bullish (at downtrend bottom)",
        "stop_loss": "Below wick low",
        "generator": lambda: generate_pattern("Hammer")
    },
    "Shooting Star": {
        "desc": "Small body at BOTTOM, LONG UPPER wick (2x+ body), little/no lower wick", 
        "bias": "Bearish (at uptrend top)",
        "stop_loss": "Above wick high",
        "generator": lambda: generate_pattern("Shooting Star")
    },
    "Long-legged Doji": {
        "desc": "Open ≈ Close (virtually SAME price), long wicks BOTH directions",
        "bias": "Neutral (indecision)",
        "stop_loss": "Beyond both wicks",
        "generator": lambda: generate_pattern("Long-legged Doji")
    },
    "Bullish Engulfing": {
        "desc": "GREEN candle body FULLY covers previous RED candle body",
        "bias": "Bullish",
        "stop_loss": "Below green candle low",
        "generator": lambda: generate_pattern("Bullish Engulfing")
    },
    "Bearish Engulfing": {
        "desc": "RED candle body FULLY covers previous GREEN candle body",
        "bias": "Bearish",
        "stop_loss": "Above red candle high",
        "generator": lambda: generate_pattern("Bearish Engulfing")
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# ENHANCED PATTERN GENERATOR WITH VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_pattern(pattern_type):
    """Generate patterns with STRICT adherence to WR Trading PDF rules"""
    df = generate_base_candles(20)
    idx = -2  # Pattern forms at 2nd to last candle
    
    body_size = random.uniform(0.3, 0.8)
    base_price = df.iloc[idx]['open']
    
    if pattern_type == "Hammer":
        # CRITICAL: Must have REAL BODY (not Doji), long LOWER shadow ONLY
        df.at[df.index[idx], 'open'] = base_price
        df.at[df.index[idx], 'close'] = base_price + body_size  # Green hammer
        df.at[df.index[idx], 'high'] = base_price + body_size * 1.1  # Tiny upper wick
        df.at[df.index[idx], 'low'] = base_price - (body_size * 3)  # Long lower wick
        # Ensure upper wick is < 10% of body size
        assert df.iloc[idx]['high'] - max(df.iloc[idx]['open'], df.iloc[idx]['close']) < body_size * 0.1
    
    elif pattern_type == "Shooting Star":
        # CRITICAL: Must have REAL BODY, long UPPER shadow ONLY
        df.at[df.index[idx], 'open'] = base_price + body_size
        df.at[df.index[idx], 'close'] = base_price  # Red star
        df.at[df.index[idx], 'high'] = base_price + (body_size * 3)  # Long upper wick
        df.at[df.index[idx], 'low'] = base_price - body_size * 0.1  # Tiny lower wick
        # Ensure lower wick is < 10% of body size
        assert min(df.iloc[idx]['open'], df.iloc[idx]['close']) - df.iloc[idx]['low'] < body_size * 0.1
    
    elif pattern_type == "Long-legged Doji":
        # CRITICAL: Open ≈ Close (difference < 1% of range), long wicks BOTH directions
        df.at[df.index[idx], 'open'] = base_price
        df.at[df.index[idx], 'close'] = base_price + random.uniform(-0.02, 0.02)
        df.at[df.index[idx], 'high'] = base_price + body_size * 2.5  # Long upper wick
        df.at[df.index[idx], 'low'] = base_price - body_size * 2.5   # Long lower wick
        # Validate Doji: open/close difference < 5% of total range
        total_range = df.iloc[idx]['high'] - df.iloc[idx]['low']
        assert abs(df.iloc[idx]['open'] - df.iloc[idx]['close']) < total_range * 0.05
    
    elif pattern_type == "Bullish Engulfing":
        prev_idx = idx - 1
        # Previous: small red body
        df.at[df.index[prev_idx], 'open'] = base_price + body_size * 0.4
        df.at[df.index[prev_idx], 'close'] = base_price - body_size * 0.4
        df.at[df.index[prev_idx], 'high'] = base_price + body_size * 0.5
        df.at[df.index[prev_idx], 'low'] = base_price - body_size * 0.5
        
        # Current: big green engulfs previous body
        df.at[df.index[idx], 'open'] = base_price - body_size * 0.8
        df.at[df.index[idx], 'close'] = base_price + body_size * 1.5
        df.at[df.index[idx], 'high'] = base_price + body_size * 1.6
        df.at[df.index[idx], 'low'] = base_price - body_size * 0.9
        
        # Validate engulfing: green body fully covers red body
        prev_body = abs(df.iloc[prev_idx]['open'] - df.iloc[prev_idx]['close'])
        curr_body = abs(df.iloc[idx]['open'] - df.iloc[idx]['close'])
        assert curr_body > prev_body * 1.5  # Ensure significant engulfing
    
    elif pattern_type == "Bearish Engulfing":
        prev_idx = idx - 1
        # Previous: small green body
        df.at[df.index[prev_idx], 'open'] = base_price - body_size * 0.4
        df.at[df.index[prev_idx], 'close'] = base_price + body_size * 0.4
        df.at[df.index[prev_idx], 'high'] = base_price + body_size * 0.5
        df.at[df.index[prev_idx], 'low'] = base_price - body_size * 0.5
        
        # Current: big red engulfs previous body
        df.at[df.index[idx], 'open'] = base_price + body_size * 0.8
        df.at[df.index[idx], 'close'] = base_price - body_size * 1.5
        df.at[df.index[idx], 'high'] = base_price + body_size * 0.9
        df.at[df.index[idx], 'low'] = base_price - body_size * 1.6
        
        # Validate engulfing: red body fully covers green body
        prev_body = abs(df.iloc[prev_idx]['open'] - df.iloc[prev_idx]['close'])
        curr_body = abs(df.iloc[idx]['open'] - df.iloc[idx]['close'])
        assert curr_body > prev_body * 1.5
    
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
        st.session_state.time_limit = 20  # Increased for analysis

# ─────────────────────────────────────────────────────────────────────────────
# CHART DRAWING
# ─────────────────────────────────────────────────────────────────────────────

def draw_chart(df, is_cheat_sheet=False):
    """Draw candlestick chart with clear pattern marking"""
    fig = go.Figure(data=go.Candlestick(
        x=df['date'], open=df['open'], high=df['high'], low=df['low'], close=df['close'],
        increasing_line_color='#26d367', decreasing_line_color='#ff4757',
        increasing_fillcolor='#26d367', decreasing_fillcolor='#ff4757',
        line=dict(width=2)
    ))
    
    if is_cheat_sheet:
        fig.update_layout(height=150, margin=dict(l=10, r=10, t=10, b=10),
                         xaxis_visible=False, yaxis_visible=False)
    else:
        fig.update_layout(
            title="📊 CHART ACTION - <span style='color:yellow'>YELLOW LINE shows pattern candle to identify</span>",
            title_font_color="#ff6b6b", title_font_size=22,
            xaxis_rangeslider_visible=False, height=500,
            margin=dict(l=20, r=20, t=100, b=20),
            xaxis=dict(color='white'), yaxis=dict(color='white')
        )
    
    fig.update_layout(
        template='plotly_dark', paper_bgcolor='#1a1a2e', plot_bgcolor='#16213e'
    )
    
    # Mark pattern candle with vertical line
    pattern_date = df['date'].iloc[-2]
    fig.add_vline(x=pattern_date, line_width=4, line_color="yellow", opacity=0.8)
    
    # Add annotation
    pattern_candle = df.iloc[-2]
    fig.add_annotation(
        x=pattern_date, y=pattern_candle['high'] + (pattern_candle['high']*0.01),
        text="PATTERN CANDLE", showarrow=True, arrowhead=2, arrowcolor="yellow",
        font=dict(color="yellow", size=12, family="Arial Black")
    )
    
    return fig

def calculate_score(pattern_correct, time_remaining):
    """Calculate score with time bonus"""
    base = 100
    pattern_mult = 1.0 if pattern_correct else 0
    time_bonus = 0.15 * time_remaining
    return int(base * pattern_mult * (1 + time_bonus/10))

def handle_submission(selected_pattern, time_remaining):
    """Process user submission"""
    correct = selected_pattern == st.session_state.correct_pattern
    st.session_state.total_trades += 1
    
    if correct:
        st.session_state.winning_trades += 1
        st.session_state.streak += 1
        
        points = calculate_score(True, time_remaining)
        st.session_state.score += points
        
        profit = st.session_state.account * 0.02 * 2
        st.session_state.account += profit
        
        st.success(f"✅ CORRECT! +${profit:,.2f} | +{points} pts | Streak: {st.session_state.streak} 🔥")
    else:
        st.session_state.streak = 0
        loss = st.session_state.account * 0.02
        st.session_state.account -= loss
        st.error(f"❌ WRONG! -${loss:,.2f} | Pattern was: **{st.session_state.correct_pattern}**")
    
    st.session_state.game_active = False
    time.sleep(2.5)
    st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="Pattern Sensei Pro", layout="wide")
    
    st.markdown("""<style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700&display=swap');
        body{background-color:#1a1a2e;color:white;}
        .stButton>button{font-size:20px;font-weight:bold;background:linear-gradient(90deg,#ff6b6b 0%,#feca57 100%);color:black;border-radius:10px;padding:15px 30px;font-family:'Orbitron',sans-serif;}
        .scoreboard{font-family:'Orbitron',sans-serif;font-size:26px;color:#feca57;text-align:center;padding:20px;background:rgba(255,255,255,0.1);border-radius:15px;border:2px solid #feca57;}
        .timer{font-size:48px;color:#ff6b6b;font-weight:bold;text-align:center;text-shadow:0 0 10px #ff6b6b;}
        .pattern-info{font-size:14px;padding:10px;background:rgba(0,0,0,0.3);border-radius:8px;margin-top:10px;}
    </style>""", unsafe_allow_html=True)
    
    initialize_session()
    
    # ENHANCED CHEAT SHEET WITH VALIDATION
    with st.sidebar:
        st.markdown("## 📖 PATTERN CHEAT SHEET")
        st.info("**Study these perfect examples before trading!**")
        
        with st.expander("CLICK FOR PERFECT EXAMPLES", expanded=False):
            for pattern_name in PATTERNS.keys():
                st.divider()
                
                bias = PATTERNS[pattern_name]['bias']
                bias_color = "#26d367" if "Bullish" in bias else "#ff4757"
                if "Neutral" in bias:
                    bias_color = "#feca57"
                
                st.markdown(f"<h4 style='color:{bias_color};margin-bottom:5px;'>{pattern_name.upper()}</h4>", unsafe_allow_html=True)
                st.markdown(f"<span style='color:{bias_color};font-size:12px;'>{bias}</span>", unsafe_allow_html=True)
                
                df, _ = generate_pattern(pattern_name)
                df = df.iloc[-6:]  # Show more context
                fig = draw_chart(df, is_cheat_sheet=True)
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                
                pattern_candle = df.iloc[-2]
                st.markdown(
                    f"<div class='pattern-info'>"
                    f"<strong>Pattern Candle OHLC:</strong><br>"
                    f"O: {pattern_candle['open']:.2f} | "
                    f"H: {pattern_candle['high']:.2f} | "
                    f"L: {pattern_candle['low']:.2f} | "
                    f"C: {pattern_candle['close']:.2f}<br>"
                    f"<strong>Rule:</strong> {PATTERNS[pattern_name]['desc']}"
                    f"</div>", 
                    unsafe_allow_html=True
                )
    
    # MAIN DASHBOARD
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        win_rate = (st.session_state.winning_trades / max(st.session_state.total_trades, 1)) * 100
        st.markdown(
            f'<div class="scoreboard">💰 ACCOUNT<br>${st.session_state.account:,.2f}<br>'
            f'<span style="font-size:16px;">Win Rate: {win_rate:.1f}%</span></div>', 
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            """<h1 style='text-align:center;color:#ff6b6b;font-family:Orbitron;margin-bottom:0;'>
            PATTERN SENSEI PRO
            </h1>""", 
            unsafe_allow_html=True
        )
        st.markdown(
            "<p style='text-align:center;color:#48dbfb;font-size:14px;margin-top:0;'>"
            "Master the patterns. Trade with precision.</p>",
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            f'<div class="scoreboard">⚡ SCORE<br>{st.session_state.score}<br>'
            f'<span style="font-size:16px;">Streak: {st.session_state.streak}</span></div>', 
            unsafe_allow_html=True
        )
    
    # PHASE INDICATOR
    phases = ['Initiate','Apprentice','Journeyman','Expert','Master']
    current_phase = min(st.session_state.phase - 1, len(phases)-1)
    st.markdown(
        f"<h3 style='text-align:center;color:#48dbfb;margin-bottom:20px;'>"
        f"Phase {st.session_state.phase} - {phases[current_phase].upper()}</h3>",
        unsafe_allow_html=True
    )
    
    # MAIN GAME LOGIC
    if not st.session_state.game_active:
        if st.button("🎯 START NEW TRADE", key="start", use_container_width=True):
            pattern_name = random.choice(list(PATTERNS.keys()))
            df, pattern = generate_pattern(pattern_name)
            st.session_state.current_chart = df
            st.session_state.correct_pattern = pattern
            st.session_state.start_time = time.time()
            st.session_state.game_active = True
            st.rerun()
    else:
        elapsed = time.time() - st.session_state.start_time
        remaining = max(0, st.session_state.time_limit - elapsed)
        
        if remaining > 0:
            # Display timer
            st.markdown(f'<div class="timer">{remaining:.1f}s</div>', unsafe_allow_html=True)
            
            # Display chart
            fig = draw_chart(st.session_state.current_chart)
            st.plotly_chart(fig, use_container_width=True, key="main_chart")
            
            # Pattern selection buttons
            st.markdown("### 🎯 IDENTIFY THE PATTERN CANDLE")
            cols = st.columns(3)
            pattern_options = list(PATTERNS.keys())
            
            for i, pattern in enumerate(pattern_options):
                with cols[i % 3]:
                    if st.button(
                        pattern, 
                        key=f"pat_{pattern}_{int(time.time())}", 
                        use_container_width=True
                    ):
                        handle_submission(pattern, remaining)
            
            # Hint expander
            with st.expander("💡 Need a hint?"):
                st.info("Look at the **YELLOW LINE** candle. Check: 1) Body size/position, 2) Which shadow is long, 3) Color")
                
        else:
            # Time's up
            st.error(f"⏰ TIME'S UP! The pattern was: **{st.session_state.correct_pattern}**")
            st.session_state.streak = 0
            st.session_state.game_active = False
            
            if st.button("➡️ NEXT CHART", use_container_width=True):
                st.rerun()

if __name__ == "__main__":
    main()
