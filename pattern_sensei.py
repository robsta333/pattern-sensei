def main():
    st.set_page_config(page_title="Pattern Sensei", layout="wide", initial_sidebar_state="expanded")
    
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
    .cheat-sheet-chart {
        height: 150px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    initialize_session()
    
    # ──────────────────────────────────────────────────────────────
    # COLLAPSIBLE CHEAT SHEET IN SIDEBAR
    # ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 📖 PATTERN CHEAT SHEET")
        with st.expander("CLICK TO EXPAND (AVAILABLE DURING GAME)", expanded=False):
            st.markdown("### Master These 5 Patterns")
            
            for pattern_name in PATTERNS.keys():
                st.markdown(f"---")
                st.markdown(f"#### **{pattern_name.upper()}**")
                
                # Generate perfect example
                if pattern_name == "Hammer":
                    example_df = create_perfect_hammer()
                elif pattern_name == "Shooting Star":
                    example_df = create_perfect_shooting_star()
                elif pattern_name == "Doji":
                    example_df = create_perfect_doji()
                elif pattern_name == "Bullish Engulfing":
                    example_df = create_perfect_engulfing("bullish")
                elif pattern_name == "Bearish Engulfing":
                    example_df = create_perfect_engulfing("bearish")
                
                # Mini chart
                fig = go.Figure(data=go.Candlestick(
                    x=example_df['date'],
                    open=example_df['open'],
                    high=example_df['high'],
                    low=example_df['low'],
                    close=example_df['close'],
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
                
                # Key points
                st.markdown(f"""
                **Visual:** {PATTERNS[pattern_name]['desc']}  
                **Bias:** {PATTERNS[pattern_name]['bias']}  
                **Stop:** {PATTERNS[pattern_name]['stop_loss']}  
                **⏱️ ID Time:** {['5 sec', '5 sec', '7 sec', '6 sec', '6 sec'][list(PATTERNS.keys()).index(pattern_name)]}
                """)
    
    # ──────────────────────────────────────────────────────────────
    # MAIN GAME UI (unchanged)
    # ──────────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        st.markdown(f'<div class="scoreboard">💰 ACCOUNT<br>${st.session_state.account:,.2f}</div>', unsafe_allow_html=True)
    with col2:
        st.markdown("<h1 style='text-align: center; color: #ff6b6b; font-family: Orbitron;'>PATTERN SENSEI</h1>", unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="scoreboard">⚡ SCORE<br>{st.session_state.score}</div>', unsafe_allow_html=True)
    
    st.markdown(f"<h3 style='text-align: center; color: #48dbfb;'>Phase {st.session_state.phase} - {['Initiate', 'Apprentice', 'Journeyman'][st.session_state.phase-1].upper()}</h3>", unsafe_allow_html=True)
    
    if not st.session_state.game_active:
        if st.button("🎯 START TRADE", key="start"):
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
                    if st.button(pattern, key=f"pat_{pattern}"):
                        st.session_state.selected_pattern = pattern
                        st.rerun()
            
            if 'selected_pattern' in st.session_state:
                handle_submission(st.session_state.selected_pattern, remaining)
                del st.session_state.selected_pattern
            
        else:
            st.error("⏰ TIME'S UP! Pattern was: " + st.session_state.correct_pattern)
            st.session_state.streak = 0
            st.session_state.game_active = False
            if st.button("NEXT CHART"):
                st.rerun()

# ──────────────────────────────────────────────────────────────
# PERFECT EXAMPLE GENERATORS (for cheat sheet)
# ──────────────────────────────────────────────────────────────

def create_perfect_hammer():
    """Create textbook hammer example"""
    df = pd.DataFrame({
        'date': [datetime.now() - timedelta(minutes=i) for i in range(4, 0, -1)],
        'open': [100, 100.5, 99.8, 100],
        'high': [101, 101.2, 100.1, 100.3],
        'low': [99.5, 99.8, 97.5, 99.9],
        'close': [100.3, 100.8, 99.9, 100.2]
    })
    # Make candle #2 a perfect hammer
    df.at[df.index[1], 'low'] = 97.0  # Long lower wick
    df.at[df.index[1], 'open'] = 99.8
    df.at[df.index[1], 'close'] = 100.0  # Small body
    df.at[df.index[1], 'high'] = 100.2
    return df

def create_perfect_shooting_star():
    df = pd.DataFrame({
        'date': [datetime.now() - timedelta(minutes=i) for i in range(4, 0, -1)],
        'open': [100, 100, 100.5, 100.2],
        'high': [101, 103, 101.2, 100.8],
        'low': [99.5, 99.8, 100.3, 100.0],
        'close': [100.3, 100.1, 100.4, 100.3]
    })
    # Make candle #1 a perfect shooting star
    df.at[df.index[1], 'high'] = 103.5  # Long upper wick
    df.at[df.index[1], 'open'] = 100.0
    df.at[df.index[1], 'close'] = 100.1  # Small body
    df.at[df.index[1], 'low'] = 99.9
    return df

def create_perfect_doji():
    df = pd.DataFrame({
        'date': [datetime.now() - timedelta(minutes=i) for i in range(4, 0, -1)],
        'open': [100, 100, 100.5, 100.2],
        'high': [101, 101.5, 101.2, 100.8],
        'low': [99.5, 98.5, 99.8, 100.0],
        'close': [100.3, 100, 100.4, 100.3]
    })
    # Make candle #1 perfect doji
    df.at[df.index[1], 'open'] = 100.0
    df.at[df.index[1], 'close'] = 100.01  # Open ≈ Close
    return df

def create_perfect_engulfing(bias):
    df = pd.DataFrame({
        'date': [datetime.now() - timedelta(minutes=i) for i in range(5, 0, -1)],
        'open': [100, 100.5, 100, 100.3, 100.1],
        'high': [100.8, 101, 100.5, 101.5, 100.5],
        'low': [99.5, 100.2, 99.8, 99.5, 99.9],
        'close': [100.5, 100.7, 100.1, 101.2, 100.3]
    })
    
    if bias == "bullish":
        df.at[df.index[2], 'open'] = 100.5  # Small red candle
        df.at[df.index[2], 'close'] = 100.2
        df.at[df.index[1], 'open'] = 100.0  # Big green engulfs it
        df.at[df.index[1], 'close'] = 100.8
    else:
        df.at[df.index[2], 'open'] = 99.8   # Small green candle
        df.at[df.index[2], 'close'] = 100.3
        df.at[df.index[1], 'open'] = 100.4  # Big red engulfs it
        df.at[df.index[1], 'close'] = 99.7
        
    return df
