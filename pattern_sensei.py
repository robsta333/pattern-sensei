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

# Cache cheat sheet examples so they only generate once
@st.cache_data
def create_perfect_examples():
    """Generate perfect pattern examples ONCE and cache them"""
    examples = {}
    
    # Hammer
    df = pd.DataFrame({
        'date': [datetime.now() - timedelta(minutes=i) for i in range(4, 0, -1)],
        'open': [100, 100, 99.8, 100],
        'high': [101, 101, 100.1, 100.3],
        'low': [99.5, 99.8, 96.5, 99.9],
        'close': [100.3, 100.8, 99.9, 100.2]
    })
    df.at[df.index[1], 'low'] = 96.0
    examples['Hammer'] = df
    
    # Shooting Star
    df = pd.DataFrame({
        'date': [datetime.now() - timedelta(minutes=i) for i in range(4, 0, -1)],
        'open': [100, 100, 100.5, 100.2],
        'high': [101, 104, 101.2, 100.8],
        'low': [99.5, 99.8, 100.3, 100.0],
        'close': [100.3, 100.1, 100.4, 100.3]
    })
    df.at[df.index[1], 'high'] = 104.5
    examples['Shooting Star'] = df
    
    # Doji
    df = pd.DataFrame({
        'date': [datetime.now() - timedelta(minutes=i)
