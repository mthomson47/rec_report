import streamlit as st
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image

st.set_page_config(page_title="Trade Recommendations", layout="wide")
st.title("Daily Trade Recommendations - "+str(dt.datetime.today().date()))

img = Image.open("tradeclasses.png")

# Recommendation data
@st.cache_data
def load_recs(path="data/trade_recs.csv"):
    return pd.read_csv(path)

df = load_recs("data/trade_recs.csv").drop('RECOMMENDATION_DATE', axis=1)
df = df.reset_index(drop=True)

@st.cache_data
def make_ttf_data():
    today = dt.datetime.today().date()
    np.random.seed(None)
    # Past 60 business days
    dates_hist = pd.bdate_range(end=today, periods=30)
    base = 50 + np.cumsum(np.random.randn(len(dates_hist)))
    ohlc = pd.DataFrame({
        "date": dates_hist,
        "open":  base + np.random.randn(len(base))*0.5,
        "high":  base + np.random.rand(len(base))*1.5,
        "low":   base - np.random.rand(len(base))*1.5,
        "close": base + np.random.randn(len(base))*0.5,
    }).set_index("date")
    # Next 30 business days
    dates_fc = pd.bdate_range(start=today + pd.Timedelta(days=1), periods=30)
    last = ohlc["close"].iloc[-1]
    fc = pd.DataFrame({
        "date": dates_fc,
        "nc":  np.full(30, last),
        "c":   last + np.linspace(0, 2, 30),
        "ct":  last + np.linspace(0, 1, 30) + 0.01*np.arange(30),
        "ctt": last + 0.5*(np.linspace(0,1,30)**2),
    }).set_index("date")
    return ohlc, fc

def ttf_figure(ohlc, fc):
    fig = make_subplots(specs=[[{"secondary_y": False}]])
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=ohlc.index, open=ohlc["open"], high=ohlc["high"],
        low=ohlc["low"], close=ohlc["close"],
        increasing=dict(line_color="green", fillcolor="green"),
        decreasing=dict(line_color="red",   fillcolor="red"),
        name="History"
    ))
    # Forecast lines
    colors = {"nc":"orange","c":"cyan","ct":"magenta","ctt":"yellow"}
    for col in fc.columns:
        fig.add_trace(go.Scatter(
            x=fc.index, y=fc[col], mode="lines",
            line=dict(color=colors[col], width=2),
            name=f"FC {col.upper()}"
        ))
    fig.update_layout(
        title="TTF: Historical OHLC + VAR Price Forecasts",
        xaxis=dict(color="white"), yaxis=dict(color="white"),
        plot_bgcolor="black", paper_bgcolor="black",
        legend=dict(orientation="h", y=1.02),
        font=dict(color="white"),
        margin=dict(t=50, b=30, l=30, r=30)
    )
    return fig

ohlc, fc = make_ttf_data()

# define tabs
tab_recs, tab_ttf, tab_hh, tab_wti, tab_brent = st.tabs([
    "Overview",
    "TTF",
    "HH",
    "WTI",
    "BRENT"
])

# Overview tab
with tab_recs:
    st.header("Overview")
    st.dataframe(df, use_container_width=True)

    st.subheader("Trade Class Visualization")
    spacer1, content_col, spacer2 = st.columns([1, 4, 1], vertical_alignment="center")

    with content_col:
       
        col_img, col_tex = st.columns([1, 1], vertical_alignment="center")
        with col_img:
            st.image(img, use_container_width=True)
        with col_tex:
            st.latex(r"""
                \begin{align*}
                    F_P &= \text{forecasted price trend} \\[8pt]
                    F_V &= \text{forecasted volatility trend} \\[8pt]
                    \theta &= \arctan\frac{F_V}{F_P} \\[8pt]
                    r &= \sqrt{F_P^2 + F_V^2}
                \end{align*}
            """)
    
    

# TTF tab
with tab_ttf:
    st.header("TTF")
    # st.subheader("Price forecasts (VAR)")
    st.plotly_chart(ttf_figure(ohlc, fc), use_container_width=True)

# HH tab
with tab_hh:
    st.header("HH")
    st.write("… your HH charts …")

# WTI tab
with tab_wti:
    st.header("WTI")
    st.write("… your WTI charts …")

# Brent tab
with tab_brent:
    st.header("Brent")
    st.write("… your Brent charts …")

