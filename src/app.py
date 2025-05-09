import streamlit as st
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import re
from PIL import Image
from utils.utils import *

# TODO:
# - get real data
# - fix dodgy IV syndata
# - error handling

st.set_page_config(page_title="Trade Recommendations", layout="wide")
st.title("Daily Trade Recommendations - "+str(dt.datetime.today().date()))

# load and prepare trade recs df
@st.cache_data
def load_and_prepare(path):
    raw = pd.read_csv(path, parse_dates=["RECOMMENDATION_DATE"])
    return transform_recs(raw)

df = load_and_prepare("data/trade_recs.csv")
df = df.reset_index(drop=True)

# main content function
def render_contract_subtabs(product_prefix):
    ctrs = [c for c in df["Contract"].unique() if c.startswith(product_prefix)]
    if not ctrs:
        st.write(f"No recommendations for {product_prefix}")
        return

    sub_tabs = st.tabs(ctrs)
    for contract, sub in zip(ctrs, sub_tabs):
        with sub:
            st.header(contract)
            df_c = df[df["Contract"] == contract]
            st.dataframe(df_c.drop(columns=["Contract"]), use_container_width=True)

            st.header("Price")
            st.plotly_chart(price_figure(contract), use_container_width=True)
            # st.write("(*Move tabs on edges of bottom subplot to change date range)")

            st.header("ATM Volatility")
            st.plotly_chart(vol_figure(contract), use_container_width=True)

            st.header("IV Smile Evolution")
            st.plotly_chart(iv_smile_fig(contract), use_container_width=True)

# plotting functions
@st.cache_data
def price_figure(contract):
    ticker = contract_to_ticker(contract)
    ohlc = pd.read_csv(f"data/{ticker}_ohlc.csv").set_index('date')
    fc = pd.read_csv(f"data/{ticker}_fc.csv").set_index('date')
    fig = go.Figure()
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=ohlc.index, open=ohlc["open"], high=ohlc["high"],
        low=ohlc["low"], close=ohlc["close"],
        increasing=dict(line_color="green", fillcolor="green"),
        decreasing=dict(line_color="red",   fillcolor="red"),
        name="History"
    ))
    # Forecast lines
    colors = {"nc":"#FF6600","c":"cyan","ct":"magenta","ctt":"#B8860B"}
    for col in fc.columns:
        fig.add_trace(go.Scatter(
            x=fc.index, y=fc[col], mode="lines",
            line=dict(color=colors[col], width=2),
            name=f"FC {col.upper()}"
        ))

    recs = df
    strikes = recs.loc[recs["Contract"] == contract, "Strike"].unique()
    if len(strikes):
        x0, x1 = fc.index.min(), fc.index.max()
        for strike in strikes:
            fig.add_trace(go.Scatter(
                x=[x0, x1], y=[strike, strike],
                mode="lines",
                line=dict(color="gray", width=1, dash="dash"),
                showlegend=False
            ))

            fig.add_annotation(
                    x=x1, y=strike,
                    text=f"Strike {strike:.2f}",
                    showarrow=False,
                    xanchor="left", 
                    xshift =-50,
                    yanchor="bottom",
                    yshift=0,
                    font=dict(color="gray", size=10),
                )

    fig.update_layout(
        title=f"{contract}: Historical OHLC + VAR Price Forecasts",
         xaxis=dict(
            color="white",
            rangeslider=dict(visible=False)
        ),
        yaxis=dict(color="white"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.02),
        font=dict(color="gray"),
        margin=dict(t=50, b=30, l=30, r=30)
    )
    return fig

@st.cache_data
def vol_figure(contract):
    ticker = contract_to_ticker(contract)
    hist = pd.read_csv(f'data/{ticker}_hist.csv').set_index('date')
    IV = pd.read_csv(f'data/{ticker}_IVfc.csv').set_index('date')
    
    new_cols = np.round(IV.columns.astype(float), 2)
    IV.columns = new_cols
    fig = go.Figure()

    # Historical vol
    fig.add_trace(go.Scatter(
        x=hist.index, y=hist["vol"],
        mode="lines", name="History",
        line=dict(color="orange", width=2)
    ))
    # Single forecast
    fig.add_trace(go.Scatter(
        x=IV.index, y=IV[1.00],
        mode="lines", name="Forecast",
        line=dict(color="cyan", width=2, dash="dash")
    ))
    fig.update_layout(
        title=f"{contract}: ATM Volatility & Forecast",
        xaxis_title="Date",
        yaxis_title="Volatility (%)",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        legend=dict(orientation="h", y=1.02),
        margin=dict(t=50, b=30, l=30, r=30)
    )
    return fig

@st.cache_datas
def iv_smile_fig(contract):
    ticker = contract_to_ticker(contract)
    # Read wide DataFrame (date index, each moneyness as a column)
    df_wide = (
        pd.read_csv(f"data/{ticker}_IVfc.csv", index_col="date")
        .sort_index()
    )

    # Melt back to long form: one row per (date, moneyness)
    df_long = (
        df_wide
        .reset_index()
        .melt(id_vars="date", var_name="moneyness", value_name="iv")
    )
    df_long["moneyness"] = df_long["moneyness"].astype(float)

    fig = px.line(
        df_long,
        x="moneyness",
        y="iv",
        animation_frame="date",
        range_y=[df_long.iv.min() - 1, df_long.iv.max() + 1],
        labels={"moneyness": "Moneyness", "iv": "IV (%)"},
        title=f"{contract}: Implied Volatility Forecast"
    )
    fig.update_traces(line=dict(width=2))
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        # font=dict(color="white"),
        sliders=dict(transition=dict(duration=300)),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True)
    return fig


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
    st.write('(*Bid/offer prices are combined for composite positions, e.g. risk reversals)')
    st.subheader("Trade Class Visualization")
    spacer1, content_col, spacer2 = st.columns([1, 4, 1], vertical_alignment="center")

    with content_col:
       
        col_img, col_tex = st.columns([1, 1], vertical_alignment="center")
        with col_img:
            img = Image.open("tradeclasses.png")
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
    render_contract_subtabs("TTF")

# HH tab
with tab_hh:
    render_contract_subtabs("HH")

# WTI tab
with tab_wti:
    render_contract_subtabs("WTI")

# Brent tab
with tab_brent:
    render_contract_subtabs("BRENT")

