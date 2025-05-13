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

# load and prepare trade recs df
@st.cache_data
def load_and_prepare(path):
    raw = pd.read_csv(path, parse_dates=["RECOMMENDATION_DATE"])
    return transform_recs(raw), raw

df, raw = load_and_prepare("data/trade_recs.csv")
df = df.reset_index(drop=True)

today = dt.date.today()
selected_date = st.sidebar.date_input(
    "View data as of", 
    value=today,
    min_value=raw["RECOMMENDATION_DATE"].min().date(),
    max_value=raw["RECOMMENDATION_DATE"].max().date()
)
st.title("Daily Trade Recommendations - "+str(selected_date))

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

            st.header("IV Smile")
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
    colors = {
        "nc":  "#444444",   # darkest grey
        "c":   "#666666",   # medium-dark
        "ct":  "#888888",   # medium-light
        "ctt": "#AAAAAA",   # light grey
    }
    for col in fc.columns:
        fig.add_trace(go.Scatter(
            x=fc.index, y=fc[col], mode="lines",
            line=dict(color=colors[col], width=2),
            name=f"FC {col.upper()}"
        ))

    

    recs_c = df[df["Contract"] == contract]
    x0, x1 = fc.index.min(), fc.index.max()

    for _, row in recs_c.iterrows():
        strike   = row["Strike"]
        sign     = row["Long/Short"]
        pc       = row["Put/Call"]
        expiry   = row["Expiry"]
        # derive leg description
        leg_desc = f"{'Long' if sign>0 else 'Short'} {'Call' if pc=='C' else 'Put'}"
        bullbear = f"{'green' if leg_desc=='Long Call' or leg_desc=='Short Put' else 'red'}"

        # dashed horizontal line
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[strike, strike],
            mode="lines", showlegend=False,
            line=dict(color="gray", width=1, dash="dash")
        ))

        txt = (
            f"<span style='color:{bullbear};'>{contract} {leg_desc} @{strike:.2f}<br></span>"
            f"Exp: {expiry}"
        )
        # annotation at right end, just above the line
        fig.add_annotation(
            x=1.0, y=strike,
            xref="paper", yref="y",
            text=txt,
            showarrow=False,
            xanchor="left",
            xshift=-150,
            yanchor="middle",
            yshift=0,
            align='right',
            font=dict(color="gray", size=12),
        )

    today   = pd.Timestamp(dt.date.today()-dt.timedelta(days=3))
    expiry  = pd.to_datetime(
        df.loc[df["Contract"] == contract, "Expiry"].iloc[0]
    ).normalize()

    for date, label, color in [
        (today,  f"Inception: \n{today.date()}",  "#1E90FF"),
        # (expiry, f"Expiry \n{expiry.date()}","red")
    ]:
        fig.add_shape(
            type="line",
            x0=date, x1=date, xref="x",
            y0=0,    y1=1,    yref="paper",
            line=dict(color=color, width=1, dash="dot"),
        )
        fig.add_annotation(
            x=date, y=1,
            xref="x", yref="paper",
            text=label,
            showarrow=False,
            xanchor="left", yanchor="bottom",
            yshift=-20,
            font=dict(color=color, size=12)
        )

    fig.update_layout(
        autosize=False,     
        height=700,      
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

    baseline = min(hist['vol'].min(), IV[1.00].min())
    hist_rel = hist["vol"] - baseline
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=hist.index, 
        y=[baseline]*len(hist),
        mode='lines',
        line=dict(color='rgba(0,0,0,0)'),  # fully transparent
        showlegend=False
    ))

    N = 20
    for i in range(1, N+1):
        frac    = i / N
        opacity = frac * 0.3  # max 30% opacity
        # Build absolute y-values: baseline + frac*(hist-baseline)
        y_i = baseline + hist_rel * frac
        fig.add_trace(go.Scatter(
            x=hist.index,
            y=y_i,
            mode='none',
            fill='tonexty',
            fillcolor=f'rgba(255,165,0,{opacity})',
            showlegend=False
        ))

    # Historical vol
    fig.add_trace(go.Scatter(
        x=hist.index, y=hist["vol"],
        mode="lines", name="History",
        line=dict(color="orange", width=2),
        # fill="tonexty",
        # fillcolor="rgba(255,165,0,0.3)" 
    ))

    # Single forecast
    fig.add_trace(go.Scatter(
        x=IV.index, y=IV[1.00],
        mode="lines", name="Forecast",
        line=dict(color="#888888", width=2, dash="dash")
    ))
    fig.update_layout(
        autosize=False,     
        height=500,      
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

@st.cache_data
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
        autosize=False,     
        height=700,      
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

