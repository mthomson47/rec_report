import streamlit as st
import pandas as pd
from pandas.tseries.offsets import BDay
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
# - HOLIDAYS
# - error handling
# - deployment

st.set_page_config(page_title="Trade Recommendations", layout="wide", initial_sidebar_state="collapsed")
# pd.options.display.float_format = '{:.2f}'.format

# load and prepare trade recs df
# @st.cache_data
def load_and_prepare(path):
    raw = pd.read_parquet(path)
    return transform_recs(raw), raw

rec_df, rec_raw = load_and_prepare(r"T:\Besold Temp\TradeRecs\trade_recs_all.prqt")
price_forecast_df_all = pd.read_parquet(r"T:\Besold Temp\PriceForecasts\price_forecasts_all.prqt").sort_index()
mkt_df = pd.read_parquet(r"T:\Besold Temp\MarketDataPriceVol\total_current.parquet")
vol_forecast_df_all = pd.read_parquet(r"T:\Besold Temp\VolForecasts\vol_forecasts_all.prqt").sort_index()

today = dt.date.today()
selected_date = today

col_title, col_date = st.columns([5, 1])

with col_title:
    title_slot = st.title(f"Daily Trade Recommendations — {selected_date}")

with col_date:
    selected_date = st.date_input(
        "",
        value=selected_date,
        min_value=rec_raw["RECOMMENDATION_DATE"].min(),
        max_value=today
    )

# Update the title to reflect the new selection
title_slot.title(f"Daily Trade Recommendations — {selected_date}")
df = rec_df[rec_df["RECOMMENDATION_DATE"] == selected_date]

# price_forecasts = price_forecast_df_all[price_forecast_df_all["valDate"] == selected_date].sort_values(by="TRADING_DATE")

selected_date = pd.to_datetime(selected_date)
historical_start = selected_date - pd.tseries.offsets.BDay(100)
historical_start_vol = selected_date - pd.tseries.offsets.BDay(40)
realized_end = selected_date + pd.tseries.offsets.BDay(30-2)
realized_end_vol = selected_date + pd.tseries.offsets.BDay(10)
dates = mkt_df.index.get_level_values("TRADING_DATE")
mask = (dates >= historical_start) & (dates <= realized_end)
mask_vol = (dates >= historical_start_vol) & (dates <= realized_end_vol)
price_historical = np.exp(mkt_df.loc[mask]['PRICE'])
vol_historical = mkt_df.loc[mask_vol]
ATM_historical = vol_historical['1.0000']

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
            st.dataframe(df_c.drop(columns=["RECOMMENDATION_DATE","Contract"]), use_container_width=True)

            st.header("Price")
            st.plotly_chart(price_figure(contract), use_container_width=True)
            # st.write("(*Move tabs on edges of bottom subplot to change date range)")
            st.write("(*Click on the legend to hide/show lines. Trend 'CT' is used for trade class selection.)")

            st.header("ATM Volatility")
            st.plotly_chart(vol_figure(contract), use_container_width=True)

            st.header("IV Smile")
            st.plotly_chart(iv_smile_fig(contract), use_container_width=True)

# plotting functions
# @st.cache_data
def price_figure(contract):
    ticker = contract_to_ticker(contract)

    fig = go.Figure()

    # # Candlestick
    # fig.add_trace(go.Candlestick(
    #     x=ohlc.index, open=ohlc["open"], high=ohlc["high"],
    #     low=ohlc["low"], close=ohlc["close"],
    #     increasing=dict(line_color="green", fillcolor="green"),
    #     decreasing=dict(line_color="red",   fillcolor="red"),
    #     name="History"
    # ))
    
    hist = price_historical.loc[ticker]
    fc = price_forecast_df_all.loc[(ticker, selected_date)].sort_values(by="TRADING_DATE")
    fc = fc.set_index("TRADING_DATE")
    fc.index = pd.to_datetime(fc.index)
    first_date = fc.index[0]
    start = first_date - BDay()
    new_idx = pd.date_range(start=start, periods=len(fc), freq=BDay())
    fc.index = new_idx
    recs_c = df[df["Contract"] == contract]
    baseline = min(hist.min(), fc[['nc', 'c', 'ct', 'ctt']].min().min(), recs_c["Strike"].min())
    hist_rel = hist - baseline

    fig.add_trace(go.Scatter(
        x=hist.index, 
        y=[baseline]*len(hist),
        mode='lines',
        line=dict(color='rgba(0,0,0,0)'),  # fully transparent
        showlegend=False
    ))
    
    N = 50
    for i in range(1, N+1):
        frac    = i / N
        opacity = frac * 0.3  # max 30% opacity
        # Build absolute y-values: baseline + frac*(hist-baseline)
        y_i = baseline + hist_rel * frac
        fig.add_trace(go.Scatter(
            x=hist.loc[:selected_date.date()].index,  # only show up to selected date
            y=y_i,
            mode='none',
            fill='tonexty',
            fillcolor=f'rgba(30,144,255,{opacity})',
            hoverinfo='skip',
            showlegend=False
        ))

    fig.add_trace(go.Scatter(
            x=hist.index, y=hist,
            mode="lines", name="Realized",
            line=dict(color="#1E90FF", width=2)
        ))
    colors = {
        "nc":  "#444444",   # darkest grey
        "c":   "#666666",   # medium-dark
        "ct":  "#888888",   # medium-light
        "ctt": "#AAAAAA",   # light grey
    }
    for col in colors.keys():
        fig.add_trace(go.Scatter(
            x=fc.index, y=fc[col], mode="lines",
            line=dict(color=colors[col], width=2),
            name=f"FC {col.upper()}"
        ))

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

    # today   = pd.Timestamp(dt.date.today())
    expiry  = pd.to_datetime(
        df.loc[df["Contract"] == contract, "Expiry"].iloc[0]
    ).normalize()

    for date, label, color in [
        (selected_date.date(),  f"Inception: \n{selected_date.date()}",  "#1E90FF"),
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
        title=f"{contract}: Realized Price + VAR Forecasts",
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

# @st.cache_data
def vol_figure(contract):
    ticker = contract_to_ticker(contract)
    hist = ATM_historical.loc[ticker]
    fc = vol_forecast_df_all.loc[(ticker, selected_date)].sort_index().sort_values(by="TRADING_DATE")
    fc = fc.set_index("TRADING_DATE")
    fc.index = pd.to_datetime(fc.index)
    first_date = fc.index[0]
    start = first_date
    new_idx = pd.date_range(start=start-BDay(1), periods=len(fc), freq=BDay())
    fc.index = new_idx
    ATM = fc['1.0000']
    sd = pd.to_datetime(selected_date)-BDay(1)
    last_hist_val = hist.asof(sd)
    prepend = pd.Series([last_hist_val], index=[sd])
    ATM = pd.concat([prepend, ATM]).sort_index()

    baseline = min(hist.min(), ATM.min())
    hist_rel = hist - baseline
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=hist.index, 
        y=[baseline]*len(hist),
        mode='lines',
        line=dict(color='rgba(0,0,0,0)'),  # fully transparent
        showlegend=False
    ))

    N = 50
    for i in range(1, N+1):
        frac    = i / N
        opacity = frac * 0.3  # max 30% opacity
        # Build absolute y-values: baseline + frac*(hist-baseline)
        y_i = baseline + hist_rel * frac
        fig.add_trace(go.Scatter(
            x=hist.loc[:selected_date.date()].index,
            y=y_i,
            mode='none',
            fill='tonexty',
            fillcolor=f'rgba(255,165,0,{opacity})',
            hoverinfo='skip',
            showlegend=False
        ))

    # Historical vol
    fig.add_trace(go.Scatter(
        x=hist.index, y=hist,
        mode="lines", name="Realized",
        line=dict(color="orange", width=2),
        # fill="tonexty",
        # fillcolor="rgba(255,165,0,0.3)" 
    ))

    # Single forecast
    fig.add_trace(go.Scatter(
        x=ATM.index, y=ATM,
        mode="lines", name="Forecast",
        line=dict(color="#888888", width=2, dash="dash")
    ))

    for date, label, color in [
        (selected_date.date(),  f"Inception: \n{selected_date.date()}",  "#1E90FF"),
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
        height=500,      
        title=f"{contract}: Realized ATM Volatility + Forecast",
        xaxis_title="Date",
        yaxis_title="Volatility",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        legend=dict(orientation="h", y=1.02),
        margin=dict(t=50, b=30, l=30, r=30)
    )
    return fig

# @st.cache_data
def iv_smile_fig(contract):
    ticker = contract_to_ticker(contract)

    # Read wide DataFrame (date index, each moneyness as a column)
    df_wide_hist = vol_historical.loc[ticker].drop(columns=["SHORT_NAME","PRICE", "EXPIRY"])
    df_wide_hist = df_wide_hist.loc[(df_wide_hist.index < selected_date) & (df_wide_hist.index >= selected_date - BDay(10))]

    df_wide = vol_forecast_df_all.loc[(ticker, selected_date)].sort_index().sort_values(by="TRADING_DATE").drop(columns=["SHORT_NAME"])
    df_wide = df_wide.set_index("TRADING_DATE")

    # Remap forecast index to BDay frequency
    first_date = df_wide.index[0]
    start = first_date
    new_idx = pd.date_range(start=start-BDay(1), periods=len(df_wide), freq=BDay())
    df_wide.index = new_idx
    df_wide.index.rename("TRADING_DATE", inplace=True)

    # Melt back to long form: one row per (date, moneyness)
    df_long_hist = (
        df_wide_hist
        .reset_index()
        .melt(id_vars="TRADING_DATE", var_name="moneyness", value_name="iv")
    )
    df_long_hist["moneyness"] = df_long_hist["moneyness"].astype(float)
    df_long_hist["TRADING_DATE"] = df_long_hist["TRADING_DATE"].dt.date

    df_long = (
        df_wide
        .reset_index()
        .melt(id_vars="TRADING_DATE", var_name="moneyness", value_name="iv")
    )
    df_long["moneyness"] = df_long["moneyness"].astype(float)
    df_long["TRADING_DATE"] = df_long["TRADING_DATE"].dt.date

    df_long_hist["Source"] = "Realized"
    df_long["Source"]      = "Forecast"


    df_all = pd.concat([df_long_hist, df_long], ignore_index=True)

    # Bullshit to fix legend
    first_date = df_all["TRADING_DATE"].min()
    for src in ["Realized", "Forecast"]:
        mask = (df_all["TRADING_DATE"] == first_date) & (df_all["Source"] == src)
        if not mask.any():
            m = df_all.loc[df_all["Source"] == src, "moneyness"].iloc[0]
            v = df_all.loc[df_all["Source"] == src, "iv"].iloc[0]
            df_all = pd.concat([
                df_all,
                pd.DataFrame({
                    "TRADING_DATE": [ first_date ],
                    "moneyness":    [ m ],
                    "iv":           [ v ],
                    "Source":       [ src ]
                })
            ], ignore_index=True)


    fig = px.line(
        df_all,
        x="moneyness",
        y="iv",
        color="Source",
        animation_frame="TRADING_DATE",
        range_y=[df_all.iv.min() - 0.01, df_all.iv.max() + 0.01],
        labels={"moneyness": "Moneyness", "iv": "Implied Volatility", "Source": ""},
        title=f"{contract}: Realized Implied Volatility + Forecast",
        color_discrete_map={"Realized":"#4CC9F0", "Forecast":"#1E90FF"},
    )
    fig.update_traces(line=dict(width=2))
    fig.update_layout(
        legend=dict(
            orientation="h",
            y=1.02
        )
    )
    fig.update_layout(
        autosize=False,
        height=700,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        sliders=dict(transition=dict(duration=150)),
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
    st.dataframe(df.drop(['RECOMMENDATION_DATE'], axis=1), use_container_width=True)
    # st.write('(*Bid/offer prices are combined for composite positions, e.g. risk reversals)')
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

