import streamlit as st
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from PIL import Image

st.set_page_config(page_title="Trade Recommendations", layout="wide")
st.title("Daily Trade Recommendations - "+str(dt.datetime.today().date()))

img = Image.open("tradeclasses.png")

@st.cache_data
def load_recs(path="data/trade_recs.csv"):
    return pd.read_csv(path)

df = load_recs("data/trade_recs.csv").drop('RECOMMENDATION_DATE', axis=1)
df = df.reset_index(drop=True)

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
                \theta = \arctan\frac{\mathrm{vol}}{\mathrm{price}} \\[8pt]
                r = \sqrt{\mathrm{price}^2 + \mathrm{vol}^2}
            """)
    
    

# TTF tab
with tab_ttf:
    st.header("TTF")
    st.write("… your TTF charts …")

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

