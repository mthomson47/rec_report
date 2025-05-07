import streamlit as st

st.set_page_config(page_title="Trade Recommendations", layout="wide")
st.title("Daily Trade Recommendations")

# Create the tabs
tab_recommendations, tab_ttf, tab_hh, tab_wti, tab_brent = st.tabs([
    "Recommendations", "TTF", "HH", "WTI", "BRENT"
])

# Contents tab
with tab_recommendations:
    st.header("Recommendations")
    st.markdown("""
    1. [TTF (Title)](#)
    2. [HH (Title)](#)
    3. [WTI (Title)](#)
    4. [Brent (Title)](#)
    """)

# TTF tab
with tab_ttf:
    st.header("TTF")
    st.write("Place your TTF price & IV charts here")
    # e.g. st.plotly_chart(your_ttf_figure)

# HH tab
with tab_hh:
    st.header("HH")
    st.write("Place your HH price & IV charts here")
    # e.g. st.plotly_chart(your_hh_figure)

# WTI tab
with tab_wti:
    st.header("WTI")
    st.write("Place your WTI price & IV charts here")
    # e.g. st.plotly_chart(your_wti_figure)

# Brent tab
with tab_brent:
    st.header("BRENT")
    st.write("Place your Brent price & IV charts here")
    # e.g. st.plotly_chart(your_brent_figure)
