# main app
# see version change log for new feats: https://docs.streamlit.io/library/changelog
import streamlit as st

st.set_page_config(page_title="NDP Data Papers", layout="wide", initial_sidebar_state='expanded')

# hide image fullscreen
st.markdown("""
<style>
[data-testid="stMetricDelta"] svg {
        display: none;}
button[title="View fullscreen"]{
    visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# content
st.title('working?')
