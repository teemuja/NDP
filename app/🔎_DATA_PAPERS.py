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

#logo
#from streamlit_extras.app_logo import add_logo #https://extras.streamlit.app
#add_logo('utils/NDP_150.png',height=120)
from PIL import Image
image = Image.open('utils/NDP_500.png')
l1,l2 =st.columns([1,5])
l1.image(image,use_column_width='auto')

# content
l2.title('NDP Data Papers')
header_ing = '''
<p style="font-family:sans-serif; color:dimgrey; font-size: 14px;">
Naked Density Project (NDP) is a PhD research project by <a href="https://research.aalto.fi/en/persons/teemu-jama" target="_blank">Teemu Jama</a> in Aalto University Finland.  
NDP project studies correlation between urban density and <a href="https://sdgs.un.org/goals" target="_blank">SDG-goals</a> by applying latest spatial data analytics and machine learning. \
</p>
'''
l2.markdown(header_ing, unsafe_allow_html=True)
