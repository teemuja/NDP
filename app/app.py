import streamlit as st

st.set_page_config(page_title="NDP", layout="wide", initial_sidebar_state='expanded')
st.markdown("""
<style>
button[title="View fullscreen"]{
        visibility: hidden;}
</style>
""", unsafe_allow_html=True)

#title
st.sidebar.header("NDP Data Papers",divider='green')
header_ing = '''
<p style="font-family:sans-serif; color:dimgrey; font-size: 14px;">
CFUA is a PhD research project by <a href="https://orcid.org/0000-0003-0168-7062" target="_blank">Teemu Jama</a>.  \
</p>
'''
st.sidebar.markdown(header_ing, unsafe_allow_html=True)
spons = '''
<p style="font-family:sans-serif; color:dimgrey; font-size: 11px;">  
Funding (direct & indirect)<br> 
<a href="https://emilaaltonen.fi" target="_blank">Eemil Aaltosen Säätiö</a><br>
<a href="https://www.aalto.fi/en/department-of-architecture" target="_blank">Aalto dep. of Architecture</a><br> 
<a href="https://www.aalto.fi/en/department-of-built-environment/geoinformatics" target="_blank">Aalto Geoinformatics</a><br> 
<a href="https://english.hi.is" target="_blank">University Of Iceland</a><br> 
</p>
'''
st.sidebar.markdown(spons, unsafe_allow_html=True)


home_page = st.Page("dps/home.py", title="Home", default=True)


dp1 = st.Page("dps/DP1_ChangeInScale.py", title="Change In Scale")
dp2 = st.Page("dps/DP2_DensityAndAmenities.py", title="Density And Amenities")
dp3 = st.Page("dps/DP3_TrueDensityNomograms.py", title="True Density Nomograms")
dp4 = st.Page("dps/DP4_TehokkuusGradientitSuomessa.py", title="Tehokkuus Gradientit Suomessa")
#dp5 = st.Page("dps/DP5_PopGrowthInDensityClasses.py", title="Väestökasvu eri maankäytön tiiviysluokissa")

menu = st.navigation(
    {
        #"Home": [home_page],
        "Data Papers": [dp1,dp2,dp3,dp4]
    }
)
menu.run()

#sidebar footer
st.sidebar.markdown('---')
footer_title = '''
[![MIT license](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/teemuja/NDP/blob/main/LICENSE) 
'''
st.sidebar.markdown(footer_title)
disclamer = 'Data papers are constant work in progress and will be upgraded, changed & fixed while research go on.'
st.sidebar.caption('Disclaimer: ' + disclamer)