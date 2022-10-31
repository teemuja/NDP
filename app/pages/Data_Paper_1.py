# NDP app always beta a lot
import pandas as pd
import geopandas as gpd
import numpy as np
import streamlit as st
import shapely.speedups
shapely.speedups.enable()
import plotly.express as px
px.set_mapbox_access_token(st.secrets['MAPBOX_TOKEN'])
my_style = st.secrets['MAPBOX_STYLE']
from pathlib import Path
import h3pandas as h3
from shapely import wkt


# page setup ---------------------------------------------------------------
st.set_page_config(page_title="Data Paper #1", layout="wide", initial_sidebar_state='collapsed')
padding = 1
st.markdown(f""" <style>
    .reportview-container .main .block-container{{
        padding-top: {padding}rem;
        padding-right: {padding}rem;
        padding-left: {padding}rem;
        padding-bottom: {padding}rem;
    }}
    </style> """, unsafe_allow_html=True)
st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #fab43a;
        color:#ffffff;
    }
    div.stButton > button:hover {
        background-color: #e75d35; 
        color:#ffffff;
        }
    [data-testid="stMetricDelta"] svg {
            display: none;}
    button[title="View fullscreen"]{
        visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

header = '<p style="font-family:sans-serif; color:grey; font-size: 12px;">\
        NDP data paper #1 V0.1\
        </p>'
st.markdown(header, unsafe_allow_html=True)
# plot size setup
#px.defaults.width = 600
px.defaults.height = 700

# page header
header_title = '''
**Naked Density Project**
'''
st.subheader(header_title)
header_text = '''
<p style="font-family:sans-serif; color:dimgrey; font-size: 10px;">
Naked Density Project is a PhD research project by <a href="https://research.aalto.fi/en/persons/teemu-jama" target="_blank">Teemu Jama</a> in Aalto University Finland.  
NDP project studies correlation between urban density and <a href="https://sdgs.un.org/goals" target="_blank">SDG-goals</a> by applying latest spatial data analytics and machine learning. \
</p>
'''
st.markdown(header_text, unsafe_allow_html=True)
st.markdown("----")
# content
st.title("Data Paper #1")
st.subheader("Change in scale in detail plans in Helsinki")
ingress = '''
<p style="font-family:sans-serif; color:Black; font-size: 14px;">
This data paper studies the change in GFA and plan unit sizes in post-war detail plans by decades
</p>
'''
st.markdown(ingress, unsafe_allow_html=True)

# get the data
@st.experimental_memo()
def load_data():
    path = Path(__file__).parent / 'data/hki_ak_data_202210.csv'
    with path.open() as f:
        data = pd.read_csv(f, index_col='kaavatunnus', header=0)#.astype(str)
    return data
try:
    df_data = load_data()
except:
    st.warning('Dataan ei yhteyttä.')
    st.stop()

df_data['geometry'] = df_data['geometry'].apply(wkt.loads)
hki_ak_data = gpd.GeoDataFrame(df_data, crs=4326, geometry='geometry')
# drop zero GFAs
hki_ak_data = hki_ak_data.loc[hki_ak_data['rakennusoikeus'] != 0]

# select designations
all_list = hki_ak_data['kayttotarkoitusluokka_koodi'].unique().tolist()
c1,c2 = st.columns(2)
use_list = c1.multiselect('Select land use types to include',all_list,default=['C','AK','AP'])
years = c2.slider('Set decades to include',1940,2020,(1940,2020),step=10)
trendline = st.radio('Trendline model',['lowess','ols'], horizontal=True)
# create plot gdf
plot = hki_ak_data.loc[(hki_ak_data['vuosikymmen'] >= years[0]) & (hki_ak_data['vuosikymmen'] <= years[1])]
plot = plot[plot['kayttotarkoitusluokka_koodi'].isin(use_list)]
plot['vuosikymmen'] = plot['vuosikymmen'].astype(str)

# scatt plot
plot = plot.loc[plot['rekisteriala'] < plot['rekisteriala'].quantile(0.99)]
plot = plot.loc[plot['rakennusoikeus'] < plot['rakennusoikeus'].quantile(0.99)]
fig = px.scatter(plot , color="vuosikymmen", x="rekisteriala", y="rakennusoikeus", opacity=0.6, trendline=trendline,
                 labels={'rakennusoikeus':'GFA','rekisteriala':'Plan area','vuosikymmen':'Decade'},
                 title=f'GFA and plan sizes in detail plans in Helsinki by decade (trendline={trendline})',
                 color_discrete_sequence=px.colors.qualitative.Dark24)
fig.update_traces(patch={"line": {"width": 5}}, selector={"legendgroup": "2020"})
st.plotly_chart(fig, use_container_width=True)

# summaries
plot['tehokkuus'] = plot['rakennusoikeus']/plot['rekisteriala']
gfa_1970 = plot.loc[plot['vuosikymmen'] == '1970']['rakennusoikeus'].median()
e_1970 = plot.loc[plot['vuosikymmen'] == '1970']['tehokkuus'].median()
gfa_1980 = plot.loc[plot['vuosikymmen'] == '1980']['rakennusoikeus'].median()
e_1980 = plot.loc[plot['vuosikymmen'] == '1980']['tehokkuus'].median()
gfa_1990 = plot.loc[plot['vuosikymmen'] == '1990']['rakennusoikeus'].median()
e_1990 = plot.loc[plot['vuosikymmen'] == '1990']['tehokkuus'].median()
gfa_2000 = plot.loc[plot['vuosikymmen'] == '2000']['rakennusoikeus'].median()
e_2000 = plot.loc[plot['vuosikymmen'] == '2000']['tehokkuus'].median()
gfa_2010 = plot.loc[plot['vuosikymmen'] == '2010']['rakennusoikeus'].median()
e_2010 = plot.loc[plot['vuosikymmen'] == '2010']['tehokkuus'].median()
gfa_2020 = plot.loc[plot['vuosikymmen'] == '2020']['rakennusoikeus'].median()
e_2020 = plot.loc[plot['vuosikymmen'] == '2020']['tehokkuus'].median()

m1,m2,m3,m4,m5,m6 = st.columns(6)
m1.metric(label=f"Median GFA in 1970s", value=f"{gfa_1970:,.0f} sqrm", delta=f"median e={e_1970:.2f}")
m2.metric(label=f"Median GFA in 1980s", value=f"{gfa_1980:,.0f} sqrm", delta=f"median e={e_1980:.2f}")
m3.metric(label=f"Median GFA in 1990s", value=f"{gfa_1990:,.0f} sqrm", delta=f"median e={e_1990:.2f}")
m4.metric(label=f"Median GFA in 2000s", value=f"{gfa_2000:,.0f} sqrm", delta=f"median e={e_2000:.2f}")
m5.metric(label=f"Median GFA in 2010s", value=f"{gfa_2010:,.0f} sqrm", delta=f"median e={e_2010:.2f}")
m6.metric(label=f"Median GFA in 2020s", value=f"{gfa_2020:,.0f} sqrm", delta=f"median e={e_2020:.2f}")

# trendline info
def trend_values(fig):
    model = px.get_trendline_results(fig)
    n = len(model)
    df = pd.DataFrame(columns=['decade','constant','slope'])
    for i in range(n):
        r = model.iloc[i]["px_fit_results"]
        alpha = r.params[0] # constant
        beta = r.params[1] # slope
        dec = years[0] + (i*10)
        df.loc[i] = [dec]+[alpha]+[beta]
    df['decade'] = df['decade'].astype(int).astype(str)
    return df

if trendline == 'ols':
    with st.expander('OLS-trendline details',expanded=False):
        trend = trend_values(fig)
        fig2 = px.scatter(trend , color="decade", x="constant", y="slope", opacity=0.9,
                        labels={'decade':'Decade','constant':'Constant','slope':'Slope'},
                        title='Details of trendlines of different decades',
                        color_discrete_sequence=px.colors.qualitative.Dark24)
        st.plotly_chart(fig2, use_container_width=True)
        st.caption('Slope value indicates the density of the plans while constant value indicates the volume in the plans. '
                    'Bigger the slope, denser the plans. Bigger the constant, more GFA in the plans.')

# map plot
st.markdown('###')
mapit = st.button('Create map of plan units')
if mapit:
    with st.spinner("Preparing map of plans..."):
        lat = plot.unary_union.centroid.y
        lon = plot.unary_union.centroid.x
        mymap = px.choropleth_mapbox(plot,
                                    geojson=plot.geometry,
                                    locations=plot.index,
                                    title='Plan units on map',
                                    color="vuosikymmen",
                                    hover_name='kayttotarkoitusluokka_koodi',
                                    hover_data=['vuosi','rakennusoikeus','kaavayksikkotunnus'],
                                    labels={"vuosikymmen": 'Decade'},
                                    mapbox_style=my_style,
                                    color_discrete_sequence=px.colors.qualitative.D3,
                                    center={"lat": lat, "lon": lon},
                                    zoom=10,
                                    opacity=0.8,
                                    width=1200,
                                    height=700
                                    )
        st.plotly_chart(mymap, use_container_width=True)
        plancount = len(plot)
        st.caption(f'Total {plancount} plans of types {use_list} between {years[0]} and {years[1]}. Zero GFA plans excluded.')
        source = '''
        <p style="font-family:sans-serif; color:dimgrey; font-size: 10px;">
        Data: <a href="https://hri.fi/data/fi/dataset/helsingin-kaavayksikot" target="_blank">HRI & Helsinki</a>
        </p>
        '''
        st.markdown(source, unsafe_allow_html=True)

#footer
footer_title = '''
---
**Naked Density Project**
[![MIT license](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/teemuja/NDP/blob/main/LICENSE) 
'''
st.markdown(footer_title)