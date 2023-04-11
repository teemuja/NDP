#python we go
import pandas as pd
import geopandas as gpd
import numpy as np
from scipy.stats import boxcox
import streamlit as st
import shapely.speedups
shapely.speedups.enable()
import plotly.express as px
import plotly.graph_objs as go
px.set_mapbox_access_token(st.secrets['MAPBOX_TOKEN'])
my_style = st.secrets['MAPBOX_STYLE']
from pathlib import Path
import h3pandas as h3
from shapely import wkt


# page setup ---------------------------------------------------------------
st.set_page_config(page_title="Data Paper #4", layout="wide", initial_sidebar_state='collapsed')
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
        NDP data paper #4 V0.91\
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
st.title("Data Paper #4")
st.subheader("Change in density in HMA")
ingress = '''
<p style="font-family:sans-serif; color:Black; font-size: 14px;">
This data paper studies the growth of the Helsinki metropolitan area (HMA) in the zones designated by 
<a href="https://ckan.ymparisto.fi/en/dataset/harva-ja-tihea-taajama-alue" target="_blank">SYKE</a>.  
Data is prepared in cooperation with PhD(cand) Mathew Page from University of Helsinki.
</p>
'''
st.markdown(ingress, unsafe_allow_html=True)
st.markdown("###")

# data loader
@st.cache_data()
def load_data():
    path = Path(__file__).parent / 'data/SYKE_grid_h10.csv'
    with path.open() as f:
        data = pd.read_csv(f, index_col='h3_10', header=0)#.astype(str)
    # to gdf
    data['geometry'] = data['geometry'].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(data, crs=4326, geometry='geometry')
    # center tuple
    lat = gdf.unary_union.centroid.y
    lon = gdf.unary_union.centroid.x
    center = [lat,lon]
    # rename
    gdf.rename(columns={'pop19':'pop20'}, inplace=True)
    gdf.rename(columns={'GFA19':'GFA20'}, inplace=True)
    return gdf, center

mygdf,center = load_data()

# reso
c1,c2,c3,c4 = st.columns(4)
yrs = {
    '2020':'Type20',
    '2010':'Type10',
    '2000':'Type00',
    '1990':'Type90'
}
#year
year = c1.selectbox('Zoning year', ['2020','2010','2000','1990',])
#zone
zone = c2.selectbox('Zone',['Tiheä taajama','Harva taajama','Kylät','Pienkylät','Maaseutuasutus'])
#reso
reso = c3.slider('Resolution',6,9,9,1)
#feature
feat = c4.selectbox('Visualise',['Population','GFA'])
#legend
keys = {
    'Tiheä taajama':1,
    'Harva taajama':2,
    'Kylät':3,
    'Pienkylät':4,
    'Maaseutuasutus':5
    }
# filter
plot = mygdf.loc[mygdf[yrs[year]] == keys[zone]].h3.geo_to_h3_aggregate(reso)

colormap = {
    1:'darkbrown',
    2:'brown',
    3:'darkgoldenrod',
    4:'goldenrod',
    5:'gold'
    }
# color
if feat == 'Population':
    color = f'pop{year[-2:]}'
    plot = plot.loc[plot[color] > plot[color].quantile(0.1)]
else:
    color = f'GFA{year[-2:]}'
    plot = plot.loc[plot[color] > plot[color].quantile(0.1)]

range_min = plot[color].quantile(0.1)
range_max = plot[color].quantile(0.9)

# map
with st.expander('Map', expanded=False):
    lat = center[0]
    lon = center[1]
    fig = px.choropleth_mapbox(plot,
                            geojson=plot.geometry,
                            locations=plot.index,
                            title=f'Zones based on year {year}',
                            color=color,
                            #color_discrete_sequence=colormap,
                            range_color=(range_min, range_max),
                            color_continuous_scale=px.colors.sequential.Inferno[::-1],
                            center={"lat": lat, "lon": lon},
                            mapbox_style=my_style,
                            zoom=9,
                            opacity=0.5,
                            width=1200,
                            height=700
                            )

    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, height=700)

    st.plotly_chart(fig, use_container_width=True)

# map
with st.expander('Graphs', expanded=True):
    quant = st.radio('Set quantile to check',[10,25,50,75,90],horizontal=True)
    plot = plot.h3.cell_area(unit='m^2')
    yr_list = ['90','00','10','20']
    for yr in yr_list:
        plot[f'den_pop{yr}'] = round(plot[f'pop{yr}'] / (plot['h3_cell_area']/10000),-1)
        plot[f'den_gfa{yr}'] = round(plot[f'GFA{yr}'] / plot['h3_cell_area'],3)

    #growth goes..
    def growth_df(df,q=0.9):
        d = {'Year': [1990,2000,2010,2020],
            'Population':[df['pop90'].quantile(q),df['pop00'].quantile(q),df['pop10'].quantile(q),df['pop20'].quantile(q)],
            'GFA': [df['GFA90'].quantile(q),df['GFA00'].quantile(q),df['GFA10'].quantile(q),df['GFA20'].quantile(q)],
            'Density (e)': [plot['den_gfa90'].quantile(q),plot['den_gfa00'].quantile(q),plot['den_gfa10'].quantile(q),plot['den_gfa20'].quantile(q)],
            'Density (pop)': [plot['den_pop90'].quantile(q),plot['den_pop00'].quantile(q),plot['den_pop10'].quantile(q),plot['den_pop20'].quantile(q)]
            }
        dfg = pd.DataFrame(data=d)
        return dfg
    
    dfg = growth_df(plot,q=quant/100)
    #st.dataframe(dfg)

    #growth plot
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    figg = make_subplots(specs=[[{"secondary_y": True}]])
    # Add traces https://plotly.com/python/multiple-axes/#multiple-axes
    figg.add_trace(
        go.Scatter(x=dfg['Year'], y=dfg['Population'], name="Population"),
        secondary_y=False,
    )
    figg.add_trace(
        go.Scatter(x=dfg['Year'], y=dfg['Density (pop)'], name="Density"),
        secondary_y=True,
    )
    figg.update_layout(
        title_text=f"Growth in the '{zone}' (quantile {quant}%)"
    )
    figg.update_xaxes(title_text="Year")
    figg.update_yaxes(title_text="Population", range=[0,dfg['Population'].max()*1.1], secondary_y=False)
    figg.update_yaxes(title_text="Density (pop/ha)",range=[0,dfg['Density (pop)'].max()*1.1], secondary_y=True)
    st.plotly_chart(figg, use_container_width=True)

    def gshare(plot,quant):
        g90 = round(plot.loc[plot['pop90'] > plot['pop90'].quantile(quant/100)]['pop90'].sum(),-3)
        g00 = round(plot.loc[plot['pop00'] > plot['pop00'].quantile(quant/100)]['pop00'].sum(),-3)
        g10 = round(plot.loc[plot['pop10'] > plot['pop10'].quantile(quant/100)]['pop10'].sum(),-3)
        g20 = round(plot.loc[plot['pop20'] > plot['pop20'].quantile(quant/100)]['pop20'].sum(),-3)
        d = {
            'year': ['1990','2000','2010','2020'],
            'share %': [round(g90/plot['pop90'].sum(),2)*100,
                      round(g00/plot['pop00'].sum(),2)*100,
                      round(g10/plot['pop10'].sum(),2)*100,
                      round(g20/plot['pop20'].sum(),2)*100]
        }
        df = pd.DataFrame(data=d)
        return df
    g_shares = gshare(plot=plot,quant=quant)
    st.markdown(f"**Share of total population above the cells of the quantile {quant}% in the zone '{zone}'**")
    st.dataframe(g_shares.set_index('year').T)


#footer
st.markdown('---')
footer_title = '''
**Naked Density Project**
[![MIT license](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/teemuja/NDP/blob/main/LICENSE) 
'''
st.markdown(footer_title)
disclamer = 'Data papers are constant work in progress and will be upgraded, changed & fixed for errors while research go on.'
st.caption('Disclaimer: ' + disclamer)