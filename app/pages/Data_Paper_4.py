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
c1,c2,c3 = st.columns(3)
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

# map
with st.expander('Map', expanded=False):
    #feature to plot
    feat = st.radio('Density of',['Population','GFA'], horizontal=True)
    #calc densities
    plot = plot.h3.cell_area(unit='m^2')
    yr_list = ['90','00','10','20']
    for yr in yr_list:
        plot[f'den_pop{yr}'] = round(plot[f'pop{yr}'] / (plot['h3_cell_area']/10000),-1)
        plot[f'den_gfa{yr}'] = round(plot[f'GFA{yr}'] / plot['h3_cell_area'],3)
        plot[f'class{yr}'] = 'less'
        if feat == 'Population':
            #color = f'pop{year[-2:]}'
            #plot = plot.loc[plot[color] > plot[color].quantile(0.1)]
            plot.loc[plot[f'den_pop{yr}'] > 1, f'class{yr}'] = 'sprawl'
            plot.loc[plot[f'den_pop{yr}'] > 10, f'class{yr}'] = 'spacious'
            plot.loc[plot[f'den_pop{yr}'] > 50, f'class{yr}'] = 'compact'
            plot.loc[plot[f'den_pop{yr}'] > 70, f'class{yr}'] = 'dense'
        else:
            #color = f'GFA{year[-2:]}'
            #plot = plot.loc[plot[color] > plot[color].quantile(0.1)]
            plot.loc[plot[f'den_gfa{yr}'] > 0.10, f'class{yr}'] = 'sprawl'
            plot.loc[plot[f'den_gfa{yr}'] > 0.15, f'class{yr}'] = 'spacious'
            plot.loc[plot[f'den_gfa{yr}'] > 0.30, f'class{yr}'] = 'compact'
            plot.loc[plot[f'den_gfa{yr}'] > 0.50, f'class{yr}'] = 'dense'

    colormap = {
        "dense": "darkgoldenrod",
        "compact": "darkolivegreen",
        "spacious": "lightgreen",
        "sprawl": "lightblue",
        "less":"lightcyan"
    }
    #range_min = plot[color].quantile(0.1)
    #range_max = plot[color].quantile(0.9)

    lat = center[0]
    lon = center[1]
    fig = px.choropleth_mapbox(plot,
                            geojson=plot.geometry,
                            locations=plot.index,
                            title=f"Zone '{zone}' based on year {year}",
                            color=f'class{yr}',
                            hover_data=[f'den_pop{yr}',f'den_gfa{yr}'],
                            color_discrete_map=colormap,
                            labels={f'class{yr}': f'Density of {feat}'},
                            category_orders={f'class{yr}': ['dense','compact','spacious','sprawl']},
                            #range_color=(range_min, range_max),
                            #color_continuous_scale=px.colors.sequential.Inferno[::-1],
                            center={"lat": lat, "lon": lon},
                            mapbox_style=my_style,
                            zoom=9,
                            opacity=0.5,
                            width=1200,
                            height=700
                            )

    fig.update_layout(margin={"r": 10, "t": 50, "l": 10, "b": 10}, height=700,
                                legend=dict(
                                    yanchor="top",
                                    y=0.97,
                                    xanchor="left",
                                    x=0.02
                                )
                                )

    st.plotly_chart(fig, use_container_width=True)

# map
with st.expander('Graphs', expanded=True):

    #growth plot
    import plotly.graph_objects as go
    # Add traces https://plotly.com/python/multiple-axes/#multiple-axes

    # func to generate pop shares by quantiles for each year
    def qshare(plot):
        q_list = [90,75,50,25,10]
        q_dfs = []
        for q in q_list:
            g90 = round(plot.loc[plot['pop90'] > plot['pop90'].quantile(q/100)]['pop90'].sum(),0)
            g00 = round(plot.loc[plot['pop00'] > plot['pop00'].quantile(q/100)]['pop00'].sum(),0)
            g10 = round(plot.loc[plot['pop10'] > plot['pop10'].quantile(q/100)]['pop10'].sum(),0)
            g20 = round(plot.loc[plot['pop20'] > plot['pop20'].quantile(q/100)]['pop20'].sum(),0)
            d = {
                #'year': ['1990','2000','2010','2020'],
                f'share_{q}': [round(g90/plot['pop90'].sum(),2)*100,
                               round(g00/plot['pop00'].sum(),2)*100,
                               round(g10/plot['pop10'].sum(),2)*100,
                               round(g20/plot['pop20'].sum(),2)*100]
            }
            q_df = pd.DataFrame(data=d, index=['1990','2000','2010','2020'])
            q_dfs.append(q_df)

        #dfs = [df.set_index('year') for df in q_dfs]
        df_out = pd.concat(q_dfs, axis=1)
        return df_out
    
    q_shares = qshare(plot=plot)

    def share_plot(df):
        linecolors = px.colors.qualitative.Plotly
        fig = go.Figure()
        fig.add_traces(go.Scatter(x=df.index, y = df['share_90'],name='90%', mode = 'lines', line=dict(color=linecolors[0])))
        fig.add_traces(go.Scatter(x=df.index, y = df['share_75'],name='75%', mode = 'lines', line=dict(color=linecolors[1])))
        fig.add_traces(go.Scatter(x=df.index, y = df['share_50'],name='50%', mode = 'lines', line=dict(color=linecolors[2])))
        fig.add_traces(go.Scatter(x=df.index, y = df['share_25'],name='25%', mode = 'lines', line=dict(color=linecolors[3])))
        fig.add_traces(go.Scatter(x=df.index, y = df['share_10'],name='10%', mode = 'lines', line=dict(color=linecolors[4])))
        fig.update_layout(title_text=f"Change of population share in density quantiles in the '{zone}' ")
        fig.update_xaxes(title='Year')
        fig.update_yaxes(title='% of total population above quantile')
        return fig
    p1,p2 =  st.columns(2)
    p1.plotly_chart(share_plot(q_shares), use_container_width=True)
    p2.markdown('###')
    p2.markdown('###')
    p2.markdown('###')
    p2.markdown('###')
    p2.markdown(f'Shares in resolution H{reso}')
    p2.dataframe(q_shares.T)

#footer
st.markdown('---')
footer_title = '''
**Naked Density Project**
[![MIT license](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/teemuja/NDP/blob/main/LICENSE) 
'''
st.markdown(footer_title)
disclamer = 'Data papers are constant work in progress and will be upgraded, changed & fixed for errors while research go on.'
st.caption('Disclaimer: ' + disclamer)