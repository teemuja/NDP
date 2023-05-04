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
zone = c2.selectbox('Zone',['Tiheä taajama','Harva taajama','Kylät ja maaseutu'])
#reso
resoh = c3.radio('Resolution',['H9','H8','H7'], horizontal=True) #['~5km²','~1km²','~1ha']
reso = int(resoh[-1])

#legend
keys = {
    'Tiheä taajama':1,
    'Harva taajama':2,
    'Kylät':3,
    'Pienkylät':4,
    'Maaseutuasutus':5
    }
# filter
if zone != 'Kylät ja maaseutu':
    plot = mygdf.loc[mygdf[yrs[year]] == keys[zone]].h3.geo_to_h3_aggregate(reso)
else:
    subzones = ['Kylät','Pienkylät','Maaseutuasutus'] #=[3,4,5]
    plot = mygdf.loc[mygdf[yrs[year]].isin([3,4,5])].h3.geo_to_h3_aggregate(reso)

# map
with st.expander('Map', expanded=False):
    #feature to plot
    densityof = st.radio('Density of',['Population','GFA'], horizontal=True)
    #calc densities
    plot = plot.h3.cell_area(unit='m^2')
    yr_list = ['90','00','10','20']
    for yr in yr_list:
        plot[f'den_pop{yr}'] = round(plot[f'pop{yr}'] / (plot['h3_cell_area']/10000),-1)
        plot[f'den_gfa{yr}'] = round(plot[f'GFA{yr}'] / plot['h3_cell_area'],3)
        plot[f'class_pop{yr}'] = 'less'
        plot[f'class_gfa{yr}'] = 'less'
        #popdens
        plot.loc[plot[f'den_pop{yr}'] > 1, f'class_pop{yr}'] = 'sprawl'
        plot.loc[plot[f'den_pop{yr}'] > 10, f'class_pop{yr}'] = 'spacious'
        plot.loc[plot[f'den_pop{yr}'] > 50, f'class_pop{yr}'] = 'compact'
        plot.loc[plot[f'den_pop{yr}'] > 70, f'class_pop{yr}'] = 'dense'
        #gfadense
        plot.loc[plot[f'den_gfa{yr}'] > 0.10, f'class_gfa{yr}'] = 'sprawl'
        plot.loc[plot[f'den_gfa{yr}'] > 0.15, f'class_gfa{yr}'] = 'spacious'
        plot.loc[plot[f'den_gfa{yr}'] > 0.30, f'class_gfa{yr}'] = 'compact'
        plot.loc[plot[f'den_gfa{yr}'] > 0.60, f'class_gfa{yr}'] = 'dense'

    colormap = {
        "dense": "darkgoldenrod",
        "compact": "darkolivegreen",
        "spacious": "lightgreen",
        "sprawl": "lightblue",
        "less":"lightcyan"
    }

    lat = center[0]
    lon = center[1]
    if densityof == 'Population':
        mycolor = f'class_pop{yr}'
        myclass = 'pop'
    else:
        mycolor = f'class_gfa{yr}'
        myclass = 'gfa'

    fig = px.choropleth_mapbox(plot,
                            geojson=plot.geometry,
                            locations=plot.index,
                            title=f"Zone '{zone}' based on year {year} in resolution H{reso}.",
                            color=mycolor,
                            hover_data=[f'den_pop{yr}',f'den_gfa{yr}'],
                            color_discrete_map=colormap,
                            labels={f'class_{myclass}{yr}': f'Density of {densityof}'},
                            category_orders={f'class_{myclass}{yr}': ['dense','compact','spacious','sprawl','less']},
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

    #st.dataframe(plot.drop(columns='geometry'))

# graphs
st.subheader(f"Population shares in density classes. ")
hex_area = round(plot['h3_cell_area'].mean()/10000,1)
if hex_area < 100:
    st.markdown(f'Zone "{zone}" in resolution H{reso} (hexagon area: {hex_area} ha)')
else:
    st.markdown(f'Zone "{zone}" in resolution H{reso} (hexagon area: {round(hex_area/100,1)} km²)')

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
    fig.update_layout(title_text=f"Population share in pop. density quantiles (H{reso}).")
    fig.update_xaxes(title='Year')
    fig.update_yaxes(title='% of total population above quantile')
    return fig

# pop class shares
def density_class_share(plot,feat='pop'):
    den_list = ["dense","compact","spacious","sprawl","less"]
    list_of_df_shares = []
    for d in den_list:
        pop_shares_in_class_d = []
        for yr in ['90','00','10','20']:
            popsum = plot.loc[plot[f'class_{feat}{yr}'] == d][f'pop{yr}'].sum()
            popshare_yr = round(popsum/plot[f'pop{yr}'].sum(),2)*100
            pop_shares_in_class_d.append(popshare_yr)
        # pop_shares_in_class_d -> df
        d = {f'share_{d}': pop_shares_in_class_d}
        df_shares_of_d = pd.DataFrame(data=d, index=['1990','2000','2010','2020'])
        list_of_df_shares.append(df_shares_of_d)

    df_out = pd.concat(list_of_df_shares, axis=1)
    return df_out

pop_shares = density_class_share(plot,feat='pop')
gfa_shares = density_class_share(plot,feat='gfa')

def pop_share_plot(df):
    linecolors = px.colors.qualitative.Plotly
    fig = go.Figure()
    fig.add_traces(go.Scatter(x=df.index, y = df['share_dense'],name='dense', mode = 'lines', line=dict(color=linecolors[0])))
    fig.add_traces(go.Scatter(x=df.index, y = df['share_compact'],name='compact', mode = 'lines', line=dict(color=linecolors[1])))
    fig.add_traces(go.Scatter(x=df.index, y = df['share_spacious'],name='spacious', mode = 'lines', line=dict(color=linecolors[2])))
    fig.add_traces(go.Scatter(x=df.index, y = df['share_sprawl'],name='sprawl', mode = 'lines', line=dict(color=linecolors[3])))
    fig.add_traces(go.Scatter(x=df.index, y = df['share_less'],name='less', mode = 'lines', line=dict(color=linecolors[4])))
    fig.update_layout(title_text=f"Population share by pop. density classes (H{reso}).")
    fig.update_xaxes(title='Year')
    fig.update_yaxes(title='% of total population in class')
    return fig

def gfa_share_plot(df):
    linecolors = px.colors.qualitative.Plotly
    fig = go.Figure()
    fig.add_traces(go.Scatter(x=df.index, y = df['share_dense'],name='dense', mode = 'lines', line=dict(color=linecolors[0])))
    fig.add_traces(go.Scatter(x=df.index, y = df['share_compact'],name='compact', mode = 'lines', line=dict(color=linecolors[1])))
    fig.add_traces(go.Scatter(x=df.index, y = df['share_spacious'],name='spacious', mode = 'lines', line=dict(color=linecolors[2])))
    fig.add_traces(go.Scatter(x=df.index, y = df['share_sprawl'],name='sprawl', mode = 'lines', line=dict(color=linecolors[3])))
    fig.add_traces(go.Scatter(x=df.index, y = df['share_less'],name='less', mode = 'lines', line=dict(color=linecolors[4])))
    fig.update_layout(title_text=f"Population share by GFA density classes (H{reso}).")
    fig.update_xaxes(title='Year')
    fig.update_yaxes(title='% of total population in class')
    return fig

#plots in tabs
tab1,tab2 =  st.tabs(['In pop density classes','In GFA density classes'])
with tab1:
    st.plotly_chart(pop_share_plot(pop_shares), use_container_width=True)
with tab2:
    st.plotly_chart(gfa_share_plot(gfa_shares), use_container_width=True)

selite = """
<b>Density classification:</b><br>
<i>
Dense: e > 0.6 | pop/ha > 70 <br>
Compact: 0.6 - 0.3 | 70 - 50 <br>
Spacious: 0.3 - 0.15 | 50 - 10 <br>
Sprawl: 0.15 - 0.10 | 10 - 2 <br>
Less: e < 0.10 | pop/ha < 2 <br>
</i>
<br>
"""
st.markdown(selite, unsafe_allow_html=True)

#footer
st.markdown('---')
footer_title = '''
**Naked Density Project**
[![MIT license](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/teemuja/NDP/blob/main/LICENSE) 
'''
st.markdown(footer_title)
disclamer = 'Data papers are constant work in progress and will be upgraded, changed & fixed for errors while research go on.'
st.caption('Disclaimer: ' + disclamer)