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
import json


# page setup ---------------------------------------------------------------
st.set_page_config(page_title="Data Paper #1", layout="wide")
padding = 1
st.markdown(f""" <style>
    .reportview-container .main .block-container{{
        padding-top: {padding}rem;
        padding-right: {padding}rem;
        padding-left: {padding}rem;
        padding-bottom: {padding}rem;
    }}
    </style> """, unsafe_allow_html=True)

header = '<p style="font-family:sans-serif; color:grey; font-size: 12px;">\
        NDP data paper #1 V0.9\
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
st.subheader("Correlation between urban density and amenities")
ingress = '''
<p style="font-family:sans-serif; color:Black; font-size: 14px;">
This data paper visualise the change in correlation between urban density and urban amenities.
Research quest here is to see how often used argument of positive density impact on local amenities in
urban planning works in different geographical scales. The research method is Pearson correlation calculations between
gross floor area (GFA) and urban amenities in different scales.
</p>
'''
st.markdown(ingress, unsafe_allow_html=True)
st.markdown("###")
# translate dict
eng_feat = {
    'kem_2000':'Total GFA in 2000',
    'askem_2000':'Residential GFA in 2000',
    'kem_2016':'Total GFA in 2016',
    'askem_2016':'Residential GFA in 2016',
    'kem_muutos':'GFA change 2000-2016',
    'askem_muutos':'Residential GFA change 2000-2016',
    'palv_pien_2000':'One person companies (OPC) in urban amenities in 2000',
    'palv_2000':'Urban amenities (OPC excluded) in 2000',
    'kaup_2000':'Wholesale and retail trade in 2000',
    'pt_2000':'Crocery stores and kiosks in 2000',
    'palv_pien_2016':'One person companies (OPC) in urban amenities in 2016',
    'palv_2016':'Urban amenities (OPC excluded) in 2016',
    'kaup_2016':'Wholesale and retail trade in 2016',
    'pt_2016':'Crocery stores and kiosks in 2016',
    'palv_pien_muutos':'Change in one person companies (OPC) in urban amenities 2000-2016',
    'palv_muutos':'Cange in Urban amenities (OPC excluded) 2000-2016',
    'kaup_muutos':'Change in wholesale and retail trade 2000-2016',
    'pt_muutos':'Change in Crocery stores and kiosks 2000-2016',
}

@st.cache(allow_output_mutation=True)
def load_data():
    path = Path(__file__).parent / 'data/h3_10_PKS.csv'
    with path.open() as f:
        data = pd.read_csv(f, index_col='h3_10', header=0)#.astype(str)
    # translate columns
    eng_data = data.rename(columns=eng_feat)
    return eng_data

gdf = load_data()

centre_pnos = [
"Helsinki keskusta - Etu-Töölö",
"Punavuori - Bulevardi",
"Kruununhaka",
"Kaartinkaupunki",
"Kaivopuisto - Ullanlinna",
"Punavuori - Eira - Hernesaari",
"Katajanokka",
"Kamppi - Ruoholahti",
"Jätkäsaari",
"Länsi-Pasila",
"Pohjois-Meilahti",
"Meilahden sairaala-alue",
"Taka-Töölö",
"Keski-Töölö",
"Munkkiniemi",
"Kallio",
"Vallila - Hermanni",
"Sörnäinen - Harju",
"Toukola - Kumpula - Vanhakaupunki",
"Kalasatama - Kyläsaari",
"Kalasatama - Sompasaari",
"Alppila - Vallila"
]

s1,s2 = st.columns(2)
pnolista = gdf['pno'].unique()
tapa = s1.selectbox('Select...',['By City','By Neighbourhood'])
if tapa == 'By City':
    kuntani = s2.selectbox(' ',['Helsinki','Espoo','Vantaa','Helsinki centre','Helsinki suburbs'])
    if kuntani == 'Helsinki centre':
        mygdf = gdf.loc[gdf.pno.isin(centre_pnos)]
    elif kuntani == 'Helsinki suburbs':
        mygdf = gdf.loc[gdf.kunta == 'Helsinki']
        mygdf = mygdf.loc[~mygdf.pno.isin(centre_pnos)]
    else:
        mygdf = gdf.loc[gdf.kunta == kuntani]
else:
    pnos = s2.multiselect(' ', pnolista,
                            default=['Tapiola','Pohjois-Tapiola','Otaniemi'])
    if pnos is not None:
        mygdf = gdf.loc[gdf.pno.isin(pnos)]
    else:
        st.warning('Select city or neighbourhoods.')
        st.stop()

# filters..
col_list = mygdf.drop(columns=['kunta','pno']).columns.to_list()
default_ix = col_list.index('Residential GFA in 2016')
p1,p2 = st.columns(2)
color = p1.selectbox('Filter by feature quantiles (%)', col_list, index=default_ix)
q_range = p2.slider(' ',0,100,(10,90),10)
mygdf = mygdf.loc[mygdf[f'{color}'].astype(int) > mygdf[f'{color}'].astype(int).quantile(q_range[0]/100)] 
mygdf = mygdf.loc[mygdf[f'{color}'].astype(int) < mygdf[f'{color}'].astype(int).quantile(q_range[1]/100)]
mapplace = st.empty()
l1,l2 = st.columns(2)
level = l1.slider('H3-resolution in map (H6-H9)',6,9,9,1)
l1.caption('https://h3geo.org/docs/core-library/restable/')
# map plot
if len(mygdf) > 1:
    plot = mygdf.h3.h3_to_parent_aggregate(level)
    # map plot
    lat = plot.unary_union.centroid.y
    lon = plot.unary_union.centroid.x
    range_min = plot[color].quantile(0.05)
    range_max = plot[color].quantile(0.95)
    fig = px.choropleth_mapbox(plot,
                            geojson=plot.geometry,
                            locations=plot.index,
                            color=color,
                            center={"lat": lat, "lon": lon},
                            mapbox_style=my_style,
                            range_color=(range_min, range_max),
                            color_continuous_scale=px.colors.sequential.Inferno[::-1],
                            #color_continuous_scale=px.colors.sequential.Blackbody[::-1],
                            #labels={'palv':'Palveluiden määrä'},
                            zoom=9,
                            opacity=0.5,
                            width=1200,
                            height=700
                            )
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, height=700)
    fig.update_layout(coloraxis_showscale=False)
    with mapplace:
        st.plotly_chart(fig, use_container_width=True)
    
else:
    st.stop()

# scat plot
import plotly.graph_objects as go
from plotly.subplots import make_subplots
df = plot.copy()
trace1 = go.Scatter(
    x=df['Residential GFA in 2000'],
    y=df['Crocery stores and kiosks in 2000'],
    name='2000',
    mode='markers'
)
trace2 = go.Scatter(
    x=df['Residential GFA in 2016'],
    y=df['Crocery stores and kiosks in 2016'],
    name='2016',
    yaxis='y2',
    mode='markers'
)
scat = make_subplots(specs=[[{"secondary_y": True}]],
                        x_title='Residential GFA in area',y_title='Crocery stores and kiosks in area')
scat.add_trace(trace1)
scat.add_trace(trace2,secondary_y=True)
scat.update_layout(title=f'Number of croceries&kiosks vs residential GFA on resolution {level}')

#scat = px.scatter(plot, x=x, y=y)
st.plotly_chart(scat, use_container_width=True)

st.markdown('---')

# corr graphs
st.subheader('Correlation loss')

@st.cache(allow_output_mutation=True)
def corr_loss(df,h=10,corr_type='year'):
    if corr_type == '2000':
        x_list=['Total GFA in 2000',
                'Residential GFA in 2000']
        y_list=['One person companies (OPC) in urban amenities in 2000',
                'Urban amenities (OPC excluded) in 2000',
                'Wholesale and retail trade in 2000',
                'Crocery stores and kiosks in 2000']
    elif corr_type == '2016':
        x_list=['Total GFA in 2016',
                'Residential GFA in 2016']
        y_list=['One person companies (OPC) in urban amenities in 2016',
                'Urban amenities (OPC excluded) in 2016',
                'Wholesale and retail trade in 2016',
                'Crocery stores and kiosks in 2016']
    elif corr_type == 'year':
        x_list=['Total GFA',
                'Residential GFA']
        y_list=['One person companies (OPC) in urban amenities',
                'Urban amenities (OPC excluded)',
                'Wholesale and retail trade',
                'Crocery stores and kiosks']
    elif corr_type == 'change':
        x_list=['GFA change 2000-2016',
                'Residential GFA change 2000-2016']
        y_list=['Change in one person companies (OPC) in urban amenities 2000-2016',
                'Cange in Urban amenities (OPC excluded) 2000-2016',
                'Change in wholesale and retail trade 2000-2016',
                'Change in Crocery stores and kiosks 2000-2016',]
        
    frames = []
    for x in x_list:
        for y in y_list:
            corr_list = []
            for i in range(1,5):
                df_i = df.h3.h3_to_parent_aggregate(h-i,return_geometry=False)
                corr_i = df_i.corr()[x][y]
                corr_list.append(corr_i)
            corr_y = pd.DataFrame(corr_list, index=['h9','h8','h7','h6'], columns=[x+' VS '+y])
            frames.append(corr_y)
    corr_df = pd.concat(frames, axis=1, ignore_index=False)
    return corr_df

# data in use for corr
if tapa == 'By City':
    st.caption(f'Data in use: {color} -value quantiles {q_range[0]}-{q_range[1]}% in {kuntani}')
    graph_title = kuntani
else:
    st.caption(f'Data in use: {color} -value quantiles {q_range[0]}-{q_range[1]}% in neighbourhoods {pnos}')
    graph_title = pnos
st.caption('Click the legend to select/unselect correlation pairs. Save the graph using camera-icon when howered over.')
# corrs
# use similar col names for facet plot
facet_feat = {
    'Total GFA in 2000':'Total GFA',
    'Total GFA in 2016':'Total GFA',
    'Residential GFA in 2000':'Residential GFA',
    'Residential GFA in 2016':'Residential GFA',
    'One person companies (OPC) in urban amenities in 2000':'One person companies (OPC) in urban amenities',
    'One person companies (OPC) in urban amenities in 2016':'One person companies (OPC) in urban amenities',
    'Urban amenities (OPC excluded) in 2000':'Urban amenities (OPC excluded)',
    'Urban amenities (OPC excluded) in 2016':'Urban amenities (OPC excluded)',
    'Wholesale and retail trade in 2000':'Wholesale and retail trade',
    'Wholesale and retail trade in 2016':'Wholesale and retail trade',
    'Crocery stores and kiosks in 2000':'Crocery stores and kiosks',
    'Crocery stores and kiosks in 2016':'Crocery stores and kiosks',
}
facet_col_list_2000 = [
    'Total GFA in 2000',
    'Residential GFA in 2000',
    'One person companies (OPC) in urban amenities in 2000',
    'Urban amenities (OPC excluded) in 2000',
    'Wholesale and retail trade in 2000',
    'Crocery stores and kiosks in 2000'
]
facet_col_list_2016 = [
    'Total GFA in 2016',
    'Residential GFA in 2016',
    'One person companies (OPC) in urban amenities in 2016',
    'Urban amenities (OPC excluded) in 2016',
    'Wholesale and retail trade in 2016',
    'Crocery stores and kiosks in 2016'
]
corr_2000 = corr_loss(mygdf[facet_col_list_2000].rename(columns=facet_feat),corr_type='year')
corr_2000['year'] = 2000
corr_2016 = corr_loss(mygdf[facet_col_list_2016].rename(columns=facet_feat),corr_type='year')
corr_2016['year'] = 2016
corrs = corr_2000.append(corr_2016)

fig_corr = px.line(corrs,
                   labels = {'index':'H3-resolution','value':'Correlation','variable':'Corr pairs'},
                   title=f'Correlation loss in {graph_title}', facet_col='year' )
fig_corr.update_xaxes(autorange="reversed")#, side='top')
fig_corr.update_layout(legend=dict(orientation="h",yanchor="bottom",y=-0.5,xanchor="right",x=1))
st.plotly_chart(fig_corr, use_container_width=True)
        
with st.expander('Classification', expanded=False):        
    class_expl = """
    **Urban amenities** are all company business space locations which belong
    to the following finnish TOL-industry classes:  
    _Wholesale and retail_  
    _Accomondation and food service activites_  
    _Information and communication_  
    _Financial and insurance activities_  
    _Other service activities_  

    Original raw data is from
    <a href="https://research.aalto.fi/fi/projects/l%C3%A4hi%C3%B6iden-kehityssuunnat-ja-uudelleenkonseptointi-2020-luvun-segr " target="_blank">Re:Urbia</a>
    -research project data retrieved from the data products "SeutuCD 2002" and "SeutuCD 2018" by Statistical Finland.
    """
    st.markdown(class_expl, unsafe_allow_html=True)

#footer
footer_title = '''
---
**Naked Density Project**
[![MIT license](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/teemuja/ndp_app4/blob/main/LICENSE) 
'''
st.markdown(footer_title)