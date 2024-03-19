# NDP app always beta a lot
import pandas as pd
import geopandas as gpd
import numpy as np
import streamlit as st
import h3pandas as h3
import numpy as np
from pathlib import Path
from shapely import wkt
import json
import geocoder
from geopandas import points_from_xy
from owslib.wfs import WebFeatureService
from owslib.fes import *
from owslib.etree import etree
import plotly.express as px
px.set_mapbox_access_token(st.secrets['MAPBOX_TOKEN'])
mbtoken = st.secrets['MAPBOX_TOKEN']
my_style = st.secrets['MAPBOX_STYLE']
import math
import statistics

# page setup ---------------------------------------------------------------
st.set_page_config(page_title="Data Paper #3", layout="wide", initial_sidebar_state='collapsed')
padding = 1
st.markdown(f""" <style>
    .reportview-container .main .block-container{{
        padding-top: {padding}rem;
        padding-right: {padding}rem;
        padding-left: {padding}rem;
        padding-bottom: {padding}rem;
    }} </style> """, unsafe_allow_html=True)
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
        NDP Data Paper 4 V1.1 \
        </p>'
st.markdown(header, unsafe_allow_html=True)
# plot size setup
#px.defaults.width = 600
px.defaults.height = 700

st.markdown("----")
# content
st.title("Data Paper #4")

st.markdown('---')
path = Path(__file__).parent / 'data/kunta_dict.csv'
with path.open() as f1:
    kuntakoodit = pd.read_csv(f1, index_col=False, header=0).astype(str)
kuntakoodit['koodi'] = kuntakoodit['koodi'].str.zfill(3)
kuntalista = kuntakoodit['kunta'].tolist()
#st.title(':point_down:')
st.title('Väestötiheyden kehitys')
st.subheader('Tällä tutkimusappilla voit katsoa, miten seudun väestötiheys on kehittynyt.')
# kuntavalitsin
valinnat = st.multiselect('Valitse kunnat (max 7) - kattavuus koko Suomi', kuntalista, default=['Helsinki','Espoo','Vantaa'])
st.caption('Ensin valittua käytetään väestögradientin keskipisteenä.')
vuodet = st.slider('Aseta aikajakso',2010, 2022, (2015, 2022),step=1)
#st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
k = st.empty()

#statgrid change
@st.cache_data()
def muutos_h3(kunta_list,y1=2015,y2=2022): #h3 resolution 7 for 1x1km census grid
    url = 'http://geo.stat.fi/geoserver/vaestoruutu/wfs'
    wfs11 = WebFeatureService(url=url, version='1.1.0')
    path = Path(__file__).parent / 'data/kunta_dict.csv'
    with path.open() as f1:
        kuntakoodit = pd.read_csv(f1, index_col=False, header=0).astype(str)
    kuntakoodit['koodi'] = kuntakoodit['koodi'].str.zfill(3)
    kunta_dict_inv = pd.Series(kuntakoodit.koodi.values, index=kuntakoodit.kunta).to_dict()
    cols = ['grd_id','kunta','vaesto','ika_0_14','ika_65_','geometry']
    yrs = [y1,y2]
    # loop
    grid = pd.DataFrame()
    for kunta in kunta_list:
        koodi = kunta_dict_inv.get(kunta)
        filter = PropertyIsLike(propertyname='kunta', literal=koodi, wildCard='*')
        filterxml = etree.tostring(filter.toXML()).decode("utf-8")
        grid_kunta = pd.DataFrame()
        for y in yrs:
            response = wfs11.getfeature(typename=f'vaestoruutu:vaki{y}_1km_kp', filter=filterxml, outputFormat='json')
            griddata = gpd.read_file(response)[cols]
            griddata['vuosi'] = y
            grid_kunta = pd.concat([grid_kunta,griddata], ignore_index=True)
        grid = pd.concat([grid,grid_kunta], ignore_index=True)
    # yr pop
    grid.loc[grid['vuosi'] == y1,f'{y1}_tot'] = grid['vaesto']
    grid.loc[grid['vuosi'] == y2,f'{y2}_tot'] = grid['vaesto']
    # yr pop_lap
    grid.loc[grid['vuosi'] == y1,f'{y1}_lap'] = grid['ika_0_14']
    grid.loc[grid['vuosi'] == y2,f'{y2}_lap'] = grid['ika_0_14']
    # yr pop_van
    grid.loc[grid['vuosi'] == y1,f'{y1}_van'] = grid['ika_65_']
    grid.loc[grid['vuosi'] == y2,f'{y2}_van'] = grid['ika_65_']
    # prepare merge sum
    grid.replace({-1:0}, inplace=True)
    grid.loc[grid['vuosi'] == y1,'vaesto'] = -abs(grid['vaesto']) # make first yr value negative for merge sums below
    grid.loc[grid['vuosi'] == y1,'ika_0_14'] = -abs(grid['ika_0_14'])
    grid.loc[grid['vuosi'] == y1,'ika_65_'] = -abs(grid['ika_65_'])
    # count change with groupby..
    sums = grid.drop(columns='geometry').groupby(by='grd_id').sum().reset_index()
    sums_df = pd.merge(sums,grid[['grd_id','geometry']],on='grd_id')
    # create gdf
    sums_gdf = gpd.GeoDataFrame(sums_df,geometry='geometry',crs=3067)
    h3_out = sums_gdf.to_crs(4326).h3.geo_to_h3_aggregate(7)
    # count ratios of change
    h3_out['vaestosuht'] = round((h3_out['vaesto'] / h3_out[f'{y1}_tot'])*100,0)
    h3_out['ika_0_14suht'] = round((h3_out['ika_0_14'] / h3_out[f'{y1}_lap'])*100,0)
    h3_out['ika_65_suht'] = round((h3_out['ika_65_'] / h3_out[f'{y1}_van'])*100,0)

    return h3_out


#selectors
if len(valinnat) == 0:
    st.warning('Valitse ensin kunnat.')
    st.stop()
elif len(valinnat) > 7:
    st.warning('Voit valita max 7.')
    st.stop()
else:
    # generate regional data
    plot = muutos_h3(valinnat, y1=vuodet[0], y2=vuodet[1])
    
    # and scale cirlce
    try:
        loc = geocoder.osm(valinnat[0]) #eka kaupunki listalla
        ring = gpd.GeoDataFrame(pd.DataFrame(), geometry=points_from_xy(loc.lng, loc.lat))
        ring['geometry'] = ring.geometry.buffer(5000)
    except:
        ring = None

    # render map
    mapholder = st.empty()
    k1,k2 = st.columns([1,2])
    ratio = k1.checkbox('Näytä suhteellinen kasvu')
    plot_mode = k2.radio('Muutosdata',('vaesto','ika_0_14','ika_65_'),horizontal=True)

    # plot mode
    if ratio == 1:
        color_value = f'{plot_mode}suht'
    else:
        color_value = plot_mode
    
    # discrete colorscales
    bin_labels = ['taantumaa','hiipumaa','ei muutosta','karttumaa','kasvua','TOP3']
    color_col = color_value
    min1 = plot.loc[plot[color_col] < 0][color_col].quantile(0.75)
    min2 = plot.loc[plot[color_col] < 0][color_col].quantile(0.25)
    max1 = plot.loc[plot[color_col] > 0][color_col].quantile(0.25)
    max2 = plot.loc[plot[color_col] > 0][color_col].quantile(0.75)
    top3 = plot[color_col].sort_values(ascending = False).head(4).min()
    plot['Muutos'] = pd.cut(x=plot[color_col],bins=[-np.inf,min2,min1,max1,max2,top3,np.inf],labels=bin_labels)

    #colors
    bin_colors = {
        'taantumaa':'cornflowerblue',
        'hiipumaa':'lightblue',
        'ei muutosta':'ghostwhite',
        'karttumaa':'burlywood',
        'kasvua':'brown',
        'TOP3':'red'
    }

    # set range    
    #range_min = plot[color_value].quantile(0.05)
    #range_max = plot[color_value].quantile(0.95)
    #mid_point = abs(0 - range_min / (range_max - range_min))

    #if math.isnan(mid_point):
    #    mid_point = 0.5
    #    st.warning('Väriskaalahäiriö')

    #colorscale = [[0, 'rgba(100, 149, 237, 0.85)'],
    #                [mid_point, 'rgba(255, 255, 255, 0.85)'],
    #                [1, 'rgba(214, 39, 40, 0.85)']]

    #plot_mode
    if plot_mode == 'vaesto':
        graph_value = 'tot'
        value_title = 'kaikki ikäluokat'
    elif plot_mode == 'ika_0_14':
        graph_value = 'lap'
        value_title = 'lapset 0-14v'
    else:
        graph_value = 'van'
        value_title = 'väestö yli 65v'

    # plot
    lat = plot.unary_union.centroid.y
    lon = plot.unary_union.centroid.x
    fig = px.choropleth_mapbox(plot,
                                geojson=plot.geometry,
                                locations=plot.index,
                                title=f'Väestötiheyden muutos {vuodet[0]}-{vuodet[1]}, ({value_title})',
                                color='Muutos',
                                hover_data=['vaesto','ika_0_14','ika_65_','vaestosuht','ika_0_14suht','ika_65_suht'],
                                center={"lat": lat, "lon": lon},
                                mapbox_style=my_style,
                                color_discrete_map=bin_colors,
                                category_orders={'Muutos':bin_labels},
                                #color_continuous_scale=colorscale,
                                #range_color=(range_min,range_max),
                                labels={'vaesto':'Muutos','ika_0_14':'Muutos lapset','ika_65_':'Muutos vanh.',
                                        'vaestosuht':'Muutos%','ika_0_14suht':'Muutos% lapset','ika_65_suht':'Muutos% vanh.'
                                        },
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
    if ring is not None:
        fig.update_layout(
                    mapbox={
                        "layers": [
                            {
                                "source": json.loads(ring.to_crs(4326).to_json()),
                                #"below": "traces",
                                "type": "line",
                                "color": "black",
                                "line": {"width": 0.5,"dash":[5,5]},
                            }
                        ]
                    }
                )
    mapholder.plotly_chart(fig, use_container_width=True)

st.markdown('---')

# metrics
st.subheader('TOP3-kohteiden osuus kasvusta')
def topN_share(df,col,n=4):
    net_g = df[col].sum()
    #if net_g > 0:
    #    g_share = round((df[col].sort_values(ascending = False).head(n).sum() / net_g)*100,1)
    #else:
    g_share = round((df[df[col] > 0][col].sort_values(ascending = False).head(n).sum()) / (df[df[col] > 0][col].sum())*100,1)
    return round(net_g,-1),g_share

net,net_s = topN_share(plot,'vaesto')
lap,lap_n = topN_share(plot,'ika_0_14')
van,van_n = topN_share(plot,'ika_65_')

m1, m2, m3 = st.columns(3)
m1.metric(label="Väestökasvu", value=f"{net:.0f}", delta=f"TOP3 osuus {net_s}%")
m2.metric(label="Lapsikasvu", value=f"{lap:.0f}", delta=f"TOP3 osuus {lap_n}%")
m3.metric(label="Seniorikasvu", value=f"{van:.0f}", delta=f"TOP3 osuus {van_n}%")
st.caption('Kasvu on ko. ryhmän nettokasvu, mutta TOP3-alueiden osuus on laskettu vain kasvualueista, ei nettokasvusta.')
st.markdown('---')
# graph placeholder
st.subheader('Väestögradientti')
den_holder = st.empty()

def den_grad(_h3_df,center_add,value,reso=7,rings=7):
    # create center hex
    loc = geocoder.mapbox(center_add, key=mbtoken)
    center_df = pd.DataFrame({'lat':loc.lat,'lng':loc.lng},index=[0])
    h3_center = center_df.h3.geo_to_h3(reso)

    # create grad_df to sum medians from the rings
    grad_df = pd.DataFrame()
    grads = []
    popsums = []
    # create ring columns around center_df
    for i in range(1,rings+1):
        ring = pd.DataFrame()
        h3_center[f'h3_r{i}'] = h3_center.h3.k_ring(i)['h3_k_ring']
        ring[f'h3_0{reso}'] = h3_center[f'h3_r{i}'][0]
        ring[value] = 0 #np.NaN
        ring.set_index(f'h3_0{reso}', inplace=True)
        ring[value].update(_h3_df[value])
        # remove zeros
        ring = ring.loc[ring[value] != 0]
        # calc median
        popmedian = ring[value].median()
        popsum = ring[value].sum()
        grads.append(popmedian)
        popsums.append(popsum)
    grad_df['pop_median_km2'] = grads
    grad_df['pop_sum_ring'] = popsums
    # create ring names
    grad_df.reset_index(drop=False, inplace=True)
    grad_df.rename(columns={'index':'ring'}, inplace=True)
    grad_df['ring'] = 'R'+ grad_df['ring'].astype(str)
    return grad_df


# and density gradients + rings
den0 = den_grad(_h3_df=plot,center_add=valinnat[0],value=f'{vuodet[0]}_{graph_value}',reso=7,rings=16)
den1 = den_grad(_h3_df=plot,center_add=valinnat[0],value=f'{vuodet[1]}_{graph_value}',reso=7,rings=16)

# graph plotter
import plotly.graph_objects as go
def generate_den_graphs(den0,den1):
    # 
    def plot_muutos(df1,df2):
        fig = go.Figure(layout=go.Layout(title=go.layout.Title(text=f'Väestötiheys keskustaetäisyyden mukaan (mediaani, {value_title})')))
        fig.add_trace(go.Scatter(x=df1['ring'],y=df1['pop_median_km2'],name=f'{vuodet[0]}',
                                fill='tozeroy',fillcolor='rgba(222, 184, 135, 0.5)',
                                mode='lines', line=dict(width=0.5, color='rgb(0, 0, 0)')))
        fig.add_trace(go.Scatter(x=df2['ring'],y=df2['pop_median_km2'],name=f'{vuodet[1]}',
                                fill='tonexty',fillcolor='rgba(205, 127, 50, 0.5)', mode='none'))
        fig.update_xaxes(range=[1,15])
        fig.update_layout(margin={"r": 10, "t": 50, "l": 10, "b": 10}, height=500,
                                legend=dict(
                                    yanchor="top",
                                    y=0.97,
                                    xanchor="right",
                                    x=0.99
                                )
                                )
        fig.update_layout(shapes=[
                            dict(
                                type= 'line',
                                yref= 'paper', y0= 0, y1= 1,
                                xref= 'x', x0= 3, x1= 3,
                                line=dict(
                                            color="Black",
                                            width=0.5,
                                            dash="dash",
                                        )
                                )
                        ])
        return fig
    fig = plot_muutos(den0,den1)
    return st.plotly_chart(fig, use_container_width=True)

with den_holder:
    generate_den_graphs(den0,den1)
st.caption("data: [stat.fi](https://www.stat.fi/org/avoindata/paikkatietoaineistot/tilastoruudukko_1km.html)")
    
with st.expander('Selite',expanded=False):
    selite = ''' 
        Heksagonihila muodostuu h3geo.org -kirjaston heksagoneista, joihin on summattu asukasmäärät 
        kuhunkin heksagoniin osuvien 1x1km väestöruututietojen keskipisteiden mukaan (soveltuu vain seudulliseen tarkasteluun). Väestögradientti on muodostettu 
        heksagonikehistä (R1,R2,R3..) jotka ovat noin 2km välein. Katkoviivalla on merkitty n. 5km keskustaetäisyys. 
        Luokittelu on dynaaminen seudun arvojen ala- ja yläkvarttaalien mukaan:
        Heksat, joissa muutos on yli kasvun/vähenemän alakvarttaalin on luokiteltu karttumaksi/hiipumaksi ja 
        heksat, joissa muutos on yli yläkvarttaalien ovat vastaavasti kasvua/taantumaa.
        '''
    st.caption(selite)

#footer
st.markdown('---')
footer_title = '''
**NDP Project**
[![MIT license](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/teemuja/NDP/blob/main/LICENSE) 
'''
st.markdown(footer_title)
disclamer = 'Data papers are constant work in progress and will be upgraded, changed & fixed while research go on.'
st.caption('Disclaimer: ' + disclamer)