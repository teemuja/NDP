# NDP app always beta a lot
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
import json


# page setup ---------------------------------------------------------------
st.set_page_config(page_title="Data Paper #2", layout="wide", initial_sidebar_state='collapsed')
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
        NDP data paper #2 V0.91\
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
st.title("Data Paper #2")
st.subheader("Correlation between urban density and amenities")
ingress = '''
<p style="font-family:sans-serif; color:Black; font-size: 14px;">
This data paper visualise the change in correlation between urban density and urban amenities.
Research quest here is to see how an often used argument of positive density impact on local amenities in
urban planning works in different geographical scales.
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
    'palv_pien_2000':'One person companies (OPC) in urban amenities in 2000',
    'palv_suur_2000':'Urban amenities (OPC excluded) in 2000',
    'kaup_2000':'Wholesale and retail trade in 2000',
    'pt_sup_2000':'Consumer daily goods and kiosks in 2000',
    'pt_laaja_2000':'Retail trade in 2000',
    'palv_pien_2016':'One person companies (OPC) in urban amenities in 2016',
    'palv_suur_2016':'Urban amenities (OPC excluded) in 2016',
    'kaup_2016':'Wholesale and retail trade in 2016',
    'pt_sup_2016':'Consumer daily goods and kiosks in 2016',
    'pt_laaja_2016':'Retail trade in 2016'
}

@st.cache_data()
def load_data():
    path = Path(__file__).parent / 'data/h3_10_PKS_kem2_VS_palv.csv'
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
col_list_all = mygdf.drop(columns=['kunta','pno']).columns.to_list()
# removelist = if needed
#col_list = [i for i in col_list_all if i not in remove_list]
default_ix = col_list_all.index('Residential GFA in 2016')
p1,p2,p3 = st.columns(3)
color = p1.selectbox('Filter by feature quantiles (%)', col_list_all, index=default_ix)
q_range = p2.slider(' ',0,100,(0,100),10)
level = p3.slider('..and/or by H3-resolution (H6-H9)',6,9,9,1)
p3.caption('https://h3geo.org/docs/core-library/restable/')
mygdf = mygdf.loc[mygdf[f'{color}'].astype(int) > mygdf[f'{color}'].astype(int).quantile(q_range[0]/100)] 
mygdf = mygdf.loc[mygdf[f'{color}'].astype(int) < mygdf[f'{color}'].astype(int).quantile(q_range[1]/100)]

# the map
with st.expander('Filtered data on map', expanded=False):
    mapplace = st.empty()
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
                                zoom=10,
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


# corr graphs
st.subheader('Correlation loss')

@st.cache_data()
def corr_loss(df,h=10,corr_type='year',method='pearson'):
    if corr_type == '2000':
        x_list=['Total GFA in 2000',
                'Residential GFA in 2000']
        y_list=['One person companies (OPC) in urban amenities in 2000',
                'Urban amenities (OPC excluded) in 2000',
                'Wholesale and retail trade in 2000',
                'Consumer daily goods and kiosks in 2000',
                'Retail trade in 2000']
    elif corr_type == '2016':
        x_list=['Total GFA in 2016',
                'Residential GFA in 2016']
        y_list=['One person companies (OPC) in urban amenities in 2016',
                'Urban amenities (OPC excluded) in 2016',
                'Consumer daily goods and kiosks in 2016',
                'Retail trade in 2016']
    elif corr_type == 'year':
        x_list=['Total GFA',
                'Residential GFA']
        y_list=['One person companies (OPC) in urban amenities',
                'Urban amenities (OPC excluded)',
                'Wholesale and retail trade',
                'Consumer daily goods and kiosks',
                'Retail trade']

    # prepare corrs    
    frames = []
    for x in x_list:
        for y in y_list:
            corr_list = []
            for i in range(1,5):
                df_i = df.h3.h3_to_parent_aggregate(h-i,return_geometry=False)
                # apply box cox transform to values..
                if method == 'pearson':
                    # use only positive values as boxcox applied..
                    df_i = df_i[(df_i[df_i.columns] > 0).all(axis=1)]
                    df_i[x] = boxcox(df_i[x])[0] #boxcox return two: transformed data and lambda value
                    df_i[y] = boxcox(df_i[y])[0]
                # use as transformed
                corr_i = df_i.corr(method=method)[x][y]
                corr_list.append(corr_i)
            corr_y = pd.DataFrame(corr_list, index=['h9','h8','h7','h6'], columns=[x+' VS '+y])
            frames.append(corr_y)
    corr_df = pd.concat(frames, axis=1, ignore_index=False)
    return corr_df


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
    'Consumer daily goods and kiosks in 2000':'Consumer daily goods and kiosks',
    'Consumer daily goods and kiosks in 2016':'Consumer daily goods and kiosks',
    'Retail trade in 2000':'Retail trade',
    'Retail trade in 2016':'Retail trade'
}
facet_col_list_2000 = [
    'Total GFA in 2000',
    'Residential GFA in 2000',
    'One person companies (OPC) in urban amenities in 2000',
    'Urban amenities (OPC excluded) in 2000',
    'Wholesale and retail trade in 2000',
    'Consumer daily goods and kiosks in 2000',
    'Retail trade in 2000'
]
facet_col_list_2016 = [
    'Total GFA in 2016',
    'Residential GFA in 2016',
    'One person companies (OPC) in urban amenities in 2016',
    'Urban amenities (OPC excluded) in 2016',
    'Wholesale and retail trade in 2016',
    'Consumer daily goods and kiosks in 2016',
    'Retail trade in 2016'
]

my_method = st.radio('Correlation method',('pearson','spearman'))
corr_2000 = corr_loss(mygdf[facet_col_list_2000].rename(columns=facet_feat),corr_type='year',method=my_method)
corr_2000['year'] = 2000
corr_2016 = corr_loss(mygdf[facet_col_list_2016].rename(columns=facet_feat),corr_type='year',method=my_method)
corr_2016['year'] = 2016
corrs = corr_2000.append(corr_2016)

# select feat for corrs
plot_list = corrs.columns.to_list()[:-1]
my_plot_list = ['Residential GFA VS Urban amenities (OPC excluded)',
                'Residential GFA VS Consumer daily goods and kiosks',
                'Residential GFA VS Retail trade'
                ]
scat_list = st.multiselect('Choose data for the correlation plot', plot_list,default=my_plot_list)
if len(scat_list) > 0:
    scat_list.extend(['year'])
    corr_plot = corrs[corrs.columns.intersection(scat_list)]
else:
    st.stop()

# data in use for corr
if tapa == 'By City':
    st.caption(f'Data in use: {color} -value quantiles {q_range[0]}-{q_range[1]}% in {kuntani}')
    graph_title = kuntani
else:
    st.caption(f'Data in use: {color} -value quantiles {q_range[0]}-{q_range[1]}% in neighbourhoods {pnos}')
    graph_title = pnos

# plot
fig_corr = px.line(corr_plot,
                   labels = {'index':'H3-resolution','value':'Correlation','variable':'Correlation pairs'},
                   title=f'Correlation loss in {graph_title}', facet_col='year' )
fig_corr.update_xaxes(autorange="reversed")#, side='top')
fig_corr['layout'].update(shapes=[{'type': 'line','y0':0.5,'y1': 0.5,'x0':str(corr_plot.index[0]), 
                              'x1':str(corr_plot.index[-1]),'xref':'x1','yref':'y1',
                              'line': {'color': 'black','width': 0.5,'dash':'dash'}},
                             {'type': 'line','y0':0.5,'y1': 0.5,'x0':str(corr_plot.index[0]), 
                              'x1':str(corr_plot.index[-1]),'xref':'x2','yref':'y2',
                              'line': {'color': 'black','width': 0.5,'dash':'dash'}}])
fig_corr.update_layout(yaxis_range=[0,1])
#fig_corr.update_layout(legend=dict(orientation="h",yanchor="bottom",y=-0.2,xanchor="left",x=0))
st.plotly_chart(fig_corr, use_container_width=True)

with st.expander('Statistical checks', expanded=False):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    df = plot.copy()
    # select feat
    x1,y1 = st.columns(2)
    xvalue = x1.selectbox('Choose X axis data',['Residential GFA','Total GFA'])
    yvalue = y1.selectbox('Choose y axis data',['Urban amenities (OPC excluded)',
                                                'One person companies (OPC) in urban amenities',
                                                'Consumer daily goods and kiosks',
                                                'Retail trade',
                                                'Wholesale and retail trade'])
    st.caption('Change H3-resolution for scatter plot using filter selectors')
    # plots
    trace1 = go.Scatter(
        x=df[f'{xvalue} in 2000'],
        y=df[f'{yvalue} in 2000'],
        name='2000',
        mode='markers',
        marker=dict(
                color='Brown',
                size=7)
    )
    trace2 = go.Scatter(
        x=df[f'{xvalue} in 2016'],
        y=df[f'{yvalue} in 2016'],
        name='2016',
        yaxis='y2',
        mode='markers',
        marker=dict(
                color='Orange',
                size=7)
    )
    scat = make_subplots(specs=[[{"secondary_y": True}]],
                            x_title=f'{xvalue}',y_title=f'{yvalue}')
    scat.add_trace(trace1)
    scat.add_trace(trace2,secondary_y=True)
    if 'kuntani' not in globals():
        kuntani = 'selected neighborhoods'
    scat.update_layout(title=f'Scatter plot on resolution H{level} in {kuntani}')

    st.plotly_chart(scat, use_container_width=True)

    #check
    yvalue_2000sum = df[f'{yvalue} in 2000'].sum()
    yvalue_2016sum = df[f'{yvalue} in 2016'].sum()
    yvalue_2000max = df[f'{yvalue} in 2000'].max() #.quantile(0.9)
    yvalue_2016max = df[f'{yvalue} in 2016'].max() #.quantile(0.9)
    m1,m2 = st.columns(2)
    m1.metric(label=f"Sum of {yvalue} in 2000", value=f"{yvalue_2000sum}", delta=f"max_H{level}: {yvalue_2000max}")
    m2.metric(label=f"Sum of {yvalue} in 2016", value=f"{yvalue_2016sum}", delta=f"max_H{level}: {yvalue_2016max}")
    st.markdown('---')

    # histogram for data in current resollution level
    df_ = df[(df.T != 0).any()].drop(columns='geometry')
    #x2000 = go.Histogram(x=df_[f'{xvalue} in 2000'],opacity=0.75,name=f'{xvalue} in 2000')
    #x2016 = go.Histogram(x=df_[f'{xvalue} in 2016'],opacity=0.75,name=f'{xvalue} in 2016')
    y2000 = go.Histogram(x=df_[f'{yvalue} in 2000'],opacity=0.75,name=f'{yvalue} in 2000')
    y2016 = go.Histogram(x=df_[f'{yvalue} in 2016'],opacity=0.75,name=f'{yvalue} in 2016')
    #traces_x = [x2000,x2016]
    traces_y = [y2000,y2016]
    layout = go.Layout(title='Histograms',barmode='overlay')
    #fig_x = go.Figure(data=traces_x, layout=layout)
    fig_y = go.Figure(data=traces_y, layout=layout).update_yaxes(range=[0, 200])
    #fig_hist = px.histogram(traces, x=yvalue, color='year')
    #m1.plotly_chart(fig_x, use_container_width=True)
    st.plotly_chart(fig_y, use_container_width=True)
    
    # plot box cox histogram version..
    if my_method == 'pearson':
        df_box = df_[(df_[df_.columns] > 0).all(axis=1)]
        df_box[f'{yvalue} in 2000'] = boxcox(df_box[f'{yvalue} in 2000'])[0]
        df_box[f'{yvalue} in 2016'] = boxcox(df_box[f'{yvalue} in 2016'])[0]
        y2000b = go.Histogram(x=df_box[f'{yvalue} in 2000'],opacity=0.75,name=f'{yvalue} in 2000')
        y2016b = go.Histogram(x=df_box[f'{yvalue} in 2016'],opacity=0.75,name=f'{yvalue} in 2016')
        layout_b = go.Layout(title='Box Cox transformed histograms',barmode='overlay')
        traces_y_box = [y2000b,y2016b]
        fig_y_box = go.Figure(data=traces_y_box, layout=layout_b) #.update_yaxes(range=[0, 200])
        st.plotly_chart(fig_y_box, use_container_width=True)
        st.caption('Box Cox power tranformation is applied to data for Pearson correlation approach.')


with st.expander('Classification', expanded=False):       
    class_expl = """
    **Urban amenities** are all company business space locations which belong
    to the following finnish TOL-industry classes:  
    _Wholesale and retail_  
    _Accomondation and food service activites_  
    _Information and communication_  
    _Financial and insurance activities_  
    _Other service activities_  
    <p style="font-family:sans-serif; color:grey; font-size: 12px;">
    Original raw data is from
    <a href="https://research.aalto.fi/fi/projects/l%C3%A4hi%C3%B6iden-kehityssuunnat-ja-uudelleenkonseptointi-2020-luvun-segr " target="_blank">Re:Urbia</a>
    -research project data retrieved from the data products "SeutuCD 2002" and "SeutuCD 2018" by Statistical Finland. 
    Data for company facilities in SeutuCD -products are two years older than the publishing year of the product while data for buildings is roughly one year old.  
    Despite this small timespan inconsistency building data is treated as the data for the companies. 
    The construction which adds a bit of gross floor area in some neighbourhoods within one year of time is analysed to be not relevant in amount for the validity issue in the scope of the study. 
    Based on this alignment the data paper analyze the “eras” 2000 and 2016.
    """
    st.markdown(class_expl, unsafe_allow_html=True)



#footer
st.markdown('---')
footer_title = '''
**Naked Density Project**
[![MIT license](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/teemuja/NDP/blob/main/LICENSE) 
'''
st.markdown(footer_title)
disclamer = 'Data papers are constant work in progress and will be upgraded, changed & fixed for errors while research go on.'
st.caption('Disclaimer: ' + disclamer)