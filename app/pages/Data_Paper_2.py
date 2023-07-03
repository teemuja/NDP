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
        NDP data paper #2 V0.98\
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
    'GFAtot_2000':'Total GFA in 2000',
    'GFAres_2000':'Residential GFA in 2000',
    'GFAtot_2016':'Total GFA in 2016',
    'GFAres_2016':'Residential GFA in 2016',
    'opc_2000':'One person companies (OPC) in urban amenities in 2000',
    'urban_2000':'Urban amenities (OPC excluded) in 2000',
    'opc_2016':'One person companies (OPC) in urban amenities in 2016',
    'urban_2016':'Urban amenities (OPC excluded) in 2016'
}

@st.cache_data()
def load_data():
    path = Path(__file__).parent / 'data/h3_10_pks_corr.csv'
    with path.open() as f:
        data = pd.read_csv(f, index_col='h3_10', header=0) #.astype(str)
    # translate columns
    eng_data = data.rename(columns=eng_feat)
    return eng_data

@st.cache_data()
def load_data24():
    path = Path(__file__).parent / 'data/h3_10_pks_corr24.csv'
    with path.open() as f:
        data = pd.read_csv(f, index_col='h3_10', header=0) #.astype(str)
    # translate columns
    eng_data = data.rename(columns=eng_feat)
    return eng_data

gdf = load_data()
gdf24 = load_data24()

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

s1,s2,s3 = st.columns(3)
kuntani = s1.selectbox('Select study area',['Helsinki','Espoo','Vantaa','Helsinki centre','Helsinki suburbs','All suburbs'])
if kuntani == 'Helsinki centre':
    mygdf = gdf.loc[gdf.pno.isin(centre_pnos)]
    #for day pop study..
    mygdf24 = gdf24.loc[gdf24.pno.isin(centre_pnos)]
elif kuntani == 'Helsinki suburbs':
    mygdf = gdf.loc[gdf.kunta == 'Helsinki']
    mygdf = mygdf.loc[~mygdf.pno.isin(centre_pnos)]
    #for day pop study..
    mygdf24 = gdf24.loc[gdf24.kunta == 'Helsinki']
    mygdf24 = mygdf24.loc[~mygdf24.pno.isin(centre_pnos)]
elif kuntani == 'All suburbs':
    mygdf = gdf.loc[~gdf.pno.isin(centre_pnos)]
    #for day pop study use whole material
    mygdf24 = gdf24.loc[~gdf24.pno.isin(centre_pnos)]
else:
    mygdf = gdf.loc[gdf.kunta == kuntani]
    mygdf24 = gdf24.loc[gdf24.kunta == kuntani]

#q_range = s3.slider(' ',0,100,(0,100),10) 
# filter accordingly..
#mygdf = mygdf.loc[mygdf[f'{color}'].astype(int) > mygdf[f'{color}'].astype(int).quantile(q_range[0]/100)] 
#mygdf = mygdf.loc[mygdf[f'{color}'].astype(int) < mygdf[f'{color}'].astype(int).quantile(q_range[1]/100)]

# the checks
with st.expander('Data validation', expanded=False):
    #
    mapplace = st.empty()
    m1,m2 = st.columns(2)
    
    # func to purge col names by characters in them
    col_list_all = mygdf.drop(columns=['kunta','pno']).columns.to_list()
    def purge(mylist,purge_list):
        for i in purge_list:
            mylist = [c for c in mylist if i not in c]
        return mylist
    purgelist = ['WO','SA','SU']
    feat_list = purge(col_list_all,purge_list=purgelist)
    default_ix = feat_list.index('Residential GFA in 2016')
    color = m1.selectbox('Check features on map', feat_list, index=default_ix)
    level = m2.radio('Change H3-resolution for validation checks',(7,8,9),horizontal=True)
    m2.caption('https://h3geo.org/docs/core-library/restable/')

    # map plot
    if len(mygdf) > 1:
        plot = mygdf.h3.h3_to_parent_aggregate(level)
        # exclude zero hexas
        plot = plot[plot[color] > 0]
        # map plot
        lat = plot.unary_union.centroid.y
        lon = plot.unary_union.centroid.x
        range_min = plot[color].min() #.quantile(0.001)
        range_max = plot[color].max()  #.quantile(0.999)
        fig = px.choropleth_mapbox(plot,
                                geojson=plot.geometry,
                                locations=plot.index,
                                #title='Filtered data on map',
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
    
    st.markdown('---')
    st.subheader('Sample checks')
    # stat checks here
    my_method = 'pearson' #st.radio('Correlation method',('pearson','spearman'))
#with st.expander('Statistical checks', expanded=False):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    df = plot.copy()
    # select feat
    ycols = ['Urban amenities (OPC excluded)',
             'One person companies (OPC) in urban amenities']
    x1,y1 = st.columns(2)
    xvalue = x1.selectbox('Choose X axis data',['Residential GFA','Total GFA'])
    yvalue = y1.selectbox('Choose y axis data',ycols)
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
    scat.update_layout(title=f"Scatter plot on resolution H{level} in {kuntani} for '{yvalue}' ")

    st.plotly_chart(scat, use_container_width=True)

    #x
    xvalue_2000sum = df[f'{xvalue} in 2000'].sum()
    xvalue_2016sum = df[f'{xvalue} in 2016'].sum()
    xvalue_2000max = df[f'{xvalue} in 2000'].max()
    xvalue_2016max = df[f'{xvalue} in 2016'].max()
    #y
    yvalue_2000sum = df[f'{yvalue} in 2000'].sum()
    yvalue_2016sum = df[f'{yvalue} in 2016'].sum()
    yvalue_2000max = df[f'{yvalue} in 2000'].max()
    yvalue_2016max = df[f'{yvalue} in 2016'].max()

    m1,m2,m3,m4 = st.columns(4)
    m1.metric(label=f"Sum of {xvalue} in 2000", value=f"{xvalue_2000sum}", delta=f"max_H{level}: {xvalue_2000max}")
    m2.metric(label=f"Sum of {xvalue} in 2016", value=f"{xvalue_2016sum}", delta=f"max_H{level}: {xvalue_2016max}")
    m3.metric(label=f"Sum of {yvalue} in 2000", value=f"{yvalue_2000sum}", delta=f"max_H{level}: {yvalue_2000max}")
    m4.metric(label=f"Sum of {yvalue} in 2016", value=f"{yvalue_2016sum}", delta=f"max_H{level}: {yvalue_2016max}")
    #
    count_2000 = df[f'{xvalue} in 2000'].count()
    count_2016 = df[f'{xvalue} in 2016'].count()
    
    #st.markdown('---')

    # histogram for data in current resollution level --use only positive values
    df_ = df[(df.T != 0).any()][feat_list] #.drop(columns='geometry')
    df_ = df_.apply(pd.to_numeric, errors='coerce')
    #st.table(df_.describe())
    #st.stop()
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

    # FIX boxplot check below!!
    st.plotly_chart(fig_y, use_container_width=True)
    st.metric(label=f"Sample size 2000/2016", value=f"{count_2000}/{count_2016}")
    
    # plot box cox histogram version..
    if my_method == 'pearson':
        try:
            df_box2000 = df_[df_[f'{yvalue} in 2000'] > 0]
            df_box2016 = df_[df_[f'{yvalue} in 2016'] > 0]
            #st.table(df_box.describe())
            #st.stop()
            df_box2000[f'{yvalue} in 2000'],lam2000 = boxcox(df_box2000[f'{yvalue} in 2000'])
            df_box2016[f'{yvalue} in 2016'],lam2016 = boxcox(df_box2016[f'{yvalue} in 2016'])
            y2000b = go.Histogram(x=df_box2000[f'{yvalue} in 2000'],opacity=0.75,name=f'{yvalue} in 2000',nbinsx=20)
            y2016b = go.Histogram(x=df_box2016[f'{yvalue} in 2016'],opacity=0.75,name=f'{yvalue} in 2016',nbinsx=20)
            layout_b = go.Layout(title='Box Cox transformed histograms',barmode='overlay')
            traces_y_box = [y2000b,y2016b]
            fig_y_box = go.Figure(data=traces_y_box, layout=layout_b) #.update_yaxes(range=[0, 200])
            st.plotly_chart(fig_y_box, use_container_width=True)
            lam1 = round(lam2000,2)
            lam2 = round(lam2016,2)
            st.write(f'Box Cox lambda: 2000= {lam1}, 2016= {lam2} in resolution H{level}.')
            st.write('**NOTE** Sample of the whole region (All suburbs) was used for resolution H6 in each study to avoid  '
                        'biases of too small samples in large scale resolution.  '
                        'This way H6 resolution acts as a reference correlation in this data paper.  '
                        'Correlation loss -graph shows how correlation falls in particular part of the city  '
                        'when zooming in to the more local level scales.')
        except Exception as e:
            st.warning(e)
            pass

# corr graphs
st.subheader('Correlation loss')

def corr_loss(df,h=10,corr_type='year',method='pearson'): # h-value is one more than generated corr-levels
    if corr_type == '2000':
        x_list=['Total GFA in 2000',
                'Residential GFA in 2000']
        y_list=['One person companies (OPC) in urban amenities in 2000',
                'Urban amenities (OPC excluded) in 2000']
    elif corr_type == '2016':
        x_list=['Total GFA in 2016',
                'Residential GFA in 2016']
        y_list=['One person companies (OPC) in urban amenities in 2016',
                'Urban amenities (OPC excluded) in 2016']
    elif corr_type == 'year':
        x_list=['Total GFA',
                'Residential GFA']
        y_list=['One person companies (OPC) in urban amenities',
                'Urban amenities (OPC excluded)']

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
    'Urban amenities (OPC excluded) in 2016':'Urban amenities (OPC excluded)'
}
facet_col_list_2000 = [
    'Total GFA in 2000',
    'Residential GFA in 2000',
    'One person companies (OPC) in urban amenities in 2000',
    'Urban amenities (OPC excluded) in 2000'
]
facet_col_list_2016 = [
    'Total GFA in 2016',
    'Residential GFA in 2016',
    'One person companies (OPC) in urban amenities in 2016',
    'Urban amenities (OPC excluded) in 2016'
]

# corrs combined
def corrs_combine(mygdf):
    # all subs for H6 reference values for all 
    allsub = gdf #.loc[~gdf.pno.isin(centre_pnos)]
    # 2000
    corr_2000 = corr_loss(mygdf[facet_col_list_2000].rename(columns=facet_feat),corr_type='year',method=my_method)
    corr_2000_all = corr_loss(allsub[facet_col_list_2000].rename(columns=facet_feat),corr_type='year',method=my_method)
    corr_2000.loc[corr_2000.index == 'h6', list(corr_2000.columns)] = corr_2000_all[list(corr_2000.columns)]
    corr_2000['year'] = 2000
    # 2016
    corr_2016 = corr_loss(mygdf[facet_col_list_2016].rename(columns=facet_feat),corr_type='year',method=my_method)
    corr_2016_all = corr_loss(allsub[facet_col_list_2016].rename(columns=facet_feat),corr_type='year',method=my_method)
    corr_2016.loc[corr_2016.index == 'h6', list(corr_2016.columns)] = corr_2016_all[list(corr_2016.columns)]
    corr_2016['year'] = 2016
    # combine
    corrs_all = pd.concat([corr_2000,corr_2016]) #corr_2000.append(corr_2016)
    return corrs_all

# do the corrs
df4corr = mygdf[feat_list] # use only feat_cols
try:
    corrs = corrs_combine(mygdf=df4corr)
except Exception as e:
    st.warning(f"Issue occured with BoxCox transformation: {e}")
    st.stop()

# select feat for corrs
plot_list = corrs.columns.to_list()[:-1]
my_plot_list = ['Residential GFA VS Urban amenities (OPC excluded)',
                'Residential GFA VS One person companies (OPC) in urban amenities'
                ]
scat_list = my_plot_list #st.multiselect('Choose data for the correlation plot', plot_list,default=my_plot_list, max_selections=2)
if len(scat_list) == 2:
    scat_list.extend(['year'])
    corr_plot = corrs[corrs.columns.intersection(scat_list)]
else:
    st.stop()

# plot
graph_title = kuntani
fig_corr = px.line(corr_plot,line_dash='variable',line_dash_map={my_plot_list[0]:'solid',my_plot_list[1]:'dash'},
                   labels = {'index':'H3-resolution','value':'Correlation coefficient','variable':'Correlation pairs'},
                   title=f'Correlation loss in {graph_title}', facet_col='year', facet_col_spacing=0.05)
fig_corr.update_xaxes(autorange="reversed")#, side='top')
fig_corr['layout'].update(shapes=[{'type': 'line','y0':0.5,'y1': 0.5,'x0':str(corr_plot.index[0]), 
                              'x1':str(corr_plot.index[-1]),'xref':'x1','yref':'y1',
                              'line': {'color': 'black','width': 0.5,'dash':'dash'}},
                             {'type': 'line','y0':0.5,'y1': 0.5,'x0':str(corr_plot.index[0]), 
                              'x1':str(corr_plot.index[-1]),'xref':'x2','yref':'y2',
                              'line': {'color': 'black','width': 0.5,'dash':'dash'}}])
#fig_corr.add_vrect(
#    x0=corr_plot.index[-2], x1=corr_plot.index[-1],
#    fillcolor="white", opacity=0.8,
#    layer="above", line_width=0,
#)
fig_corr.update_layout(#margin={"r": 10, "t": 50, "l": 10, "b": 50}, height=700,
                legend=dict(
                    yanchor="top",
                    y=-0.15,
                    xanchor="left",
                    x=-0.0
                )
                )
graph_place = st.empty()
fixed = st.checkbox('Use fixed scale')
if fixed:
    minimi = -0.5
else:
    minimi = corr_plot.stack().min()
fig_corr.update_layout(yaxis_range=[minimi,1])
fig_corr.update_xaxes(type='category')
#fig_corr.update_layout(legend=dict(orientation="h",yanchor="bottom",y=-0.2,xanchor="left",x=0))
with graph_place:
    st.plotly_chart(fig_corr, use_container_width=True)


with st.expander('Classification', expanded=False):       
    class_expl = """
    For used resolutions, see: <a href="https://h3geo.org/docs/core-library/restable/" target="_blank">h3geo.org</a>

    **Urban amenities** are all company business space locations which belong
    to the following finnish TOL-industry classes (tol95 and tol2008):  
    _Wholesale and retail_  
    _Accomondation and food service activites_  
    _Information and communication_  
    _Financial and insurance activities_  
    _Other service activities_  
    More info: <a href="https://www.stat.fi/en/luokitukset/toimiala/" target="_blank">Stat.fi</a>  
      
    OPC = One Person Companies according to the information in national business space location registry (YrTp)  
    Correlation values are <a href="https://en.wikipedia.org/wiki/Pearson_correlation_coefficient" target="_blank">Pearson</a> 
    correlation coefficient (r) -values computed using <a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html" target="_blank">Pandas</a> library.  

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

# **Consumer daily goods and kiosks** refers to the TOL-classes 5211+5212(TOL1995) for 2000 data and 4711+4719 (TOL2008) for 2016 data.  
    

with st.expander('Case studies', expanded=False):
    # study level
    case_level = st.radio('Set H3-resolution for case studies',(7,8,9), horizontal=True)
    # use mygdf which has h10 resolution!
    df = mygdf24.drop(columns=['kunta','pno'])
    df = df.h3.h3_to_parent_aggregate(case_level).rename(columns={'pub_trans_2016':'Public transit use 2016'})
    
    st.markdown('---')
    st.subheader('Daytime population')

    def grouping(df,reg='WO'):
        dataframe = df.filter(regex=reg)
        # Rename columns with underscore
        renamed_columns = []
        for column in dataframe.columns:
            if "_" in column:
                renamed_columns.append(column.split("_")[1])
            else:
                renamed_columns.append(column)
        dataframe.columns = renamed_columns

        # Extract the hour values from the column names
        hour_values = dataframe.columns.str[1:].astype(int)

        # Group the hour columns into four parts: noon, afternoon, evening, night
        grouped_data = dataframe.groupby(hour_values // 6, axis=1).median().rename(columns={0:'Night',1:'Noon',2:'Afternoon',3:'Evening'})

        df_out = grouped_data.join(df[['Total GFA in 2016','Residential GFA in 2016']])
        return df_out

    def generate_scatter_map(dataframe,gfa='Total GFA in 2016',y_max=7000):
        # Extract selected GFA values for plotting
        gfa_values = dataframe[gfa]

        # Extract the grouped medians for each day category and time group
        medians_noon = dataframe['Noon']
        medians_afternoon = dataframe['Afternoon']
        medians_evening = dataframe['Evening']
        medians_night = dataframe['Night']
        dataframe['All'] = dataframe[['Noon','Afternoon','Evening','Night']].median(axis=1)

        # Create the scatter plot
        fig = px.scatter(dataframe, x=gfa_values, y=dataframe['All'], trendline="ols", color_discrete_sequence=['grey'])
        fig.add_scatter(x=gfa_values, y=medians_noon, mode='markers', name='Noon', marker=dict(color='red'))
        fig.add_scatter(x=gfa_values, y=medians_afternoon, mode='markers', name='Afternoon', marker=dict(color='orange'))
        fig.add_scatter(x=gfa_values, y=medians_evening, mode='markers', name='Evening', marker=dict(color='skyblue'))
        fig.add_scatter(x=gfa_values, y=medians_night, mode='markers', name='Night', marker=dict(color='violet', opacity=0.3))

        # Customize the plot layout
        fig.update_layout(xaxis_title=gfa, yaxis_title='Daytime population medians',yaxis_range=[0,y_max])
        fig.update_layout(title=f"Correlation at resolution H{case_level} in {kuntani} at '{day}' ")
        return fig

    # day selector
    s1,s2,s3 = st.columns(3)
    day = s1.radio('Select time category',('Working day','Saturday','Sunday'),horizontal=True)
    gfa_set = s2.radio ('Select GFA',('Residential GFA in 2016','Total GFA in 2016'),horizontal=True)
    quant = s3.checkbox('Remove high deciles of GFA')
    if quant:
        df = df.loc[df['Total GFA in 2016'] < df['Total GFA in 2016'].quantile(0.9)]

    if day == 'Working day':
        df_for_plot = grouping(df,reg='WO')            
    elif day == 'Saturday':
        df_for_plot = grouping(df,reg='SA')
    else:
        df_for_plot = grouping(df,reg='SU')

    ymax = df_for_plot.drop(columns=['Residential GFA in 2016','Total GFA in 2016']).to_numpy().max()
    
    st.plotly_chart(generate_scatter_map(df_for_plot,gfa=gfa_set,y_max=ymax), use_container_width=True)
    
    source_24h = """
    Data source: <a href="https://zenodo.org/record/3247564#.ZGxysC9Bzyw" target="_blank">Helsinki Region Travel Time Matrix</a>
    """
    st.markdown(source_24h, unsafe_allow_html=True)
    
    # PUBLIC TRANS

    st.markdown('---')
    st.subheader('Public transit use 2016')
    # plots
    traceRES = go.Scatter(
        x=df['Residential GFA in 2016'],
        y=df['Public transit use 2016'],
        name='Residential GFA',
        text=df.index,
        hovertemplate=
        "<b>Haxagon %{text}</b><br><br>" +
        "GFA: %{x:,.0f} sqr-m<br>" +
        "Takeoffs: %{y}<br>" +
        #"Population: %{marker.size:,}" +
        "<extra></extra>",
        mode='markers',
        marker=dict(
                color='Brown',
                size=7)
    )
    traceTOT = go.Scatter(
        x=df['Total GFA in 2016'],
        y=df['Public transit use 2016'],
        name='Total GFA',
        text=df.index,
        hovertemplate=
        "<b>Haxagon %{text}</b><br><br>" +
        "GFA: %{x:,.0f} sqr-m<br>" +
        "Takeoffs: %{y}<br>" +
        #"Population: %{marker.size:,}" +
        "<extra></extra>",
        yaxis='y2',
        mode='markers',
        marker=dict(
                color='Orange',
                size=7)
    )
    scat_pub = make_subplots(specs=[[{"secondary_y": True}]],
                            x_title='GFA in location',y_title='Takeoffs')
    scat_pub.add_trace(traceRES)
    scat_pub.add_trace(traceTOT,secondary_y=False)
    if 'kuntani' not in globals():
        kuntani = 'selected neighborhoods'
    scat_pub.update_layout(title=f"Scatter plot on resolution H{case_level} in {kuntani} for 'Public transit use 2016' ")
    st.plotly_chart(scat_pub, use_container_width=True)
    pub_expl = """
    Data source: <a href="https://www.avoindata.fi/data/en_GB/dataset/hsl-n-nousijamaarat-pysakeittain" target="_blank">Avoindata.fi</a>
    """
    st.markdown(pub_expl, unsafe_allow_html=True)

    
with st.expander('PDF downloads', expanded=False):
    import io
    @st.cache_data()
    def gen_pdf(fig):
        buffer_fig = io.BytesIO()
        # https://github.com/plotly/plotly.py/issues/3469
        temp_fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])
        temp_fig.write_image(file=buffer_fig, format="pdf")
        import time
        time.sleep(1)
        # replace temp_fig in buffer
        fig.write_image(file=buffer_fig, format="pdf")
        return buffer_fig
    
    pdf_out = None
    d1,d2 = st.columns([1,2])
    with d1.form("my_form",clear_on_submit=True):
        my_sel = st.selectbox('',['Select the graph..','Correlation loss','Public transit case','Daytime pop case'])
        if my_sel == 'Correlation loss':
            my_fig = fig_corr
        elif my_sel == 'Public transit case':
            my_fig = scat_pub
        elif my_sel == 'Daytime pop case':
            my_fig = scat24
        else:
            my_fig = None

        submitted = st.form_submit_button("Generate PDF")

        if submitted:
            if my_fig is not None:
                #update layout for pdf plot
                my_fig.update_layout(
                    margin={"r": 100, "t": 100, "l": 100, "b": 100}, height=700,
                    legend=dict(
                        yanchor="top",
                        y=-0.15,
                        xanchor="left",
                        x=-0.0
                    )
                    )
                pdf_out = gen_pdf(my_fig) #pdf.generate_pdf_report(my_fig)
            else:
                st.warning('Select figure')
                st.stop()

    # download button must be outside the form
    if pdf_out is not None:
        d2.markdown('###')
        d2.markdown('###')
        d2.download_button(
            label="Download pdf",
            data=pdf_out,
            file_name=f"{my_sel} {graph_title}.pdf",
            mime="application/pdf",
            )


#footer
st.markdown('---')
footer_title = '''
**Naked Density Project**
[![MIT license](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/teemuja/NDP/blob/main/LICENSE) 
'''
st.markdown(footer_title)
disclamer = 'Data papers are constant work in progress and will be upgraded, changed & fixed for errors while research go on.'
st.caption('Disclaimer: ' + disclamer)