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
        NDP data paper #2 V1.2\
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
st.markdown('This data paper visualise the change in correlation between _**urban density, amenities and daytime population**_. '
            'Research quest here is to see how typical urban design argument of positive density impact on local amenities and livability works in different geographical scales.'
            )
#ingress, unsafe_allow_html=True)
            
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

s1,s2 = st.columns(2)
kuntani = s1.selectbox('Select study area',['Helsinki','Espoo','Vantaa','Helsinki centre','Helsinki suburbs','All suburbs'])
# feature selector place_holder
feature_selector = s2.empty()
feat_selector_warning = st.empty()

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



# ------------------------- AMENITYS STUDY -------------------------------
st.markdown('---')
st.subheader('Urban amenity study')

# main graphs before sample checks
graph_place = st.empty()

# the checks
with st.expander('Sample checks', expanded=False):
    #
    mapplace = st.empty()

    m1,m2,m3 = st.columns(3)
    
    # func to purge col names by characters in them for map viz
    def purge(mylist,purge_list):
        for i in purge_list:
            mylist = [c for c in mylist if i not in c]
        return mylist
    purgelist = ['WO','SA','SU']
    col_list_all = mygdf.drop(columns=['kunta','pno']).columns.to_list()
    feat_list = purge(col_list_all,purge_list=purgelist)
    default_ix = feat_list.index('Residential GFA in 2016')
    color = m1.selectbox('Check features on map', feat_list, index=default_ix)
    level = m2.radio('Change H3-resolution for validation checks',(6,7,8,9),horizontal=True)
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
        fig.update_layout(
                            margin={"r": 0, "t": 30, "l": 0, "b": 0},
                            height=700,
                            coloraxis_showscale=False,
                            title_text='Study area data on map (zero and none value hexas removed)'
                        )
        with mapplace:
            st.plotly_chart(fig, use_container_width=True)

        # cell area info in column 3
        avg_cell_area = round(plot.h3.cell_area()['h3_cell_area'].mean(),3)
        m3.markdown('###')
        m3.markdown(f'Cell area in H{level}: **{avg_cell_area} km²**')
        # Calculate average radius
        import math
        avg_radius = round(math.sqrt((2 * avg_cell_area) / (3 * math.sqrt(3))),2)
        m3.markdown(f'Cell diameter in H{level}: **{2*avg_radius} km**')
    else:
        st.stop()
    
    st.markdown('---')
    st.subheader('Sample checks')
    # stat checks here
    my_method = st.radio('Correlation method',('pearson','spearman'))
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

    st.markdown('---')
    # calssifications        
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




# ----------------------------------- corr calculations for graph_place above --------------------------------

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
                    #df_i = df_i[(df_i[df_i.columns] > 0).all(axis=1)]
                    #df_i[x] = boxcox(df_i[x])[0] #as boxcox return two: transformed data and lambda value
                    #df_i[y] = boxcox(df_i[y])[0]

                    # Ensure values are positive before applying boxcox for x
                    non_positive_values_x = df_i[df_i[x] <= 0]
                    if not non_positive_values_x.empty:
                        df_i[x] = df_i[x].clip(lower=0.01)
                        df_i[x] = boxcox(df_i[x])[0]
                    else:
                        df_i[x] = boxcox(df_i[x])[0]
                    # Ensure values are positive before applying boxcox for y
                    non_positive_values_y = df_i[df_i[y] <= 0]
                    if not non_positive_values_y.empty:
                        df_i[y] = df_i[y].clip(lower=0.01)
                        df_i[y] = boxcox(df_i[y])[0]
                    else:
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
def corrs_combine(mygdf,ref_gdf):
    # all subs for H6 reference values for all 
    allsub = ref_gdf #.loc[~gdf.pno.isin(centre_pnos)]
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
    corrs = corrs_combine(mygdf=df4corr,ref_gdf=gdf) # all subs for H6 reference values for all
except Exception as e:
    st.warning(f"Issue occured with BoxCox transformation: {e}")
    st.stop()

# select feat for corrs ...
plot_list = corrs.columns.to_list()[:-1]
# using only these..
my_plot_list = ['Residential GFA VS Urban amenities (OPC excluded)',
                'Residential GFA VS One person companies (OPC) in urban amenities'
                ]
# ..or select from data
with feature_selector:
    scat_list = st.multiselect('Choose data for the correlation plot', plot_list,default=my_plot_list, max_selections=2)
if len(scat_list) == 2:
    scat_list.extend(['year'])
    corr_plot = corrs[corrs.columns.intersection(scat_list)]
else:
    with feat_selector_warning:
        st.warning('Select two features from the data to plot the correlarion loss.')
    st.stop()

# plotting ..
graph_title = kuntani

fig_corr = px.line(corr_plot,line_dash='variable',line_dash_map={my_plot_list[0]:'solid',my_plot_list[1]:'dash'},
                   labels = {'index':'Spatial resolution','value':'Correlation coefficient','variable':'Correlation pairs'},
                   title=f'Correlation loss in {graph_title} in urban amenities', facet_col='year', facet_col_spacing=0.05)

fig_corr.update_xaxes(autorange="reversed")#, side='top')
fig_corr['layout'].update(shapes=[{'type': 'line','y0':0.5,'y1': 0.5,'x0':str(corr_plot.index[0]), 
                              'x1':str(corr_plot.index[-1]),'xref':'x1','yref':'y1',
                              'line': {'color': 'black','width': 0.5,'dash':'dash'}},
                             {'type': 'line','y0':0.5,'y1': 0.5,'x0':str(corr_plot.index[0]), 
                              'x1':str(corr_plot.index[-1]),'xref':'x2','yref':'y2',
                              'line': {'color': 'black','width': 0.5,'dash':'dash'}}])
#
fig_corr.update_layout(#margin={"r": 10, "t": 50, "l": 10, "b": 50}, height=700,
                legend=dict(
                    yanchor="top",
                    y=-0.15,
                    xanchor="left",
                    x=-0.0
                )
                )
# Extract unique year values directly from the dataframe
year_vals = corr_plot['year'].unique()
new_labels = {f"year={y}": str(y) for y in year_vals}
# Update the annotations with new labels and increase the font size
fig_corr.for_each_annotation(lambda a: a.update(text=new_labels.get(a.text, a.text), font=dict(size=16, family="Arial Bold")))

minimi = -0.25
#minimi = corr_plot.stack().min()
fig_corr.update_layout(yaxis_range=[minimi,1])
fig_corr.update_xaxes(type='category')
#fig_corr.update_layout(legend=dict(orientation="h",yanchor="bottom",y=-0.2,xanchor="left",x=0))

with graph_place:
    st.plotly_chart(fig_corr, use_container_width=True)

st.markdown('###')

st.markdown('---')


# ------------------------------------DAYTIME POP STUDY -------------------------------------------------
st.subheader('Daytime population study')

# selectors
s1,s2,s3 = st.columns(3)
day = s1.radio('Select time category',('Working day','Saturday','Sunday'),horizontal=True)
use_values = s2.radio('Use values',('median','average','max'),horizontal=True)
filter = s3.radio('Filter top decile GFA',('None','Residential GFA','Total GFA'),horizontal=True)

# the corr plot
corr_holder = st.empty()

#the scatter section
with st.expander('Sample checks', expanded=False):
    sc1,sc2 = st.columns(2)
    gfa_set = sc1.radio('Select GFA for plot',('Residential GFA in 2016','Total GFA in 2016'),horizontal=True)
    case_level = sc2.radio('Set H3-resolution for plot',(7,8,9), horizontal=True)

    # use mygdf which has h10 resolution!
    df_h10 = mygdf24.drop(columns=['kunta','pno'])

    # df for scatterplots
    df = df_h10.h3.h3_to_parent_aggregate(case_level).rename(columns={'pub_trans_2016':'Public transit use 2016'})
    
    #map holder and settigs under expander
    scat_holder = st.empty()
    st.markdown('###')

    if filter == 'Total GFA in 2016':
        df = df.loc[df['Total GFA in 2016'] < df['Total GFA in 2016'].quantile(0.9)]
        mytitle = f"{kuntani}: Resolution H{case_level} at '{day}' using '{use_values}' values. (high total GFA locations filtered)"
    elif filter == 'Total GFA in 2016':
        df = df.loc[df['Residential GFA in 2016'] < df['Residential GFA in 2016'].quantile(0.9)]
        mytitle = f"{kuntani}: Resolution H{case_level} at '{day}' using '{use_values}' values. (high res. GFA locations filtered)"
    else:
        mytitle = f"{kuntani}: Resolution H{case_level} at '{day}' using '{use_values}' values. "

    source_24h = """
    Data source: <a href="https://zenodo.org/record/4724389" target="_blank">A 24-hour dynamic population distribution dataset based on mobile phone data from Helsinki Metropolitan Area, Finland</a>
    """
    st.markdown(source_24h, unsafe_allow_html=True)


#----------------functions-------------------

#noon,afternoon... grouping
def grouping(df,reg='WO',values='median'):
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
    if values=='median':
        grouped_data = dataframe.groupby(hour_values // 6, axis=1).median().rename(columns={0:'Night',1:'Noon',2:'Afternoon',3:'Evening'})
    elif values=='average':
        grouped_data = dataframe.groupby(hour_values // 6, axis=1).mean().rename(columns={0:'Night',1:'Noon',2:'Afternoon',3:'Evening'})
    else:
        grouped_data = dataframe.groupby(hour_values // 6, axis=1).max().rename(columns={0:'Night',1:'Noon',2:'Afternoon',3:'Evening'})
    
    df_out = grouped_data.join(df[['Total GFA in 2016','Residential GFA in 2016']])
    return df_out

@st.cache_data(max_entries=1)
def generate_scatter_map(dataframe,title,gfa='Total GFA in 2016',y_max=7000):
    # Extract selected GFA values for plotting
    gfa_values = dataframe[gfa]

    # Extract the grouped medians for each day category and time group
    noon = dataframe['Noon']
    afternoon = dataframe['Afternoon']
    evening = dataframe['Evening']
    night = dataframe['Night']
    dataframe['All'] = dataframe[['Noon','Afternoon','Evening','Night']].mean(axis=1)
    avg = dataframe['All']

    # Create the scatter plot
    fig = px.scatter(dataframe, x=gfa_values, y=avg, trendline="ols", color_discrete_sequence=['lightgrey'], opacity=0.5)
    fig.add_scatter(x=gfa_values, y=noon, mode='markers', name='Noon', marker=dict(color='red'))
    fig.add_scatter(x=gfa_values, y=afternoon, mode='markers', name='Afternoon', marker=dict(color='orange'))
    fig.add_scatter(x=gfa_values, y=evening, mode='markers', name='Evening', marker=dict(color='skyblue'))
    fig.add_scatter(x=gfa_values, y=night, mode='markers', name='Night', marker=dict(color='violet', opacity=0.5))
    # invert plot order
    fig.data = fig.data[::-1]

    #trendline
    trendline_res = px.get_trendline_results(fig)
    summary = trendline_res.iloc[0]["px_fit_results"].summary()
    coef = round(px.get_trendline_results(fig).px_fit_results.iloc[0].rsquared,2)

    # Customize the plot layout
    fig.update_layout(xaxis_title=f'{gfa[:-8]} in location', yaxis_title='Daytime population',yaxis_range=[0,y_max],legend_traceorder="reversed")
    fig.update_layout(title=title)
    fig.update_layout(legend=dict(orientation="h",x=0.05))
    #fig.add_annotation(text=f'OLS-Trendline of average values in grey (R-squared coef: {coef})',
    #            align='left',
    #            showarrow=False,
    #            xref='paper',
    #            yref='paper',
    #            x=0.05,
    #            y=-0.07,
    #            #bordercolor='black', borderwidth=1
    #            )
    return fig,summary

def corr_loss24(df, h=10, method='pearson'):
    # Determine the GFA column by removing the time columns from the DataFrame's columns
    gfa_column = [col for col in df.columns if col not in ['Night', 'Noon', 'Afternoon', 'Evening']][0]
    y_list = ['Night', 'Noon', 'Afternoon', 'Evening']
    frames = []
    for y in y_list:
        corr_list = []
        for i in range(1, 5):
            df_i = df.h3.h3_to_parent_aggregate(h-i, return_geometry=False)
            
            # apply box cox transform to values..
            if method == 'pearson':
                # Ensure values are positive before applying boxcox for GFA column
                non_positive_values_gfa = df_i[df_i[gfa_column] <= 0]
                if not non_positive_values_gfa.empty:
                    df_i[gfa_column] = df_i[gfa_column].clip(lower=0.01)
                    df_i[gfa_column] = boxcox(df_i[gfa_column])[0]
                else:
                    df_i[gfa_column] = boxcox(df_i[gfa_column])[0]

                # Ensure values are positive before applying boxcox for time of day
                non_positive_values_y = df_i[df_i[y] <= 0]
                if not non_positive_values_y.empty:
                    df_i[y] = df_i[y].clip(lower=0.01)
                    df_i[y] = boxcox(df_i[y])[0]
                else:
                    df_i[y] = boxcox(df_i[y])[0]
                    
            # use as transformed
            corr_i = df_i.corr(method=method)[gfa_column][y]
            corr_list.append(corr_i)
        corr_y = pd.DataFrame(corr_list, index=['h9', 'h8', 'h7', 'h6'], columns=['GFA vs ' + y])
        frames.append(corr_y)
        
    corr_df = pd.concat(frames, axis=1, ignore_index=False)
    return corr_df

# corrs combined
def corrs24_combine(my_gdf24,ref_gdf24):
    # calcs .rename(columns={'Total GFA in 2016':'Total GFA'}) .rename(columns={'Residential GFA in 2016':'Residential GFA'})
    facet_cols_total_gfa = ['Total GFA in 2016','Night','Noon','Afternoon','Evening']
    facet_cols_res_gfa = ['Residential GFA in 2016','Night','Noon','Afternoon','Evening']
    corr_tot_GFA = corr_loss24(my_gdf24[facet_cols_total_gfa],method=my_method)
    corr_tot_GFA_ref = corr_loss24(ref_gdf24[facet_cols_total_gfa],method=my_method)
    corr_res_GFA = corr_loss24(my_gdf24[facet_cols_res_gfa],method=my_method)
    corr_res_GFA_ref = corr_loss24(ref_gdf24[facet_cols_res_gfa],method=my_method)
    # replace h6 values with reference values for both
    corr_tot_GFA.loc[corr_tot_GFA.index == 'h6', list(corr_tot_GFA.columns)] = corr_tot_GFA_ref[list(corr_tot_GFA.columns)]
    corr_tot_GFA['GFA_type'] = 'Total GFA'
    corr_res_GFA.loc[corr_res_GFA.index == 'h6', list(corr_res_GFA.columns)] = corr_res_GFA_ref[list(corr_res_GFA.columns)]
    corr_res_GFA['GFA_type'] = 'Residential GFA'
    # combine
    corrs_all = pd.concat([corr_tot_GFA,corr_res_GFA])
    return corrs_all

import h3pandas
import pandas as pd

import h3pandas
import pandas as pd

def filter_hexagons_by_neighbors(df,use_col,q=0.9):
    """
    Filter H3 hexagons by the 0.9 quantile value calculated from the sum of GFA values of each hexagon's neighboring hexagons.
    Parameters:
    - df (DataFrame): Input DataFrame of H3 hexagons of resolution level 10 (h10) indexed by h3_id with a 'GFA' column
    Returns:
    - DataFrame: Filtered DataFrame
    """
    def sum_neighboring_col(h3_index):
        # Get neighboring hexagons
        neighbors = h3pandas.h3.k_ring(h3_index, k=1)
        # Calculate sum of GFA values for neighboring hexagons
        return df.loc[df.index.intersection(neighbors), use_col].sum()
    # Calculate the sum of GFA values for neighboring hexagons for each hexagon
    df['neighbors_sum'] = df.index.to_series().apply(sum_neighboring_col)
    # Calculate the 0.9 quantile value
    quantile = df['neighbors_sum'].quantile(q)
    # Filter hexagons based on the 0.9 quantile
    filtered_df = df[df['neighbors_sum'] > quantile]
    # Drop the temporary columns
    filtered_df = filtered_df.drop(columns=['neighbors_sum'])
    return filtered_df


#coor per daytime
def corrs24_generate(df_h10,df_ref,my_reg,my_values,filter=None):
    #filter high gfa hexas of h10 original dana
    if filter == 'Total GFA in 2016':
        df_h10_use = filter_hexagons_by_neighbors(df_h10,use_col=filter,q=0.9)
        ref_use = filter_hexagons_by_neighbors(df_ref,use_col=filter,q=0.9)
    elif filter == 'Residential GFA in 2016':
        df_h10_use = filter_hexagons_by_neighbors(df_h10,use_col=filter,q=0.9)
        ref_use = filter_hexagons_by_neighbors(df_ref,use_col=filter,q=0.9)
    else: #originals
        df_h10_use = df_h10
        ref_use = df_ref
    # use original h10 data for grouping..
    df_for_corr = grouping(df_h10_use,reg=my_reg,values=my_values)
    ref_df_corr = grouping(ref_use,reg=my_reg,values=my_values) # for reference corrs of H6
    # and then calc loss using that
    corrs24_out = corrs24_combine(my_gdf24=df_for_corr,ref_gdf24=ref_df_corr)
    return corrs24_out, df_for_corr, ref_df_corr #corrs24_out

def corrs24_plotter(corr_plot,fixed=True):
    # line_dash_map = which pairs to plot with line types
    line_map = {'GFA vs Noon':'solid','GFA vs Afternoon':'solid','GFA vs Evening':'dash','GFA vs Night':'dash'}
    fig_corr = px.line(corr_plot,line_dash='variable',line_dash_map=line_map,
                labels = {'index':'Spatial resolution','value':'Correlation coefficient','variable':'Correlation pairs'},
                title=f'Correlation loss in daytime population in {graph_title} at {day}', facet_col='GFA_type', facet_col_spacing=0.05)
    # Define a dictionary to map values to line widths
    line_width_map = {'GFA vs Noon': 0.5,'GFA vs Afternoon': 3,'GFA vs Evening': 3,'GFA vs Night': 0.5}

    # Loop over the traces and set the line width
    for trace in fig_corr.data:
        trace_name = trace.name  # Getting the name of the trace
        if trace_name in line_width_map:
            #print(f"Setting line width for {trace_name} to {line_width_map[trace_name]}")  # Debug print
            trace.line.width = line_width_map[trace_name]
    
    #xaxis reversed
    fig_corr.update_xaxes(autorange="reversed")#, side='top')

    #help lines
    fig_corr['layout'].update(shapes=[{'type': 'line','y0':0.5,'y1': 0.5,'x0':str(corr_plot.index[0]), 
                                'x1':str(corr_plot.index[-1]),'xref':'x1','yref':'y1',
                                'line': {'color': 'black','width': 0.5,'dash':'dash'}},
                                {'type': 'line','y0':0.5,'y1': 0.5,'x0':str(corr_plot.index[0]), 
                                'x1':str(corr_plot.index[-1]),'xref':'x2','yref':'y2',
                                'line': {'color': 'black','width': 0.5,'dash':'dash'}}])
    
    #legend
    fig_corr.update_layout(#margin={"r": 10, "t": 50, "l": 10, "b": 50}, height=700,
                    legend=dict(
                        yanchor="top",
                        y=-0.15,
                        xanchor="left",
                        x=-0.0
                    )
                    )
    minimi = -0.25
    #minimi = corr_plot.stack().min()
    fig_corr.update_layout(yaxis_range=[minimi,1])
    fig_corr.update_xaxes(type='category')

    # Extract unique year values directly from the dataframe
    year_vals = corr_plot['GFA_type'].unique()
    new_labels = {f"GFA_type={y}": str(y) for y in year_vals}
    # Update the annotations with new labels and increase the font size
    fig_corr.for_each_annotation(lambda a: a.update(text=new_labels.get(a.text, a.text), font=dict(size=16, family="Arial Bold")))

    return fig_corr


# --------------- DATA to VIZs --------------------------
if day == 'Working day':
    df_for_scat = grouping(df,reg='WO',values=use_values)
    df_for_corrs = corrs24_generate(df_h10=df_h10,df_ref=gdf24,my_reg='WO',my_values=use_values,filter=filter)
elif day == 'Saturday':
    df_for_scat = grouping(df,reg='SA',values=use_values)
    df_for_corrs = corrs24_generate(df_h10=df_h10,df_ref=gdf24,my_reg='SA',my_values=use_values,filter=filter)
else:
    df_for_scat = grouping(df,reg='SU',values=use_values)
    df_for_corrs = corrs24_generate(df_h10=df_h10,df_ref=gdf24,my_reg='SU',my_values=use_values,filter=filter)

# figures..
ymax = df_for_scat.drop(columns=['Residential GFA in 2016','Total GFA in 2016']).to_numpy().max()
scat24,summary = generate_scatter_map(df_for_scat,title=mytitle,gfa=gfa_set,y_max=ymax)
corrs24_fig = corrs24_plotter(df_for_corrs[0])

# ----------- viz the data in place holders ---------
with scat_holder:
    scat1,scat2 = st.columns(2)
    scat1.markdown('---')
    scat1.plotly_chart(scat24, use_container_width=True)
    scat2.markdown('---')
    scat2.write(summary)

with corr_holder:
    st.plotly_chart(corrs24_fig, use_container_width=True)


# for corr loss plot..
#with st.expander('data',expanded=False):
#    c1,c2,c3 = st.columns(3)
#    c1.dataframe(df_for_corrs[0])
#    c2.dataframe(df_for_corrs[1])
#    c3.dataframe(df_for_corrs[2])


st.markdown('---')
import io
@st.cache_data()
def gen_pdf(fig):
    buffer_fig = io.BytesIO()
    # https://github.com/plotly/plotly.py/issues/3469
    temp_fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])
    temp_fig.write_image(file=buffer_fig, format="pdf")
    import time
    time.sleep(1)
    # adjust layout
    fig.update_layout(
                margin={"r": 100, "t": 100, "l": 100, "b": 100}, height=700,
                legend=dict(
                    yanchor="top",
                    y=-0.15,
                    xanchor="left",
                    x=-0.0
                    )
                )
    # replace temp_fig in buffer
    fig.write_image(file=buffer_fig, format="pdf")
    return buffer_fig
        
with st.spinner():
    st.subheader('Downloads')
    #
    fig_corr_pdf = gen_pdf(fig_corr)
    fig_corr_pdf_name = f"Correlation loss in urban amenities in {graph_title}.pdf"
    corrs24_fig_pdf = gen_pdf(corrs24_fig)
    corrs24_fig_pdf_name = f"Correlation loss in daytime population in {graph_title}.pdf"
    
    st.download_button(
        label=fig_corr_pdf_name,
        data=fig_corr_pdf,
        file_name=fig_corr_pdf_name,
        mime="application/pdf",
        )

    st.download_button(
        label=corrs24_fig_pdf_name,
        data=corrs24_fig_pdf,
        file_name=corrs24_fig_pdf_name,
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