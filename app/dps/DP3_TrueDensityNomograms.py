# NDP app always beta a lot
from ast import Pass
from unittest import case
import pandas as pd
import geopandas as gpd
import numpy as np
import streamlit as st
import osmnx as ox
import momepy
import plotly.express as px
px.set_mapbox_access_token(st.secrets['MAPBOX_TOKEN'])
my_style = st.secrets['MAPBOX_STYLE']
import math
import statistics


st.title("Data Paper #3")
st.markdown("Density measurements using Open Street Map data")
st.markdown("###")
st.title(':point_down:')

@st.cache_data(ttl=900, max_entries=5)
def get_data(add, tags, radius=500):
    gdf = ox.features_from_address(add, tags, radius)
    fp_proj = ox.project_gdf(gdf).reset_index()
    fp_proj = fp_proj[fp_proj["element_type"] == "way"]
    use_cols = ["osmid", "geometry", "building", 'building:levels','addr:street']
    if 'building:levels' in fp_proj:
        fp_poly = fp_proj[use_cols]
    else:
        fp_proj['building:levels'] = None #[ random.randint(1,2) for k in fp_proj.index]
        fp_poly = fp_proj[use_cols]
    fp_poly["area"] = fp_poly.area
    # exclude all small footprints !!!
    fp_poly = fp_poly[fp_poly["area"] > 50]
    # levels to numeric
    if 'building:levels' not in gdf.columns:
        gdf['building:levels'] = None
    fp_poly["building:levels"] = pd.to_numeric(fp_poly["building:levels"], errors='coerce', downcast='float')
    return fp_poly # projected

# USER INPUT -------------------------------------------------------------
user_input = st.text_input('Type address or place on earth')
import re
add = re.sub(' +', ' ', f'{user_input}')
tags = {'building': True}
radius = 500
if add:
    try:
        buildings = get_data(add, tags, radius)
    except:
        st.write('Check the address!')
        st.stop()
else:
    st.stop()

# cut out edge footprints (incomplete sum values) and viz circle..
with st.spinner(text='preparing map...'):
    union = buildings.unary_union
    env = union.envelope
    focus = gpd.GeoSeries(env)
    focus_area = gpd.GeoSeries(focus)
    focus_circle = focus_area.centroid.buffer(radius)
    focus_gdf = gpd.GeoDataFrame(focus_circle, geometry=0)
    fp_cut = gpd.overlay(buildings, focus_gdf, how='intersection')  # CRS projected for both


plot = fp_cut.to_crs(4326)
# tag reorder
tag_order = plot.building.value_counts().index.tolist()
lat = plot.unary_union.centroid.y
lon = plot.unary_union.centroid.x
mymap = px.choropleth_mapbox(plot,
                                geojson=plot.geometry,
                                locations=plot.index,
                                title=f'{add}',
                                color="building",
                                hover_name='building',
                                hover_data=['building:levels','addr:street'],
                                labels={"building": 'Building tags in use sorted by count'},
                                category_orders={"building":tag_order},
                                mapbox_style=my_style,
                                color_discrete_sequence=px.colors.qualitative.D3,
                                center={"lat": lat, "lon": lon},
                                zoom=14,
                                opacity=0.8,
                                width=1200,
                                height=700
                                )
with st.expander("Map", expanded=True):
    st.plotly_chart(mymap, use_container_width=True)
#st.dataframe(plot.drop(columns='geometry'))
flr_rate = 100 - round(buildings['building:levels'].isna().sum() / len(buildings.index) * 100,0)
floor_med = buildings['building:levels'].median()
st.caption(f'Floor number information in {flr_rate}% of buildings with median value of {floor_med}. The rest will be estimated using nearby medians.')

# -------------------------------------------------------------------

@st.cache_data(ttl=120)
def osm_densities(_buildings):
    # projected crs for momepy calculations
    gdf = buildings.to_crs(3067)
    # check tessellation input & filter bad ones
    check = momepy.CheckTessellationInput(gdf)
    l1 = check.collapse['osmid'].tolist()
    l2 = check.split['osmid'].tolist()
    l3 = check.overlap['osmid'].tolist()
    filterlist = l1 + l2 + l3
    filtered_series = ~gdf.osmid.isin(filterlist)
    gdf = gdf[filtered_series]
    # prepare uIDs
    gdf['uID'] = momepy.unique_id(gdf)
    limit = momepy.buffered_limit(gdf)
    tessellation = momepy.Tessellation(gdf, unique_id='uID', limit=limit).tessellation
    sw = momepy.sw_high(k=2, gdf=tessellation, ids='uID')
    
    # calculate GSI = ground space index = coverage = CAR = coverage area ratio
    tess_GSI = momepy.AreaRatio(tessellation, gdf,
                                momepy.Area(tessellation).series,
                                momepy.Area(gdf).series, 'uID')
    gdf['GSI'] = round(tess_GSI.series,3)

    # floor num info:
    tessellation = tessellation.merge(gdf[['uID', 'building:levels']])
    # get mean floor num of the neighborhood for NaN values
    gdf['ND_med_floors'] = round(momepy.AverageCharacter(tessellation, values='building:levels', spatial_weights=sw,unique_id='uID').median,0)
    gdf['ND_med_floors'].fillna(1, inplace=True)
    # prepare GFAs
    if gdf["building:levels"] is not None:
        gdf["GFA"] = gdf["area"] * gdf['ND_med_floors']
    else:
        gdf["GFA"] = gdf["area"] * gdf["building:levels"]
    gdf['GFA'] = round(gdf['GFA'],0)
    # calculate FSI = floor space index = FAR = floor area ratio
    gdf['FSI'] = round(gdf['GFA'] / momepy.Area(tessellation).series,3)
    # calculate OSR = open space ratio = spaciousness
    gdf['OSR'] = round((1 - gdf['GSI']) / gdf['FSI'],3)

    # ND calculations
    # queen contiguity for 2 degree neighbours = "perceived neighborhood"
    tessellation = tessellation.merge(
        gdf[['uID', 'area', 'GFA', 'OSR']])  # add selected values from buildings to tess-areas
    sw = momepy.sw_high(k=2, gdf=tessellation, ids='uID')  # degree of nd
    gdf['GSI_ND'] = round(
        momepy.Density(tessellation, values='area', spatial_weights=sw, unique_id='uID').series, 2)
    gdf['FSI_ND'] = round(momepy.Density(tessellation, values='GFA', spatial_weights=sw, unique_id='uID').series,
                          2)
    gdf['OSR_ND'] = round((1 - gdf['GSI_ND']) / gdf['FSI_ND'], 2)
    gdf['OSR_ND_mean'] = round(
        momepy.AverageCharacter(tessellation, values='OSR', spatial_weights=sw, unique_id='uID').mean, 2)
    # remove infinite values of osr if needed..
    gdf['OSR_ND'].clip(upper=gdf['OSR'].quantile(0.99), inplace=True)
    gdf['OSR_ND_mean'].clip(upper=gdf['OSR'].quantile(0.99), inplace=True)
    # remove infinite values of osr
    gdf['OSR'].clip(upper=gdf['OSR'].quantile(0.99), inplace=True)
    gdf.rename(columns={'building:levels': 'floors', 'area': 'footprint'},inplace=True)
    gdf_out = gdf.to_crs(4326)
    return gdf_out

def classify_density(density_data):
    density_data['OSR_class'] = 'nan'
    density_data.loc[density_data['OSR'] > 0, 'OSR_class'] = 'close'
    density_data.loc[density_data['OSR'] > 1, 'OSR_class'] = 'dense'
    density_data.loc[density_data['OSR'] > 2, 'OSR_class'] = 'compact'
    density_data.loc[density_data['OSR'] > 4, 'OSR_class'] = 'spacious'
    density_data.loc[density_data['OSR'] > 8, 'OSR_class'] = 'airy'
    density_data.loc[density_data['OSR'] > 16, 'OSR_class'] = 'spread'
    density_data['OSR_ND_class'] = 'nan'
    density_data.loc[density_data['OSR_ND'] > 0, 'OSR_ND_class'] = 'close'
    density_data.loc[density_data['OSR_ND'] > 1, 'OSR_ND_class'] = 'dense'
    density_data.loc[density_data['OSR_ND'] > 2, 'OSR_ND_class'] = 'compact'
    density_data.loc[density_data['OSR_ND'] > 4, 'OSR_ND_class'] = 'spacious'
    density_data.loc[density_data['OSR_ND'] > 8, 'OSR_ND_class'] = 'airy'
    density_data.loc[density_data['OSR_ND'] > 16, 'OSR_ND_class'] = 'spread'
    return density_data


# CALCULATE DENSITIES ----------------------------------

st.markdown('---')
tags = buildings['building'].unique().tolist()
# get most common list of tags in use in area
top_tags = tag_order[:9] # top 9 
mytags = st.multiselect('Select tags(building types) to include for density analysis',tags,default=top_tags)
st.caption('Top-9 tags mostly used as a default selection set. See sorter legend of map')
my_buildings = buildings.loc[buildings['building'].isin(mytags)]

run = st.checkbox('Autocalculate densities', value=False)
if run:
    density_data = osm_densities(my_buildings)
    case_data = classify_density(density_data)
else:
    st.stop()

# for density plot
colormap_osr = {
    "close": "red",
    "dense": "darkgoldenrod",
    "compact": "darkolivegreen",
    "spacious": "lightgreen",
    "airy": "cornflowerblue",
    "spread": "lightblue",
    "nan":"grey"
}
# Density expander...
with st.expander(f"Density nomograms for {add}", expanded=True):
    # maxes
    FSI_scale_max = case_data['FSI'].quantile(0.9)
    #OSR
    fig_OSR = px.scatter(case_data, title='Buildings colored by OSR per (mophological) plot',
                                      x='GSI', y='FSI', color='OSR_class', #size='GFA',
                                      log_y=False,
                                      hover_name='building',
                                      hover_data=['addr:street','floors','GFA','OSR','OSR_ND'],
                                      labels={"OSR_class": 'Plot density'},
                                      category_orders={'OSR_class': ['close','dense','compact','spacious','airy','spread']},
                                      color_discrete_map=colormap_osr
                                      )
    fig_OSR.update_layout(xaxis_range=[0, 0.75], yaxis_range=[0, FSI_scale_max])
    fig_OSR.update_xaxes(rangeslider_visible=True)

    #OSR_ND
    fig_OSR_ND = px.scatter(case_data, title='Buildings colored by OSR per neighbourhood',
                            x='GSI', y='FSI', color='OSR_ND_class', #size='GFA',
                            log_y=False,
                            hover_name='building',
                            hover_data=['addr:street','floors','GFA','OSR','OSR_ND'],
                            labels={"OSR_ND_class": 'Neigbourhood density'},
                            category_orders={'OSR_ND_class': ['close','dense','compact','spacious','airy','spread']},
                            color_discrete_map=colormap_osr
                            )
    fig_OSR_ND.update_layout(xaxis_range=[0, 0.75], yaxis_range=[0, FSI_scale_max])
    fig_OSR_ND.update_xaxes(rangeslider_visible=True)
    
    # charts..
    col1, col2 = st.columns(2)
    col1.plotly_chart(fig_OSR, use_container_width=True)
    col2.plotly_chart(fig_OSR_ND, use_container_width=True)
    
    # summary
    bu_count = len(case_data)
    tot_gfa = round(case_data['GFA'].sum(),-2)
    e_area = tot_gfa/785375
    #st.markdown(f'Total GFA: **{tot_gfa:,}** , Median plot density FSI/e=**{e_plot}**')
    # calc tag accounts
    tags = case_data.groupby(['building'])['building'].count()
    toptags = tags.sort_values(ascending=False).head(3)
    m1,m2 = st.columns(2)
    m1.metric(label=f"Total GFA in {add} in 500m radius ({bu_count} buildings)", value=f"{tot_gfa:,.0f} sqrm", delta=f"Areal density = {e_area:.2f}")
    #m2.metric(label=f"GFA {toptags.index[0]}", value=f"{toptags[0][0]:.0f}", delta=f"{e_plot:.0f}")
    st.caption('Values are based on footprints and floor number information. Underground GFA is excluded.')
    # continue some day..

    st.markdown('---')
    # describe_table
    st.markdown(f'{add} data described')
    des = case_data.drop(columns=['osmid', 'uID']).describe()
    st.dataframe(des)
    #st.dataframe(density_data.drop(columns='geometry'))

    # prepare save..
    save_data = gpd.overlay(density_data.to_crs(3067), focus_gdf.set_crs(3067), how='intersection').to_crs(4326)
    save_data.insert(0, 'TimeStamp', pd.to_datetime('now').replace(microsecond=0))
    save_data['date'] = save_data['TimeStamp'].dt.date
    save_me = save_data.drop(columns=(['uID', 'TimeStamp','OSR_class','OSR_ND_class'])).assign(location=add)
    save_me = save_me.assign(flr_rate=flr_rate)
    raks = save_me.to_csv().encode('utf-8')
    save = st.download_button(label="Save density data as CSV", data=raks, file_name=f'buildings_{add}.csv',mime='text/csv')

# expl container
with st.expander("What is this?", expanded=False):
    st.markdown('Density measures in the nomogram above are derived from the latest density research by'
                ' Meta Berghouser Pont and Per Haupt (2021), Kim Dowey and Elek Pafka (2014) as well as'
                ' from Finnish seminal work by O-I Meurman in 1947.')
    # expl
    selite = '''
    **Density measures**<br>
    **GFA** = Gross Floor Area = Total area of in-door space in building including all floors<br>
    **FSI** = Floor Space Index = FAR = Floor Area Ratio = Ratio of floor area per total area of _morphological plot_<br>
    **GSI** = Ground Space Index = Coverage = Ratio of building footprint per total area of _morphological plot_<br>
    **OSR** = Open Space Ratio = Ratio of non-build space per square meter of gross floor area<br>
    **i_ND** = Value of the _i_-index in neighborhood scale<br>
    **OSR_ND_mean** = Average OSR of plots in nearby neighborhood<br>
    
    Density classification is based on OSR-values:<br>
    <i>
    Close: OSR < 1 <br>
    Dense: OSR 1-2 <br>
    Compact: OSR 2-4 <br>
    Spacious: OSR 4-8 <br>
    Airy: OSR 8-16 <br>
    Spread: OSR > 16 <br>
    </i>
    <br>
    _Morfological plot_ is a plot generated using polygonal tessellation around buildings using 
    <a href="http://docs.momepy.org/en/stable/user_guide/elements/tessellation.html" target="_blank">Momepy</a>.<br>
    Nearby neighborhood in OSR_ND calculation is based on queen contiguity for 2 degree neighbours 
    (border neighbors and their neighbours as an "experienced neighborhood").<br>
    <br>
    Average OSR values of morphological plots classify urban density well as they combine both
    the volume of architecture (FSI) and the compactness of urban planning (GSI).
    '''
    soveltaen = '''
    <p style="font-family:sans-serif; color:Dimgrey; font-size: 12px;">
    References:<br><i>
    Berghauser Pont, Meta, and Per Haupt. 2021. Spacematrix: Space, Density and Urban Form. Rotterdam: nai010 publishers.<br>
    Dovey, Kim, Pafka, Elek. 2014. The urban density assemblage: Modelling multiple measures. Urban Des Int 19, 66–76<br>
    Meurman, Otto-I. 1947. Asemakaavaoppi. Helsinki: Rakennuskirja.<br>
    Fleischmann, Martin. 2019. momepy: Urban Morphology Measuring Toolkit. Journal of Open Source Software, 4(43), 1807<br>
    Boeing, G. 2017. “OSMnx: New Methods for Acquiring, Constructing, Analyzing, and Visualizing Complex Street Networks.” Computers, Environment and Urban Systems. 65, 126-139.<br>
    </i>
    </p>
    '''
    st.markdown(selite, unsafe_allow_html=True)
    cs1, cs2, cs3 = st.columns(3)
    cs1.latex(r'''
            OSR = \frac {1-GSI} {FSI}
            ''')  # https://katex.org/docs/supported.html

    st.markdown(soveltaen, unsafe_allow_html=True)
