#python we go
import pandas as pd
import geopandas as gpd
import numpy as np
import json
from scipy.stats import boxcox
import streamlit as st
import plotly.express as px
import plotly.graph_objs as go
px.set_mapbox_access_token(st.secrets['MAPBOX_TOKEN'])
my_style = st.secrets['MAPBOX_STYLE']
from pathlib import Path
import h3
from shapely import wkt
from shapely.geometry import shape


# content
st.title("Data Paper #4")
st.subheader("Väestömuutos eri maankäytön tehokkuuden luokissa")
ingress = '''
<p style="font-family:sans-serif; color:Black; font-size: 14px;">
Tämä tutkimusappi tarkastelee väestökasvua 
<a href="https://ckan.ymparisto.fi/en/dataset/harva-ja-tihea-taajama-alue" target="_blank">SYKEn</a>  
laatimissa maankäytön tehokkuuden luokissa. Tarkastelu toiminut taustamateriaalina artikkelissa:  
<a href="https://doi.org/10.1080/09654313.2024.2370314" target="_blank">Density as an indicator of sustainable urban development: insights from Helsinki?</a>
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
    return gdf.reset_index(), center

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
year = c1.selectbox('Vuosi', ['2020','2010','2000','1990',])
#zone
zone = c2.selectbox('Vyöhyke',['Tiheä taajama','Harva taajama','Kylät ja maaseutu'])
#reso
resoh = c3.radio('Resoluutio',['H9','H8','H7'], horizontal=True) #['~5km²','~1km²','~1ha']
reso = int(resoh[-1])

#legend
keys = {
    'Tiheä taajama':1,
    'Harva taajama':2,
    'Kylät':3,
    'Pienkylät':4,
    'Maaseutuasutus':5
    }


# filter & agg
def aggregate_h3_resolution(gdf_in, h3_col="h3_10", target_reso=9):
    df = gdf_in.drop(columns='geometry')
    df["h3_parent"] = df[h3_col].apply(lambda x: h3.cell_to_parent(x, target_reso))
    agg_df = (
        df.groupby("h3_parent")
        .agg('sum')
        .reset_index()
        ).drop(columns=h3_col)
    agg_df = agg_df.rename(columns={"h3_parent": "h3_id"})
    agg_df['geojsonpolygon'] = agg_df["h3_id"].apply(lambda cell: h3.cells_to_geo([cell], tight=True))
    agg_df['geometry'] = agg_df['geojsonpolygon'].apply(lambda p: shape(p))
    agg_gdf = gpd.GeoDataFrame(agg_df,geometry='geometry',crs=4326)
    return agg_gdf.drop(columns='geojsonpolygon')

if zone != 'Kylät ja maaseutu':
    filtered = mygdf.loc[mygdf[yrs[year]] == keys[zone]]
    plot = aggregate_h3_resolution(filtered, h3_col='h3_10', target_reso=reso)

else:
    subzones = ['Kylät','Pienkylät','Maaseutuasutus'] #=[3,4,5]
    filtered = mygdf.loc[mygdf[yrs[year]].isin([3,4,5])]
    plot = aggregate_h3_resolution(filtered, target_reso=reso)

# map
with st.expander('Tehokkuus kartalla', expanded=False):
    #feature to plot
    densityof = st.radio('Tehokkuusmittari',['Väestö','Kerrosala'], horizontal=True)
    #calc densities
    plot['h3_cell_area'] = plot['h3_id'].apply(lambda x: h3.cell_area(h=x,unit='m^2'))
    yr_list = ['90','00','10','20']
    for yr in yr_list:
        plot[f'den_pop{yr}'] = round(plot[f'pop{yr}'] / (plot['h3_cell_area']/10000),-1)
        plot[f'den_gfa{yr}'] = round(plot[f'GFA{yr}'] / plot['h3_cell_area'],3)
        plot[f'class_pop{yr}'] = 'harva'
        plot[f'class_gfa{yr}'] = 'harva'
        #popdens
        plot.loc[plot[f'den_pop{yr}'] > 1, f'class_pop{yr}'] = "hajanainen" #'sprawl'
        plot.loc[plot[f'den_pop{yr}'] > 10, f'class_pop{yr}'] = "väljä" #'spacious'
        plot.loc[plot[f'den_pop{yr}'] > 50, f'class_pop{yr}'] = "kompakti" #'compact'
        plot.loc[plot[f'den_pop{yr}'] > 70, f'class_pop{yr}'] = "tiivis" #'dense'
        #gfadense
        plot.loc[plot[f'den_gfa{yr}'] > 0.10, f'class_gfa{yr}'] = 'hajanainen'
        plot.loc[plot[f'den_gfa{yr}'] > 0.15, f'class_gfa{yr}'] = 'väljä'
        plot.loc[plot[f'den_gfa{yr}'] > 0.30, f'class_gfa{yr}'] = 'kompakti'
        plot.loc[plot[f'den_gfa{yr}'] > 0.60, f'class_gfa{yr}'] = 'tiivis'

    colormap = {
        "tiivis": "darkgoldenrod",
        "kompakti": "darkolivegreen",
        "väljä": "lightgreen",
        "hajanainen": "lightblue",
        "harva":"lightcyan"
    }

    lat = center[0]
    lon = center[1]
    if densityof == 'Väestö':
        mycolor = f'class_pop{yr}'
        myclass = 'pop'
    else:
        mycolor = f'class_gfa{yr}'
        myclass = 'gfa'

    fig_map = px.choropleth_mapbox(plot,
                            geojson = plot.geometry, #plot.geometry.iloc[0].__geo_interface__,
                            locations=plot.index,
                            title=f"Vyöhyke '{zone}' vuoden {year} luokittelun mukaan resoluutiolla H{reso}.",
                            color=mycolor,
                            hover_data=[f'den_pop{yr}',f'den_gfa{yr}'],
                            color_discrete_map=colormap,
                            labels={f'class_{myclass}{yr}': f'{densityof}'},
                            category_orders={f'class_{myclass}{yr}': ['tiivis','kompakti','väljä','hajanainen','harva']},
                            #range_color=(range_min, range_max),
                            #color_continuous_scale=px.colors.sequential.Inferno[::-1],
                            center={"lat": lat, "lon": lon},
                            mapbox_style=my_style,
                            zoom=9,
                            opacity=0.5,
                            width=1200,
                            height=700
                            )

    fig_map.update_layout(margin={"r": 10, "t": 50, "l": 10, "b": 10}, height=700,
                                legend=dict(
                                    yanchor="top",
                                    y=0.97,
                                    xanchor="left",
                                    x=0.02
                                )
                                )

    st.plotly_chart(fig_map, use_container_width=True)

    #st.dataframe(plot.drop(columns='geometry'))

# graphs
st.subheader(f"Väestöosuudet eri luokissa")
hex_area = round(plot['h3_cell_area'].mean()/10000,1)

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
    fig.update_layout(title_text=f"Väestön osuus eri luokissa (H{reso}).")
    fig.update_xaxes(title='Year')
    fig.update_yaxes(title='% of total population above quantile')
    return fig

# pop class shares
den_list = ['tiivis','kompakti','väljä','hajanainen','harva'] #["dense","compact","spacious","sprawl","less"]
def density_class_share(plot,feat,den_list):
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

pop_shares = density_class_share(plot,feat='pop',den_list=den_list)
gfa_shares = density_class_share(plot,feat='gfa',den_list=den_list)

def pop_share_plot(df):
    linecolors = px.colors.qualitative.Plotly
    fig = go.Figure()
    fig.add_traces(go.Scatter(x=df.index, y = df[f'share_{den_list[0]}'],name=den_list[0], mode = 'lines', line=dict(color=linecolors[0])))
    fig.add_traces(go.Scatter(x=df.index, y = df[f'share_{den_list[1]}'],name=den_list[1], mode = 'lines', line=dict(color=linecolors[1])))
    fig.add_traces(go.Scatter(x=df.index, y = df[f'share_{den_list[2]}'],name=den_list[2], mode = 'lines', line=dict(color=linecolors[2])))
    fig.add_traces(go.Scatter(x=df.index, y = df[f'share_{den_list[3]}'],name=den_list[3], mode = 'lines', line=dict(color=linecolors[3])))
    fig.add_traces(go.Scatter(x=df.index, y = df[f'share_{den_list[4]}'],name=den_list[4], mode = 'lines', line=dict(color=linecolors[4])))
    fig.update_layout(title_text=f"Väestöosuudet tiiviysluokissa väestön mukaan")
    fig.update_xaxes(title='Vuosi')
    fig.update_yaxes(title='% väestöstä luokassa')
    return fig

def gfa_share_plot(df):
    linecolors = px.colors.qualitative.Plotly
    fig = go.Figure()
    fig.add_traces(go.Scatter(x=df.index, y = df[f'share_{den_list[0]}'],name=den_list[0], mode = 'lines', line=dict(color=linecolors[0])))
    fig.add_traces(go.Scatter(x=df.index, y = df[f'share_{den_list[1]}'],name=den_list[1], mode = 'lines', line=dict(color=linecolors[1])))
    fig.add_traces(go.Scatter(x=df.index, y = df[f'share_{den_list[2]}'],name=den_list[2], mode = 'lines', line=dict(color=linecolors[2])))
    fig.add_traces(go.Scatter(x=df.index, y = df[f'share_{den_list[3]}'],name=den_list[3], mode = 'lines', line=dict(color=linecolors[3])))
    fig.add_traces(go.Scatter(x=df.index, y = df[f'share_{den_list[4]}'],name=den_list[4], mode = 'lines', line=dict(color=linecolors[4])))
    fig.update_layout(title_text=f"Väestöosuudet tiiviysluokissa kerrosalan mukaan")
    fig.update_xaxes(title='Vuosi')
    fig.update_yaxes(title='% väestöstä luokassa')
    return fig

#plots in tabs
tab1,tab2 =  st.tabs(['Väestötiheyden mukaan','Kerrosalamäärän mukaan'])

with tab1:
    fig_pop_share = pop_share_plot(pop_shares)
    st.plotly_chart(fig_pop_share, use_container_width=True)
with tab2:
    fig_gfa_share = gfa_share_plot(gfa_shares)
    st.plotly_chart(fig_gfa_share, use_container_width=True)

selite = """
<b>Luokittelu:</b><br>
<i>
Tiivis: ae > 0.6 | as/ha > 70 <br>
Kompakti: 0.6 - 0.3 | 70 - 50 <br>
Väljä: 0.3 - 0.15 | 50 - 10 <br>
Hajanainen: 0.15 - 0.10 | 10 - 2 <br>
Harva: ae < 0.10 | as/ha < 2 <br>
[ ae = aluetehokkuus ]
</i>
<br>
"""
st.markdown(selite, unsafe_allow_html=True)
