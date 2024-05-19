# NDP app always beta a lot
import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely import wkt
from shapely.geometry import MultiPoint, Point
import osmnx as ox
import math
import geocoder
from sklearn.cluster import DBSCAN
import boto3
import requests
import io
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots

#ML
from scipy import stats
import pingouin as pg
import statsmodels.api as sm
import statsmodels.formula.api as smf

#g
from streamlit_gsheets import GSheetsConnection


px.set_mapbox_access_token(st.secrets['MAPBOX_TOKEN'])
mbtoken = st.secrets['MAPBOX_TOKEN']
my_style = st.secrets['MAPBOX_STYLE']
key = st.secrets['bucket']['key']
secret = st.secrets['bucket']['secret']
url = st.secrets['bucket']['url']
cfua_allas = st.secrets['allas']['url']
allas_key = st.secrets['allas']['access_key_id']
allas_secret = st.secrets['allas']['secret_access_key']


# page setup ---------------------------------------------------------------
st.set_page_config(page_title="Data Paper #5", layout="wide", initial_sidebar_state='collapsed')
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
        NDP data paper #5 V1.3\
        </p>'
st.markdown(header, unsafe_allow_html=True)
# plot size setup
#px.defaults.width = 600
px.defaults.height = 700

st.markdown("----")
# content
st.title("Data Paper #5")
st.subheader("CFUA - Climate Friendly Urban Architecture")
st.markdown('This data paper is seeking urban features which correlate with low carbon lifestyles.')
st.caption("Authors: Teemu Jama, Jukka Heinonen @[Háskóli Íslands](https://uni.hi.is/heinonen/), Henrikki Tenkanen @[Aalto GIST Lab](https://gistlab.science)")
st.markdown("###")

def check_password():
    def password_entered():
        if (
            st.session_state["password"]
            == st.secrets["passwords"]["cfua"]
        ):
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False
    if "password_correct" not in st.session_state:
        st.text_input(label="password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input(label="password", type="password", on_change=password_entered, key="password")
        return False
    else:
        return True

#data handler
def spaces_csv_handler(file_name=None, folder_name="ndp", operation=None, data_frame=None):
    bucket_name='ana'
    session = boto3.session.Session()
    client = session.client('s3',
                            region_name='ams3',
                            endpoint_url=url,
                            aws_access_key_id=key, 
                            aws_secret_access_key=secret 
                            )
    
    def download_csv_from_spaces(client, bucket_name, file_name):
        obj = client.get_object(Bucket=bucket_name, Key=file_name)
        df = pd.read_csv(obj['Body'], on_bad_lines='skip')
        return df

    def upload_csv_to_spaces(client, bucket_name, file_name, data_frame):
        csv_buffer = data_frame.to_csv(index=False)
        client.put_object(Bucket=bucket_name, Key=file_name, Body=csv_buffer, ContentType='text/csv')

    def list_files_from_bucket(client,bucket_name,folder_name):
        # List objects in the specified folder
        objects = client.list_objects_v2(Bucket=bucket_name, Prefix=folder_name)

        # Initialize a list to hold CSV file names
        csv_files = []

        # Iterate over each object in the specified folder
        for obj in objects.get('Contents', []):
            file_name = obj['Key']
            # Check if the file is a CSV
            if file_name.endswith('.csv'):
                csv_files.append(file_name)

        return csv_files
    
    if operation == 'download':
        return download_csv_from_spaces(client, bucket_name, file_name)
    elif operation == 'upload' and data_frame is not None:
        upload_csv_to_spaces(client, bucket_name, file_name, data_frame)
    elif operation == 'list':
        return list_files_from_bucket(client,bucket_name,folder_name)
    else:
        raise ValueError("Invalid operation or missing data for upload")

#@st.cache_data()
def allas_get(filepath):
    r = requests.get(filepath, stream=True)
    data = io.BytesIO(r.content)
    cfua_data = pd.read_csv(data)
    cfua_data = cfua_data.loc[:, ~cfua_data.columns.str.startswith('Unnamed')]
    return cfua_data
    


auth = check_password()
if not auth:
    st.stop()




# ------------------ plotters ------------------

#map plotter
def plot_sample_clusters(gdf_in,cf_col="Total footprint"):

    gdf = gdf_in.copy()
    lat = gdf.unary_union.centroid.y
    lon = gdf.unary_union.centroid.x
    #
    # Define fixed labels and colors
    labels_colors = {
        'Bottom': 'rgba(144, 238, 144, 0.6)',
        'Low': 'rgba(254, 220, 120, 0.8)',
        'High': 'rgba(254, 170, 70, 1)',
        'Top': 'rgba(253, 100, 80, 1)'
    }

    # Get unique sorted values
    sorted_unique_values = sorted(gdf[cf_col].unique())

    # Determine number of bins
    num_bins = min(len(sorted_unique_values), 4)  # Limit the number of bins to 4

    # Generate bin edges ensuring they are unique and cover the range of values
    bin_edges = np.percentile(gdf[cf_col], np.linspace(0, 100, num_bins + 1))

    # Ensure unique bin edges
    bin_edges = np.unique(bin_edges)

    # Adjust the number of labels based on the number of bin edges
    num_labels = len(bin_edges) - 1
    labels = list(labels_colors.keys())[:num_labels]

    # Assign quartile class
    gdf['cf_class'] = pd.cut(gdf[cf_col], bins=bin_edges, labels=labels, include_lowest=True)
    #ensure all classes exist
    for label in labels_colors.keys():
        # Check if the label is already present
        if label not in gdf['cf_class'].tolist():
            # Create a dummy row DataFrame with the missing category
            dummy_row_df = pd.DataFrame([{cf_col: None, 'cf_class': label}])
            # Use pd.concat to append the dummy row DataFrame
            gdf = pd.concat([gdf, dummy_row_df], ignore_index=True)

    fig_map = px.choropleth_mapbox(gdf,
                            geojson=gdf.geometry,
                            locations=gdf.index,
                            title="Sample areas of on map",
                            color='cf_class',
                            hover_name='cf_class',
                            color_discrete_map=labels_colors,
                            category_orders={"cf_class":['Top','High','Low','Bottom']},
                            labels={'cf_class': f'{cf_col} level'},
                            center={"lat": lat, "lon": lon},
                            mapbox_style=my_style,
                            zoom=10,
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
    return fig_map







# --------------------- THE CONTENT ------------------------

tab1,tab2,tab3 = st.tabs(['Clusterizer','Urban data','Regressor'])

with tab1:
    @st.cache_data()
    def get_orig():
        df = spaces_csv_handler(file_name="ndp/cfua/CFUA_rev1.csv",operation="download")
        if 'Unnamed: 0' in df.columns:
            df.drop(columns='Unnamed: 0', inplace=True)
        for col in df.columns:
            # Only process if column is not in the excluded column
            if col not in ["city"]:
                # Convert to numeric float
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    
    with st.status('CF data'):
        data = get_orig()
        st.text(f"N = {len(data)}")
        featlist = data.columns.tolist()
        st.data_editor(data)
    
    #clusterizer
    def cluster_and_generate_polygons(df, radius=500, min_samples=2, max_size=9, ratio=None):
        # Calculate average latitude for more accurate distance approximation
        average_latitude = df['lat'].mean()
        latitude_radians = math.radians(average_latitude)
        eps = radius / (111320 * math.cos(latitude_radians))
        
        # Convert lat and lng columns to Shapely Point geometries
        df['geometry'] = [Point(xy) for xy in zip(df['lng'], df['lat'])]
        
        # Perform initial DBSCAN clustering
        df['cluster'] = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(df[['lat', 'lng']])
        
        # Identify and process oversized clusters
        clusters_to_split = df['cluster'].value_counts()[df['cluster'].value_counts() > max_size].index.tolist()
        next_cluster_id = df['cluster'].max() + 1  # Start new cluster IDs from here
        
        for cluster_id in clusters_to_split:
            # Filter points belonging to the oversized cluster
            mask = df['cluster'] == cluster_id
            points = df.loc[mask, ['lat', 'lng']].to_numpy()
            
            # Re-cluster with adjusted parameters to split the oversized cluster
            re_eps = eps * 0.75
            new_clusters = DBSCAN(eps=re_eps, min_samples=min_samples).fit_predict(points)
            
            # Update the cluster labels in the DataFrame
            new_labels = np.where(new_clusters == -1, -1, new_clusters + next_cluster_id)
            df.loc[mask, 'cluster'] = new_labels
            next_cluster_id += new_clusters.max() + 1  # Update for the next possible cluster ID
        
        # Filter out noise points
        clustered_df = df[df['cluster'] != -1]
        
        rows = []
        for cluster_id in clustered_df['cluster'].unique():
            # Subset points in the current cluster
            cluster_points = clustered_df[clustered_df['cluster'] == cluster_id]
            
            # Create a polygon (convex hull) around points in the cluster
            polygon = MultiPoint(cluster_points['geometry'].tolist()).convex_hull
            
            # Calculate median values for all other columns
            median_values = cluster_points.median(numeric_only=True)
            
            # Create a row for the GeoDataFrame
            row = median_values.to_dict()
            row['geometry'] = polygon
            rows.append(row)
            
        # Create a GeoDataFrame from the rows
        gdf = gpd.GeoDataFrame(rows)
        
        if ratio is not None:
            max_area = gdf['geometry'].area.max()
            target_area = max_area * ratio
            
            def calculate_buffer(area, target_area):
                if area < target_area:
                    return (target_area - area) ** 0.5 * 0.5
                return 0

            gdf['buffer_distance'] = gdf['geometry'].area.apply(lambda x: calculate_buffer(x, target_area))
            gdf['geometry'] = gdf.apply(lambda row: row['geometry'].buffer(row['buffer_distance']), axis=1)
        return gdf.drop(columns=['cluster','lat', 'lng'])



    # select city
    top50 = pd.DataFrame(data=data['city'].value_counts()).head(50)
    city_list = top50.index.to_list()
    city_list.pop(0)
    city_list.insert(0,'Greater Helsinki')
    city_list.insert(0,'..')

    c1,c2 = st.columns(2)
    my_city = c1.selectbox('Select city',city_list)
    my_def = featlist.index('Total footprint')
    my_col = c2.selectbox('Choose value', featlist, index=my_def)

    if my_city == '..':
        plot_filtered = None
    elif my_city == 'Greater Helsinki':
        gh_cities = ['Helsinki','Espoo','Vantaa','Kauniainen']
        plot_filtered = data.loc[data['city'].isin(gh_cities)]
    else:
        plot_filtered = data.loc[data['city'] == my_city]
        
    #cluster widgets
    s1,s2 = st.columns(2)
    radius = s1.slider('Search radius',100,900,500,100)
    min_max = s2.slider('Min/max points',3,9,[3,7],1)
    harmo = True #st.toggle('Harmonize')
    
    #clusters
    if my_city != '..':
        if harmo:
            r=0.1
        else:
            r = None
        sample = cluster_and_generate_polygons(plot_filtered, radius=radius, ratio=r,
                                               min_samples=min_max[0], max_size=min_max[1])
        st.text(f"Clusters generated for {my_city} {len(sample)}")
        with st.status('clusters', expanded=True):
            map_plot = plot_sample_clusters(sample,cf_col=my_col)
            st.plotly_chart(map_plot, use_container_width=True, config = {'displayModeBar': False} )
        
        with st.expander(f'Save sample for {my_city}',expanded=False):
            
            def add_centroid_id(gdf, id_name='clusterID'):
                # Assuming 'gdf' is your GeoDataFrame with polygon geometries
                gdf['centroid'] = gdf.geometry.centroid
                gdf['lat'] = gdf.centroid.y
                gdf['lng'] = gdf.centroid.x

                def get_street_name(lat, lng):
                    g = geocoder.osm([lat, lng], method='reverse')
                    if g.ok:
                        return g.street
                    return None

                # Apply the function to get street names
                gdf['street_name'] = gdf.apply(lambda x: get_street_name(x['lat'], x['lng']), axis=1)

                # Replace None or NaN in street_name with a placeholder like 'unknownstreet'
                gdf['street_name'].fillna('unknown_street', inplace=True)

                # Handle potential duplicates in street names by appending a unique suffix if needed
                gdf[id_name] = gdf.groupby('street_name').cumcount().astype(str).radd(gdf['street_name'] + '_')

                gdf.drop(columns=['centroid', 'lat', 'lng'], inplace=True)

                return gdf
            
            #add geom as wkt for save
            sample['wkt'] = sample['geometry'].apply(lambda x: x.wkt)
            sample["city"] = my_city

            col_order = ['clusterID','city', 'Number of persons in household','Income Level decile',
                        'Total footprint',
                        'Goods and services footprint',
                        'Public transportation footprint',
                        'Vehicle possession footprint',
                        'Housing footprint',
                        'Leisure travel footprint',
                        'Summer house footprint',
                        'Diet footprint',
                        'Pets footprint',
                        'wkt'
                        ]
            st.data_editor(sample[col_order[1:]])

            d1,d2 = st.columns(2)
            #save to bucket
            repo_save = d1.button('Save in database')
            if repo_save:
                #add cluster ID
                cluster_id_name = "clusterID"
                filtered_buffers = add_centroid_id(gdf=sample,id_name=cluster_id_name)
                df_out = filtered_buffers[col_order]
                #prepare save
                specs = f"R{radius}Pmin{min_max[0]}Pmax{min_max[1]}N{len(sample)}"
                cityname = my_city.replace(" ", "")
                file_name = f"ndp/cfua/CFDATA_{cityname}_({specs}).csv"
                spaces_csv_handler(file_name=file_name,operation="upload",data_frame=df_out)






with tab2:
    st.markdown('Open notebook in [https://www.puhti.csc.fi/](https://www.puhti.csc.fi/) to generate urban data..')


with tab3:
    reg_file = f"{cfua_allas}REG/cf_reg_ready.csv"
    cf_reg_raw = allas_get(reg_file)
    
    import re
    import string
    #clean col names from white spaces etc..
    def clean_col_names(df):
        df.columns = [
            ''.join(filter(lambda x: x in string.printable, re.sub(r'[^\w\s]', '', re.sub(r'\s+', '_', col.strip())))).lower()
            for col in df.columns
        ]
        return df
    
    cf_reg_all = clean_col_names(cf_reg_raw)
    
    def bin_age(df):
        bins = [0, 20, 40, 60, float('inf')]
        labels = ['young', 'adult', 'senior', 'seasoned']
        df['age_class'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)
        return df
    
    cf_reg_all = bin_age(cf_reg_all)
    
    def calculate_per_capita(df, cluster_col, cols):
        grouped = df.groupby(cluster_col)
        per_capita = round(grouped[cols].sum().div(grouped.size(), axis=0),-1)
        return per_capita
    
    def remove_outliers_and_normalize(df, cols, norm='Percent'):
            if norm == 'IQR':
                for col in cols:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    df_out = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                    df_out[col] = np.log1p(df_out[col])
            elif norm == 'Percent':
                for col in cols:
                    lower_bound = df[col].quantile(0.01)
                    upper_bound = df[col].quantile(0.91)
                    df_out = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                    df_out[col] = np.log1p(df_out[col])
            else:
                for col in cols:
                    lower_bound = df[col].quantile(0.05)
                    upper_bound = df[col].quantile(0.95)
                    df_out = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                    
            return df_out
    
    with st.expander('Per capita values in cities'):
        
        st.text(f"Sample size {len(cf_reg_all)}")
    
        reg_cols = cf_reg_all.columns.tolist()[3:12]
        
        city_results = calculate_per_capita(cf_reg_all, 'city', reg_cols).reset_index()
        
        norm = st.toggle('Remove outliers')
        if norm:
            city_results = remove_outliers_and_normalize(city_results, reg_cols, norm='none')
        else:
            city_results = city_results.copy()
        #the plot
        bivar_fig = px.bar(city_results, x="city", y=reg_cols[:-1], title="Per capita footprints in Cities")
        st.plotly_chart(bivar_fig, use_container_width=True, config = {'displayModeBar': False} )
        
    st.markdown('---')
    
    st.subheader('Multi-regression study')
    #cf_reg_all.columns = cf_reg_all.columns.str.replace(r'\W+', '_').str.strip('_').str.replace(r'^(\d+)', r'_\1')
    #myindepvars = cf_reg_all.columns.tolist()[12:19]
    #var_cols = c2.multiselect('Independent vars',myindepvars,default=myindepvars)
    
    with st.expander('Normalized data'):
        method = st.radio('Method for outliers',['Percent','IQR'],horizontal=True)
        cf_normalized = remove_outliers_and_normalize(cf_reg_all, reg_cols, norm=method)
        
        st.data_editor(cf_normalized)
        hist_place = st.empty()

    default_inx = reg_cols.index('total_footprint')
    c1,c2,c3 = st.columns(3)
    cf_col = c1.selectbox('Domain to study',options=reg_cols,index=default_inx)
    #indepcols = ['age_class','education_level','income_level_decile','household_type','urban_degree','country']
    #ipcols = c2.multiselect('Indep.vars',indepcols,default=indepcols.remove('age_class'))
    hist = px.histogram(cf_normalized, x=cf_col, color="country", title='Normalized (log1p) distribution')
    hist_place.plotly_chart(hist, use_container_width=True, config = {'displayModeBar': False} )

    if st.toggle('Calculate regression'):
        
        with st.status('Calculating..',expanded=True) as stat:
            def ols_reg(df,cf_col='total_footprint'):
                base_formula = f'{cf_col} ~ C(age_class) + C(education_level) + C(income_level_decile) + C(household_type) + C(urban_degree) + C(country)'
                ext_formula = f'{cf_col} ~ C(age_class) + C(education_level) + C(income_level_decile) + C(household_type) + C(urban_degree) + C(country) + res_gfa_loc + res_gfa_nd + other_gfa_loc + other_gfa_nd + services_loc + services_nd + sdi_loc + sdi_nd'
                base_model = smf.ols(formula=base_formula, data=df)
                base_results = base_model.fit()
                ext_model = smf.ols(formula=ext_formula, data=df)
                ext_results = ext_model.fit()
                return base_results, ext_results

            base_results, ext_results = ols_reg(df=cf_normalized,cf_col=cf_col)
            
            with st.container(height=300):
                s1,s2 = st.columns(2)
                s1.markdown('**Without urban features** (Base model)')
                s1.text(base_results.summary())
                s2.markdown('**With urban features** (Ext model)')
                s2.text(ext_results.summary())
            
            reg_results = pd.DataFrame({
                            'Base β': base_results.params,
                            'Base p': base_results.pvalues,
                            'Ext β': ext_results.params,
                            'Ext p': ext_results.pvalues,
                        })
            st.data_editor(reg_results,use_container_width=True)
            
            stat.update(label="Done!", state="complete", expanded=True)

            #the plot
            #reg_fig = px.bar(reg_results, x=['Base β','Base p','Ext β','Ext p'], y=reg_results.index,
            #                 range_x=[-1,1],title="Comparison")
            #st.plotly_chart(reg_fig, use_container_width=True, config = {'displayModeBar': False} )

#footer
st.markdown('---')
footer_title = '''
**NDP Project**
[![MIT license](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/teemuja/NDP/blob/main/LICENSE) 
'''
st.markdown(footer_title)
disclamer = 'Data papers are constant work in progress and will be upgraded, changed & fixed while research go on.'
st.caption('Disclaimer: ' + disclamer)
