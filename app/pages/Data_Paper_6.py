# NDP app always beta a lot
import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely import wkt
from shapely.geometry import MultiPoint, Point
import math
import geocoder
from sklearn.cluster import DBSCAN
import boto3
import re
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pingouin as pg
from sklearn.preprocessing import MinMaxScaler


px.set_mapbox_access_token(st.secrets['MAPBOX_TOKEN'])
my_style = st.secrets['MAPBOX_STYLE']
key = st.secrets['bucket']['key']
secret = st.secrets['bucket']['secret']
url = st.secrets['bucket']['url']


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
        NDP data paper #5 V1.2\
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

tab1,tab2,tab3 = st.tabs(['Clusterizer','Analyzer','Classifier'])

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
    ver = "v2" #st.radio("Sample set version",['v1','v2'],horizontal=True)

    csv_list = spaces_csv_handler(operation="list",folder_name="ndp/cfua")

    names = []
    for file_name in csv_list:
        file_name_with_extension = file_name.split('/')[-1]
        name = file_name_with_extension.split('.')[0]
        names.append(name)
    if ver == 'v1':
        pattern = f"CFUADATA" #r'_sample_N\d+$'
    else:
        pattern = f"CFUADATA_v2" #r'_sample_N\d+$'
    filtered_names = [name for name in names if re.search(pattern, name)]
    selectbox_names = filtered_names.copy()
    selectbox_names.insert(0,"...")
    selectbox_names.append("All_samples")
    selected_urb_file = st.selectbox('Select sample file to analyze',selectbox_names)

    #map plotter 2
    def plot_sample_areas(df,cf_col="Total footprint"):
        #get gdf
        df['geometry'] = df.wkt.apply(wkt.loads)
        case_data = gpd.GeoDataFrame(df,geometry='geometry')
            
        lat = case_data.unary_union.centroid.y
        lon = case_data.unary_union.centroid.x
        #
        # Define fixed labels and colors
        labels_colors = {
            'Bottom': 'rgba(144, 238, 144, 0.6)',
            'Low': 'rgba(64,224,208, 0.8)',
            'High': 'rgba(254, 170, 70, 1)',
            'Top': 'rgba(253, 100, 80, 1)'
        }

        # Get unique sorted values
        sorted_unique_values = sorted(case_data[cf_col].unique())

        # Determine number of bins
        num_bins = min(len(sorted_unique_values), 4)  # Limit the number of bins to 4

        # Generate bin edges ensuring they are unique and cover the range of values
        bin_edges = np.percentile(case_data[cf_col], np.linspace(0, 100, num_bins + 1))

        # Ensure unique bin edges
        bin_edges = np.unique(bin_edges)

        # Adjust the number of labels based on the number of bin edges
        num_labels = len(bin_edges) - 1
        labels = list(labels_colors.keys())[:num_labels]

        # Assign quartile class
        case_data['cf_class'] = pd.cut(case_data[cf_col], bins=bin_edges, labels=labels, include_lowest=True)
        #ensure all classes exist
        for label in labels_colors.keys():
            # Check if the label is already present
            if label not in case_data['cf_class'].tolist():
                # Create a dummy row DataFrame with the missing category
                dummy_row_df = pd.DataFrame([{cf_col: None, 'cf_class': label}])
                # Use pd.concat to append the dummy row DataFrame
                case_data = pd.concat([case_data, dummy_row_df], ignore_index=True)

        fig_map = px.choropleth_mapbox(case_data,
                                geojson=case_data.geometry,
                                locations=case_data.index,
                                title="Sample areas of on map",
                                color='cf_class',
                                hover_name='clusterID',
                                hover_data = ['area'],
                                color_discrete_map=labels_colors,
                                category_orders={"cf_class":['Top','High','Low','Bottom']},
                                labels={'cf_class': f'{cf_col} level'},
                                center={"lat": lat, "lon": lon},
                                mapbox_style=my_style,
                                zoom=11,
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

    #scat plotter
    def carbon_vs_pois_scatter(case_data_in,
                            hovername=None,
                            cf_col=None,
                            x_col=None,
                            y_col=None,
                            z_col=None,
                            scale_axis=True,
                            trendline=False,
                            bins=None,
                            title=None):

        #make a copy
        case_data = case_data_in.copy()

        # Get unique sorted values
        sorted_unique_values = sorted(case_data[cf_col].unique())

        # Determine number of bins
        num_bins = min(len(sorted_unique_values), 4)  # Limit the number of bins to 4

        # Define fixed labels and colors
        labels_colors = {
            'Bottom': 'rgba(35,140,35, 1)',
            'Low': 'rgba(64,224,208, 0.8)',
            'High': 'rgba(254, 170, 70, 0.8)',
            'Top': 'rgba(255, 87, 51, 1)'
        }
        
        if bins is None:
            # Generate bin edges ensuring they are unique and cover the range of values
            bin_edges = np.percentile(case_data[cf_col], np.linspace(0, 100, num_bins + 1))
            # Adjust the number of labels based on the number of bin edges
            num_labels = len(bin_edges) - 1
            labels = list(labels_colors.keys())[:num_labels]
            # Assign quartile class
            case_data['cf_class'] = pd.cut(case_data[cf_col], bins=bin_edges, labels=labels, include_lowest=True)
        else:
            bin_edges = bins
            num_labels = len(bin_edges) - 1
            labels = list(labels_colors.keys())[:num_labels]
            case_data['cf_class'] = pd.cut(case_data[cf_col], bins=bin_edges, labels=labels, include_lowest=True)

        #ensure all classes exist
        for label in labels_colors.keys():
            # Check if the label is already present
            if label not in case_data['cf_class'].tolist():
                # Create a dummy row DataFrame with the missing category
                dummy_row_df = pd.DataFrame([{cf_col: None, 'cf_class': label, x_col: None, y_col: None, z_col: 0}])
                # Use pd.concat to append the dummy row DataFrame
                case_data = pd.concat([case_data, dummy_row_df], ignore_index=True)

        # Filtering out nan values and ensuring only valid labels are included
        unique_labels = [label for label in case_data['cf_class'].dropna().unique() if label in labels_colors]

        # Dynamically create colormap for quartile classes used in cf_class
        quartile_colormap = {label: labels_colors[label] for label in unique_labels}

        #cat orders
        desired_order = ["Top", "High", "Low", "Bottom"]
        present_categories = case_data['cf_class'].tolist()
        adjusted_category_orders = [cat for cat in desired_order if cat in present_categories]

        # Create a new column for custom hover text
        case_data['custom_hover_text'] = case_data.apply(lambda row: f"footprint {row[cf_col]}", axis=1)

        if scale_axis:
            buff = 1.1
            # Calculate 99th quantile and max values for x and y columns
            x_quantile = case_data[x_col].quantile(0.95)
            y_quantile = case_data[y_col].quantile(0.95)
            x_max = case_data[x_col].max()
            y_max = case_data[y_col].max()

            # Determine if the difference between max and 99th quantile is large for x and y
            x_large_diff = (x_max - x_quantile) > (x_quantile * 0.3) # threshold n% of the 95th quantile
            y_large_diff = (y_max - y_quantile) > (y_quantile * 0.3)

            # Set axis range based on the above logic
            x_range = [-5, (x_quantile if x_large_diff else x_max) * buff]
            y_range = [0, (y_quantile if y_large_diff else y_max) * buff]
        else:
            x_range = None
            y_range = None

        #scale marker sizes
        min_size=50
        max_size=500
        # Initialize the scaler with the desired min and max sizes
        scaler = MinMaxScaler(feature_range=(min_size, max_size))
        # Fit the scaler to your data and transform the z_col to the scaled sizes
        # Reshape(-1, 1) is needed because the data needs to be in a 2D array format
        case_data['adjusted_size'] = scaler.fit_transform(case_data[z_col].values.reshape(-1, 1)).flatten()
        
        #hovers
        hover_data = {column: True for column in [z_col,'cf_class','Income Level decile']}
        # Now, set all other columns to False
        for column in case_data.columns:
            if column not in hover_data:
                hover_data[column] = False

        #trendline
        if trendline:
            trend_line="ols"
        else:
            trend_line=None

        # Create the scatter plot
        fig = px.scatter(case_data, title=title,
                            x=x_col, y=y_col, color='cf_class', size='adjusted_size',
                            log_y=False,
                            hover_name=hovername,
                            hover_data = hover_data,
                            labels={'cf_class': f'{cf_col} level'},
                            category_orders={"cf_class": adjusted_category_orders},
                            color_discrete_map=quartile_colormap,
                            trendline=trend_line,
                            range_x=x_range,
                            range_y=y_range
                            )
            
        fig.update_layout(
            margin={"r": 10, "t": 50, "l": 10, "b": 10}, height=700,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )

        return fig




    #----------------------------------- MAIN ----------------------------------------
    
    #area calc
    def calc_area(df_in):
        df = df_in.copy()
        df['geometry'] = df.wkt.apply(wkt.loads)
        gdf = gpd.GeoDataFrame(df,geometry='geometry',crs=4326)
        #utm..
        utm = gdf.estimate_utm_crs()
        df_in['area'] = round(gdf.to_crs(utm).area,-3)
        orig_cols = list(df_in.columns)
        orig_cols.insert(2, orig_cols.pop(orig_cols.index('area')))
        df_out = df_in[orig_cols]
        return df_out
    
    # Shannon index for a row
    def shannon_index(row):
        # Filter out zero values to avoid log(0)
        filtered_row = row[row > 0]
        total = filtered_row.sum()
        proportions = filtered_row / total
        shannon_index = -np.sum(proportions * np.log(proportions))
        return shannon_index
    
    @st.cache_data(max_entries=1)
    def get_sample(selected_urb_file,filtered_names):
        if selected_urb_file != "All_samples":
            cfua_data = spaces_csv_handler(file_name=f"ndp/cfua/{selected_urb_file}.csv",operation="download")
            cfua_data = calc_area(cfua_data)
            scale_axis = False
        else: #get all city samples and concat
            cfua_data = pd.DataFrame()
            for name in filtered_names:
                sample_df = spaces_csv_handler(file_name=f"ndp/cfua/{name}.csv",operation="download")
                sample_df = calc_area(sample_df)
                cfua_data = pd.concat([cfua_data,sample_df])
                del sample_df
            scale_axis = True
        #change e to FAR and gfa to GFA if needed..
        if 'e' in cfua_data.columns:
            cfua_data.rename(columns={'e': 'FAR', 'gfa': 'GFA', 'numfloors': 'Floors'}, inplace=True)
        
        return cfua_data, scale_axis
    
    
    # ------- functions ----------
        
    def diversity_index(df_in,use_cols):
        df = df_in.copy()
        df['shannon_index'] = round(df[use_cols].apply(shannon_index, axis=1),3)
        min_shannon = df['shannon_index'].min()
        max_shannon = df['shannon_index'].max()
        df_in['mixed-use'] = round(((df['shannon_index'] - min_shannon) / (max_shannon - min_shannon)) * 100,2)
        orig_cols = list(df_in.columns)
        orig_cols.insert(9, orig_cols.pop(orig_cols.index('mixed-use')))
        df_out = df_in[orig_cols]
        del df_in
        del df
        return df_out

    def residential_vol(df_in,land_use_cols = ['residential-all','commercial','public']):
        df = df_in.copy()
        df['all_buildings'] = df['residential-all'] + df['commercial'] + df['public']
        df['residential_large_share'] = round(df['residential-large'] / df['all_buildings'],3)
        min_share  =df['residential_large_share'].min()
        max_share = df['residential_large_share'].max()
        df_in['high-res-index'] = round(((df['residential_large_share'] - min_share) / (max_share - min_share)) * 100,2)
        orig_cols = list(df_in.columns)
        orig_cols.insert(9, orig_cols.pop(orig_cols.index('high-res-index')))
        df_out = df_in[orig_cols]
        del df_in
        del df
        return df_out
    
    def amenity_densities(df_in):
        df = df_in.copy()
        
        df['Consumer_urb_index'] = round(df['consumer_urbanism'] / df['GFA'],3)
        min_con = df['Consumer_urb_index'].min()
        max_con = df['Consumer_urb_index'].max()
        df_in['Consumer_urb_index'] = round(((df['Consumer_urb_index'] - min_con) / (max_con - min_con)) * 100,2)
        
        df['Time_spending_index'] = round(df['time_spending'] / df['GFA'],3)
        min_tim = df['Time_spending_index'].min()
        max_tim = df['Time_spending_index'].max()
        df_in['Time_spending_index'] = round(((df['Time_spending_index'] - min_tim) / (max_tim - min_tim)) * 100,2)
        
        orig_cols = list(df_in.columns)
        orig_cols.insert(12, orig_cols.pop(orig_cols.index('Consumer_urb_index')))
        orig_cols.insert(12, orig_cols.pop(orig_cols.index('Time_spending_index')))
        df_out = df_in[orig_cols]
        del df_in
        del df
        return df_out
    
    #prepare data func to be used also in tab3..
    def prepare_data(selected_urb_file):
        cfua_data, scale_axis = get_sample(selected_urb_file=selected_urb_file,filtered_names=filtered_names)
        cfua_data['residential-all'] = cfua_data['residential-small'] + cfua_data['residential-large']
        land_use_cols = ['residential-all','commercial','public']
        cfua_data = diversity_index(cfua_data,use_cols=land_use_cols)
        cfua_data = residential_vol(cfua_data,land_use_cols=land_use_cols)
        cfua_data = amenity_densities(cfua_data)
        return cfua_data, scale_axis
    
    cfua_df = None
    if selected_urb_file != "...":

        cfua_data, scale_axis = prepare_data(selected_urb_file)
        
        dropcols = ['city','wkt','other','miscellaneous','residential-all']
        cfua_df = cfua_data.drop(columns=dropcols)
        #st.data_editor(cfua_df)
        #st.stop()
        
        # ------- cols for features --------
        density_cols = cfua_df.drop(columns='clusterID').columns.tolist()[1:4]
        land_use_cols = cfua_df.drop(columns='clusterID').columns.tolist()[7:9] + ['Consumer_urb_index','Time_spending_index']
        cf_cols = cfua_df.drop(columns='clusterID').columns.tolist()[17:]
        classification_sets = ['Carbon footprint','Consumer_urb_index','Time_spending_index']

        c1,c2,c3,c4 = st.columns(4)
        yax = c1.selectbox('Density metric (y)',density_cols,index=2)
        xax = c2.selectbox('Land-use index (x)',land_use_cols,index=1)
        size = c3.selectbox('Carbon footprint (size)',cf_cols,index=0)
        classify = c4.selectbox('Classification (color)',classification_sets,index=0)
        
        #classify
        if classify == 'Carbon footprint':
            classifier = size
        else:
            classifier = classify
        
        if yax != xax:
            
            if selected_urb_file == "All_samples":
                #drop large areas
                cfua_df = cfua_df[cfua_df['area'] < 4000000] # > 4km2
                
            #st.data_editor(cfua_df)
            if selected_urb_file != "All_samples":
                custom_bins = None
            else:
                custom_bins = np.percentile(cfua_df[classifier].dropna(), [0, 25, 50, 75, 90])
            
            scat_plot = carbon_vs_pois_scatter(cfua_df,
                                hovername='clusterID',
                                cf_col=classifier,
                                x_col=xax,
                                y_col=yax,
                                z_col=size,
                                scale_axis=scale_axis,
                                trendline=False,
                                bins=custom_bins,
                                title=f"CFUA scatter, Sample N = {len(cfua_df)}")
            
            st.plotly_chart(scat_plot, use_container_width=True, config = {'displayModeBar': False} )
            
            if selected_urb_file != "All_samples":
                with st.expander('Cluster on map'):
                    map_plot = plot_sample_areas(cfua_data,cf_col=classifier)
                    st.plotly_chart(map_plot, use_container_width=True, config = {'displayModeBar': False} )
                    st.data_editor(cfua_df.drop(columns=cfua_df.columns[-1],  axis=1))
                
        with st.expander("Correlation matrix",expanded=True):
            colorscale_cont = [
                    [0.0, 'cornflowerblue'],
                    [0.8/3.0, 'skyblue'],
                    [0.8/2.0, 'orange'],
                    [1.0, 'darkred']
                ]
            colorscale_dicr = [
                [0.0, 'cornflowerblue'], [0.33, 'cornflowerblue'],
                [0.33, 'skyblue'], [0.5, 'skyblue'],
                [0.5, 'orange'], [0.66, 'orange'],
                [0.66, 'darkred'], [1.0, 'darkred']
            ]

            def single_corr_matrix(df, color_scale, sample_name, control_var=None):
                # Validate control_var
                if control_var and control_var not in df.columns:
                    st.warning("Control variable {control_var} not found in DataFrame columns.")
                    raise ValueError(f"Control variable {control_var} not found in DataFrame columns.")
                
                # Compute partial correlations if control_var is specified
                if control_var:
                    # Initialize an empty DataFrame for partial correlations
                    partial_corrs = df.corr()  # Starting with full correlation as a template for indices
                    corr_result_df = pd.DataFrame() #df to concat p-value results
                    for col1 in df.columns:
                        for col2 in df.columns:
                            if col1 != col2:
                                # Compute partial correlation, excluding control_var from analysis if it matches either column
                                if col1 == control_var or col2 == control_var:
                                    partial_corrs.loc[col1, col2] = df[[col1, col2]].corr().iloc[0, 1]
                                else:
                                    partial_corr_result_df = pg.partial_corr(data=df, x=col1, y=col2, covar=control_var)
                                    partial_corrs.loc[col1, col2] = partial_corr_result_df['r'].values[0]
                                    corr_result_df = pd.concat([corr_result_df,partial_corr_result_df])
                                    
                            else:
                                # Set diagonal to 1 for self-correlation
                                partial_corrs.loc[col1, col2] = 1.0

                    # Remove the control_var from the correlation matrix
                    partial_corrs = partial_corrs.drop(index=control_var, columns=control_var, errors='ignore')
                    corr = partial_corrs
                    control_var_text = "income level controlled"
                
                else:
                    # Compute regular correlation if no control_var specified
                    corr = df.corr()
                    control_var_text = "income level not controlled"
                    corr_result_df = None
                
                trace = go.Heatmap(z=corr.values,
                                x=corr.index.values,
                                y=corr.columns.values,
                                colorscale=color_scale)
                fig = go.Figure()
                fig.add_trace(trace)
                fig.update_layout(margin={"r": 10, "t": 50, "l": 10, "b": 10}, height=800, title_text=f"{sample_name}, {control_var_text}")
                return fig, corr_result_df

                
            def facet_corr_matrix_by_city(df, color_scale, control_var=None):
                unique_cities = df['city'].unique()
                n = len(unique_cities)
                
                # Create subplot layout: adjust rows and cols as per your preference
                rows = int(n**0.5) + (1 if n % int(n**0.5) > 0 else 0)
                cols = int(n / rows) + (n % rows > 0)
                
                fig = make_subplots(rows=rows, cols=cols, subplot_titles=unique_cities)
                
                for i, city in enumerate(unique_cities, start=1):
                    city_df = df[df['city'] == city]
                    heatmap_fig, corr_df = single_corr_matrix(city_df.drop(columns=['city','clusterID','wkt']), color_scale, city, control_var)
                    del corr_df
                    # For each subplot, add the heatmap. Note: we need to extract the trace from heatmap_fig
                    for trace in heatmap_fig.data:
                        trace.showscale = False
                        fig.add_trace(trace, row=(i-1)//cols + 1, col=(i-1) % cols + 1)
                
                fig.update_layout(height=800 * rows, width=500 * cols, showlegend=False,
                                title_text="Correlation Heatmaps by City, income level not controlled")
                return fig
        
            if selected_urb_file == "All_samples":
                corvar = st.toggle('Income level controlled')
                if corvar:
                    fig, corr_df = single_corr_matrix(cfua_df.drop(columns=['clusterID']),color_scale=colorscale_dicr,sample_name=selected_urb_file,
                                            control_var='Income Level decile')
                    st.plotly_chart(fig, use_container_width=True, config = {'displayModeBar': False} )
                    st.markdown('**Partial correlation results**')
                    st.data_editor(corr_df.describe())
                else:
                    fig, corr_df = single_corr_matrix(cfua_df.drop(columns=['clusterID']),color_scale=colorscale_dicr,sample_name=selected_urb_file,
                                            control_var=None)
                    st.plotly_chart(fig, use_container_width=True, config = {'displayModeBar': False} )
                

                #..and by city
                fig = facet_corr_matrix_by_city(cfua_data,color_scale=colorscale_dicr,control_var=None) #not big enough sample
                st.plotly_chart(fig, use_container_width=True, config = {'displayModeBar': False} )
                
            else:
                fig, corr_df = single_corr_matrix(cfua_df.drop(columns=['clusterID']),color_scale=colorscale_dicr,sample_name=selected_urb_file)
                st.plotly_chart(fig, use_container_width=True, config = {'displayModeBar': False} )
                

#classifier
with tab3:
        
    # ML classifier
    def classifier(df_in,k=4):
        feats = ['Floors','FAR','high-res-index','mixed-use','Time_spending_index','Consumer_urb_index']
        df = df_in[feats].copy()
        from sklearn.cluster import KMeans
        from sklearn.tree import DecisionTreeClassifier, plot_tree
        import matplotlib.pyplot as plt
        # Applying K-Means clustering
        kmeans = KMeans(n_clusters=k, random_state=42)
        df['cluster'] = kmeans.fit_predict(df)
        # Preparing the data
        X = df.drop('cluster', axis=1)  # Features
        y = df['cluster']  # Cluster labels as target
        # Training a decision tree
        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(X, y)
        # Plotting the decision tree
        plt.figure(figsize=(20,10))
        plot_tree(clf, filled=True, feature_names=X.columns, class_names=True)
        mytree = plt.show()
        return st.pyplot(mytree)
    
    selected_urb_file2 = st.selectbox('Select sample to classify',selectbox_names)
    if selected_urb_file2 != "...":
        cfua_data, scalefix_notused = prepare_data(selected_urb_file2)
        
        with st.form('Urban types'):
            st.subheader('Define metrics for urban types')
            #classifier form
            c0,c1,c2,c3 = st.columns([2,3,3,3])
            c0.subheader("'Frank'") #LL
            LL_far = c1.slider('FAR',0.0,2.0,[0.0,0.2],0.1,key='LL_far')
            LL_floors = c2.slider('Floors',1,9,[1,2],1,key='LL_floors')
            LL_urb = c3.slider('Urbanism',0,100,[1,100],10,key='LL_urb')
            c0.markdown("###")
            c0.markdown("###")
            c0.subheader("'Jacobs'") #LH
            LH_far = c1.slider('FAR',0.0,2.0,[0.3,1.0],0.1,key='LH_far')
            LH_floors = c2.slider('Floors',1,9,[1,3],1,key='LH_floors')
            LH_urb = c3.slider('Urbanism',0,100,[0,50],10,key='LH_urb')
            c0.markdown("###")
            c0.markdown("###")
            c0.subheader("'Corbu'") #HL
            HL_far = c1.slider('FAR',0.0,2.0,[0.8,2.0],0.1,key='HL_far')
            HL_floors = c2.slider('Floors',1,9,[5,9],1,key='HL_floors')
            HL_urb = c3.slider('Urbanism',0,100,[0,50],10,key='HL_urb')
            c0.markdown("###")
            c0.markdown("###")
            c0.subheader("'Cerda'") #HH
            HH_far = c1.slider('FAR',0.0,2.0,[0.5,2.0],0.1,key='HH_far')
            HH_floors = c2.slider('Floors',1,9,[4,9],1,key='HH_floors')
            HH_urb = c3.slider('Urbanism',0,100,[10,100],10,key='HH_urb')
            c0.markdown("###")
            c0.markdown("###")
            
            update = st.form_submit_button('Apply')
            #st.markdown("---")
        
        #manual classifier
        classification_rules = {
            # low-low = LL
            'Frank': [
                {'F1': 'FAR', 'min': LL_far[0], 'max': LL_far[1],
                 'F2': 'Floors', 'min2': LL_floors[0], 'max2': LL_floors[1]}
            ],
            # low-high = LH
            'Jacobs': [
                {'F1': 'FAR', 'min': LH_far[0], 'max': LH_far[1],
                 'F2': 'Floors', 'min2': LH_floors[0], 'max2': LH_floors[1],
                 'F3': 'Consumer_urb_index', 'min3': LH_urb[0], 'max3': LH_urb[1]}
            ],
            # high-low = HL
            'Corbu': [
                {'F1': 'FAR', 'min': HL_far[0], 'max': HL_far[1],
                 'F2': 'Floors', 'min2': HL_floors[0], 'max2': HL_floors[1],
                 'F3': 'Consumer_urb_index', 'min3': HL_urb[0], 'max3': HL_urb[1]}
            ],
            # high-high = HH
            'Cerda': [
                {'F1': 'FAR', 'min': HH_far[0], 'max': HH_far[1],
                 'F2': 'Floors', 'min2': HH_floors[0], 'max2': HH_floors[1],
                 'F3': 'Consumer_urb_index', 'min3': HH_urb[0], 'max3': HH_urb[1]}
            ]
        }
        
        def classify_combined(row, rules):
            for class_name, conditions in rules.items():
                for condition in conditions:
                    if condition['min'] <= row[condition['F1']] < condition['max'] and \
                    condition['min2'] <= row[condition['F2']] < condition['max2'] and \
                    ('F3' not in condition or (condition['min3'] <= row[condition['F3']] < condition['max3'])) and \
                    ('F4' not in condition or (condition['min4'] <= row[condition['F4']] < condition['max4'])):
                        return class_name
            return 'Uncat'
        
        def classify_sequentially(df, classification_rules):
            # Initialize a column for classification results if not already present
            if 'Urban_class' not in df.columns:
                df['Urban_class'] = None

            for class_name, conditions in classification_rules.items():
                for condition in conditions:
                    # Constructing the condition for current class
                    # Start with rows not yet classified
                    current_condition = df['Urban_class'].isnull()

                    # Check each specified feature condition
                    current_condition &= df[condition['F1']].between(condition['min'], condition['max'], inclusive='left')
                    current_condition &= df[condition['F2']].between(condition['min2'], condition['max2'], inclusive='left')
                    if 'F3' in condition and 'min3' in condition and 'max3' in condition:
                        current_condition &= df[condition['F3']].between(condition['min3'], condition['max3'], inclusive='left')
                    
                    # Update classification for rows that meet the current condition
                    df.loc[current_condition, 'Urban_class'] = class_name
            df.loc[df['Urban_class'].isna(),'Urban_class'] = "Uncat"
            return df

        if update:
            #drop large areas
            df_classified = cfua_data.copy()

            #df_classified['Urban_class'] =df_classified.apply(classify_combined, rules=classification_rules, axis=1)
            df_classified = classify_sequentially(df_classified,classification_rules)
            df_classified.loc[(df_classified['Urban_class'] == "Uncat") & 
                              (df_classified['mixed-use'] > 30) &
                              (df_classified['Consumer_urb_index'] < 30),
                               'Urban_class'] = "Mixed"
            df_classified.loc[(df_classified['area'] > 4000000) &
                              (df_classified['Consumer_urb_index'] > 50),
                              'Urban_class'] = "Glaeser"
            
            feats = ['FAR','Floors','mixed-use','Consumer_urb_index']
            
            urb_type_colors = {
                'Glaeser':'cornflowerblue',
                'Cerda':'orange',
                'Corbu':'Grey',
                'Jacobs':'brown',
                'Frank':'burlywood',
                'Mixed':'violet',
                'Uncat':'white'
            }
            def check_plot(df):
                df['geometry'] = df.wkt.apply(wkt.loads)
                gdf = gpd.GeoDataFrame(df,geometry='geometry',crs=4326)
                lat = gdf.unary_union.centroid.y
                lon = gdf.unary_union.centroid.x
                check_fig = px.choropleth_mapbox(gdf,
                                geojson=gdf.geometry,
                                locations=gdf.index,
                                title="Classified areas",
                                color='Urban_class',
                                hover_name='clusterID',
                                hover_data = feats,
                                color_discrete_map=urb_type_colors,
                                #category_orders={"cf_class":['Top','High','Low','Bottom']},
                                #labels={'cf_class': f'{cf_col} level'},
                                center={"lat": lat, "lon": lon},
                                mapbox_style=my_style,
                                zoom=10,
                                opacity=0.5,
                                width=900,
                                height=900
                                )
                check_fig.update_layout(margin={"r": 10, "t": 50, "l": 10, "b": 10}, height=700,
                                    legend=dict(
                                        yanchor="top",
                                        y=0.97,
                                        xanchor="left",
                                        x=0.02
                                    )
                                    )
                return check_fig
            
            st.plotly_chart(check_plot(df_classified), use_container_width=True, config = {'displayModeBar': False} )
        
            #distribution
            fig_bar = px.bar(df_classified, x='GFA', y='Urban_class', color='Urban_class',
                             orientation='h', title='Volume distribution')
            fig_bar.update_layout(showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True, config = {'displayModeBar': False} )
        

#footer
st.markdown('---')
footer_title = '''
**NDP Project**
[![MIT license](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/teemuja/NDP/blob/main/LICENSE) 
'''
st.markdown(footer_title)
disclamer = 'Data papers are constant work in progress and will be upgraded, changed & fixed while research go on.'
st.caption('Disclaimer: ' + disclamer)