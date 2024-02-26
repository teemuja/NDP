# NDP app always beta a lot
import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import boto3
import re
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots


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
        NDP data paper #5 V1.1\
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
st.title("Data Paper #5")
st.subheader("CFUA - Climate Friendly Urban Architecture")
st.markdown('This data paper is seeking urban features which correlate with low carbon lifestyles.')
st.caption("Authors: Teemu Jama, Jukka Heinonen @[Háskóli Íslands](https://uni.hi.is/heinonen/), Henrikki Tenkanen @[Aalto GIST Lab](https://gistlab.science)")
st.markdown("###")

def check_password():
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if (
            st.session_state["username"] in st.secrets["passwords"]
            and st.session_state["password"]
            == st.secrets["passwords"][st.session_state["username"]]
        ):
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False
    if "password_correct" not in st.session_state:
        # First run, show inputs for username + password.
        st.text_input(label="user name", on_change=password_entered, key="username")
        st.text_input(label="password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(label="username", on_change=password_entered, key="username")
        st.text_input(label="password", type="password", on_change=password_entered, key="password")
        st.error("incorrect")
        return False
    else:
        # Password correct.
        return True

#data handler
@st.cache_data(max_entries=1)
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
        df = pd.read_csv(obj['Body'])
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



#selectors
csv_list = spaces_csv_handler(operation="list",folder_name="ndp/cfua")

names = []
for file_name in csv_list:
    file_name_with_extension = file_name.split('/')[-1]
    name = file_name_with_extension.split('.')[0]
    names.append(name)
pattern = "CFUADATA" #r'_sample_N\d+$'
filtered_names = [name for name in names if re.search(pattern, name)]
selectbox_names = filtered_names.copy()
selectbox_names.insert(0,"...")
selectbox_names.append("All_samples")
selected_urb_file = st.selectbox('Select sample file',selectbox_names)

#map plotter
from shapely import wkt
def plot_sample_areas(df,cf_col="Total footprint"):
    df['geometry'] = df.wkt.apply(wkt.loads)
    case_data = gpd.GeoDataFrame(df,geometry='geometry')
    lat = case_data.unary_union.centroid.y
    lon = case_data.unary_union.centroid.x
    #
    # Define fixed labels and colors
    labels_colors = {
        'Bottom': 'rgba(144, 238, 144, 0.6)',
        'Low': 'rgba(254, 220, 120, 0.8)',
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

    fig_map = px.choropleth_mapbox(case_data,
                            geojson=case_data.geometry,
                            locations=case_data.index,
                            title="Sample areas of on map",
                            color='cf_class',
                            hover_name='clusterID',
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
def carbon_vs_pois_scatter(case_data,
                           hovername=None,
                           cf_col=None,
                           x_col=None,
                           y_col=None,
                           z_col=None,
                           title=None):

    # Define fixed labels and colors
    labels_colors = {
        'Bottom': 'rgba(144, 238, 144, 0.6)',
        'Low': 'rgba(254, 220, 120, 0.8)',
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

    # Dynamically create colormap for quartile classes used in cf_class
    unique_labels = case_data['cf_class'].cat.categories
    quartile_colormap = {label: labels_colors[label] for label in unique_labels}

    # Create a new column for custom hover text
    case_data['custom_hover_text'] = case_data.apply(lambda row: f"footprint {row[cf_col]}", axis=1)

    # Calculate 99th quantile and max values for x and y columns
    x_99_quantile = case_data[x_col].quantile(0.90)
    y_99_quantile = case_data[y_col].quantile(0.90)
    x_max = case_data[x_col].max()
    y_max = case_data[y_col].max()

    # Determine if the difference between max and 99th quantile is large for x and y
    x_large_diff = (x_max - x_99_quantile) > (x_99_quantile * 0.1) # threshold 10% of the 99th quantile
    y_large_diff = (y_max - y_99_quantile) > (y_99_quantile * 0.1)

    # Set axis range based on the above logic
    x_range = [0, x_99_quantile if x_large_diff else x_max]
    y_range = [0, y_99_quantile if y_large_diff else y_max]

    # Create the scatter plot
    fig = px.scatter(case_data, title=title,
                         x=x_col, y=y_col, color='cf_class', size=z_col*2, #size scaled
                         log_y=False,
                         hover_name=hovername,
                         labels={'cf_class': f'{cf_col} level'},
                         color_discrete_map=quartile_colormap,
                         range_x=x_range,
                         range_y=y_range
                         )
        
    fig.update_layout(
        margin={"r": 10, "t": 50, "l": 10, "b": 10}, height=700,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    return fig

#plot
if selected_urb_file != "...":
    if selected_urb_file != "All_samples":
        cfua_data = spaces_csv_handler(file_name=f"ndp/cfua/{selected_urb_file}.csv",operation="download")
    else:
        cfua_data = pd.DataFrame()
        for name in filtered_names:
            sample_df = spaces_csv_handler(file_name=f"ndp/cfua/{name}.csv",operation="download")
            cfua_data = pd.concat([cfua_data,sample_df])
      
    cfua_plot = cfua_data.drop(columns=['city','wkt'])
    dropcols = ['city','clusterID','wkt']
    cfua_df = cfua_data.drop(columns=dropcols)
    corr = cfua_df.corr()
    
    c1,c2,c3,c4 = st.columns(4)
    
    yax = c1.selectbox('Density (y)',cfua_df.columns.tolist()[:3],index=2)
    xax = c2.selectbox('Building types (x)',cfua_df.columns.tolist()[3:9],index=3)
    size = c3.selectbox('Amenities (size)',cfua_df.columns.tolist()[9:16],index=0)
    cf = c4.selectbox('CF (color)',cfua_df.columns.tolist()[18:],index=1)
    
    
    if yax != xax:
        try:
            scat_plot = carbon_vs_pois_scatter(cfua_plot,
                                hovername='clusterID',
                                cf_col=cf,
                                x_col=xax,
                                y_col=yax,
                                z_col=size,
                                title="CFUA scatter")
            st.plotly_chart(scat_plot, use_container_width=True, config = {'displayModeBar': False} )
            
        except:
            st.warning('Cannot create scatter')
        
        if selected_urb_file != "All_samples":
            with st.expander('Cluster on map'):
                map_plot = plot_sample_areas(cfua_data,cf_col=cf)
                st.plotly_chart(map_plot, use_container_width=True, config = {'displayModeBar': False} )
                st.data_editor(cfua_df.drop(columns=cfua_df.columns[-1],  axis=1))
            
    with st.expander("Correlation matrix",expanded=True):
        colorscale = [
            [0.0, 'cornflowerblue'],
            [0.8/3.0, 'skyblue'],
            [0.8/2.0, 'orange'],
            [1.0, 'darkred']
        ]
        trace = go.Heatmap(z=corr.values,
                        x=corr.index.values,
                        y=corr.columns.values,
                        colorscale=colorscale)

        fig = go.Figure()
        fig.add_trace(trace)
        fig.update_layout(margin={"r": 10, "t": 50, "l": 10, "b": 10}, height=700)
        st.plotly_chart(fig, use_container_width=True, config = {'displayModeBar': False} )

#footer
st.markdown('---')
footer_title = '''
**Naked Density Project**
[![MIT license](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/teemuja/NDP/blob/main/LICENSE) 
'''
st.markdown(footer_title)
disclamer = 'Data papers are constant work in progress and will be upgraded, changed & fixed while research go on.'
st.caption('Disclaimer: ' + disclamer)