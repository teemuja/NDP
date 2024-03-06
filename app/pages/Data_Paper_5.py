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


# Shannon index for a row
def shannon_index(row):
    # Filter out zero values to avoid log(0)
    filtered_row = row[row > 0]
    total = filtered_row.sum()
    proportions = filtered_row / total
    shannon_index = -np.sum(proportions * np.log(proportions))
    return shannon_index

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
                           title=None):

    #make a copy
    case_data = case_data_in.copy()

    # Get unique sorted values
    sorted_unique_values = sorted(case_data[cf_col].unique())

    # Determine number of bins
    num_bins = min(len(sorted_unique_values), 4)  # Limit the number of bins to 4

    # Generate bin edges ensuring they are unique and cover the range of values
    bin_edges = np.percentile(case_data[cf_col], np.linspace(0, 100, num_bins + 1))

    # Ensure unique bin edges
    bin_edges = np.unique(bin_edges)

    # Define fixed labels and colors
    labels_colors = {
        'Bottom': 'rgba(35,140,35, 1)',
        'Low': 'rgba(254, 220, 120, 0.8)',
        'High': 'rgba(254, 170, 70, 0.8)',
        'Top': 'rgba(255, 87, 51, 1)'
    }
    
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
            dummy_row_df = pd.DataFrame([{cf_col: None, 'cf_class': label, x_col: None, y_col: None, z_col: 0}])
            # Use pd.concat to append the dummy row DataFrame
            case_data = pd.concat([case_data, dummy_row_df], ignore_index=True)

    # Dynamically create colormap for quartile classes used in cf_class
    unique_labels = case_data['cf_class'].tolist()
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
    hover_data = {column: True for column in [z_col,'Income Level decile']}
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
                         labels={'cf_class': f'{cf_col} level','mixed-use':'mixed-use index'},
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
if selected_urb_file != "...":
    if selected_urb_file != "All_samples":
        cfua_data = spaces_csv_handler(file_name=f"ndp/cfua/{selected_urb_file}.csv",operation="download")
        scale_axis = False
    else: #get all city samples and concat
        cfua_data = pd.DataFrame()
        for name in filtered_names:
            sample_df = spaces_csv_handler(file_name=f"ndp/cfua/{name}.csv",operation="download")
            cfua_data = pd.concat([cfua_data,sample_df])
        scale_axis = True

    #change e to FAR and gfa to GFA if needed..
    if 'e' in cfua_data.columns:
        cfua_data.rename(columns={'e': 'FAR', 'gfa': 'GFA', 'numfloors': 'Floors'}, inplace=True)

    #add shannon index as mixed land use indicator
    cfua_data['residential-all'] = cfua_data['residential-small'] + cfua_data['residential-large']
    land_use_cols = ['residential-all','commercial','public']
    def diversity_index(df_in,use_cols):
        df = df_in.copy()
        df['shannon_index'] = round(df[use_cols].apply(shannon_index, axis=1),3)
        min_shannon = df['shannon_index'].min()
        max_shannon = df['shannon_index'].max()
        df_in['mixed-use'] = round(((df['shannon_index'] - min_shannon) / (max_shannon - min_shannon)) * 100,2)
        orig_cols = list(df_in.columns)
        orig_cols.insert(9, orig_cols.pop(orig_cols.index('mixed-use')))
        df_out = df_in[orig_cols]
        return df_out
    cfua_data = diversity_index(cfua_data,use_cols=land_use_cols)

    dropcols = ['city','wkt','other','miscellaneous','residential-all']
    cfua_df = cfua_data.drop(columns=dropcols)

    #cols for features
    density_cols = cfua_df.drop(columns='clusterID').columns.tolist()[:3]
    land_use_cols = cfua_df.drop(columns='clusterID').columns.tolist()[3:8]
    amenity_cols = cfua_df.drop(columns='clusterID').columns.tolist()[8:11]
    cf_cols = cfua_df.drop(columns='clusterID').columns.tolist()[13:]

    c1,c2,c3,c4 = st.columns(4)
    yax = c1.selectbox('Density metric (y)',density_cols,index=2)
    xax = c2.selectbox('Land-use index (x)',land_use_cols,index=4)
    size = c3.selectbox('Amenity count (size)',amenity_cols,index=0)
    cf = c4.selectbox('CF-class (color)',cf_cols,index=0)
    
    if yax != xax:
        if selected_urb_file == "All_samples":
            use_trendline = True
        else:
            use_trendline = False

        scat_plot = carbon_vs_pois_scatter(cfua_df,
                            hovername='clusterID',
                            cf_col=cf,
                            x_col=xax,
                            y_col=yax,
                            z_col=size,
                            scale_axis=scale_axis,
                            trendline=use_trendline,
                            title="CFUA scatter")
        
        st.plotly_chart(scat_plot, use_container_width=True, config = {'displayModeBar': False} )
        
        if selected_urb_file != "All_samples":
            with st.expander('Cluster on map'):
                map_plot = plot_sample_areas(cfua_data,cf_col=cf)
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
            

#footer
st.markdown('---')
footer_title = '''
**Naked Density Project**
[![MIT license](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/teemuja/NDP/blob/main/LICENSE) 
'''
st.markdown(footer_title)
disclamer = 'Data papers are constant work in progress and will be upgraded, changed & fixed while research go on.'
st.caption('Disclaimer: ' + disclamer)