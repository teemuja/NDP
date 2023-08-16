#sidebar logo
import streamlit as st
from pathlib import Path
import os
import base64

def add_ndp_logo_with_link():
    
    def get_base64_of_bin_file(bin_file):
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()

    def get_img_with_href(local_img_path, target_url):
        img_format = os.path.splitext(local_img_path)[-1].replace('.', '')
        bin_str = get_base64_of_bin_file(local_img_path)
        html_code = f'''
            <a href="{target_url}">
                <img src="data:image/{img_format};base64,{bin_str}" />
            </a>'''
        return html_code
    gif_html = get_img_with_href('logo/NDP_180.png', '#')
    
    return st.markdown(gif_html, unsafe_allow_html=True)

def add_bg_from_local(image_path):
    image_file = Path(__file__).parent / f'{image_path}'
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: tile
    }}
    </style>
    """,
    unsafe_allow_html=True
    )