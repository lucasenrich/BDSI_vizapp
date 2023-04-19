import sys
import streamlit as st
import folium
from folium import raster_layers
from folium.plugins import HeatMap
import shutil
from folium import Choropleth, Circle, Marker
from streamlit_folium import folium_static
import geopandas as gpd
import ee
import os
import numpy as np
from rasterio.windows import Window
from rasterio.merge import merge
from urllib import request
from zipfile import *
from osgeo import gdal
from unetseg.predict import PredictConfig, predict
from shapely.geometry import box
import glob
import operator
import rasterio as rio
from rasterio.mask import mask
import requests as r
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvas
from rasterio.plot import reshape_as_image



def style_function(feature):
    return {
        'fillOpacity': 0,
        'color': 'red',
        'weight': 3,
    }

@st.cache_data
def read_renabap():   
    try:
        print("ANTES")
        df = gpd.read_file('./data/amba.geojson')
        print("DESPUES")
        return df
    except Exception as e:
        st.error(f"Error reading GeoPandas dataframe: {e}")
        return 0
    

#
df = read_renabap()





selected_option = st.selectbox('Select an option:', df.renabap_id.tolist())
gdf = df[df.renabap_id==selected_option].reset_index(drop=True)

predtif = r.get(f'http://localhost:8000/run/{selected_option}').json()

raster = rio.open(predtif)

# Read the raster layer as a NumPy array
array = raster.read(1)
array[array < 150] = raster.nodata
# Define the colormap to use

# Convert the raster data to a colormap image using Matplotlib
m = folium.Map(location=[raster.bounds[1], raster.bounds[0]], zoom_start=12)
folium.TileLayer(
    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attr='Esri',
    name='Esri Satellite',
    overlay=True,
    control=True
).add_to(m)

# Add the raster as a TileLayer
#folium.raster_layers.ImageOverlay(
#    image=raster.read(1),
#    bounds=[[raster.bounds[1], raster.bounds[0]], [raster.bounds[3], raster.bounds[2]]],
#    opacity=0.5
#).add_to(m)


poliresult = r.get(f'http://localhost:8001/p2r?p2r={selected_option}').json()['msg'][0]
poliresultdf = gpd.read_file(poliresult)

# Add the GeoDataFrame as a polygon layer
folium.GeoJson(gdf, style_function=style_function).add_to(m)
folium.GeoJson(poliresultdf).add_to(m)
# Add the colormap image overlay to the map

folium_static(m)