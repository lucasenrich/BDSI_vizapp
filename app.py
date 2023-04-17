#!/usr/bin/env python3.11
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


START_DATE = '2021-01-01'
END_DATE = '2021-04-30'
CLOUD_FILTER = 1
CLD_PRB_THRESH = 1
NIR_DRK_THRESH = 0.15
CLD_PRJ_DIST = 1
BUFFER = 50


def croppge(of,nom,min_lon, min_lat, max_lon, max_lat, delta=0, idx=0):
    #min_lon, min_lat, max_lon, max_lat = df.geometry[idx].bounds
    min_lon = min_lon-delta
    min_lat = min_lat-delta

    max_lon = max_lon+delta
    max_lat = max_lat+delta

    bbox = box(min_lon, min_lat, max_lon, max_lat)
    crs = 'EPSG:4326' # WGS84

    with rio.open(of[1]) as src:
        # Crop the raster using the bounding box
        out_image, out_transform = mask(src, [bbox], crop=True)
        out_meta = src.meta.copy()
        out_meta.update({"driver": "GTiff",
                         "height": out_image.shape[1],
                         "width": out_image.shape[2],
                         "transform": out_transform,
                         "crs": crs})
    cropped_file = './data/raster_cropped/'+nom+"_cropped.tif"
    with rio.open(cropped_file, "w", **out_meta) as dest:
        dest.write(out_image)
    fl_300 = True
    with rio.open(cropped_file) as src:
        print(src.width,src.height)
        if src.width<300 and src.height <300:
            fl_300 = False
    if not fl_300:
            #os.remove(cropped_file)
            croppge(of,nom,min_lon, min_lat, max_lon, max_lat,delta = 0.005)
    return cropped_file

def custom_merge_works(old_data, new_data, old_nodata, new_nodata, index=None, roff=None, coff=None):
    old_data[:] = np.maximum(old_data, new_data)  # <== NOTE old_data[:] updates the old data array *in place*


def get_s2_sr_cld_col(aoi, start_date, end_date):
    # Import and filter S2 SR.
    s2_sr_col = (ee.ImageCollection('COPERNICUS/S2_SR')
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', CLOUD_FILTER)))

    # Import and filter s2cloudless.
    s2_cloudless_col = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
        .filterBounds(aoi)
        .filterDate(start_date, end_date))

    # Join the filtered s2cloudless collection to the SR collection by the 'system:index' property.
    return ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(**{
        'primary': s2_sr_col,
        'secondary': s2_cloudless_col,
        'condition': ee.Filter.equals(**{
            'leftField': 'system:index',
            'rightField': 'system:index'
        })
    }))


def get_raster_gearth(df,iy=6,anio=2021,delta = 0.01, TCI = False,path_output= os.getcwd()):
    nom = df.renabap_id[iy].astype(str)
    prov = df.provincia[iy]
    sep17 = df[(df.renabap_id.astype(str)==nom)].reset_index(drop=True)
    
    area_barrio = sep17['geometry'].to_crs({'init': 'epsg:4326'})\
           .map(lambda p: p.area )[0]

    min_lon, min_lat, max_lon, max_lat = df.geometry[iy].bounds
    
    min_lon -= delta
    min_lat -= delta
    max_lon += delta
    max_lat += delta
    
    coords = [[min_lon,min_lat],[min_lon,max_lat],[max_lon,max_lat],[max_lon,min_lat]]
    aoi = ee.Geometry.Polygon(coords)

    AOI = ee.Geometry.Polygon(coords)

    START_DATE = str(anio)+'-09-01'
    END_DATE = str(anio)+'-12-31'
    if anio == 2020:
        START_DATE = str(anio)+'-01-01'
        END_DATE = str(anio)+'-12-31'
    if anio == 2022:
        START_DATE = str(anio)+'-02-01'
        END_DATE = str(anio)+'-03-31'


    s2_sr_cld_col_eval = get_s2_sr_cld_col(AOI, START_DATE, END_DATE)
    a = anio


    ffa_db = ee.Image(ee.ImageCollection('COPERNICUS/S2_SR') 
                   .filterBounds(aoi) 
                   .filterDate(ee.Date(START_DATE), ee.Date(END_DATE))
                   .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', CLOUD_FILTER)) 
                   .first() 
                   .clip(aoi))
    try:
        if TCI:
            bandas = ['TCI_R','TCI_G','TCI_B']
            link = ffa_db.select('TCI_R','TCI_G','TCI_B').getDownloadURL({
                'scale': 1,
                'crs': 'EPSG:4326',
                'fileFormat': 'GeoTIFF'})
        else:
            bandas = ['B2','B3','B4','B8','B11']
            link = ffa_db.select('B2','B3','B4','B8','B11').getDownloadURL({
                'scale': 1,
                'crs': 'EPSG:4326',
                'fileFormat': 'GeoTIFF'})
    except Exception as e:
        print(e)
        if 'total request size' in str(e).lower():
            t,of = get_raster_gearth(df=df, iy=iy,anio=anio,delta = delta-0.001, TCI = TCI, path_output = path_output)
            if t == True:
                return True,of
        else: 
            return e 
    
    
    file = 'raster.zip'
    
    
    response = request.urlretrieve(link, file)
    
    
    
    if not os.path.isdir(path_output):
        os.mkdir(path_output)
    with ZipFile(file, 'r') as zObject:
        zObject.extractall(path=path_output)
        
        
        
    raster_files = [os.path.join(path_output,i) for i in os.listdir(os.path.join(path_output))]
    rf = []
    for i in bandas:
        for j in raster_files:
            if i in j:
                rf.append(j)
    if not TCI:
        out10m_B11 = os.path.join(os.getcwd(),"B11_10m".join(rf[4].split("B11")))

        src = gdal.Open(rf[0])
        xres, yres = operator.itemgetter(1,5)(src.GetGeoTransform())
        gdal.Warp(out10m_B11, rf[4], xRes=xres, yRes=yres)
        os.remove(rf[4])

    
    raster_files = [os.path.join(path_output,i) for i in os.listdir(os.path.join(path_output))]
    rf = []
    for i in bandas:#['B2','B3','B4','B8','B11']:
        for j in raster_files:
            if i in j:
                rf.append(j)  
    with rio.open(rf[0]) as blue_raster:
        blue = blue_raster.read(1, masked=True)
        out_meta = blue_raster.meta.copy()
        out_meta.update({"count": len(rf)})
        if TCI:
            out_img = os.path.join(path_output,f'./data/raster/{nom}_{anio}_TCI.tif')
        else:
            out_img = os.path.join(path_output,f'./data/raster/{nom}_{anio}.tif')
        file_list = [rio.open(i) for i in rf]
        with rio.open(out_img, 'w', **out_meta) as dest:
            for band_nr, src in enumerate(file_list, start=1):
                dest.write(src.read(1, masked=True), band_nr)
    for k in file_list:
        k.close()
    for h in rf:
        os.remove(h)
    os.remove('raster.zip')
    return True, out_img

def predict_model(path,nom,sw = 3):
    
    ge = rio.open(path)
    ix = int(np.ceil(ge.height/300))
    jx = int(np.ceil(ge.width/300))
    ix=ix*sw
    jx=jx*sw
    imagespath = './cropped/images/'
    if not os.path.isdir(imagespath):
        os.mkdir(imagespath)
    for i in range(ix):
        for j in range(jx):
            # Open the raster file
            with rio.open(path) as src:
                # Define the window for the crop
                win_height = 300  # height of the window
                win_width = 300  # width of the window
                row_start = 0+j*(300/sw)  # starting row for the window
                col_start = 0+i*(300/sw)  # starting column for the window
                window = Window(col_start, row_start, win_width, win_height)
                # Read the data for the window
                data = src.read(window=window)
                # Update the metadata for the cropped image
                out_meta = src.meta.copy()
                out_meta.update({
                    "height": win_height,
                    "width": win_width,
                    "transform": rio.windows.transform(window, src.transform)
                })
                # Write the cropped image to a new file

                p = os.path.join(imagespath,f'output_crop_{win_height}_{win_height}_{row_start}_{col_start}.tif')

                with rio.open(p, 'w', **out_meta) as dst:
                    dst.write(data)
    ge.close()
    predict_config = PredictConfig(images_path='./cropped/', # ruta a las imagenes sobre las cuales queremos predecir
                                    results_path='./cropped/out/', # ruta de destino para nuestra predicción
                                    batch_size=1,
                                    model_path=os.path.join(os.getcwd(),'model', 'model.h5'),  #  ruta al modelo (.h5)
                                    height=160,
                                    width=160,
                                    n_channels=5,
                                    n_classes=1,
                                    class_weights= [1]) 
    predict(predict_config)
    anio = '2021'
    r = './cropped/out/'
    files_to_mosaic = glob.glob(r+"\\*")
    m = [rio.open(files_to_mosaic[x]) for x in range(len(files_to_mosaic))]
    arr, out_trans = merge(m, method = custom_merge_works)
    output_meta = m[0].meta.copy()
    output_meta.update(
        {"driver": "GTiff",
            "height": arr.shape[1],
            "width": arr.shape[2],
            "transform": out_trans,
        })


    outputdir = "./output_pred/"#os.path.join("C:/Users/HP/OneDrive/pytorch/torchgeo/renabap/output",prov)
    if not os.path.isdir(outputdir):
        os.mkdir(outputdir)

    fname = nom+".tif"#path.split("\\")[len(path.split("\\"))-1]

    with rio.open(os.path.join(outputdir,fname), "w", **output_meta) as p:
        p.write(arr)


    for k in m:
        k.close()
    for h in files_to_mosaic:
        os.remove(h)
    shutil.rmtree(imagespath)
    
    return True,os.path.join(outputdir,fname)


def style_function(feature):
    return {
        'fillOpacity': 0,
        'color': 'red',
        'weight': 3,
    }

@st.cache_data
def initee():
    ee.Authenticate()
    ee.Initialize()
    return True


@st.cache_data
def read_renabap():
    
    try:
        print("ANTES")
        df = gpd.read_file('./data/amba.geojson')
        print("DESPUES")
    except Exception as e:
        st.error(f"Error reading GeoPandas dataframe: {e}")
    
    return df

#
df = read_renabap()
initee()




selected_option = st.selectbox('Select an option:', df.nombre_barrio.tolist())

gdf = df[df.nombre_barrio==selected_option].reset_index(drop=True)
min_lon, min_lat, max_lon, max_lat = gdf.geometry[0].bounds


ofTCI = get_raster_gearth(df=gdf, iy=0,TCI=True)
of = get_raster_gearth(df=gdf, iy=0,TCI=False)
pf = croppge(of,selected_option,min_lon, min_lat, max_lon, max_lat, idx = 0)
# Create a Leaflet map
_, predtif = predict_model(pf,selected_option)



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


poliresult = r.get(f'http://localhost:8000/p2r?p2r={selected_option}.tif').json()
poliresult = poliresult['msg'][0]
poliresultdf = gpd.read_file(poliresult)

# Add the GeoDataFrame as a polygon layer
folium.GeoJson(gdf, style_function=style_function).add_to(m)
folium.GeoJson(poliresultdf).add_to(m)
# Add the colormap image overlay to the map

# Display the map in Streamlit
col1, col2 = st.columns(2)
with col1:
    folium_static(m)

