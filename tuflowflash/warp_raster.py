from pyproj import Proj
from pyproj import Transformer
from shapely.geometry import mapping
from typing import List
from datetime import datetime, timedelta
from pathlib import Path
from osgeo import gdal

import cftime
import ftplib
import geopandas
import glob
import gzip
import logging
import netCDF4 as nc
import numpy as np
import os
import pandas as pd
import pytz
import requests
import rioxarray
import shutil
import time

import os

os.chdir(r"C:\Townsville_Flash\Tvlle_Combined\runs")

# Create timezone objecty of Sydney Australia
aus_tz = pytz.timezone("Australia/Sydney")
# Find time right now
time_now = datetime.now(aus_tz)

# Generate the name of the .nc file to download (for example sm_pct_2023.nc). One nc file exists for each year and the file is updated daily
nc_soil_moisture_filename = "sm_pct_" + str(time_now.year) + ".nc"
# nc_soil_moisture_filename = "sm_pct_2024.nc"
nc_soil_moisture_file = (
    self.settings.soil_moisture_folder / nc_soil_moisture_filename
)

logging.info(nc_soil_moisture_file)
# logging.info(type(nc_soil_moisture_file))

# Generate request for file from the AWO HTTP server
awra_l_url = self.settings.soil_moisture_awra_l_url + nc_soil_moisture_filename
response = requests.get(awra_l_url)

if response.status_code == 200:
    with open(nc_soil_moisture_file, "wb") as file:
        file.write(response.content)
    logging.info("succesfully downloaded %s", nc_soil_moisture_filename)

# Input files
soil_moisture_nc_file = "../bc_dbase/Soils/sm_pct_2025.nc"
soil_moisture_tif_file = "../bc_dbase/Soils/sm_pct_2025_latest.tif"
soil_moisture_tif_reprojected_file = "../bc_dbase/Soils/sm_pct_2025_latest_EPSG28355.tif"
soil_root_zone_depth_tif = "../model/gis/Soils/Depth/DES_000_200_EV_N_P_AU_TRN_C_20190901_Tville_Clip_GDA94_80mGrid.tif"
soil_depth_tif = "../bc_dbase/Soils/sm_pct_2025_latest_EPSG28355_depth.tif"

# Open the NetCDF variable
soil_moisture_nc = gdal.Open(f'NETCDF:\"{soil_moisture_nc_file}\":sm_pct')

# Get the last band (latest time)
band = soil_moisture_nc.GetRasterBand(soil_moisture_nc.RasterCount)

# Save as GeoTIFF
gdal.Translate(soil_moisture_tif_file, soil_moisture_nc, bandList=[soil_moisture_nc.RasterCount])

# the resolution and bounds of the depth of soil root zone
x_res = 80
y_res = 80
xmin = 388000
xmax = 523040
ymin = 7772960
ymax = 7918000

# # Hardcoded bounds (in meters, EPSG:28355)
# left, bottom, right, top = 388000, 7772960, 523040, 7918000

gdal.Warp(
    soil_moisture_tif_reprojected_file,
    soil_moisture_tif_file, 
    srcSRS="EPSG:4326", 
    dstSRS="EPSG:28355", 
    outputBounds=(xmin, ymin, xmax, ymax),
    xRes=x_res,
    yRes=y_res,
    resampleAlg="nearest")

from osgeo import gdal
import numpy as np

# Input and output paths
r1 = gdal.Open(soil_moisture_tif_reprojected_file)
r2 = gdal.Open(soil_root_zone_depth_tif)
out_path = soil_depth_tif

# Read arrays and NoData
a1 = r1.GetRasterBand(1)
a2 = r2.GetRasterBand(1)
d1 = a1.ReadAsArray()
d2 = a2.ReadAsArray()
nodata = a1.GetNoDataValue() or -9999

# Multiply with NoData mask
mask = (d1 != nodata) & (d2 != nodata)
result = np.full_like(d1, nodata, dtype=np.float32)
result[mask] = d1[mask] * d2[mask]

# Save result
driver = gdal.GetDriverByName("GTiff")
out = driver.Create(out_path, r1.RasterXSize, r1.RasterYSize, 1, gdal.GDT_Float32)
out.SetGeoTransform(r1.GetGeoTransform())
out.SetProjection(r1.GetProjection())
out.GetRasterBand(1).WriteArray(result)
out.GetRasterBand(1).SetNoDataValue(nodata)
out.FlushCache()
