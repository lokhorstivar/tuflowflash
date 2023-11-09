from osgeo import gdalconst
from osgeo import osr
from time import sleep

import os
import pandas as pd


try:
    import gdal
except:
    from osgeo import gdal

from pathlib import Path

import datetime
import glob
import json
import logging
import netCDF4 as nc
import numpy as np
import pytz
import requests
import shutil
from tuflowflash.impact_module import impactModule

logger = logging.getLogger(__name__)
RASTER_SOURCES_URL = (
    "https://rhdhv.lizard.net/api/v4/rastersources/"  # use rastersources endpoint
)

TIMESERIES_URL = "https://rhdhv.lizard.net/api/v4/timeseries/"

MAXWAITTIME_RASTER_UPLOAD = 120


class ProcessFlash:
    def __init__(self, settings):
        self.settings = settings

    def process_tuflow(self):
        # self.convert_flt_to_tiff()
        self.project_geotiff_rasters()
        logger.info("Tuflow results converted to tiff")
        if hasattr(self.settings, "waterlevel_result_uuid_file"):
            self.post_timeseries()

        if hasattr(self.settings, "waterdepth_raster_upload_list"):
            waterdepth_filenames, timestamps = self.select_rasters_to_upload(
                self.settings.waterdepth_raster_upload_list
            )
            self.post_temporal_raster_to_lizard(
                waterdepth_filenames, self.settings.depth_raster_uuid, timestamps
            )

        if hasattr(self.settings, "waterlevel_raster_upload_list"):
            filenames, timestamps = self.select_rasters_to_upload(
                self.settings.waterlevel_raster_upload_list
            )
            self.post_temporal_raster_to_lizard(
                filenames, self.settings.waterlevel_raster_uuid, timestamps
            )

        if self.settings.determine_impact:
            self.process_depth_to_impact(waterdepth_filenames)
        logger.info("Tuflow results posted to Lizard")

    def tuflow_tif_output_to_relative_timestamp(self, filename):
        file_stem = Path(filename).stem
        if file_stem.endswith("_00_00"):
            file_timestamp = float(file_stem[-9:-6])
        elif file_stem.endswith("_00"):
            file_timestamp = float(file_stem[-6:-3])
        else:
            file_timestamp = float(file_stem[-3:])
        timestamp = self.settings.start_time + datetime.timedelta(
            hours=float(file_timestamp)
        )
        return timestamp

    def process_depth_to_impact(self, waterdepth_filenames):
        impact_module = impactModule(self.settings, self.settings.end_result_type)
        raster_filenames = []
        timestamps = []
        for raster in waterdepth_filenames:
            vector_list = []
            vector_list.append(
                impact_module.determine_vulnerability_roads(
                    self.settings.roads_file, self.settings.projection, raster
                )
            )
            vector_list.append(
                impact_module.determine_vulnerability_buildings(
                    self.settings.buildings_file, self.settings.projection, raster
                )
            )
            if self.settings.end_result_type == "raster":
                raster_filenames.append(
                    impact_module.create_impact_raster(vector_list, raster)
                )
                timestamps.append(self.tuflow_tif_output_to_relative_timestamp(raster))
            else:
                logger.info("Geoserver vulnerability not implemented yet")
        if self.settings.end_result_type == "raster":
            self.post_temporal_raster_to_lizard(
                raster_filenames, self.settings.impact_raster_uuid, timestamps
            )
        else:
            logger.info("Geoserver vulnerability not implemented yet")

    def select_rasters_to_upload(self, raster_list):
        filenames = []
        for raster in raster_list:
            filenames.append(
                os.path.join(self.settings.raster_output_folder, raster + ".tif")
            )
        timestamps = []
        for (
            file
        ) in (
            filenames
        ):  # please note: might cause problems with water level rasters as well
            timestamps.append(self.tuflow_tif_output_to_relative_timestamp(file))
        return filenames, timestamps

    def project_geotiff_rasters(self):
        file_path_list = []
        if hasattr(self.settings, "waterdepth_raster_upload_list"):
            for file in self.settings.waterdepth_raster_upload_list:
                file_path_list.append(
                    os.path.join(self.settings.raster_output_folder, file + ".tif")
                )
        if hasattr(self.settings, "waterlevel_raster_upload_list"):
            for file in self.settings.waterlevel_raster_upload_list:
                file_path_list.append(
                    os.path.join(self.settings.raster_output_folder, file + ".tif")
                )
        for file in file_path_list:
            ds = gdal.Open(file, gdal.GA_Update)
            if ds:
                print("Updating projection for " + file)
                srs_wkt = self.create_projection(self.settings.projection)
                res = ds.SetProjection(srs_wkt)
                if res != 0:
                    print("Setting projection failed " + str(res))
                ds = None  # save, close
            else:
                print("Could not open with GDAL: " + file)

    def upload_bom_precipitation(self):
        self.NC_to_tiffs(Path("temp"))

        filenames = glob.glob(os.path.join("temp", "*.tif"))
        timestamps = []
        for file in filenames:
            rain_timestamp = Path(file).stem
            timestamp = self.settings.start_time + datetime.timedelta(
                hours=float(rain_timestamp)
            )
            timestamps.append(timestamp)
        self.post_temporal_raster_to_lizard(
            filenames, self.settings.rainfall_raster_uuid, timestamps
        )
        logger.info("Bom rainfall posted to Lizard")

    def archive_simulation(self):
        folder_time_string = (
            str(self.settings.start_time).replace(":", "_").replace(" ", "_")
        )
        result_folder = os.path.join(
            self.settings.archive_folder, "results_" + folder_time_string
        )
        os.mkdir(result_folder)
        shutil.copytree("Log", os.path.join(result_folder, "log"))
        shutil.copytree(
            self.settings.output_folder, os.path.join(result_folder, "results")
        )
        if hasattr(self.settings, "rain_grids_csv"):
            shutil.copyfile(
                self.settings.rain_grids_csv,
                os.path.join(result_folder, "rain_grids.csv"),
            )
            shutil.copytree(
                self.settings.rain_grids_csv.parent / "RFG",
                os.path.join(result_folder, "RFG"),
            )
        elif hasattr(self.settings, "gauge_rainfall_file"):
            shutil.copyfile(
                self.settings.gauge_rainfall_file,
                os.path.join(result_folder, "gauge_rain.csv"),
            )
        if hasattr(self.settings, "boundary_csv_tuflow_file"):
            shutil.copyfile(
                self.settings.boundary_csv_tuflow_file,
                os.path.join(result_folder, "boundary_csv_tuflow_file.csv"),
            )
        if self.settings.get_bom_forecast:
            shutil.copyfile(
                os.path.join("temp", "forecast_rain.nc"),
                os.path.join(result_folder, "forecast_rain.nc"),
            )
        if self.settings.get_bom_nowcast:
            shutil.copyfile(
                os.path.join("temp", "radar_rain.nc"),
                os.path.join(result_folder, "radar_rain.nc"),
            )
        self.remove_flts_from_archive(result_folder)
        shutil.make_archive(result_folder, "zip", result_folder)
        shutil.rmtree(result_folder)
        logging.info("succesfully archived files to: %s", result_folder)

    def remove_flts_from_archive(self, result_folder):
        for dirname, dirs, files in os.walk(result_folder):
            for file in files:
                if file.endswith(".flt"):
                    source_file = os.path.join(dirname, file)
                    os.remove(source_file)

    def clear_in_output(self):
        shutil.rmtree("Log")
        shutil.rmtree(self.settings.output_folder)

        if hasattr(self.settings, "rain_grids_csv"):
            os.remove(self.settings.rain_grids_csv)
        if hasattr(self.settings, "netcdf_forecast_rainfall_file"):
            os.remove(self.settings.netcdf_forecast_rainfall_file)
        if hasattr(self.settings, "netcdf_nowcast_rainfall_file"):
            os.remove(self.settings.netcdf_nowcast_rainfall_file)
        if hasattr(self.settings, "gauge_rainfall_file"):
            os.remove(self.settings.gauge_rainfall_file)
        if hasattr(self.settings, "boundary_csv_tuflow_file"):
            os.remove(self.settings.boundary_csv_tuflow_file)

    def create_projection(self, projection):
        """obtain wkt definition of the tuflow spatial projection. Used to write
        geotiff format files with gdal.
        """
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(projection)
        srs_wkt = srs.ExportToWkt()
        return srs_wkt

    def convert_flt_to_tiff(self):
        gdal.UseExceptions()
        file_path_list = []
        if self.settings.waterdepth_raster_upload_list:
            for file in self.settings.waterdepth_raster_upload_list:
                file_path_list.append(
                    os.path.join(self.settings.raster_output_folder, file + ".flt")
                )
        if self.settings.waterlevel_raster_upload_list:
            for file in self.settings.waterlevel_raster_upload_list:
                file_path_list.append(
                    os.path.join(self.settings.raster_output_folder, file + ".flt")
                )

        for file in file_path_list:
            data = gdal.Open(file, gdalconst.GA_ReadOnly)
            nodata = data.GetRasterBand(1).GetNoDataValue()
            data_array = data.GetRasterBand(1).ReadAsArray()
            geo_transform = data.GetGeoTransform()
            proj = self.create_projection(self.settings.projection)
            x_res = data.RasterXSize
            y_res = data.RasterYSize

            output = file.replace(".flt", ".tif")
            target_ds = gdal.GetDriverByName("GTiff").Create(
                output, x_res, y_res, 1, gdal.GDT_Float32, options=["COMPRESS=Deflate"]
            )
            target_ds.SetGeoTransform(geo_transform)
            target_ds.GetRasterBand(1).WriteArray(data_array)
            target_ds.SetProjection(proj)
            target_ds.GetRasterBand(1).SetNoDataValue(nodata)
            target_ds.FlushCache()
            target_ds = None
        return

    def create_post_element(self, series, shift):
        data = []
        aus_now = datetime.datetime.now(pytz.timezone("Australia/Sydney"))
        timezone_stamp = (
            "+" + str(int(aus_now.utcoffset().total_seconds() / 3600)).zfill(2) + ":00"
        )
        for index, value in series.iteritems():
            data.append(
                {
                    "time": index.isoformat() + timezone_stamp,
                    "value": str(float(value) + shift),
                }
            )
        return data

    def post_timeseries(self):
        username = "__key__"
        password = self.settings.apikey
        headers = {
            "username": username,
            "password": password,
            "Content-Type": "application/json",
        }

        result_ts_uuids = pd.read_csv(self.settings.waterlevel_result_uuid_file)
        # temp
        file_name = os.path.join(
            self.settings.output_folder,
            str(self.settings.tcf_file.name).replace(".tcf", "_PO.csv"),
        )

        results_dataframe = pd.read_csv(file_name)
        results_dataframe = results_dataframe.iloc[1:]

        for index, row in results_dataframe.iterrows():
            results_dataframe.at[
                index, "datetime"
            ] = self.settings.reference_time + datetime.timedelta(
                hours=float(row["Location"])
            )  # weird row order in po file
        results_dataframe.set_index("datetime", inplace=True)
        for index, row in result_ts_uuids.iterrows():
            timeserie = self.create_post_element(
                results_dataframe[row["po_name"]], row["shift"]
            )
            url = TIMESERIES_URL + row["ts_uuid"] + "/events/"
            r = requests.delete(url=url, headers=headers)
            r = requests.post(url=url, data=json.dumps(timeserie), headers=headers)

    def NC_to_tiffs(self, Output_folder):
        nc_data_obj = nc.Dataset(self.settings.netcdf_rainfall_file)
        Lon = nc_data_obj.variables["y"][:]
        Lat = nc_data_obj.variables["x"][:]
        precip_arr = np.asarray(
            nc_data_obj.variables["rainfall_depth"]
        )  # read data into an array
        # the upper-left and lower-right coordinates of the image
        LonMin, LatMax, LonMax, LatMin = [Lon.min(), Lat.max(), Lon.max(), Lat.min()]

        # resolution calculation
        N_Lat = len(Lat)
        N_Lon = len(Lon)
        Lon_Res = (LonMax - LonMin) / (float(N_Lon) - 1)
        Lat_Res = (LatMax - LatMin) / (float(N_Lat) - 1)

        for i in range(len(precip_arr[:])):
            # to create. tif file
            driver = gdal.GetDriverByName("GTiff")
            out_tif_name = os.path.join(
                Output_folder, str(nc_data_obj.variables["time"][i]) + ".tif"
            )
            out_tif = driver.Create(out_tif_name, N_Lon, N_Lat, 1, gdal.GDT_Float32)

            #  set the display range of the image
            # -Lat_Res must be - the
            geotransform = (LonMin, Lon_Res, 0, LatMax, 0, -Lat_Res)
            out_tif.SetGeoTransform(geotransform)

            # get geographic coordinate system information to select the desired geographic coordinate system
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(
                self.settings.projection
            )  #  the coordinate system of the output
            out_tif.SetProjection(
                srs.ExportToWkt()
            )  #  give the new layer projection information

            # data write
            out_tif.GetRasterBand(1).WriteArray(
                precip_arr[i]
            )  #  writes data to memory, not to disk at this time
            out_tif.FlushCache()  #  write data to hard disk
            out_tif = None  #  note that the tif file must be closed

    def post_temporal_raster_to_lizard(self, filenames, raster_uuid, timestamps):
        username = "__key__"
        password = self.settings.apikey
        headers = {
            "username": username,
            "password": password,
        }

        json_headers = {
            "username": username,
            "password": password,
            "Content-Type": "application/json",
        }
        raster_url = RASTER_SOURCES_URL + raster_uuid + "/"
        url = raster_url + "data/"

        aus_now = datetime.datetime.now(pytz.timezone("Australia/Sydney"))
        timezone_stamp = (
            "+" + str(int(aus_now.utcoffset().total_seconds() / 3600)).zfill(2) + ":00"
        )
        requests.delete(url=url, headers=json_headers)

        for file, timestamp in zip(filenames, timestamps):
            logger.debug("posting file %s to lizard", file)
            lizard_timestamp = timestamp.strftime("%Y-%m-%dT%H:%M:00")
            lizard_timestamp = lizard_timestamp + timezone_stamp
            file = {"file": open(file, "rb")}
            data = {"timestamp": lizard_timestamp}
            r = requests.post(url=url, data=data, files=file, headers=headers)

            try:
                r.raise_for_status()
            except:
                logger.error("Error, file post for %s failed", file)

            # waittime = 0
            # while (
            # requests.get(url=r.json()["url"]).json()["status"]
            # not in ["SUCCESS", "FAILURE"]
            # ) and (waittime <= MAXWAITTIME_RASTER_UPLOAD):
            # logger.info(
            # "raster: %s upload status: %s",
            # file,
            # requests.get(url=r.json()["url"]).json()["status"],
            # )
            # waittime += 10
            # sleep(10)
        return

    def track_historic_forecasts_in_lizard(self):
        headers = {
            "username": "__key__",
            "password": self.settings.apikey,
            "Content-Type": "application/json",
        }
        params = {"page_size": 100000}

        historic_admin = pd.read_csv(self.settings.historic_forecast_administration_csv)
        for index, row in historic_admin.iterrows():
            for x in range(len(row) - 1, 0, -1):
                url_to_update = TIMESERIES_URL + row[x] + "/events/"
                source_data_url = TIMESERIES_URL + row[x - 1] + "/events/"
                requests.delete(url=url_to_update, headers=headers)
                r = requests.get(url=source_data_url, params=params, headers=headers)
                source_df = pd.DataFrame(r.json()["results"])
                try:
                    source_df["time"] = pd.to_datetime(source_df["time"])
                    source_df.set_index("time", inplace=True)
                    timeserie_data = []
                    for index, value in source_df["value"].iteritems():
                        timeserie_data.append(
                            {"time": index.isoformat(), "value": str(value)}
                        )
                    r = requests.post(
                        url=url_to_update,
                        data=json.dumps(timeserie_data),
                        headers=headers,
                    )
                except:
                    logger.warning("Did not fill historic timeserie: %s", row[x])
                    pass
