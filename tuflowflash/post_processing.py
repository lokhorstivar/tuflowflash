from osgeo import gdalconst
from osgeo import osr
from time import sleep
import os
import pandas as pd
import geopandas as gpd

try:
    import gdal
except:
    from osgeo import gdal

from pathlib import Path

import datetime
from datetime import timezone
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
gdal.SetConfigOption("GDAL_HTTP_UNSAFESSL", "YES")
TIMESERIES_URL = "https://rhdhv.lizard.net/api/v4/timeseries/"

MAXWAITTIME_RASTER_UPLOAD = 240


class ProcessFlash:
    def __init__(self, settings):
        self.settings = settings

    def process_tuflow(self):
        #self.convert_flt_to_tiff()
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
            # convert waterdepth filenames to waterlevel filenames at that timestamp
            waterlevel_filenames = [f.replace("_d_HR_", "_h_HR_") for f in waterdepth_filenames]
            self.process_waterlevel_to_impact(waterlevel_filenames)

        logger.info("Tuflow results posted to Lizard")

    def tuflow_tif_output_to_relative_timestamp(self, filename):
        file_stem = Path(filename).stem
        if file_stem.endswith("_00_00"):
            file_timestamp = float(file_stem[-9:-6])
        elif file_stem.endswith("_00"):
            file_timestamp = float(file_stem[-6:-3])
        else:
            file_timestamp = float(file_stem[-3:])
        timestamp = self.settings.reference_time + datetime.timedelta(
            hours=float(file_timestamp)
        )
        return timestamp

    def process_waterlevel_to_impact(self, waterlevel_filenames):
        logger.info("Processing waterlevel to impact")

        impact_module = impactModule(self.settings, self.settings.end_result_type)

        impact_output_files = (self.settings.output_folder / "impact").resolve()

        if not impact_output_files.exists():
            impact_output_files.mkdir()
            logger.info("Created impact directory")

        raster_filenames = []
        timestamps = []
        
        input_path = Path(r"D:\FLASH\01_Modelling\impact_module")
    
        buildings = gpd.read_file(input_path / "input" / "tville_impact_shapes.gpkg", layer="Buildings")
        road_closure_points = gpd.read_file(input_path / "input" / "tville_impact_shapes.gpkg", layer="Road Flooding Points")
        council_assets = gpd.read_file(input_path / "input" / "tville_impact_shapes.gpkg", layer="Council Services")
        evacuation_centres = gpd.read_file(input_path / "input" / "tville_impact_shapes.gpkg", layer="Evacuation Centres")
        
        dem_path = r"..\model\DEM\2012_w2018_19_Clipped_grid_DTM_LiDAR_v1.tif"

        max_waterlevel_raster = Path(f"{waterlevel_filenames[0].split('h_HR')[0]}h_HR_Max.tif").resolve()

        shutil.copy(max_waterlevel_raster, Path(r"D:\FLASH\01_Modelling\impact_module\input\latest_max_wl_rasters") / max_waterlevel_raster.name)
        # building_path_factsheet = impact_module.determine_vulnerability_buildings(
        #     buildings, max_waterlevel_raster, reference_level_column_name="FLOOR"
        # )
        # evacuation_centre_path_factsheet = impact_module.determine_vulnerability_evacuation_centres(
        #     evacuation_centres, max_waterlevel_raster, reference_level_column_name="Floor_Leve"
        # )
        # road_closure_point_path_factsheet = impact_module.determine_vulnerability_road_closure_points(
        #     road_closure_points, max_waterlevel_raster, reference_level_column_name="ROAD_LEVEL"
        # )
        # council_assets_path_factsheet = impact_module.determine_vulnerability_council_assets(
        #     council_assets, max_waterlevel_raster
        # )
        impact_vector_merged = impact_output_files / "impact_vector_factsheet.gpkg"

        logger.info("Creating impact vector")

        # impact_module.create_impact_vector(
        #     layer_dict={
        #         "Buildings": building_path_factsheet,
        #         "Road Closure Points": road_closure_point_path_factsheet,
        #         "Evacuation Centre": evacuation_centre_path_factsheet,
        #         "Council Assets": council_assets_path_factsheet,
        #     },
        #     impact_vector_output_path=impact_vector_merged,
        # )
        #generate_factsheet(impact_vector_path=impact_vector_merged)

        logger.info("Creating impact rasters")      

        for waterlevel_raster in waterlevel_filenames:
            building_path = impact_module.determine_vulnerability_buildings(
                buildings, Path(waterlevel_raster), reference_level_column_name="FLOOR"
            )
            evacuation_centre_path = impact_module.determine_vulnerability_evacuation_centres(
                evacuation_centres, Path(waterlevel_raster), reference_level_column_name="Floor_Leve"
            )
            road_closure_point_path = impact_module.determine_vulnerability_road_closure_points(
                road_closure_points, Path(waterlevel_raster), reference_level_column_name="ROAD_LEVEL"
            )
            council_assets_path = impact_module.determine_vulnerability_council_assets(
                council_assets, Path(waterlevel_raster)
            )

            if self.settings.end_result_type == "raster":
                impact_raster_location = impact_module.create_impact_raster(
                    layer_dict={
                        "Buildings": building_path,
                        "Road Closure Points": road_closure_point_path,
                        "Council Assets": council_assets_path,
                        "Evacuation Centre": evacuation_centre_path,
                    },
                    depth_raster=Path(waterlevel_raster),
                    output_path=impact_output_files / f"{Path(waterlevel_raster).stem}_vulnerability_raster.tif"
                )
                raster_filenames.append((impact_output_files / f"{Path(waterlevel_raster).stem}_vulnerability_raster.tif").as_posix())
                timestamps.append(self.tuflow_tif_output_to_relative_timestamp(waterlevel_raster))
            else:
                logger.info("Geoserver vulnerability not implemented yet")

        logger.info("Impact raster filenames: %s", raster_filenames)
        logger.info("Impact raster timestamps: %s", timestamps)

        if self.settings.end_result_type == "raster":
            self.post_temporal_raster_to_lizard(
                raster_filenames, self.settings.impact_raster_uuid, timestamps
            )
        else:
            logger.info("Geoserver vulnerability not implemented yet")

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
        try:
            shutil.rmtree("Log")
        except:
            pass

        try:
            shutil.rmtree(self.settings.output_folder)
        except:
            pass

        try:
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
        except:
            pass

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
        for index, value in series.iteritems():
            #logger.info(index.to_pydatetime().astimezone())
            #logger.info(index.to_pydatetime().astimezone(timezone.utc))
            data.append(
                {
                    "time": index.to_pydatetime().astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
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
        logger.info("Reference time: %s", self.settings.reference_time)
        logger.info(" results_dataframe: %s", results_dataframe.head())

        for index, row in results_dataframe.iterrows():
            results_dataframe.at[
                index, "datetime"
            ] = self.settings.reference_time + datetime.timedelta(
                hours=float(row["Location"])
            )  # weird row order in po file
        results_dataframe.set_index("datetime", inplace=True)
        logger.info(results_dataframe.index.tolist())
        for index, row in result_ts_uuids.iterrows():
            try:
                timeserie = self.create_post_element(
                    results_dataframe[row["po_name"]], row["shift"]
                )
                logger.info(row["ts_uuid"])
                url = TIMESERIES_URL + row["ts_uuid"] + "/events/"
                r = requests.delete(url=url, headers=headers)
                r = requests.post(url=url, data=json.dumps(timeserie), headers=headers)
            except:
                logger.info("Error when posting %s", row["po_name"])

    def NC_to_tiffs(self, Output_folder):
        nc_data_obj = nc.Dataset(self.settings.netcdf_forecast_rainfall_file)
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
            out_tif = driver.Create(out_tif_name, N_Lat, N_Lon, 1, gdal.GDT_Float32) # SWAPPED LON LAT > LAT LON
            
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

        requests.delete(url=url, headers=json_headers)

        for file_name, timestamp in zip(filenames, timestamps):
            timestamp_utc = timestamp.astimezone(timezone.utc)

            lizard_timestamp_utc = timestamp_utc.strftime("%Y-%m-%dT%H:00:00Z")

            file = {"file": open(file_name, "rb")}
            data = {"timestamp": lizard_timestamp_utc}
    
            r = requests.post(url=url, data=data, files=file, headers=headers)

            try:
                r.raise_for_status()
            except:
                logger.error("Error, file post for %s failed", file)

            waittime = 0
            while (requests.get(url=r.json()["url"]).json()["status"] not in ["SUCCESS", "FAILURE"]) and (waittime <= MAXWAITTIME_RASTER_UPLOAD):
                x = requests.get(url=r.json()["url"]).json()["status"]
                waittime += 10
                sleep(10)
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
