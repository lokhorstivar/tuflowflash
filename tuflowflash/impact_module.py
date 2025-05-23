from rasterstats import zonal_stats, point_query
import geopandas as gpd
import rasterio
from rasterio import features
import numpy as np
import pandas as pd
from rasterio.crs import CRS
from shapely.geometry import box

CLASSES = {"low": 1, "medium": 2, "high": 3, "very high": 4}

BUFFER_STYLE = {
    "Buildings": "square",
    "Road Closure Points": "round",
    "Evacuation Centre": "round",
    "Council Assets": "square",
}
BUFFER_SIZE = {
    "Buildings": 10,
    "Road Closure Points": 7,
    "Evacuation Centre": 30,
    "Council Assets": 20,
}


def save_array_to_tiff(outfile, array, profile):
    """Saves rasterio or numpy array to tiff file

    Args:
        outfile (str): destination path for the file (tiff)
        array (np.array): array with the data
        profile (rasterio): raster metadata profile (can be derived from other raster)
    """
    with rasterio.open(outfile, "w", **profile) as dst:
        dst.write(array, 1)


def rasterize_vector_column(
    geodataframe, value_column, outfile, transform, depth_raster, layer_name
):
    """Process vector_file to raster

    Args:
        vector_file (str): any gdal supported feature file. Defaults = shape_file
        value_column (str): column name of the vector's attribute table containing the burn value
        outfile (str): file to burn into
        transform (transform): geotransform of the target raster
        layer (str, optional): layer name in case of geopackage
        bbox (bbox, optional): bounding box to clip geopackage
    """

    with rasterio.open(depth_raster) as src:
        shape = src.shape

    geodataframe.geometry = geodataframe.buffer(cap_style=BUFFER_STYLE.get(layer_name), distance=BUFFER_SIZE.get(layer_name))

    shapes = (
        (geom, value)
        for geom, value in zip(geodataframe["geometry"], geodataframe[value_column])
    )

    features.rasterize(
        shapes,
        out_shape=shape,
        out=outfile,
        transform=transform,
        all_touched=False,
        default_value=1,
    )


class impactModule:
    def __init__(self, settings, end_result_type):
        self.settings = settings
        self.end_result_type = end_result_type

    def translate_to_classes(self, vector_layer, threshold0, threshold1, threshold2):
        vector_layer["max_depth"].fillna(0, inplace=True)
        vector_layer.loc[
            vector_layer["max_depth"] <= threshold0, ["vulnerability_class"]
        ] = CLASSES["low"]

        vector_layer.loc[
            (vector_layer["max_depth"] > threshold0)
            & (vector_layer["max_depth"] < threshold1),
            ["vulnerability_class"],
        ] = CLASSES["medium"]

        vector_layer.loc[
            (vector_layer["max_depth"] >= threshold1)
            & (vector_layer["max_depth"] < threshold2),
            ["vulnerability_class"],
        ] = CLASSES["high"]

        vector_layer.loc[
            vector_layer["max_depth"] >= threshold2, ["vulnerability_class"]
        ] = CLASSES["very high"]

        return vector_layer

    def sample_raster(
        self,
        raster_location,
        shapes,
        shapes_reference_level_colname,
        buffer_size=0,
        fill_value=0,
    ):
        with rasterio.open(raster_location) as src:
            if buffer_size > 0:
                zonal_statistics = zonal_stats(
                    shapes.buffer(buffer_size).tolist(),
                    src.read(1),
                    affine=src.transform,
                    stats=["max", "sum"],
                )
            else:
                zonal_statistics = point_query(
                    shapes["geometry"].tolist()[0], src.read(1), affine=src.transform
                )
        shapes = shapes.fillna(value=np.nan)
        shapes["max_wl"] = list(map(lambda x: x["max"], zonal_statistics))
        shapes["volume"] = list(map(lambda x: x["sum"], zonal_statistics))

        shapes["max_depth"] = shapes.apply(
            lambda row: (
                row["max_wl"] - row[shapes_reference_level_colname]
                if not pd.isnull(row["max_wl"])
                else fill_value
            ),
            axis=1,
        )
        return shapes

    def determine_floor_level(
        self, building_shape_path, floor_level_shape_path, floor_level_column
    ):
        buildings = gpd.read_file(building_shape_path)
        floor_level_points = gpd.read_file(floor_level_shape_path)

        # intersect floor level points with building footprints (with small buffer to include)
        buildings["uuid"] = [f"BUILDING_{int(x)}" for x in range(len(buildings))]

        buildings_buff = buildings.copy()
        buildings_buff["geometry"] = buildings_buff.buffer(2)
        buildings_buff_floor_level = buildings_buff.sjoin(
            floor_level_points[[floor_level_column, "geometry"]],
            predicate="intersects",
            how="left",
        )

        buildings_buff_floor_level_grouped = buildings_buff_floor_level.groupby("uuid")[
            [floor_level_column]
        ].min()

        def get_floor_level(row):
            floor_level = buildings_buff_floor_level_grouped.loc[
                buildings_buff_floor_level_grouped.index == row.uuid
            ]

            if np.isnan(floor_level[floor_level_column].iloc[0]):
                # use dem zonal statistics > min > + 0.15 m for missing values
                return row.DTM_STATSmin + 0.15
            else:
                return floor_level[floor_level_column].iloc[0]

        buildings["floor_level"] = buildings.apply(
            lambda row: get_floor_level(row), axis=1
        )

        return buildings

    def determine_vulnerability_buildings(
        self,
        buildings_geopackage,
        waterlevel_raster,
        reference_level_column_name="Level",
        buffer_size=2,
    ):
        buildings_sampled = self.sample_raster(
            raster_location=waterlevel_raster,
            shapes=buildings_geopackage,
            shapes_reference_level_colname=reference_level_column_name,
            buffer_size=buffer_size,
        )

        vulnerable_buildings_results = self.translate_to_classes(
            buildings_sampled, 0, 0.25, 1
        )

        with rasterio.open(waterlevel_raster, "r") as src:
            bounds =box(*tuple(src.bounds))
            
        vulnerable_buildings_results = vulnerable_buildings_results.clip(bounds)
        print(waterlevel_raster, type(waterlevel_raster))
        output_filename = waterlevel_raster.as_posix().replace(".tif", "_buildings.gpkg")
        vulnerable_buildings_results.to_file(
            str(output_filename), layer="vulnerable_buildings"
        )
        return output_filename

    def determine_vulnerability_road_closure_points(
        self,
        road_closure_pt_geopackage,
        waterlevel_raster,
        buffer_size=15,
        reference_level_column_name="Level",
    ):
        # set fill value to -9999, which means that when no waterlevel is found, the risk is assessed at < -0.3 m WD instead of 0, which would classify this as a medium risk
        road_closure_points_sampled = self.sample_raster(
            raster_location=waterlevel_raster,
            shapes=road_closure_pt_geopackage,
            shapes_reference_level_colname=reference_level_column_name,
            buffer_size=buffer_size,
            fill_value=-9999,
        )

        vulnerable_road_closure_points_results = self.translate_to_classes(
            road_closure_points_sampled, -0.3, 0, 0.15
        )

        with rasterio.open(waterlevel_raster, "r") as src:
            bounds =box(*tuple(src.bounds))
            
        vulnerable_road_closure_points_results = vulnerable_road_closure_points_results.clip(bounds)

        output_filename = waterlevel_raster.as_posix().replace(".tif", "_road_closure_points.gpkg")
        vulnerable_road_closure_points_results.to_file(
            str(output_filename), layer="vulnerable_road_closure_points"
        )
        return output_filename

    def determine_vulnerability_evacuation_centres(
        self,
        evac_centre_geopackage,
        waterlevel_raster,
        buffer_size=100,
        reference_level_column_name="Level",
    ):
        # set fill value to -9999, which means that when no waterlevel is found, the risk is assessed at < -0.3 m WD instead of 0, which would classify this as a medium risk
        evacuation_centres_sampled = self.sample_raster(
            raster_location=waterlevel_raster,
            shapes=evac_centre_geopackage,
            shapes_reference_level_colname=reference_level_column_name,
            buffer_size=buffer_size,
            fill_value=-9999,
        )

        vulnerable_evacuation_centres_results = self.translate_to_classes(
            evacuation_centres_sampled, -0.3, 0, 0.15
        )

        with rasterio.open(waterlevel_raster, "r") as src:
            bounds =box(*tuple(src.bounds))
            
        vulnerable_evacuation_centres_results = vulnerable_evacuation_centres_results.clip(bounds)

        output_filename = waterlevel_raster.as_posix().replace(".tif", "_evacuation_centres.gpkg")
        vulnerable_evacuation_centres_results.to_file(
            str(output_filename), layer="vulnerable_evacuation_centres"
        )
        return output_filename

    def determine_vulnerability_council_assets(
        self,
        council_assets_geopackage,
        waterlevel_raster,
        buffer_size=2,
        reference_level_column_name="Level",
    ):
        # set fill value to -9999, which means that when no waterlevel is found, the risk is assessed at < -0.3 m WD instead of 0, which would classify this as a medium risk
        council_assets_sampled = self.sample_raster(
            raster_location=waterlevel_raster,
            shapes=council_assets_geopackage,
            shapes_reference_level_colname=reference_level_column_name,
            buffer_size=buffer_size,
            fill_value=-9999,
        )
        vulnerable_council_assets_results = self.translate_to_classes(
            council_assets_sampled, -0.3, 0, 0.15
        )

        with rasterio.open(waterlevel_raster, "r") as src:
            bounds =box(*tuple(src.bounds))
            
        vulnerable_council_assets_results = vulnerable_council_assets_results.clip(bounds)

        output_filename = waterlevel_raster.as_posix().replace(".tif", "_council_assets.gpkg")
        vulnerable_council_assets_results.to_file(
            str(output_filename), layer="vulnerable_council_assets"
        )
        return output_filename

    def create_impact_raster(self, layer_dict, depth_raster, output_path):
        with rasterio.open(depth_raster) as src:
            profile = src.profile
            transform = profile["transform"]
        impact_data = (
            np.empty((profile["height"], profile["width"])) + profile["nodata"]
        ).astype(dtype=np.float32)

        for _, (layer_name, layer_location) in enumerate(layer_dict.items()):
            geodataframe = gpd.read_file(layer_location, layer_name=layer_name)
            rasterize_vector_column(
                geodataframe,
                "vulnerability_class",
                impact_data,
                transform,
                depth_raster,
                layer_name,
            )

        profile["crs"] = CRS.from_string("EPSG:28355")
        save_array_to_tiff(output_path, impact_data, profile
        )
        return output_path

    def create_impact_vector(self, layer_dict, impact_vector_output_path):
        for i, (layer_name, layer_location) in enumerate(layer_dict.items()):
            points = gpd.read_file(layer_location)
            # if "Name" in points.columns:
            #     points = points[["Name", "vulnerability_class", "geometry"]]
            # else:
            #     points = points[["vulnerability_class", "geometry"]]
            points.to_file(impact_vector_output_path, driver="GPKG", layer=layer_name)
