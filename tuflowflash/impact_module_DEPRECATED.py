from rasterstats import zonal_stats
import geopandas as gpd
import rasterio
from rasterio import features
import numpy as np

BUFFER_SIZE = 5
CAP_STYLE = 2
CLASSES = {"low": 1, "medium": 2, "high": 3}

def save_array_to_tiff(outfile, array, profile):
    """Saves rasterio or numpy array to tiff file

    Args:
        outfile (str): destination path for the file (tiff)
        array (np.array): array with the data
        profile (rasterio): raster metadata profile (can be derived from other raster)
    """
    with rasterio.open(outfile, "w", **profile) as dst:
        dst.write(array, 1)

def rasterize_vector_column(geodataframe, value_column, outfile, transform,depth_raster):
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

    ones_list = list(geodataframe[geodataframe[value_column]==1]["geometry"])
    if ones_list:
        features.rasterize(
            ones_list,
            out_shape=shape,
            out=outfile,
            transform=transform,
            all_touched=False,
            default_value=1,
        )

    twos_list = list(geodataframe[geodataframe[value_column]==2]["geometry"])
    if twos_list:
        features.rasterize(
            twos_list,
            out_shape=shape,
            out=outfile,
            transform=transform,
            all_touched=False,
            default_value=2,
        )

    threes_list = list(geodataframe[geodataframe[value_column]==3]["geometry"])
    if threes_list:
        features.rasterize(
            threes_list,
            out_shape=shape,
            out=outfile,
            transform=transform,
            all_touched=False,
            default_value=3,
        )

class impactModule:
    def __init__(self, settings, end_result_type):
        self.settings = settings
        self.end_result_type = end_result_type

    def translate_to_classes(self, vector_layer, threshold1, threshold2):
        vector_layer["max_depth"].fillna(0, inplace=True)
        vector_layer.loc[
            vector_layer["max_depth"] < threshold1, ["vulnerability_class"]
        ] = CLASSES["low"]
        vector_layer.loc[
            (vector_layer["max_depth"] >= threshold1)
            & (vector_layer["max_depth"] < threshold2),
            ["vulnerability_class"],
        ] = CLASSES["medium"]
        vector_layer.loc[
            vector_layer["max_depth"] >= threshold2, ["vulnerability_class"]
        ] = CLASSES["high"]

        return vector_layer

    def determine_vulnerability_roads(self, geopackage, crs, depth_raster):
        roads = gpd.read_file(geopackage, crs=crs)
        geometry_linestrings = roads.geometry
        # from linestrings to polygons by buffering, one value for each road within the buffered road
        roads.geometry = roads.geometry.buffer(BUFFER_SIZE, cap_style=CAP_STYLE)
        with rasterio.open(depth_raster) as src:
            zonal_statistics = zonal_stats(
                roads, src.read(1), affine=src.transform, stats=["max", "sum"]
            )

            roads["max_depth"] = list(map(lambda x: x["max"], zonal_statistics))
            roads["volume"] = list(map(lambda x: x["sum"], zonal_statistics))

            vulnerable_roads_results = self.translate_to_classes(roads, 0.15, 0.3)
            if self.end_result_type == "geoserver":
                vulnerable_roads_results.geometry = geometry_linestrings
            output_filename = depth_raster.replace(".tif", "_roads.gpkg")
            vulnerable_roads_results.to_file(
                str(output_filename), layer="vulnerable_roads"
            )
        return output_filename

    def determine_vulnerability_buildings(self, geopackage, crs, depth_raster):
        buildings = gpd.read_file(geopackage, crs=crs)
        with rasterio.open(depth_raster) as src:
            zonal_statistics = zonal_stats(
                buildings, src.read(1), affine=src.transform, stats=["max", "sum"]
            )

            buildings["max_depth"] = list(map(lambda x: x["max"], zonal_statistics))
            buildings["volume"] = list(map(lambda x: x["sum"], zonal_statistics))

            vulnerable_buildings_results = self.translate_to_classes(
                buildings, 0.15, 0.3
            )
            output_filename = depth_raster.replace(".tif", "_buildings.gpkg")
            vulnerable_buildings_results.to_file(
                str(output_filename), layer="vulnerable_buildings"
            )
        return output_filename

    def create_impact_raster(self, vector_list, depth_raster):
        with rasterio.open(depth_raster) as src:
            profile = src.profile
            transform = profile["transform"]
        impact_data = (
            np.empty((profile["height"], profile["width"])) + profile["nodata"]
        ).astype(dtype=np.float32)
        for vector_file in vector_list:
            geodataframe = gpd.read_file(vector_file)
            rasterize_vector_column(
                geodataframe,
                "vulnerability_class",
                impact_data,
                transform,
                depth_raster,
            )
        save_array_to_tiff(depth_raster.replace(".tif", "_vulnerability.tif"), impact_data, profile)
        return depth_raster.replace(".tif", "_vulnerability.tif")
