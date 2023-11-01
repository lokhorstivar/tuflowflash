from rasterstats import zonal_stats
import geopandas as gpd
import rasterio
from rasterio import features

BUFFER_SIZE = 5
CAP_STYLE = 2
CLASSES = {"low": 1, "medium": 2, "high": 3}


def rasterize_vector_column(geodataframe, value_column, outfile, transform):
    """Process vector_file to raster

    Args:
        vector_file (str): any gdal supported feature file. Defaults = shape_file
        value_column (str): column name of the vector's attribute table containing the burn value
        outfile (str): file to burn into
        transform (transform): geotransform of the target raster
        layer (str, optional): layer name in case of geopackage
        bbox (bbox, optional): bounding box to clip geopackage
    """

    for index, row in geodataframe.iterrows():
        geom = row["geometry"]
        value = row[value_column]
        features.rasterize(
            [geom],
            out=outfile,
            transform=transform,
            all_touched=False,
            default_value=value,
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
            output_filename = depth_raster.replace(".tif", "_roads.gpkg")
            vulnerable_buildings_results.to_file(
                str(output_filename), layer="vulnerable_buildings"
            )
        return output_filename

    def create_impact_raster(self, vector_list, depth_raster):
        with rasterio.open(depth_raster) as src:
            profile = src.profile
            transform = profile["transform"]
        for vector_file in vector_list:
            geodataframe = gpd.read_file(vector_file)
            rasterize_vector_column(
                geodataframe,
                "vulnerability_class",
                depth_raster.replace(".tif", "_vulnerability.tif"),
                transform,
            )
        return depth_raster.replace(".tif", "_vulnerability.tif")
