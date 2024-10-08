import os
from pathlib import Path
import pandas as pd
from fpdf import FPDF
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
from matplotlib.colors import LinearSegmentedColormap
import contextily as cx
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import rasterio
from PIL import Image
import random
import warnings
from shapely.geometry import box
from fpdf.enums import XPos, YPos
from fpdf.fonts import FontFace

warnings.filterwarnings("ignore")


MAP_EXTENT = Path(__file__).parent / "styling" / "map_extent.gpkg"

TEMP_MAP_SAVELOC_FOLDER = Path(__file__).parent / "styling"

GREEN_RGB = (218, 242, 208)
BLUE_RGB = (159, 214, 239)
YELLOW_RGB = (255, 229, 147)
RED_RGB = (245, 189, 221)


def create_zone_breakdown_table(impact_vector_path, zone_file_path, layer_names, type):
    impact_assessment_statistics_buildings = gpd.read_file(
        impact_vector_path, layer=layer_names[0]
    )
    impact_assessment_statistics_road_closure_points = gpd.read_file(
        impact_vector_path, layer=layer_names[1]
    )
    impact_assessment_statistics_evacuation_centre = gpd.read_file(
        impact_vector_path, layer=layer_names[2]
    )
    impact_assessment_statistics_council_assets = gpd.read_file(
        impact_vector_path, layer=layer_names[3]
    )

    zones = gpd.read_file(zone_file_path)
    zones = zones.loc[zones["Type"] == type]

    def get_stats(row, impact_df, t):
        impact_df_subset = impact_df.clip(row.geometry)
        vulnerability_count = impact_df_subset["vulnerability_class"].value_counts()
        if 1.0 in vulnerability_count.index:
            ic1 = vulnerability_count[1.0]
        else:
            ic1 = 0

        if 2.0 in vulnerability_count.index:
            ic2 = vulnerability_count[2.0]
        else:
            ic2 = 0

        if 3.0 in vulnerability_count.index:
            ic3 = vulnerability_count[3.0]
        else:
            ic3 = 0

        if 4.0 in vulnerability_count.index:
            ic4 = vulnerability_count[4.0]
        else:
            ic4 = 0
        building_count = len(impact_df_subset)
        return pd.Series([t, building_count, ic1, ic2, ic3, ic4])

    zone_buildings = zones.copy()
    zone_buildings[
        [
            "Type",
            "#",
            "None",
            "Minor",
            "Moderate",
            "Severe",
        ]
    ] = zone_buildings.apply(
        lambda row: get_stats(row, impact_assessment_statistics_buildings, "Buildings"),
        axis=1,
    )

    zone_rcp = zones.copy()
    zone_rcp[
        [
            "Type",
            "#",
            "None",
            "Minor",
            "Moderate",
            "Severe",
        ]
    ] = zone_rcp.apply(
        lambda row: get_stats(
            row, impact_assessment_statistics_road_closure_points, "Road Closure Points"
        ),
        axis=1,
    )

    zone_evac = zones.copy()
    zone_evac[
        [
            "Type",
            "#",
            "None",
            "Minor",
            "Moderate",
            "Severe",
        ]
    ] = zone_evac.apply(
        lambda row: get_stats(
            row, impact_assessment_statistics_evacuation_centre, "Evacuation Centre"
        ),
        axis=1,
    )

    zone_ca = zones.copy()
    zone_ca[
        [
            "Type",
            "#",
            "None",
            "Minor",
            "Moderate",
            "Severe",
        ]
    ] = zone_ca.apply(
        lambda row: get_stats(
            row, impact_assessment_statistics_council_assets, "Council Assets"
        ),
        axis=1,
    )
    zone_summary_stats = pd.concat(
        [zone_buildings, zone_rcp, zone_evac, zone_ca], axis=0
    )
    zone_summary_stats = zone_summary_stats.drop(["geometry", "OBJECTID"], axis=1)
    zone_summary_stats = zone_summary_stats.groupby(["Name", "Type"]).sum()

    return zone_summary_stats


def create_asset_breakdown_table(impact_vector_path, zone_file_path, layer_name, type):
    impact_assessment_statistics = gpd.read_file(impact_vector_path, layer=layer_name)

    if layer_name == "Evacuation Centre":
        pass
    elif layer_name == "Council Assets":
        impact_assessment_statistics = impact_assessment_statistics.rename(
            columns={"Comment": "Name"}
        )
    elif layer_name == "Road Closure Points":
        impact_assessment_statistics = impact_assessment_statistics.rename(
            columns={"LOCATION": "Name"}
        )
    else:
        pass

    regions = gpd.read_file(zone_file_path)
    suburbs = regions.loc[regions["Type"] == "Suburb"]

    if "Name" not in impact_assessment_statistics.columns:
        impact_assessment_statistics["Name"] = impact_assessment_statistics.apply(
            lambda row: f"Name_{row.name}", axis=1
        )

    impact_assessment_statistics["suburb_zone"] = impact_assessment_statistics.apply(
        lambda row: suburbs.loc[
            suburbs.geometry.intersects(row.geometry.centroid), "Name"
        ].iloc[0],
        axis=1,
    )

    def assign_impact_category(row):
        vulnerability_class = row.vulnerability_class
        if vulnerability_class == 1:
            return "None"
        elif vulnerability_class == 2:
            return "Minor"
        elif vulnerability_class == 3:
            return "Moderate"
        elif vulnerability_class == 4:
            return "Severe"
        else:
            return None

    impact_assessment_statistics["max_depth"] = impact_assessment_statistics.apply(
        lambda row: ("N/A" if row.max_depth == -9999 else str(round(row.max_depth, 2))),
        axis=1,
    )
    impact_assessment_statistics["Impact"] = impact_assessment_statistics.apply(
        lambda row: assign_impact_category(row), axis=1
    )
    impact_assessment_statistics = impact_assessment_statistics.drop(
        ["geometry", "vulnerability_class"], axis=1
    )
    return impact_assessment_statistics


def create_building_table(impact_vector_path, zone_file_path, layer_name):
    impact_assessment_statistics = gpd.read_file(impact_vector_path, layer=layer_name)
    regions = gpd.read_file(zone_file_path)

    impact_assessment_statistics = impact_assessment_statistics.rename(
        columns={"_StrucType": "Type"}
    )
    suburbs = regions.loc[regions["Type"] == "Suburb"]

    impact_assessment_statistics["Suburb"] = impact_assessment_statistics.apply(
        lambda row: suburbs.loc[
            suburbs.geometry.intersects(row.geometry.centroid), "Name"
        ].iloc[0],
        axis=1,
    )

    def assign_impact_category(row):
        vulnerability_class = row.vulnerability_class
        if vulnerability_class == 1:
            return pd.Series([1, 0, 0, 0])
        elif vulnerability_class == 2:
            return pd.Series([0, 1, 0, 0])
        elif vulnerability_class == 3:
            return pd.Series([0, 0, 1, 0])
        elif vulnerability_class == 4:
            return pd.Series([0, 0, 0, 1])
        else:
            return None

    impact_assessment_statistics["max_depth"] = impact_assessment_statistics.apply(
        lambda row: ("N/A" if row.max_depth == -9999 else str(round(row.max_depth, 2))),
        axis=1,
    )
    impact_assessment_statistics[["None", "Minor", "Moderate", "Severe"]] = (
        impact_assessment_statistics.apply(
            lambda row: assign_impact_category(row), axis=1
        )
    )

    impact_assessment_statistics = impact_assessment_statistics.groupby(
        ["Suburb", "Type"]
    )[["None", "Minor", "Moderate", "Severe"]].sum()

    suburbs = impact_assessment_statistics.index.get_level_values("Suburb")
    res_type = impact_assessment_statistics.index.get_level_values("Type")

    appendages = (
        []
    )  # not all suburbs have all types of buildings. To keep uniformity in the layout, the following code completes the categories with zero filled values

    for unique_suburb in np.unique(suburbs):
        subset = impact_assessment_statistics[
            impact_assessment_statistics.index.get_level_values("Suburb")
            == unique_suburb
        ]
        res_type = subset.index.get_level_values("Type")

        expected_types = ["Commercial", "Residential", "Industrial"]

        for et in expected_types:
            if et not in res_type:
                appendages.append(pd.Series([0, 0, 0, 0], name=(unique_suburb, et)))

    missing_categories = pd.concat(appendages, axis=1).T
    missing_categories.columns = ["None", "Minor", "Moderate", "Severe"]

    impact_assessment_statistics = pd.concat(
        [impact_assessment_statistics, missing_categories], axis=0
    )
    impact_assessment_statistics = impact_assessment_statistics.sort_index(level=0)

    # for i, row in impact_assessment_statistics.iterrows():

    return impact_assessment_statistics


def create_impact_guide_table():
    buildings_impact_guide = pd.DataFrame(
        data={
            "Waterdepth over floor level [m]": [
                "> 1.0",
                "0.25 - 1.0",
                "0.0 - 0.25",
                "< 0",
            ],
            "Impact Category": ["Severe", "Moderate", "Minor", "None"],
        }
    )

    roads_impact_guide = pd.DataFrame(
        data={
            "Waterdepth over road level [m]": [
                "> 0.15",
                "0 - 0.15",
                "-0.3 - 0",
                "< -0.3",
            ],
            "Impact Category": ["Severe", "Moderate", "Minor", "None"],
        }
    )

    council_services_impact_guide = pd.DataFrame(
        data={
            "Waterdepth over base level [m]": [
                "> 0.15",
                "0 - 0.15",
                "-0.3 - 0",
                "< -0.3",
            ],
            "Impact Category": ["Severe", "Moderate", "Minor", "None"],
        }
    )

    evac_centers_impact_guide = pd.DataFrame(
        data={
            "Waterdepth over floor level [m]": [
                "> 0.15",
                "0 - 0.15",
                "-0.3 - 0",
                "< -0.3",
            ],
            "Impact Category": ["Severe", "Moderate", "Minor", "None"],
        }
    )

    return (
        buildings_impact_guide,
        roads_impact_guide,
        council_services_impact_guide,
        evac_centers_impact_guide,
    )


class PdfFile(FPDF):
    def __init__(self):
        super().__init__()
        self.alias_nb_pages()
        self.contents = {}

    def initiate_layout(self):
        self.set_auto_page_break(False, margin=0)
        self.font_name = "Helvetica"
        self.add_page()
        self.set_font(self.font_name, size=6)

    def header(self):
        # Select Arial bold 15
        self.set_font("helvetica", "B", 20)
        # Move to the right
        self.add_text(
            x=3,
            y=10,
            alignment="L",
            text="Townsville Impact Overview",
            fontsize=18,
            bold=False,
            color=(15, 24, 96),
        )
        self.set_xy(x=175, y=0)
        self.image(
            r"d:\Royal HaskoningDHV\P-PA3396-Townsville-FLASH - WIP\python\impact_module_update\tville_logo.png",
            link="",
            type="png",
            w=225 / 7,
            h=225 / 7 * 1,
        )
        self.line(3, 34, 200, 34)
        self.line(3, 40, 200, 40)
        self.ln(20)

    def footer(self):
        self.ln(20)
        self.line(3, 280, 200, 280)
        self.add_text(
            x=3,
            y=275,
            alignment="C",
            text="Page %s" % self.page_no(),
            fontsize=8,
            color=(15, 24, 96),
        )

    def add_content_line(self):
        for page in range(1, self.pages_count + 1):
            x = 3

            self.page = page
            if len(self.contents) > 0:
                for i, (txt, lnk) in enumerate(self.contents.items()):
                    self.set_font(self.font_name, size=6)
                    self.set_font_size(6)  # (self.font_name, size=6)
                    self.add_text(
                        x=x,
                        y=27,
                        alignment="L",
                        text=txt,
                        fontsize=6,
                        bold=False,
                        color=(15, 24, 96),
                        link=lnk,
                    )
                    x += len(txt) * 1.3 + 2

    def add_text(
        self, x, y, alignment, text, fontsize, bold=False, color=(0, 0, 0), link=""
    ):
        self.set_xy(x, y)
        if bold:
            self.set_font("helvetica", "B", fontsize)
        else:
            self.set_font("helvetica", "", fontsize)
        self.set_text_color(*color)
        self.cell(w=210.0, h=20.0, align=alignment, text=text, border=0, link=link)

    def add_table(
        self,
        x,
        y,
        table_data,
        title="",
        data_size=6,
        title_size=8,
        align_data="L",
        align_header="L",
        cell_width="even",
        x_start="x_default",
        emphasize_data=[],
        emphasize_style="",
        emphasize_color=(0, 0, 0),
        conditional_formatting=False,
    ):
        """
        table_data:
                    list of lists with first element being list of headers
        title:
                    (Optional) title of table (optional)
        data_size:
                    the font size of table data
        title_size:
                    the font size fo the title of the table
        align_data:
                    align table data
                    L = left align
                    C = center align
                    R = right align
        align_header:
                    align table data
                    L = left align
                    C = center align
                    R = right align
        cell_width:
                    even: evenly distribute cell/column width
                    uneven: base cell size on lenght of cell/column items
                    int: int value for width of each cell/column
                    list of ints: list equal to number of columns with the widht of each cell / column
        x_start:
                    where the left edge of table should start
        emphasize_data:
                    which data elements are to be emphasized - pass as list
                    emphasize_style: the font style you want emphaized data to take
                    emphasize_color: emphasize color (if other than black)

        """
        self.set_xy(x, y)
        default_style = self.font_style

        if emphasize_style is None:
            emphasize_style = default_style

        # default_font = self.font_family
        # default_size = self.font_size_pt
        # default_style = self.font_style
        # default_color = self.color # This does not work

        # Convert dict to lol
        # Why? because i built it with lol first and added dict func after
        # Is there performance differences?

        if isinstance(table_data, dict):
            header = [key for key in table_data]
            data = []
            for key in table_data:
                value = table_data[key].values()
                data.append(value)
            # need to zip so data is in correct format (first, second, third --> not first, first, first)
            data = [list(a) for a in zip(*data)]

        else:
            header = table_data[0]
            data = table_data[1:]

        # Get Width of Columns
        def get_col_widths():
            col_width = cell_width
            if col_width == "even":
                col_width = (self.epw / 5) / len(data[0]) - 1
            elif col_width == "uneven":
                col_widths = []

                # searching through columns for largest sized cell (not rows but cols)
                for col in range(len(table_data[0])):  # for every row
                    longest = 0
                    for row in range(len(table_data)):
                        cell_value = str(table_data[row][col])
                        value_length = self.get_string_width(cell_value)
                        if value_length > longest:
                            longest = value_length
                    col_widths.append(longest + 4)  # add 4 for padding
                col_width = col_widths

                # compare columns

            elif isinstance(cell_width, list):
                col_width = cell_width  # TODO: convert all items in list to int
            else:
                # TODO: Add try catch
                col_width = int(col_width)
            return col_width

        line_height = 2.5

        col_width = get_col_widths()

        self.set_font(self.font_name, size=title_size, style="B")

        # Get starting position of x
        # Determine width of table to get x starting point for centred table
        if x_start == "C":
            table_width = 0
            if isinstance(col_width, list):
                for width in col_width:
                    table_width += width
            else:  # need to multiply cell width by number of cells to get table width
                table_width = col_width * len(table_data[0])
            # Get x start by subtracting table width from pdf width and divide by 2 (margins)
            margin_width = self.w - table_width
            # TODO: Check if table_width is larger than pdf width

            center_table = margin_width / 2  # only want width of left margin not both
            x_start = center_table
            self.set_x(x_start)
        elif isinstance(x_start, int):
            self.set_x(x_start)
        elif x_start == "x_default":
            x_start = self.set_x(self.l_margin)

        # TABLE CREATION #

        # add title
        if title != "":
            self.multi_cell(
                0,
                line_height,
                title,
                border=0,
                align="l",
                ln=3,
                max_line_height=self.font_size,
            )
            self.ln(2.1 * line_height)  # move cursor back to the left margin

        self.set_font(self.font_name, size=data_size)

        # add header
        y1 = self.get_y()
        if x_start:
            x_left = x_start
        else:
            x_left = self.get_x()

        if not isinstance(col_width, list):
            if x_start:
                self.set_x(x_start)
            for datum in header:
                self.multi_cell(
                    col_width,
                    line_height,
                    str(datum),
                    border=0,
                    align="C",
                    ln=3,
                    max_line_height=self.font_size,
                )
                x_right = self.get_x()

            self.ln(line_height * 2.2)  # move cursor back to the left margin
            y2 = self.get_y()
            self.line(x_left, y1, x_right, y1)
            self.line(x_left, y2, x_right, y2)

            for row in data:
                if x_start:  # not sure if I need this
                    self.set_x(x_start)

                for datum in row:
                    if conditional_formatting:
                        if datum in emphasize_data:
                            self.set_text_color(*emphasize_color)
                            self.set_font(self.font_name, style=emphasize_style)
                            self.multi_cell(
                                col_width,
                                line_height,
                                datum,
                                border=0,
                                align=align_data,
                                ln=3,
                                max_line_height=self.font_size,
                            )
                            self.set_text_color(0, 0, 0)
                            self.set_font(self.font_name, style=default_style)
                        else:
                            if int(datum) < 0:
                                self.set_text_color(255, 0, 0)
                                self.multi_cell(
                                    col_width,
                                    line_height,
                                    datum,
                                    border=0,
                                    align=align_data,  # ln=3,
                                    max_line_height=self.font_size,
                                )  # ln = 3 - move cursor to right with same vertical offset # this uses an object named self
                            elif int(datum) == 0:
                                self.set_text_color(0, 0, 0)
                                self.multi_cell(
                                    col_width,
                                    line_height,
                                    datum,
                                    border=0,
                                    align=align_data,  # ln=3,
                                    max_line_height=self.font_size,
                                )  # ln = 3 - move cursor to right with same vertical offset # this uses an object named self
                            else:
                                self.set_text_color(50, 205, 50)
                                self.multi_cell(
                                    col_width,
                                    line_height,
                                    datum,
                                    border=0,
                                    align=align_data,
                                    ln=3,
                                    max_line_height=self.font_size,
                                )  # ln = 3 - move cursor to right with same vertical offset # this uses an object named self

                    else:
                        if datum in emphasize_data:
                            self.set_text_color(*emphasize_color)
                            self.set_font(self.font_name, style=emphasize_style)
                            self.multi_cell(
                                col_width,
                                line_height,
                                str(datum),
                                border=0,
                                align=align_data,
                                ln=3,
                                max_line_height=self.font_size,
                            )
                            self.set_text_color(0, 0, 0)
                        else:
                            self.multi_cell(
                                col_width,
                                line_height,
                                str(datum),
                                border=0,
                                align=align_data,
                                ln=3,
                                max_line_height=self.font_size,
                            )
                self.ln(line_height)  # move cursor back to the left margin

        else:
            if x_start:
                self.set_x(x_start)
            for i in range(len(header)):
                datum = header[i]
                self.multi_cell(
                    col_width[i],
                    line_height,
                    datum,
                    border=0,
                    align=align_header,
                    ln=3,
                    max_line_height=self.font_size,
                )
                x_right = self.get_x()
            self.ln(line_height)  # move cursor back to the left margin
            y2 = self.get_y()
            self.line(x_left, y1, x_right, y1)
            self.line(x_left, y2, x_right, y2)

            def is_number(n):
                try:
                    float(n)  # Type-casting the string to `float`.
                    # If string is not a valid `float`,
                    # it'll raise `ValueError` exception
                except ValueError:
                    return False
                return True

            for i in range(len(data)):
                if x_start:
                    self.set_x(x_start)
                row = data[i]
                for i in range(len(row)):
                    datum = row[i]
                    if not isinstance(datum, str):
                        datum = str(datum)
                    adjusted_col_width = col_width[i]
                    if conditional_formatting is True:
                        if datum in emphasize_data:
                            self.set_text_color(*emphasize_color)
                            self.set_font(self.font_name, style=emphasize_style)
                            self.multi_cell(
                                adjusted_col_width,
                                line_height,
                                datum,
                                border=0,
                                align=align_data,
                                ln=3,
                                max_line_height=self.font_size,
                            )
                            self.set_text_color(0, 0, 0)
                            self.set_font(self.font_name, style=default_style)
                        else:
                            if is_number(datum) is True:
                                if float(datum) < 0:
                                    self.set_text_color(255, 0, 0)
                                    self.multi_cell(
                                        adjusted_col_width,
                                        line_height,
                                        datum,
                                        border=0,
                                        align=align_data,
                                        ln=3,
                                        max_line_height=self.font_size,
                                    )  # ln = 3 - move cursor to right with same vertical offset # this uses an object named self
                                elif float(datum) == 0:
                                    self.set_text_color(0, 0, 0)
                                    self.multi_cell(
                                        adjusted_col_width,
                                        line_height,
                                        datum,
                                        border=0,
                                        align=align_data,
                                        ln=3,
                                        max_line_height=self.font_size,
                                    )  # ln = 3 - move cursor to right with same vertical offset # this uses an object named self
                                else:
                                    self.set_text_color(50, 205, 50)
                                    self.multi_cell(
                                        adjusted_col_width,
                                        line_height,
                                        datum,
                                        border=0,
                                        align=align_data,
                                        ln=3,
                                        max_line_height=self.font_size,
                                    )  # ln = 3 - move cursor to right with same vertical offset # this uses an object named self
                            else:
                                self.set_text_color(0, 0, 0)
                                self.multi_cell(
                                    adjusted_col_width,
                                    line_height,
                                    datum,
                                    border=0,
                                    align=align_data,
                                    ln=3,
                                    max_line_height=self.font_size,
                                )  # ln = 3 - move cursor to right with same vertical offset # this uses an object named self
                    else:
                        if datum in emphasize_data:
                            self.set_text_color(*emphasize_color)
                            self.set_font(self.font_name, style=emphasize_style)
                            self.multi_cell(
                                adjusted_col_width,
                                line_height,
                                datum,
                                border=0,
                                align=align_data,
                                ln=3,
                                max_line_height=self.font_size,
                            )
                            self.set_text_color(0, 0, 0)
                            self.set_font(self.font_name, style=default_style)
                        else:
                            self.multi_cell(
                                adjusted_col_width,
                                line_height,
                                datum,
                                border=0,
                                align=align_data,
                                ln=3,
                                max_line_height=self.font_size,
                            )  # ln = 3 - move cursor to right with same vertical offset # this uses an object named self
                self.ln(line_height)  # move cursor back to the left margin
        y3 = self.get_y()
        self.line(x_left, y3, x_right, y3)

    def add_main_summary_table(
        self,
        x,
        y,
        table_data,
        title="",
        data_size=6,
        title_size=8,
        align_data="L",
        align_header="L",
        cell_width="even",
        x_start="x_default",
        emphasize_data=[],
        emphasize_style="",
        emphasize_color=(0, 0, 0),
        conditional_formatting=False,
    ):
        """
        table_data:
                    list of lists with first element being list of headers
        title:
                    (Optional) title of table (optional)
        data_size:
                    the font size of table data
        title_size:
                    the font size fo the title of the table
        align_data:
                    align table data
                    L = left align
                    C = center align
                    R = right align
        align_header:
                    align table data
                    L = left align
                    C = center align
                    R = right align
        cell_width:
                    even: evenly distribute cell/column width
                    uneven: base cell size on lenght of cell/column items
                    int: int value for width of each cell/column
                    list of ints: list equal to number of columns with the widht of each cell / column
        x_start:
                    where the left edge of table should start
        emphasize_data:
                    which data elements are to be emphasized - pass as list
                    emphasize_style: the font style you want emphaized data to take
                    emphasize_color: emphasize color (if other than black)

        """
        self.set_xy(x, y)
        default_style = self.font_style

        if emphasize_style is None:
            emphasize_style = default_style

        # default_font = self.font_family
        # default_size = self.font_size_pt
        # default_style = self.font_style
        # default_color = self.color # This does not work

        # Convert dict to lol
        # Why? because i built it with lol first and added dict func after
        # Is there performance differences?

        if isinstance(table_data, dict):
            header = [key for key in table_data]
            data = []
            for key in table_data:
                value = table_data[key].values()
                data.append(value)
            # need to zip so data is in correct format (first, second, third --> not first, first, first)
            data = [list(a) for a in zip(*data)]

        else:
            header = table_data[0]
            data = table_data[1:]

        # Get Width of Columns
        def get_col_widths():
            col_width = cell_width
            if col_width == "even":
                col_width = (self.epw / 5) / len(data[0]) - 1
            elif col_width == "uneven":
                col_widths = []

                # searching through columns for largest sized cell (not rows but cols)
                for col in range(len(table_data[0])):  # for every row
                    longest = 0
                    for row in range(len(table_data)):
                        cell_value = str(table_data[row][col])
                        value_length = self.get_string_width(cell_value)
                        if value_length > longest:
                            longest = value_length
                    col_widths.append(longest + 4)  # add 4 for padding
                col_width = col_widths

                # compare columns

            elif isinstance(cell_width, list):
                col_width = cell_width  # TODO: convert all items in list to int
            else:
                # TODO: Add try catch
                col_width = int(col_width)
            return col_width

        line_height = 2.5

        col_width = get_col_widths()

        self.set_font(self.font_name, size=title_size, style="B")

        # Get starting position of x
        # Determine width of table to get x starting point for centred table
        if x_start == "C":
            table_width = 0
            if isinstance(col_width, list):
                for width in col_width:
                    table_width += width
            else:  # need to multiply cell width by number of cells to get table width
                table_width = col_width * len(table_data[0])
            # Get x start by subtracting table width from pdf width and divide by 2 (margins)
            margin_width = self.w - table_width
            # TODO: Check if table_width is larger than pdf width

            center_table = margin_width / 2  # only want width of left margin not both
            x_start = center_table
            self.set_x(x_start)
        elif isinstance(x_start, int):
            self.set_x(x_start)
        elif x_start == "x_default":
            x_start = self.set_x(self.l_margin)

        # TABLE CREATION #

        # add title
        if title != "":
            self.multi_cell(
                0,
                line_height,
                title,
                border=0,
                align="l",
                ln=3,
                max_line_height=self.font_size,
            )
            self.ln(2.1 * line_height)  # move cursor back to the left margin

        self.set_font(self.font_name, size=data_size)

        # add header
        y1 = self.get_y()
        if x_start:
            x_left = x_start
        else:
            x_left = self.get_x()

        if not isinstance(col_width, list):
            if x_start:
                self.set_x(x_start)
            for datum in header:
                self.multi_cell(
                    col_width,
                    line_height,
                    str(datum),
                    border=0,
                    align="C",
                    ln=3,
                    max_line_height=self.font_size,
                )
                x_right = self.get_x()

            self.ln(line_height * 2.2)  # move cursor back to the left margin
            y2 = self.get_y()
            self.line(x_left, y1, x_right, y1)
            self.line(x_left, y2, x_right, y2)

            for row in data:
                if x_start:  # not sure if I need this
                    self.set_x(x_start)

                for datum in row:
                    if conditional_formatting:
                        if datum in emphasize_data:
                            self.set_text_color(*emphasize_color)
                            self.set_font(self.font_name, style=emphasize_style)
                            self.multi_cell(
                                col_width,
                                line_height,
                                datum,
                                border=0,
                                align=align_data,
                                ln=3,
                                max_line_height=self.font_size,
                            )
                            self.set_text_color(0, 0, 0)
                            self.set_font(self.font_name, style=default_style)
                        else:
                            if int(datum) < 0:
                                self.set_text_color(255, 0, 0)
                                self.multi_cell(
                                    col_width,
                                    line_height,
                                    datum,
                                    border=0,
                                    align=align_data,  # ln=3,
                                    max_line_height=self.font_size,
                                )  # ln = 3 - move cursor to right with same vertical offset # this uses an object named self
                            elif int(datum) == 0:
                                self.set_text_color(0, 0, 0)
                                self.multi_cell(
                                    col_width,
                                    line_height,
                                    datum,
                                    border=0,
                                    align=align_data,  # ln=3,
                                    max_line_height=self.font_size,
                                )  # ln = 3 - move cursor to right with same vertical offset # this uses an object named self
                            else:
                                self.set_text_color(50, 205, 50)
                                self.multi_cell(
                                    col_width,
                                    line_height,
                                    datum,
                                    border=0,
                                    align=align_data,
                                    ln=3,
                                    max_line_height=self.font_size,
                                )  # ln = 3 - move cursor to right with same vertical offset # this uses an object named self

                    else:
                        if datum in emphasize_data:
                            self.set_text_color(*emphasize_color)
                            self.set_font(self.font_name, style=emphasize_style)
                            self.multi_cell(
                                col_width,
                                line_height,
                                str(datum),
                                border=0,
                                align=align_data,
                                ln=3,
                                max_line_height=self.font_size,
                            )
                            self.set_text_color(0, 0, 0)
                        else:
                            self.multi_cell(
                                col_width,
                                line_height,
                                str(datum),
                                border=0,
                                align=align_data,
                                ln=3,
                                max_line_height=self.font_size,
                            )
                self.ln(line_height)  # move cursor back to the left margin

        else:
            if x_start:
                self.set_x(x_start)

            for i in range(len(header)):
                datum = header[i]

                self.multi_cell(
                    col_width[i],
                    line_height,
                    datum,
                    border=0,
                    align=align_header,
                    ln=3,
                    max_line_height=self.font_size,
                )
                x_right = self.get_x()

            self.ln(line_height)  # move cursor back to the left margin
            y2 = self.get_y()
            self.line(x_left, y1, x_right, y1)
            self.line(x_left, y2, x_right, y2)

            def is_number(n):
                try:
                    float(n)  # Type-casting the string to `float`.
                    # If string is not a valid `float`,
                    # it'll raise `ValueError` exception
                except ValueError:
                    return False
                return True

            for i in range(len(data)):
                if x_start:
                    self.set_x(x_start)
                row = data[i]
                if i / 4 - int(i / 4) == 0:
                    self.line(x_left, self.get_y(), x_right, self.get_y())

                for j in range(len(row)):
                    datum = row[j]
                    if not isinstance(datum, str):
                        datum = str(datum)
                    adjusted_col_width = col_width[j]
                    conditional_formatting = True
                    if j > 2 and row[j] > 0:
                        if j == 3:
                            self.set_fill_color(GREEN_RGB)
                        elif j == 4:
                            self.set_fill_color(BLUE_RGB)
                        elif j == 5:
                            self.set_fill_color(YELLOW_RGB)
                        elif j == 6:
                            self.set_fill_color(RED_RGB)
                        else:
                            pass
                        self.set_font(self.font_name, style="")
                        self.multi_cell(
                            adjusted_col_width,
                            line_height,
                            datum,
                            border=0,
                            fill=True,
                            align=align_data,
                            ln=3,
                            max_line_height=self.font_size,
                        )
                        self.set_text_color(0, 0, 0)
                        self.set_fill_color(0, 0, 0)
                        self.set_font(self.font_name, style="")
                    else:
                        self.set_text_color(0, 0, 0)
                        self.set_fill_color(0, 0, 0)
                        self.multi_cell(
                            adjusted_col_width,
                            line_height,
                            datum,
                            border=0,
                            align=align_data,
                            ln=3,
                            max_line_height=self.font_size,
                        )  # ln = 3 - move cursor to right with same vertical offset # this uses an object named self

                self.ln(line_height)  # move cursor back to the left margin
        y3 = self.get_y()
        self.line(x_left, y3, x_right, y3)

    def add_asset_table(self, x, y, table_data, emphasize_column, title):

        self.set_font(self.font_name, size=6)

        table_data_processed = [list(table_data.columns)]

        for _, row in table_data.iterrows():
            table_data_processed.append(row.values)

        blue = (0, 0, 0)
        grey = (200, 200, 200)

        self.set_xy(x, y)

        if x < 80:
            align = "LEFT"
        else:
            align = "RIGHT"

        self.set_font(self.font_name, size=8, style="B")
        self.multi_cell(
            0,
            2.5,
            title,
            border=0,
            align="l",
            ln=3,
            max_line_height=self.font_size,
        )

        self.set_xy(x, y + 5)

        self.set_font(self.font_name, size=6, style="")

        headings_style = FontFace(emphasis="BOLD", color=blue, fill_color=grey)
        with self.table(
            line_height=self.font_size,
            width=90,
            col_widths=(40, 20, 10, 10),
            headings_style=headings_style,
            text_align="LEFT",
            padding=1,
            align=align,
        ) as table:
            for data_row in table_data_processed:
                row = table.row()
                for i, datum in enumerate(data_row):
                    if i == emphasize_column:
                        if datum == "Severe":
                            style = FontFace(fill_color=RED_RGB)
                        elif datum == "Moderate":
                            style = FontFace(fill_color=YELLOW_RGB)
                        elif datum == "Minor":
                            style = FontFace(fill_color=BLUE_RGB)
                        elif datum == "None":
                            style = FontFace(fill_color=GREEN_RGB)
                        else:
                            style = None

                        row.cell(datum, style=style)
                    else:
                        row.cell(datum)

    def add_building_asset_table(self, x, y, table_data):

        self.set_font(self.font_name, size=6)

        table_data_processed = [["Suburb", "Type"] + list(table_data.columns)]

        for i, (_, row) in enumerate(table_data.iterrows()):
            data = [row.name[0], row.name[1]] + list(row.values)
            table_data_processed.append(data)

        blue = (0, 0, 0)
        grey = (200, 200, 200)

        self.set_xy(x, y)

        if x < 80:
            align = "LEFT"
        else:
            align = "RIGHT"

        headings_style = FontFace(emphasis="BOLD", color=blue, fill_color=grey)

        with self.table(
            line_height=self.font_size,
            width=90,
            col_widths=(20, 20, 10, 10, 13, 10),
            headings_style=headings_style,
            text_align="LEFT",
            padding=1,
            align=align,
        ) as table:
            for r_no, data_row in enumerate(table_data_processed):
                row = table.row()

                for i, datum in enumerate(data_row):
                    if i == 0:
                        if r_no > 1:
                            previous_suburb_name = table_data_processed[r_no - 1][i]
                            suburb_name = table_data_processed[r_no][i]
                            if previous_suburb_name != suburb_name:
                                row.cell(str(datum), rowspan=3)
                        elif r_no == 1:
                            row.cell(str(datum), rowspan=3)
                        else:
                            row.cell(str(datum))
                    else:
                        if i == 2 and r_no > 0 and datum > 0:
                            style = FontFace(fill_color=GREEN_RGB)
                        elif i == 3 and r_no > 0 and datum > 0:
                            style = FontFace(fill_color=BLUE_RGB)
                        elif i == 4 and r_no > 0 and datum > 0:
                            style = FontFace(fill_color=YELLOW_RGB)
                        elif i == 5 and r_no > 0 and datum > 0:
                            style = FontFace(fill_color=RED_RGB)
                        else:
                            style = None

                        row.cell(str(datum), style=style)

    def add_asset_table_old(
        self,
        x,
        y,
        table_data,
        title="",
        data_size=6,
        title_size=8,
        align_data="L",
        align_header="L",
        cell_width="even",
        x_start="x_default",
        emphasize_data=[],
        emphasize_style="",
        emphasize_color=(0, 0, 0),
        conditional_formatting=False,
        column_to_format=2,
    ):
        """
        table_data:
                    list of lists with first element being list of headers
        title:
                    (Optional) title of table (optional)
        data_size:
                    the font size of table data
        title_size:
                    the font size fo the title of the table
        align_data:
                    align table data
                    L = left align
                    C = center align
                    R = right align
        align_header:
                    align table data
                    L = left align
                    C = center align
                    R = right align
        cell_width:
                    even: evenly distribute cell/column width
                    uneven: base cell size on lenght of cell/column items
                    int: int value for width of each cell/column
                    list of ints: list equal to number of columns with the widht of each cell / column
        x_start:
                    where the left edge of table should start
        emphasize_data:
                    which data elements are to be emphasized - pass as list
                    emphasize_style: the font style you want emphaized data to take
                    emphasize_color: emphasize color (if other than black)

        """
        self.set_xy(x, y)
        default_style = self.font_style

        if emphasize_style is None:
            emphasize_style = default_style

        if isinstance(table_data, dict):
            header = [key for key in table_data]
            data = []
            for key in table_data:
                value = table_data[key].values()
                data.append(value)
            # need to zip so data is in correct format (first, second, third --> not first, first, first)
            data = [list(a) for a in zip(*data)]

        else:
            header = table_data[0]
            data = table_data[1:]

        # Get Width of Columns
        def get_col_widths():
            col_width = cell_width
            if col_width == "even":
                col_width = (self.epw / 5) / len(data[0]) - 1
            elif col_width == "uneven":
                col_widths = []

                # searching through columns for largest sized cell (not rows but cols)
                for col in range(len(table_data[0])):  # for every row
                    longest = 0
                    for row in range(len(table_data)):
                        cell_value = str(table_data[row][col])
                        value_length = self.get_string_width(cell_value)
                        if value_length > longest:
                            longest = value_length
                    col_widths.append(longest + 4)  # add 4 for padding
                col_width = col_widths

                # compare columns

            elif isinstance(cell_width, list):
                col_width = cell_width  # TODO: convert all items in list to int
            else:
                # TODO: Add try catch
                col_width = int(col_width)
            return col_width

        line_height = 2.5

        col_width = get_col_widths()

        self.set_font(self.font_name, size=title_size, style="B")

        # Get starting position of x
        # Determine width of table to get x starting point for centred table
        if x_start == "C":
            table_width = 0
            if isinstance(col_width, list):
                for width in col_width:
                    table_width += width
            else:  # need to multiply cell width by number of cells to get table width
                table_width = col_width * len(table_data[0])
            # Get x start by subtracting table width from pdf width and divide by 2 (margins)
            margin_width = self.w - table_width
            # TODO: Check if table_width is larger than pdf width

            center_table = margin_width / 2  # only want width of left margin not both
            x_start = center_table
            self.set_x(x_start)
        elif isinstance(x_start, int):
            self.set_x(x_start)
        elif x_start == "x_default":
            x_start = self.set_x(self.l_margin)

        # TABLE CREATION #

        # add title
        if title != "":
            self.multi_cell(
                0,
                line_height,
                title,
                border=0,
                align="l",
                ln=3,
                max_line_height=self.font_size,
            )
            self.ln(2.1 * line_height)  # move cursor back to the left margin

        self.set_font(self.font_name, size=data_size)

        # add header
        y1 = self.get_y()
        if x_start:
            x_left = x_start
        else:
            x_left = self.get_x()

        if not isinstance(col_width, list):
            if x_start:
                self.set_x(x_start)
            for datum in header:
                self.multi_cell(
                    col_width,
                    line_height,
                    str(datum),
                    border=0,
                    align="C",
                    ln=3,
                    max_line_height=self.font_size,
                )
                x_right = self.get_x()

            self.ln(line_height * 2.2)  # move cursor back to the left margin
            y2 = self.get_y()
            self.line(x_left, y1, x_right, y1)
            self.line(x_left, y2, x_right, y2)

            for row in data:
                if x_start:  # not sure if I need this
                    self.set_x(x_start)

                for datum in row:
                    if conditional_formatting:
                        if datum in emphasize_data:
                            self.set_text_color(*emphasize_color)
                            self.set_font(self.font_name, style=emphasize_style)
                            self.multi_cell(
                                col_width,
                                line_height,
                                datum,
                                border=0,
                                align=align_data,
                                ln=3,
                                max_line_height=self.font_size,
                            )
                            self.set_text_color(0, 0, 0)
                            self.set_font(self.font_name, style=default_style)
                        else:
                            if int(datum) < 0:
                                self.set_text_color(255, 0, 0)
                                self.multi_cell(
                                    col_width,
                                    line_height,
                                    datum,
                                    border=0,
                                    align=align_data,  # ln=3,
                                    max_line_height=self.font_size,
                                )  # ln = 3 - move cursor to right with same vertical offset # this uses an object named self
                            elif int(datum) == 0:
                                self.set_text_color(0, 0, 0)
                                self.multi_cell(
                                    col_width,
                                    line_height,
                                    datum,
                                    border=0,
                                    align=align_data,  # ln=3,
                                    max_line_height=self.font_size,
                                )  # ln = 3 - move cursor to right with same vertical offset # this uses an object named self
                            else:
                                self.set_text_color(50, 205, 50)
                                self.multi_cell(
                                    col_width,
                                    line_height,
                                    datum,
                                    border=0,
                                    align=align_data,
                                    ln=3,
                                    max_line_height=self.font_size,
                                )  # ln = 3 - move cursor to right with same vertical offset # this uses an object named self

                    else:
                        if datum in emphasize_data:
                            self.set_text_color(*emphasize_color)
                            self.set_font(self.font_name, style=emphasize_style)
                            self.multi_cell(
                                col_width,
                                line_height,
                                str(datum),
                                border=0,
                                align=align_data,
                                ln=3,
                                max_line_height=self.font_size,
                            )
                            self.set_text_color(0, 0, 0)
                        else:
                            self.multi_cell(
                                col_width,
                                line_height,
                                str(datum),
                                border=0,
                                align=align_data,
                                ln=3,
                                max_line_height=self.font_size,
                            )
                self.ln(line_height)  # move cursor back to the left margin

        else:
            if x_start:
                self.set_x(x_start)
            for i in range(len(header)):
                datum = header[i]
                self.multi_cell(
                    col_width[i],
                    line_height,
                    datum,
                    new_x=XPos.RIGHT,
                    new_y=YPos.TOP,
                    border=0,
                    align=align_header,
                    max_line_height=self.font_size,
                )
                x_right = self.get_x()

            self.ln(line_height)  # move cursor back to the left margin
            y2 = self.get_y()
            self.line(x_left, y1, x_right, y1)
            self.line(x_left, y2, x_right, y2)

            def is_number(n):
                try:
                    float(n)  # Type-casting the string to `float`.
                    # If string is not a valid `float`,
                    # it'll raise `ValueError` exception
                except ValueError:
                    return False
                return True

            for i in range(len(data)):
                if x_start:
                    self.set_x(x_start)
                row = data[i]
                for j in range(len(row)):
                    datum = row[j]
                    if not isinstance(datum, str):
                        datum = str(datum)
                    adjusted_col_width = col_width[j]

                    if j == column_to_format:

                        if row[j] == "None":
                            self.set_fill_color(GREEN_RGB)
                        elif row[j] == "Minor":
                            self.set_fill_color(BLUE_RGB)
                        elif row[j] == "Moderate":
                            self.set_fill_color(YELLOW_RGB)
                        elif row[j] == "Severe":
                            self.set_fill_color(RED_RGB)
                        else:
                            self.set_fill_color(0, 0, 0)

                        self.multi_cell(
                            w=adjusted_col_width,
                            h=None,
                            text=datum,
                            new_x=XPos.RIGHT,
                            new_y=YPos.TOP,
                            border=0,
                            fill=True,
                            align=align_data,
                            max_line_height=self.font_size,
                        )
                    else:
                        self.multi_cell(
                            w=adjusted_col_width,
                            h=None,
                            text=datum,
                            new_x=XPos.RIGHT,
                            new_y=YPos.TOP,
                            border=0,
                            align=align_data,
                            max_line_height=self.font_size,
                        )
                self.ln(line_height)
        y3 = self.get_y()
        self.line(x_left, y3, x_right, y3)

    def add_impact_guide_table(
        self,
        x,
        y,
        table_data,
        title="",
        data_size=6,
        title_size=8,
        align_data="L",
        align_header="L",
        cell_width="even",
        x_start="x_default",
        emphasize_data=[],
        emphasize_style="",
        emphasize_color=(0, 0, 0),
        conditional_formatting=False,
    ):
        """
        table_data:
                    list of lists with first element being list of headers
        title:
                    (Optional) title of table (optional)
        data_size:
                    the font size of table data
        title_size:
                    the font size fo the title of the table
        align_data:
                    align table data
                    L = left align
                    C = center align
                    R = right align
        align_header:
                    align table data
                    L = left align
                    C = center align
                    R = right align
        cell_width:
                    even: evenly distribute cell/column width
                    uneven: base cell size on lenght of cell/column items
                    int: int value for width of each cell/column
                    list of ints: list equal to number of columns with the widht of each cell / column
        x_start:
                    where the left edge of table should start
        emphasize_data:
                    which data elements are to be emphasized - pass as list
                    emphasize_style: the font style you want emphaized data to take
                    emphasize_color: emphasize color (if other than black)

        """
        self.set_xy(x, y)
        default_style = self.font_style

        if emphasize_style is None:
            emphasize_style = default_style

        if isinstance(table_data, dict):
            header = [key for key in table_data]
            data = []
            for key in table_data:
                value = table_data[key].values()
                data.append(value)
            # need to zip so data is in correct format (first, second, third --> not first, first, first)
            data = [list(a) for a in zip(*data)]

        else:
            header = table_data[0]
            data = table_data[1:]

        # Get Width of Columns
        def get_col_widths():
            col_width = cell_width
            if col_width == "even":
                col_width = (self.epw / 5) / len(data[0]) - 1
            elif col_width == "uneven":
                col_widths = []

                # searching through columns for largest sized cell (not rows but cols)
                for col in range(len(table_data[0])):  # for every row
                    longest = 0
                    for row in range(len(table_data)):
                        cell_value = str(table_data[row][col])
                        value_length = self.get_string_width(cell_value)
                        if value_length > longest:
                            longest = value_length
                    col_widths.append(longest + 4)  # add 4 for padding
                col_width = col_widths

                # compare columns

            elif isinstance(cell_width, list):
                col_width = cell_width  # TODO: convert all items in list to int
            else:
                # TODO: Add try catch
                col_width = int(col_width)
            return col_width

        line_height = 2.5

        col_width = get_col_widths()

        self.set_font(self.font_name, size=title_size, style="B")

        # Get starting position of x
        # Determine width of table to get x starting point for centred table
        if x_start == "C":
            table_width = 0
            if isinstance(col_width, list):
                for width in col_width:
                    table_width += width
            else:  # need to multiply cell width by number of cells to get table width
                table_width = col_width * len(table_data[0])
            # Get x start by subtracting table width from pdf width and divide by 2 (margins)
            margin_width = self.w - table_width
            # TODO: Check if table_width is larger than pdf width

            center_table = margin_width / 2  # only want width of left margin not both
            x_start = center_table
            self.set_x(x_start)
        elif isinstance(x_start, int):
            self.set_x(x_start)
        elif x_start == "x_default":
            x_start = self.set_x(self.l_margin)

        # TABLE CREATION #

        # add title
        if title != "":
            self.multi_cell(
                0,
                line_height,
                title,
                border=0,
                align="l",
                ln=3,
                max_line_height=self.font_size,
            )
            self.ln(2.1 * line_height)  # move cursor back to the left margin

        self.set_font(self.font_name, size=data_size)

        # add header
        y1 = self.get_y()
        if x_start:
            x_left = x_start
        else:
            x_left = self.get_x()

        if not isinstance(col_width, list):
            if x_start:
                self.set_x(x_start)
            for datum in header:
                self.multi_cell(
                    col_width,
                    line_height,
                    str(datum),
                    border=0,
                    align="C",
                    ln=3,
                    max_line_height=self.font_size,
                )
                x_right = self.get_x()

            self.ln(line_height * 2.2)  # move cursor back to the left margin
            y2 = self.get_y()
            self.line(x_left, y1, x_right, y1)
            self.line(x_left, y2, x_right, y2)

            for row in data:
                if x_start:  # not sure if I need this
                    self.set_x(x_start)

                for datum in row:
                    if conditional_formatting:
                        if datum in emphasize_data:
                            self.set_text_color(*emphasize_color)
                            self.set_font(self.font_name, style=emphasize_style)
                            self.multi_cell(
                                col_width,
                                line_height,
                                datum,
                                border=0,
                                align=align_data,
                                ln=3,
                                max_line_height=self.font_size,
                            )
                            self.set_text_color(0, 0, 0)
                            self.set_font(self.font_name, style=default_style)
                        else:
                            if int(datum) < 0:
                                self.set_text_color(255, 0, 0)
                                self.multi_cell(
                                    col_width,
                                    line_height,
                                    datum,
                                    border=0,
                                    align=align_data,  # ln=3,
                                    max_line_height=self.font_size,
                                )  # ln = 3 - move cursor to right with same vertical offset # this uses an object named self
                            elif int(datum) == 0:
                                self.set_text_color(0, 0, 0)
                                self.multi_cell(
                                    col_width,
                                    line_height,
                                    datum,
                                    border=0,
                                    align=align_data,  # ln=3,
                                    max_line_height=self.font_size,
                                )  # ln = 3 - move cursor to right with same vertical offset # this uses an object named self
                            else:
                                self.set_text_color(50, 205, 50)
                                self.multi_cell(
                                    col_width,
                                    line_height,
                                    datum,
                                    border=0,
                                    align=align_data,
                                    ln=3,
                                    max_line_height=self.font_size,
                                )  # ln = 3 - move cursor to right with same vertical offset # this uses an object named self

                    else:
                        if datum in emphasize_data:
                            self.set_text_color(*emphasize_color)
                            self.set_font(self.font_name, style=emphasize_style)
                            self.multi_cell(
                                col_width,
                                line_height,
                                str(datum),
                                border=0,
                                align=align_data,
                                ln=3,
                                max_line_height=self.font_size,
                            )
                            self.set_text_color(0, 0, 0)
                        else:
                            self.multi_cell(
                                col_width,
                                line_height,
                                str(datum),
                                border=0,
                                align=align_data,
                                ln=3,
                                max_line_height=self.font_size,
                            )
                self.ln(line_height)  # move cursor back to the left margin

        else:
            if x_start:
                self.set_x(x_start)
            for i in range(len(header)):
                datum = header[i]
                self.multi_cell(
                    col_width[i],
                    line_height,
                    datum,
                    border=0,
                    align=align_header,
                    ln=3,
                    max_line_height=self.font_size,
                )
                x_right = self.get_x()
            self.ln(line_height)  # move cursor back to the left margin
            y2 = self.get_y()
            self.line(x_left, y1, x_right, y1)
            self.line(x_left, y2, x_right, y2)

            def is_number(n):
                try:
                    float(n)  # Type-casting the string to `float`.
                    # If string is not a valid `float`,
                    # it'll raise `ValueError` exception
                except ValueError:
                    return False
                return True

            for i in range(len(data)):
                if x_start:
                    self.set_x(x_start)
                row = data[i]
                for j in range(len(row)):
                    datum = row[j]
                    if not isinstance(datum, str):
                        datum = str(datum)
                    adjusted_col_width = col_width[j]
                    if j == 1:
                        if i == 0:
                            self.set_fill_color(RED_RGB)
                        elif i == 1:
                            self.set_fill_color(YELLOW_RGB)
                        elif i == 2:
                            self.set_fill_color(BLUE_RGB)
                        elif i == 3:
                            self.set_fill_color(GREEN_RGB)
                        else:
                            self.set_fill_color(0, 0, 0)

                        self.multi_cell(
                            adjusted_col_width,
                            line_height,
                            datum,
                            border=0,
                            fill=True,
                            align=align_data,
                            ln=3,
                            max_line_height=self.font_size,
                        )

                    else:
                        if datum in emphasize_data:
                            self.set_text_color(*emphasize_color)
                            self.set_font(self.font_name, style=emphasize_style)
                            self.multi_cell(
                                adjusted_col_width,
                                line_height,
                                datum,
                                border=0,
                                align=align_data,
                                ln=3,
                                max_line_height=self.font_size,
                            )
                            self.set_text_color(0, 0, 0)
                            self.set_font(self.font_name, style=default_style)
                        else:
                            self.set_fill_color(0, 0, 0)
                            self.multi_cell(
                                adjusted_col_width,
                                line_height,
                                datum,
                                border=0,
                                align=align_data,
                                ln=3,
                                max_line_height=self.font_size,
                            )  # ln = 3 - move cursor to right with same vertical offset # this uses an object named self
                self.ln(line_height)  # move cursor back to the left margin
        y3 = self.get_y()
        self.line(x_left, y3, x_right, y3)


class CreateReport:
    def __init__(
        self,
        impact_vector_path,
        zone_file,
        zone_file_name_col="name",
        layer_names=[
            "Buildings",
            "Road Closure Points",
            "Evacuation Centre",
            "Council Assets",
        ],
    ):
        self.impact_vector_path = impact_vector_path
        self.layer_names = layer_names

        # zones file
        self.zone_file = zone_file
        self.zone_file_name_col = zone_file_name_col

        # initiate pdf
        self.pdf = PdfFile()
        self.pdf.initiate_layout()

    def create_map(self, map_region, y_top=45, add_link=False):

        with rasterio.open(
            rf"d:\Royal HaskoningDHV\P-PA3396-Townsville-FLASH - WIP\python\impact_module_update\background_map_{map_region}.png"
        ) as src:
            bounds_geom = box(*src.bounds)

        fig, ax = plt.subplots(1, figsize=(14, 8))

        marker_legend_handles = []

        markers = {
            "Buildings": "o",
            "Evacuation Centre": "*",
            "Council Assets": "v",
            "Road Closure Points": "x",
        }

        exclude_non_impacted = ["Buildings", "Road Closure Points"]

        for layer in self.layer_names:
            shape = gpd.read_file(self.impact_vector_path, layer=layer)
            shape = shape.loc[shape.within(bounds_geom)]

            if layer in exclude_non_impacted:
                cmap = LinearSegmentedColormap.from_list(
                    "townsville_impact",
                    [(0, (0, 0, 0, 0)), (0.33, "blue"), (0.66, "yellow"), (1, "red")],
                )

                shape.plot(
                    ax=ax,
                    marker=markers[layer],
                    markersize=30,
                    column="vulnerability_class",
                    cmap=cmap,
                    vmin=1,
                    vmax=4,
                )
            else:
                cmap = LinearSegmentedColormap.from_list(
                    "townsville_impact",
                    [(0, "green"), (0.33, "blue"), (0.66, "yellow"), (1, "red")],
                )
                shape.plot(
                    ax=ax,
                    marker=markers[layer],
                    markersize=50,
                    column="vulnerability_class",
                    # edgecolor='black',
                    cmap=cmap,
                    vmin=1,
                    vmax=4,
                )

        ax.axis("off")

        legend_risk = plt.legend(
            [
                Line2D([0], [0], color=cmap(0.0), lw=4),
                Line2D([0], [0], color=cmap(0.33), lw=4),
                Line2D([0], [0], color=cmap(0.66), lw=4),
                Line2D([0], [0], color=cmap(0.99), lw=4),
            ],
            ["Low", "Medium", "High", "Very High"],
            title="Impact Assessment",
            loc="upper right",
        )

        zones = gpd.read_file(self.zone_file)
        suburbs = zones.loc[zones["Type"] == "Suburb"]
        police_sector = zones.loc[zones["Type"] == "Police Sector"]

        suburbs.plot(
            ax=ax,
            label="Suburbs",
            facecolor="none",
            edgecolor="black",
        )
        police_sector.plot(
            ax=ax,
            label="Police Sectors",
            facecolor="none",
            edgecolor="fuchsia",
            linestyle=":",
            linewidth=2,
        )

        custom_handle1 = Line2D(
            [],
            [],
            color="fuchsia",
            linestyle=":",
            label="Police Sector",
        )
        custom_handle2 = Line2D(
            [],
            [],
            color="black",
            label="Suburb",
        )

        marker_delineation_handles = []
        marker_delineation_handles.extend([custom_handle1, custom_handle2])
        legend_delineation = plt.legend(
            handles=marker_delineation_handles,
            loc="lower left",
            title="Areas of interest",
        )

        for marker_name, marker_style in markers.items():
            marker_legend_handles.append(
                Line2D(
                    [],
                    [],
                    color="black",
                    marker=marker_style,
                    linestyle="None",
                    markersize=10,
                    label=marker_name,
                )
            )

        legend_markers = plt.legend(
            handles=marker_legend_handles,
            loc="lower right",
            title="Overlays and markers",
        )

        ax.add_artist(legend_risk)
        ax.add_artist(legend_markers)
        ax.add_artist(legend_delineation)
        with rasterio.open(
            rf"d:\Royal HaskoningDHV\P-PA3396-Townsville-FLASH - WIP\python\impact_module_update\background_map_{map_region}.png"
        ) as src:
            bounds = list(src.bounds)

        with Image.open(
            rf"d:\Royal HaskoningDHV\P-PA3396-Townsville-FLASH - WIP\python\impact_module_update\background_map_{map_region}.png"
        ) as background_map:
            ax.imshow(
                background_map,
                extent=[bounds[0], bounds[2], bounds[1], bounds[3]],
                aspect="equal",
            )

        plt.savefig(
            TEMP_MAP_SAVELOC_FOLDER / f"temp_{map_region}.png",
            bbox_inches="tight",
            dpi=600,
        )

        # add map to pdf
        self.pdf.set_xy(x=3, y=y_top)

        with rasterio.open(TEMP_MAP_SAVELOC_FOLDER / f"temp_{map_region}.png") as src:
            profile = src.profile

        desired_width = 200
        wh_ratio = profile["height"] / profile["width"]

        self.pdf.image(
            TEMP_MAP_SAVELOC_FOLDER / f"temp_{map_region}.png",
            link="",
            type="png",
            w=desired_width,
            h=desired_width * wh_ratio,
        )
        if add_link:
            self.pdf.contents["| Map |"] = self.pdf.add_link()

    def create_main_table(self, y_start):
        summary_tables = {}

        for layer in self.layer_names:
            impact_assessment_statistics = gpd.read_file(
                self.impact_vector_path, layer=layer
            )
            vulnerability_class = [1, 2, 3, 4]

            statistics_df = pd.DataFrame(data={"Impact category": vulnerability_class})
            statistics_df["Amount"] = statistics_df.apply(
                lambda row: len(
                    impact_assessment_statistics.loc[
                        impact_assessment_statistics["vulnerability_class"]
                        == row["Impact category"]
                    ]
                ),
                axis=1,
            )
            statistics_df["[%]"] = statistics_df.apply(
                lambda row: round(
                    (
                        len(
                            impact_assessment_statistics.loc[
                                impact_assessment_statistics["vulnerability_class"]
                                == row["Impact category"]
                            ]
                        )
                        / len(impact_assessment_statistics)
                    )
                    * 100,
                    2,
                ),
                axis=1,
            )

            no_of_features = len(impact_assessment_statistics)
            summary_table_name = f"{layer} [{no_of_features}]\n"
            summary_tables[summary_table_name] = statistics_df

        x_pos = [5, 55, 105, 155]
        for i, (name, table) in enumerate(summary_tables.items()):
            self.pdf.add_table(
                x_pos[i], y_start, table.to_dict(), x_start=x_pos[i], title=name
            )
        summary_table_link = self.pdf.add_link()
        self.pdf.set_link(summary_table_link, y=210)
        self.pdf.contents["| Impact Summary |"] = summary_table_link

    def add_zone_breakdown_table(self):
        table = create_zone_breakdown_table(
            impact_vector_path=self.impact_vector_path,
            zone_file_path=self.zone_file,
            layer_names=self.layer_names,
            type="Suburb",
        )

        table_police = create_zone_breakdown_table(
            impact_vector_path=self.impact_vector_path,
            zone_file_path=self.zone_file,
            layer_names=self.layer_names,
            type="Police Sector",
        ).reset_index()

        table_police_dict = table_police.to_dict()

        table_p1 = table.reset_index().iloc[0:80, :]
        table_dict_p1 = table_p1.to_dict()

        table_p2 = table.reset_index().iloc[80:160, :]
        table_dict_p2 = table_p2.to_dict()

        table_p3 = table.reset_index().iloc[160:248, :]
        table_dict_p3 = table_p3.to_dict()

        table_p4 = table.reset_index().iloc[248:336, :]
        table_dict_p4 = table_p4.to_dict()

        table_p5 = table.reset_index().iloc[336:424, :]
        table_dict_p5 = table_p5.to_dict()

        start_x = [5, 110, 5, 110, 5, 110]

        self.pdf.contents["| Suburb Breakdown Table |"] = self.pdf.add_link()

        for i, t in enumerate(
            [
                table_dict_p1,
                table_dict_p2,
                table_dict_p3,
                table_dict_p4,
                table_dict_p5,
                table_police_dict,
            ]
        ):
            if i in [2, 4]:
                self.pdf.add_page()

            if i in [0, 1]:
                y_start = 55
            else:
                y_start = 45

            if i != 5:
                title = f"Suburbs [{i+1}/5]"
            else:
                title = "Police Sectors"
            n = "-"
            for k, v in t["Name"].items():
                if v != n:
                    if not v == "":
                        n = v
                else:
                    t["Name"][k] = ""

            self.pdf.add_main_summary_table(
                x=start_x[i],
                y=y_start,
                table_data=t,
                data_size=6,
                title=title,
                x_start=start_x[i],
                cell_width=[15, 25, 10, 8, 8, 12, 12],
            )

    def add_building_table(self, content_line_name):
        building_asset_table = create_building_table(
            impact_vector_path=self.impact_vector_path,
            zone_file_path=self.zone_file,
            layer_name="Buildings",
        )

        y = 50
        x = 5

        self.pdf.contents[content_line_name] = self.pdf.add_link()

        desired_number_of_rows_per_table = 54
        split_dfs = [
            building_asset_table[i : i + desired_number_of_rows_per_table]
            for i in range(
                0, building_asset_table.shape[0], desired_number_of_rows_per_table
            )
        ]

        for i, df in enumerate(split_dfs):
            if i % 2 == 0:
                x = 5
            else:
                x = 100
            self.pdf.add_building_asset_table(
                x=x,
                y=y,
                table_data=df,
            )
            if i % 2 == 0:
                pass
            else:
                self.pdf.add_page()

    def add_table_per_asset_type(self, type, content_line_name):
        suburb_asset_table = create_asset_breakdown_table(
            impact_vector_path=self.impact_vector_path,
            zone_file_path=self.zone_file,
            layer_name=self.layer_names[type],
            type="Suburb",
        )

        unique_suburbs = np.unique(suburb_asset_table["suburb_zone"].tolist())

        y = 50
        x = 5

        self.pdf.contents[content_line_name] = self.pdf.add_link()

        for sub in unique_suburbs:
            assets_in_suburb = suburb_asset_table.loc[
                suburb_asset_table["suburb_zone"] == sub
            ]
            # assets_in_suburb = assets_in_suburb.reset_index()
            if self.layer_names[type] == "Evacuation Centre":
                assets_in_suburb = assets_in_suburb.drop("suburb_zone", axis=1)[
                    ["Name", "Type", "max_depth", "Impact"]
                ]
                emphasize_column = 3
            else:
                assets_in_suburb = assets_in_suburb.drop("suburb_zone", axis=1)[
                    ["Name", "max_depth", "Impact"]
                ]
                emphasize_column = 2

            assets_in_suburb = assets_in_suburb.rename(
                columns={"max_depth": "Max. Depth [m]"}
            )

            if y + 2.5 * len(assets_in_suburb) > 250:
                if x == 5:
                    y = 50
                    x = 105
                else:
                    x = 5
                    y = 50
                    self.pdf.add_page()

            self.pdf.add_asset_table(
                x=x,
                y=y,
                table_data=assets_in_suburb,  # .to_dict(),
                emphasize_column=emphasize_column,
                title=sub,
            )

            y = self.pdf.get_y() + 5
            if y > 250 and x == 5:
                y = 50
                x = 105
            elif y > 250 and x == 105:
                x = 5
                y = 50
                self.pdf.add_page()
            else:
                pass

    def add_impact_guide(self):
        (
            buildings_impact_guide,
            roads_impact_guide,
            council_services_impact_guide,
            evac_centers_impact_guide,
        ) = create_impact_guide_table()

        y = 50
        x = 5

        self.pdf.contents["| Impact Guide |"] = self.pdf.add_link()

        for title, impact_guide in {
            "Buildings": buildings_impact_guide,
            "Roads": roads_impact_guide,
            "Council Services": council_services_impact_guide,
            "Evacuation Centres": evac_centers_impact_guide,
        }.items():

            self.pdf.add_impact_guide_table(
                x=x,
                y=y,
                table_data=impact_guide.to_dict(),
                x_start=x,
                cell_width=[50, 30],
                title=title,
            )

            y = self.pdf.y + 5

    def save_report(self):
        self.pdf.output(
            os.path.join(
                r"d:\Royal HaskoningDHV\P-PA3396-Townsville-FLASH - WIP\python\impact_module_update\output\factsheet.pdf"
            ),
            "F",
        )


if __name__ == "__main__":
    cm = CreateReport(
        zone_file_name_col="Name",
        impact_vector_path=r"d:\Royal HaskoningDHV\P-PA3396-Townsville-FLASH - WIP\python\impact_module_update\output\impact_vector.gpkg",
        zone_file=r"d:\Royal HaskoningDHV\P-PA3396-Townsville-FLASH - WIP\python\impact_module_update\AnalysisAreas\Impact_Module_Areas.shp",
    )

    cm.pdf.add_text(
        x=3,
        y=45,
        text="Overall Impact Assessment Summary",
        alignment="L",
        fontsize=12,
        bold=True,
    )
    cm.create_main_table(y_start=65)

    cm.pdf.add_page()

    cm.create_map(map_region="magnetic_island", y_top=50, add_link=True)
    cm.pdf.add_text(
        x=3,
        y=40,
        text="Magnetic Island",
        alignment="L",
        fontsize=8,
        bold=True,
    )

    cm.create_map(map_region="balgal_bay", y_top=170, add_link=True)
    cm.pdf.add_text(
        x=3,
        y=160,
        text="Balgal Bay",
        alignment="L",
        fontsize=8,
        bold=True,
    )

    cm.pdf.add_page()
    
    cm.create_map(map_region="townsville_north", y_top=50)
    cm.pdf.add_text(
        x=3,
        y=40,
        text="Townsville North",
        alignment="L",
        fontsize=8,
        bold=True,
    )

    cm.create_map(map_region="townsville_south", y_top=170)
    cm.pdf.add_text(
        x=3,
        y=160,
        text="Townsville South",
        alignment="L",
        fontsize=8,
        bold=True,
    )

    cm.pdf.add_page()

    cm.pdf.add_text(
        x=3,
        y=35,
        text="Impact Assessment Breakdown Per Suburb",
        alignment="L",
        fontsize=12,
        bold=True,
    )
    cm.add_zone_breakdown_table()

    cm.pdf.add_page()

    cm.pdf.add_text(
        x=3,
        y=35,
        text="Breakdown Per Asset Type - Buildings",
        alignment="L",
        fontsize=12,
        bold=True,
    )
    cm.add_building_table(content_line_name="| Buildings |")

    cm.pdf.add_page()

    cm.pdf.add_text(
        x=3,
        y=35,
        text="Breakdown Per Asset Type - Evacuation Centres",
        alignment="L",
        fontsize=12,
        bold=True,
    )
    cm.add_table_per_asset_type(type=2, content_line_name="| Evacuation Centres |")

    cm.pdf.add_page()

    cm.pdf.add_text(
        x=3,
        y=35,
        text="Breakdown Per Asset Type - Council Services",
        alignment="L",
        fontsize=12,
        bold=True,
    )
    cm.add_table_per_asset_type(type=3, content_line_name="| Council Services |")

    cm.pdf.add_page()

    cm.pdf.add_text(
        x=3,
        y=35,
        text="Breakdown Per Asset Type - Road Flooding Points",
        alignment="L",
        fontsize=12,
        bold=True,
    )
    cm.add_table_per_asset_type(type=1, content_line_name="| Road Flooding Points |")

    cm.pdf.add_page()

    cm.pdf.add_text(
        x=3,
        y=35,
        text="Impact Guide",
        alignment="L",
        fontsize=12,
        bold=True,
    )

    cm.add_impact_guide()
    cm.pdf.add_page()

    cm.pdf.add_content_line()
    cm.save_report()
