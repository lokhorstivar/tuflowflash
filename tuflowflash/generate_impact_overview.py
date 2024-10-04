import os
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

warnings.filterwarnings("ignore")


MAP_EXTENT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "styling", "map_extent.gpkg"
)
TEMP_MAP_SAVELOC = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "styling", "temp_map.png"
)


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
            "IC 1",
            "IC 2",
            "IC 3",
            "IC 4",
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
            "IC 1",
            "IC 2",
            "IC 3",
            "IC 4",
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
            "IC 1",
            "IC 2",
            "IC 3",
            "IC 4",
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
            "IC 1",
            "IC 2",
            "IC 3",
            "IC 4",
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


def create_asset_breakdown_table(impact_vector_path, zone_file_path, layer_names, type):
    impact_assessment_statistics_evacuation_centre = gpd.read_file(
        impact_vector_path, layer=layer_names[2]
    )

    regions = gpd.read_file(zone_file_path)
    suburbs = regions.loc[regions["Type"] == "Suburb"]
    impact_assessment_statistics_evacuation_centre["suburb_zone"] = (
        impact_assessment_statistics_evacuation_centre.apply(
            lambda row: suburbs.loc[
                suburbs.geometry.intersects(row.geometry), "Name"
            ].iloc[0],
            axis=1,
        )
    )

    impact_assessment_statistics_evacuation_centre["Asset Name"] = (
        impact_assessment_statistics_evacuation_centre.apply(
            lambda row: f"Fake evacuation center {row.name}", axis=1
        )
    )

    impact_assessment_statistics_evacuation_centre["Water Depth [m]"] = (
        impact_assessment_statistics_evacuation_centre.apply(
            lambda row: round(random.random() * 2, 2), axis=1
        )
    )

    def assign_impact_category(row):
        vulnerability_class = row.vulnerability_class
        if vulnerability_class == 1:
            return "1 - Low"
        elif vulnerability_class == 2:
            return "2 - Moderate"
        elif vulnerability_class == 3:
            return "3 - High"
        elif vulnerability_class == 4:
            return "4 - Very High"
        else:
            return None
        return None

    impact_assessment_statistics_evacuation_centre["Impact Category"] = (
        impact_assessment_statistics_evacuation_centre.apply(
            lambda row: assign_impact_category(row), axis=1
        )
    )
    impact_assessment_statistics_evacuation_centre = (
        impact_assessment_statistics_evacuation_centre.drop(
            ["geometry", "vulnerability_class"], axis=1
        )
    )
    return impact_assessment_statistics_evacuation_centre


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
            text="Townsville Impact Module",
            fontsize=18,
            bold=False,
            color=(15, 24, 96),
        )
        self.set_xy(x=170, y=5)
        self.image(
            r"d:\Royal HaskoningDHV\P-PA3396-Townsville-FLASH - WIP\python\impact_module_update\tville_logo.png",
            link="",
            type="png",
            w=225 / 7,
            h=225 / 7 * 1,
        )

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
                    print(self.font_size)
                    x += len(txt) * 1.5 + 5

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
                            self.set_fill_color(0, 128, 0)
                        elif j == 4:
                            self.set_fill_color(3, 3, 255)
                        elif j == 5:
                            self.set_fill_color(254, 254, 1)
                        elif j == 6:
                            self.set_fill_color(255, 6, 0)
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

    def create_map(self):
        map_extent = gpd.read_file(MAP_EXTENT)
        fig, ax = plt.subplots(1, figsize=(14, 8))

        marker_legend_handles = []

        cmap = LinearSegmentedColormap.from_list(
            "townsville_impact",
            [(0, "green"), (0.33, "blue"), (0.66, "yellow"), (1, "red")],
        )

        markers = {"Evacuation Centre": "*", "Council Assets": "v"}

        for layer in self.layer_names:
            shape = gpd.read_file(self.impact_vector_path, layer=layer)
            shape = shape.loc[shape.within(map_extent.geometry.iloc[0])]
            if layer in markers:
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
                shape.plot(
                    column="vulnerability_class", ax=ax, cmap=cmap, vmin=1, vmax=4
                )
            crs = shape.crs

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
            edgecolor="orange",
            linestyle=":",
            linewidth=2,
        )

        custom_handle1 = Line2D(
            [],
            [],
            color="orange",
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
            r"d:\Royal HaskoningDHV\P-PA3396-Townsville-FLASH - WIP\python\impact_module_update\background_map.png"
        ) as src:
            bounds = list(src.bounds)

        with Image.open(
            r"d:\Royal HaskoningDHV\P-PA3396-Townsville-FLASH - WIP\python\impact_module_update\background_map.png"
        ) as background_map:
            ax.imshow(
                background_map,
                extent=[bounds[0], bounds[2], bounds[1], bounds[3]],
                aspect="equal",
            )

        plt.savefig(TEMP_MAP_SAVELOC, bbox_inches="tight", dpi=600)

        # add map to pdf
        self.pdf.set_xy(x=3, y=45)

        with rasterio.open(TEMP_MAP_SAVELOC) as src:
            profile = src.profile

        desired_width = 200
        wh_ratio = profile["height"] / profile["width"]

        self.pdf.image(
            TEMP_MAP_SAVELOC,
            link="",
            type="png",
            w=desired_width,
            h=desired_width * wh_ratio,
        )
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
        self.pdf.set_link(summary_table_link, y=230)
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

        start_x = [13, 103, 13, 103, 13, 103]

        self.pdf.contents["| Zone Breakdown Table |"] = self.pdf.add_link()

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
                cell_width=[15, 25, 10, 7, 7, 7, 7],
            )

    def create_table_per_zone(self, y_start, type):  # DEPRECATED
        zones = gpd.read_file(self.zone_file)
        zones = zones.loc[zones["Type"] == type]

        if len(zones) > 0:
            current_y = y_start
            for _, row in zones.iterrows():
                self.pdf.add_text(
                    x=10,
                    y=current_y,
                    text=f"{row[self.zone_file_name_col]}:",
                    alignment="L",
                    fontsize=12,
                )
                summary_tables = {}

                for layer in self.layer_names:
                    impact_assessment_statistics = gpd.read_file(
                        self.impact_vector_path, layer=layer
                    )
                    impact_assessment_statistics = impact_assessment_statistics.clip(
                        row.geometry
                    )
                    if len(impact_assessment_statistics) > 0:
                        vulnerability_class = [1, 2, 3, 4]

                        statistics_df = pd.DataFrame(
                            data={"Impact category": vulnerability_class}
                        )
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
                                            impact_assessment_statistics[
                                                "vulnerability_class"
                                            ]
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
                    else:
                        vulnerability_class = [1, 2, 3, 4]
                        statistics_df = pd.DataFrame(
                            data={
                                "Impact category": vulnerability_class,
                                "Amount": [0, 0, 0, 0],
                                "[%]": [0, 0, 0, 0],
                            }
                        )
                        no_of_features = len(impact_assessment_statistics)
                        summary_table_name = f"{layer} [{no_of_features}]\n"
                        summary_tables[summary_table_name] = statistics_df

                x_pos = [50, 90, 130, 170]  #
                for i, (name, table) in enumerate(summary_tables.items()):
                    self.pdf.add_table(
                        x_pos[i],
                        current_y,
                        table.to_dict(),
                        x_start=x_pos[i],
                        title=name,
                    )

                if current_y > 220:  # go to next page
                    self.pdf.add_page()
                    current_y = 50
                else:
                    current_y += 30

    def add_table_per_asset_type(self):
        suburb_asset_table = create_asset_breakdown_table(
            impact_vector_path=self.impact_vector_path,
            zone_file_path=self.zone_file,
            layer_names=self.layer_names,
            type="Suburb",
        )

        unique_suburbs = np.unique(suburb_asset_table["suburb_zone"].tolist())

        y = 50
        x = 5

        self.pdf.contents["| Asset Breakdown Table |"] = self.pdf.add_link()

        for sub in unique_suburbs:
            assets_in_suburb = suburb_asset_table.loc[
                suburb_asset_table["suburb_zone"] == sub
            ]
            # assets_in_suburb = assets_in_suburb.reset_index()
            assets_in_suburb = assets_in_suburb.drop("suburb_zone", axis=1)

            self.pdf.add_table(
                x=x,
                y=y,
                table_data=assets_in_suburb.to_dict(),
                x_start=x,
                cell_width=[30, 30, 20, 20],
                title=sub,
            )

            y = self.pdf.y + 5
            if y > 230 and x == 5:
                y = 50
                x = 100
            elif y > 230 and x == 100:
                x = 5
                y = 50
                self.pdf.add_page()

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
        y=210,
        text="Overall Impact Assessment Summary",
        alignment="L",
        fontsize=12,
        bold=True,
    )

    cm.create_map()
    cm.pdf.add_text(
        x=3,
        y=210,
        text="Overall Impact Assessment Summary",
        alignment="L",
        fontsize=12,
        bold=True,
    )
    cm.create_main_table(y_start=230)
    cm.pdf.add_page()
    cm.pdf.add_text(
        x=3,
        y=35,
        text="Impact Assessment Breakdown Per Region",
        alignment="L",
        fontsize=12,
        bold=True,
    )
    cm.add_zone_breakdown_table()

    cm.pdf.add_page()

    cm.pdf.add_text(
        x=3,
        y=35,
        text="Breakdown Per Asset Type - Evacuation Centres",
        alignment="L",
        fontsize=12,
        bold=True,
    )
    cm.add_table_per_asset_type()
    cm.pdf.add_page()
    cm.pdf.add_content_line()
    cm.save_report()
