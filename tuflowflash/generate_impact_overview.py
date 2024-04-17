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

MAP_EXTENT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "styling", "map_extent.gpkg"
)
TEMP_MAP_SAVELOC = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "styling", "temp_map.png"
)


class PdfFile(FPDF):
    def initiate_layout(self):
        self.set_auto_page_break(False, margin=0)
        self.font_name = "Helvetica"
        self.add_page()
        self.set_font(self.font_name, size=6)
        self.add_text(
            x=3,
            y=10,
            alignment="L",
            text="Townsville Impact Module",
            fontsize=14,
            bold=True,
        )

    def add_text(self, x, y, alignment, text, fontsize, bold=False):
        self.set_xy(x, y)
        if bold:
            self.set_font("helvetica", "B", fontsize)
        else:
            self.set_font("helvetica", "", fontsize)
        self.set_text_color(0, 0, 0)
        self.cell(w=210.0, h=20.0, align=alignment, text=text, border=0)

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
        emphasize_style="B",
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

        self.set_font(self.font_name, size=title_size)

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
        for _, row in zones.iterrows():
            gpd.GeoDataFrame(row).T.plot(
                ax=ax,
                label=row["name"],
                facecolor="none",
                edgecolor=np.random.rand(
                    3,
                ),
            )

        for plotted_item in ax.get_children():
            label = plotted_item.get_label()

            labels = zones["name"].tolist()
            if label in labels:
                edgecolor = plotted_item.get_edgecolor()
                custom_handle = mpatches.Patch(facecolor=edgecolor, label=label)
                marker_legend_handles.append(custom_handle)

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

        cx.add_basemap(ax, crs=crs, zoom=14)

        plt.savefig(TEMP_MAP_SAVELOC, bbox_inches="tight", dpi=600)

        # add map to pdf
        self.pdf.set_xy(x=3, y=20)

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
            summary_table_name = f"Impact assessment \n{layer} [{no_of_features}]"
            summary_tables[summary_table_name] = statistics_df

        x_pos = [5, 55, 105, 155]
        for i, (name, table) in enumerate(summary_tables.items()):
            self.pdf.add_table(
                x_pos[i], y_start, table.to_dict(), x_start=x_pos[i], title=name
            )

    def create_table_per_zone(self, y_start):
        zones = gpd.read_file(self.zone_file)
        if len(zones) > 0:
            current_y = y_start
            for _, row in zones.iterrows():
                self.pdf.add_text(
                    x=0,
                    y=current_y,
                    text=f"{row[self.zone_file_name_col]} Impact Assessment",
                    alignment="C",
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
                        summary_table_name = (
                            f"Impact assessment \n{layer} [{no_of_features}]"
                        )
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
                        summary_table_name = (
                            f"Impact assessment \n{layer} [{no_of_features}]"
                        )
                        summary_tables[summary_table_name] = statistics_df

                x_pos = [5, 55, 105, 155]
                for i, (name, table) in enumerate(summary_tables.items()):
                    self.pdf.add_table(
                        x_pos[i],
                        current_y + 15,
                        table.to_dict(),
                        x_start=x_pos[i],
                        title=name,
                    )

                current_y += 20

    def save_report(self):
        self.pdf.output(
            os.path.join(
                r"d:\Royal HaskoningDHV\P-PA3396-Townsville-FLASH - WIP\python\impact_module_update\output\factsheet.pdf"
            ),
            "F",
        )


if __name__ == "__main__":
    cm = CreateReport(
        impact_vector_path=r"d:\Royal HaskoningDHV\P-PA3396-Townsville-FLASH - WIP\python\impact_module_update\output\impact_vector.gpkg",
        zone_file=r"d:\Royal HaskoningDHV\P-PA3396-Townsville-FLASH - WIP\python\impact_module_update\testdata\impact_zone.gpkg",
    )
    cm.create_map()
    cm.pdf.add_text(
        x=0, y=120, text="Overall Impact Assessment", alignment="C", fontsize=12
    )
    cm.create_main_table(y_start=160)
    cm.create_table_per_zone(y_start=185)
    cm.save_report()
