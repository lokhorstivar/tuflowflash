from pathlib import Path

import configparser as ConfigParser
import datetime
import logging


logger = logging.getLogger(__name__)

Tuflow_settings = {
    "tuflow_executable": str,
    "tcf_file": Path,
    "manage_states": bool,
    "initial_states_folder": Path,
    "export_states_folder": Path,
    "states_expiry_time_days": int,
    "gauge_rainfall_file": Path,
    "boundary_csv_input_file": Path,
    "boundary_csv_tuflow_file": Path,
    "custom_residual_script": Path,
    "netcdf_forecast_rainfall_file": Path,
    "netcdf_nowcast_rainfall_file": Path,
    "raster_output_folder": Path,
    "rain_grids_folder": Path,
    "rain_grids_csv": Path,
    "archive_folder": Path,
    "soil_moisture_awra_l_url": str,
    "soil_moisture_folder": Path,
    "soil_moisture_depth_file": Path,
    "soil_root_zone_depth_file": Path,
    "soil_moisture_xres": int,
    "soil_moisture_yres": int,
    "soil_moisture_xmin": int,
    "soil_moisture_ymin": int,
    "soil_moisture_xmax": int,
    "soil_moisture_ymax": int,
    "projection": int,
    "pu": int,
}

lizard_settings = {
    "apikey": str,
    "precipitation_uuid_file": Path,
    "depth_raster_uuid": str,
    "waterlevel_raster_uuid": str,
    "rainfall_raster_uuid": str,
    "waterlevel_result_uuid_file": Path,
    "waterdepth_raster_upload_list": list,
    "waterlevel_raster_upload_list": list,
    "historic_forecast_administration_csv": Path,
}

impact_module_settings = {
    "end_result_type": str,
    "roads_file": Path,
    "buildings_file": Path,
    "assets_file": Path,
    "impact_raster_uuid": str,
}

switches_settings = {
    "get_historical_precipitation": bool,
    "convert_csv_to_bc": bool,
    "custom_residual_tide": bool,
    "get_bom_forecast": bool,
    "get_bom_nowcast": bool,
    "use_bom_historical": bool,
    "use_soil_moisture": bool,
    "run_simulation": bool,
    "post_to_lizard": bool,
    "archive_simulation": bool,
    "clear_input_output": bool,
    "track_historic_forecasts": bool,
    "determine_impact": bool,
}

bom_settings = {
    "bom_username": str,
    "bom_password": str,
    "bom_url": str,
    "bom_forecast_file": str,
    "bom_nowcast_file": str,
    "historic_rain_folder": str,
    "forecast_clipshape": Path,
}

email_settings = {
    "email_adress": str,
    "email_password": str,
    "email_attendees": str,
    "email_text_file": Path,
    "email_subject": str,
}


class MissingFileException(Exception):
    pass


class MissingSettingException(Exception):
    pass


class FlashSettings:
    def __init__(self, settingsFile: Path, reference_time=None):
        self.settingsFile = settingsFile
        self.config = ConfigParser.RawConfigParser()
        try:
            self.config.read(settingsFile)
        except FileNotFoundError as e:
            msg = f"Settings file '{settingsFile}' not found"
            raise MissingFileException(msg) from e

        self.read_settings_file(Tuflow_settings, "tuflow")
        self.read_settings_file(lizard_settings, "lizard")
        self.read_settings_file(switches_settings, "switches")
        if self.determine_impact:
            self.read_settings_file(impact_module_settings, "impact_module")
        self.read_settings_file(bom_settings, "bom")
        self.read_settings_file(email_settings, "email")

        self.read_tcf_parameters(self.tcf_file)

        self.reference_time, self.start_time = self.convert_relative_time(
            self.tuflow_start_time, reference_time
        )

        self.reference_time, self.end_time = self.convert_relative_time(
            self.tuflow_end_time, reference_time
        )

        logger.info("settings have been read")

    def extract_variable_from_tcf(self, line):
        variable = line.split("==")[1]
        variable = variable.split("!")[0].strip()
        return variable

    def read_tcf_parameters(self, tcf_file):
        with open(tcf_file) as f:
            lines = f.readlines()
        for line in lines:
            if line.lower().startswith("start time"):
                setattr(
                    self,
                    "tuflow_start_time",
                    float(self.extract_variable_from_tcf(line)),
                )
            if line.lower().startswith("end time"):
                setattr(
                    self,
                    "tuflow_end_time",
                    float(self.extract_variable_from_tcf(line)),
                )
            if line.lower().startswith("output folder"):
                setattr(
                    self, "output_folder", Path(self.extract_variable_from_tcf(line))
                )
            if line.lower().startswith("shp projection"):
                setattr(self, "prj_file", Path(self.extract_variable_from_tcf(line)))
            if line.lower().startswith("read restart file"):
                setattr(
                    self, "restart_file", Path(self.extract_variable_from_tcf(line))
                )

    def roundTime(self, dt=None, roundTo=60):
        """Round a datetime object to any time lapse in seconds
        dt : datetime.datetime object, default now.
        roundTo : Closest number of seconds to round to, default 1 minute.
        """
        if dt is None:
            dt = datetime.datetime.now()
        seconds = (dt.replace(tzinfo=None) - dt.min).seconds
        rounding = (seconds + roundTo / 2) // roundTo * roundTo
        return dt + datetime.timedelta(0, rounding - seconds, -dt.microsecond)

    def convert_relative_time(self, relative_time, reference_time=None):
        if reference_time is None:
            reference_time = self.roundTime()
        else:
            reference_time = datetime.datetime.strptime(
                reference_time, "%Y-%m-%dT%H:%M"
            )
        time = reference_time + datetime.timedelta(hours=float(relative_time))
        return reference_time, time

    def read_settings_file(self, variables, variable_header):
        # maak de output van deze functie aan
        for variable, datatype in variables.items():
            value = self.config.get(variable_header, variable)
            if len(value) > 0:
                try:
                    if datatype == int:
                        setattr(self, variable, int(value))
                    if datatype == Path:
                        try:
                            setattr(self, variable, Path(value))
                        except:
                            logger.warning("no value found for %s", variable)
                    if datatype == str:
                        setattr(self, variable, str(value))
                    if datatype == bool:
                        setattr(
                            self,
                            variable,
                            value.lower() == "true",
                        )
                    if datatype == list:
                        input_list = string_to_list(value)
                        setattr(self, variable, input_list)
                except:
                    raise MissingSettingException(
                        f"Setting: '{variable}' is missing in " f"{self.settingsFile}."
                    )


def string_to_list(string):
    return string.split(",")
