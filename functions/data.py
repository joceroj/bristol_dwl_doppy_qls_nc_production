"""Created on April 29, 2025.

Creating the Doppler Wind Lidar Data.

@author: Jonnathan Céspedes
"""  # noqa: INP001

import datetime as dt
from dataclasses import dataclass, field
from typing import Any

import doppy
import xarray as xr
from functions.builder_xr import build_wind_dataset
from matplotlib.path import Path
from utils.tools import (
    check_overwrite,
    get_list_files,
    remove_encoding_conflicting_attrs,
    setup_logger,
    write_netcdf,
)

logger = setup_logger()


@dataclass(slots=True)
class DataLoader:
    """A class to load DWL data for VAD and Stare observations from a given site.

    Attributes:
        conf (dict): Global configuration dictionary parsed from TOML.
        init_dates (list): List of dates (datetime-like) to process.
        site (str): Site identifier in uppercase (e.g., 'BRDIFU').
        output_dir (Path): Creates an output directory.
        dir_mask (str): Filepath pattern for the site's data directory.
        list_back_files (list): Background signal files.
        list_vertical_files (list): Files for vertical stare scanning.
        list_vad_files (list): Files for VAD (Velocity-Azimuth Display) scanning.

    """

    conf: dict[str, Any]
    site: str
    init_dates: list[dt.datetime]

    dir_mask: str = field(init=False)
    site_conf: dict[str, Any] = field(init=False)
    files: dict[str, list[Path]] = field(init=False)

    def __post_init__(self) -> None:
        """Initialize derived attributes after dataclass instantiation.

        This method sets the `dir_mask`: tha mask of the data directory.
        It is automatically called after the dataclass has initialized
        all declared fields.
        """
        # messages for user
        logger.info("Available sites: %s", list(self.conf["sites"].keys()))
        logger.info("Requested site: %s", self.site)

        self.validate_site()
        self.site_conf = self.conf["sites"][self.site]
        self.dir_mask = self.site_conf["dir_mask"]
        self.files = {}

    def validate_site(self) -> None:
        """Validate that the required site exist in the config file."""
        if self.site not in self.conf["sites"]:
            msg = f"Site '{self.site}' is not in configuration."
            raise ValueError(msg)

    def load_files(self) -> None:
        """Load the *.hpl for VAD and Stare, and *.txt for background files."""
        # mask for every type of file
        masks = self.conf["all_files_mask"]

        # getting all files
        self.files["background"] = get_list_files(
            self.init_dates,
            self.dir_mask,
            masks["file_back_mask"],
        )
        self.files["vertical"] = get_list_files(
            self.init_dates,
            self.dir_mask,
            masks["file_vertical_mask"],
        )
        self.files["vad"] = get_list_files(
            self.init_dates,
            self.dir_mask,
            masks["file_vad_mask"],
        )

        missing = [k for k, v in self.files.items() if not v]
        if missing:
            msg = f"No files found for: {', '.join(missing)}"
            raise FileNotFoundError(msg)


@dataclass(slots=True)
class WindProcessor:
    """A class to process DWL data for wind and stare retrievals from a given site.

    Attributes:
        conf (dict): Global configuration dictionary parsed from TOML.
        site (str): Site identifier in uppercase (e.g., 'BRDIFU').
        output_dir (Path): Creates an output directory.
        files (list): Background signal files.

    """

    conf: dict[str, Any]
    site: str
    site_conf: dict[str, Any]
    init_dates: list
    output_dir: Path
    files: dict[str, list[Path]]

    def process_stare(self) -> xr.Dataset:
        """Process vertical stare mode files into an xarray Dataset.

        Returns:
            xr.Dataset: Cleaned and validated vertical wind profile dataset.

        Raises:
            ValueError: If the resulting dataset is empty or invalid.

        """
        stare = doppy.product.Stare.from_halo_data(
            data=self.files["vertical"],
            data_bg=self.files["background"],
            bg_correction_method=doppy.options.BgCorrectionMethod.FIT,
        )
        return self._validate_and_clean(build_wind_dataset(stare), "vertical")

    def process_wind(self) -> xr.Dataset:
        """Process VAD scanning mode files into an xarray Dataset.

        Returns:
            xr.Dataset: Cleaned and validated horizontal wind (VAD) dataset.

        Raises:
            ValueError: If the resulting dataset is empty or invalid.

        """
        wind = doppy.product.Wind.from_halo_data(
            data=self.files["vad"],
            options=doppy.product.wind.Options(
                azimuth_offset_deg=self.site_conf["wd_offset"],
            ),
        )
        return self._validate_and_clean(build_wind_dataset(wind), "horizontal")

    def _validate_and_clean(self, ds: xr.Dataset, label: str) -> xr.Dataset:
        """Validate and clean a dataset.

        Check for emptiness, clean encodings, and remove the near-range gates.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset to validate and clean.
        label : str
            Label for error reporting (e.g., 'vertical' or 'horizontal').

        Returns
        -------
        xr.Dataset
            Cleaned dataset with encoding attributes removed and near range filtered.

        Raises
        ------
        ValueError
            If the dataset is None or has no dimensions.

        """
        if ds is None or len(ds.dims) == 0:
            msg = f"{label.capitalize()} wind dataset is empty or invalid."
            raise ValueError(msg)

        ds = remove_encoding_conflicting_attrs(ds)
        return self._remove_near_range(ds)

    def _remove_near_range(self, ds: xr.Dataset) -> xr.Dataset:
        """Remove the first three range gates from all variables in the dataset.

        This is often necessary to exclude unreliable measurements near the instrument.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset containing a 'range' coordinate.

        Returns
        -------
        xr.Dataset
            Dataset with the first three range gates removed.

        """
        n_ =4
        if "range" not in ds.coords or ds.dims.get("range", 0) < n_:
            return ds  # Not enough range bins or missing, skip trimming
        return ds.sel(range=ds.range[3:])

    def to_netcdf_vertical(self, ds: xr.Dataset) -> None:
        """Write vertical wind profile dataset to a NetCDF file.

        The filename includes the site and date range.

        Args:
            ds (xr.Dataset): Dataset containing vertical wind profile data.

        Raises:
            FileExistsError: If the output file already exists.

        """
        name = (
            f"{self.site}_Vertical_wind_profile_"
            f"{self.init_dates[0]:%Y-%m-%d}_{self.init_dates[-1]:%Y-%m-%d}.nc"
        )
        output_file = self.output_dir / name
        check_overwrite(output_file)
        config = self.conf["netCDF"]["vertical"]
        write_netcdf(ds, config, output_file)

    def to_netcdf_horizontal(self, ds: xr.Dataset) -> None:
        """Write horizontal wind (VAD) dataset to a NetCDF file.

        The filename includes the site and date range.

        Args:
            ds (xr.Dataset): Dataset containing horizontal wind (VAD) data.

        Raises:
            FileExistsError: If the output file already exists.

        """
        name = (
            f"{self.site}_Horizontal_wind_VAD_"
            f"{self.init_dates[0]:%Y-%m-%d}_{self.init_dates[-1]:%Y-%m-%d}.nc"
        )
        output_file = self.output_dir / name
        check_overwrite(output_file)
        config = self.conf["netCDF"]["horizontal"]
        write_netcdf(ds, config, output_file)

