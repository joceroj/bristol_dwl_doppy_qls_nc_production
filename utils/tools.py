"""Created on April 29, 2025.

Different tools used to explore doppy.

@author: Jonnathan Céspedes
"""  # noqa: INP001

import datetime as dt
import logging
import sys
from pathlib import Path

import click
import xarray as xr


def setup_logger(
    name: str = "dwl_processor", level: int = logging.INFO,
) -> logging.Logger:
    """Set up a logger with stdout output and formatted messages."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(message)s",
            "%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


def get_list_files(
    dates: list[dt.datetime],
    dir_mask: str,
    file_mask: str,
) -> list[Path]:
    """Get the list of files to read based on dates and files mask.

    Parameters
    ----------
    dates : list[dt.datetime]
        The list of dates to search files at.
    dir_mask : str
        Mask of the directory where files are stored.
    file_mask : str
        Mask of the files.

    Returns
    -------
    list[Path]
        The list of files found corresponding to the dates and masks.

    """
    list_files = []
    for date in dates:
        dir_name = Path(date.strftime(dir_mask))
        file_name = date.strftime(file_mask)

        list_files += sorted(dir_name.glob(file_name))

    return list_files


def remove_encoding_conflicting_attrs(
    data: xr.Dataset | xr.DataArray,
    extra_conflicts: list[str] | None = None,
) -> xr.Dataset | xr.DataArray:
    """Remove attributes that may conflict with NetCDF encoding from an xarray Dataset.

    Parameters
    ----------
    data : xarray.Dataset or xarray.DataArray
        Input dataset or data array.
    extra_conflicts : list of str, optional
        Additional attribute names to remove beyond the standard set.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Copy of the input with problematic attributes removed.

    """
    # Default set of known encoding-related attributes
    default_conflicts = [
        "units",
        "calendar",
        "scale_factor",
        "add_offset",
        "_FillValue",
        "fill_value",
        "dtype",
        "missing_value",
    ]
    conflict_attrs = set(default_conflicts + (extra_conflicts or []))

    data = data.copy()

    # Clean global attributes
    for attr in list(data.attrs):
        if attr in conflict_attrs:
            del data.attrs[attr]

    # Clean attributes in all variables (including coordinates)
    for var in list(data.variables):
        for attr in list(data[var].attrs):
            if attr in conflict_attrs:
                del data[var].attrs[attr]

    return data


def check_overwrite(path: Path) -> bool:
    """Check if file exists and ask user to overwrite it."""
    if path.exists():
        click.confirm(f"File {path.name} already exists. Overwrite?", abort=True)
    return True


def write_netcdf(data: xr.Dataset, conf: dict, file: Path) -> None:
    """Create the netCDF file.

    Parameters
    ----------
    data : xr.Dataset
        The data to write.
    conf : dict
        dictionary containing the netCDF values.
    file : Path
        The path to the file to create.

    """
    # Add global attributes
    if "global_attributes" in conf:
        data.attrs.update(conf["global_attributes"])

    # create encoding dict
    encoding = {
        key: conf["variables"][key]["encoding"]
        for key in conf["variables"]
        if "encoding" in conf["variables"][key]
    }

    # add metadata looping over variables
    for var_name in conf["variables"]:
        if var_name in data:
            if "metadata" in conf["variables"][var_name]:
                data[var_name].attrs.update(conf["variables"][var_name]["metadata"])
        else:
            click.echo(
                f" Variable '{var_name}' not found in dataset. Skipping metadata.",
            )

    # add metadata to coordinates too
    for coord in data.coords:
        if coord in conf["variables"] and "metadata" in conf["variables"][coord]:
            data[coord].attrs.update(conf["variables"][coord]["metadata"])

    # Verify if data is empty
    if not data or len(data.dims) == 0:
        msg = f" Dataset is empty. Not writing to {file}"
        raise ValueError(msg)

    # write to NetCDF
    try:
        data.to_netcdf(file, encoding=encoding)
    except OSError as e:
        msg = f"Failed to write NetCDF to {file}: {e}"
        raise RuntimeError(msg) from e
