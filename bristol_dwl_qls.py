#!/usr/bin/env python  # noqa: EXE001

"""Created on April 29, 2025.

Script to create time-height cross sections (quick-look QLs) from Doppler Wind Lidar.

@author: Jonnathan Céspedes <j.cespedes@reading.ac.uk>
"""

import traceback
from pathlib import Path
from typing import Any

import click
import pandas as pd
from functions.data import DataLoader, WindProcessor
from functions.ql import HorizontalPlots
from utils.cli_utils import validate_site
from utils.tools import setup_logger

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


__version__ = "0.0.1"
__author__ = "j.cespedes@reading.ac.uk"

logger = setup_logger()


@click.command()
@click.argument(
    "conf-file",
    type=click.Path(
        exists=True,
        readable=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
        path_type=Path,
    ),
)
@click.argument(
    "date_start",
    type=click.DateTime(formats=["%Y%m%d", "%Y-%m-%d"]),
)
@click.argument(
    "date_end",
    type=click.DateTime(formats=["%Y%m%d", "%Y-%m-%d"]),
)
@click.argument(
    "site",
    type=str,
    callback=validate_site,
)
@click.option("--save-wind-ql", "-sql", is_flag=True, help="Save PNG quick-look")
# @click.option("--save-nc", "-snc", is_flag=True, help="Save NetCDF files")
@click.option(
    "--output-dir",
    "-o",
    default=Path(__file__).parent / "outputs",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Output directory",
)
def main(**kwargs: dict[str, Any]) -> None:
    """Entry point for the CLI."""
    run_pipeline(**kwargs)


def run_pipeline(  # noqa: PLR0913
    conf_file: Path,
    date_start: click.DateTime,
    date_end: click.DateTime,
    site: str,
    # save_nc: bool,
    save_wind_ql: bool,
    output_dir: Path,
) -> None:
    """Process the main logic of this tool.

    Args:
    ----
        conf_file: Path
            The path to the configuration file.
        date_start: click.DateTime
            The start date for processing (in '%Y%m%d' or '%Y-%m-%d' format).
        date_end: click.DateTime
            The end date for processing (in '%Y%m%d' or '%Y-%m-%d' format).
        site: str
            The name of the site (must be an uppercase string, at least 2 characters).
        save_nc: bool
            Argument to save the NetCDF file.
        save_wind_ql: bool
            Argument to save the quick-look.
        output_dir: Path
            Creates an output directory.

    Returns:
    -------
        None: This function doesn't return anything but processes the command-line.

    """
    click.echo(f"Config: {conf_file}")
    click.echo(f"Start: {date_start}")
    click.echo(f"End: {date_end}")
    click.echo(f"Site: {site}")

    # ---------------------------------------------------------------------
    # # 0 - Load config file
    # ---------------------------------------------------------------------
    with conf_file.open("rb") as fp:
        conf = tomllib.load(fp)

    # ---------------------------------------------------------------------
    # # 1 - Get list of dates of the study period
    # ---------------------------------------------------------------------
    # check the dates are correct.
    if date_start > date_end:
        msg = "Start date must be before end date."
        raise click.BadParameter(msg)
    init_dates = sorted(set(pd.date_range(date_start, date_end, freq="1D")))

    # ---------------------------------------------------------------------
    # # 2 - create output directory
    # ---------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------
    # # 3 - Loading data
    # ---------------------------------------------------------------------

    loader = DataLoader(conf, site, init_dates)
    loader.load_files()

    # ---------------------------------------------------------------------
    # # 4 - Creating the dataset with the reconstructed wind
    # ---------------------------------------------------------------------

    processor = WindProcessor(
        conf=conf,
        site=site,
        site_conf=loader.site_conf,
        init_dates=init_dates,
        output_dir=output_dir,
        files=loader.files,
    )

    stare_ds = processor.process_stare()
    wind_ds = processor.process_wind()

    # ---------------------------------------------------------------------
    # # 4 - saving NetCDF files with processed data
    # ---------------------------------------------------------------------

    processor.to_netcdf_vertical(stare_ds)
    processor.to_netcdf_horizontal(wind_ds)

    # ---------------------------------------------------------------------
    # # 5 - Plotting and saving QLs with processed data
    # ---------------------------------------------------------------------

    plotter = HorizontalPlots(
        ds=wind_ds,
        conf=conf,
        site=site,
        init_dates=init_dates,
        output_dir=output_dir,
    )

    plotter.do_plot()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger = setup_logger()
        logger.exception("Unexpected error occurred:")
        logger.exception(traceback.format_exc())
        raise SystemExit(1) from None
