"""Created on April 29, 2025.

Plot the Doppler Wind Lidar Data.

@author: Jonnathan Céspedes
"""  # noqa: INP001

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from datetime import timedelta
from typing import TYPE_CHECKING, Any

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(slots=True)
class HorizontalPlots:
    """A class for generating horizontal wind component plots from DWL data."""

    ds: xr.Dataset
    conf: dict[str, Any]
    site: str
    init_dates: list[dt.datetime]
    output_dir: Path

    dir_mask: str = field(init=False)
    site_conf: dict[str, Any] = field(init=False)

    def __post_init__(self) -> None:
        """Validate input data and initialize derived attributes."""
        if not isinstance(self.ds, xr.Dataset) or self.ds.time.size == 0:
            msg = "Input dataset is empty or not a valid xarray.Dataset."
            raise ValueError(msg)

        if not self.init_dates:
            msg = "init_dates list is empty."
            raise ValueError(msg)

        if self.site not in self.conf["sites"]:
            msg = f"Site '{self.site}' not found in configuration."
            raise KeyError(msg)

        self.site_conf = self.conf["sites"][self.site]
        self.dir_mask = self.site_conf["dir_mask"]

    def _add_colorbar(
        self,
        fig: mpl.figure.Figure,
        mesh: mpl.cm.ScalarMappable,
    ) -> None:
        """Add colorbar to the plot."""
        cbar_ax = fig.add_axes(self.conf["QL_params"]["Colorbar_size"])
        cb = fig.colorbar(mesh, cax=cbar_ax, cmap="bwr")
        cb.ax.set_ylabel(
            r"$ms^{-1}$",
            fontsize=self.conf["QL_params"]["Axis_label_font_size"],
        )
        cb.ax.tick_params(labelsize=self.conf["QL_params"]["Ticks_label_font_size"])

    def _plot_u_v(
        self,
        dates: dt.datetime,
        var: xr.DataArray,
        ax: mpl.axes.Axes,
        fig: mpl.figure.Figure,
    ) -> None:
        """Plot quicklook (QL) with u&V wind components data."""
        # Plotting the height-time cross section of horizontal wind speed

        # defining intervals to plot the QL
        start_t = f"{dates:%Y-%m-%d} 00:00"
        end_t = f"{(dates + timedelta(days=1)):%Y-%m-%d} 23:59"

        mesh = ax.pcolormesh(
            self.ds.sel(time=slice(start_t, end_t)).time.values,
            self.ds.range.values,
            var.sel(time=slice(start_t, end_t)).T.values,
            cmap="bwr",
            vmin=-30,
            vmax=30,
        )

        ax.set_facecolor(self.conf["QL_params"]["QL_background_color"])
        # creating the colorbar for the height-time cross section
        self._add_colorbar(fig, mesh)

        # Defining axis labels
        ax.set_xlabel(
            self.conf["QL_params"]["X_label"],
            fontsize=self.conf["QL_params"]["Axis_label_font_size"],
        )
        ax.set_ylabel(
            self.conf["QL_params"]["Y_label"],
            fontsize=self.conf["QL_params"]["Axis_label_font_size"],
        )

        # Defining parameters of axis, ticks, label and other
        ax.grid(axis="y", linestyle="--")
        ax.grid(axis="x", which="minor", linestyle="--")

        plt.setp(ax.spines.values(), lw=1.5)
        ax.tick_params(
            which="major",
            labelsize=self.conf["QL_params"]["Ticks_label_font_size"],
        )

        ax.tick_params(
            axis="x",
            which="minor",
            labelsize=self.conf["QL_params"]["Ticks_label_font_size"],
        )

        # Plotting inside the tickslabels...
        ax.tick_params(axis="both", direction="in")
        ax.tick_params(which="minor", direction="in", length=5)
        ax.tick_params(which="major", direction="in", length=10)
        ax.yaxis.set_ticks_position("both")
        ax.xaxis.set_ticks_position("both")

        # Giving space between the line and the labels...
        ax.tick_params(axis="x", which="minor", pad=10)
        ax.tick_params(axis="x", which="major", pad=25)

        # Xaxis lebel...
        ax.xaxis.set_minor_formatter(mdates.DateFormatter("%H"))
        ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=np.arange(0, 24, 4)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))

        # Yaxis lebel...
        ax.set_ylim(0, self.conf["QL_params"]["Y_axis_lim"])
        ax.yaxis.set_major_locator(
            mticker.MultipleLocator(self.conf["QL_params"]["Y_ticker_max"]),
        )
        ax.yaxis.set_minor_locator(
            mticker.MultipleLocator(self.conf["QL_params"]["Y_ticker_min"]),
        )
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())

    def do_plot(self) -> None:
        """Generate and save horizontal wind component plots for each date."""
        for date in self.init_dates:
            fig, axes = plt.subplots(2, 1, sharex=True, figsize=[12, 10])
            ax = axes.flatten()

            axe = ax[0]
            self._plot_u_v(
                date,
                self.ds.v_wind,
                axe,
                fig,
            )

            axe.set_title(
                "Screened meridional wind (v)",
                fontsize=self.conf["QL_params"]["Axis_label_font_size"],
            )
            axe.set_xlabel("")

            axe = ax[1]
            self._plot_u_v(
                date,
                self.ds.v_wind_raw,
                axe,
                fig,
            )

            axe.set_title(
                "Non-screened meridional wind (v)",
                fontsize=self.conf["QL_params"]["Axis_label_font_size"],
            )

            fig.tight_layout()
            filename = f"{self.site}_Horizontal_wind_component_{date:%Y-%m-%d}.png"
            output_file = self.output_dir / filename
            plt.savefig(output_file, bbox_inches="tight", dpi=300)
            plt.close(fig)
