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
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(slots=True)
class BasePlotter:
    """Base class containing shared logic for plotting DWL data."""

    ds: xr.Dataset
    conf: dict[str, Any]
    site: str
    init_dates: list[dt.datetime]
    output_dir: Path

    dir_mask: str = field(init=False)
    site_conf: dict[str, Any] = field(init=False)

    def __post_init__(self) -> None:
        """Validate input data and initialize site-specific configuration."""
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

    def _plot_component(  # noqa: PLR0913
        self,
        dates: dt.datetime,
        var: xr.DataArray,
        ax: mpl.axes.Axes,
        fig: mpl.figure.Figure,
        cmap: str | mcolors.Colormap,
        vmin: float | None,
        vmax: float | None,
        label_colorbar: str,
        norm: mcolors.Normalize | None = None,
    ) -> None:
        """Plot height-time cross-section of a wind variable."""
        start_t = f"{dates:%Y-%m-%d} 00:00"
        end_t = f"{(dates + timedelta(days=1)):%Y-%m-%d} 00:01"

        subset = self.ds.sel(time=slice(start_t, end_t))
        data = var.sel(time=slice(start_t, end_t)).T

        mesh_kwargs = {"cmap": cmap}
        if norm is not None:
            mesh_kwargs["norm"] = norm
        else:
            mesh_kwargs["vmin"] = vmin
            mesh_kwargs["vmax"] = vmax

        mesh = ax.pcolormesh(
            subset.time.values,
            self.ds.range.values,
            data.values,
            shading="auto",
            **mesh_kwargs,
        )
        ax.set_facecolor(self.conf["QL_params"]["QL_background_color"])

        cb = fig.colorbar(mesh, ax=ax, pad=0.02, aspect=20)
        cb.ax.set_ylabel(
            label_colorbar,
            fontsize=self.conf["QL_params"]["Axis_label_font_size"],
        )
        cb.ax.tick_params(labelsize=self.conf["QL_params"]["Ticks_label_font_size"])

        ax.set_xlabel(
            self.conf["QL_params"]["X_label"],
            fontsize=self.conf["QL_params"]["Axis_label_font_size"],
        )
        ax.set_ylabel(
            self.conf["QL_params"]["Y_label"],
            fontsize=self.conf["QL_params"]["Axis_label_font_size"],
        )

        ax.grid(axis="y", linestyle="--")
        ax.grid(axis="x", which="minor", linestyle="--")
        plt.setp(ax.spines.values(), lw=1.5)

        ax.tick_params(
            which="major", labelsize=self.conf["QL_params"]["Ticks_label_font_size"]
        )
        ax.tick_params(
            axis="x",
            which="minor",
            labelsize=self.conf["QL_params"]["Ticks_label_font_size"],
        )

        ax.tick_params(axis="both", direction="in")
        ax.tick_params(which="minor", direction="in", length=5)
        ax.tick_params(which="major", direction="in", length=10)
        ax.yaxis.set_ticks_position("both")
        ax.xaxis.set_ticks_position("both")

        ax.tick_params(axis="x", which="minor", pad=10)
        ax.tick_params(axis="x", which="major", pad=25)

        ax.xaxis.set_minor_formatter(mdates.DateFormatter("%H"))
        ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=np.arange(0, 24, 4)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))

        ax.set_ylim(0, self.conf["QL_params"]["Y_axis_lim"])
        ax.yaxis.set_major_locator(
            mticker.MultipleLocator(self.conf["QL_params"]["Y_ticker_max"]),
        )
        ax.yaxis.set_minor_locator(
            mticker.MultipleLocator(self.conf["QL_params"]["Y_ticker_min"]),
        )
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())


@dataclass(slots=True)
class HorizontalPlots(BasePlotter):
    """A class for generating horizontal wind component plots from DWL data."""

    def _u_v(
        self,
        dates: dt.datetime,
        var: xr.DataArray,
        ax: mpl.axes.Axes,
        fig: mpl.figure.Figure,
    ) -> None:
        """Plot quicklook (QL) with u&V wind components data."""
        self._plot_component(
            dates=dates,
            var=var,
            ax=ax,
            fig=fig,
            cmap="RdBu_r",
            vmin=-20,
            vmax=20,
            label_colorbar=r"$ms^{-1}$",
        )

    def _custom_ws_colormap(self) -> tuple[mcolors.Colormap, mcolors.Normalize]:
        """Create custom colormap for wind speed with white low values."""
        vmin = 0
        vmax = 30
        base_cmap = plt.get_cmap("viridis")
        colors = base_cmap(np.linspace(0, 1, 256))
        colors[:10] = [1, 1, 1, 1]  # white for the lowest 10 bins
        cmap = mcolors.ListedColormap(colors)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        return cmap, norm

    def _ws(
        self,
        dates: dt.datetime,
        var: xr.DataArray,
        ax: mpl.axes.Axes,
        fig: mpl.figure.Figure,
    ) -> None:
        """Plot quicklook (QL) with horizontal wind speed."""
        cmap, norm = self._custom_ws_colormap()
        self._plot_component(
            dates=dates,
            var=var,
            ax=ax,
            fig=fig,
            cmap=cmap,
            vmin=0,
            vmax=30,
            norm=norm,
            label_colorbar=r"Wind speed [$ms^{-1}$]",
        )

    def _wd(
        self,
        dates: dt.datetime,
        var: xr.DataArray,
        ax: mpl.axes.Axes,
        fig: mpl.figure.Figure,
    ) -> None:
        """Plot quicklook (QL) with horizontal wind direction."""
        self._plot_component(
            dates=dates,
            var=var,
            ax=ax,
            fig=fig,
            cmap="hsv",
            vmin=0,
            vmax=360,
            label_colorbar=r"Wind direction [°]",
        )

    def do_plot_hor(self) -> None:
        """Generate and save horizontal wind component plots for each date."""
        for date in self.init_dates:
            for component, screened, raw, label in [
                ("Meridional", self.ds.v_wind, self.ds.v_wind_raw, "v"),
                ("Zonal", self.ds.u_wind, self.ds.u_wind_raw, "u"),
            ]:
                fig, axes = plt.subplots(
                    2,
                    1,
                    sharex=True,
                    figsize=[11, 9],
                    constrained_layout=True,
                )
                ax = axes.flatten()

                self._u_v(date, screened, ax[0], fig)
                ax[0].set_title(
                    f"Screened {component.lower()} wind ({label})",
                    fontsize=self.conf["QL_params"]["Axis_label_font_size"],
                )
                ax[0].set_xlabel("")

                self._u_v(date, raw, ax[1], fig)
                ax[1].set_title(
                    f"Non-screened {component.lower()} wind ({label})",
                    fontsize=self.conf["QL_params"]["Axis_label_font_size"],
                )

                filename = f"{self.site}_{component}_wind_component_{date:%Y-%m-%d}.png"
                output_file = self.output_dir / filename
                plt.savefig(output_file, bbox_inches="tight", dpi=300)
                plt.close(fig)

            # Plot WS and WD
            fig, axes = plt.subplots(
                2,
                1,
                sharex=True,
                figsize=[11, 9],
                constrained_layout=True,
            )
            ax = axes.flatten()

            self._ws(date, self.ds.ws, ax[0], fig)
            ax[0].set_xlabel("")

            self._wd(date, self.ds.wd, ax[1], fig)

            filename = f"{self.site}_Horizontal_ws_wd_{date:%Y-%m-%d}.png"
            output_file = self.output_dir / filename
            plt.savefig(output_file, bbox_inches="tight", dpi=300)
            plt.close(fig)


@dataclass(slots=True)
class VerticalPlots(BasePlotter):
    """A class for generating vertical stare products from DWL data."""

    def _w(
        self,
        dates: dt.datetime,
        var: xr.DataArray,
        ax: mpl.axes.Axes,
        fig: mpl.figure.Figure,
    ) -> None:
        """Plot quicklook (QL) with w wind component data."""
        self._plot_component(
            dates=dates,
            var=var,
            ax=ax,
            fig=fig,
            cmap="RdBu_r",
            vmin=-3,
            vmax=3,
            label_colorbar=r"$ms^{-1}$",
        )

    def _custom_beta_colormap(self) -> tuple[mcolors.Colormap, mcolors.Normalize]:
        """Create custom colormap for wind speed with white low values."""
        vmin = 10e-7
        vmax = 10e-4
        cmap = "viridis"
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        return cmap, norm

    def _beta(
        self,
        dates: dt.datetime,
        var: xr.DataArray,
        ax: mpl.axes.Axes,
        fig: mpl.figure.Figure,
    ) -> None:
        """Plot quicklook (QL) with backscattering data."""
        cmap, norm = self._custom_beta_colormap()
        self._plot_component(
            dates=dates,
            var=var,
            ax=ax,
            fig=fig,
            cmap=cmap,
            vmin=10e-7,
            vmax=10e-4,
            norm=norm,
            label_colorbar=r"$sr^{-1}m^{-1}$",
        )

    def do_plot_ver(self) -> None:
        """Generate and save horizontal wind component plots for each date."""
        for date in self.init_dates:
            # Plot w and beta
            fig, axes = plt.subplots(
                2,
                1,
                sharex=True,
                figsize=[11, 9],
                constrained_layout=True,
            )
            ax = axes.flatten()

            self._w(date, self.ds.w_wind, ax[0], fig)
            ax[0].set_xlabel("")

            self._beta(date, self.ds.beta, ax[1], fig)

            filename = f"{self.site}_Vertical_w_beta_{date:%Y-%m-%d}.png"
            output_file = self.output_dir / filename
            plt.savefig(output_file, bbox_inches="tight", dpi=300)
            plt.close(fig)
