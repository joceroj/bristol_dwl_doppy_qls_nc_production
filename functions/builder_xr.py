"""Created on April 16, 2025.

Module to build a xr.Dataset from doppy output.

@author: Jonnathan Céspedes
"""

from __future__ import (
    annotations,  # This import is use the class in the "with" function
)

from typing import Any

import doppy
import numpy as np
import numpy.typing as npt
import xarray as xr


class XarrayDatasetBuilder:
    """Bulid a xarray Dataset based on the doppy "wind" output."""

    def __init__(self) -> None:
        """Initialize an empty XarrayDatasetBuilder.

        This constructor sets up internal containers to hold coordinates,
        data variables, and global attributes for building an xarray.Dataset.
        """
        self.coords = {}
        self.data_vars = {}
        self.attrs = {}

    def __enter__(self: object) -> XarrayDatasetBuilder:  # noqa: PYI034
        """Enter the context manager and return the builder instance.

        This allows the builder to be used with a `with` statement.

        Returns
        -------
        XarrayDatasetBuilder
            The builder instance for method chaining.

        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,  # noqa: ANN401, PYI036
    ) -> None:
        """Exit the context manager. No cleanup or exception handling is performed."""

    def add_attribute(self, key: str, val: str) -> XarrayDatasetBuilder:
        """Add a global attribute to the dataset.

        Parameters
        ----------
        key : str
            Name of the attribute.
        val : str
            Value of the attribute.

        Returns
        -------
        XarrayDatasetBuilder
            The builder instance for method chaining.

        """
        self.attrs[key] = val
        return self

    def add_time(
        self,
        name: str,
        dimensions: tuple[str, ...],
        data: npt.NDArray[np.datetime64],
        standard_name: str | None = None,
        long_name: str | None = None,
    ) -> XarrayDatasetBuilder:
        """Add a time coordinate.

        Parameters
        ----------
        name : str
            Name of the coordinate.
        dimensions : tuple of str
            Dimensions associated with the time coordinate.
        data : np.ndarray of datetime64[us]
            Time values.
        standard_name : str, optional
            CF-compliant standard name.
        long_name : str, optional
            Descriptive name for the variable.

        Returns
        -------
        XarrayDatasetBuilder
            The builder instance for method chaining.

        """
        self.coords[name] = xr.DataArray(
            data,
            dims=dimensions,
            attrs={
                "calendar": "standard",
                "axis": "T",
                **({"standard_name": standard_name} if standard_name else {}),
                **({"long_name": long_name} if long_name else {}),
            },
        )
        return self

    def add_variable(  # noqa: PLR0913
        self,
        name: str,
        dimensions: tuple[str, ...],
        units: str,
        data: npt.NDArray[np.float64],
        standard_name: str | None = None,
        mask: npt.NDArray[np.bool_] | None = None,
    ) -> XarrayDatasetBuilder:
        """Add a variable from wind data from doppy.

        Parameters
        ----------
        name : str
            Name of the variable.
        dimensions : tuple of str
            Dimensions associated with the coordinates.
        units : str
            Units of the added variable.
        data : np.ndarray of data values.
            Variable values.
        standard_name : str, optional
            Standard variable name.
        mask : np.ndarray of data values.
            Mask to correct raw values.

        Returns
        -------
        XarrayDatasetBuilder
            The builder instance for method chaining.

        """
        if mask is not None:
            data = np.ma.masked_array(data, mask)
        self.data_vars[name] = xr.DataArray(
            data,
            dims=dimensions,
            attrs={
                "units": units,
                **({"standard_name": standard_name} if standard_name else {}),
            },
        )
        return self

    def add_scalar_variable(
        self,
        name: str,
        units: str,
        data: float,
        standard_name: str | None = None,
    ) -> XarrayDatasetBuilder:
        """Add a variable from wind data from doppy.

        Parameters
        ----------
        name : str
            Name of the variable.
        units : str
            Units of the added variable.
        data : np.ndarray of data values.
            Variable values.
        standard_name : str, optional
            Standard variable name..

        Returns
        -------
        XarrayDatasetBuilder
            The builder instance for method chaining.

        """
        self.data_vars[name] = xr.DataArray(
            data,
            dims=(),
            attrs={
                "units": units,
                **({"standard_name": standard_name} if standard_name else {}),
            },
        )
        return self

    def to_xarray(self) -> xr.Dataset:
        """Create the xarray Dataset."""
        return xr.Dataset(
            data_vars=self.data_vars,
            coords=self.coords,
            attrs=self.attrs,
        )


def build_wind_dataset(wind: doppy.product.Wind) -> xr.Dataset:
    """Build a xarray.Dataset from wind object made by doppy."""
    with XarrayDatasetBuilder() as nc:
        if "zonal_wind" in dir(wind):
            # Coordinates
            nc.add_time(
                name="time",
                dimensions=("time",),
                standard_name="time",
                long_name="Time UTC",
                data=wind.time,
            )
            nc.add_variable(
                name="range",
                dimensions=("range",),
                units="m",
                data=wind.height,
            )
            # Horizontal wind components (u, v): each raw and corrected
            for name, data, mask_raw, mask_combined in [
                (
                    "u_wind",
                    wind.zonal_wind,
                    wind.mask_zonal_wind,
                    wind.mask | wind.mask_zonal_wind,
                ),
                (
                    "v_wind",
                    wind.meridional_wind,
                    wind.mask_meridional_wind,
                    wind.mask | wind.mask_meridional_wind,
                ),
                (
                    "w_wind",
                    wind.vertical_wind,
                    wind.mask_vertical_wind,
                    wind.mask | wind.mask_vertical_wind,
                ),
                (
                    "ws",
                    wind.horizontal_wind_speed,
                    wind.mask_meridional_wind,
                    wind.mask | wind.mask_meridional_wind,
                ),
                (
                    "wd",
                    wind.horizontal_wind_direction,
                    wind.mask_meridional_wind,
                    wind.mask | wind.mask_meridional_wind,
                ),
            ]:
                nc.add_variable(
                    name=f"{name}_raw",
                    dimensions=("time", "range"),
                    units="m s-1",
                    data=data,
                    mask=mask_raw,
                )
                nc.add_variable(
                    name=name,
                    dimensions=("time", "range"),
                    units="m s-1",
                    data=data,
                    mask=mask_combined,
                )

            # Optional scalar variable
            if getattr(wind.options, "azimuth_offset_deg", None) is not None:
                nc.add_scalar_variable(
                    name="azimuth_offset",
                    units="degrees",
                    data=wind.options.azimuth_offset_deg,
                )

        if "elevation" in dir(wind):
            # Coordinates
            nc.add_time(
                name="time",
                dimensions=("time",),
                standard_name="time",
                long_name="Time UTC",
                data=wind.time,
            )
            nc.add_variable(
                name="range",
                dimensions=("range",),
                units="m",
                data=wind.radial_distance,
            )
            for name, data, units, mask_raw in [
                (
                    "beta",
                    wind.beta,
                    "sr-1 m-1",
                    wind.mask_beta,
                ),
                (
                    "w_wind",
                    wind.radial_velocity,
                    "m s-1",
                    wind.mask_radial_velocity,
                ),
            ]:
                nc.add_variable(
                    name=f"{name}_raw",
                    dimensions=("time", "range"),
                    units=units,
                    data=data,
                )

                nc.add_variable(
                    name=name,
                    dimensions=("time", "range"),
                    units=units,
                    data=data,
                    mask=mask_raw,
                )

            nc.add_variable(
                name="elevation",
                dimensions=("time",),
                units="degrees",
                data=wind.elevation,
            )

        # Atributtes
        nc.add_attribute("serial_number", wind.system_id)
        nc.add_attribute("doppy_version", doppy.__version__)

    return nc.to_xarray()
