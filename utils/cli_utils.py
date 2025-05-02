"""Created on April 29, 2025.

Script to create time-height cross sections (quick-look QLs) from Doppler Wind Lidar.

@author: Jonnathan Céspedes <j.cespedes@reading.ac.uk>
"""  # noqa: INP001

import click


def validate_site(ctx, param, value: str) -> None:  # noqa: ANN001, ARG001, D417
    """Validate that 'site' argument is an uppercase string, at least two characters.

    Parameters
    ----------
        ctx (click.Context): The Click execution context (unused).
        param (click.Parameter): The parameter being processed (unused).
        value (str): The user-provided value for the 'site' argument.

    Returns
    -------
        str: The validated site name.

    Raises
    ------
        click.BadParameter: If the value is not uppercase or too short.

    """
    n_min = 2
    if not value.isupper():
        msg = "site name must be in uppercase letters"
        raise click.BadParameter(msg)
    if len(value) < n_min:
        msg = "site name must be at least 2 characters"
        raise click.BadParameter(msg)
    return value
