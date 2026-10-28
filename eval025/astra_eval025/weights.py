"""WeatherBench 2 cell-area latitude weights of an equiangular grid with poles.

The latitudes are linspace(-90, 90, lat_size); the weight of a latitude row is the area of its band,
sin(phi_upper) - sin(phi_lower), with the outermost bounds at +/-90 deg (pole rows are half cells), normalised to
mean 1, as in weatherbench2.metrics.get_lat_weights.
"""
from __future__ import annotations

import numpy as np


def latitude_weights(lat_size: int):
    """(lat_size,) float32 cell-area weights of the latitudes linspace(-90, 90, lat_size), mean 1."""
    import jax.numpy as jnp
    phi = np.deg2rad(np.linspace(-90.0, 90.0, lat_size, dtype=np.float64))
    mid = 0.5 * (phi[1:] + phi[:-1])
    bounds = np.concatenate([phi[:1], mid, phi[-1:]])
    w = np.sin(bounds[1:]) - np.sin(bounds[:-1])
    w = w / w.mean()
    return jnp.asarray(w, jnp.float32)


def weights_for_points(lat_size: int, lon_size: int):
    """(P,) weights for a flattened grid P = lon_size * lat_size with latitude varying fastest."""
    import jax.numpy as jnp
    w_lat = latitude_weights(lat_size)
    w = jnp.broadcast_to(w_lat[None, :], (lon_size, lat_size)).reshape(-1)
    return w


def area_weights(lat: int, lon: int):
    """(P,) float64 copy of the float32 cell-area weights, latitude varying fastest (the layout the scorer expects)."""
    return np.asarray(weights_for_points(int(lat), int(lon)), np.float64)
