"""Short variable names and the WeatherBench 2 (variable, level) of the 0.25 deg result files."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

LEVELS = (500, 700, 850)


@dataclass(frozen=True)
class WB2Var:
    ours: str
    wb2: str
    level: Optional[int]


def _ua(prefix, wb2):
    return [WB2Var(f"{prefix}{lv}", wb2, lv) for lv in LEVELS]


VARS: tuple = tuple(
    _ua("z", "geopotential")
    + _ua("t", "temperature")
    + _ua("q", "specific_humidity")
    + _ua("u", "u_component_of_wind")
    + _ua("v", "v_component_of_wind")
    + _ua("ws", "wind_speed")
    + [WB2Var("msl", "mean_sea_level_pressure", None),
       WB2Var("t2m", "2m_temperature", None),
       WB2Var("u10", "10m_u_component_of_wind", None),
       WB2Var("v10", "10m_v_component_of_wind", None),
       WB2Var("ws10", "10m_wind_speed", None),
       WB2Var("tp24h", "total_precipitation_24hr", None)])

BY_OURS = {v.ours: v for v in VARS}


def present(names) -> list:
    """Scorecard rows available in a per-initialisation score file, in VARS order."""
    have = set(str(n) for n in names)
    return [v for v in VARS if v.ours in have]
