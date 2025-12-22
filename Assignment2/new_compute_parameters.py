"""
Data structure for Assignment 2
Uses dataclasses + pandas DataFrames with airport/aircraft labels
"""

# ---- Imports ----
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


# Dataclass container
@dataclass(frozen=True)
class ProblemData:
    # Core sets
    airports: List[str]                 # list of airport codes/names (length 20)
    aircraft_types: List[str]           # list of aircraft type names (length 3)

    # Route / airport data
    airport_info: pd.DataFrame          # index=airports, cols=["lat_deg","lon_deg","runway_m"]
    demand: pd.DataFrame                # index=airports, cols=airports (OD demand)
    hour_coeff: pd.DataFrame            # index=airports, cols=0..23

    # Derived
    distance_km: pd.DataFrame           # index=airports, cols=airports
    yield_per_km: pd.DataFrame          # index=airports, cols=airports

    # Aircraft data
    aircraft: pd.DataFrame              # index=aircraft_types, cols=[speed_kmh,seats,TAT_h,range_km,runway_req_m,lease_cost,fixed_op_cost,hourly_cost,fuel_cost,fleet]

    # Derived multi-dim parameters
    hourly_demand: pd.DataFrame         # MultiIndex (origin,dest,hour) -> column ["demand"]
    operating_cost: Dict[str, pd.DataFrame]  # per aircraft_type: DataFrame index=airports, cols=airports


# ---- Loading functions ----
def load_data_routes(airports_path: str, hours_path: str) -> tuple[List[str], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Reads route-related Excel files and returns:
    - airports (list[str])
    - airport_info DataFrame with lat/lon/runway
    - demand DataFrame (OD)
    - hour_coeff DataFrame (airport x 24h)
    """

    # Matches your original logic
    # DemandGroup7.xlsx
    aircraft_data = pd.read_excel(airports_path, skiprows=2, usecols="C:V")

    # HourCoefficients.xlsx
    hours_df = pd.read_excel(hours_path, skiprows=1, usecols="D:AA")

    # ---- slicing (as in your current code) ----
    airports = list(aircraft_data.iloc[0, :].astype(str))

    coord = aircraft_data.iloc[2:4, :].to_numpy(dtype=float)    # shape (2, 20)
    runways = aircraft_data.iloc[4, :].to_numpy(dtype=float)    # shape (20,)
    demand = aircraft_data.iloc[8:28, :].to_numpy(dtype=float)  # shape (20, 20)

    # hour coefficients: shape (20, 24)
    hour_coeff = hours_df.to_numpy(dtype=float)

    # Build labelled DataFrames
    airport_info = pd.DataFrame(
        {
            "lat_deg": coord[0, :],
            "lon_deg": coord[1, :],
            "runway_m": runways,
        },
        index=airports,
    )

    demand_df = pd.DataFrame(demand, index=airports, columns=airports)

    hour_cols = list(range(24))
    hour_coeff_df = pd.DataFrame(hour_coeff, index=airports, columns=hour_cols)

    return airports, airport_info, demand_df, hour_coeff_df


def load_data_aircraft(aircraft_path: str) -> tuple[List[str], pd.DataFrame]:
    """
    Reads FleetType.xlsx and returns:
    - aircraft_types: list[str]
    - aircraft_df: index=aircraft_types, columns:
      ["speed_kmh","seats","TAT_h","range_km","runway_req_m","lease_cost",
       "fixed_op_cost","hourly_cost","fuel_cost","fleet"]
    """

    raw = pd.read_excel(aircraft_path, skiprows=0, usecols="B:D")

    # Column headers should be aircraft type names; if not, fall back to generic names
    aircraft_types = [str(c) for c in raw.columns]
    if any(a.startswith("Unnamed") for a in aircraft_types):
        aircraft_types = [f"type_{i+1}" for i in range(len(aircraft_types))]

    # Rows are parameters in fixed order (matching your original iloc rows)
    params = [
        "speed_kmh",
        "seats",
        "TAT_h",
        "range_km",
        "runway_req_m",
        "lease_cost",
        "fixed_op_cost",
        "hourly_cost",
        "fuel_cost",
        "fleet",
    ]

    if raw.shape[0] < len(params):
        raise ValueError(
            f"Expected at least {len(params)} rows in aircraft file, got {raw.shape[0]}."
        )

    param_matrix = raw.iloc[0:len(params), :].to_numpy(dtype=float)  # shape (10, 3)
    aircraft_types = ["T1", "T2", "T3"] # Overwrite the name of the aircrafts for convenience

    aircraft_df = pd.DataFrame(param_matrix, index=params, columns=aircraft_types).T
    return aircraft_types, aircraft_df



# ---- Compute functions ----

def comp_distances_km(airport_info: pd.DataFrame) -> pd.DataFrame:
    """
    Great-circle distance in km using the same formula as your NumPy version.
    Returns a DataFrame index=airports, cols=airports.
    """
    airports = list(airport_info.index)
    lat = np.radians(airport_info["lat_deg"].to_numpy(dtype=float))
    lon = np.radians(airport_info["lon_deg"].to_numpy(dtype=float))

    n = len(airports)
    RE = 6371.0  # km

    dist = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(n):
            sqrt_arg = (
                np.sin((lat[i] - lat[j]) / 2.0) ** 2
                + np.cos(lat[i]) * np.cos(lat[j]) * np.sin((lon[i] - lon[j]) / 2.0) ** 2
            )
            d_sigma = 2.0 * np.arcsin(np.sqrt(sqrt_arg))
            dist[i, j] = RE * d_sigma

    return pd.DataFrame(dist, index=airports, columns=airports)


def comp_yield(distance_km: pd.DataFrame) -> pd.DataFrame:
    """
    Yield matrix, same piecewise logic as your current code:
      if distance==0 -> 0
      else -> 5.9 * d^-0.76 + 0.043
    """
    d = distance_km.to_numpy(dtype=float)
    y = np.zeros_like(d)

    mask = d != 0
    y[mask] = 5.9 * (d[mask] ** (-0.76)) + 0.043

    return pd.DataFrame(y, index=distance_km.index, columns=distance_km.columns)


def comp_hourly_demand(demand: pd.DataFrame, hour_coeff: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a *long* DataFrame with a MultiIndex (origin, dest, hour) and one column: ["demand"].
    This avoids a messy 20x20x24 NumPy cube while staying very explicit.
    """
    airports = list(demand.index)
    hours = list(hour_coeff.columns)

    rows = []
    for i in airports:
        coeff_i = hour_coeff.loc[i, :].to_numpy(dtype=float)  # (24,)
        for j in airports:
            base = float(demand.loc[i, j])
            # vector of 24 values
            hd = base * coeff_i
            for h_idx, h in enumerate(hours):
                rows.append((i, j, int(h), float(hd[h_idx])))

    out = pd.DataFrame(rows, columns=["origin", "dest", "hour", "demand"])
    out.set_index(["origin", "dest", "hour"], inplace=True)
    return out


def comp_operating_costs(
    aircraft: pd.DataFrame,
    distance_km: pd.DataFrame,
    fuel_price_usd_per_gallon: float = 1.42,
) -> Dict[str, pd.DataFrame]:
    """
    Operating cost per leg per aircraft type.
    Keeps your exact formula structure, but returns a dict:
      operating_cost[type] -> DataFrame (origin x dest)
    """
    airports = list(distance_km.index)
    dist = distance_km.to_numpy(dtype=float)

    out: Dict[str, pd.DataFrame] = {}

    for ac_type, row in aircraft.iterrows():
        speed = float(row["speed_kmh"])
        fixed = float(row["fixed_op_cost"])
        hourly = float(row["hourly_cost"])
        fuel_cost = float(row["fuel_cost"])

        # Same expression as your original code:
        # fixed + hourly * (d / speed) + fuel_cost * FUEL * d / 1.5
        cost = fixed + hourly * (dist / speed) + (fuel_cost * fuel_price_usd_per_gallon * dist / 1.5)

        out[str(ac_type)] = pd.DataFrame(cost, index=airports, columns=airports)

    return out


# ---- Build problem data ----
def build_problem_data(
    airports_path: str,
    hours_path: str,
    aircraft_path: str,
) -> ProblemData:
    """
    Loads everything and computes all derived parameters.
    """
    airports, airport_info, demand, hour_coeff = load_data_routes(airports_path, hours_path)
    aircraft_types, aircraft = load_data_aircraft(aircraft_path)

    distance_km = comp_distances_km(airport_info)
    yield_per_km = comp_yield(distance_km)
    hourly_demand = comp_hourly_demand(demand, hour_coeff)
    operating_cost = comp_operating_costs(aircraft, distance_km)

    return ProblemData(
        airports=airports,
        aircraft_types=aircraft_types,
        airport_info=airport_info,
        demand=demand,
        hour_coeff=hour_coeff,
        distance_km=distance_km,
        yield_per_km=yield_per_km,
        aircraft=aircraft,
        hourly_demand=hourly_demand,
        operating_cost=operating_cost,
    )