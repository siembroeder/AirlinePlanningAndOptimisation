import numpy as np
import pandas as pd

# Loading data from Excel files

def load_data_routes(airports_path, hours_path):

    """Load route data: airports, airport_info, demand, hour_coeff"""

    aircraft_data = pd.read_excel(airports_path, skiprows=2, usecols="C:V")
    hours_df = pd.read_excel(hours_path, skiprows=1, usecols="D:AA")

    airports = list(aircraft_data.iloc[0, :].astype(str))
    coord = aircraft_data.iloc[2:4, :].to_numpy(dtype=float)
    runways = aircraft_data.iloc[4, :].to_numpy(dtype=float)
    demand = aircraft_data.iloc[8:28, :].to_numpy(dtype=float)
    hour_coeff = hours_df.to_numpy(dtype=float)

    airport_info = pd.DataFrame(
        {"lat": coord[0, :], "lon": coord[1, :], "runway": runways},
        index=airports,
    )

    demand_df = pd.DataFrame(demand, index=airports, columns=airports)
    hour_coeff_df = pd.DataFrame(hour_coeff, index=airports, columns=range(24))

    return airports, airport_info, demand_df, hour_coeff_df


def load_data_aircraft(aircraft_path):

    """Load aircraft data"""

    data = pd.read_excel(aircraft_path, skiprows=0, usecols="B:D")
    aircraft_types = ["Prop", "SmallJet", "BigJet"]
    
    params = ["speed", "seats", "TAT", "range", "runway",
              "lease_cost", "fixed_op_cost", "hourly_cost", "fuel_cost", "fleet"]
    
    param_matrix = data.iloc[:len(params), :].to_numpy(dtype=float)
    aircraft_df = pd.DataFrame(param_matrix, index=params, columns=aircraft_types).T
    
    return aircraft_types, aircraft_df


# Compute parameters

def comp_distances(airport_info):

    """Great-circle distance in km"""

    airports = list(airport_info.index)
    lat = np.radians(airport_info["lat"].to_numpy(dtype=float))
    lon = np.radians(airport_info["lon"].to_numpy(dtype=float))

    n = len(airports)
    RE = 6371.0
    dist = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            sqrt_arg = (np.sin((lat[i] - lat[j]) / 2) ** 2 +
                        np.cos(lat[i]) * np.cos(lat[j]) * np.sin((lon[i] - lon[j]) / 2) ** 2)
            dist[i, j] = RE * 2 * np.arcsin(np.sqrt(sqrt_arg))
    
    distance = pd.DataFrame(dist, index=airports, columns=airports)

    return distance


def comp_yield(distance):

    """Yield matrix in revenue-passenger-km"""

    d = distance.to_numpy(dtype=float)
    y = np.zeros_like(d)

    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            if d[i, j] == 0:
                y[i, j] = 0
            else:
                y[i, j] = 5.9 * (d[i, j] ** (-0.76)) + 0.043

    yield_rpk = pd.DataFrame(y, index=distance.index, columns=distance.columns)

    return yield_rpk

def comp_hourly_demand(demand, hour_coeff):

    """Hourly demand with MultiIndex (origin, dest, hour)"""

    demand_per_hour = []
    for i in demand.index:
        coeff_i = hour_coeff.loc[i].to_numpy(dtype=float)
        for j in demand.columns:
            base = float(demand.loc[i, j])
            for h, coeff in enumerate(coeff_i):
                demand_per_hour.append((i, j, h, base * coeff))

    hourly_demand = pd.DataFrame(demand_per_hour, columns=["origin", "dest", "hour", "demand"])
    hourly_demand = hourly_demand.set_index(["origin", "dest", "hour"])["demand"]

    return hourly_demand


def comp_operating_costs(aircraft, distance, fuel_price=1.42):

    """Operating cost per leg per aircraft type"""

    airports = list(distance.index)
    dist = distance.to_numpy(dtype=float)
    operating_cost = {}

    for ac_type, row in aircraft.iterrows():
        cost = (float(row["fixed_op_cost"]) + 
                float(row["hourly_cost"]) * (dist / float(row["speed"])) +
                float(row["fuel_cost"]) * fuel_price * dist / 1.5)
        operating_cost[ac_type] = pd.DataFrame(cost, index=airports, columns=airports)

    return operating_cost

def build_problem_data(airports_path, hours_path, aircraft_path):

    """Load and compute all parameters"""

    airports, airport_info, demand, hour_coeff = load_data_routes(airports_path, hours_path)
    aircraft_types, aircraft = load_data_aircraft(aircraft_path)

    return {
        "airports": airports,
        "aircraft_types": aircraft_types,
        "airport_info": airport_info,
        "demand": demand,
        "hour_coeff": hour_coeff,
        "distance": comp_distances(airport_info),
        "yield": comp_yield(comp_distances(airport_info)),
        "aircraft": aircraft,
        "hourly_demand": comp_hourly_demand(demand, hour_coeff),
        "operating_cost": comp_operating_costs(aircraft, comp_distances(airport_info)),
    }
