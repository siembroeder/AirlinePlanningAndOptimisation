# Import modules
import pandas as pd
import numpy as np
from pathlib import Path

# Import functions
from compute_parameters import load_data_routes, comp_distances, comp_yield, comp_hourly_demand, load_aircraft_data, comp_operating_costs

# Define paths of input data
BASE_DIR = Path(__file__).resolve().parent
airports_path = BASE_DIR / "Data" / "DemandGroup7.xlsx"
aircraft_path = BASE_DIR / "Data" / "FleetType.xlsx"
hours_path    = BASE_DIR / "Data" / "HourCoefficients.xlsx"

# Load parameters from excel files
destinations, coord, runways, demand, hours = load_data_routes(airports_path, hours_path)

# Compute distances between destinations from coordinates
distance = comp_distances(coord)

# Compute the yield between all airports
YIELD = comp_yield(distance)

# Compute demand per hour for all routes
hourly_demand = comp_hourly_demand(demand, hours)

# Load aircraft data for fleet
speed, seats, TAT, ac_range, runway_req, lease_cost =  load_aircraft_data(aircraft_path)[0:6]
n_ac_in_fleet = load_aircraft_data(aircraft_path)[-1]

# Compute total operating cost per flight leg for each aircraft
operating_cost = comp_operating_costs(aircraft_path, distance)

# print(f"operating cost: {operating_cost}")


# # Find indices for London and Helsinki
# i = int(np.where(destinations == "London")[0][0])
# j = int(np.where(destinations == "Helsinki")[0][0])

# # Pick aircraft type k = 0/1/2
# # 0 = Type 1 (turboprop), 1 = Type 2 (regional jet), 2 = Type 3 (single-aisle)
# for k in range(3):
#     print(k, operating_cost[i, j, k])