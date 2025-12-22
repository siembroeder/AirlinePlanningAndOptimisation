import pandas as pd
import numpy as np

from compute_parameters import load_data_routes, comp_distances, comp_yield, comp_hourly_demand, load_data_aicraft, comp_operating_costs

"""

cd "C:/Users/pop_r/OneDrive - Delft University of Technology/Documents/SVV/AirlinePlanningAndOptimisation/Assignment2"

"""
airports_path = "C:/Users/pop_r/OneDrive - Delft University of Technology/Documents/SVV/AirlinePlanningAndOptimisation/Assignment2/Data/DemandGroup7.xlsx"
aircraft_path = "C:/Users/pop_r/OneDrive - Delft University of Technology/Documents/SVV/AirlinePlanningAndOptimisation/Assignment2/Data/FleetType.xlsx"
hours_path = "C:/Users/pop_r/OneDrive - Delft University of Technology/Documents/SVV/AirlinePlanningAndOptimisation/Assignment2/Data/HourCoefficients.xlsx"


names, coord, runways, demand, hours = load_data_routes(airports_path, hours_path)

distance = comp_distances(coord)

y = comp_yield(distance)

hourly_demand = comp_hourly_demand(demand, hours)

speed, seats, TAT, range, runway_req, lease_cost =  load_data_aicraft(aircraft_path)[0:6]
fleet = load_data_aicraft(aircraft_path)[-1]

operating_cost = comp_operating_costs(aircraft_path, distance)