# Imports
from new_compute_parameters import build_problem_data
from pathlib import Path

# Define paths of input data
BASE_DIR = Path(__file__).resolve().parent
airports_path = BASE_DIR / "Data" / "DemandGroup7.xlsx"
aircraft_path = BASE_DIR / "Data" / "FleetType.xlsx"
hours_path    = BASE_DIR / "Data" / "HourCoefficients.xlsx"

# Build data 
data = build_problem_data(airports_path, hours_path, aircraft_path)

# Examples for usage:
print(f"distance London-Helsinki: {data.distance_km.loc["London", "Helsinki"]}")
print(f"Operating cost type 1 London-Helsinki: {data.operating_cost["T2"].loc["London", "Helsinki"]}")   # (type name depends on your Excel column header)
print(f"Hourly demand London-Helsinki: {data.hourly_demand.loc[("London", "Helsinki"), "demand"]}")

# Print toal obtained data
# print(f"Total data: {data}")