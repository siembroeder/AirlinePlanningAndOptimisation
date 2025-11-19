# Import self-written functions
from Read_input import read_excel_pandas
from DistancesLatLong import compute_pairwise_distance, compute_dij
from GravityOLS import execute_OLS_fit, convert_to_ij_format
from PredictDemand import predict_demand_gravity, extrapolate_gdp_pop, plot_demand_cities


# Import packages
import statsmodels.api as sm # Error but works so ... ?
import numpy   as np
import pandas  as pd
from   pathlib import Path

# Constants:
FUEL = 1.42 # Eur/Gallon


def main():
    # aircraft_path = r"C:\Users\siemb\Documents\Year5\AirlinePlannningOptimisation\Assignment1\Data\AircraftData.xlsx"

    pop_path   = r"C:\Users\siemb\Documents\Year5\AirlinePlannningOptimisation\Assignments\Assignment1\Data\pop_gdp.xlsx"
    pop_sheets = ["GDP_country","Population_city"]

    demand_path   = r"C:\Users\siemb\Documents\Year5\AirlinePlannningOptimisation\Assignments\Assignment1\Data\DemandGroup7.xlsx"
    demand_sheets = ["Airport_data", "Demand_week"]

    GDP_data, pop_data             = read_excel_pandas(pop_path, pop_sheets, indx=None)
    airport_data, demand_data_week = read_excel_pandas(demand_path, demand_sheets)  # type(airport_data)=df, type(d_d_w)=matrix
    demand_data_2021               = demand_data_week.values * 52                   # Demand per year
    
    # print(airport_data.head())
    # print(GDP_data.head())
    # print(pop_data.head())
    # print(GDP_data.loc['PRT'].keys())
    
    ij_data_2021   = convert_to_ij_format(airport_data, GDP_data, pop_data, year=2021, f=FUEL, demand_data=demand_data_2021)
    gravity_coeffs = execute_OLS_fit(ij_data_2021)

    ij_data_2024   = convert_to_ij_format(airport_data, GDP_data, pop_data, year=2024, f=FUEL)
    demand_2024_ij = predict_demand_gravity(gravity_coeffs, ij_data_2024) 

    gdp_2026, pop_2026 = extrapolate_gdp_pop(GDP_data, pop_data, pred_year=2026)

    ij_data_2026       = convert_to_ij_format(airport_data, gdp_2026, pop_2026, year=2026, f=FUEL)
    demand_2026_ij     = predict_demand_gravity(gravity_coeffs, ij_data_2026)
 
    plot_demand_cities(ij_data_2021, demand_2024_ij, demand_2026_ij, [('London', 'Paris'), ('Paris', 'London'), ('Amsterdam', 'Berlin'), ('Frankfurt', 'Madrid'), ('Warsaw', 'Berlin')])
    
    
    


if __name__ == "__main__":
    main()
