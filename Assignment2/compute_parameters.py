# Import modules
from dataclasses import dataclass
import pandas as pd
import numpy as np


def load_data_routes(airports_path, hours_path):
    """This function transforms aircraft_data from excel files to np arrays using their file paths"""

    # Read aircraft_data from the provided file path
    aircraft_data = pd.read_excel(airports_path, skiprows=2, usecols="C:V")
    hours = pd.read_excel(hours_path, skiprows=1, usecols="D:AA")

    # Divide aircraft_data into different DataFrames based on row ranges and transform into np.arrays for easier manipulation
    destinations = np.array(aircraft_data.iloc[0, :])
    coord = np.array(aircraft_data.iloc[2:4, :])
    runways = np.array(aircraft_data.iloc[4, :])
    demand = np.array(aircraft_data.iloc[8:28, :])

    # Do the same for hours
    hours = np.array(hours)

    return destinations, coord, runways, demand, hours


def comp_distances(coord):
    """This function computes distances in km from coordinates"""

    distance = np.zeros(shape=(20,20)) # Set up an array with zeroes
    RE = 6371 # Radius of Earth in km

    # Loop over all entries in the distance matrix and calculate the correct distance in km
    for i in range(20):
        for j in range(20):

            # Retrieve latitute and longitude of departure and arrival airport 
            lat_dep = np.radians(coord[0][i])
            lat_arr = np.radians(coord[0][j])

            long_dep = np.radians(coord[1][i])
            long_arr = np.radians(coord[1][j])

            # Calculate distance between departure and arrival airport
            sqrt_arg = np.sin((lat_dep - lat_arr) / 2) ** 2 + np.cos(lat_dep) * np.cos(lat_arr) * np.sin((long_dep-long_arr) / 2) ** 2
            d_sigma = 2 * np.arcsin(np.sqrt(sqrt_arg))
            distance[i][j] = RE * d_sigma

    return distance


def comp_yield(distance):
    """This function computes the yield of a given flight"""

    # Set up dummy matrix of shape 20x20 with only zeroes
    YIELD = np.zeros(shape=(20,20))

    # Loop over all entries of the dummy matrix
    for i in range(20):
        for j in range(20):
            
            # If distance equals zero, the yield is 0
            if distance[i][j] == 0:
                YIELD[i][j] = 0

            # If the distance is nonzero, the yield is calculated with the formula from appendix B
            else:
                YIELD[i][j] = 5.9 * distance[i][j] ** -0.76 + 0.043

    return YIELD


def comp_hourly_demand(demand, hours):
    """This function computes the hourly demand"""

    # Set up a dummy matrix with only zeroes
    hourly_demand = np.zeros(shape=(20,20,24))

    # Loop over all entries in the dummy matrix
    for i in range(20):
        for j in range(20):
            for k in range(24):
                
                # Hourly demand equals total demand times  the hourly coefficients 
                hourly_demand[i][j][k] = demand[i][j] * hours[i][k]
    
    return hourly_demand


def load_aircraft_data(aircraft_path):
    """This function loads the data of different aircraft and returns their parameters"""

    # Create dataframe from aircraft data excel
    aircraft_data = pd.read_excel(aircraft_path, skiprows=0, usecols="B:D")

    # Load all aircraft data into numpy arrays
    speed = np.array(aircraft_data.iloc[0, :])
    seats = np.array(aircraft_data.iloc[1, :])
    TAT = np.array(aircraft_data.iloc[2, :])
    range = np.array(aircraft_data.iloc[3, :])
    runway_req = np.array(aircraft_data.iloc[4, :])
    lease_cost = np.array(aircraft_data.iloc[5, :])
    fixed_operating_cost = np.array(aircraft_data.iloc[6, :])
    hourly_cost = np.array(aircraft_data.iloc[7, :])
    fuel_cost = np.array(aircraft_data.iloc[8, :])
    fleet = np.array(aircraft_data.iloc[9, :])

    return speed, seats, TAT, range, runway_req, lease_cost, fixed_operating_cost, hourly_cost, fuel_cost, fleet


def comp_operating_costs(aircraft_path, distance):
    """This function computes the operating cost per aircraft for each flight leg"""

    # Define cost and speed parameters
    speed = load_aircraft_data(aircraft_path)[0]
    fixed_operating_cost, hourly_cost, fuel_cost = load_aircraft_data(aircraft_path)[6:9]
    FUEL = 1.42 # Fuel price in USD/gallon

    # Set up dummy array with zeroes
    operating_cost = np.zeros(shape=(20,20,3))

    # Loop over all entries in the array
    for i in range(20):
        for j in range(20):
            for k in range(3):

                # Total operating cost for a flight leg 
                operating_cost[i][j][k] = fixed_operating_cost[k] + (hourly_cost[k] * distance[i][j] / speed[k]) + (fuel_cost[k] * FUEL * distance[i][j] / 1.5)

    return operating_cost
