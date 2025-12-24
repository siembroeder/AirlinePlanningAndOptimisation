import pandas as pd
import numpy as np

def read_file(file_path, skiprows, usecols):

    # Read EXCEL file

    df = pd.read_excel(file_path, skiprows=skiprows, usecols=usecols)
    return df

def load_data_routes(airports_path, hours_path):

    # Read data from the provided file path
    data = read_file(airports_path, skiprows=2, usecols="C:V")
    hours = read_file(hours_path, skiprows=1, usecols="D:AA")


    # Divide data into different DataFrames based on row ranges
    # and transform into np arrays for easier manipulation
    names = np.array(data.iloc[0, :])
    coord = np.array(data.iloc[2:4, :])
    runways = np.array(data.iloc[4, :])
    demand = np.array(data.iloc[8:28, :])

    hours = np.array(hours)

    return names, coord, runways, demand, hours


def comp_distances(coord):
    
    distance = np.zeros(shape=(20,20))
    
    RE = 6371 # Radius of Earth in km

    for i in range(20):
        for j in range(20):

            lat_dep = np.radians(coord[0][i])
            lat_arr = np.radians(coord[0][j])

            long_dep = np.radians(coord[1][i])
            long_arr = np.radians(coord[1][j])

            sqrt_arg = np.sin((lat_dep-lat_arr)/2)**2 +np.cos(lat_dep)*np.cos(lat_arr)*np.sin((long_dep-long_arr)/2)**2
            d_sigma = 2*np.arcsin(np.sqrt(sqrt_arg))

            distance[i][j] = RE * d_sigma

    return distance


def comp_yield(distance):

    y = np.zeros(shape=(20,20))

    for i in range(20):
        for j in range(20):

            if distance[i][j] == 0:
                y[i][j] = 0

            else:
                y[i][j] = 5.9 * distance[i][j] ** -0.76 + 0.043

    return y

def comp_hourly_demand(demand, hours):

    hourly_demand = np.zeros(shape=(20,20,24))

    for i in range(20):
        for j in range(20):
            for k in range(24):
                hourly_demand[i][j][k] = demand[i][j] * hours[i][k]
    return hourly_demand

def load_data_aicraft(aircraft_path):

    data = read_file(aircraft_path, skiprows=0, usecols="B:D")

    speed = np.array(data.iloc[0, :])
    seats = np.array(data.iloc[1, :])
    TAT = np.array(data.iloc[2, :])
    range = np.array(data.iloc[3, :])
    runway_req = np.array(data.iloc[4, :])
    lease_cost = np.array(data.iloc[5, :])
    fixed_operating_cost = np.array(data.iloc[6, :])
    hourly_cost = np.array(data.iloc[7, :])
    fuel_cost = np.array(data.iloc[8, :])
    fleet = np.array(data.iloc[9, :])

    return speed, seats, TAT, range, runway_req, lease_cost, fixed_operating_cost, hourly_cost, fuel_cost, fleet

def comp_operating_costs(aircraft_path, distance):

    speed = load_data_aicraft(aircraft_path)[0]
    fixed_operating_cost, hourly_cost, fuel_cost = load_data_aicraft(aircraft_path)[6:9]

    FUEL = 1.42
    operating_cost = np.zeros(shape=(20,20,3))

    for i in range(20):
        for j in range(20):
            for k in range(3):
                operating_cost[i][j][k] = fixed_operating_cost[k] + hourly_cost[k] * distance[i][j] / speed[k] + fuel_cost[k] * FUEL / 1.5 * distances[i][j]

    return operating_cost
