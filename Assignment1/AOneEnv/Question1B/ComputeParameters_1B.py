# General imports
import numpy as np 
import pandas as pd

# Custom imports 
from Question1A.DistancesLatLong import compute_dij
from Costs import compute_all_C_ijk, leasing_cost


def distances(airport_data):
    '''
    Compute the distance matrix between all airports.
    '''
    airports = airport_data.columns
    len_airports = len(airports)
    
    distance = np.zeros((len_airports, len_airports))
    
    for i, porti in enumerate(airports):
        for j, portj in enumerate(airports):
            if i == j:
                distance[i, j] = 0
            else:
                distance[i, j] = compute_dij(airport_data, porti, portj)
    
    return distance


def load_aircraft_params(path="AircraftData.xlsx"):
    """
    Reads AircraftData.xlsx (layout like your screenshot) and builds
    aircraft_params = {
        "Aircraft 1": {"C_X": ..., "c_T": ..., "V": ..., "c_F": ..., "L": ...},
        ...
    }
    """
    # Use row 2 as header (Aircraft 1..4) and column A as index (Speed, Lease cost, etc.)
    df = pd.read_excel(path, header=1, index_col=0)

    aircraft_params = {}

    for ac_type in df.columns:          # 'Aircraft 1', 'Aircraft 2', ...
        V   = df.at["Speed [km/h]", ac_type]
        L   = df.at["Weekly lease cost [€]", ac_type]
        C_X = df.at["Fixed operating cost C_X [€]", ac_type]
        c_T = df.at["Time cost parameter C_T [€/hr]", ac_type]
        c_F = df.at["Fuel cost parameter C_F", ac_type]

        aircraft_params[ac_type] = {
            "C_X": C_X,
            "c_T": c_T,
            "V":   V,
            "c_F": c_F,
            "L":   L,
        }

    return aircraft_params

