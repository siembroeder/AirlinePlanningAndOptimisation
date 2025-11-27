# General imports
import numpy as np 
import pandas as pd

# Custom imports 
from Assignment1.AOneEnv.Question1A.DistancesLatLong import compute_dij


# Constants 
FUEL_PRICE = 1.42


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


def yields(distances):

    '''
    Compute the yield matrix based on the distance matrix.
    '''
    y = np.zeros(distances.shape)
    for i in range(distances.shape[0]):
        for j in range(distances.shape[0]):
            if i == j:
                y[i,j] = 0
            else:
                y[i,j] = 5.9 * distances[i,j]**-0.76 + 0.043
    return y
    

def load_aircraft_params(path):
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


def C_X_k(ac_type, aircraft_params):
    """
    :param ac_type: Aircraft type
    :param aircraft_params: Aircraft parameters
    
    This function calculates the fixed operating costs per flight leg and represent costs such as landing rights, 
    parking fees and fixed fuel cost. They depend only on the aircraft type k"""

    return aircraft_params[ac_type]["C_X"]


def C_T_ij_k(d_ij, ac_type, aircraft_params):
    """  
    :param d_ij: The distance in km between origin i and destination j
    :param ac_type: Type of aircraft used in flight leg
    :param aircraft_params: Aircraft parameters
    
    This function calculates the time-based operating costs. 
    These are defined in euros per flight hour and represent dime-dependent operating costs such as cabin- and flight crew.
    They depend only on the distance of the flight leg and the aircraft type k."""

    c_T = aircraft_params[ac_type]["c_T"]
    V   = aircraft_params[ac_type]["V"]
    
    return c_T * (d_ij / V)


def C_F_ij_k(d_ij, ac_type, aircraft_params, fuel_price=FUEL_PRICE):
    """
    :param d_ij: The distance in km between origin i and destination j
    :param ac_type: Aircraft type
    :param aircraft_params: Aircraft parameters
    :param fuel_price: Fuel price
    """
    c_F = aircraft_params[ac_type]["c_F"]
    
    return c_F * (fuel_price ** 1.5) * d_ij


def C_ij_k(origin, dest, d_ij, ac_type, aircraft_params, hub):
    """
    :param origin: Origin airport
    :param dest: Destination airport
    :param d_ij: Distance in km between origin and destination
    :param ac_type: Aircraft type
    :param aircraft_params: Aircraft parameters
    :param hub: Hub airport
    
    This function calculates the total operating costs for a flight leg between airports i and j operated by aircraft type k. 
    This takes into account that flights departing or landing at the hub airport have 30% lower operating costs due to econonmics of scale. """

    Cx = C_X_k(ac_type, aircraft_params)
    Ct = C_T_ij_k(d_ij, ac_type, aircraft_params)
    Cf = C_F_ij_k(d_ij, ac_type, aircraft_params)

    C_total = Cx + Ct + Cf

    # HUB DISCOUNT (Appendix B — 30% lower)
    if origin == hub or dest == hub:
        C_total *= 0.7

    return C_total


def leasing_cost(ac_type, aircraft_params):
    """
    :param ac_type: Aircraft type
    :param aircraft_params: Aircraft parameters

    This function retrieves the leasing cost of an aircraft type
    """
    return aircraft_params[ac_type]["L"]


def compute_all_C_ijk(airports, dist_matrix, aircraft_params, hub):
    """
    :param airports: iterable of airport IDs (indices or codes)
    :param dist_matrix: structure with dist_matrix[i][j] = distance in km
    :param aircraft_params: dict as used in other functions
    :param hub: hub airport ID (same type as elements in 'airports')

    :return: dict with keys (i, j, ac_type) and values C_ij^k
    """
    costs = {}

    for i in airports:
        for j in airports:
            if i == j:
                continue  # skip self-legs

            d_ij = dist_matrix[i][j]

            for ac_type in aircraft_params.keys():
                costs[(i, j, ac_type)] = C_ij_k(
                    origin=i,
                    dest=j,
                    d_ij=d_ij,
                    ac_type=ac_type,
                    aircraft_params=aircraft_params,
                    hub=hub
                )

    return costs




# The code below is only runned when running the file directly
if __name__ == "__main__":
    print(f"This should be printed now")

    ac_data_path = r"C:\Users\jobru\Documents\TU Delft\MSc AE\Year 1\Courses Q2\APandO\Assignment files\AirlinePlanningAndOptimisation\Assignment1\Data\AircraftData.xlsx"

    ac_parameters = load_aircraft_params(ac_data_path)

    print(f"ac_parametes: {ac_parameters}")
    
    pass
