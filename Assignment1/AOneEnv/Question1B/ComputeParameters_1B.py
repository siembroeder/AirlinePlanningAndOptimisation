import numpy as np 
import pandas as pd

# Functions imports 
from Question1A.DistancesLatLong import compute_dij
from Question1A.main import main

def demand_list(airport_data, q):
    """
    Compute the demand matrix based on airport data and demand dataframe computed in Question 1A.

    """
    airports = airport_data.columns  
    demand_list = np.zeros((len(airports), len(airports)), dtype=int)
    
    for i, porti in enumerate(airports):
        for j, portj in enumerate(airports):
            if i == j:
                demand_list[i][j] = 0
            else:
                demand_list[i][j] = int(q.loc[(q['i'] == porti) & (q['j'] == portj), 'Dij'].iloc[0])

    return demand_list

def load_airport_params(airport_data):
    """
    Compute airport parameters from airport data Excel file.

    """
    airports = airport_data.columns
    len_airports = len(airports)
    
    distance = np.zeros((len_airports, len_airports))       # Distance matrix between all airports
    r = np.zeros((len_airports, len_airports))              # Min runway length matrix for all routes - most critical runway length
    ls = np.zeros(len_airports)                             # Available weekly landing slots at each airport
    g = np.ones(len_airports)                               # Hub indicator (0 if hub, 1 otherwise)    
    
    for i, porti in enumerate(airports):

        if( airport_data[porti]['Hub'] == 'Yes' ):
            g[i] = 0
            ls[i] = 10000           # No limit (big M value set) on landing slots at hub
        else:
            ls[i] = airport_data[porti]['Available slots']

        for j, portj in enumerate(airports):
            if i == j:
                distance[i, j] = 0  # Distance from/to same airport is zero
            else:
                distance[i, j] = compute_dij(airport_data, porti, portj)

            r[i, j] = min(airport_data[porti]['Runway (m)'], airport_data[portj]['Runway (m)']) # Min runway length for route ij
    
    return distance, r, ls, g


def yields(distances):
    '''
    Compute the yield matrix based on the distance matrix.
    '''
    y = np.zeros(distances.shape)
    for i in range(distances.shape[0]):
        for j in range(distances.shape[0]):
            if i == j:
                y[i,j] = 0      # No yield for same airport
            else:
                y[i,j] = 5.9 * distances[i,j]**-0.76 + 0.043
    return y

def load_aircraft_params(aircraft_data):
    """
    Load aircraft parameters from aircraft data Excel file.
    """
    aircraft_types = aircraft_data.columns          # List of aircraft types

    sp = np.zeros(len(aircraft_types))    # Speed in km/h
    s = np.zeros(len(aircraft_types))     # Seating capacity
    TAT = np.zeros(len(aircraft_types))   # Average turn around time in hours
    R = np.zeros(len(aircraft_types))     # Range in km
    RW = np.zeros(len(aircraft_types))    # Required runway length in m

    C_L = np.zeros(len(aircraft_types))     # Weekly lease cost in Eur
    C_X = np.zeros(len(aircraft_types))   # Fixed operating cost in Eur
    c_T = np.zeros(len(aircraft_types))   # Time cost parameter in Eur/hr
    c_F = np.zeros(len(aircraft_types))   # Fuel cost parameter

    for k, aircraft in enumerate(aircraft_types):

        sp[k] = aircraft_data[aircraft]['Speed [km/h]']
        s[k]  = aircraft_data[aircraft]['Seats']
        TAT[k]= aircraft_data[aircraft]['Average TAT [mins]'] / 60      # Convert to hours
        R[k]  = aircraft_data[aircraft]['Maximum range [km]']
        RW[k] = aircraft_data[aircraft]['Runway required [m]']

        C_L[k]  = aircraft_data[aircraft]['Weekly lease cost [€]']
        C_X[k]= aircraft_data[aircraft]['Fixed operating cost C_X [€]']
        c_T[k]= aircraft_data[aircraft]['Time cost parameter C_T [€/hr]']
        c_F[k]= aircraft_data[aircraft]['Fuel cost parameter C_F']
    
    return aircraft_types, sp, s, TAT, R, RW, C_L, C_X, c_T, c_F


def operating_costs(airport_data, aircraft_data, distances, sp, FUEL, C_X, c_T, c_F):

    """
    Compute the operating cost matrix for all routes and aircraft types using Appendix B formulas.

    """

    operating_costs = np.zeros((len(airport_data.columns), len(airport_data.columns), len(aircraft_data.columns)))

    for k, aircraft in enumerate(aircraft_data.columns):
        for i, porti in enumerate(airport_data.columns):
            for j, portj in enumerate(airport_data.columns):

                operating_costs[i,j,k] = C_X[k] + c_T[k] * (distances[i,j] / sp[k]) + c_F[k] * FUEL / 1.5 * distances[i,j]
    
    operating_costs = operating_costs * 0.7   # Apply hub discount of 30% to all flights

    return operating_costs


# def C_T_ij_k(d_ij, ac_type, aircraft_params):
#     """  
#     :param d_ij: The distance in km between origin i and destination j
#     :param ac_type: Type of aircraft used in flight leg
#     :param aircraft_params: Aircraft parameters
    
#     This function calculates the time-based operating costs. 
#     These are defined in euros per flight hour and represent dime-dependent operating costs such as cabin- and flight crew.
#     They depend only on the distance of the flight leg and the aircraft type k."""

#     c_T = aircraft_params[ac_type]["c_T"]
#     V   = aircraft_params[ac_type]["V"]
    
#     return c_T * (d_ij / V)


# def C_F_ij_k(d_ij, ac_type, aircraft_params, fuel_price=FUEL_PRICE):
#     """
#     :param d_ij: The distance in km between origin i and destination j
#     :param ac_type: Aircraft type
#     :param aircraft_params: Aircraft parameters
#     :param fuel_price: Fuel price
#     """
#     c_F = aircraft_params[ac_type]["c_F"]
    
#     return c_F * (fuel_price ** 1.5) * d_ij


# def C_ij_k(origin, dest, d_ij, ac_type, aircraft_params, hub):
#     """
#     :param origin: Origin airport
#     :param dest: Destination airport
#     :param d_ij: Distance in km between origin and destination
#     :param ac_type: Aircraft type
#     :param aircraft_params: Aircraft parameters
#     :param hub: Hub airport
    
#     This function calculates the total operating costs for a flight leg between airports i and j operated by aircraft type k. 
#     This takes into account that flights departing or landing at the hub airport have 30% lower operating costs due to econonmics of scale. """

#     Cx = C_X_k(ac_type, aircraft_params)
#     Ct = C_T_ij_k(d_ij, ac_type, aircraft_params)
#     Cf = C_F_ij_k(d_ij, ac_type, aircraft_params)

#     C_total = Cx + Ct + Cf

#     # HUB DISCOUNT (Appendix B — 30% lower)
#     if origin == hub or dest == hub:
#         C_total *= 0.7

#     return C_total


# def leasing_cost(ac_type, aircraft_params):
#     """
#     :param ac_type: Aircraft type
#     :param aircraft_params: Aircraft parameters

#     This function retrieves the leasing cost of an aircraft type
#     """
#     return aircraft_params[ac_type]["L"]


# def compute_all_C_ijk(airports, dist_matrix, aircraft_params, hub):
#     """
#     :param airports: iterable of airport IDs (indices or codes)
#     :param dist_matrix: structure with dist_matrix[i][j] = distance in km
#     :param aircraft_params: dict as used in other functions
#     :param hub: hub airport ID (same type as elements in 'airports')

#     :return: dict with keys (i, j, ac_type) and values C_ij^k
#     """
#     costs = {}

#     for i in airports:
#         for j in airports:
#             if i == j:
#                 continue  # skip self-legs

#             d_ij = dist_matrix[i][j]

#             for ac_type in aircraft_params.keys():
#                 costs[(i, j, ac_type)] = C_ij_k(
#                     origin=i,
#                     dest=j,
#                     d_ij=d_ij,
#                     ac_type=ac_type,
#                     aircraft_params=aircraft_params,
#                     hub=hub
#                 )

#     return costs

