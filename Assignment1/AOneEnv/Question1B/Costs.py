

# Check whether running file 
print(f"Costs file is running!")

# Imports - General
import math


FUEL_PRICE = 1.42


def C_X_k(ac_type, aircraft_params):
    """
    Docstring for C_X_k
    
    :param ac_type: Aircraft type
    :param aircraft_params: Aircraft parameters
    
    This function calculates the fixed operating costs per flight leg and represent costs such as landing rights, 
    parking fees and fixed fuel cost. They depend only on the aircraft type k"""

    return aircraft_params[ac_type]["C_X"]


def C_T_ij_k(d_ij, ac_type, aircraft_params):
    """
    Docstring for C_T_ij_k
    
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
    Docstring for C_F_ij_k
    
    :param d_ij: The distance in km between origin i and destination j
    :param ac_type: Aircraft type
    :param aircraft_params: Aircraft parameters
    :param fuel_price: Fuel price
    """

    c_F = aircraft_params[ac_type]["c_F"]
    
    return c_F * (fuel_price ** 1.5) * d_ij

def C_ij_k(origin, dest, d_ij, ac_type, aircraft_params, hub):
    """
    Docstring for C_ij_k
    
    :param origin: Origin airport
    :param dest: Destination airport
    :param d_ij: Distance in km between origin and destination
    :param ac_type: Aircraft type
    :param aircraft_params: Aircraft parameters
    :param hub: Hub airport
    
    This function calculates the total operating costs for a flight leg between airports i and j operated by aircraft type k. 
    This takes into account that flights departing or landing at the hub airport have 30% lower operating costs due to econonmics of scale.
    """

    Cx = C_X_k(ac_type, aircraft_params)
    Ct = C_T_ij_k(d_ij, ac_type, aircraft_params)
    Cf = C_F_ij_k(d_ij, ac_type, aircraft_params)

    C_total = Cx + Ct + Cf

    # HUB DISCOUNT (Appendix B — 30% lower)
    if origin == hub or dest == hub:
        C_total *= 0.7

    return C_total
