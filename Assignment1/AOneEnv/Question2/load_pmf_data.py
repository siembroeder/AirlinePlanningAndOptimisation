
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gurobipy import *

from Question1A.Read_input import read_excel_pandas



def load_assignment_data():
    ex_sheets   = ["Flights", "Itineraries", "Recapture"]
    ex_path = r"..\..\Assignment1\Data\Group_7_PMF.xlsx"
    flights, itins, recaps    = read_excel_pandas(ex_path, ex_sheets, indx = None)

    # Standardize column names
    flights.rename(columns={"O":"Origin", "D":"Destination", "DTime":"DepartureTime", "RTime":"ArrivalTime", "Cap":"Capacity"}, inplace=True)
    itins.rename(columns={'Price [EUR]':'Fare', 'Flight 1':'Leg1', 'Flight 2':'Leg2'}, inplace=True)
    recaps.rename(columns={"From Itinerary":"OldItin", "To Itinerary":"NewItin", "Recapture Rate":"RecapRate"}, inplace=True)

    # Shift indices by 1, in input data (only in assignment) they start at 0
    itins['Itinerary'] = itins['Itinerary'] + 1
    recaps['OldItin'] = recaps['OldItin'] + 1
    recaps['NewItin'] = recaps['NewItin'] + 1

    itins      = itins.set_index('Itinerary').to_dict('index')
    flight_idx = flights['Flight No.'].tolist()     # Can't use flights as indices as their labels include Letters

    return flights, itins, recaps, flight_idx

def load_exercise_data():
    ex_sheets   = ["Flights", "Itineraries", "Recapture"]
    ex_path = r"..\..\Assignment1\Data\AE4423_PMF_Exercise_Input.xlsx" # Execute from AOneEnv
    flights, itins, recaps    = read_excel_pandas(ex_path, ex_sheets)

    # Standardize column names
    flights.rename(columns={"O":"Origin", "D":"Destination", "DTime":"DepartureTime", "RTime":"ArrivalTime", "Cap":"Capacity"}, inplace=True)
    recaps.rename(columns={"From":"OldItin", "To":"NewItin", "Rate":"RecapRate"}, inplace=True)

    itins      = itins.to_dict('index')
    flight_idx = flights.index 

    return flights, itins, recaps, flight_idx

