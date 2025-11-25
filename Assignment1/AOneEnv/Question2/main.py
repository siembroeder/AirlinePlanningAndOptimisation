import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gurobipy import *

from Question1A.Read_input import read_excel_pandas


def main():

    ex_path = r"C:\Users\siemb\Documents\Year5\AirlinePlannningOptimisation\Assignments\Assignment1\Data\AE4423_PMF_Exercise_Input.xlsx"
    ex_sheets = ["Flights", "Itineraries", "Recapture"]
    flights, itins, recaps    = read_excel_pandas(ex_path, ex_sheets)

    print(flights.head())
    print(itins.head())
    print(recaps.head())

    


    return ...








if __name__ == "__main__":
    main()