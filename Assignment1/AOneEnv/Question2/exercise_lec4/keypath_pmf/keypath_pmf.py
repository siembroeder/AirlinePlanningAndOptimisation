import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gurobipy import *

from Question1A.Read_input import read_excel_pandas



def main():
    # Load Exercise data from lecture 4
    ex_path = r"C:\Users\siemb\Documents\Year5\AirlinePlannningOptimisation\Assignments\Assignment1\Data\AE4423_PMF_Exercise_Input.xlsx"
    ex_sheets = ["Flights", "Itineraries", "Recapture"]
    flights, itins, recaps    = read_excel_pandas(ex_path, ex_sheets)

    # Standardize column names
    flights.rename(columns={"O":"Origin", "D":"Destination", "DTime":"DepartureTime", "RTime":"ArrivalTime", "Cap":"Capacity"}, inplace=True)
    recaps.rename(columns={"From":"OldItin", "To":"NewItin", "Rate":"RecapRate"}, inplace=True)

    flight_nums = flights.index
    capacity    = flights["Capacity"]
    itins       = itins.to_dict('index')

    revenue     = {p: itins[p]['Fare']   for p in itins}    # Revenue per itinerary
    demand      = {p: itins[p]['Demand'] for p in itins}    # Demand per itinerary


    # Incidence matrix
    delta = {p: {i: int(i in [itins[p]['Leg1'], itins[p]['Leg2']]) for i in flight_nums} for p in itins}

    # Recapture rates b^r_p
    b = {p: {r: 0.0 for r in itins} for p in itins}
    for idx, row in recaps.iterrows():
        old_itin = int(row['OldItin'])
        new_itin = int(row['NewItin'])
        b[old_itin][new_itin] = row['RecapRate']


    # Daily unconstrained demand per flight
    Q = {f: sum(delta[p][f]*demand[p] for p in itins) for f in flight_nums}


    # Initialize model
    m = Model('keypath')
    m.params.LogFile = 'Question2/exercise_lec4/keypath_pmf/Keypath.log'


    # Decision variables t^r_p >= 0, integer
    t = {}
    for p in itins:
        for r in itins:
            if r != p and b[p][r] > 0:   # only reallocated passengers
                t[p,r] = m.addVar(lb=0.0, vtype=GRB.INTEGER, name=f"t_{p}_{r}")


    m.setObjective(quicksum((revenue[p] - b[p][r]*revenue[r])*t[p,r] for (p,r) in t), GRB.MINIMIZE)

    # Capacity
    for i in flight_nums:
        lhs_removed     = quicksum(delta[p][i] * t[p,r] for (p,r) in t)
        lhs_recaptured  = quicksum(delta[p][i] * b[r][p] * t[r,p] for (r,p) in t if b[r][p] > 0)

        rhs = Q[i] - flights.loc[i,'Capacity']
        m.addConstr(lhs_removed - lhs_recaptured >= rhs, name=f"cap_{i}")


    # passengers <= demand
    for p in itins:
        lhs = quicksum(t[p,r] for r in itins if (p,r) in t)
        m.addConstr(lhs <= demand[p], name=f"demand_{p}")


    
    m.optimize()

    if m.status == GRB.OPTIMAL or m.status == GRB.TIME_LIMIT:
        m.write('Question2/exercise_lec4/keypath_pmf/Keypath_model.lp')
        print("\nOptimal objective value:", m.objVal)
        print("\nPassenger reallocation (t^r_p):")
        for (p,r) in sorted(t):
            if t[p,r].X > 0.001:
                print(f"{t[p,r].X:.2f} passengers originally on {p} reallocated to {r}")
    else:
        print("Model not solved to optimality, status:", m.status)
















if __name__ == '__main__':
    main()