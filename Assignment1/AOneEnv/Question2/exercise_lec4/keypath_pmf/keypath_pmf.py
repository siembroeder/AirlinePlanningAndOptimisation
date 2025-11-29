import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gurobipy import *

from Question1A.Read_input import read_excel_pandas



def main():
    # Load PMF Data
    source_data = "Assignment" # Options: "Exercise", "Assignment"

    ex_sheets = ["Flights", "Itineraries", "Recapture"]

    if source_data == "Exercise":
        ex_path = r"..\..\Assignment1\Data\AE4423_PMF_Exercise_Input.xlsx"
        flights, itins, recaps    = read_excel_pandas(ex_path, ex_sheets)

        # Standardize column names
        flights.rename(columns={"O":"Origin", "D":"Destination", "DTime":"DepartureTime", "RTime":"ArrivalTime", "Cap":"Capacity"}, inplace=True)
        recaps.rename(columns={"From":"OldItin", "To":"NewItin", "Rate":"RecapRate"}, inplace=True)

        
        itins       = itins.to_dict('index')
        revenue     = {p: itins[p]['Fare']   for p in itins}    # Revenue per itinerary

        # Incidence matrix
        flight_nums = flights.index        
        delta = {p: {i: int(i in [itins[p]['Leg1'], itins[p]['Leg2']]) for i in flight_nums} for p in itins}
    
    elif source_data == "Assignment":
        ex_path = r"..\..\Assignment1\Data\Group_7_PMF.xlsx"
        flights, itins, recaps    = read_excel_pandas(ex_path, ex_sheets, indx = None)

        # Standardize column names
        recaps.rename(columns={"From Itinerary":"OldItin", "To Itinerary":"NewItin", "Recapture Rate":"RecapRate"}, inplace=True)
        flights.rename(columns={"O":"Origin", "D":"Destination", "DTime":"DepartureTime", "RTime":"ArrivalTime", "Cap":"Capacity"}, inplace=True)

        itins       = itins.to_dict('index')
        revenue     = {p: itins[p]['Price [EUR]']   for p in itins}    # Revenue per itinerary 
    
        # Incidence matrix
        flight_nums = flights.index
        delta = {p: {i: int(i in [itins[p]['Flight 1'], itins[p]['Flight 2']]) for i in flight_nums} for p in itins}

    capacity    = flights["Capacity"]
    demand      = {p: itins[p]['Demand'] for p in itins}    # Demand per itinerary
    
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
            if r != p:   # only reallocated passengers
                t[p,r] = m.addVar(lb=0.0, vtype=GRB.INTEGER, name=f"t_{p}_{r}")

    m.setObjective(quicksum((revenue[p] - b[p][r]*revenue[r])*t[p,r] for (p,r) in t), GRB.MINIMIZE)

    # Capacity
    for i in flight_nums:
        lhs_removed     = quicksum(delta[p][i] * t[p,r]           for (p,r) in t)
        lhs_recaptured  = quicksum(delta[p][i] * b[r][p] * t[r,p] for (r,p) in t)

        rhs = Q[i] - capacity[i]
        m.addConstr(lhs_removed - lhs_recaptured >= rhs, name=f"cap_{i}")

    # passengers <= demand
    for p in itins:
        lhs = quicksum(t[p,r] for (pp,r) in t if pp==p)
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



    # total_revenue = 0.0
    # stay_rev      = 0.0
    # recap_rev     = 0.0

    # for p in itins:
    #     # Revenue from passengers who stayed
    #     stay_p = demand[p] - sum(t[p,r].X for r in itins if (p,r) in t)
    #     stay_rev += revenue[p] * stay_p

    #     # Revenue from reallocated passengers
    #     for r in itins:
    #         if (p,r) in t:
    #             recap_rev += b[p][r] * revenue[r] * t[p,r].X # recapture rate corresponds to increase in revenue, recaptured pax don't bring full revenue

    # total_revenue = stay_rev + recap_rev

    
    # print("\nTotal stayrevenue across all flights: {:.2f}".format(stay_rev))
    # print("\nTotal recaprevenue across all flights: {:.2f}".format(recap_rev))
    # print("\nTotal revenue across all flights: {:.2f}".format(total_revenue))

    # print("\nRevenue breakdown by itinerary:")
    # for p in itins:
    #     stay_p = demand[p] - sum(t[p,r].X for r in itins if (p,r) in t)
    #     stay_rev = revenue[p] * stay_p
    #     realloc_rev = sum(b[p][r] * revenue[r] * t[p,r].X for r in itins if (p,r) in t)
        # print(f"Itinerary {p}: stayed {stay_p:.2f} pax, revenue = {stay_rev:.2f}; "
            # f"reallocated revenue = {realloc_rev:.2f}, total = {stay_rev + realloc_rev:.2f}")



    for (p, r), var in t.items():
        if var.X > 1e-6:
            net_loss = revenue[p] - b[p][r] * revenue[r]
            print(f"p={p}, r={r}, t={var.X:.1f}, net_loss_per_pax={net_loss:.2f}")









if __name__ == '__main__':
    main()