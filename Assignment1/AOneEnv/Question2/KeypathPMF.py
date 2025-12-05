import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gurobipy import *

from Question1A.Read_input import read_excel_pandas
from Question2.load_pmf_data import load_assignment_data, load_exercise_data
from Question2.calc_profit import calculate_profit_difference, calculate_total_profit


def main():
    # Load PMF Data
    # flights, itins, recaps, flight_idx = load_exercise_data()
    flights, itins, recaps, flight_idx = load_assignment_data()


    capacity = dict(zip(flight_idx, flights['Capacity']))
    demand   = {p: itins[p]['Demand'] for p in itins}    # Demand per itinerary
    revenue  = {p: itins[p]['Fare']   for p in itins}    # Revenue per itinerary
    delta    = {p: {f: int(f in [itins[p]['Leg1'], itins[p]['Leg2']]) for f in flight_idx} for p in itins} # Incidence matrix
    Q        = {i: sum(delta[p][i] * itins[p]['Demand'] for p in itins) for i in flight_idx} # Daily unconstrained demand per flight, ie all passengers that'd like to travel on flight i if no constraints
    
    # Recapture rates b^r_p, 0 unless in recaps df
    b = {p: {r: 0.0 for r in itins} for p in itins}
    for idx, row in recaps.iterrows():
        old_itin = int(row['OldItin'])
        new_itin = int(row['NewItin'])
        b[old_itin][new_itin] = row['RecapRate']

    


    # Initialize model
    m = Model('keypath')
    m.params.LogFile = f'Question2/log_files/Keypath.log'

    # Decision variables t^r_p >= 0, integer
    print("Constructing Decision Variables")
    t = {}
    for p in itins:
        for r in itins:
            if r != p:   # only reallocated passengers
                t[r,p] = m.addVar(lb=0.0, vtype=GRB.INTEGER, name=f"t_{p}_{r}")

    m.setObjective(quicksum((revenue[p] - b[p][r]*revenue[r])*t[p,r] for (p,r) in t), GRB.MINIMIZE)

    print("Constructing Constraints")
    # Capacity
    for i in flight_idx:
        lhs_removed     = quicksum(delta[p][i] * t[p,r]           for (p,r) in t)
        lhs_recaptured  = quicksum(delta[p][i] * b[r][p] * t[r,p] for (r,p) in t)

        rhs = Q[i] - capacity[i]
        m.addConstr(lhs_removed - lhs_recaptured >= rhs, name=f"cap_{i}")
 
    # passengers <= demand
    for p in itins:
        lhs = quicksum(t[p,r] for (pp,r) in t if pp==p)
        m.addConstr(lhs <= demand[p], name=f"demand_{p}")




    # Execute optimization
    m.Params.TimeLimit = 100*60
    m.optimize()

    results = calculate_total_profit(m, t, revenue, itins, demand, 
                                        b=b, verbose=True)
    print(f"\nFinal Total Profit: ${results['total_profit']:.2f}")

    if m.status == GRB.OPTIMAL or m.status == GRB.TIME_LIMIT:
        m.write('Question2/log_files/KeypathModel.lp')
        print("\nOptimal objective value:", m.objVal)
        # print("\nPassenger reallocation (t^r_p):")
        # for (p,r) in sorted(t):
        #     if t[p,r].X > 0.001:
        #         print(f"{t[p,r].X:.2f} passengers originally on {p} reallocated to {r}")

        
        

    else:
        print("Model not solved to optimality, status:", m.status)

    








if __name__ == '__main__':
    main()