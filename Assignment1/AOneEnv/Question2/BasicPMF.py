import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gurobipy import *
import time

from Question2.calc_profit import calculate_total_profit_basic
from Question2.load_pmf_data import load_assignment_data, load_exercise_data


def main():
    # flights, itins, recaps, flight_idx = load_exercise_data() # optimal spillage 198
    t1 = time.time()
    flights, itins, recaps, flight_idx = load_assignment_data()

    # print(flights.head())
    # print(itins.head())
    # print(recaps.head())

    capacity = dict(zip(flight_idx, flights['Capacity']))

    revenue = {i: itins[i]['Fare'] for i in itins}      # Revenue per itinerary
    demand  = {i: itins[i]['Demand'] for i in itins}    # Demand per itinerary
    delta   = {i: {f: int(f in [itins[i]['Leg1'], itins[i]['Leg2']]) for f in flight_idx} for i in itins} # Incidence matrix =1 if itin i uses flight f, else 0

    
    print(f'Total capacity: {sum(capacity.values())}')
    total_demand = sum(demand[p] for p in itins)
    print(f'Total demand: {total_demand}')
    print(f'Optimal spillage: {sum(demand[p] for p in itins) - sum(capacity.values())}')

    recap_rates = {p: {r:(1.0 if p==r else 0.0) for r in itins} for p in itins} # Build recapture matrix, zero except for given data
    for idx, row in recaps.iterrows():
        old_itin = int(row['OldItin'])
        new_itin = int(row['NewItin'])
        recap_rates[old_itin][new_itin] = row['RecapRate']

    # Initialize model
    m = Model('basicPMF')
    m.params.LogFile = 'Question2/log_files/basicPMF.log'

    x = {}
    num_x = 0
    num_p = 0
    for p in itins:
        for r in itins:
            if recap_rates[p][r] > 0.0:
                x[p,r] = m.addVar(lb=0.0, ub= GRB.INFINITY, vtype=GRB.INTEGER, name=f"x_{p}_{r}") 
                num_x +=1

    num_pr = 0

    for p in itins:
        num_p +=1    
        for idx,row in recaps.iterrows():
            if p == row['OldItin']:
                num_pr += 1


    print(f'\n\n\n Total number of x variables: {num_x}')
    print(f'|P| + sum_pinP |P_p|  = {num_p} + {num_pr} = {num_p + num_pr}')

                
    m.setObjective(quicksum(revenue[r]*x[p,r] for (p,r) in x), GRB.MAXIMIZE)

    # Capacity constraint
    for f in flight_idx:
        m.addConstr(quicksum(delta[r][f] * x[p,r] for (p,r) in x) <= capacity[f], name = f"cap_{f}")

    # Demand constraint
    for p in itins:
        lhs = quicksum(x[p,r] / recap_rates[p][r] for r in itins if (p,r) in x) # and recap_rates[p][r]<1.0
        rhs = demand[p]
        m.addConstr(lhs <= rhs, name = f'recap_{p}')

    
    m.optimize()
    

    print(f"\nObjective value: {m.ObjVal}")
    
    j = 0
    recap_pax = 0
    stay_pax  = 0
    for (p,r) in x:
        if x[p,r].X > 0.001 and p != r:
            j +=1
            print(f"x[{p},{r}] = {x[p,r].X:.2f}")
            recap_pax += x[p,r].X
        elif x[p,r].X > 0.001 and p == r:
            j +=1
            print(f"x[{p},{r}] = {x[p,r].X:.2f}")
            stay_pax += x[p,r].X
    print(f"Non-zero variables: {j}")
    print(f'Recaptured Pax: {recap_pax}')
    print(f'Stay Pax: {stay_pax}')    


    results = calculate_total_profit_basic(m,x,revenue,itins,demand,b=recap_rates,verbose=False)
    print(f'Final Total Profit: {results}')

    total_served = 0
    for p in itins:
        # Calculate how many from demand p were served
        served = sum(x[p,r].X for r in itins if (p,r) in x) # /recap_rates[p][r]

        total_served += served
    print(f"\nTotal served passengers: {total_served:.2f}")
    print(f'Total spilled passengers: {total_demand - total_served}')

    t2 = time.time()
    print(f'Script took {t2-t1} seconds to run')
    

    if m.status == GRB.OPTIMAL or m.status == GRB.TIME_LIMIT:
        m.write('Question2/log_files/basicPMF.lp')
    else:
        print("Model not solved to optimality, status:", m.status)









if __name__ == "__main__":
    main()