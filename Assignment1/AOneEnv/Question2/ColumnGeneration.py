import pandas as pd
import numpy as np
from gurobipy import *
from Question2.load_pmf_data import load_assignment_data, load_exercise_data
from Question2.calc_profit import calculate_total_profit_pathbased, get_first_five_flight_duals, get_first_five_itins, first_five_vars
import time

def solve_master_problem(itins, flight_idx, delta, capacity, Q, revenue, b, current_columns):
    """
    Solve the restricted master problem with current columns
    Returns: model, dual values (pi, sigma)
    """
    m = Model('master')
    m.Params.OutputFlag = 0  # silent
    
    # If no columns yet, return zero duals
    if len(current_columns) == 0:
        pi = {i: 0.0 for i in flight_idx}
        sigma = {p: 0.0 for p in itins}
        return None, pi, sigma
    
    # Decision variables for current columns only
    t = {}
    for (p, r) in current_columns:
        t[p, r] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"t_{p}_{r}")
    
    # Objective
    m.setObjective(quicksum((revenue[p] - b[p][r]*revenue[r]) * t[p,r] for (p,r) in t), GRB.MINIMIZE)
    
    # Capacity constraints
    cap_constrs = {}
    for i in flight_idx:
        lhs_removed     = quicksum(delta[p][i] * t[p,r]           for (p,r) in t)
        lhs_recaptured  = quicksum(delta[p][i] * b[r][p] * t[r,p] for (r,p) in t)
        rhs            = Q[i] - capacity[i]
        cap_constrs[i] = m.addConstr(lhs_removed - lhs_recaptured >= rhs, name=f"cap_{i}")
    
    # Demand constraints
    demand_constrs = {}
    for p in itins:
        lhs = quicksum(t[pp,r] for (pp,r) in t if pp == p)
        demand_constrs[p] = m.addConstr(lhs <= itins[p]['Demand'], name=f"demand_{p}")

    
    m.optimize()


    # m.computeIIS()
    # m.write("master_infeasible.ilp")
    
    # Extract dual values
    pi = {i: cap_constrs[i].Pi for i in flight_idx}
    sigma = {p: demand_constrs[p].Pi for p in itins}
    
    return m, pi, sigma


def compute_reduced_cost(p, r, itins, flight_idx, delta, revenue, b, pi, sigma):
    # Modified fare in itinerary p
    modified_fare_p = revenue[p] - sum(pi[i] for i in flight_idx if delta[p][i] > 0)
    
    # Modified fare in alternative itinerary r
    modified_fare_r = revenue[r] - sum(pi[j] for j in flight_idx if delta[r][j] > 0)
    
    # Reduced cost formula from slide
    c_pr = modified_fare_p - b[p][r] * modified_fare_r - sigma[p]
    
    return c_pr



def find_negative_columns(itins, flight_idx, delta, revenue, b, pi, sigma, threshold=0.0, max_columns=None):
    """ Find columns with negative reduced cost
    Parameters:
    - threshold: only return columns with RC below this (default 0.0)
    - max_columns: maximum number of columns to return (None = all) """

    negative_cols = []
    
    for p in itins:
        for r in itins:
            if r == p:
                continue
            
            rc = compute_reduced_cost(p, r, itins, flight_idx, delta, revenue, b, pi, sigma)

            if rc < threshold:
                negative_cols.append((p, r, rc))

    # Sort by reduced cost
    negative_cols.sort(key=lambda x: x[2])

    if max_columns is not None:
        return negative_cols[:max_columns]
    else:
        return negative_cols


   
def column_generation(flights, itins, recaps, flight_idx, capacity, demand, revenue, delta, Q, b, DUMMY, 
                      threshold=0.0, columns_per_iteration=None):
    # Initialize columns with spill only
    current_columns = {(p,DUMMY) for p in itins if p!=DUMMY}
    print(f"Starting with {len(current_columns)}, spill columns only")

    iteration = 0
    max_iterations = 1000
    
    while iteration < max_iterations:
        iteration += 1
        print(f"Iteration {iteration}")
        
        # Solve master problem
        master, pi, sigma = solve_master_problem(itins, flight_idx, delta, capacity, Q, revenue, b, current_columns)
        
        if master is not None:
            print(f"  Master objective: {master.ObjVal}")
        else:
            print(f"  Master objective: 0.00 (no columns yet)")
        
        negative_cols = find_negative_columns(itins, flight_idx, delta, revenue, b, pi, sigma, 
                                              threshold=threshold, 
                                              max_columns=columns_per_iteration)
        
        # Add new columns
        added_count = 0
        for (p, r, rc) in negative_cols:
            if (p, r) not in current_columns:
                print(f"Adding Column ({p}, {r}): RC = {rc}")
                current_columns.add((p, r))
                added_count += 1
        
        print(f"  Added {added_count} new columns. Total columns: {len(current_columns)}")

        if added_count == 0 or len(negative_cols) == 0:
            print("Column Generation Converged")
            # duals_info = get_first_five_duals(master, itins, pi, sigma)
            t = {(p,r): master.getVarByName(f"t_{p}_{r}") for (p,r) in current_columns}
            duals_info = get_first_five_itins(master, itins, pi, sigma, t, revenue, b, delta, flight_idx)
            flight_duals = get_first_five_flight_duals(pi, capacity, Q, flight_idx, delta, itins, t)
            break


       
    print(f"\nFinal number of columns: {len(current_columns)}")
    
    # Solve final master as integer problem
    m_final = Model('final')
    m_final.params.LogFile = 'Question2/log_files/CG.log'
    
    t = {}
    for (p, r) in current_columns:
        t[p, r] = m_final.addVar(lb=0.0, vtype=GRB.INTEGER, name=f"t_{p}_{r}")
    
    m_final.setObjective(quicksum((revenue[p] - b[p][r]*revenue[r]) * t[p,r] for (p,r) in t), GRB.MINIMIZE)
    
    for i in flight_idx:
        lhs_removed    = quicksum(delta[p][i] * t[p,r] for (p,r) in t)
        lhs_recaptured = quicksum(delta[r][i] * b[p][r] * t[p,r] for (p,r) in t)
        rhs            = Q[i] - capacity[i]
        m_final.addConstr(lhs_removed - lhs_recaptured >= rhs, name=f"cap_{i}")
    
    for p in itins:
        lhs = quicksum(t[pp,r] for (pp,r) in t if pp == p)
        m_final.addConstr(lhs <= itins[p]['Demand'], name=f"demand_{p}")
    
    m_final.optimize()

    # results = calculate_total_profit_pathbased(m_final, t, revenue, itins, demand, b=b, verbose=True)
    # print(f"\nFinal Total Profit: ${results['total_profit']:.2f}")

    for (p,r) in t:
        if r== 1 or p == 1:
            print(f't_{p}^{r} = {t[p,r].X}')
            print(itins[p])

    first_five_vars(m_final, t, revenue, itins, demand, b=b, verbose=False)

    if m_final.status == GRB.OPTIMAL:          
        m_final.write('Question2/log_files/CG.lp')
        print(f"\nFinal Integer Objective: {m_final.ObjVal}")
        # print("\nPassenger reallocations:")
        # for (p, r) in sorted(t.keys()):
        #     if t[p,r].X > 0.001:
        #         print(f"  {t[p,r].X:.2f} passengers from itinerary {p} to {r}")
        



    else:
        print(f"\nModel status: {m_final.status}")
        if m_final.status == GRB.INFEASIBLE:
            print("Model is infeasible")
            m_final.computeIIS()
            m_final.write('infeasible.ilp')
    
    return m_final, current_columns



def main():
    t1 = time.time()
    # flights, itins, recaps, flight_idx = load_exercise_data()
    # flights, itins, recaps, flight_idx = load_assignment_data(modified=True)
    flights, itins, recaps, flight_idx = load_assignment_data()

    thrshld    = -0.0001 # Only add meaningful negative columns -> numerical precision
    clmns_iter = 100000 # If large number, add all negative rc columns -> Suggested by teacher.
    
    capacity = dict(zip(flight_idx, flights['Capacity']))
    demand   = {p: itins[p]['Demand'] for p in itins}
    revenue  = {p: itins[p]['Fare'] for p in itins}
    delta    = {p: {f: int(f in [itins[p]['Leg1'], itins[p]['Leg2']]) for f in flight_idx} for p in itins}
    Q        = {i: sum(delta[p][i] * itins[p]['Demand'] for p in itins) for i in flight_idx}
    
    b = {p: {r: 0.0 for r in itins} for p in itins}
    for idx, row in recaps.iterrows():
        old_itin = int(row['OldItin'])
        new_itin = int(row['NewItin'])
        b[old_itin][new_itin] = row['RecapRate']
        


    print(f'Total capacity: {sum(capacity.values())}')
    print(f'Total demand: {sum(demand[p] for p in itins)}')
    print(f'Optimal spillage: {sum(demand[p] for p in itins) - sum(capacity.values())}')



    # Define dummy itineray for proper initialization
    DUMMY = 999
    itins[DUMMY]   = {'Demand': 1e10, 'Fare':0, 'Leg1':None, 'Leg2':None}
    revenue[DUMMY] = 0
    delta[DUMMY]   = {i:0 for i in flight_idx}
    b[DUMMY] = {r:0 for r in itins}
    for p in itins:
        b[p][DUMMY] = 0
        
    
    # Run column generation
    final_model, final_columns = column_generation(flights, itins, recaps, flight_idx, capacity, demand, revenue, delta, Q, b, DUMMY,
                                                   threshold = thrshld, columns_per_iteration=clmns_iter)
    
    t2 = time.time()

    print(f'CG took: {(t2-t1):.2f} seconds')


if __name__ == '__main__':
    main()