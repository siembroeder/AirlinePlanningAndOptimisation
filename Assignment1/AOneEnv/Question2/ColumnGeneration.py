import pandas as pd
import numpy as np
from gurobipy import *
from Question1A.Read_input import read_excel_pandas
from Question2.load_pmf_data import load_assignment_data, load_exercise_data



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

    if m.Status == GRB.INF_OR_UNBD:
        m.Params.DualReductions = 0
        m.optimize()
        print("Refined status:", m.Status)

    m.computeIIS()
    m.write("master_infeasible.ilp")
    
    # Extract dual values
    pi = {i: cap_constrs[i].Pi for i in flight_idx}
    sigma = {p: demand_constrs[p].Pi for p in itins}
    
    return m, pi, sigma


def compute_reduced_cost(p, r, itins, flight_idx, delta, revenue, b, pi, sigma):
    """
    Compute reduced cost for column (p,r) according to slide formula
    """
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
    ncols_rc_zero = 0
    rc_negative = False
    cols_rc_zero = []
    for p in itins:
        for r in itins:
            if r == p:
                continue
            
            rc = compute_reduced_cost(p, r, itins, flight_idx, delta, revenue, b, pi, sigma)

            if rc == 0.0:
                ncols_rc_zero += 1


            if rc < threshold:
                negative_cols.append((p, r, rc))
                rc_negative = True # Assume threshold = 0


            if rc_negative == False and rc == 0.0:
                cols_rc_zero.append((p,r,rc))

    print(f"Columns with rc=0: {ncols_rc_zero}")
    
    # Sort by reduced cost (most negative first)
    negative_cols.sort(key=lambda x: x[2])
    

    if max_columns is not None and rc_negative == True:
        return negative_cols[:max_columns]
    elif max_columns is not None and rc_negative == False: # In this case add columns with rc=0
        return cols_rc_zero[:max_columns]
    else:
        return negative_cols


   
def column_generation(flights, itins, recaps, flight_idx, capacity, demand, revenue, delta, Q, b, 
                      threshold=0.0, columns_per_iteration=None):
    # Initialize columns
    current_columns = set()
    # for p in itins:
    #     for r in itins:
    #         if r != p and b[p][r] > 0:  # Only if recapture rate exists
    #             current_columns.add((p, r))
    #             break  # Just add one per itinerary to start
    # print(f"Starting with {len(current_columns)} initial columns\n")

    # New: ensure for each flight with RHS>0 we add at least one (p,r) that uses that flight
    rhs_by_flight = {i: Q[i] - capacity[i] for i in flight_idx}
    for i in flight_idx:
        if rhs_by_flight[i] > 0:
            # find an itinerary p that uses flight i
            for p in itins:
                if delta[p][i] > 0:
                    # find an r != p with a positive recapture rate
                    found = False
                    for r in itins:
                        if r != p and b[p][r] > 0:
                            current_columns.add((p, r))
                            found = True
                            break
                    # if no positive recapture exists, you might still add (p, some r) or handle separately
                    if found:
                        break
    
    
    # as a fallback add one column per itinerary as before (optional)
    for p in itins:
        if not any(pp == p for (pp,rr) in current_columns):
            for r in itins:
                if r != p and b[p][r] > 0:
                    current_columns.add((p, r))
                    break

    print(f"CurrentColumns: {current_columns}")

    iteration = 0
    max_iterations = 1000
    
    while iteration < max_iterations:
        iteration += 1
        print(f"Iteration {iteration}")
        
        # Solve master problem
        master, pi, sigma = solve_master_problem(itins, flight_idx, delta, capacity, Q, revenue, b, current_columns)
        
        if master is not None:
            print(f"  Master objective: {master.ObjVal:.2f}")
        else:
            print(f"  Master objective: 0.00 (no columns yet)")
        
        negative_cols = find_negative_columns(itins, flight_idx, delta, revenue, b, pi, sigma, 
            threshold=threshold, 
            max_columns=columns_per_iteration)
        
        # Add new columns
        added_count = 0
        for (p, r, rc) in negative_cols:
            if (p, r) not in current_columns:
                print(f"Adding Column ({p}, {r}): RC = {rc:.4f}")
                current_columns.add((p, r))
                added_count += 1

        if added_count == 0 or len(negative_cols) == 0:
            print("Column Generation Converged")
            break

        print(f"  Added {added_count} new columns. Total columns: {len(current_columns)}")
       
    print(f"\nFinal number of columns: {len(current_columns)}")
    
    # Solve final master as integer problem
    m_final = Model('final')
    
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
    
    if m_final.status == GRB.OPTIMAL:
        print(f"\nFinal Integer Objective: {m_final.ObjVal:.2f}")
        print("\nPassenger reallocations:")
        for (p, r) in sorted(t.keys()):
            if t[p,r].X > 0.001:
                print(f"  {t[p,r].X:.2f} passengers from itinerary {p} to {r}")
    else:
        print(f"\nModel status: {m_final.status}")
        if m_final.status == GRB.INFEASIBLE:
            print("Model is infeasible")
            m_final.computeIIS()
            m_final.write('infeasible.ilp')
    
    return m_final, current_columns



def main():
    # flights, itins, recaps, flight_idx = load_exercise_data()
    flights, itins, recaps, flight_idx = load_assignment_data()
    # print(flights,itins,recaps, flight_idx)

    thrshld    = 0.0
    clmns_iter = 10
    
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
        
    
    # Run column generation
    final_model, final_columns = column_generation(flights, itins, recaps, flight_idx, capacity, demand, revenue, delta, Q, b,
                                                   threshold = thrshld, columns_per_iteration=clmns_iter)

if __name__ == '__main__':
    main()