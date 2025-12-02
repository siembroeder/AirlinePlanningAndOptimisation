import pandas as pd
import numpy as np
from gurobipy import *
from Question1A.Read_input import read_excel_pandas

def solve_master_problem(itins, flight_idx, delta, capacity, Q, revenue, b, current_columns):
    """
    Solve the restricted master problem with current columns
    Returns: model, dual values (pi, sigma)
    """
    m = Model('master')
    m.Params.OutputFlag = 0  # silent
    
    # Decision variables for current columns only
    t = {}
    for (p, r) in current_columns:
        t[p, r] = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"t_{p}_{r}")
    
    # Objective
    m.setObjective(
        quicksum((revenue[p] - b[p][r]*revenue[r]) * t[p,r] for (p,r) in t),
        GRB.MINIMIZE
    )
    
    # Capacity constraints
    cap_constrs = {}
    for i in flight_idx:
        lhs_removed = quicksum(delta[p][i] * t[p,r] for (p,r) in t)
        lhs_recaptured = quicksum(delta[r][i] * b[p][r] * t[p,r] for (p,r) in t)
        rhs = Q[i] - capacity[i]
        cap_constrs[i] = m.addConstr(lhs_removed - lhs_recaptured >= rhs, name=f"cap_{i}")
    
    # Demand constraints
    demand_constrs = {}
    for p in itins:
        lhs = quicksum(t[pp,r] for (pp,r) in t if pp == p)
        demand_constrs[p] = m.addConstr(lhs <= itins[p]['Demand'], name=f"demand_{p}")
    
    m.optimize()
    
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

def find_most_negative_column(itins, flight_idx, delta, revenue, b, pi, sigma):
    """
    Find the column with most negative reduced cost
    Returns: (p, r, reduced_cost) or None if all non-negative
    """
    best_col = None
    best_rc = 0.0
    
    for p in itins:
        for r in itins:
            if r == p:
                continue
            
            rc = compute_reduced_cost(p, r, itins, flight_idx, delta, revenue, b, pi, sigma)
            
            if rc < best_rc:
                best_rc = rc
                best_col = (p, r)
    
    return best_col, best_rc

def column_generation(flights, itins, recaps, flight_idx, capacity, demand, revenue, delta, Q, b):
    """
    Main column generation loop
    """
    print("\n=== Starting Column Generation ===\n")
    
    # Initialize with empty set of columns
    current_columns = set()
    
    iteration = 0
    max_iterations = 1000
    
    while iteration < max_iterations:
        iteration += 1
        print(f"Iteration {iteration}")
        
        # Solve master problem
        master, pi, sigma = solve_master_problem(
            itins, flight_idx, delta, capacity, Q, revenue, b, current_columns
        )
        
        if len(current_columns) > 0:
            print(f"  Master objective: {master.ObjVal:.2f}")
        else:
            print(f"  Master objective: 0.00 (no columns yet)")
        
        # Pricing: find column with most negative reduced cost
        new_col, reduced_cost = find_most_negative_column(
            itins, flight_idx, delta, revenue, b, pi, sigma
        )
        
        if new_col is None:
            print(f"  No column with negative reduced cost found")
            print("\n=== Column Generation Converged ===\n")
            break
        
        print(f"  Best reduced cost: {reduced_cost:.4f} for column {new_col}")
        
        # Add column to master
        current_columns.add(new_col)
        print(f"  Added column {new_col}. Total columns: {len(current_columns)}")
    
    print(f"\nFinal number of columns: {len(current_columns)}")
    
    # Solve final master as integer program
    print("\n=== Solving Final Integer Program ===\n")
    m_final = Model('final')
    
    t = {}
    for (p, r) in current_columns:
        t[p, r] = m_final.addVar(lb=0.0, vtype=GRB.INTEGER, name=f"t_{p}_{r}")
    
    m_final.setObjective(
        quicksum((revenue[p] - b[p][r]*revenue[r]) * t[p,r] for (p,r) in t),
        GRB.MINIMIZE
    )
    
    for i in flight_idx:
        lhs_removed = quicksum(delta[p][i] * t[p,r] for (p,r) in t)
        lhs_recaptured = quicksum(delta[r][i] * b[p][r] * t[p,r] for (p,r) in t)
        rhs = Q[i] - capacity[i]
        m_final.addConstr(lhs_removed - lhs_recaptured >= rhs, name=f"cap_{i}")
    
    for p in itins:
        lhs = quicksum(t[pp,r] for (pp,r) in t if pp == p)
        m_final.addConstr(lhs <= itins[p]['Demand'], name=f"demand_{p}")
    
    m_final.optimize()
    
    print(f"\nFinal Integer Objective: {m_final.ObjVal:.2f}")
    print("\nPassenger reallocations:")
    for (p, r) in sorted(t.keys()):
        if t[p,r].X > 0.001:
            print(f"  {t[p,r].X:.2f} passengers from itinerary {p} to {r}")
    
    return m_final, current_columns

def main():
    # [Keep your existing data loading code exactly as is]
    source_data = "Assignment"
    ex_sheets = ["Flights", "Itineraries", "Recapture"]
    
    if source_data == "Assignment":
        ex_path = r"..\..\Assignment1\Data\Group_7_PMF.xlsx"
        flights, itins, recaps = read_excel_pandas(ex_path, ex_sheets, indx=None)
        
        flights.rename(columns={"O":"Origin", "D":"Destination", "DTime":"DepartureTime", 
                                "RTime":"ArrivalTime", "Cap":"Capacity"}, inplace=True)
        itins.rename(columns={'Price [EUR]':'Fare', 'Flight 1':'Leg1', 'Flight 2':'Leg2'}, inplace=True)
        recaps.rename(columns={"From Itinerary":"OldItin", "To Itinerary":"NewItin", 
                               "Recapture Rate":"RecapRate"}, inplace=True)
        
        itins['Itinerary'] = itins['Itinerary'] + 1
        recaps['OldItin'] = recaps['OldItin'] + 1
        recaps['NewItin'] = recaps['NewItin'] + 1
        
        itins = itins.set_index('Itinerary').to_dict('index')
        flight_idx = flights['Flight No.'].tolist()
    
    capacity = dict(zip(flight_idx, flights['Capacity']))
    demand = {p: itins[p]['Demand'] for p in itins}
    revenue = {p: itins[p]['Fare'] for p in itins}
    delta = {p: {f: int(f in [itins[p]['Leg1'], itins[p]['Leg2']]) for f in flight_idx} for p in itins}
    Q = {i: sum(delta[p][i] * itins[p]['Demand'] for p in itins) for i in flight_idx}
    
    b = {p: {r: 0.0 for r in itins} for p in itins}
    for idx, row in recaps.iterrows():
        old_itin = int(row['OldItin'])
        new_itin = int(row['NewItin'])
        b[old_itin][new_itin] = row['RecapRate']
    
    # Run column generation
    final_model, final_columns = column_generation(
        flights, itins, recaps, flight_idx, capacity, demand, revenue, delta, Q, b
    )

if __name__ == '__main__':
    main()