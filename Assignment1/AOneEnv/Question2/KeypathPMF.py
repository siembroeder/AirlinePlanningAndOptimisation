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

    

    """ 
    # Perform postprocessing on optimization results.

    tolerance = 1e-6  # floating-point tolerance
    violated_constraints = []
    print("Checking all constraints for violations...\n")

    for constr in m.getConstrs():
        sense = constr.Sense       # '<=' = -1, '=' = 0, '>=' = 1
        slack = constr.getAttr("Slack")
        violated = False
    if sense == '<':       # <=
        if slack < -tolerance:
            violated = True
    elif sense == '>':     # >=
        if slack < -tolerance:
            violated = True
    elif sense == '=':     # ==
        if abs(slack) > tolerance:
            violated = True

    if violated:
        violated_constraints.append((constr.ConstrName, slack))
        print(f"Constraint VIOLATED: {constr.ConstrName}, Slack = {slack}")

    if not violated_constraints:
        print("All constraints satisfied within tolerance.")
    else:
        print(f"\nTotal violated constraints: {len(violated_constraints)}")

    # vehicles = list(capacity.keys())  # all flight IDs you have capacities for
    # for v in vehicles:  # or planes, trains, etc.
    #     total_on_v = sum(var.X for var in m.getVars() if f"_{v}" in var.VarName)
    #     if total_on_v > capacity[v] + 1e-6:  # small tolerance
    #         print(f"Capacity exceeded on {v}: {total_on_v} / {capacity[v]}")

    frac_vars = [v for v in m.getVars() if abs(v.X - round(v.X)) > 1e-6]
    print(f"Fractional allocations: {len(frac_vars)}")

    # obj_calc = sum(var.Obj * var.X for var in m.getVars())
    # print(f"Objective check: calculated = {obj_calc}, reported = {m.ObjVal}")


    def inspect_constraint(m, constr_name, show_terms=True, tol_coeff=1e-12):
        # find constraint by name
        constr = m.getConstrByName(constr_name)
        if constr is None:
            print(f"Constraint '{constr_name}' not found.")
            return

        # Extract RHS, sense and slack from Gurobi
        rhs = constr.getAttr("RHS")
        sense = constr.Sense   # '<' = <=, '>' = >=, '=' = ==
        slack = constr.getAttr("Slack")  # defined by Gurobi (see note below)

        # Compute LHS by summing coeff * var.X for all variables that appear in this constraint
        lhs = 0.0
        contributions = []   # list of (varname, coeff, var.X, coeff*var.X)
        for var in m.getVars():
            coeff = m.getCoeff(constr, var)  # coefficient of var in this constraint (0 if not present)
            if abs(coeff) > tol_coeff:
                val = var.X
                contrib = coeff * val
                lhs += contrib
                contributions.append((var.VarName, coeff, val, contrib))

        # Another (sanity) way to compute lhs from Gurobi attributes:
        # For sense '<' (<=): Slack = RHS - LHS  => LHS = RHS - Slack
        # For sense '>' (>=): Slack = LHS - RHS  => LHS = RHS + Slack
        if sense == '<':
            lhs_from_slack = rhs - slack
        elif sense == '>':
            lhs_from_slack = rhs + slack
        else:  # '='
            # Gurobi Slack for '=' is typically LHS - RHS, but compute consistent fallback:
            lhs_from_slack = rhs + slack  # this will match above when slack = LHS - RHS

        # Print report
        print(f"Constraint: {constr_name}")
        print(f"  Sense: '{sense}', RHS = {rhs}, Slack = {slack}")
        print(f"  Computed LHS (sum coeff*var.X) = {lhs:.6f}")
        print(f"  LHS computed from RHS/slack = {lhs_from_slack:.6f}")
        print(f"  Difference between methods = {lhs - lhs_from_slack:.6e}")

        # Show top contributing terms if requested
        if show_terms:
            # sort by absolute contribution
            contributions.sort(key=lambda x: abs(x[3]), reverse=True)
            print("\nContributions (var, coeff, value, coeff*value):")
            for name, coeff, val, contrib in contributions:  # show all terms
                if val > 0:
                    print(f"  {name:>25s}  {coeff:8.4f}  {val:10.2f}  {contrib:12.2f}")
            if len(contributions) > 200:
                print(f"({len(contributions)} terms in total)")

    # # Example: inspect cap_101
    inspect_constraint(m, "cap_NA3794")
    inspect_constraint(m, "cap_NA1013")
    inspect_constraint(m, "demand_1")
    # inspect_constraint(m, "demand_2")
    # inspect_constraint(m, "demand_3")
    # inspect_constraint(m, "demand_379")
    # inspect_constraint(m, "demand_380")
    inspect_constraint(m, "demand_381")

 """
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



    # for (p, r), var in t.items():
    #     if var.X > 1e-6:
    #         net_loss = revenue[p] - b[p][r] * revenue[r]
    #         print(f"p={p}, r={r}, t={var.X:.1f}, net_loss_per_pax={net_loss:.2f}")









if __name__ == '__main__':
    main()