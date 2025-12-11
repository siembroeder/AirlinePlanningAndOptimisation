import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from gurobipy import *
import time

from Question1A.Read_input import read_excel_pandas
from Question2.load_pmf_data import load_assignment_data, load_exercise_data
from Question2.calc_profit import calculate_total_profit_pathbased, dv_first_five, get_first_five_itins


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
    b = {p: {r: (1.0 if p==r else 0.0) for r in itins} for p in itins}
    for idx, row in recaps.iterrows():
        old_itin = int(row['OldItin'])
        new_itin = int(row['NewItin'])
        b[old_itin][new_itin] = row['RecapRate']



    print(f'Total capacity: {sum(capacity.values())}')
    print(f'Total demand: {sum(demand[p] for p in itins)}')
    print(f'Optimal spillage: {sum(demand[p] for p in itins) - sum(capacity.values())}')

    # Initialize model
    t1 = time.time()
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

    t2 = time.time()
    print(f'Constructing model takes {t2-t1} seconds')
    iter_log = []

    def mip_callback(model, where):
        if where == GRB.Callback.MIP:
            iters = model.cbGet(GRB.Callback.MIP_ITRCNT)
            incumbent = model.cbGet(GRB.Callback.MIP_OBJBST)
            bound = model.cbGet(GRB.Callback.MIP_OBJBND)
            runtime = model.cbGet(GRB.Callback.RUNTIME)
            if incumbent == 0:
                gap = math.inf
            else:
                gap = abs(incumbent - bound) / abs(incumbent)

            iter_log.append((iters, incumbent, bound, gap, runtime))


    # Execute optimization
    m.Params.TimeLimit = 10*60
    m.optimize(mip_callback)


    plot=True
    if plot == True:

        df = pd.DataFrame(iter_log, columns=["iterations", "incumbent", "bound", "gap", "time"])

        df_plot = df.copy()

        # Drop rows with invalid early values
        df_plot = df_plot.replace([np.inf, -np.inf], np.nan).dropna()

        # Convert gap to percentage
        df_plot["gap_pct"] = 100 * df_plot["gap"]

        N = 1
        df_plot = df_plot[df_plot['iterations'] >= N]

        fig, ax1 = plt.subplots(figsize=(8, 5))
        incumbent_line, = ax1.plot(
            df_plot["iterations"],
            df_plot["incumbent"],
            label="Incumbent",
            color= 'green')

        bound_line, = ax1.plot(
            df_plot["iterations"],
            df_plot["bound"],
            linestyle=":",
            color="black",
            label="Bound")

        ax1.set_xlabel("Iterations")
        ax1.set_ylabel("Objective Value")
        # ax1.set_yscale('log')
        ax1.set_xscale('log')

        incumbent_color = incumbent_line.get_color()
        ax1.tick_params(axis="y", colors=incumbent_color)
        ax1.yaxis.label.set_color(incumbent_color)

        ax2 = ax1.twinx()

        gap_line, = ax2.plot(
            df_plot["iterations"],
            df_plot["gap_pct"],
            linestyle="--",
            label="Optimality gap",
            color= 'blue')

        ax2.set_ylabel("Percentage")
        # ax2.set_yscale('log')
        
        ax2.tick_params(axis="y", colors='blue')
        ax2.yaxis.label.set_color('blue')

        # ---- LEGEND (COMBINED) ----
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

        # ---- GRID + TITLE ----
        ax1.grid(True)
        # plt.title("MILP Solver Progress")

        plt.tight_layout()
        plt.xlabel('')
        plt.show()

    results = calculate_total_profit_pathbased(m, t, revenue, itins, demand, b=b, verbose=True)
    print(f"\nFinal Total Profit: ${results['total_profit']:.2f}")

    dv_first_five(m, t, revenue, itins, demand, b=b, verbose=True)
    

    if m.status == GRB.OPTIMAL or m.status == GRB.TIME_LIMIT:
        m.write('Question2/log_files/KeypathModel2.lp')
        # ff = get_first_five_itins(m, t, itins, demand, revenue, b)
        print("\nOptimal objective value:", m.objVal)
        # print("\nPassenger reallocation (t^r_p):")
        # for (p,r) in sorted(t):
        #     if t[p,r].X > 0.001:
        #         print(f"{t[p,r].X:.2f} passengers originally on {p} reallocated to {r}")

        
        

    else:
        print("Model not solved to optimality, status:", m.status)

    








if __name__ == '__main__':
    main()