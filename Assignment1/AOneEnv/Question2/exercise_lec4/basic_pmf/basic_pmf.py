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

    # print(flights.head())
    # print(itins.head())
    # print(recaps.head())

    flight_nums = flights.index
    capacity    = flights["Capacity"]
    itins       = itins.to_dict('index')

    revenue     = {i: itins[i]['Fare'] for i in itins}      # Revenue per itinerary
    demand      = {i: itins[i]['Demand'] for i in itins}    # Demand per itinerary

    A = {i: {f: int(f in [itins[i]['Leg1'], itins[i]['Leg2']]) for f in flight_nums} for i in itins} # Incidence matrix =1 if itin i uses flight f, else 0

    recap_rates = {j: {i:0.0 for i in itins} for j in itins} # Build recapture matrix, zero except for given data
    for idx, row in recaps.iterrows():
        old_itin = int(row['OldItin'])
        new_itin = int(row['NewItin'])
        recap_rates[old_itin][new_itin] = row['RecapRate']


    # Initialize model
    m = Model('ex4')
    m.params.LogFile = 'Question2/exercise_lec4/basic_pmf/PMF_optimizer.log'

    x = {}
    for p in itins:
        for r in itins:
            if r == p or (r in recap_rates[p] and recap_rates[p][r] > 0):
                x[p,r] = m.addVar(lb=0.0, ub= GRB.INFINITY, vtype=GRB.INTEGER, name=f"x_{p}_{r}") # non-negative x for staying on same flight r==p, and if positive recap_rate exists for p,r

    m.setObjective(quicksum(revenue[r]*x[p,r] for (p,r) in x), GRB.MAXIMIZE)

    # Capacity constraint
    for f in flight_nums:
        m.addConstr(quicksum(A[r][f] * x[p,r] for (p,r) in x) <= capacity[f], name = f"cap_{f}")

    # Demand constraint
    for r in itins:
        lhs = quicksum(x[p,r] / (recap_rates[p][r] if p != r else 1) for p in itins if (p,r) in x)
        rhs = demand[r]
        m.addConstr(lhs <= rhs, name = f'recap_{r}')

    
    m.optimize()
    

    # if m.status == GRB.OPTIMAL or m.status == GRB.TIME_LIMIT:

    #     m.write('Question2/exercise_lec4/basic_pmf/PMF_model.lp')

    #     print("\nOptimal revenue:", m.objVal)
    #     print("\nAccepted passengers per recapture flow (p -> r):")
    #     for (p,r) in sorted(x):  # sort for readability
    #         # if x[p,r].X > 0.001:  # only print non-zero flows
    #         print(f"Passengers originally on itinerary {p} traveling on itinerary {r}: {x[p,r].X:.2f}")
    # else:
    #     print("Model not solved to optimality, status:", m.status)


    if m.status == GRB.OPTIMAL or m.status == GRB.TIME_LIMIT:
        m.write('Question2/exercise_lec4/basic_pmf/PMF_model.lp')
        
        # Separate revenue streams
        original_revenue = 0.0
        recaptured_revenue = 0.0
        
        original_pax = []
        recaptured_pax = []
        
        print("\n" + "="*80)
        print("REVENUE STREAM ANALYSIS")
        print("="*80)
        
        print("\n--- ORIGINAL ITINERARY PASSENGERS (p == r) ---")
        for (p, r) in sorted(x):
            if p == r and x[p,r].X > 0.001:
                pax = x[p,r].X
                rev = revenue[r] * pax
                original_revenue += rev
                original_pax.append((p, pax, rev))
                print(f"Itinerary {p}: {pax:.2f} passengers, Revenue: ${rev:.2f}")
        
        print(f"\nTotal Original Revenue: ${original_revenue:.2f}")
        print(f"Total Original Passengers: {sum([p[1] for p in original_pax]):.2f}")
        
        print("\n--- RECAPTURED PASSENGERS (p != r) ---")
        for (p, r) in sorted(x):
            if p != r and x[p,r].X > 0.001:
                pax = x[p,r].X
                rev = revenue[r] * pax
                recaptured_revenue += rev
                recaptured_pax.append((p, r, pax, rev))
                print(f"Itin {p} → Itin {r}: {pax:.2f} passengers, Revenue: ${rev:.2f}")
        
        print(f"\nTotal Recaptured Revenue: ${recaptured_revenue:.2f}")
        print(f"Total Recaptured Passengers: {sum([p[2] for p in recaptured_pax]):.2f}")
        
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        total_revenue = original_revenue + recaptured_revenue
        print(f"Original Itinerary Revenue:    ${original_revenue:>12.2f} ({original_revenue/total_revenue*100:.1f}%)")
        print(f"Recaptured Itinerary Revenue:  ${recaptured_revenue:>12.2f} ({recaptured_revenue/total_revenue*100:.1f}%)")
        print(f"Total Revenue:                 ${total_revenue:>12.2f}")
        print("="*80)
        
        # Create visualization
        create_revenue_visualization(original_revenue, recaptured_revenue, 
                                    original_pax, recaptured_pax)
    else:
        print("Model not solved to optimality, status:", m.status)


def create_revenue_visualization(orig_rev, recap_rev, orig_pax, recap_pax):
    """Create visualizations comparing the two revenue streams"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Passenger Mix and Revenue Analysis', fontsize=16, fontweight='bold')
    
    # 1. Revenue Pie Chart
    ax1 = axes[0, 0]
    revenues = [orig_rev, recap_rev]
    labels = ['Original Itinerary', 'Recaptured']
    colors = ['#2E86AB', '#A23B72']
    explode = (0.05, 0.05)
    
    ax1.pie(revenues, labels=labels, autopct='%1.1f%%', startangle=90,
            colors=colors, explode=explode, shadow=True)
    ax1.set_title('Revenue Distribution')
    
    # 2. Revenue Bar Chart
    ax2 = axes[0, 1]
    categories = ['Original', 'Recaptured']
    bar_heights = [orig_rev, recap_rev]
    bars = ax2.bar(categories, bar_heights, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Revenue ($)')
    ax2.set_title('Revenue by Stream')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'${height:,.0f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Passenger Count Comparison
    ax3 = axes[1, 0]
    orig_pax_count = sum([p[1] for p in orig_pax])
    recap_pax_count = sum([p[2] for p in recap_pax])
    
    pax_categories = ['Original', 'Recaptured']
    pax_counts = [orig_pax_count, recap_pax_count]
    bars = ax3.bar(pax_categories, pax_counts, color=colors, alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Number of Passengers')
    ax3.set_title('Passenger Count by Stream')
    ax3.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Revenue per Passenger
    ax4 = axes[1, 1]
    rev_per_pax_orig = orig_rev / orig_pax_count if orig_pax_count > 0 else 0
    rev_per_pax_recap = recap_rev / recap_pax_count if recap_pax_count > 0 else 0
    
    rpp_categories = ['Original', 'Recaptured']
    rpp_values = [rev_per_pax_orig, rev_per_pax_recap]
    bars = ax4.bar(rpp_categories, rpp_values, color=colors, alpha=0.7, edgecolor='black')
    ax4.set_ylabel('Revenue per Passenger ($)')
    ax4.set_title('Average Revenue per Passenger')
    ax4.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'${height:.0f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('Question2/exercise_lec4/basic_pmf/revenue_stream_analysis.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved to: Question2/exercise_lec4/basic_pmf/revenue_stream_analysis.png")
    plt.show()








if __name__ == "__main__":
    main()