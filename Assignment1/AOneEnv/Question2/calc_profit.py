import pandas as pd
import numpy as np
from gurobipy import *


def calculate_total_profit(model, x_vars, revenue, itins, demand, b=None, verbose=True):
    """
    Calculate total profit from both original and recaptured passengers.
    
    Parameters:
    -----------
    model : gurobipy.Model
        Solved Gurobi model
    x_vars : dict
        Dictionary of decision variables with keys (p, r) where:
        - For basic PMF: x[p,r] = passengers originally on p, traveling on r
        - For column generation: t[p,r] = passengers DENIED on p, recaptured on r
    revenue : dict
        Revenue per itinerary {itinerary_id: fare}
    itins : dict
        Itinerary information with 'Demand' field
    demand : dict
        Demand per itinerary {itinerary_id: demand_value}
    b : dict, optional
        Recapture rates {p: {r: rate}}. If None, will be computed from x_vars
    verbose : bool
        If True, print detailed breakdown
    
    Returns:
    --------
    dict containing:
        - 'total_profit': total revenue
        - 'original_profit': revenue from passengers on original itinerary
        - 'recaptured_profit': revenue from recaptured passengers
        - 'original_passengers': count of passengers on original itinerary
        - 'recaptured_passengers': count of recaptured passengers
        - 'total_passengers': total passenger count
        - 'original_flows': detailed list of original flows
        - 'recaptured_flows': detailed list of recaptured flows
    """
    
    # if model.status != GRB.OPTIMAL and model.status != GRB.TIME_LIMIT:
        # print(f"Warning: Model not solved to optimality. Status: {model.status}")
        # return None
    
    # Initialize counters
    original_revenue = 0.0
    recaptured_revenue = 0.0
    original_pax = 0.0
    recaptured_pax = 0.0
    
    original_flows = []
    recaptured_flows = []
    
    # Calculate total denied passengers per itinerary (from x/t variables)
    denied_pax = {p: 0.0 for p in itins}
    for (p, r) in x_vars:
        if x_vars[p, r].X > 0.001:
            denied_pax[p] += x_vars[p, r].X
    
    # Calculate original passengers (demand minus denied)
    for p in itins:
        if p == 0:  # Skip dummy itinerary
            continue
        
        original_demand = demand.get(p, itins[p].get('Demand', 0))
        denied = denied_pax.get(p, 0.0)
        
        # Original passengers are those who were NOT denied
        original = max(0, original_demand - denied)
        
        if original > 0.001:
            original_pax += original
            original_revenue += original * revenue[p]
            original_flows.append({
                'itinerary': p,
                'passengers': original,
                'revenue': original * revenue[p],
                'fare': revenue[p]
            })
    
    # Calculate recaptured passengers from decision variables
    for (p, r) in x_vars:
        if p == 0 or r == 0:  # Skip dummy itinerary
            continue
            
        pax_count = x_vars[p, r].X
        
        # Skip negligible flows
        if pax_count < 0.001:
            continue
        
        # These are denied passengers who were recaptured
        # Revenue depends on recapture rate b[p][r]
        if b is not None and p in b and r in b[p]:
            recapture_rate = b[p][r]
        else:
            recapture_rate = 1.0  # Basic PMF assumes full revenue
        
        if b[p][r] > 0:
            # Calculate revenue: recapture_rate * fare * passengers
            flow_revenue = recapture_rate * revenue[r] * pax_count
            
            recaptured_revenue += flow_revenue
            recaptured_pax += pax_count
            recaptured_flows.append({
                'from_itin': p,
                'to_itin': r,
                'passengers': pax_count,
                'revenue': flow_revenue,
                'fare': revenue[r],
                'recapture_rate': recapture_rate,
                'effective_fare': recapture_rate * revenue[r]
            })
        else:
            continue
    
    # Calculate totals
    total_revenue = original_revenue + recaptured_revenue
    total_pax = original_pax + recaptured_pax
    
    # Print detailed breakdown if requested
    if verbose:
        print("\n" + "="*80)
        print("PROFIT CALCULATION - DETAILED BREAKDOWN")
        print("="*80)
        
        print("\n--- ORIGINAL ITINERARY PASSENGERS (p == r) ---")
        for flow in sorted(original_flows, key=lambda x: x['itinerary']):
            print(f"Itinerary {flow['itinerary']}: "
                  f"{flow['passengers']:>8.2f} passengers × ${flow['fare']:>6.2f} = "
                  f"${flow['revenue']:>10.2f}")
        print(f"\nSubtotal - Original: {original_pax:>8.2f} passengers, "
              f"${original_revenue:>10.2f} revenue")
        
        print("\n--- RECAPTURED PASSENGERS (p → r) ---")
        for flow in sorted(recaptured_flows, key=lambda x: (x['from_itin'], x['to_itin'])):
            print(f"Itin {flow['from_itin']:>3} → {flow['to_itin']:>3}: "
                  f"{flow['passengers']:>8.2f} pax × ${flow['fare']:>6.2f} × {flow['recapture_rate']:.3f} = "
                  f"${flow['revenue']:>10.2f}")
        print(f"\nSubtotal - Recaptured: {recaptured_pax:>8.2f} passengers, "
              f"${recaptured_revenue:>10.2f} revenue")
        
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        if total_revenue > 0:
            print(f"Original Revenue:    ${original_revenue:>12.2f} "
                  f"({original_revenue/total_revenue*100:>5.1f}%) | "
                  f"{original_pax:>8.2f} pax")
            print(f"Recaptured Revenue:  ${recaptured_revenue:>12.2f} "
                  f"({recaptured_revenue/total_revenue*100:>5.1f}%) | "
                  f"{recaptured_pax:>8.2f} pax")
        else:
            print(f"Original Revenue:    ${original_revenue:>12.2f} | {original_pax:>8.2f} pax")
            print(f"Recaptured Revenue:  ${recaptured_revenue:>12.2f} | {recaptured_pax:>8.2f} pax")
        print("-" * 80)
        print(f"TOTAL PROFIT:        ${total_revenue:>12.2f} | {total_pax:>8.2f} pax")
        print("="*80 + "\n")
    
    # Return comprehensive results
    return {
        'total_profit': total_revenue,
        'original_profit': original_revenue,
        'recaptured_profit': recaptured_revenue,
        'original_passengers': original_pax,
        'recaptured_passengers': recaptured_pax,
        'total_passengers': total_pax,
        'original_flows': original_flows,
        'recaptured_flows': recaptured_flows
    }


def calculate_profit_difference(model, x_vars, revenue, itins, demand):
    """
    Calculate profit gained/lost compared to unconstrained demand scenario.
    
    Parameters:
    -----------
    model : gurobipy.Model
        Solved Gurobi model
    x_vars : dict
        Decision variables {(p,r): var}
    revenue : dict
        Revenue per itinerary
    itins : dict
        Itinerary information
    demand : dict
        Original demand per itinerary
    
    Returns:
    --------
    dict with spillage and recapture analysis
    """
    
    # Calculate actual profit
    results = calculate_total_profit(model, x_vars, revenue, itins, verbose=False)
    actual_profit = results['total_profit']
    
    # Calculate theoretical maximum (unconstrained)
    max_profit = sum(demand[p] * revenue[p] for p in demand if p in revenue)
    
    # Calculate spillage
    total_demand = sum(demand[p] for p in demand if p in revenue)
    served_pax = results['total_passengers']
    spilled_pax = total_demand - served_pax
    
    profit_loss = max_profit - actual_profit
    
    print("\n" + "="*80)
    print("PROFIT ANALYSIS vs UNCONSTRAINED DEMAND")
    print("="*80)
    print(f"Maximum Possible Profit (unconstrained): ${max_profit:>12.2f}")
    print(f"Actual Profit (optimized):               ${actual_profit:>12.2f}")
    print(f"Profit Loss due to Capacity:             ${profit_loss:>12.2f} "
          f"({profit_loss/max_profit*100:.1f}%)")
    print("-" * 80)
    print(f"Total Demand:                            {total_demand:>12.2f} pax")
    print(f"Passengers Served:                       {served_pax:>12.2f} pax")
    print(f"Passengers Spilled:                      {spilled_pax:>12.2f} pax "
          f"({spilled_pax/total_demand*100:.1f}%)")
    print(f"Recapture Rate:                          "
          f"{results['recaptured_passengers']/served_pax*100:.1f}% of served")
    print("="*80 + "\n")
    
    return {
        'max_profit': max_profit,
        'actual_profit': actual_profit,
        'profit_loss': profit_loss,
        'total_demand': total_demand,
        'served_passengers': served_pax,
        'spilled_passengers': spilled_pax,
        'recapture_efficiency': results['recaptured_passengers']/served_pax if served_pax > 0 else 0
    }


# Example usage for both scripts:
"""
# For basic PMF (script 1):
if m.status == GRB.OPTIMAL:
    results = calculate_total_profit(m, x, revenue, itins, demand, b=recap_rates, verbose=True)
    print(f"Total Profit: ${results['total_profit']:.2f}")
    
    # With demand comparison:
    analysis = calculate_profit_difference(m, x, revenue, itins, demand)

# For Column Generation (script 2):
if final_model.status == GRB.OPTIMAL:
    results = calculate_total_profit(final_model, t, revenue, itins, demand, b=b, verbose=True)
    print(f"Total Profit: ${results['total_profit']:.2f}")
    
    # With demand comparison:
    analysis = calculate_profit_difference(final_model, t, revenue, itins, demand)
"""