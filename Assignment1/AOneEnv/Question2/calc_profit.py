import pandas as pd
import numpy as np
from gurobipy import *


def calculate_total_profit_keypath(model, x_vars, revenue, itins, demand, b=None, verbose=True):
        
    # Initialize counters
    original_revenue = 0.0
    recaptured_revenue = 0.0
    original_pax = 0.0
    recaptured_pax = 0.0
    
    original_flows = []
    recaptured_flows = []
    all_pax = 0
    # Calculate total denied passengers per itinerary (from x/t variables)
    denied_pax = {p: 0.0 for p in itins}
    for (p, r) in x_vars:
        if x_vars[p, r].X > 0.001:
            all_pax += x_vars[p,r].X
            denied_pax[p] += x_vars[p, r].X
    
    print(f'\n\n\n allpax: {all_pax}\n\n\n')
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
                'effective_fare': recapture_rate * revenue[r]})
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



"""
def calculate_total_profit_pathbased(model, x_vars, revenue, itins, demand, b=None, verbose=True):
   
    
    # Initialize counters
    original_revenue = 0.0
    recaptured_revenue = 0.0
    original_pax = 0.0
    recaptured_pax = 0.0
    
    original_flows = []
    recaptured_flows = []
    
    # Process all decision variables
    for (p, r) in x_vars:
        if p == 0 or r == 0:  # Skip dummy itinerary if exists
            continue
            
        pax_count = x_vars[p, r].X
        
        # Skip negligible flows
        if pax_count < 0.001:
            continue
        
        # Case 1: Original passengers (p == r)
        if p == r:
            flow_revenue = pax_count * revenue[p]
            original_pax += pax_count
            original_revenue += flow_revenue
            
            original_flows.append({
                'itinerary': p,
                'passengers': pax_count,
                'revenue': flow_revenue,
                'fare': revenue[p]
            })
        
        # Case 2: Recaptured passengers (p != r)
        else:
            # Determine recapture rate for reporting purposes
            if b is not None and p in b and r in b[p]:
                recapture_rate = b[p][r]
            else:
                recapture_rate = 1.0
            
            # Only count if there's actual recapture possibility
            if recapture_rate > 0:
                # Revenue = passengers × fare of itinerary r (FULL fare, not discounted)
                # The recapture rate affects constraint (acceptance), not revenue (payment)
                flow_revenue = pax_count * revenue[r]
                
                recaptured_pax += pax_count
                recaptured_revenue += flow_revenue
                
                recaptured_flows.append({
                    'from_itin': p,
                    'to_itin': r,
                    'passengers': pax_count,
                    'revenue': flow_revenue,
                    'fare': revenue[r],
                    'recapture_rate': recapture_rate,
                    'effective_fare': revenue[r]  # Full fare paid
                })
    
    # Calculate totals
    total_revenue = original_revenue + recaptured_revenue
    total_pax = original_pax + recaptured_pax
    
    # Calculate denied passengers (demand not satisfied)
    # NOTE: In the constraint Σ(x_p^r / b_p^r) ≤ D_p, the LHS represents effective demand satisfied
    total_denied = 0.0
    denied_by_itin = {}
    
    for p in demand:
        if p == 0:  # Skip dummy
            continue
        
        # Calculate satisfied demand using the same formula as constraint C2
        # Σ(x_p^r / b_p^r) represents the effective demand satisfied for type p
        satisfied = 0.0
        for r in itins:
            if (p, r) in x_vars and b is not None and p in b and r in b[p] and b[p][r] > 0:
                satisfied += x_vars[p, r].X / b[p][r]
        
        denied = max(0, demand[p] - satisfied)
        
        if denied > 0.001:
            total_denied += denied
            denied_by_itin[p] = denied
    
    # Print detailed breakdown if requested
    if verbose:
        print("\n" + "="*80)
        print("PROFIT CALCULATION - PATH-BASED FORMULATION")
        print("="*80)
        
        print("\n--- ORIGINAL ITINERARY PASSENGERS (p == r) ---")
        for flow in sorted(original_flows, key=lambda x: x['itinerary']):
            print(f"Itinerary {flow['itinerary']:>3}: "
                  f"{flow['passengers']:>8.2f} passengers × ${flow['fare']:>6.2f} = "
                  f"${flow['revenue']:>10.2f}")
        print(f"\nSubtotal - Original: {original_pax:>8.2f} passengers, "
              f"${original_revenue:>10.2f} revenue")
        
        if recaptured_flows:
            print("\n--- RECAPTURED PASSENGERS (p → r, where p ≠ r) ---")
            for flow in sorted(recaptured_flows, key=lambda x: (x['from_itin'], x['to_itin'])):
                print(f"Itin {flow['from_itin']:>3} → {flow['to_itin']:>3}: "
                      f"{flow['passengers']:>8.2f} pax × ${flow['fare']:>6.2f} × {flow['recapture_rate']:.3f} = "
                      f"${flow['revenue']:>10.2f}")
            print(f"\nSubtotal - Recaptured: {recaptured_pax:>8.2f} passengers, "
                  f"${recaptured_revenue:>10.2f} revenue")
        else:
            print("\n--- RECAPTURED PASSENGERS (p → r, where p ≠ r) ---")
            print("None (basic PMF without recapture)")
        
        if denied_by_itin:
            print("\n--- DENIED PASSENGERS (no available capacity) ---")
            for p in sorted(denied_by_itin.keys()):
                print(f"Itinerary {p:>3}: {denied_by_itin[p]:>8.2f} passengers denied")
            print(f"\nTotal Denied: {total_denied:>8.2f} passengers")
        
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        if total_revenue > 0:
            print(f"Original Revenue:    ${original_revenue:>12.2f} "
                  f"({original_revenue/total_revenue*100:>5.1f}%) | "
                  f"{original_pax:>8.2f} pax")
            if recaptured_revenue > 0:
                print(f"Recaptured Revenue:  ${recaptured_revenue:>12.2f} "
                      f"({recaptured_revenue/total_revenue*100:>5.1f}%) | "
                      f"{recaptured_pax:>8.2f} pax")
        else:
            print(f"Original Revenue:    ${original_revenue:>12.2f} | {original_pax:>8.2f} pax")
            if recaptured_revenue > 0:
                print(f"Recaptured Revenue:  ${recaptured_revenue:>12.2f} | {recaptured_pax:>8.2f} pax")
        print("-" * 80)
        print(f"TOTAL PROFIT:        ${total_revenue:>12.2f} | {total_pax:>8.2f} pax")
        if total_denied > 0:
            print(f"Passengers Denied:   {total_denied:>8.2f} pax")
        print("="*80 + "\n")
    
    # Return comprehensive results
    return {
        'total_profit': total_revenue,
        'original_profit': original_revenue,
        'recaptured_profit': recaptured_revenue,
        'original_passengers': original_pax,
        'recaptured_passengers': recaptured_pax,
        'total_passengers': total_pax,
        'denied_passengers': total_denied,
        'original_flows': original_flows,
        'recaptured_flows': recaptured_flows,
        'denied_by_itinerary': denied_by_itin
    }

"""


def calculate_total_profit_basic(model, x_vars, revenue, itins, demand, b=None, verbose=True):
        
    total_rev = 0.0

    for (p,r) in x_vars:
        if x_vars[p,r].X > 0.0001:
            total_rev += revenue[r] * x_vars[p,r].X

    return total_rev

def calculate_total_profit_pathbased(model, t, revenue, itins, demand, b=None, verbose=False):
    """
    Calculate total profit for the path-based passenger reallocation model.
    
    Parameters:
    -----------
    model : Gurobi model
        The optimized model
    t : dict
        Decision variables t[p,r] = passengers from itinerary p reallocated to r
    revenue : dict
        Revenue per itinerary {p: fare}
    itins : dict
        Itinerary data
    demand : dict
        Demand per itinerary {p: demand}
    b : dict, optional
        Recapture rates b[p][r]
    verbose : bool
        Print detailed breakdown
        
    Returns:
    --------
    dict with keys:
        - total_profit: final total profit
        - total_spilled: total number of passengers spilled
        - revenue_accepted: revenue from accepted passengers
        - revenue_lost_spilled: revenue lost from spilled passengers
        - passengers_accepted: number of passengers accepted
        - passengers_spilled: number of passengers spilled
        - passengers_recaptured: number of passengers recaptured
    """
    
    # Extract solution values
    t_sol = {}
    for (p, r) in t:
        t_sol[p, r] = t[p, r].X if hasattr(t[p, r], 'X') else 0.0
    
    # Calculate passengers denied on each itinerary p
    passengers_denied = {}
    for p in itins:
        denied = sum(t_sol.get((p, r), 0.0) for r in itins if r != p)
        passengers_denied[p] = denied
    
    # Calculate passengers recaptured onto each itinerary r
    passengers_recaptured_to = {}
    for r in itins:
        if b is not None:
            recaptured = sum(b[p][r] * t_sol.get((p, r), 0.0) for p in itins if p != r)
        else:
            recaptured = 0.0
        passengers_recaptured_to[r] = recaptured
    
    # Calculate final accepted passengers on each itinerary
    passengers_accepted = {}
    total_accepted = 0
    total_spilled = 0
    
    for p in itins:
        if p == 0: # skip dummy
            continue
        # Original demand minus denied plus recaptured
        # print(demand)
        accepted = demand[p] - passengers_denied[p] + passengers_recaptured_to[p]
        passengers_accepted[p] = accepted
        total_accepted += accepted
        
        # Spilled = denied but not recaptured elsewhere
        spilled = passengers_denied[p] - sum(
            b[p][r] * t_sol.get((p, r), 0.0) for r in itins if r != p
        ) if b is not None else passengers_denied[p]
        total_spilled += spilled
    
    # Calculate revenues
    revenue_accepted = sum(passengers_accepted[p] * revenue[p] for p in itins if p != 0)
    revenue_lost_spilled = sum(
        (passengers_denied[p] - sum(
            b[p][r] * t_sol.get((p, r), 0.0) for r in itins if r != p
        )) * revenue[p] for p in itins
    ) if b is not None else sum(passengers_denied[p] * revenue[p] for p in itins)
    
    total_profit = revenue_accepted
    total_recaptured = sum(passengers_recaptured_to.values())
    
    if verbose:
        print("\n" + "="*60)
        print("PROFIT CALCULATION SUMMARY")
        print("="*60)
        print(f"\nTotal Original Demand: {sum(demand.values()):.0f} passengers")
        print(f"Total Passengers Accepted: {total_accepted:.0f} passengers")
        print(f"Total Passengers Spilled: {total_spilled:.0f} passengers")
        print(f"Total Passengers Recaptured: {total_recaptured:.0f} passengers")
        print(f"\nRevenue from Accepted Passengers: ${revenue_accepted:,.2f}")
        print(f"Revenue Lost from Spilled Passengers: ${revenue_lost_spilled:,.2f}")
        print(f"\nTotal Profit: ${total_profit:,.2f}")
        print("="*60)
        
        # Show top 5 itineraries by passengers denied
        if any(passengers_denied.values()):
            print("\nTop Itineraries by Passengers Denied:")
            sorted_denied = sorted(passengers_denied.items(), key=lambda x: x[1], reverse=True)
            for i, (p, denied) in enumerate(sorted_denied[:5]):
                if denied > 0.1:
                    print(f"  Itinerary {p}: {denied:.0f} passengers denied")
    
    return {
        'total_profit': total_profit,
        'total_spilled': total_spilled,
        'revenue_accepted': revenue_accepted,
        'revenue_lost_spilled': revenue_lost_spilled,
        'passengers_accepted': total_accepted,
        'passengers_spilled': total_spilled,
        'passengers_recaptured': total_recaptured,
        'passengers_accepted_by_itin': passengers_accepted,
        'passengers_denied_by_itin': passengers_denied
    }

def dv_first_five(model, t, revenue, itins, demand, b, verbose=False):

    for (p,r) in t:
        if (r <= 5 or p <= 5) and t[p,r].X>0.001 and r != 0.0 and p !=0.0:
            print(f't_{p}_{r}: {t[p,r].X}')

def get_first_five_itins(master_model, itins, pi, sigma, t, revenue, b, delta, flight_idx):

    first_five = sorted([p for p in itins.keys() if p != 0])[:5]
    results = {}
    
    for p in first_five:
        print(f"\n--- ITINERARY {p} ---")
        
        # Decision variables: reallocations FROM itinerary p
        total_from = 0
        for (pp, r) in t:
            if pp == p and t[pp, r].X > 0.001:
                print(f"  t[{pp},{r}] = {t[pp,r].X:.2f} → Itin {r} (Fare ${revenue[r]:.2f})")
                total_from += t[pp, r].X
        print(f"  Total: {total_from:.2f}")
        
        # Decision variables: recaptures TO itinerary p
        total_to = 0
        for (r, pp) in t:
            if pp == p and t[r, pp].X > 0.001 and b[r][p] > 0:
                recaptured = t[r, pp].X * b[r][p]
                print(f"  t[{r},{pp}] = {t[r,pp].X:.2f} × {b[r][p]:.2%} = {recaptured:.2f}")
                total_to += recaptured
        print(f"  Total: {total_to:.2f}")
        print(f"  Net change: {total_to - total_from:+.2f}")
        
        results[p] = {
            'reallocated_from': total_from,
            'recaptured_to': total_to,
            'net_change': total_to - total_from}
    
    return results

def get_first_five_flight_duals(pi, capacity, Q, flight_idx, delta, itins, t):
    first_five_flights = flight_idx[:5]
    print(f'First five flights: {first_five_flights}')
    
    results = {}
    
    for i in first_five_flights:
        # Calculate actual usage
        removed = sum(delta[p][i] * t[p,r].X for (p,r) in t if delta[p][i] > 0)
        recaptured = sum(delta[r][i] * itins[p].get('RecapRate', {}).get(r, 0) * t[p,r].X 
                        for (p,r) in t if delta[r][i] > 0)
        net_load = Q[i] - (removed - recaptured)
        slack = net_load - capacity[i]
        
        # Find itineraries using this flight
        itins_using = [p for p in itins if delta[p][i] > 0 and p != 0]
        
        print(f"\nFlight {i}:")
        # print(f"  Capacity: {capacity[i]:.0f}, Unconstrained demand: {Q[i]:.0f}")
        print(f"Slack: {slack:.2f}")
        print(f"π_{i} = {pi[i]:.6f} → {'slack (spare capacity)' if pi[i] > 0.001 else 'BINDING (at capacity)'}")
        print(f"Used by {len(itins_using)} itineraries")
        
        results[i] = {
            'capacity': capacity[i],
            'unconstrained_demand': Q[i],
            'net_load': net_load,
            'slack': slack,
            'pi': pi[i],
            'itineraries_using': itins_using}
    
    return results





















