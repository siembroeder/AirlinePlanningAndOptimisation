import pandas as pd
import numpy as np
from gurobipy import *


def calculate_total_profit_basic(model, x_vars, revenue, itins, demand, b=None, verbose=True):
        
    total_rev = 0.0

    for (p,r) in x_vars:
        if x_vars[p,r].X > 0.0001:
            total_rev += revenue[r] * x_vars[p,r].X

    return total_rev

def calculate_total_profit_pathbased(model, t, revenue, itins, demand, b=None, verbose=False):
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
        if p == 999: # skip dummy
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
    revenue_accepted = sum(passengers_accepted[p] * revenue[p] for p in itins if p != 999)
    revenue_lost_spilled = sum(
        (passengers_denied[p] - sum(
            b[p][r] * t_sol.get((p, r), 0.0) for r in itins if r != p
        )) * revenue[p] for p in itins
    ) if b is not None else sum(passengers_denied[p] * revenue[p] for p in itins)
    
    total_profit = revenue_accepted
    total_recaptured = sum(passengers_recaptured_to.values())
    
    if verbose:
        print(f"\nTotal Original Demand: {sum(demand.values()):.0f} passengers")
        print(f"Total Passengers Accepted: {total_accepted:.0f} passengers")
        print(f"Total Passengers Spilled: {total_spilled:.0f} passengers")
        print(f"Total Passengers Recaptured: {total_recaptured:.0f} passengers")
        print(f"\nRevenue from Accepted Passengers: ${revenue_accepted:,.2f}")
        print(f"Revenue Lost from Spilled Passengers: ${revenue_lost_spilled:,.2f}")
        print(f"\nTotal Profit: ${total_profit:,.2f}")

    
    return {
        'total_profit': total_profit,
        'total_spilled': total_spilled,
        'revenue_accepted': revenue_accepted,
        'revenue_lost_spilled': revenue_lost_spilled,
        'passengers_accepted': total_accepted,
        'passengers_spilled': total_spilled,
        'passengers_recaptured': total_recaptured,
        'passengers_accepted_by_itin': passengers_accepted,
        'passengers_denied_by_itin': passengers_denied}

def first_five_vars(model, t, revenue, itins, demand, b, verbose=False):
    n = 4
    single_outflow = 0
    double_outflow = 0
    tosingle = 0
    for (p,r) in t:
        if (r <= n or p <= n) and t[p,r].X>0.001 and r != 999 and p !=999:
            print(f't_{p}_{r}: {t[p,r].X}')
            if p <= 145:
                single_outflow +=1
            elif p >=146:
                double_outflow +=1
                if r <=145:
                    tosingle +=1
    print(f'singleoutflow: {single_outflow}, doubleoutflow: {double_outflow}, tosingle: {tosingle}')
    
def get_first_five_itins(master_model, itins, pi, sigma, t, revenue, b, delta, flight_idx):

    first_five = sorted([p for p in itins.keys() if p != 999])[:5]
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
        print(f"π_{i} = {pi[i]:.6f}")
        print(f"Used by {len(itins_using)} itineraries")
        
        results[i] = {
            'capacity': capacity[i],
            'unconstrained_demand': Q[i],
            'net_load': net_load,
            'slack': slack,
            'pi': pi[i],
            'itineraries_using': itins_using}
    
    return results





















