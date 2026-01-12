"""
GROUP 7

Siem Broeder - 6565662
Job Ruijters - 5073138
Radu Pop - 5716527

"""

# Imports
from compute_parameters import build_problem_data
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

# Define paths of input data
BASE_DIR = Path(__file__).resolve().parent
airports_path = BASE_DIR / "Data" / "DemandGroup7.xlsx"
aircraft_path = BASE_DIR / "Data" / "FleetType.xlsx"
hours_path    = BASE_DIR / "Data" / "HourCoefficients.xlsx"

# Build data 
data = build_problem_data(airports_path, hours_path, aircraft_path)

# Parameters
TIME_STEP   = 6                 # minutes
TOTAL_TIME = 24 * 60            # total time in minutes
TIMES      = list(range(0, TOTAL_TIME + 1, TIME_STEP))  # time steps

AIRPORTS = data['distance'].index.tolist()  # list of airports names
HUB = 'Amsterdam'
MIN_BLOCK = 360  # 6 hours minimum total block time

# Transfer parameters
MIN_CONNECTION = 40 # minimum connection time in minutes
MAX_CONNECTION = 180 # maximum connection time in minutes

# Initialize transfer demand tracking (spoke to spoke demand)
transfer_demand = {}
for orig in AIRPORTS:
    if orig == HUB:
        continue
    for dest in AIRPORTS:
        if dest == HUB or dest == orig:
            continue
        # Initialize with total daily demand for spoke-to-spoke routes
        total_daily = sum(
            data['hourly_demand'].loc[(orig, dest, h)]
            for h in range(24)
        )
        transfer_demand[(orig, dest)] = total_daily

# Initialize final routes list
final_routes = []

# Initialize arriving passengers at hub for transfering
hub_arrivals = {}

# Main loop over aircraft types
for ac_type in reversed(data['aircraft_types']):
    fleet_size = int(data['aircraft']['fleet'][ac_type])
    speed      = data['aircraft'].loc[ac_type, 'speed']
    seats      = data['aircraft'].loc[ac_type, 'seats']
    TAT        = int(data['aircraft'].loc[ac_type, 'TAT'])
    max_range  = data['aircraft'].loc[ac_type, 'range']
    runway_req = data['aircraft'].loc[ac_type, 'runway']
    lease_cost = data['aircraft'].loc[ac_type, 'lease_cost']

    # Loop over individual planes
    for ac_id in range(fleet_size):

        # Initialize value and policy to obtain that value
        V = {}
        policy = {}

        # Terminal condition: only allow ending at hub, if not penalize
        for a in AIRPORTS:
            if a == HUB:
                V[(TOTAL_TIME, a)] = 0         
            else:
                V[(TOTAL_TIME, a)] = -1e9       

        # Backward DP
        for t in reversed(TIMES[:-1]):

            hour = min(int(t / 60), 23)         # current hour (0-23), 24 not included

            # Loop over all airports
            for i in AIRPORTS:

                # Initial condition: only allow starting at hub, if not penalize
                if t == 0 and i != HUB:
                    V[(t, i)] = -1e9
                    policy[(t, i)] = ('invalid',)
                    continue
                
                # Option 1: Stay idle
                best_value = V[(t + TIME_STEP, i)]
                best_action = ('stay', i, t + TIME_STEP)

                # Option 2: Fly to another airport
                for j in AIRPORTS:
                    if i == j:                      # no self-loops
                        continue
                    if i != HUB and j != HUB:       # no spoke-to-spoke flights
                        continue
                    
                    dist = data['distance'].loc[i, j]

                    # Check range and runway constraints
                    if dist > max_range:
                        continue
                    if data['airport_info'].loc[j, 'runway'] < runway_req:
                        continue

                    flight_time = int((dist / speed) * 60 + 30)         # flight time in minutes + 15 min for climb and 15 min for descent

                    t_ready = ((t + flight_time + TAT + TIME_STEP - 1) // TIME_STEP) * TIME_STEP        # next available time (including TAT), rounded up to next time step

                    # Cannot arrive after end of day
                    if t_ready > TOTAL_TIME:
                        continue
                    
                    # Demand (last 3 hours)
                    total_direct_demand = sum(
                        data['hourly_demand'].loc[(i, j, h)]
                        for h in range(max(0, hour - 2), hour + 1)
                    )

                    required_pax = int(seats * 0.8)         # Assumed 80% load factor for all flights
                    direct_pax = int(min(total_direct_demand, required_pax))       
                    
                    transfer_pax = 0
                    transfer_revenue = 0
                    transfer_breakdown = {}                 # Initialize transfer origin/destination
                    
                    remaining_needed = required_pax - direct_pax        # Check if there is place on-board for transfer passengers
                    
                    # Get transfer passengers for OUTBOUND flights
                    if i == HUB and j != HUB and remaining_needed > 0:    

                        # Loop over all hub transfering arrivals (computed at a later stage)                      
                        for (origin_spoke, arr_time), available_pax in hub_arrivals.items():

                            if available_pax <= 0 or remaining_needed <= 0:
                                continue
                            
                            connection_time = t - arr_time
                            
                            if MIN_CONNECTION <= connection_time <= MAX_CONNECTION:          # Make sure the connection time is within set limits
                                if (origin_spoke, j) in transfer_demand:                     # Check if there is demand between the spoke - spoke route
                                    available_transfer_demand = transfer_demand[(origin_spoke, j)]
                                    
                                    if available_transfer_demand > 0:
                                        can_transfer = min(
                                            available_pax,
                                            remaining_needed,
                                            available_transfer_demand
                                        )                               # If there is, the pax that can transfer is the minimum between the transferring passengers waiting 
                                                                        # at the hub, the empty seats on the flight for 80% LF and the demand
                                        
                                        if can_transfer > 0:
                                            transfer_pax += can_transfer        # Track total number of transferring pax
                                            remaining_needed -= can_transfer    # Remove taken seats from flight
                                            transfer_breakdown[origin_spoke] = can_transfer     # Track origin of transferring passengers
                                            
                                            # Compute revenue for transferring passengers: use the spoke - spoke yield but only 2nd leg distance 
                                            leg2_dist = data['distance'].loc[HUB, j]
                                            transfer_revenue += can_transfer * data['yield'].loc[origin_spoke, j] * leg2_dist
                                            
                                            # Make sure all transferring passengers have a seat
                                            if remaining_needed <= 0:
                                                break

                    # Compute total number of pax for the flight                                                  
                    total_pax = direct_pax + transfer_pax
                    
                    # Force exactly 80% loading factor
                    if total_pax != required_pax:
                        continue
                    
                    # Compute total revenue
                    direct_revenue = direct_pax * data['yield'].loc[i, j] * dist
                    total_revenue = direct_revenue + transfer_revenue
                    
                    # Compute operating cost and profit 
                    cost = data['operating_cost'][ac_type].loc[i, j]
                    profit = total_revenue - cost

                    # Discard unprofitable flights
                    if profit <= 0:         
                        continue
                    
                    # Update total value
                    value = profit + V[(t_ready, j)]

                    # Update best action if better
                    if value > best_value:
                        best_value = value
                        best_action = ('fly', j, t_ready, direct_pax, transfer_pax, profit, flight_time + TAT, transfer_breakdown.copy())

                V[(t, i)] = best_value
                policy[(t, i)] = best_action

        # Reconstruct routes using optimal policy
        t = 0                   # initialize time
        loc = HUB               # initialize location
        route = []              # initialize route list
        total_block = 0         # initialize total block time

        while t < TOTAL_TIME:
            action = policy[(t, loc)]       # get optimal policy

            if action[0] == 'stay':         # if stay, just update time
                t = action[2]
            elif action[0] == 'fly':        # if fly, record flight details
                _, dest, t_next, direct_pax, transfer_pax, profit, block, transfer_breakdown = action
                
                route.append({
                    'aircraft_type': ac_type,
                    'aircraft_id': ac_id,
                    'dep_time': t,
                    'origin': loc,
                    'dest': dest,
                    'arr_time': t_next,
                    'direct_passengers': direct_pax,
                    'transfer_passengers': transfer_pax,
                    'total_passengers': direct_pax + transfer_pax,
                    'profit': profit,
                    'block_time': block,
                    'transfer_from': transfer_breakdown
                })
                total_block += block        # update total block time
                t = t_next                  # update time: arrival time at destination
                loc = dest                  # update location: arrival airport
            else:
                break

        # Only keep aircraft that meet operational constraints
        if loc == HUB and total_block >= MIN_BLOCK and route:
            flight_profit = sum(f['profit'] for f in route)
            lease_cost = data['aircraft'].loc[ac_type, 'lease_cost']
            net_profit = flight_profit - lease_cost
            
            # Verify all flights have exactly 80% load factor
            all_flights_at_80 = all(
                f['total_passengers'] == int(data['aircraft'].loc[ac_type, 'seats'] * 0.8)
                for f in route
            )
            
            # Only include aircraft with positive net profit and correct load factor
            if net_profit > 0 and all_flights_at_80:
                final_routes.append(route)

                # Update demand and transfer tracking for future aircraft
                for r in route:
                    hour = int(r['dep_time'] / 60)
                    
                    # Deduct served direct demand from available hourly demand
                    remaining_direct = r['direct_passengers']
                    for h in range(hour, max(-1, hour - 3), -1):        # loop backwards through last 3 hours
                        if remaining_direct <= 0:
                            break
                        idx = (r['origin'], r['dest'], h)
                        available = data['hourly_demand'].loc[idx]
                        served = min(available, remaining_direct)
                        data['hourly_demand'].loc[idx] -= served        # update remaining demand
                        remaining_direct -= served
                    
                    # Track passengers arriving at hub for potential transfers
                    if r['dest'] == HUB:
                        key = (r['origin'], r['arr_time'])
                        hub_arrivals[key] = r['direct_passengers']
                    
                    # Update transfer demand and hub arrivals when transfers are used
                    if r['origin'] == HUB and r['transfer_passengers'] > 0:
                        for origin_spoke, transfer_count in r.get('transfer_from', {}).items():

                            # Deduct from spoke-spoke transfer demand
                            if (origin_spoke, r['dest']) in transfer_demand:
                                transfer_demand[(origin_spoke, r['dest'])] -= transfer_count
                            
                            # Deduct transferring passengers from hub arrivals
                            for (arr_origin, arr_time), available_pax in list(hub_arrivals.items()):
                                if arr_origin == origin_spoke:
                                    connection_time = r['dep_time'] - arr_time
                                    if MIN_CONNECTION <= connection_time <= MAX_CONNECTION:
                                        deduct = min(available_pax, transfer_count)
                                        hub_arrivals[(arr_origin, arr_time)] -= deduct
                                        transfer_count -= deduct
                                        if transfer_count <= 0:
                                            break

# VERIFICATION OF RESULTS AND PRINTING
# Compute total profit and verify load factors again
total_profit = 0
total_flights = 0
total_pax = 0
total_capacity = 0

# Loop over all aircraft
for ac in final_routes:
    ac_type = ac[0]['aircraft_type']
    seats = data['aircraft'].loc[ac_type, 'seats']
    flight_profit = sum(f['profit'] for f in ac)
    lease_cost = data['aircraft'].loc[ac_type, 'lease_cost']
    total_profit += flight_profit - lease_cost
    
    # Sum to make sure all passengers are accounted for (all departing pax have arrived)
    for f in ac:
        total_flights += 1
        total_pax += f['total_passengers']
        total_capacity += seats * 0.8

# Network used capacity should be 100% (pax vs seats - see comment above)
network_used_capacity = total_pax / total_capacity if total_capacity > 0 else 0

# Print summary statistics
print(f"\nTotal network profit (after lease): {total_profit:,.0f}")
print(f"Network-wide load factor verification: {network_used_capacity:.2%} (should be 100%)")
print(f"Total flights: {total_flights}")
print(f"Total passengers: {total_pax:,}\n")

# Convert minutes to HH:MM format for display
def min_to_hhmm(t):
    h = t // 60
    m = t % 60
    return f"{int(h):02d}:{int(m):02d}"

# Calculate passenger breakdown by type
total_direct = sum(f['direct_passengers'] for ac in final_routes for f in ac)
total_transfer = sum(f['transfer_passengers'] for ac in final_routes for f in ac)
print(f"Total direct passengers: {total_direct:,}")
print(f"Total transfer passengers: {total_transfer:,}")
print(f"Transfer percentage: {100*total_transfer/(total_direct+total_transfer):.1f}%\n")

# Create summary table for each aircraft
rows = []
for ac in final_routes:
    ac_type = ac[0]['aircraft_type']
    ac_id   = ac[0]['aircraft_id']
    flight_time_mins = sum(f['arr_time'] - f['dep_time'] for f in ac)
    direct_pax = sum(f['direct_passengers'] for f in ac)
    transfer_pax = sum(f['transfer_passengers'] for f in ac)
    profit = sum(f['profit'] for f in ac)
    lease  = data['aircraft'].loc[ac_type, 'lease_cost']

    rows.append({
        "Aircraft": f"{ac_type} #{ac_id}",
        "Block Time (h)": round(flight_time_mins / 60, 2),
        "Direct Pax": direct_pax,
        "Transfer Pax": transfer_pax,
        "Total Pax": direct_pax + transfer_pax,
        "Flight Profit (€)": round(profit, 0),
        "Lease Cost (€)": lease,
        "Net Profit (€)": round(profit - lease, 0)
    })

df = pd.DataFrame(rows)
print("=== AIRCRAFT SUMMARY ===\n")
print(df)

# PLOTS AND CSV SAVING
# Visualization 1: Gantt chart of the complete aircraft schedule

# Use IATA codes to save space
iata_map = {
    'London': 'LHR', 'Paris': 'CDG', 'Amsterdam': 'AMS', 'Frankfurt': 'FRA',
    'Madrid': 'MAD', 'Barcelona': 'BCN', 'Munich': 'MUC', 'Rome': 'FCO',
    'Dublin': 'DUB', 'Stockholm': 'ARN', 'Lisbon': 'LIS', 'Berlin': 'TXL',
    'Helsinki': 'HEL', 'Warsaw': 'WAW', 'Edinburgh': 'EDI', 'Bucharest': 'OTP',
    'Heraklion': 'HER', 'Reykjavik': 'KEF', 'Palermo': 'PMO', 'Madeira': 'FNC'
}

# Define start time for visualization 
START_DISPLAY = 4 * 60

# Create Gantt chart
fig, ax = plt.subplots(figsize=(18, max(6, len(final_routes) * 0.8)))

# Color mapping for routes
route_colors = {}
color_idx = 0
colors = plt.cm.tab20.colors + plt.cm.tab20b.colors

# Plot each aircraft's schedule
for idx, ac in enumerate(reversed(final_routes)):
    label = f"{ac[0]['aircraft_type']} #{ac[0]['aircraft_id']}"
    
    # Plot each flight as a horizontal bar
    for f in ac:
        # Assign same color to each route
        route_key = f"{f['origin']}→{f['dest']}"
        if route_key not in route_colors:
            route_colors[route_key] = colors[color_idx % len(colors)]
            color_idx += 1
        
        duration = f['arr_time'] - f['dep_time']
        midpoint = f['dep_time'] + duration / 2
        
        # Draw flight bar
        ax.barh(label, duration, left=f['dep_time'], color=route_colors[route_key], 
                edgecolor='black', height=0.6)
        
        # Add route label (IATA codes)
        org_iata = iata_map.get(f['origin'], f['origin'][:3].upper())
        des_iata = iata_map.get(f['dest'], f['dest'][:3].upper())
        cell_text = f"{org_iata} - {des_iata}"
        ax.text(midpoint, label, cell_text, ha='center', va='center', 
                fontsize=8, color='black', clip_on=True)

# Configure time axis
ax.set_xticks(range(START_DISPLAY, 1441, 120))
ax.set_xticklabels([min_to_hhmm(t) for t in range(START_DISPLAY, 1441, 120)])
ax.set_xlim(START_DISPLAY, TOTAL_TIME)
ax.set_xlabel("Time (HH:MM)")
ax.set_title("Aircraft Schedules")
ax.grid(axis='x', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.show()

# Visualization 2: Net profit per aircraft

# Prepare data 
labels, profits = [], []
for ac in final_routes:
    ac_type = ac[0]['aircraft_type']
    ac_id   = ac[0]['aircraft_id']
    lease   = data['aircraft'].loc[ac_type, 'lease_cost']
    flight_profit = sum(f['profit'] for f in ac)
    labels.append(f"{ac_type} #{ac_id}")
    profits.append(flight_profit - lease)

# Create bar chart
plt.figure(figsize=(12,6))
plt.bar(labels, profits, edgecolor='black')
plt.axhline(0, color='red')
plt.xticks(rotation=45)
plt.ylabel("€")
plt.title("Net Profit per Aircraft")
plt.grid(axis='y', alpha=0.4)
plt.tight_layout()
plt.show()

# Save complete flight schedule as csv file CSV

# Create mapping of inbound flights to their outbound connections
inbound_to_outbound = {}

for ac_idx, ac in enumerate(final_routes):
    for flight_idx, f in enumerate(ac):
        flight_key = (ac[0]['aircraft_type'], ac[0]['aircraft_id'], flight_idx)
        
        # For inbound flights (spoke - hub), find where passengers transferred to
        if f['dest'] == HUB and f['direct_passengers'] > 0:
            outbound_connections = {}           # dictionary to store destinations
            origin_spoke = f['origin']
            arrival_time = f['arr_time']
            
            # Search all outbound flights for transfer passengers from this origin
            for other_ac in final_routes:
                for other_f in other_ac:
                    if other_f['origin'] == HUB and other_f['dest'] != HUB:
                        connection_time = other_f['dep_time'] - arrival_time
                        
                        # Check if connection time is valid
                        if MIN_CONNECTION <= connection_time <= MAX_CONNECTION:
                            # Check if outbound flight carried passengers from this origin
                            if origin_spoke in other_f.get('transfer_from', {}):
                                pax_count = other_f['transfer_from'][origin_spoke]
                                if pax_count > 0:
                                    # Store passengers by destination
                                    if other_f['dest'] not in outbound_connections:
                                        outbound_connections[other_f['dest']] = 0
                                    outbound_connections[other_f['dest']] += pax_count
            
            # Store connections if any exist
            if outbound_connections:
                inbound_to_outbound[flight_key] = list(outbound_connections.items())

# Build detailed schedule with transfer information
rows = []
for ac_idx, ac in enumerate(final_routes):
    if not ac: continue
    
    ac_type, ac_id = ac[0]['aircraft_type'], ac[0]['aircraft_id']
    
    for flight_idx, f in enumerate(ac):
        # Get IATA codes
        org_iata = iata_map.get(f['origin'], f['origin'])
        des_iata = iata_map.get(f['dest'], f['dest'])
        duration = f['arr_time'] - f['dep_time']
        
        transfer_details = ""
        
        # For outbound flights (hub - spoke): show origin of transfer passengers
        if f['origin'] == HUB and f['transfer_passengers'] > 0:
            if f.get('transfer_from'):
                origins_list = [f"{iata_map.get(o, o[:3].upper())}({p})" 
                               for o, p in f['transfer_from'].items()]
                transfer_details = f"From: {', '.join(origins_list)}"
        
        # For inbound flights (spoke - hub): show destination of transfer passengers
        inbound_transfer_pax = 0
        if f['dest'] == HUB and f['direct_passengers'] > 0:
            flight_key = (ac_type, ac_id, flight_idx)
            if flight_key in inbound_to_outbound:
                destinations_list = [f"{iata_map.get(d, d[:3].upper())}({p})" 
                                    for d, p in inbound_to_outbound[flight_key]]
                inbound_transfer_pax = sum(p for _, p in inbound_to_outbound[flight_key])
                if transfer_details: 
                    transfer_details += "; "
                transfer_details += f"To: {', '.join(destinations_list)}"
        
        # Recalculate passenger breakdown for inbound flights
        if f['dest'] == HUB:
            actual_direct_pax = max(0, f['direct_passengers'] - inbound_transfer_pax)
            actual_transfer_pax = inbound_transfer_pax
        else:
            actual_direct_pax = f['direct_passengers']
            actual_transfer_pax = f['transfer_passengers']
        
        # Build row
        rows.append({
            "Aircraft": f"{ac_type} #{ac_id}", 
            "Origin": org_iata, 
            "Destination": des_iata,
            "Departure Time": min_to_hhmm(f['dep_time']), 
            "Arrival Time": min_to_hhmm(f['arr_time']),
            "Flight Duration": min_to_hhmm(duration), 
            "Direct Passengers": actual_direct_pax,
            "Transfer Passengers": actual_transfer_pax, 
            "Total Passengers": f['total_passengers'],
            "Transfer Details": transfer_details,
            "Load Factor": round(f['total_passengers'] / (0.8 * data['aircraft'].loc[ac_type, 'seats']), 2) 
                          if (0.8 * data['aircraft'].loc[ac_type, 'seats']) > 0 else 0,
            "Flight Profit [€]": round(f['profit'], 1),
        })

df_export = pd.DataFrame(rows)

# Export schedule to CSV
df_export.to_csv("aircraft_schedules.csv", index=False)
print("\nCSV exported:\n - aircraft_schedules.csv")