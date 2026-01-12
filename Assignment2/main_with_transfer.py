
# Imports
from compute_parameters import build_problem_data
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

# Define paths of input data
BASE_DIR = Path(__file__).resolve().parent
airports_path = BASE_DIR / "Data" / "DemandGroup7.xlsx"
aircraft_path = BASE_DIR / "Data" / "FleetType.xlsx"
hours_path    = BASE_DIR / "Data" / "HourCoefficients.xlsx"

# Build data 
data = build_problem_data(airports_path, hours_path, aircraft_path)

# Store original demand for verification
original_demand = {}
for orig in data['distance'].index:
    for dest in data['distance'].index:
        if orig != dest:
            total = sum(data['hourly_demand'].loc[(orig, dest, h)] for h in range(24))
            original_demand[(orig, dest)] = total

# Parameters
TIME_STEP   = 6                 # minutes
TOTAL_TIME = 24 * 60            # total time in minutes
TIMES      = list(range(0, TOTAL_TIME + 1, TIME_STEP))  # time steps

AIRPORTS = data['distance'].index.tolist()  # list of airports names
HUB = 'Amsterdam'
MIN_BLOCK = 360  # 6 hours minimum total block time

# Transfer parameters
MIN_CONNECTION = 0   # minimum connection time (minutes)
MAX_CONNECTION = 180# maximum connection time (4 hours)

# Initialize transfer demand tracking (spoke -> spoke demand)
# This tracks remaining transfer demand between spoke airports
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

# Track arriving passengers at hub for transfer opportunities
# Structure: {(origin_spoke, arrival_time): passengers_available}
hub_arrivals = {}

# Main loop over aircraft types and individual aircraft
for ac_type in reversed(data['aircraft_types']):
    fleet_size = int(data['aircraft']['fleet'][ac_type])
    speed      = data['aircraft'].loc[ac_type, 'speed']
    seats      = data['aircraft'].loc[ac_type, 'seats']
    TAT        = int(data['aircraft'].loc[ac_type, 'TAT'])
    max_range  = data['aircraft'].loc[ac_type, 'range']
    runway_req = data['aircraft'].loc[ac_type, 'runway']
    lease_cost = data['aircraft'].loc[ac_type, 'lease_cost']

    for ac_id in range(fleet_size):
        # Initialize lists for storing optimal value function and policy
        V = {}
        policy = {}

        # Terminal condition: only allow ending at hub
        for a in AIRPORTS:
            if a == HUB:
                V[(TOTAL_TIME, a)] = 0
            else:
                V[(TOTAL_TIME, a)] = -1e9

        # Backward DP
        for t in reversed(TIMES[:-1]):
            hour = min(int(t / 60), 23)

            for i in AIRPORTS:
                # Initial condition: only allow starting at hub
                if t == 0 and i != HUB:
                    V[(t, i)] = -1e9
                    policy[(t, i)] = ('invalid',)
                    continue

                # Option 1: Stay idle
                best_value = V[(t + TIME_STEP, i)]
                best_action = ('stay', i, t + TIME_STEP)

                # Option 2: Fly to another airport
                for j in AIRPORTS:
                    if i == j:
                        continue
                    if i != HUB and j != HUB:
                        continue

                    dist = data['distance'].loc[i, j]

                    # Check range and runway constraints
                    if dist > max_range:
                        continue
                    if data['airport_info'].loc[j, 'runway'] < runway_req:
                        continue

                    flight_time = int((dist / speed) * 60 + 30)
                    t_ready = ((t + flight_time + TAT + TIME_STEP - 1) // TIME_STEP) * TIME_STEP

                    if t_ready > TOTAL_TIME:
                        continue

                    # === DIRECT DEMAND (existing logic) ===
                    total_direct_demand = sum(
                        data['hourly_demand'].loc[(i, j, h)]
                        for h in range(max(0, hour - 2), hour + 1)
                    )

                    required_pax = int(seats * 0.8)  # EXACTLY 80% must be filled
                    direct_pax = int(min(total_direct_demand, required_pax))
                    
                    # === TRANSFER DEMAND (NEW) ===
                    transfer_pax = 0
                    transfer_revenue = 0
                    transfer_breakdown = {}  # Track transfer passengers by origin
                    
                    # Calculate remaining capacity needed to reach 80%
                    remaining_needed = required_pax - direct_pax
                    
                    # If departing FROM hub, check for transfer passengers to fill remaining seats
                    if i == HUB and j != HUB and remaining_needed > 0:
                        # Look for passengers arriving at hub who want to go to j
                        for (origin_spoke, arr_time), available_pax in hub_arrivals.items():
                            if available_pax <= 0 or remaining_needed <= 0:
                                continue
                            
                            connection_time = t - arr_time
                            
                            # Check connection time window
                            if MIN_CONNECTION <= connection_time <= MAX_CONNECTION:
                                # Check if there's demand for this transfer route
                                if (origin_spoke, j) in transfer_demand:
                                    available_transfer_demand = transfer_demand[(origin_spoke, j)]
                                    
                                    if available_transfer_demand > 0:
                                        # How many transfer passengers can we take?
                                        can_transfer = min(
                                            available_pax,
                                            remaining_needed,
                                            available_transfer_demand
                                        )
                                        
                                        if can_transfer > 0:
                                            transfer_pax += can_transfer
                                            remaining_needed -= can_transfer
                                            transfer_breakdown[origin_spoke] = can_transfer
                                            
                                            # Calculate transfer revenue (origin -> hub -> dest)
                                            leg1_dist = data['distance'].loc[origin_spoke, HUB]
                                            leg2_dist = data['distance'].loc[HUB, j]
                                            total_transfer_dist = leg1_dist + leg2_dist
                                            
                                            transfer_revenue += can_transfer * data['yield'].loc[origin_spoke, j] * total_transfer_dist
                                            
                                            if remaining_needed <= 0:
                                                break

                    total_pax = direct_pax + transfer_pax
                    
                    # CRITICAL: Flight can only operate if EXACTLY 80% seats are filled
                    if total_pax != required_pax:
                        continue

                    # Calculate revenue
                    direct_revenue = direct_pax * data['yield'].loc[i, j] * dist
                    total_revenue = direct_revenue + transfer_revenue
                    
                    cost = data['operating_cost'][ac_type].loc[i, j]
                    profit = total_revenue - cost

                    if profit <= 0:
                        continue

                    value = profit + V[(t_ready, j)]

                    # Update best action if better
                    if value > best_value:
                        best_value = value
                        best_action = ('fly', j, t_ready, direct_pax, transfer_pax, profit, flight_time + TAT, transfer_breakdown.copy())

                V[(t, i)] = best_value
                policy[(t, i)] = best_action

        # Reconstruct routes using optimal policy
        t = 0
        loc = HUB
        route = []
        total_block = 0

        while t < TOTAL_TIME:
            action = policy[(t, loc)]

            if action[0] == 'stay':
                t = action[2]
            elif action[0] == 'fly':
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
                    'transfer_from': transfer_breakdown  # Dict of {origin_spoke: pax_count}
                })
                total_block += block
                t = t_next
                loc = dest
            else:
                break

        # Only keep aircraft that end at hub and meet minimum block
        if loc == HUB and total_block >= MIN_BLOCK and route:
            flight_profit = sum(f['profit'] for f in route)
            lease_cost = data['aircraft'].loc[ac_type, 'lease_cost']
            net_profit = flight_profit - lease_cost
            
            # Verify all flights have exactly 80% load factor
            all_flights_at_80 = all(
                f['total_passengers'] == int(data['aircraft'].loc[ac_type, 'seats'] * 0.8)
                for f in route
            )
            
            # Only include aircraft with positive net profit AND all flights at 80% LF
            if net_profit > 0 and all_flights_at_80:
                final_routes.append(route)

                # Update demand and transfer tracking
                for r in route:
                    hour = int(r['dep_time'] / 60)
                    
                    # Update direct demand (spoke->hub or hub->spoke)
                    remaining_direct = r['direct_passengers']
                    for h in range(hour, max(-1, hour - 3), -1):
                        if remaining_direct <= 0:
                            break
                        idx = (r['origin'], r['dest'], h)
                        available = data['hourly_demand'].loc[idx]
                        served = min(available, remaining_direct)
                        data['hourly_demand'].loc[idx] -= served
                        remaining_direct -= served
                    
                    # Track hub arrivals for future transfers
                    if r['dest'] == HUB:
                        key = (r['origin'], r['arr_time'])
                        hub_arrivals[key] = r['direct_passengers']
                    
                    # Update transfer demand and hub arrivals when transfers are used
                    if r['origin'] == HUB and r['transfer_passengers'] > 0:
                        # Process each origin in the transfer breakdown
                        for origin_spoke, transfer_count in r.get('transfer_from', {}).items():
                            # Deduct from transfer demand
                            if (origin_spoke, r['dest']) in transfer_demand:
                                transfer_demand[(origin_spoke, r['dest'])] -= transfer_count
                            
                            # Deduct from hub arrivals
                            for (arr_origin, arr_time), available_pax in list(hub_arrivals.items()):
                                if arr_origin == origin_spoke:
                                    connection_time = r['dep_time'] - arr_time
                                    if MIN_CONNECTION <= connection_time <= MAX_CONNECTION:
                                        deduct = min(available_pax, transfer_count)
                                        hub_arrivals[(arr_origin, arr_time)] -= deduct
                                        transfer_count -= deduct
                                        if transfer_count <= 0:
                                            break

# Compute total profit and verify load factors
total_profit = 0
total_flights = 0
total_pax = 0
total_capacity = 0

for ac in final_routes:
    ac_type = ac[0]['aircraft_type']
    seats = data['aircraft'].loc[ac_type, 'seats']
    flight_profit = sum(f['profit'] for f in ac)
    lease_cost = data['aircraft'].loc[ac_type, 'lease_cost']
    total_profit += flight_profit - lease_cost
    
    # Aggregate for network-wide load factor verification
    for f in ac:
        total_flights += 1
        total_pax += f['total_passengers']
        total_capacity += seats * 0.8

# Verify network-wide load factor (should be exactly 100% since each flight is at 80%)
network_load_factor = total_pax / total_capacity if total_capacity > 0 else 0

print(f"\nTotal network profit (after lease): {total_profit:,.0f}")
print(f"Network-wide load factor verification: {network_load_factor:.2%} (should be 100%)")
print(f"Total flights: {total_flights}")
print(f"Total passengers: {total_pax:,}\n")

print(f"\nTotal network profit (after lease): {total_profit:,.0f}\n")

# Convert minutes to HH:MM format
def min_to_hhmm(t):
    h = t // 60
    m = t % 60
    return f"{int(h):02d}:{int(m):02d}"

# Summary statistics
total_direct = sum(f['direct_passengers'] for ac in final_routes for f in ac)
total_transfer = sum(f['transfer_passengers'] for ac in final_routes for f in ac)
print(f"Total direct passengers: {total_direct:,}")
print(f"Total transfer passengers: {total_transfer:,}")
print(f"Transfer percentage: {100*total_transfer/(total_direct+total_transfer):.1f}%\n")

# Summary table
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

# =========================
# Plot 1: Gantt chart visualization of aircraft schedules
# =========================

# Dictionary mapping for IATA codes
iata_map = {
    'London': 'LHR', 'Paris': 'CDG', 'Amsterdam': 'AMS', 'Frankfurt': 'FRA',
    'Madrid': 'MAD', 'Barcelona': 'BCN', 'Munich': 'MUC', 'Rome': 'FCO',
    'Dublin': 'DUB', 'Stockholm': 'ARN', 'Lisbon': 'LIS', 'Berlin': 'BER',
    'Helsinki': 'HEL', 'Warsaw': 'WAW', 'Edinburgh': 'EDI', 'Bucharest': 'OTP',
    'Heraklion': 'HER', 'Reykjavik': 'KEF', 'Palermo': 'PMO', 'Madeira': 'FNC'
}

# Define start time in minutes (04:00 = 4 * 60)
START_DISPLAY = 4 * 60

fig, ax = plt.subplots(figsize=(18, max(6, len(final_routes) * 0.8)))

route_colors = {}
color_idx = 0
colors = plt.cm.tab20.colors + plt.cm.tab20b.colors

for idx, ac in enumerate(reversed(final_routes)):
    label = f"{ac[0]['aircraft_type']} #{ac[0]['aircraft_id']}"

    for f in ac:
        # Generate route key for coloring
        route_key = f"{f['origin']}→{f['dest']}"
        if route_key not in route_colors:
            route_colors[route_key] = colors[color_idx % len(colors)]
            color_idx += 1

        duration = f['arr_time'] - f['dep_time']
        midpoint = f['dep_time'] + duration / 2
        
        # Draw the bar
        ax.barh(
            label,
            duration,
            left=f['dep_time'],
            color=route_colors[route_key],
            edgecolor='black',
            height=0.6
        )

        # Get IATA codes and format as single line ORG - DES
        org_iata = iata_map.get(f['origin'], f['origin'][:3].upper())
        des_iata = iata_map.get(f['dest'], f['dest'][:3].upper())
        cell_text = f"{org_iata} - {des_iata}"

        # Place route text centered inside the bar 
        ax.text(
            midpoint,
            label,
            cell_text,
            ha='center',
            va='center',
            fontsize=8,
            fontweight='normal',
            color='black',
            clip_on=True
        )

# Time axis - Adjusted to start from 04:00
ax.set_xticks(range(START_DISPLAY, 1441, 120))
ax.set_xticklabels([min_to_hhmm(t) for t in range(START_DISPLAY, 1441, 120)])
ax.set_xlim(START_DISPLAY, TOTAL_TIME) # Graph starts at 04:00

ax.set_xlabel("Time (HH:MM)")
ax.set_title("Aircraft Schedules")
ax.grid(axis='x', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.show()

# =========================
# Plot 2: Bar chart of net profit per aircraft
# =========================

labels, profits = [], []

for ac in final_routes:
    ac_type = ac[0]['aircraft_type']
    ac_id   = ac[0]['aircraft_id']
    lease   = data['aircraft'].loc[ac_type, 'lease_cost']
    flight_profit = sum(f['profit'] for f in ac)

    labels.append(f"{ac_type} #{ac_id}")
    profits.append(flight_profit - lease)

plt.figure(figsize=(12,6))
plt.bar(labels, profits, edgecolor='black')
plt.axhline(0, color='red')
plt.xticks(rotation=45)
plt.ylabel("€")
plt.title("Net Profit per Aircraft")
plt.grid(axis='y', alpha=0.4)
plt.tight_layout()
plt.show()

# =========================
# NEW: Build transfer connection mapping CORRECTLY
# =========================

# Create mapping of inbound flights to their potential outbound connections
inbound_to_outbound = {}  # {(ac_type, ac_id, flight_idx): [(dest_spoke, pax_count), ...]}

for ac_idx, ac in enumerate(final_routes):
    for flight_idx, f in enumerate(ac):
        flight_key = (ac[0]['aircraft_type'], ac[0]['aircraft_id'], flight_idx)
        
        # For inbound flights (spoke -> hub), find where those passengers actually went
        if f['dest'] == HUB and f['direct_passengers'] > 0:
            outbound_connections = {}  # Use dict to aggregate by destination
            origin_spoke = f['origin']
            arrival_time = f['arr_time']
            
            # Find all outbound flights that took passengers from this origin
            for other_ac in final_routes:
                for other_f in other_ac:
                    if other_f['origin'] == HUB and other_f['dest'] != HUB:
                        # Check connection time window
                        connection_time = other_f['dep_time'] - arrival_time
                        if MIN_CONNECTION <= connection_time <= MAX_CONNECTION:
                            # Check if this outbound flight has transfer passengers from our origin
                            if origin_spoke in other_f.get('transfer_from', {}):
                                pax_count = other_f['transfer_from'][origin_spoke]
                                if pax_count > 0:
                                    # Aggregate by destination
                                    if other_f['dest'] not in outbound_connections:
                                        outbound_connections[other_f['dest']] = 0
                                    outbound_connections[other_f['dest']] += pax_count
            
            if outbound_connections:
                # Convert to list of tuples
                inbound_to_outbound[flight_key] = list(outbound_connections.items())

# =========================
# Export detailed schedule to CSV with CORRECTED transfer information
# =========================

rows = []

for ac_idx, ac in enumerate(final_routes):
    if not ac:
        continue

    ac_type = ac[0]['aircraft_type']
    ac_id = ac[0]['aircraft_id']

    for flight_idx, f in enumerate(ac):
        org_iata = iata_map.get(f['origin'], f['origin'])
        des_iata = iata_map.get(f['dest'], f['dest'])
        duration = f['arr_time'] - f['dep_time']
        
        transfer_details = ""
        
        # OUTBOUND from hub: show where transfer passengers came FROM
        if f['origin'] == HUB and f['transfer_passengers'] > 0:
            if f.get('transfer_from'):
                origins_list = []
                for origin_spoke, pax_count in f['transfer_from'].items():
                    origin_iata = iata_map.get(origin_spoke, origin_spoke[:3].upper())
                    origins_list.append(f"{origin_iata}({pax_count})")
                transfer_details = f"From: {', '.join(origins_list)}"
        
        # INBOUND to hub: show where passengers are going TO (and count them as transfer passengers)
        inbound_transfer_pax = 0
        if f['dest'] == HUB and f['direct_passengers'] > 0:
            flight_key = (ac_type, ac_id, flight_idx)
            if flight_key in inbound_to_outbound:
                destinations_list = []
                for dest_spoke, pax_count in inbound_to_outbound[flight_key]:
                    dest_iata = iata_map.get(dest_spoke, dest_spoke[:3].upper())
                    destinations_list.append(f"{dest_iata}({pax_count})")
                    inbound_transfer_pax += pax_count
                if destinations_list:
                    if transfer_details:
                        transfer_details += "; "
                    transfer_details += f"To: {', '.join(destinations_list)}"

        # For inbound flights to hub: passengers who will transfer are "transfer passengers"
        # For outbound flights from hub: passengers are already correctly labeled
        if f['dest'] == HUB and inbound_transfer_pax > 0:
            # These passengers will connect onwards
            actual_direct_pax = max(0, f['direct_passengers'] - inbound_transfer_pax)
            actual_transfer_pax = inbound_transfer_pax
        else:
            # Use the values from the DP algorithm
            actual_direct_pax = f['direct_passengers']
            actual_transfer_pax = f['transfer_passengers']

        row = {
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
            "Load Factor": round(
                f['total_passengers'] / (0.8 * data['aircraft'].loc[ac_type, 'seats']), 2
            ) if (0.8 * data['aircraft'].loc[ac_type, 'seats']) > 0 else 0,
            "Flight Profit [€]": round(f['profit'], 1),
        }
        
        rows.append(row)

df_export = pd.DataFrame(rows)

# =========================
# NEW: Add demand matrix verification - CREATE SEPARATE CSV
# =========================

# Calculate served demand
served_demand = {}
for orig in AIRPORTS:
    for dest in AIRPORTS:
        if orig != dest:
            served_demand[(orig, dest)] = 0

# Add up all passengers served on each route
for ac in final_routes:
    for f in ac:
        # Direct passengers (hub <-> spoke only)
        if f['origin'] == HUB or f['dest'] == HUB:
            key = (f['origin'], f['dest'])
            # Count only true direct passengers (not those who will transfer)
            if f['dest'] == HUB:
                # Inbound: subtract those who will transfer
                flight_key = (f['aircraft_type'], f['aircraft_id'], ac.index(f))
                transfer_out = sum(pax for _, pax in inbound_to_outbound.get(flight_key, []))
                direct_only = f['direct_passengers'] - transfer_out
                served_demand[key] += direct_only
            else:
                # Outbound: count only direct passengers
                served_demand[key] += f['direct_passengers']
        
        # Transfer passengers (spoke -> hub -> spoke)
        if f['origin'] == HUB and f['transfer_passengers'] > 0:
            for origin_spoke, pax_count in f.get('transfer_from', {}).items():
                key = (origin_spoke, f['dest'])
                served_demand[key] += pax_count

# Create 20x20 demand matrix
demand_matrix_data = []
airport_codes = [iata_map.get(a, a) for a in AIRPORTS]

# Header row
header_row = ['Origin/Dest'] + airport_codes
demand_matrix_data.append(header_row)

for orig in AIRPORTS:
    row_data = [iata_map.get(orig, orig)]
    for dest in AIRPORTS:
        if orig == dest:
            row_data.append('—')
        else:
            original = int(original_demand.get((orig, dest), 0))
            served = int(served_demand.get((orig, dest), 0))
            row_data.append(f"{served}/{original}")
    demand_matrix_data.append(row_data)

# Create DataFrame for demand matrix
df_demand_matrix = pd.DataFrame(demand_matrix_data[1:], columns=demand_matrix_data[0])

# Export to separate CSVs
df_export.to_csv("aircraft_schedules.csv", index=False)
df_demand_matrix.to_csv("demand_matrix.csv", index=False)

print("\nCSVs exported:")
print("  - aircraft_schedules_transfer.csv (detailed flight schedules)")
print("  - demand_matrix.csv (20x20 served/original demand matrix)")
print("\nDemand matrix format: Served/Original passengers for each O-D pair")