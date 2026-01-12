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

# Parameters
TIME_STEP   = 6                 # minutes
TOTAL_TIME = 24 * 60            # total time in minutes
TIMES      = list(range(0, TOTAL_TIME + 1, TIME_STEP))  # time steps

AIRPORTS = data['distance'].index.tolist()  # list of airports names
HUB = 'Amsterdam'
MIN_BLOCK = 360  # 6 hours minimum total block time

# Initialize final routes list
final_routes = []

# Main loop over aircraft types and individual aircraft

for ac_type in reversed(data['aircraft_types']):                    # loop over aircraft types
    fleet_size = int(data['aircraft']['fleet'][ac_type])
    speed      = data['aircraft'].loc[ac_type, 'speed']
    seats      = data['aircraft'].loc[ac_type, 'seats']
    TAT        = int(data['aircraft'].loc[ac_type, 'TAT'])
    max_range  = data['aircraft'].loc[ac_type, 'range']
    runway_req = data['aircraft'].loc[ac_type, 'runway']
    lease_cost = data['aircraft'].loc[ac_type, 'lease_cost']

    for ac_id in range(fleet_size):                                 # loop over individual aircraft     

        # Initialize lists for storing optimal value function and policy (location, time)
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
            hour = min(int(t / 60), 23)      # current hour (0-23), 24 not included

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

                    if i == j:                            # no self-loops
                        continue
                    if i != HUB and j != HUB:             # no spoke-to-spoke flights
                        continue                        

                    dist = data['distance'].loc[i, j]

                    # Check range and runway constraints
                    if dist > max_range:
                        continue
                    if data['airport_info'].loc[j, 'runway'] < runway_req:
                        continue

                    flight_time = int((dist / speed) * 60 + 30)             # flight time in minutes (+30 min climb+descent)

                    t_ready = ((t + flight_time + TAT + TIME_STEP - 1) // TIME_STEP) * TIME_STEP  # next available time (including TAT), rounded up to next time step

                    if t_ready > TOTAL_TIME:                # cannot arrive after end of day
                        continue

                    # Demand (last 3 hours)
                    total_demand = sum(
                        data['hourly_demand'].loc[(i, j, h)]
                        for h in range(max(0, hour - 2), hour + 1)
                    )

                    max_pax = int(seats * 0.8)                  # max passengers (80% load factor)

                    pax = int(min(total_demand, max_pax))

                    if pax == 0:                                # continue if no demand    
                        continue

                    revenue = pax * data['yield'].loc[i, j] * dist            # total revenue

                    cost = data['operating_cost'][ac_type].loc[i, j]          # operating cost

                    profit = revenue - cost

                    if profit <= 0:                   # skip unprofitable flights
                        continue

                    value = profit + V[(t_ready, j)]    # total value

                    # Update best action if better
                    if value > best_value:
                        best_value = value
                        best_action = ('fly', j, t_ready, pax, profit, flight_time + TAT)

                V[(t, i)] = best_value
                policy[(t, i)] = best_action

        # Reconstruct routes using optimal policy

        t = 0                   # initialize time
        loc = HUB               # initialize location
        route = []              # initialize route list
        total_block = 0         # initialize total block time

        while t < TOTAL_TIME:

            action = policy[(t, loc)]       # get optimal action

            if action[0] == 'stay':           # if stay, just update time
                t = action[2]
            elif action[0] == 'fly':          # if fly, record flight details
                _, dest, t_next, pax, profit, block = action
                
                route.append({
                    'aircraft_type': ac_type,
                    'aircraft_id': ac_id,
                    'dep_time': t,
                    'origin': loc,
                    'dest': dest,
                    'arr_time': t_next,
                    'passengers': pax,
                    'profit': profit,
                    'block_time': block
                })
                total_block += block        # update total block time
                t = t_next                  # update time: arrival time at destination
                loc = dest                  # update location: arrival airport
            else:
                break

        # Only keep aircraft that end at hub and meet minimum block
        if loc == HUB and total_block >= MIN_BLOCK and route:

            # Compute aircraft-level profit (after lease)

            flight_profit = sum(f['profit'] for f in route)
            lease_cost = data['aircraft'].loc[ac_type, 'lease_cost']
            net_profit = flight_profit - lease_cost

            # Only include aircraft with positive net profit
            if net_profit > 0:

                final_routes.append(route)

                # Update demand after having final rooster for each aircraft
                for r in route:

                    hour = int(r['dep_time'] / 60)
                    remaining = r['passengers']             # remaining passengers to remove from demand

                    for h in range(hour, max(-1, hour - 3), -1):    # last 3 hours sequentially starting from current hour
                        if remaining <= 0:
                            break

                        idx = (r['origin'], r['dest'], h)
                        available = data['hourly_demand'].loc[idx]      # available demand at that hour

                        served = min(available, remaining)              # passengers served from that hour
                        data['hourly_demand'].loc[idx] -= served
                        remaining -= served


# Compute total profit (after lease)
total_profit = 0

for ac in final_routes:
    ac_type = ac[0]['aircraft_type']

    flight_profit = sum(f['profit'] for f in ac)
    lease_cost = data['aircraft'].loc[ac_type, 'lease_cost']

    total_profit += flight_profit - lease_cost

print(f"\nTotal network profit (after lease): {total_profit:,.0f}\n")


# # Print all routes
# for ac in final_routes:
#     print(f"\nRoute for aircraft type {ac[0]['aircraft_type']} ID {ac[0]['aircraft_id']}:")
#     for f in ac:
#         print(f)


# Convert minutes to HH:MM format
def min_to_hhmm(t):
    h = t // 60
    m = t % 60
    return f"{int(h):02d}:{int(m):02d}"

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
# Summary table of aircraft performance in csv format
# =========================
rows = []

for ac in final_routes:
    ac_type = ac[0]['aircraft_type']
    ac_id   = ac[0]['aircraft_id']

    # Sum of actual flight durations (excluding TAT)
    flight_time_mins = sum(f['arr_time'] - f['dep_time'] for f in ac)
    pax    = sum(f['passengers'] for f in ac)
    profit = sum(f['profit'] for f in ac)
    lease  = data['aircraft'].loc[ac_type, 'lease_cost']

    rows.append({
        "Aircraft": f"{ac_type} #{ac_id}",
        "Total Block Time (h)": round(flight_time_mins / 60, 2),
        "Passengers": pax,
        "Flight Profit (€)": round(profit, 0),
        "Lease Cost (€)": lease,
        "Net Profit (€)": round(profit - lease, 0)
    })

df = pd.DataFrame(rows)
print("\n=== AIRCRAFT SUMMARY ===\n")
print(df)

# =========================
# Export detailed schedule to CSV
# =========================
iata_map = {
    'London': 'LHR', 'Paris': 'CDG', 'Amsterdam': 'AMS', 'Frankfurt': 'FRA',
    'Madrid': 'MAD', 'Barcelona': 'BCN', 'Munich': 'MUC', 'Rome': 'FCO',
    'Dublin': 'DUB', 'Stockholm': 'ARN', 'Lisbon': 'LIS', 'Berlin': 'BER',
    'Helsinki': 'HEL', 'Warsaw': 'WAW', 'Edinburgh': 'EDI', 'Bucharest': 'OTP',
    'Heraklion': 'HER', 'Reykjavik': 'KEF', 'Palermo': 'PMO', 'Madeira': 'FNC'
}

rows = []

for ac in final_routes:
    if not ac:
        continue

    ac_type = ac[0]['aircraft_type']
    ac_id = ac[0]['aircraft_id']

    # Totals for the aircraft
    total_flight_profit = sum(f['profit'] for f in ac)
    lease_cost = data['aircraft'].loc[ac_type, 'lease_cost']
    net_profit = total_flight_profit - lease_cost
    total_flight_time = sum(f['arr_time'] - f['dep_time'] for f in ac)

    for f in ac:
        org_iata = iata_map.get(f['origin'], f['origin'])
        des_iata = iata_map.get(f['dest'], f['dest'])
        duration = f['arr_time'] - f['dep_time']

        rows.append({
            "Aircraft": f"{ac_type} #{ac_id}",  
            "Origin": org_iata,
            "Destination": des_iata,
            "Departure Time": min_to_hhmm(f['dep_time']),
            "Arrival Time": min_to_hhmm(f['arr_time']),
            "Flight Duration": min_to_hhmm(duration),
            "Passengers": f['passengers'],
            "Load factor": round(
                f['passengers'] / (0.8 * data['aircraft'].loc[ac_type, 'seats']), 2
            ),
            "Flight profit [€]": round(f['profit'], 1),
        })

df_export = pd.DataFrame(rows)
df_export.to_csv("aircraft_schedules.csv", index=False)

print("CSV exported: aircraft_schedules_direct.csv")
