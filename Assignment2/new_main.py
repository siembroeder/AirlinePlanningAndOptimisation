# Imports
from new_compute_parameters import build_problem_data
from pathlib import Path

# Define paths of input data
BASE_DIR = Path(__file__).resolve().parent
airports_path = BASE_DIR / "Data" / "DemandGroup7.xlsx"
aircraft_path = BASE_DIR / "Data" / "FleetType.xlsx"
hours_path    = BASE_DIR / "Data" / "HourCoefficients.xlsx"

# Build data 
data = build_problem_data(airports_path, hours_path, aircraft_path)

# Examples for usage :
# print(f"distance London-Helsinki: {data['distance'].loc['London', 'Helsinki']}")
# print(f"Operating cost type 2 London-Helsinki: {data['operating_cost']['SmallJet'].loc['London', 'Helsinki']}")
# print(f"Hourly demand Madrid-Helsinki: {data['hourly_demand'].loc[('Madrid', 'Helsinki'), 'demand']}")

# Print total obtained data
# print(f"Total data: {data['hourly_demand'].loc[('London', 'Helsinki')]}")

# Parameters

TIME_STEP   = 6                 # minutes
TOTAL_TIME = 24 * 60            # total time in minutes
TIMES      = list(range(0, TOTAL_TIME + 1, TIME_STEP))  # time steps


# We define block states to handle block time requirement: the model takes into account whether
#  the aircraft has met the minimum block time or not at each decision point. 

BLOCK_REQ = 360                 # 6 hours in minutes
BLOCK_STATES = [0, BLOCK_REQ]   # 0 = min block not reached, 360 = min block reached

AIRPORTS = data['distance'].index.tolist()
HUB = 'Amsterdam'

final_routes = []

# Main loop over aircraft types and individual aircraft

for ac_type in reversed(data['aircraft_types']):                 # loop over aircraft types in reverse order from largest to smallest

    fleet_size = int(data['aircraft']['fleet'][ac_type])
    speed      = data['aircraft'].loc[ac_type, 'speed']
    seats      = data['aircraft'].loc[ac_type, 'seats']
    TAT        = int(data['aircraft'].loc[ac_type, 'TAT'])
    max_range  = data['aircraft'].loc[ac_type, 'range']
    runway_req = data['aircraft'].loc[ac_type, 'runway']

    for ac_id in range(fleet_size):

        # Store optimal value function and policy
        V = {}
        policy = {}

        # Enforce that all aircraft must end at the hub and have met block time requirement, if not very large loss
        for a in AIRPORTS:
            for b in BLOCK_STATES:
                if a == HUB and b == BLOCK_REQ:
                    V[(TOTAL_TIME, a, b)] = 0
                else:
                    V[(TOTAL_TIME, a, b)] = -1e9

        # Backward dynamic programming loop

        for t in reversed(TIMES[:-1]):          # for each time step

            hour = int(t / 60)

            for i in AIRPORTS:                  # for each airport
                for b in BLOCK_STATES:          # for each block time state

                    # All aircraft must start at hub, if not very large loss
                    if t == 0 and i != HUB:
                        V[(t, i, b)] = -1e9
                        policy[(t, i, b)] = ('invalid',)
                        continue

                    # Option 1: Stay idle
                    best_value = V[(t + TIME_STEP, i, b)]
                    best_action = ('stay', i, b)

                    # Option 2: Fly
                    for j in AIRPORTS:

                        if i == j:                              # cannot fly to same airport
                            continue

                        dist = data['distance'].loc[i, j]

                        # Check aircraft range and runway requirements
                        if dist > max_range:
                            continue

                        if data['airport_info'].loc[j, 'runway'] < runway_req:
                            continue
                        
                        # Calculate flight time and ready time(including TAT)
                        flight_time = int((dist / speed) * 60 + 30)             # flight time in minutes (+30 min for climb/descent)

                        # Round up to nearest TIME_STEP
                        t_ready = ((t + flight_time + TAT + TIME_STEP - 1) // TIME_STEP) * TIME_STEP

                        # Skip if past end of day
                        if t_ready > TOTAL_TIME:
                            continue

                        # Demand (include last 3 hours)
                        total_demand = 0
                        for h in range(max(0, hour - 2), hour + 1):
                            total_demand += data['hourly_demand'].loc[(i, j, h)]

                        max_pax = int(seats * 0.8)                  # average 80% load factor
                        pax = min(total_demand, max_pax)

                        if pax == 0:                # if no passengers, skip
                            continue
                        
                        # Calculate profit
                        revenue = pax * data['yield'].loc[i, j] * dist
                        cost = data['operating_cost'][ac_type].loc[i, j]
                        profit = revenue - cost

                        if profit <= 0:             # if no profit, skip
                            continue

                        # binary block-time
                        if b + flight_time >= BLOCK_REQ:
                            new_block = BLOCK_REQ
                        else:
                            new_block = 0

                        value = profit + V[(t_ready, j, new_block)]

                        # Update best action and value if better
                        if value > best_value:
                            best_value = value
                            best_action = ('fly', j, t_ready, new_block, pax, profit)

                    V[(t, i, b)] = best_value
                    policy[(t, i, b)] = best_action

        # Reconstruct the routes for each aircraft

        t = 0
        loc = HUB
        b = 0
        route = []

        while t < TOTAL_TIME:
            action = policy[(t, loc, b)]

            if action[0] == 'stay':
                t += TIME_STEP

            elif action[0] == 'fly':
                _, dest, t_next, b_next, pax, profit = action

                route.append({
                    'aircraft_type': ac_type,
                    'aircraft_id': ac_id,
                    'dep_time': t,
                    'origin': loc,
                    'dest': dest,
                    'arr_time': t_next,
                    'passengers': pax,
                    'profit': profit
                })

                t = t_next
                loc = dest
                b = b_next

            else:
                break

        # Discard unused aircraft

        if b < BLOCK_REQ:
            continue


        # Update transported passengers in data['hourly_demand']

        for r in route:
            hour = int(r['dep_time'] / 60)
            remaining = r['passengers']

            for h in range(hour, max(-1, hour - 3), -1):        # sequentially fill demand from current hour to past 2 hours
                if remaining <= 0:
                    break

                idx = (r['origin'], r['dest'], h)
                available = data['hourly_demand'].loc[idx]

                served = min(available, remaining)
                data['hourly_demand'].loc[idx] -= served
                remaining -= served

        final_routes.append(route)

# Results

total_profit = sum(
    f['profit']
    for ac in final_routes
    for f in ac
)

print(f"\nTotal network profit: {total_profit:,.0f}\n")

for ac in final_routes:
    for f in ac:
        print(f)


# Still need to add lease cost to each aircrfat in the final profit calculation