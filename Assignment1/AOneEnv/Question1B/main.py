from gurobipy import *
import numpy as np
import pandas  as pd
import matplotlib.pyplot as plt

# Functions imports
from Question1A.main import main
from Question1A.Read_input import read_excel_pandas
from .ComputeParameters_1B import demand_list, load_airport_params, yields, load_aircraft_params, operating_costs

# Data paths
aircraft_path = r"..\..\Assignment1\Data\AircraftData.xlsx"
aircraft_sheet = ["AircraftTypes"]

"""
In order to run the code, first set the working directory to Assignment1/AOneEnv by running:

cd "C:/Users/pop_r/OneDrive - Delft University of Technology/Documents/SVV/AirlinePlanningAndOptimisation/Assignment1/AOneEnv"
cd "C:/Users/jobru/Documents/TU Delft/MSc AE/Year 1/Courses Q2/APandO/Assignment files/AirlinePlanningAndOptimisation/Assignment1/AOneEnv"

And then run:
python -m uv run -m Question1B.main
"""

# Parameters
FUEL = 1.42                     # Eur/Gallon
LF = 0.75                       # Average load factor assumed 75%           
BT = 10 * 7                     # Assumed 10 hours of block time per day per aircraft


# Airports and routes
airport_data, q = main()                                            # Using results obtained from question 1A for demand
q = demand_list(airport_data, q)                                    # Demand matrix between all airports
airports = airport_data.columns                                     # List of airport names
len_airports = len(airports)                                        # Number of airports
distance, r, ls, g = load_airport_params(airport_data)              # Distance matrix between all airports, max runway length matrix for all routes, 
                                                                    # available weekly landing slots at each airport, hub indicator


# Aircraft
aircraft_data = read_excel_pandas(aircraft_path, aircraft_sheet)    # Load aircraft data
aircraft_data = aircraft_data[0]                                    # type(aircraft_data)=df

aircraft_types, sp, s, TAT, R, RW, C_L, C_X, c_T, c_F = load_aircraft_params(aircraft_data)
len_aircraft_types = len(aircraft_types)                            # Data from aircraft data Excel file

y = yields(distance)                                                # Yield matrix based on distance matrix

C = operating_costs(airport_data, aircraft_data, distance, sp, FUEL, C_X, c_T, c_F)   # Operating cost matrix for all routes and aircraft types

# Variables for dummy constraints:
a = np.empty((len_airports, len_airports, len_aircraft_types), dtype=int)      # Dummy variable for range constraint
b = np.empty((len_airports, len_airports, len_aircraft_types), dtype=int)      # Dummy variable for runway length constraint

for i, porti in enumerate(airports):
    for j, portj in enumerate(airports):
        for k, aircraft in enumerate(aircraft_types):
            if distance[i][j] <= R[k]:                  # Range constraint
                a[i][j][k] = 10000
            else:
                a[i][j][k] = 0
            if r[i][j] >= RW[k]:                        # Runway length constraint
                b[i][j][k] = 10000
            else:
                b[i][j][k] = 0

# Start modelling optimization problem
m = Model('question1B')
x = {}      # Direct flow variables
z = {}      # Flight frequency variables
w = {}      # Transfer flow variables
ac = {}     # Aircraft count variables

# Define decision variables
for i,porti in enumerate(airports):
    for j,portj in enumerate(airports):
        x[porti,portj] = m.addVar(lb=0, vtype=GRB.INTEGER, name=''.join(['Direct flow from ', str(porti), ' to ', str(portj)]))
        w[porti,portj] = m.addVar(lb=0,vtype=GRB.INTEGER, name=''.join(['Transfer flow from ',str(porti), ' to ', str(portj)]))
        for k,aircraft in enumerate(aircraft_types):
            z[porti,portj,aircraft] = m.addVar(lb=0, vtype=GRB.INTEGER, name=''.join(['Flights from ',str(porti), ' to ', str(portj), ' by ', str(aircraft)]))

for k,aircraft in enumerate(aircraft_types):
    ac[aircraft] = m.addVar(lb=0, vtype=GRB.INTEGER, name=''.join(['Number of ', str(aircraft)]))

m.update()

# Set Objective Function
m.setObjective(quicksum(quicksum(y[i][j]*distance[i][j]*(x[porti,portj]+w[porti,portj]) for i,porti in enumerate(airports)) for j,portj in enumerate(airports))
              - quicksum(quicksum(quicksum(C[i][j][k]*z[porti,portj,aircraft] for i, porti in enumerate(airports)) for j, portj in enumerate(airports)) for k, aircraft in enumerate(aircraft_types))
              - quicksum(C_L[k]*ac[aircraft] for k, aircraft in enumerate(aircraft_types)), GRB.MAXIMIZE)  # The objective is to maximize revenue

# Add Constraints - numbering follows the report
for i,porti in enumerate(airports):
    for j,portj in enumerate(airports):
        m.addConstr(x[porti,portj] + w[porti,portj]<= q[i][j])          # Constraint 1

        m.addConstr(w[porti, portj] <= q[i][j] * g[i] * g[j])           # Constraint 2

        m.addConstr(x[porti,portj] <= q[i][j]* (2- g[i] -g[j]))         # Constraint 3

        m.addConstr(x[porti,portj] + quicksum(w[porti,portm] * (1-g[j]) for m,portm in enumerate(airports)) + quicksum(w[portm,portj] * (1-g[i]) for m,portm in enumerate(airports)) 
                    <= quicksum(z[porti,portj,aircraft] * s[k] * LF for k,aircraft in enumerate(aircraft_types)))  # Constraint 4

        for k,aircraft in enumerate(aircraft_types):
            m.addConstr(z[porti,portj,aircraft] <= a[i][j][k])  # Constraint 7
            m.addConstr(z[porti,portj,aircraft] <= b[i][j][k])  # Constraint 8

for k,aircraft in enumerate(aircraft_types):
    for i,porti in enumerate(airports):
        m.addConstr(quicksum(z[porti,portj,aircraft] for j,portj in enumerate(airports)) ==  quicksum(z[portj, porti,aircraft] for j,portj in enumerate(airports))) # Constraint 5

for j,portj in enumerate(airports):
    m.addConstr(quicksum(quicksum(z[porti,portj,aircraft] for i,porti in enumerate(airports)) for k,aircraft in enumerate(aircraft_types)) <= ls[j])  # Constraint 9

for k,aircraft in enumerate(aircraft_types):
    m.addConstr(quicksum(quicksum((distance[i][j]/sp[k]+TAT[k]*(1 + 0.5 * (1 - g[j])))*z[porti,portj,aircraft] for i,porti in enumerate(airports)) for j,portj in enumerate(airports)) <= BT*ac[aircraft]) #Constraint 6



m.update()
# m.write('test.lp')
# Set time constraint for optimization (5 minutes)
m.setParam('TimeLimit', 5 * 60)
m.optimize()
# m.write("testout.sol")
status = m.status

if status == GRB.Status.UNBOUNDED:
    print('The model cannot be solved because it is unbounded')

elif status == GRB.Status.OPTIMAL or True:
    f_objective = m.objVal
    print('\n***** RESULTS ******')
    print('\nObjective Function Value: \t %g' % f_objective)

elif status != GRB.Status.INF_OR_UNBD and status != GRB.Status.INFEASIBLE:
    print('Optimization was stopped with status %d' % status)


# # Print out Solutions
# print()
# print("Frequencies:----------------------------------")
# print()

# # Print aircraft counts, flows and frequencies
# for k, aircraft in enumerate(aircraft_types):
#     print(f'Total number of aircraft of type {aircraft}: {ac[aircraft].X:.0f}')

# for i,porti in enumerate(airports):
#     sum_transfer = 0
#     for j,portj in enumerate(airports):
#         if x[porti, portj].X > 0 :
#             print(f"Direct flow from {porti} to {portj}: {x[porti, portj].X:.0f} passengers")
#         if w[porti, portj].X > 0 :
#             print(f"Transfer flow from {porti} to {portj}: {w[porti, portj].X:.0f} passengers")
#         sum_transfer += w[porti, portj]
#     print (f"Tranfer flow from {porti} to all destinations: {sum_transfer} passengers")

# for i, porti in enumerate(airports):
#     for j, portj in enumerate(airports):
#         for aircraft in aircraft_types:
#             if z[porti, portj, aircraft].X > 0: 
#                 print(f"{porti} -> {portj} : {z[porti, portj, aircraft].X:.0f} flights using {aircraft}")

# for k, aircraft in enumerate(aircraft_types):

#     if ac[aircraft].X > 0:

#         usage_per_actype = 0

#         for i, porti in enumerate(airports):
#             for j, portj in enumerate(airports):
#                 usage_per_actype += (distance[i][j]/sp[k]+TAT[k]*(1 + 0.5 * (1 - g[j])))*z[porti,portj,aircraft].X

#         usage = usage_per_actype / (BT * ac[aircraft].X) * 100

#         print(f'Aircraft type {aircraft} is used at {usage:.2f}% of its available block time.')


#         # Create bar chart of direct flows and transfer flows by airport
#         direct_flows = {}
#         transfer_flows = {}

#         for i, porti in enumerate(airports):
#             direct_flows[porti] = 0
#             transfer_flows[porti] = 0
#             for j, portj in enumerate(airports):
#                 direct_flows[porti] += x[porti, portj].X
#                 transfer_flows[porti] += w[porti, portj].X

#         # Filter: only include airports with direct flows > 0 and exclude Amsterdam
#         filtered_flows = {airport: direct_flows[airport] for airport in direct_flows 
#              if direct_flows[airport] > 0 and airport != 'Amsterdam'}
        
#         airport_names = list(filtered_flows.keys())
#         direct_values = [filtered_flows[airport] for airport in airport_names]
#         transfer_values = [transfer_flows[airport] for airport in airport_names]

#         x_pos = np.arange(len(airport_names))
#         width = 0.35

#         plt.figure(figsize=(12, 6))
#         bars1 = plt.bar(x_pos - width/2, direct_values, width, label='Direct Flows', alpha=0.8)
#         bars2 = plt.bar(x_pos + width/2, transfer_values, width, label='Transfer Flows', alpha=0.8)

#         plt.xlabel('Departing Airport')
#         plt.ylabel('Passengers')
#         plt.title('Direct and Transfer Flows by Airport')
#         plt.xticks(x_pos, airport_names, rotation=45, ha='right')
#         plt.legend()

#         # Add value labels on bars with default color
#         for bar in bars1:
#             yval = bar.get_height()
#             plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), va='bottom', ha='center')
#         for bar in bars2:
#             yval = bar.get_height()
#             plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), va='bottom', ha='center')

#         plt.tight_layout()
#         plt.show()