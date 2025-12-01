
from gurobipy import *
import numpy as np
import pandas  as pd

# Custom imports
from Question1A.main import main
from Question1A.Read_input import read_excel_pandas
from .ComputeParameters_1B import demand_list, load_airport_params, yields, load_aircraft_params, operating_costs

# aircraft_path = r"C:\Users\jobru\Documents\TU Delft\MSc AE\Year 1\Courses Q2\APandO\Assignment files\AirlinePlanningAndOptimisation\Assignment1\Data\AircraftData.xlsx"
# aircraft_path = r"C:\Users\pop_r\OneDrive - Delft University of Technology\Desktop\AirlinePlanning\Assignment1\Data\AircraftData.xlsx"
aircraft_path = r"..\..\Assignment1\Data\AircraftData.xlsx"
aircraft_sheet = ["AircraftTypes"]

# Radu:
# cd "C:\Users\pop_r\OneDrive - Delft University of Technology\Documents\SVV\AirlinePlanningAndOptimisation\Assignment1\AOneEnv" 
#  python -m uv run -m Question1B.main


# Job:
# cd "C:\Users\jobru\Documents\TU Delft\MSc AE\Year 1\Courses Q2\APandO\Assignment files\AirlinePlanningAndOptimisation\Assignment1\AOneEnv"
# python -m uv run -m Question1B.main

# Constants:
FUEL = 1.42                                # Eur/Gallon
LF = 0.75                                  # Average load factor assumed 75%           
BT = 10 * 7                                # Assumed 10 hours of block time per day per aircraft

# Data

# Airports and routes
airport_data, q = main()                                            # Using results obtained from question 1A for demand
q = demand_list(airport_data, q)                                    # Demand matrix between all airports
airports = airport_data.columns                                     # List of airport names
len_airports = len(airports)                                        # Number of airports
distance, r, ls, g = load_airport_params(airport_data)                       # Distance matrix between all airports, max runway length matrix for all routes, 
                                                                    # available weekly landing slots at each airport, hub indicator
# Aircraft
aircraft_data = read_excel_pandas(aircraft_path, aircraft_sheet)    # Load aircraft data
aircraft_data = aircraft_data[0]                                    # type(aircraft_data)=df

aircraft_types, sp, s, TAT, R, RW, C_L, C_X, c_T, c_F = load_aircraft_params(aircraft_data)
len_aircraft_types = len(aircraft_types)

y = yields(distance)                            # Yield matrix based on distance matrix


C = operating_costs(airport_data, aircraft_data, distance, sp, FUEL, C_X, c_T, c_F)   # Operating cost matrix for all routes and aircraft types

# Variables for dummy constraints:

a = np.empty((len_airports, len_airports, len_aircraft_types), dtype=int)
b = np.empty((len_airports, len_airports, len_aircraft_types), dtype=int)

for i, porti in enumerate(airports):
    for j, portj in enumerate(airports):
        for k, aircraft in enumerate(aircraft_types):
            if distance[i][j] <= R[k]:
                a[i][j][k] = 10000
            else:
                a[i][j][k] = 0
            if r[i][j] >= RW[k]:
                b[i][j][k] = 10000
            else:
                b[i][j][k] = 0


# Start modelling optimization problem
m = Model('question1B')
x = {}
z = {}
w = {}
ac = {}

for i,porti in enumerate(airports):
    for j,portj in enumerate(airports):
        x[porti,portj] = m.addVar(lb=0, vtype=GRB.INTEGER, name=''.join(['Direct flow from ', str(porti), ' to ', str(portj)]))
        w[porti,portj] = m.addVar(lb=0,vtype=GRB.INTEGER, name=''.join(['Transfer flow from ',str(porti), ' to ', str(portj)]))
        for k,aircraft in enumerate(aircraft_types):
            z[porti,portj,aircraft] = m.addVar(lb=0, vtype=GRB.INTEGER, name=''.join(['Flights from ',str(porti), ' to ', str(portj), ' by ', str(aircraft)]))

for k,aircraft in enumerate(aircraft_types):
    ac[aircraft] = m.addVar(lb=0, vtype=GRB.INTEGER, name=''.join(['Number of ', str(aircraft)]))

m.update()
m.setObjective(quicksum(quicksum(y[i][j]*distance[i][j]*(x[porti,portj]+w[porti,portj]) for i,porti in enumerate(airports)) for j,portj in enumerate(airports))
              - quicksum(quicksum(quicksum(C[i][j][k]*z[porti,portj,aircraft] for i, porti in enumerate(airports)) for j, portj in enumerate(airports)) for k, aircraft in enumerate(aircraft_types))
              - quicksum(C_L[k]*ac[aircraft] for k, aircraft in enumerate(aircraft_types)), GRB.MAXIMIZE)  # The objective is to maximize revenue

for i,porti in enumerate(airports):
    for j,portj in enumerate(airports):
        m.addConstr(x[porti,portj] + w[porti,portj]<= q[i][j])          #C1
        m.addConstr(w[porti, portj] <= q[i][j] * g[i] * g[j])           #C2

        #NEW CONSTRAINT NEEDED?
        m.addConstr(x[porti,portj] <= q[i][j]* (1-g[i]+ 1-g[j]))        # Direct flow only to/from hub airport

        m.addConstr(x[porti,portj] + quicksum(w[porti,portm] * (1-g[j]) for m,portm in enumerate(airports)) + quicksum(w[portm,portj] * (1-g[i]) for m,portm in enumerate(airports)) 
                    <= quicksum(z[porti,portj,aircraft] * s[k] * LF for k,aircraft in enumerate(aircraft_types)))  #C3

        for k,aircraft in enumerate(aircraft_types):
            m.addConstr(z[porti,portj,aircraft] <= a[i][j][k])  # C6
            m.addConstr(z[porti,portj,aircraft] <= b[i][j][k])  # C7

for k,aircraft in enumerate(aircraft_types):
    for i,porti in enumerate(airports):
        m.addConstr(quicksum(z[porti,portj,aircraft] for j,portj in enumerate(airports)) ==  quicksum(z[portj, porti,aircraft] for j,portj in enumerate(airports))) #C4

for j,portj in enumerate(airports):
    m.addConstr(quicksum(quicksum(z[porti,portj,aircraft] for i,porti in enumerate(airports)) for k,aircraft in enumerate(aircraft_types)) <= ls[j])  # C8

for k,aircraft in enumerate(aircraft_types):
    m.addConstr(quicksum(quicksum((distance[i][j]/sp[k]+TAT[k]*(1 + 0.5 * (1 - g[j])))*z[porti,portj,aircraft] for i,porti in enumerate(airports)) for j,portj in enumerate(airports)) <= BT*ac[aircraft]) #C5



m.update()
m.write('test.lp')
# Set time constraint for optimization (5 minutes)
m.setParam('TimeLimit', 5 * 60)
# m.setParam('MIPgap', 0.009)
m.optimize()
# m.write("testout.sol")
status = m.status

if status == GRB.Status.UNBOUNDED:
    print('The model cannot be solved because it is unbounded')

elif status == GRB.Status.OPTIMAL or True:
    f_objective = m.objVal
    print('***** RESULTS ******')
    print('\nObjective Function Value: \t %g' % f_objective)

elif status != GRB.Status.INF_OR_UNBD and status != GRB.Status.INFEASIBLE:
    print('Optimization was stopped with status %d' % status)


# # Print out Solutions
print()
print("Frequencies:----------------------------------")
print()
for i, porti in enumerate(airports):
    for j, portj in enumerate(airports):
        if x[porti, portj].X > 0 :
            print(f"Direct flow from {porti} to {portj}: {x[porti, portj].X:.0f} passengers")
        if w[porti, portj].X > 0 :
            print(f"Transfer flow from {porti} to {portj}: {w[porti, portj].X:.0f} passengers")
        for aircraft in aircraft_types:
            if z[porti, portj, aircraft].X > 0:  # only print positive frequencies
                print(f"{porti} -> {portj} : {z[porti, portj, aircraft].X:.0f} flights using {aircraft}")

for k, aircraft in enumerate(aircraft_types):
    print(f'Total number of aircraft of type {aircraft}: {ac[aircraft].X:.0f}')
