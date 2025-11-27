
from gurobipy import *
import numpy as np
import pandas  as pd

# Custom imports
from Assignment1.AOneEnv.Question1A.main import main
from Assignment1.AOneEnv.Question1A.Read_input import read_excel_pandas
from Assignment1.AOneEnv.Question1B.ComputeParameters_1B import distances, yields

aircraft_path = r"C:\Users\jobru\Documents\TU Delft\MSc AE\Year 1\Courses Q2\APandO\Assignment files\AirlinePlanningAndOptimisation\Assignment1\Data\AircraftData.xlsx"
aircraft_sheet = ["AircraftTypes"]
aircraft_data = read_excel_pandas(aircraft_path, aircraft_sheet)    # Load aircraft data

# Radu:
# cd "C:\Users\pop_r\OneDrive - Delft University of Technology\Documents\SVV\AirlinePlanningAndOptimisation\Assignment1\AOneEnv"
# python -m uv run -m Question1B.main

# Job:
# cd "C:\Users\jobru\Documents\TU Delft\MSc AE\Year 1\Courses Q2\APandO\Assignment files\AirlinePlanningAndOptimisation\Assignment1\AOneEnv"
# python -m uv run -m Question1B.main

# Constants:
FUEL = 1.42                                # Eur/Gallon
LF = 0.75                                  # Average load factor assumed 75%           
BT = 10 * 7                                # Assumed 10 hours of block time per day per aircraft

# Data

# Airports and routes
airport_data, q = main()                        # Using results obtained from question 1A for demand
airports = airport_data.columns                 # List of airport names
len_airports = len(airports)                    # Number of airports
distance = distances(airport_data)              # Distance matrix between all airports

# Aircraft

aircraft_data = aircraft_data[0]                # type(aircraft_data)=df



y = yields(distance)                            # Yield matrix based on distance matrix
print("Yield matrix:\n", y)


# CASK = 0.12

# s = 120
# sp = 870
# LTO = 20/60

# AC = 2

# Variables for dummy constraints:

a = np.empty((len_airports, len_airports, len_aircraft_types), dtype=int)
b = np.empty((len_airports, len_airports, len_aircraft_types), dtype=int)

for i in airports:
    for j in airports:
        for k in aircraft_types:
            if distance[i][j] <= R[k]:
                a[i][j][k] = 10000
            else:
                a[i][j][k] = 0
            if r[i][j] <= RW[k]:
                b[i][j][k] = 10000
            else:
                b[i][j][k] = 0

# Start modelling optimization problem
m = Model('question1B')
x = {}
z = {}
w = {}
ac = {}

for i in airports:
    for j in airports:
        x[i,j] = m.addVar(lb=0, vtype=GRB.INTEGER, name=''.join(['Direct flow from ', str(i), ' to ', str(j)]))
        w[i,j] = m.addVar(lb=0,vtype=GRB.INTEGER, name=''.join(['Transfer flow from ',str(i), ' to ', str(j)]))
        for k in aircraft_types:
            z[i,j,k] = m.addVar(lb=0, vtype=GRB.INTEGER, name=''.join(['Flights from ',str(i), ' to ', str(j), ' by ', str(k)]))

for k in aircraft_types:
        ac[k] = m.addVar(lb=0, vtype=GRB.INTEGER, name=''.join(['Number of ', str(k)]))

m.update()
m.setObjective(quicksum(quicksum(y[i,j]*distance[i,j]*(x[i,j]+w[i,j]) for i in airports) for j in airports)
              - quicksum(quicksum(quicksum(totaloperatingcost*z[i,j,k] for i in airports) for j in airports) for k in aircraft_types)
              - quicksum(L*ac[k] for k in aircraft_types), GRB.MAXIMIZE)  # The objective is to maximize revenue

for i in airports:
    for j in airports:
        m.addConstr(x[i,j] + w[i,j]<= q[i][j])          #C1
        m.addConstr(w[i, j] <= q[i][j] * g[i] * g[j])   #C2
        m.addConstr(x[i,j] + quicksum(w[i,m] * (1-g[j]) for m in airports) + quicksum(w[m,j] * (1-g[i]) for m in airports) 
                    <= quicksum(z[i,j,k] * s[k] * LF for k in aircraft_types))  #C3
        
        m.addConstr(z[i,j,k] <= a[i][j][k])  #C6
        m.addConstr(z[i,j,k] <= b[i][j][k])  #C7

    for k in aircraft_types:
        m.addConstr(quicksum(z[i,j,k] for j in airports) ==  quicksum(z[j, i,k] for j in airports)) #C4

# for j in airports:

for k in aircraft_types:
    m.addConstr(quicksum(quicksum((distance[i][j]/sp[k]+TAT[k]*(1 + 0.5 * (1 - g[j])))*z[i,j,k] for i in airports) for j in airports) <= BT*ac[k]) #C5



m.update()
# m.write('test.lp')
# Set time constraint for optimization (5minutes)
# m.setParam('TimeLimit', 1 * 60)
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
# print()
# print("Frequencies:----------------------------------")
# print()
# for i in airports:
#     for j in airports:
#         if z[i,j].X >0:
#             print(Airports[i], ' to ', Airports[j], z[i,j].X)