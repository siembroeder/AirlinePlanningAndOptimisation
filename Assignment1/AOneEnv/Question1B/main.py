from gurobipy import *

from Question1A.main import main
from Question1A.Read_input import read_excel_pandas
from .ComputeParameters_1B import distances, yields
import numpy as np
import pandas  as pd

aircraft_path = r"C:\Users\pop_r\OneDrive - Delft University of Technology\Desktop\AirlinePlanning\Assignment1\Data\AircraftData.xlsx"
aircraft_sheet = ["AircraftTypes"]
aircraft_data = read_excel_pandas(aircraft_path, aircraft_sheet)    # Load aircraft data


# cd "C:\Users\pop_r\OneDrive - Delft University of Technology\Documents\SVV\AirlinePlanningAndOptimisation\Assignment1\AOneEnv"
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
m.setObjective(m.getObjective(), GRB.MAXIMIZE)  # The objective is to maximize revenue

for i in airports:
    for j in airports:
        m.addConstr(x[i,j] <= q[i][j]) #C1
        m.addConstr(x[i, j] <=z[i,j]*s*LF) #C2
    m.addConstr(quicksum(z[i,j] for j in airports) ==  quicksum(z[j, i] for j in airports)) #C3

m.addConstr(quicksum(quicksum((distance[i][j]/sp+LTO)*z[i,j] for i in airports) for j in airports) <= BT*AC) #C4


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