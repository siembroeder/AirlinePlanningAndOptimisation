

from gurobipy import *

from Question1A.main import main
from ComputeParameters_1B import distances
import numpy as np


# Constants:
FUEL = 1.42                                # Eur/Gallon
LF = 0.75                                  # Average load factor assumed 75%           
BT = 10 * 7                                # Assumed 10 hours of block time per day per aircraft



# Data
airport_data, q = main()                 # Using results obtained from question 1A for demand
airports = airport_data.columns            # List of airport names
print(airports)
len_airports = len(airports)               # Number of airports
distance = distances(airport_data)         # Distance matrix between all airports

# CASK = 0.12

# s = 120
# sp = 870
# LTO = 20/60

# AC = 2
# y = 0.18  # yield
# distance = [[0, 2236, 3201],
#           [2236, 0, 3500],
#           [3201, 3500, 0]]

# # Start modelling optimization problem
# m = Model('practice')
# x = {}
# z = {}
# for i in airports:
#     for j in airports:
#         x[i,j] = m.addVar(obj = y*distance[i][j],lb=0, vtype=GRB.INTEGER)
#         z[i,j] = m.addVar(obj = -CASK*distance[i][j]*s, lb=0,vtype=GRB.INTEGER)

# m.update()
# m.setObjective(m.getObjective(), GRB.MAXIMIZE)  # The objective is to maximize revenue

# for i in airports:
#     for j in airports:
#         m.addConstr(x[i,j] <= q[i][j]) #C1
#         m.addConstr(x[i, j] <=z[i,j]*s*LF) #C2
#     m.addConstr(quicksum(z[i,j] for j in airports) ==  quicksum(z[j, i] for j in airports)) #C3

# m.addConstr(quicksum(quicksum((distance[i][j]/sp+LTO)*z[i,j] for i in airports) for j in airports) <= BT*AC) #C4


# m.update()
# # m.write('test.lp')
# # Set time constraint for optimization (5minutes)
# # m.setParam('TimeLimit', 1 * 60)
# # m.setParam('MIPgap', 0.009)
# m.optimize()
# # m.write("testout.sol")
# status = m.status

# if status == GRB.Status.UNBOUNDED:
#     print('The model cannot be solved because it is unbounded')

# elif status == GRB.Status.OPTIMAL or True:
#     f_objective = m.objVal
#     print('***** RESULTS ******')
#     print('\nObjective Function Value: \t %g' % f_objective)

# elif status != GRB.Status.INF_OR_UNBD and status != GRB.Status.INFEASIBLE:
#     print('Optimization was stopped with status %d' % status)


# # Print out Solutions
# print()
# print("Frequencies:----------------------------------")
# print()
# for i in airports:
#     for j in airports:
#         if z[i,j].X >0:
#             print(Airports[i], ' to ', Airports[j], z[i,j].X)