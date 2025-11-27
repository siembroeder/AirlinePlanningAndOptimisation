from turtle import distance
from Question1A.DistancesLatLong import compute_dij


import numpy as np 




def distances(airport_data):
    '''
    Compute the distance matrix between all airports.
    '''
    airports = airport_data.columns
    len_airports = len(airports)
    
    distance = np.zeros((len_airports, len_airports))
    
    for i, porti in enumerate(airports):
        for j, portj in enumerate(airports):
            if i == j:
                distance[i, j] = 0
            else:
                distance[i, j] = compute_dij(airport_data, porti, portj)
    
    return distance


def yields(distance):
    '''
    Compute yield matrix based on distance matrix.
    '''
    y = np.zeros(distance.shape)
    for i in range(len(distance[0])):
        for j in range(len(distance[0])):
            if i == j:
                y[i, j] = 0
            else:
                y[i, j] = 5.9*distance[i, j]**-0.76 + 0.043
    
    return y
