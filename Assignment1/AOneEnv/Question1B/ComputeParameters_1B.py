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

