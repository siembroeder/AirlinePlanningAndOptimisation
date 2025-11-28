import numpy as np

x = 1

def compute_pairwise_distance(loc1, loc2):
    '''
    Finds the distance between two airports at locations loc1 and loc2 via the equation given in appendix C-Airport Data.
    loc: [latitude, longitude] (in degrees)
    '''
    RE = 6371 # Radius of Earth in km
    lat1, long1 = np.radians(loc1)
    lat2, long2 = np.radians(loc2)

    sqrt_arg = np.sin((lat1-lat2)/2)**2 +np.cos(lat1)*np.cos(lat2)*np.sin((long1-long2)/2)**2
    d_sigma = 2*np.arcsin(np.sqrt(sqrt_arg))
    return RE * d_sigma


def compute_dij(port_df, init, dest):
    '''
    Returns the matrix d_ij including all the great circle distance between the airports in df.
    port_df: pandas dataframe that should at least contain 2 columns, each pertaining to an airport. 
             Each column should at least include the rows "Latitude (deg)" and "Longitude (deg)" corresponding to a location on earth.
    init, dest:    column in port_df, should be city name, eg "London". Order of init and dest doesn't matter
    '''
    coords_dep = [port_df[init]['Latitude (deg)'], port_df[init]['Longitude (deg)']]
    coords_arr = [port_df[dest]['Latitude (deg)'], port_df[dest]['Longitude (deg)']]

    dist = compute_pairwise_distance(coords_dep, coords_arr)

    return dist












