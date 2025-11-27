
from .DistancesLatLong import compute_dij

import pandas as pd
import numpy  as np
import statsmodels.api as sm

def convert_to_ij_format(airport_data, gdp_data, pop_data, f=0, year = '2000', demand_data = None):
    '''
    To execute OLS fit and to calculate demand Dij we gather the data in a df with columns i, j corresponding to routes 
        and popi and popj etc for the data on that route.
    '''
    rows = []           # list of dicts with i, j, fdij, Dij data, converted to df after
    i_indices = []      # Store indices of all pairs of ij with i \neq j
    j_indices = []

    airports = airport_data.columns

    for i, porti in enumerate(airports):
        for j, portj in enumerate(airports):

            if i == j:      # Skip when both airports are the same.
                continue

            dij = compute_dij(airport_data, porti, portj)

            # Only for 2021 demand data is available.
            if demand_data is not None:
                rows.append({'i': porti, 'j': portj, 'fdij': f*dij, 'Dij':demand_data[i,j]})
            else:
                rows.append({'i': porti, 'j': portj, 'fdij': f*dij})
            
            i_indices.append(i)
            j_indices.append(j)

    pairs_df     = pd.DataFrame(rows)   # Convert to df with columns i, j, f*dij, Dij

    gdp_year = gdp_data[str(year)+'gdp']
    pop_year = pop_data[str(year)+'pop']

    # We need gdp and pop data in ij format
    airport_gdp_pop_i = pd.DataFrame({
                            'gdpi': gdp_year.values[i_indices],
                            'popi': pop_year.values[i_indices]})

    airport_gdp_pop_j = pd.DataFrame({
                            'gdpj': gdp_year.values[j_indices],
                            'popj': pop_year.values[j_indices]})

    # Collect i,j,f*dij,Dij,i_gdp,i_pop,j_gdp,j_pop in one df
    ij_format_data = pd.concat([pairs_df, airport_gdp_pop_i, airport_gdp_pop_j], axis=1)

    return ij_format_data

def execute_OLS_fit(gravity_data):
    '''
    Find the coefficients k, b1, b2, b3 of the gravity model based on 2021 data.
    Linearize in the coeffs using logarithm. Note how k and b3 are extraced to compensate for this log
    '''

    # Define log versions of data used for OLS fitting of gravity model linear in coeffs.
    gravity_log = pd.DataFrame()

    gravity_log['log_pipj'] = np.log(gravity_data['popi'] * gravity_data['popj'])
    gravity_log['log_gigj'] = np.log(gravity_data['gdpi'] * gravity_data['gdpj'])
    gravity_log['log_fdij'] = np.log(gravity_data['fdij'])
    gravity_log['log_Dij']  = np.log(gravity_data['Dij'])

    # Select only data used for fitting
    grav_demand  = gravity_log[['log_pipj','log_gigj','log_fdij']]
    grav_demand  = sm.add_constant(grav_demand)
    known_demand = gravity_log['log_Dij']

    # Execute fit
    model  = sm.OLS(known_demand, grav_demand).fit()
    coeffs = model.params

    # Extract coeffs
    b1 = coeffs['log_pipj']
    b2 = coeffs['log_gigj']
    b3 = -1*coeffs['log_fdij']
    k  = np.exp(coeffs['const'])

    # print(all_ols_data.head())
    # print(model.summary())

    print(f"OLS Result: \nb1={b1},b2={b2},b3={b3},k={k}")

    return k, b1, b2, b3


 