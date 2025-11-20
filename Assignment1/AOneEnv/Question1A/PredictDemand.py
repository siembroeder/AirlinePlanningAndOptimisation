
import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def predict_demand_gravity(coeffs, ij_year):
    '''
    Calculate the demand from coefficients, population/gdp data, fuel price, distances
    '''

    k, b1, b2, b3 = coeffs

    popi = ij_year['popi']
    popj = ij_year['popj']

    gdpi = ij_year['gdpi']
    gdpj = ij_year['gdpj']

    fdij = ij_year['fdij']
    
    pred_demand = pd.DataFrame()
    pred_demand['i'] = ij_year['i']
    pred_demand['j'] = ij_year['j']

    pred_demand['Dij'] = k*(popi*popj)**b1*(gdpi*gdpj)**b2 / (fdij)**b3

    return pred_demand

def extrapolate_gdp_pop(gdp_data, pop_data, pred_year = None):
    '''
    Extrapolate the constant growth in years 2021 and 2024 in gdp and pop data to pred_year

    gdp_data: df that should include columns "Country", "2021gdp", "2024gdp" 
    pop_data: df that should include columns "Cities", "2021pop", "2024pop"
    pred_year: Int64
    '''
    year1 = 2021
    year2 = 2024

    dx           = year2 - year1
    extrap_range = pred_year - year2

    cities    = pop_data['City'].tolist() # preserves order
    countries = gdp_data['Country'].tolist()

    pop_2021 = pop_data['2021pop']
    pop_2024 = pop_data['2024pop']

    gdp_2021 = gdp_data['2021gdp']
    gdp_2024 = gdp_data['2024gdp']

    gdp_rows = []
    pop_rows = []
    for i, (countryi, cityi) in enumerate(zip(countries, cities)):

        pop_2021_i = pop_2021.loc[i]
        pop_2024_i = pop_2024.loc[i]
        gdp_2021_i = gdp_2021.loc[i]
        gdp_2024_i = gdp_2024.loc[i]

        dpop       = pop_2024_i - pop_2021_i
        pop_dyear  = dpop/dx
        pop_pred_year = pop_2024_i + pop_dyear * extrap_range

        dgdp       = gdp_2024_i - gdp_2021_i
        gdp_dyear  = dgdp/dx
        gdp_pred_year = gdp_2024_i + gdp_dyear * extrap_range

        pred_year_popkey = str(pred_year) + 'pop'
        pred_year_gdpkey = str(pred_year) + 'gdp'
        pop_rows.append({'City': cityi, 'yearly_change':pop_dyear, pred_year_popkey:pop_pred_year})
        gdp_rows.append({'Country': countryi, 'yearly_change':gdp_dyear, pred_year_gdpkey:gdp_pred_year})

    return pd.DataFrame(gdp_rows), pd.DataFrame(pop_rows)


def plot_demand_routes(dem_21, dem_24, dem_26, routes):
    '''
    Plotting function for demand across years 2021, 2024, 2026.
    Plots all routes in the same graph.
    '''

    dem_21 = dem_21.set_index(['i', 'j'])
    dem_24 = dem_24.set_index(['i', 'j'])
    dem_26 = dem_26.set_index(['i', 'j'])
    # print(dem_21['Dij'].loc[('London', 'Paris')])

    years = [21, 24, 26]
    
    cmap = cm.tab10  
    colors = {route: cmap(i % 10) for i, route in enumerate(routes)}


    fig, ax = plt.subplots(figsize=(10, 6))
    for route in routes:
        dem = [dem_21['Dij'].loc[route], dem_24['Dij'].loc[route], dem_26['Dij'].loc[route]]

        # ax.scatter(years, dem, color=colors[route])
        ax.plot(years, dem, color=colors[route], linestyle='--', label = route)

    max_demand = dem_26['Dij'].max()
    plt.ylim((0, max_demand*1.1 ))
    plt.legend()
    plt.show()
    











