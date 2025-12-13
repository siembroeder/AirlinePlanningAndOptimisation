

from Question2.load_pmf_data import load_assignment_data

def main():
    '''
    File to calculate some basic statistics from the input data.
    Such as,
    avg fare of single and double itineraries
    avg recapture rate
    demand for single and double itineraries
    '''

    flights, itins, recaps, flight_idx = load_assignment_data()

    flight_demand_freqs = {i: 0 for i in flights['Flight No.']}

    total_fare_single = 0
    total_fare_double = 0
    num_itins_single = 0
    num_itins_double = 0

    single_dem = 0
    double_dem = 0
    

    for itin in itins:
        itin = itins[itin]
        f1 = itin['Leg1']
        f2 = itin['Leg2']


        flight_demand_freqs[f1] +=1

        if type(f2) == str:
            flight_demand_freqs[f2] +=1
            total_fare_double += itin['Fare']
            num_itins_double +=1
            double_dem += itin['Demand']
        else:
            total_fare_single += itin['Fare']
            num_itins_single +=1
            single_dem += itin['Demand']
    
    avg_fare_single = total_fare_single / num_itins_single
    avg_fare_double = total_fare_double / num_itins_double

    for k, v in sorted(flight_demand_freqs.items(), key=lambda item: item[1]):
        print(k, v)

    print(f'Avg fare s: {total_fare_single} / {num_itins_single} = {avg_fare_single}')
    print(f'Avg fare d: {total_fare_double} / {num_itins_double} = {avg_fare_double}')
    print(f'singledem {single_dem}, doubledem {double_dem}')


    total_rate = 0
    num_rate   = 0
    for rate in recaps['RecapRate']:

        total_rate += rate
        num_rate +=1

    avg_rate = total_rate / num_rate
    print(f'Avg rate: {total_rate} / {num_rate} = {avg_rate}')
    

if __name__ == "__main__":
    main()






