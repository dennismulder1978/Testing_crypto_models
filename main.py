from const import *
from func import *

coin_data = pd.read_csv('./DOGE_trade_data__per_1h__20814_items__created_21-Dec-2023_16-07.csv')
coin_array = np.array(coin_data.drop(coin_data[['Vol.', 'Date']], axis=1))[:30]

N_STEPS = 24

test_array = create_arrays(coin_array=coin_array, N_STEPS=N_STEPS)


for each in test_array:
    print(each.reshape(1,24,4).shape)
    
    
print(test_array.shape)