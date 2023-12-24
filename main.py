from const import *
from func import *

## input values:
file_name = './DOGE_trade_data__per_1h__20814_items__created_21-Dec-2023_16-07.csv'
N_STEPS = 24
model_path = './models/'

# step 1: create cohorts
test_array = create_arrays(file_name=file_name, N_STEPS=N_STEPS)

# step 2: load each model
model_dict = list_of_LSTM_models(model_path=model_path) 

# step 3: Test each model at different percentages. 
test_models(test_array=test_array, model_dict=model_dict)





