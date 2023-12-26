from audioop import reverse
from func import *

## input values:
crypto_data_path = './crypto_data/'
model_path = './models/'
percentage_list = [1, 2, 3, 4, 5, 6, 7]

# step 1: determine availability of each model
model_dict, n_list = list_of_LSTM_models(model_path=model_path) 

# step 2: create cohorts
test_array_dict = create_arrays(crypto_data_path=crypto_data_path, 
                                N_STEPS_list=n_list)

# step 3: Test each model at different percentages. 
prediction_dict = test_models(test_array_dict=test_array_dict, 
                              model_dict=model_dict, 
                              percentage_list=percentage_list)

result = dict(sorted(prediction_dict.items(), key=lambda item: item[1], reverse=True))

for k, v in result.items():
    print(f'{k}: {v}')




