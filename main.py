from func import *

## input values:
crypto_data_path = './crypto_data/'
model_path = './models/'
N_STEPS = 24

# step 1: create cohorts
# test_array = create_arrays(crypto_data_path=crypto_data_path, N_STEPS=N_STEPS)

# step 2: load each model
model_dict = list_of_LSTM_models(model_path=model_path) 
print(model_dict)
# step 3: Test each model at different percentages. 
# prediction_dict = test_models(test_array=test_array, model_dict=model_dict)
# print(prediction_dict)




