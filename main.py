from func import *

## input values:
crypto_data_path = './crypto_data/'
model_path = './models/'
percentage_list = [0.5, 1, 2, 3]

# step 1: determine avaiability of each model
model_dict, n_list = list_of_LSTM_models(model_path=model_path) 
print(model_dict)
# step 2: create cohorts
test_array_dict = create_arrays(crypto_data_path=crypto_data_path, 
                                N_STEPS_list=n_list)

# step 3: Test each model at different percentages. 
prediction_dict = test_models(test_array_dict=test_array_dict,
                              model_dict=model_dict,
                              percentage_list=percentage_list)
print(prediction_dict)




