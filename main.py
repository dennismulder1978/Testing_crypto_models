from func import *

## input values:
crypto_data_path = './crypto_data/'
model_path = './models/'
percentage_list = [1, 2, 3, 4, 5, 6, 7]
saving_file_name = '2nd'

# step 1: determine availability of each model
model_dict, n_list = list_of_LSTM_models(model_path=model_path) 

# step 2: create cohorts
test_array_dict = create_arrays(crypto_data_path=crypto_data_path, 
                                N_STEPS_list=n_list)

# step 3: Test each model at different percentages. 
prediction_dict = test_models(test_array_dict=test_array_dict, 
                              model_dict=model_dict, 
                              percentage_list=percentage_list)

# step 4: saving prediction results
save_dict(saving_dict=prediction_dict, 
          file_name=saving_file_name)
