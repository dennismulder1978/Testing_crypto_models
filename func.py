import pandas as pd
import numpy as np
import os
import json

def create_arrays(file_name, N_STEPS):
    """
    Creates a numpy array with multiple N_STEPS-hours cohorts of 4 trade values of a cryptocoin. 
    Returns:
        np.array of multiple cohort in shape (amount of cohorts, N_STEPS, 4)
    """
    
    coin_data = pd.read_csv(file_name)
    coin_array = np.array(coin_data.drop(coin_data[['Vol.', 'Date']], axis=1))[:30]

    
    for i in range(0, len(coin_array)-N_STEPS):
        if i == 0:
            test_array = np.array([coin_array[0:N_STEPS]])
        else:
            temp_array = np.array([coin_array[i:i+N_STEPS]])
            test_array = np.append(test_array, temp_array, axis=0)
    return test_array


def list_of_LSTM_models(model_path):
    temp_dict = {}
    file_list = os.listdir(path=model_path)
    for each in file_list:
        temp = each.split('_')
        index_name = f'{temp[3]}_{temp[4]}'
        if temp_dict.get(index_name) is None:
            temp_dict[index_name] = {temp[5]: each}
        else:
            temp_dict[index_name].update({temp[5]: each})
            
    return temp_dict
            