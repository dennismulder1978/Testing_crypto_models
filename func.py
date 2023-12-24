import pandas as pd
import numpy as np
import os
import pickle
import tensorflow as tf

def create_arrays(file_name: str, N_STEPS: int):
    """
    Creates a numpy array with multiple N_STEPS-hours cohorts of 4 trade values of a cryptocoin. 
    Returns:
        np.array of multiple cohort in shape (amount of cohorts, N_STEPS, 4)
    """
    
    coin_data = pd.read_csv(file_name)
    coin_array = np.array(coin_data.drop(coin_data[['Vol.', 'Date']], axis=1))[:26]

    
    for i in range(0, len(coin_array)-N_STEPS):
        if i == 0:
            test_array = np.array([coin_array[0:N_STEPS]])
        else:
            temp_array = np.array([coin_array[i:i+N_STEPS]])
            test_array = np.append(test_array, temp_array, axis=0)
    return test_array


def list_of_LSTM_models(model_path):
    """create a list of available models and scalers

    Args:
        model_path (str): location of models/ scalers

    Returns:
        dict: different models/ scalers
    """    
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

        
def test_models(test_array:np.ndarray, model_dict: dict):
    key_list = list(model_dict.keys())
    
    for each_model in key_list:
        # loading model
        file_name_model = './Models/' + model_dict[each_model]['Model']
        model = tf.keras.models.load_model(file_name_model)
        # scaler
        file_name_scaler = './Models/' + model_dict[each_model]['Scaler']
        with open(file_name_scaler, "rb") as f:
            scaler = pickle.load(f)
        
        # Each cohort
        for each_cohort in test_array:
            scaled_array = scaler.fit_transform(each_cohort).reshape(1,24,4)
            y_pred = model.predict(scaled_array, verbose=0)[0][0]
            print(y_pred)
        input_pred = [[y_pred, y_pred, y_pred, y_pred]]
        #  scaler.inverse_transform(input_pred)[0][0]
    # except Exception as err:
    #     log(f'ERROR BALANCE,{coin},1,1,1,1,{err}')
    #     send_mail(action='Error', stringer=f'GET_CANDLES went wrong: {err}')
    #     print(err)