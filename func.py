import pandas as pd
import numpy as np
import os
import pickle
import tensorflow as tf
from datetime import datetime as dt

def create_arrays(crypto_data_path: str, N_STEPS_list: list):
    """
    Creates a numpy array with multiple N_STEPS-hours cohorts of 4 trade values of a cryptocoin.
    Returns:
        dict with np.arrays of multiple cohort in shape (amount of cohorts, N_STEPS, 4)
    """
    file_list = os.listdir(path=crypto_data_path)
    file_name = crypto_data_path + file_list[0]
    coin_data = pd.read_csv(file_name)
    column_list = coin_data.columns
    coin_array = np.array(coin_data.drop(coin_data[[column_list[0], column_list[-1]]], axis=1))
    results = {}

    for N_STEPS in N_STEPS_list:
        test_array = np.array([0])
        for i in range(0, len(coin_array)-N_STEPS):
            if i == 0:
                test_array = np.array([coin_array[0:N_STEPS]])
            else:
                temp_array = np.array([coin_array[i:i+N_STEPS]])
                test_array = np.append(test_array, temp_array, axis=0)
            results[N_STEPS] = test_array
        print(f'Done creating cohort-array with {N_STEPS} n_steps.')
    return results


def list_of_LSTM_models(model_path):
    """create a list of available models and scalers

    Args:
        model_path (str): location of models/ scalers

    Returns:
        dict: different models/ scalers
    """

    temp_dict = {}
    n_list = []
    # create list of all available files
    file_list = os.listdir(path=model_path)

    # process each file into dictionary
    for each in file_list:
        temp = each.split('_')
        index_name = f'{temp[3]}_{temp[4]}'
        if temp_dict.get(index_name) is None:
            temp_dict[index_name] = {'N_STEPS': int(temp[3][1:3])}
            temp_dict[index_name].update({'F_STEPS': int(temp[4][1:2])})
            temp_dict[index_name].update({temp[5]: each})
        else:
            temp_dict[index_name].update({temp[5]: each})
    
    # fill n_list with all different n_steps
    for each in temp_dict.keys():
        n_list.append(temp_dict[each]['N_STEPS'])
    n_list = sorted(list(set(n_list)))
    print(f'Done recording all models with n_step_list: {n_list}')
    return temp_dict, n_list


def test_models(test_array_dict: dict,
                model_dict: dict,
                percentage_list: list):
    """Makes prediction based on the model and input-arrays

    Args:
        test_array_dict (dict): dict of input arrays containing crypto-trade-data
        model_dict (dict): list of available models/ scalers
        percentage_list (list): list of testable percentages
    Returns:
        (dict): per model the results in cohorts
    """
    results = {}
    models_key_list = list(model_dict.keys())
    
    for each_model in models_key_list:
        # loading model
        file_name_model = './Models/' + model_dict[each_model]['Model']
        model = tf.keras.models.load_model(file_name_model)  # <-- model

        # scaler
        file_name_scaler = './Models/' + model_dict[each_model]['Scaler']
        with open(file_name_scaler, "rb") as f:
            scaler = pickle.load(f)  # <-- Scaler

        # selecting appropriate n_steps_array
        n_steps = model_dict[each_model]['N_STEPS']  # <-- n_steps
        f_steps = model_dict[each_model]['F_STEPS']  # <-- f_steps
        test_array = test_array_dict[n_steps]  # <-- appropriate n_steps_array

        # Each cohort
        cohort_results_list = []
        for i, each_cohort in enumerate(test_array):
            scaled_array = scaler.fit_transform(each_cohort).reshape(1,n_steps,4)
            y_pred = model.predict(scaled_array, verbose=0)[0][0]
            input_pred = [[y_pred, y_pred, y_pred, y_pred]]
            real_pred_list = [each_cohort[-1][3], scaler.inverse_transform(input_pred)[0][3]]
            cohort_results_list.append(real_pred_list)
            print('.', end='')
        print()
        print(f'Done predicting for model: {each_model}.')
        # determine profit/ loss per model per percentage based on the results.
        for buy_perc in percentage_list:
            for sell_perc in percentage_list:
                euro = 0
                coin = 1000
                for i in range (f_steps, len(cohort_results_list)):
                    temp_result = cohort_results_list[i][0] / cohort_results_list[i-f_steps][1]
                    if (euro > 0) & (temp_result > ((buy_perc / 100) + 1)):
                        # print(f'Buy action: {temp_result}, {((buy_perc / 100) + 1)}, {euro}, {coin}')
                        coin = euro / cohort_results_list[i][0]
                        euro = 0
                    elif (coin > 0) & (temp_result < (1 - (sell_perc / 100))):
                        # print(f'Sell action: {temp_result}, {(1 - (sell_perc / 100))}, {euro}, {coin}')
                        euro = coin * cohort_results_list[i][0]
                        coin = 0
                    else:
                        # print(f'No action: {temp_result}, {((buy_perc / 100) + 1)}, {(1 - (sell_perc / 100))}, {euro}, {coin}')
                        pass
                #record results
                result_name = f'{each_model}-{buy_perc}-{sell_perc}'
                if coin == 0:
                    coin = euro / cohort_results_list[-1][0]
                results[result_name] = coin
                f = open('./results/test.csv', 'a')
                f.write(f'{each_model},{buy_perc},{sell_perc},{coin}\n')
                f.close()
        print(f'Done calculating profits for model: {each_model}.')
    print('Done calculating profits for all models.')
    return results

def save_dict(saving_dict: dict, file_name: str):
    """Saves results dict

    Args:
        saving_dict (dict): _description_
        file_name (str): _description_

    Returns:
        None: None
    """    
    file_name = './results/' + file_name + '_' + str(int(dt.now().timestamp())) + '.csv'
    try:    
        f = open(file_name, 'a')
        f.write('model_name,buy_percentage,sell_percentage,result\n')
        for k, v in saving_dict.items():
            temp = str(k).split('-')
            print(temp)
            line = f'{temp[0]},{temp[1]},{temp[2]},{v}\n'
            f.write(line)
        f.close()
    except Exception as e:
        print(e)
    return None