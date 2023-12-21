import numpy as np
import pandas as pd

def create_arrays(coin_array, N_STEPS):
    """
    Creates a numpy array with multiple N_STEPS-hours cohorts of 4 trade values of a cryptocoin. 
    Returns:
        np.array of multiple cohort in shape (amount of cohorts, N_STEPS, 4)
    """
    for i in range(0, len(coin_array)-N_STEPS):
        if i == 0:
            test_array = np.array([coin_array[0:N_STEPS]])
        else:
            temp_array = np.array([coin_array[i:i+N_STEPS]])
            test_array = np.append(test_array, temp_array, axis=0)
    return test_array