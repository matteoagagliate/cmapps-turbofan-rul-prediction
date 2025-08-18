# --- LIBRARIES ---
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
import os
import random
import numpy as np
from src import models as MD
import pandas as pd


# --- UTILS ---

def plot_sensor(sensor_name, train_set):
    plt.figure(figsize=(13,5))
    for i in train_set['unit_number'].unique():
        if (i % 10 == 0):  
            subset = train_set[train_set['unit_number'] == i]
            plt.plot(subset['RUL'], subset[sensor_name], label=f'Unit {i}')
    plt.xlim(250, 0) 
    plt.xticks(np.arange(0, 275, 25))
    plt.ylabel(sensor_name)
    plt.xlabel('Remaining Useful Life')
    plt.title(f'Sensor {sensor_name} behavior over RUL')
    plt.legend(loc='upper right', fontsize='small')
    plt.grid(True)
    plt.show()

def evaluate(y_true, y_pred, label, print_results):

    y_pred = tf.reshape(y_pred, [-1]) 
    y_true = tf.reshape(y_true, [-1])
    y_true = np.array(y_true, dtype=np.float32)
    y_pred = np.array(y_pred, dtype=np.float32)
    

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    signed_error = np.mean(y_pred - y_true)

    if print_results:
        print(f'{label} set - RMSE: {rmse:.4f}, R2: {r2:.4f}, Mean Error: {signed_error:.2f}')

    return signed_error, rmse, r2

def save_results(filename, new_results_dict):
    if isinstance(new_results_dict, dict):
        new_df = pd.DataFrame([new_results_dict])
    else:
        new_df = pd.DataFrame(new_results_dict)

    if not os.path.exists(filename):
        new_df.to_csv(filename, index=False, mode='w', header=True)
    else:
        new_df.to_csv(filename, index=False, mode='a', header=False)

def asymmetric_mse(y_true, y_pred, penalty_factor=2.0):
    error = y_pred - y_true
    
    mask = tf.cast(error > 0, tf.float32)
    
    return tf.reduce_mean(
        tf.where(
            mask == 1, 
            penalty_factor * tf.square(error),
            tf.square(error) 
        )
    )


