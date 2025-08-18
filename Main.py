# --- MAIN SCRIPT ---

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import random
import numpy as np

from src import feature_engineering as FE
from src import models as MD
from src import utils as UT
from src import data_processing as DP
from src import tuning as TN

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

CONFIG = {
    'dataset_dir':'data/',
    'dataset_name':'FD004',
    'seed': 34,
    'filter_name':'No Filtering',
    'random_search': False,
    'run_linear_regression': True, 
    'run_MLP': True,              
    'run_RF_regression': False,
    'run_XGBoost_regression': False,
    'run_LSTM': False,
    'run_GRU': False,
    'run_CNN1D': False  
}

# ==================== INITIALIZATION ====================
def set_random_seeds(seed):
    """Set all random seeds for reproducibility"""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# ==================== MAIN WORKFLOW ====================
def main():

    plt.close('all')
    set_random_seeds(CONFIG['seed'])

    print("\nLoading data...")
    dir_path, index_names, setting_names, sensor_names, column_names, train_set, test_set, y_test, y_train = DP.reading_data(CONFIG['dataset_dir'], CONFIG['dataset_name'])

    train_set, test_set = DP.apply_filtering(train_set, test_set, filter_name = 'No Filtering', sensor_names = sensor_names)
    
    if CONFIG['run_linear_regression']:
        print("\n=== Running Linear Regression ===")

        MD.run_linear_regression(CONFIG['dataset_name'], train_set, test_set, y_test, sensor_names = sensor_names, seed = CONFIG['seed'],
                                apply_clipping = 1, clipping_threshold = 125, apply_scaling = 0, apply_condition_scaling = 1, scaling_method = 'z-score', apply_polynomial_features = 0, poly_degree = 3, 
                                add_lagged_vars = False, print_results = 1, export_data = 0)

    if CONFIG['run_MLP']:
        print("\n=== Running MLP ===")

        if CONFIG['random_search']:
            TN.random_search_MLP(n_iter = 100, dataset_name=CONFIG['dataset_name'], train_set = train_set, test_set = test_set, y_test = y_test, train_size = 0.8, sensor_names = sensor_names, seed = CONFIG['seed'],              
            apply_scaling=1, scaling_method='z-score', apply_clipping = 1, clipping_threshold = 125, 
            )

        else:
            MD.run_MLP(CONFIG['dataset_name'], train_set, test_set, y_test, train_size = 0.8, sensor_names = sensor_names, seed = CONFIG['seed'],
                layer_sizes = [32, 64, 128], epochs = 13, batch_size = 32, l2_param = 0.01, add_lagged_vars = 0, activation_function='tanh',            
                apply_scaling=0, apply_condition_scaling=1, scaling_method='z-score', apply_clipping = 1, clipping_threshold = 125,
                plot_history=False, verbose=1, print_results = 1,  use_asymmetric_loss = 0, export_data = 0)

    if CONFIG['run_LSTM']: 
        print("\n=== Running LSTM ===")
        if CONFIG['random_search']:

            TN.random_search_LSTM(n_iter = 10, dataset_name=CONFIG['dataset_name'], train_set = train_set, test_set = test_set, y_test = y_test, train_size = 0.8, sensor_names = sensor_names, seed = CONFIG['seed'],              
                apply_scaling=0, apply_condition_scaling = 1, scaling_method='z-score', apply_clipping = 1, clipping_threshold = 125,
                )  

        else:
            rmse_train, r2_train, rmse_dev, r2_dev, computation_time = MD.run_LSTM(CONFIG['dataset_name'], train_set, test_set, y_test, train_size = 0.8, sensor_names = sensor_names, seed =CONFIG['seed'],
            layer_sizes = [128, 64], sequence_length = 30, epochs = 20, batch_size = 32, l2_param = 0.1, activation_function = 'sigmoid',              
            apply_scaling=0, apply_condition_scaling = 1, scaling_method='z-score', apply_clipping = 1, clipping_threshold = 125,
            print_results = 1, export_data = 0)

    if CONFIG['run_GRU']: 
        print("\n=== Running GRU ===")

        if CONFIG['random_search']:

            TN.random_search_GRU(n_iter = 50, dataset_name=CONFIG['dataset_name'], train_set = train_set, test_set = test_set, y_test = y_test, train_size = 0.8, sensor_names = sensor_names, seed = CONFIG['seed'],              
                apply_scaling=0, apply_condition_scaling = 1, scaling_method='z-score', apply_clipping = 1, clipping_threshold = 125,
                ) # utils  

        else:
            rmse_train, r2_train, rmse_dev, r2_dev, computation_time = MD.run_GRU(CONFIG['dataset_name'], train_set, test_set, y_test, train_size = 0.8, sensor_names = sensor_names, seed =CONFIG['seed'],
            layer_sizes = [16], sequence_length = 30, epochs = 5, batch_size = 32, l2_param = 0.1, activation_function = 'relu',               
            apply_scaling=0, apply_condition_scaling = 1, scaling_method='z-score', apply_clipping = 1, clipping_threshold = 125,
            print_results = 1, export_data = 0)

    if CONFIG['run_CNN1D']: 
        print("\n=== Running CNN1D ===")

        if CONFIG['random_search']:
            TN.random_search_CNN1D(n_iter = 50, dataset_name=CONFIG['dataset_name'], train_set = train_set, test_set = test_set, y_test = y_test, train_size = 0.8, sensor_names = sensor_names, seed = CONFIG['seed'],              
                apply_scaling=0, apply_condition_scaling = 1, scaling_method='z-score', apply_clipping = 1, clipping_threshold = 125, 
                )  

        else:
            rmse_train, r2_train, rmse_dev, r2_dev, computation_time = MD.run_CNN1D(dataset_name=CONFIG['dataset_name'], train_set = train_set, test_set = test_set, y_test = y_test, train_size = 0.8, sensor_names = sensor_names, seed = CONFIG['seed'],
            CNN_layer_sizes = [64,128], kernel_size = 2, pool_size = 2, dense_units = 200, sequence_length = 50, epochs = 10, batch_size = 32, l2_param = 0.001, conv_activation_function = 'sigmoid', dense_activation_function = 'relu', 
            apply_scaling=0, apply_condition_scaling=1, scaling_method='z-score', apply_clipping=1, clipping_threshold=125, 
            print_results = 1, export_data = 0, use_asymmetric_loss=1)


    if CONFIG['run_RF_regression']:
        print("\n=== Running Random Forest Regression ===")
        if CONFIG['random_search']:
            TN.random_search_RF(n_iter = 60, dataset_name=CONFIG['dataset_name'], train_set = train_set, test_set = test_set, y_test = y_test, train_size = 0.8, sensor_names = sensor_names, seed = CONFIG['seed'],              
                apply_scaling=1, scaling_method='z-score', apply_clipping = 1, clipping_threshold = 125,
                ) 
            
        else:
            rmse_train, r2_train, rmse_dev, r2_dev, computation_time = MD.run_random_forest(CONFIG['dataset_name'], train_set, test_set, y_test, train_size = 0.8,  sensor_names = sensor_names, seed = CONFIG['seed'], 
                    n_estimators = 100, max_depth = 10, min_samples_leaf = 6, add_lagged_vars = 0,
                    apply_scaling = 0,  apply_condition_scaling=1,scaling_method = 'z-score', apply_clipping = 1,clipping_threshold = 125, 
                    print_results = 1, export_data=0) 

    if CONFIG['run_XGBoost_regression']:
        print("\n=== Running XGBoost Regression ===")
        if CONFIG['random_search']:
            TN.random_search_XGB(n_iter = 60, dataset_name=CONFIG['dataset_name'], train_set = train_set, test_set = test_set, y_test = y_test, train_size = 0.8, sensor_names = sensor_names, seed = CONFIG['seed'],              
                apply_scaling=1, scaling_method='z-score', apply_clipping = 1, clipping_threshold = 125,
                ) 
            
        else:
            rmse_train, r2_train, rmse_dev, r2_dev, computation_time = MD.run_XGboost(CONFIG['dataset_name'], train_set, test_set, y_test, train_size = 0.8,  sensor_names = sensor_names, seed = CONFIG['seed'], 
                    n_estimators = 300, max_depth = 3, min_samples_leaf = None, add_lagged_vars = 1, 
                    apply_scaling = 0, apply_condition_scaling=1, scaling_method = 'z-score', apply_clipping = 1, clipping_threshold = 125, 
                    print_results = 1, export_data = 0) 

    #show tuning results
    # df = pd.read_csv('outputs/results_mlp', delimiter=',')
    # df_sorted = df.sort_values(by='val_RMSE', ascending=True)
    # print(df_sorted.head(20)) 

    return

if __name__ == "__main__":
    final_results = main()
 

