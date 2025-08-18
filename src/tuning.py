import random
import numpy as np
from src import models as MD
import pandas as pd
import time
from src import utils as UT
import datetime

# --- HYPERPARAMETER TUNING FUNCTIONS ---#

def random_search_MLP(n_iter, dataset_name, train_set, test_set, y_test, train_size, sensor_names, seed,              
            apply_scaling=0,apply_condition_scaling = 0, scaling_method='z-score', apply_clipping = 0, clipping_threshold = 125,
            verbose = 1):
    
    start_time = time.time()

    epochs_list = [10,20,30]
    layer_size_list = [[16,32,64],[64,32,16], [128, 64,32], [32,64,128], [32, 64]]
    batch_size_list = [32, 64, 128]
    lags_list = [0,1] 
    l2_list = [None, 0.001, 0.01, 0.1]
    activation_function_list = ['tanh','sigmoid','relu']

    for i in range(n_iter):
        
        print(f"Iter {i+1}/{n_iter} ...")

        epochs = random.choice(epochs_list)
        layer_sizes = random.choice(layer_size_list)
        batch_size = random.choice(batch_size_list)
        l2_param = random.choice(l2_list)
        lags = random.choice(lags_list)
        activation_function = random.choice(activation_function_list)

        train_RMSE, train_R2, val_RMSE, val_R2, computation_time = MD.run_MLP(dataset_name, train_set, test_set, y_test, train_size, sensor_names, seed,
            layer_sizes = layer_sizes, epochs = epochs, batch_size = batch_size, l2_param = l2_param, add_lagged_vars = lags, activation_function = activation_function,              
            apply_scaling=apply_scaling, apply_condition_scaling= apply_condition_scaling,scaling_method=scaling_method, apply_clipping = apply_clipping, clipping_threshold = clipping_threshold,
           verbose = verbose) 
        
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        res = {
            'timestamp': timestamp,
            'dataset': dataset_name,
            'train_RMSE': train_RMSE, 
            'train_R2': train_R2,
            'val_RMSE': val_RMSE,
            'val_R2': val_R2,
            'computation_time': computation_time,
            'layer_sizes': layer_sizes, 
            'epochs': epochs,
            'l2_param': l2_param if l2_param != None else "no",
            'batch_size': batch_size,
            'lags': lags,
            'activation': activation_function,
            'scaling': scaling_method if apply_scaling != 0 else "no", 
            'clipping_threshold': clipping_threshold if apply_clipping != 0 else "no"
        }

        UT.save_results('outputs/results_mlp', res)

    print(f'total computation time: {time.time()-start_time}')

    return

def random_search_LSTM(n_iter, dataset_name, train_set, test_set, y_test, train_size, sensor_names, seed,
                       apply_scaling=0, apply_condition_scaling = 0, scaling_method='z-score', apply_clipping=0, clipping_threshold=125,
                       verbose=1):
    start_time = time.time()

    epochs_list = [5,10,20]
    layer_size_list = [[16],  [32], [16, 32], [64, 32], [32, 64], [64, 128], [128, 64]]
    batch_size_list = [32, 64, 128]
    sequence_length_list = [10, 20, 30] 
    l2_list = [None, 0.001, 0.01, 0.1]
    activation_function_list = ['tanh','sigmoid','relu']


    for i in range(n_iter):
        print(f"Iter {i+1}/{n_iter} ...")

        epochs = random.choice(epochs_list)
        layer_sizes = random.choice(layer_size_list)
        batch_size = random.choice(batch_size_list)
        sequence_length = random.choice(sequence_length_list)
        l2_param = random.choice(l2_list)
        activation_function = random.choice(activation_function_list)

        train_RMSE, train_R2, val_RMSE, val_R2, computation_time = MD.run_LSTM(
            dataset_name, train_set, test_set, y_test, train_size, sensor_names, seed,
            layer_sizes, sequence_length, epochs, batch_size, l2_param, activation_function,
            apply_scaling=apply_scaling, apply_condition_scaling = apply_condition_scaling, scaling_method=scaling_method, apply_clipping = apply_clipping, clipping_threshold = clipping_threshold, # hyperparameters that have been tested manually
            verbose = verbose)
        
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        res = {
            'timestamp': timestamp,
            'dataset': dataset_name,
            'train_RMSE': train_RMSE, 
            'train_R2': train_R2,
            'val_RMSE': val_RMSE,
            'val_R2': val_R2,
            'computation_time': computation_time,
            'LSTM_layer_sizes': layer_sizes, 
            'epochs': epochs,
            'l2_param': l2_param if l2_param != None else "no",
            'batch_size': batch_size,
            'sequence_length': sequence_length,
            'activation': activation_function,
            'scaling': scaling_method if (apply_scaling != 0 or apply_condition_scaling != 0) else "no", 
            'clipping_threshold': clipping_threshold if apply_clipping != 0 else "no"
        }

        UT.save_results('outputs/results_lstm', res)

    total_time = time.time() - start_time
    print(f'total computation time: {total_time:.2f} seconds')

    return

def random_search_GRU(n_iter, dataset_name, train_set, test_set, y_test, train_size, sensor_names, seed,
                       apply_scaling=0, apply_condition_scaling = 0, scaling_method='z-score', apply_clipping=0, clipping_threshold=125,
                       verbose=1):
    start_time = time.time()

    epochs_list = [5,10,20]
    layer_size_list = [[16],  [32], [16, 32], [64, 32], [32, 64], [64, 128], [128, 64]]
    batch_size_list = [32, 64, 128]
    sequence_length_list = [10, 20, 30] 
    l2_list = [None, 0.001, 0.01, 0.1]
    activation_function_list = ['tanh','sigmoid','relu']

    for i in range(n_iter):
        print(f"Iter {i+1}/{n_iter} ...")

        epochs = random.choice(epochs_list)
        layer_sizes = random.choice(layer_size_list)
        batch_size = random.choice(batch_size_list)
        sequence_length = random.choice(sequence_length_list)
        l2_param = random.choice(l2_list)
        activation_function = random.choice(activation_function_list)

        train_RMSE, train_R2, val_RMSE, val_R2, computation_time = MD.run_GRU(
            dataset_name, train_set, test_set, y_test, train_size, sensor_names, seed,
            layer_sizes, sequence_length, epochs, batch_size, l2_param, activation_function,
            apply_scaling=apply_scaling, apply_condition_scaling = apply_condition_scaling, scaling_method=scaling_method, apply_clipping = apply_clipping, clipping_threshold = clipping_threshold, # hyperparameters that have been tested manually
            verbose = verbose)
        
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        res = {
            'timestamp': timestamp,
            'dataset': dataset_name,
            'train_RMSE': train_RMSE, 
            'train_R2': train_R2,
            'val_RMSE': val_RMSE,
            'val_R2': val_R2,
            'computation_time': computation_time,
            'GRU_layer_sizes': layer_sizes, 
            'epochs': epochs,
            'l2_param': l2_param if l2_param != None else "no",
            'batch_size': batch_size,
            'sequence_length': sequence_length,
            'activation': activation_function,
            'scaling': scaling_method if (apply_scaling != 0 or apply_condition_scaling != 0) else "no", 
            'clipping_threshold': clipping_threshold if apply_clipping != 0 else "no"
        }

        UT.save_results('outputs/results_gru', res)

    total_time = time.time() - start_time
    print(f'total computation time: {total_time:.2f} seconds')

    return

def random_search_CNN1D(n_iter, dataset_name, train_set, test_set, y_test, train_size, sensor_names, seed,
                        apply_scaling=0, apply_condition_scaling=0, scaling_method='z-score', apply_clipping=0, clipping_threshold=125,
                        verbose=1):

    start_time = time.time()
    
    CNN_layer_sizes_list = [[16, 32], [32, 16], [32,64], [64,128], [32, 64,128], [128, 64, 32]] 
    kernel_size_list = [2, 3, 5] 
    pool_size_list = [2, 3]
    dense_units_list = [20, 50, 100, 200]
    l2_list = [None, 0.0001, 0.001, 0.01]
    activation_conv_list = ['relu', 'tanh', 'sigmoid']
    activation_dense_list = ['relu', 'tanh', 'sigmoid']
    batch_size_list = [16, 32, 64, 128]
    epochs_list = [5,7,10,20]
    sequence_length_list = [20, 30, 50]

    for i in range(n_iter):
        print(f"Iter {i+1}/{n_iter} ...")

        CNN_layer_sizes = random.choice(CNN_layer_sizes_list)
        kernel_size = random.choice(kernel_size_list)
        pool_size = random.choice(pool_size_list)
        dense_units = random.choice(dense_units_list)
        l2_param = random.choice(l2_list)
        activation_conv = random.choice(activation_conv_list)
        activation_dense = random.choice(activation_dense_list)
        batch_size = random.choice(batch_size_list)
        epochs = random.choice(epochs_list)
        sequence_length = random.choice(sequence_length_list)

        train_RMSE, train_R2, val_RMSE, val_R2, computation_time = MD.run_CNN1D(dataset_name, train_set, test_set, y_test, train_size, sensor_names, seed, 
            CNN_layer_sizes = CNN_layer_sizes, kernel_size = kernel_size, pool_size = pool_size, dense_units = dense_units, sequence_length = sequence_length, epochs = epochs,  # ...
            batch_size = batch_size, l2_param = l2_param, conv_activation_function = activation_conv, dense_activation_function = activation_dense, # hyperparameters that will be tested with RandomSearch  
            apply_scaling=apply_scaling, apply_condition_scaling=apply_condition_scaling, scaling_method=scaling_method, apply_clipping=apply_clipping, clipping_threshold=clipping_threshold, # hyperparameters that will be tested manually
            verbose = verbose) 

        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        res = {
            'timestamp': timestamp,
            'dataset': dataset_name,
            'train_RMSE': train_RMSE, 
            'train_R2': train_R2,
            'val_RMSE': val_RMSE,
            'val_R2': val_R2,
            'computation_time': computation_time,
            'CNN_layer_sizes': CNN_layer_sizes, 
            'kernel_sizes': kernel_size,
            'pool_sizes': pool_size,
            'dense_units': dense_units,
            'epochs': epochs,
            'batch_size': batch_size,
            'sequence_length': sequence_length,
            'l2_param': l2_param if l2_param != None else "no",
            'activation_conv': activation_conv,
            'activation_dense': activation_dense,
            'scaling': scaling_method if (apply_scaling != 0 or apply_condition_scaling != 0) else "no",
            'clipping_threshold': clipping_threshold if apply_clipping != 0 else "no"
        }

        UT.save_results('outputs/results_cnn1d', res)

    total_time = time.time() - start_time
    print(f'Total computation time: {total_time:.2f} seconds')

    return

def random_search_RF(n_iter, dataset_name, train_set, test_set, y_test, train_size, sensor_names, seed,              
                     apply_scaling=0, apply_condition_scaling = 0, scaling_method='z-score', apply_clipping=0, clipping_threshold=125):

    start_time = time.time()

    n_estimators_list = [50, 100, 200, 300]
    max_depth_list = [None, 3, 5, 10]
    min_samples_leaf_list = [None, 1, 2, 4, 6, 8, 10]
    lags_list = [0, 1] 

    for i in range(n_iter):
        
        print(f"Iter {i+1}/{n_iter} ...")

        n_estimators = random.choice(n_estimators_list)
        max_depth = random.choice(max_depth_list)
        min_samples_leaf = random.choice(min_samples_leaf_list)
        lags = random.choice(lags_list)

        train_RMSE, train_R2, val_RMSE, val_R2, computation_time = MD.run_random_forest(
            dataset_name=dataset_name,
            train_set=train_set,
            test_set=test_set,
            y_test=y_test,
            n_estimators=n_estimators,
            train_size=train_size,
            seed=seed,
            sensor_names=sensor_names,
            apply_scaling=apply_scaling,
            apply_condition_scaling= apply_condition_scaling,
            scaling_method=scaling_method,
            apply_clipping=apply_clipping,
            clipping_threshold=clipping_threshold,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            add_lagged_vars=lags,
            
        )
        
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        res = {
            'timestamp': timestamp,
            'dataset': dataset_name,
            'train_RMSE': train_RMSE, 
            'train_R2': train_R2,
            'val_RMSE': val_RMSE,
            'val_R2': val_R2,
            'computation_time': computation_time,
            'n_estimators': n_estimators,
            'max_depth': max_depth if max_depth != None else "no",
            'min_samples_leaf': min_samples_leaf if min_samples_leaf != None else "no",
            'lags': lags,
            'scaling': scaling_method if (apply_scaling != 0 or apply_condition_scaling != 0) else "no", 
            'clipping_threshold': clipping_threshold if apply_clipping != 0 else "no"
        }

        UT.save_results('outputs/results_rf', res)      

    print(f'total computation time: {time.time()-start_time}')

    return

def random_search_XGB(n_iter, dataset_name, train_set, test_set, y_test, train_size, sensor_names, seed,              
                     apply_scaling=0, apply_condition_scaling = 0,scaling_method='z-score', apply_clipping=0, clipping_threshold=125):

    start_time = time.time()

    n_estimators_list = [50, 100, 200, 300]
    max_depth_list = [None, 3, 5, 10]
    min_samples_leaf_list = [None, 1, 2, 4, 6, 8, 10]
    lags_list = [0, 1] 

    for i in range(n_iter):
        
        print(f"Iter {i+1}/{n_iter} ...")

        n_estimators = random.choice(n_estimators_list)
        max_depth = random.choice(max_depth_list)
        min_samples_leaf = random.choice(min_samples_leaf_list)
        lags = random.choice(lags_list)

        train_RMSE, train_R2, val_RMSE, val_R2, computation_time = MD.run_XGboost(
            dataset_name=dataset_name,
            train_set=train_set,
            test_set=test_set,
            y_test=y_test,
            n_estimators=n_estimators,
            train_size=train_size,
            seed=seed,
            sensor_names=sensor_names,
            apply_scaling=apply_scaling,
            apply_condition_scaling= apply_condition_scaling,
            scaling_method=scaling_method,
            apply_clipping=apply_clipping,
            clipping_threshold=clipping_threshold,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            add_lagged_vars=lags
        )

        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        res = {
            'timestamp': timestamp,
            'dataset': dataset_name,
            'train_RMSE': train_RMSE, 
            'train_R2': train_R2,
            'val_RMSE': val_RMSE,
            'val_R2': val_R2,
            'computation_time': computation_time,
            'n_estimators': n_estimators, 
            'max_depth': max_depth if max_depth != None else "no",
            'min_samples_leaf': min_samples_leaf if min_samples_leaf != None else "no",
            'lags': lags,
            'scaling': scaling_method if (apply_scaling != 0 or apply_condition_scaling != 0) else "no", 
            'clipping_threshold': clipping_threshold if apply_clipping != 0 else "no"
        }

        UT.save_results('outputs/results_xgb', res)      

    print(f'total computation time: {time.time()-start_time}')

    return
