# --- LIBRARIES ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from datetime import datetime
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.tree import plot_tree
import time
import os
from sklearn.model_selection import GroupShuffleSplit
import tensorflow as tf
from keras.models import Sequential
from keras.layers import GRU, LSTM, Dense, Dropout, Masking, TimeDistributed, Conv1D,  MaxPooling1D, Flatten
from keras.losses import MeanSquaredError
import xgboost as xgb
from keras.regularizers import l2

from src import feature_engineering as FE
from src import data_processing as DP
from src import utils as UT
import datetime

# ---MODEL FUNCTIONS ---
def run_linear_regression(dataset_name, train_set, test_set, y_test, sensor_names, seed, train_size = 0.8, 
                        apply_scaling = 0, apply_condition_scaling = 0, scaling_method = 'z-score', apply_clipping= 0, clipping_threshold = None, apply_polynomial_features = 0, poly_degree = 2, 
                        drop_useless_sensors = 0, add_lagged_vars = 0, print_results = 0, export_data = 0):
    
    start_time = time.time()
    train_set_copy = train_set.copy()
    test_set_copy = test_set.copy()

    if drop_useless_sensors != 0:
        train_set_copy, selected_sensors, excluded_sensors = FE.drop_sensors(train_set_copy, sensor_names)
        test_set_copy = test_set_copy.drop(excluded_sensors, axis=1)
    else:
        selected_sensors = sensor_names

    if add_lagged_vars != 0:
        train_set_copy = FE.add_specific_lags(train_set_copy, selected_sensors)
        test_set_copy = FE.add_specific_lags(test_set_copy, selected_sensors)

    test_set_copy = test_set.groupby('unit_number').last().reset_index() 
    
    if apply_clipping!=0: 
        train_set_copy = FE.clip_rul_values(train_set_copy, upper_threshold=clipping_threshold) 

    train_set_copy = DP.add_operating_condition(train_set_copy)
    test_set_copy = DP.add_operating_condition(test_set_copy)

    if apply_scaling != 0: 
        train_set_copy, test_set_copy = FE.scale_features(train_set_copy, test_set_copy, selected_sensors,method = scaling_method)
    elif apply_condition_scaling != 0:
        train_set_copy, test_set_copy = FE.scale_features_condition(train_set_copy, test_set_copy, selected_sensors, method = scaling_method)
        apply_condition_scaling = 1
    
   
    if apply_polynomial_features != 0: 
        train_set_copy, test_set_copy, selected_sensors = FE.create_polynomial_features(train_set_copy, test_set_copy, selected_sensors, poly_degree) 

    gss = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=seed)
    train_set_copy, y_train, dev_set, _= DP.split_devSet_trainSet(train_set_copy, test_set_copy, gss, groups = train_set_copy['unit_number'], print_groups=False) 
    dev_set_copy = dev_set.copy()

    X_train, X_test, X_dev, y_train, y_dev = DP.cleaning_data(train_set_copy, test_set_copy, selected_sensors, dataset_name, df_dev = dev_set_copy)

    model = LinearRegression()
    model.fit(X_train, y_train)

    if apply_polynomial_features != 0:
        select_features = SelectFromModel(model, threshold='mean', prefit=True)
        select_features.get_support()
        
        X_train = select_features.transform(X_train.values)
        X_dev = select_features.transform(X_dev.values)
        X_test = select_features.transform(X_test.values)

        model.fit(X_train, y_train.values.flatten())
        
    y_hat_train = model.predict(X_train)
    y_hat_dev = model.predict(X_dev)
    y_hat_test = model.predict(X_test)

    computation_time = time.time() - start_time
   
    if print_results:
        plt.figure(figsize=(7,7))
        plt.scatter(y_test, y_hat_test, alpha=0.5, edgecolors='k')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2) 
        plt.xlabel("Real values")
        plt.ylabel("Predicted vaues")
        plt.title(f"Scatter plot: Real values vs predicted values - {dataset_name}")
        plt.grid(True)
        plt.show()
        print(f"Results of a linear regression applied to dataset {dataset_name} | Time required to complete the computation: {computation_time}")
        print("\n")
    signed_error_train, rmse_train, r2_train = UT.evaluate(y_train, y_hat_train, "Train", print_results)
    signed_error_test, rmse_test, r2_test = UT.evaluate(y_test.values.flatten(), y_hat_test, "Test", print_results)
    signed_error_dev, rmse_dev, r2_dev = UT.evaluate(y_dev.values.flatten(), y_hat_dev, "Cross Validation", print_results)


    if export_data:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        res = {
            'timestamp': timestamp,
            'dataset': dataset_name,
            'train_RMSE': rmse_train, 
            'train_R2': r2_train,
            'val_RMSE': rmse_dev,
            'val_R2': r2_dev,
            'computation_time': computation_time,
            'scaling': scaling_method if apply_scaling != 0 else "no",
            'clipping_threshold': clipping_threshold if apply_clipping != 0 else "no",
            'polynomial_features': f"max degree {poly_degree}" if apply_polynomial_features else "no",
            'lags': add_lagged_vars
        }

        UT.save_results('outputs/results_lr', res)
    
    return rmse_train, r2_train, rmse_dev, r2_dev, computation_time

def run_MLP(dataset_name, train_set, test_set, y_test, train_size, sensor_names, seed,
            layer_sizes, epochs = 20, batch_size = 32, l2_param = None, add_lagged_vars = 0, activation_function = 'relu',              
            apply_scaling=0, apply_condition_scaling = 0, scaling_method='z-score', apply_clipping = 0, clipping_threshold = 125, 
            plot_history=False, verbose=1, print_results = 0, drop_useless_sensors = 0,  use_asymmetric_loss = 0, export_data = 0): 
    
    start_time = time.time()

    train_set_copy = train_set.copy()
    test_set_copy = test_set.copy()

    if drop_useless_sensors != 0:
        train_set_copy, selected_sensors, excluded_sensors = FE.drop_sensors(train_set_copy, sensor_names)
        test_set_copy = test_set_copy.drop(excluded_sensors, axis=1)
    else:
        selected_sensors = sensor_names

    if add_lagged_vars != 0:
        train_set_copy = FE.add_specific_lags(train_set_copy, selected_sensors)
        test_set_copy = FE.add_specific_lags(test_set_copy, selected_sensors)
     
    test_set_copy = test_set.groupby('unit_number').last().reset_index()
 
    if apply_clipping!=0:
        train_set_copy = FE.clip_rul_values(train_set_copy, upper_threshold=clipping_threshold)

    train_set_copy = DP.add_operating_condition(train_set_copy)
    test_set_copy = DP.add_operating_condition(test_set_copy)

    if apply_scaling != 0:
        train_set_copy, test_set_copy = FE.scale_features(train_set_copy, test_set_copy, selected_sensors,method = scaling_method) 

    

    elif apply_condition_scaling != 0:
        train_set_copy, test_set_copy = FE.scale_features_condition(train_set_copy, test_set_copy, selected_sensors, method = scaling_method)
        apply_condition_scaling = 1

    gss = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=seed)
    train_set_copy, y_train, dev_set, _= DP.split_devSet_trainSet(train_set_copy, test_set_copy, gss, groups = train_set_copy['unit_number'], print_groups=False)
    dev_set_copy = dev_set.copy()

    X_train, X_test, X_dev, y_train, y_dev = DP.cleaning_data(train_set_copy, test_set_copy, selected_sensors, dataset_name, df_dev = dev_set_copy)

    input_dim = X_train.shape[1]

    model = Sequential()

    model.add(Dense(layer_sizes[0], input_dim=input_dim, activation=activation_function, kernel_regularizer=l2(l2_param) if l2_param!= None else None))

    for size in layer_sizes[1:]:
        model.add(Dense(size, activation=activation_function, kernel_regularizer=l2(l2_param) if l2_param!= None else None))

    model.add(Dense(1))

    if use_asymmetric_loss:
        loss_function = lambda y_true, y_pred: UT.asymmetric_mse(y_true, y_pred, penalty_factor=2) 
    else:
        loss_function = 'mean_squared_error'

    model.compile(loss=loss_function, optimizer='adam')

    history = model.fit(
        X_train, y_train,
        validation_data=(X_dev, y_dev),
        epochs=epochs,
        batch_size = batch_size,
        verbose=verbose
    )

    if plot_history:
        plt.figure(figsize=(13,5))
        plt.plot(history.history['loss'], label='Train Loss')
        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss Progress')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    y_hat_train = model.predict(X_train)
    y_hat_test = model.predict(X_test)
    y_hat_dev = model.predict(X_dev)

    computation_time = time.time() - start_time
    
    if print_results:
        plt.figure(figsize=(7,7))
        plt.scatter(y_test, y_hat_test, alpha=0.5, edgecolors='k')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2) 
        plt.xlabel("Valori Reali (y_test)")
        plt.ylabel("Valori Predetti (y_hat_test)")
        plt.title(f"Scatter plot: Valori Reali vs Predetti - {dataset_name}")
        plt.grid(True)
        plt.show()

        print(f"Results of a LSTM applied to dataset {dataset_name} | Time required to complete the computation: {computation_time}")
        print("\n")

    signed_error_train, rmse_train, r2_train = UT.evaluate(y_train, y_hat_train, "Train", print_results)
    signed_error_test, rmse_test, r2_test = UT.evaluate(y_test.values.flatten(), y_hat_test, "Test", print_results)
    signed_error_dev, rmse_dev, r2_dev = UT.evaluate(y_dev.values.flatten(), y_hat_dev, "Cross Validation", print_results)


    if export_data:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        res = {
            'timestamp': timestamp,
            'dataset': dataset_name,
            'train_RMSE': rmse_train, 
            'train_R2': r2_train,
            'val_RMSE': rmse_dev,
            'val_R2': r2_dev,
            'computation_time': computation_time,
            'layer_sizes': layer_sizes, 
            'epochs': epochs,
            'l2_param': l2_param if l2_param != None else "no",
            'batch_size': batch_size,
            'lags': add_lagged_vars,
            'activation': activation_function,
            'scaling': scaling_method if apply_scaling != 0 else "no", 
            'clipping_threshold': clipping_threshold if apply_clipping != 0 else "no"
        }

        UT.save_results('outputs/results_mlp', res)

    return rmse_train, r2_train, rmse_dev, r2_dev, computation_time

def run_LSTM(dataset_name, train_set, test_set, y_test, train_size, sensor_names, seed,
            layer_sizes, sequence_length, epochs = 20, batch_size = 32, l2_param = None, activation_function = 'relu',              
            apply_scaling=0, apply_condition_scaling = 0, scaling_method='z-score', apply_clipping = 0, clipping_threshold = 125,
            plot_history=False, verbose=1, print_results = 0, drop_useless_sensors = 0,  use_asymmetric_loss = 0, export_data = 0):

    start_time = time.time()

    train_set_copy = train_set.copy()
    test_set_copy = test_set.copy()

    if drop_useless_sensors != 0:
        train_set_copy, selected_sensors, excluded_sensors = FE.drop_sensors(train_set_copy, sensor_names)
        test_set_copy = test_set_copy.drop(excluded_sensors, axis=1)
    else:
        selected_sensors = sensor_names

    if apply_clipping != 0:
        train_set_copy = FE.clip_rul_values(train_set_copy, upper_threshold=clipping_threshold)

    train_set_copy = DP.add_operating_condition(train_set_copy)
    test_set_copy = DP.add_operating_condition(test_set_copy)

    if apply_scaling != 0:
        train_set_copy, test_set_copy = FE.scale_features(train_set_copy, test_set_copy, selected_sensors, method=scaling_method)
        apply_scaling = 1
    elif apply_condition_scaling != 0:
        train_set_copy, test_set_copy = FE.scale_features_condition(train_set_copy, test_set_copy, selected_sensors, method = scaling_method)
        apply_condition_scaling = 1       

    gss = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=seed)
    for train_unit, val_unit in gss.split(train_set_copy['unit_number'].unique(), groups=train_set_copy['unit_number'].unique()):
        train_unit = train_set_copy['unit_number'].unique()[train_unit]  
        val_unit = train_set_copy['unit_number'].unique()[val_unit]

        train_array = DP.generate_train_sequence(train_set_copy, sequence_length, selected_sensors, train_unit)
        train_label_array = DP.generate_lables(train_set_copy, sequence_length, ['RUL'], train_unit)
        
        val_array = DP.generate_train_sequence(train_set_copy, sequence_length, selected_sensors, val_unit)
        val_label_array = DP.generate_lables(train_set_copy, sequence_length, ['RUL'], val_unit)

    generated_test_sequence = (list(DP.generate_test_data(test_set_copy[test_set_copy['unit_number']==unit_number], sequence_length, selected_sensors, -99.))
           for unit_number in test_set_copy['unit_number'].unique())
    test_array = np.concatenate(list(generated_test_sequence)).astype(np.float32)

    model = Sequential()
    model.add(Masking(mask_value=-99., input_shape=(sequence_length, train_array.shape[2])))

    for i, size in enumerate(layer_sizes):

        return_seq = True if i < len(layer_sizes) - 1 else False

        model.add(LSTM(size, activation=activation_function,
                   kernel_regularizer=l2(l2_param) if l2_param is not None else None,
                   recurrent_regularizer=l2(l2_param) if l2_param is not None else None,
                   return_sequences=return_seq))
    model.add(Dense(1))
    
    if use_asymmetric_loss:
        loss_function = lambda y_true, y_pred: UT.asymmetric_mse(y_true, y_pred, penalty_factor=2) 
    else:
        loss_function = 'mean_squared_error'

    model.compile(loss=loss_function, optimizer='adam')

    history = model.fit(train_array, train_label_array,
                        validation_data=(val_array, val_label_array),
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose = verbose)
    
    if plot_history:
        plt.figure(figsize=(13,5))
        plt.plot(history.history['loss'], label='Train Loss')
        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss Progress')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    y_hat_train = model.predict(train_array)
    y_hat_test = model.predict(test_array)
    y_hat_dev = model.predict(val_array)

    computation_time = time.time() - start_time

    if print_results:
        plt.figure(figsize=(7,7))
        plt.scatter(y_test, y_hat_test, alpha=0.5, edgecolors='k')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)  
        plt.xlabel("Valori Reali (y_test)")
        plt.ylabel("Valori Predetti (y_hat_test)")
        plt.title(f"Scatter plot: Valori Reali vs Predetti - {dataset_name}")
        plt.grid(True)
        plt.show()

        print(f"Results of a LSTM applied to dataset {dataset_name} | Time required to complete the computation: {computation_time}")
        print("\n")
    signed_error_train, rmse_train, r2_train = UT.evaluate(train_label_array, y_hat_train, "Train", print_results)
    signed_error_test, rmse_test, r2_test = UT.evaluate(y_test.values.flatten(), y_hat_test, "Test", print_results)
    signed_error_dev, rmse_dev, r2_dev = UT.evaluate(val_label_array, y_hat_dev, "Cross Validation", print_results)


    if export_data:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        res = {
            'timestamp': timestamp,
            'dataset': dataset_name,
            'train_RMSE': rmse_train,
            'train_R2': r2_train,
            'val_RMSE': rmse_dev,
            'val_R2': r2_dev,
            'computation_time': computation_time,
            'LSTM_layer_sizes': layer_sizes, 
            'epochs': epochs,
            'l2_param': l2_param if l2_param != None else "no",
            'batch_size': batch_size,
            'sequence_length': sequence_length,
            'activation': activation_function,
            'scaling': scaling_method if apply_scaling != 0 else "no",
            'clipping_threshold': clipping_threshold if apply_clipping != 0 else "no"
        }

        UT.save_results('outputs/results_lstm', res)
           
    return rmse_train, r2_train, rmse_dev, r2_dev, computation_time


def run_GRU(dataset_name, train_set, test_set, y_test, train_size, sensor_names, seed,
            layer_sizes, sequence_length, epochs = 20, batch_size = 32, l2_param = None, activation_function = 'relu',            
            apply_scaling=0, apply_condition_scaling = 0, scaling_method='z-score', apply_clipping = 0, clipping_threshold = 125, 
            plot_history=False, verbose=1, print_results = 0, drop_useless_sensors = 0,  use_asymmetric_loss = 0, export_data = 0):

    start_time = time.time()

    train_set_copy = train_set.copy()
    test_set_copy = test_set.copy()

    if drop_useless_sensors != 0:
        train_set_copy, selected_sensors, excluded_sensors = FE.drop_sensors(train_set_copy, sensor_names)
        test_set_copy = test_set_copy.drop(excluded_sensors, axis=1)
    else:
        selected_sensors = sensor_names

    if apply_clipping != 0:
        train_set_copy = FE.clip_rul_values(train_set_copy, upper_threshold=clipping_threshold)

    train_set_copy = DP.add_operating_condition(train_set_copy)
    test_set_copy = DP.add_operating_condition(test_set_copy)

    if apply_scaling != 0:
        train_set_copy, test_set_copy = FE.scale_features(train_set_copy, test_set_copy, selected_sensors, method=scaling_method)
        apply_scaling = 1
    elif apply_condition_scaling != 0:
        train_set_copy, test_set_copy = FE.scale_features_condition(train_set_copy, test_set_copy, selected_sensors, method = scaling_method)
        apply_condition_scaling = 1       

    gss = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=seed)
    for train_unit, val_unit in gss.split(train_set_copy['unit_number'].unique(), groups=train_set_copy['unit_number'].unique()):
        train_unit = train_set_copy['unit_number'].unique()[train_unit] 
        val_unit = train_set_copy['unit_number'].unique()[val_unit]

        train_array = DP.generate_train_sequence(train_set_copy, sequence_length, selected_sensors, train_unit)
        train_label_array = DP.generate_lables(train_set_copy, sequence_length, ['RUL'], train_unit)
        
        val_array = DP.generate_train_sequence(train_set_copy, sequence_length, selected_sensors, val_unit)
        val_label_array = DP.generate_lables(train_set_copy, sequence_length, ['RUL'], val_unit)

    generated_test_sequence = (list(DP.generate_test_data(test_set_copy[test_set_copy['unit_number']==unit_number], sequence_length, selected_sensors, -99.))
           for unit_number in test_set_copy['unit_number'].unique())
    test_array = np.concatenate(list(generated_test_sequence)).astype(np.float32)

    model = Sequential()
    model.add(Masking(mask_value=-99., input_shape=(sequence_length, train_array.shape[2])))

    for i, size in enumerate(layer_sizes):

        return_seq = True if i < len(layer_sizes) - 1 else False

        model.add(GRU(size, activation=activation_function,
                   kernel_regularizer=l2(l2_param) if l2_param is not None else None,
                   recurrent_regularizer=l2(l2_param) if l2_param is not None else None,
                   return_sequences=return_seq))
    model.add(Dense(1))
    
    if use_asymmetric_loss:
        loss_function = lambda y_true, y_pred: UT.asymmetric_mse(y_true, y_pred, penalty_factor=2) 
    else:
        loss_function = 'mean_squared_error'

    model.compile(loss=loss_function, optimizer='adam')

    history = model.fit(train_array, train_label_array,
                        validation_data=(val_array, val_label_array),
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose = verbose)
    
    if plot_history:
        plt.figure(figsize=(13,5))
        plt.plot(history.history['loss'], label='Train Loss')
        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss Progress')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    y_hat_train = model.predict(train_array)
    y_hat_test = model.predict(test_array)
    y_hat_dev = model.predict(val_array)

    computation_time = time.time() - start_time

    if print_results:
        plt.figure(figsize=(7,7))
        plt.scatter(y_test, y_hat_test, alpha=0.5, edgecolors='k')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)  
        plt.xlabel("Valori Reali (y_test)")
        plt.ylabel("Valori Predetti (y_hat_test)")
        plt.title(f"Scatter plot: Valori Reali vs Predetti - {dataset_name}")
        plt.grid(True)
        plt.show()

        print(f"Results of a GRU applied to dataset {dataset_name} | Time required to complete the computation: {computation_time}")
        print("\n")
    signed_error_train, rmse_train, r2_train = UT.evaluate(train_label_array, y_hat_train, "Train", print_results)
    signed_error_test, rmse_test, r2_test = UT.evaluate(y_test.values.flatten(), y_hat_test, "Test", print_results)
    signed_error_dev, rmse_dev, r2_dev = UT.evaluate(val_label_array, y_hat_dev, "Cross Validation", print_results)


    if export_data:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        res = {
            'timestamp': timestamp,
            'dataset': dataset_name,
            'train_RMSE': rmse_train, 
            'train_R2': r2_train,
            'val_RMSE': rmse_dev,
            'val_R2': r2_dev,
            'computation_time': computation_time,
            'GRU_layer_sizes': layer_sizes,
            'epochs': epochs,
            'l2_param': l2_param if l2_param != None else "no",
            'batch_size': batch_size,
            'sequence_length': sequence_length,
            'activation': activation_function,
            'scaling': scaling_method if apply_scaling != 0 else "no",
            'clipping_threshold': clipping_threshold if apply_clipping != 0 else "no"
        }

        UT.save_results('outputs/results_gru', res)
           
    return rmse_train, r2_train, rmse_dev, r2_dev, computation_time


def run_CNN1D(dataset_name, train_set, test_set, y_test, train_size, sensor_names, seed, 
            CNN_layer_sizes, kernel_size, pool_size = 2, dense_units = 50, sequence_length = 20, epochs = 10, batch_size = 32, l2_param = None, conv_activation_function = 'relu', dense_activation_function = 'relu', # hyperparameters that will be tested with RandomSearch  
            apply_scaling=0, apply_condition_scaling=0, scaling_method='z-score', apply_clipping=0, clipping_threshold=125,
            plot_history=0, verbose = 1, print_results = 0, drop_useless_sensors=0, use_asymmetric_loss = 0, export_data = 0): 

    start_time = time.time()

    train_set_copy = train_set.copy()
    test_set_copy = test_set.copy()

    if drop_useless_sensors != 0:
        train_set_copy, selected_sensors, excluded_sensors = FE.drop_sensors(train_set_copy, sensor_names)
        test_set_copy = test_set_copy.drop(excluded_sensors, axis=1)
    else:
        selected_sensors = sensor_names

    if apply_clipping != 0:
        train_set_copy = FE.clip_rul_values(train_set_copy, upper_threshold=clipping_threshold)

    train_set_copy = DP.add_operating_condition(train_set_copy)
    test_set_copy = DP.add_operating_condition(test_set_copy)

    if apply_scaling != 0:
        train_set_copy, test_set_copy = FE.scale_features(train_set_copy, test_set_copy, selected_sensors, method=scaling_method)
    elif apply_condition_scaling != 0:
        train_set_copy, test_set_copy = FE.scale_features_condition(train_set_copy, test_set_copy, selected_sensors, method=scaling_method)

    train_set_copy = DP.ema_filter(train_set_copy, selected_sensors)
    test_set_copy = DP.ema_filter(test_set_copy, selected_sensors)


    gss = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=seed)
    for train_unit, val_unit in gss.split(train_set_copy['unit_number'].unique(), groups=train_set_copy['unit_number'].unique()):
        train_unit = train_set_copy['unit_number'].unique()[train_unit]
        val_unit = train_set_copy['unit_number'].unique()[val_unit]

        train_array = DP.generate_train_sequence(train_set_copy, sequence_length, selected_sensors, train_unit)
        train_label_array = DP.generate_lables(train_set_copy, sequence_length, ['RUL'], train_unit)
        val_array = DP.generate_train_sequence(train_set_copy, sequence_length, selected_sensors, val_unit)
        val_label_array = DP.generate_lables(train_set_copy, sequence_length, ['RUL'], val_unit)

    generated_test_sequence = (list(DP.generate_test_data(test_set_copy[test_set_copy['unit_number']==unit_number], sequence_length, selected_sensors, -99.))
           for unit_number in test_set_copy['unit_number'].unique())
    test_array = np.concatenate(list(generated_test_sequence)).astype(np.float32) 
    
    model = Sequential()
    model.add(Masking(mask_value=-99., input_shape=(sequence_length, train_array.shape[2])))

    for size in CNN_layer_sizes:
        model.add(Conv1D(filters=size, kernel_size=kernel_size, activation=conv_activation_function, kernel_regularizer=l2(l2_param) if l2_param else None))   
    
    model.add(MaxPooling1D(pool_size=pool_size)) 
    model.add(Flatten()) 
    model.add(Dense(dense_units, activation=dense_activation_function, kernel_regularizer=l2(l2_param) if l2_param else None))
    model.add(Dense(1))

    if use_asymmetric_loss:
        loss_function = lambda y_true, y_pred: UT.asymmetric_mse(y_true, y_pred, penalty_factor=2) 
    else:
        loss_function = 'mean_squared_error'


    model.compile(loss='mean_squared_error', optimizer='adam')

    history = model.fit(train_array, train_label_array,
                        validation_data=(val_array, val_label_array),
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose = verbose)

    if plot_history:
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.legend()
        plt.show()

    y_hat_train = model.predict(train_array)
    y_hat_dev = model.predict(val_array)
    y_hat_test = model.predict(test_array)
    computation_time = time.time() - start_time

    if print_results:
        plt.figure(figsize=(7,7))
        plt.scatter(y_test, y_hat_test, alpha=0.5, edgecolors='k')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2) 
        plt.xlabel("Valori Reali (y_test)")
        plt.ylabel("Valori Predetti (y_hat_test)")
        plt.title(f"Scatter plot: Valori Reali vs Predetti - {dataset_name}")
        plt.grid(True)
        plt.show()

        print(f"Results of a Conv1d applied to dataset {dataset_name} | Time required to complete the computation: {computation_time}")
        print("\n")
    signed_error_train, rmse_train, r2_train = UT.evaluate(train_label_array, y_hat_train, "Train", print_results)
    signed_error_test, rmse_test, r2_test = UT.evaluate(y_test.values.flatten(), y_hat_test, "Test", print_results)
    signed_error_dev, rmse_dev, r2_dev = UT.evaluate(val_label_array, y_hat_dev, "Cross Validation", print_results)


    if export_data:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        res = {
            'timestamp': timestamp,
            'dataset': dataset_name,
            'train_RMSE': rmse_train, 
            'train_R2': r2_train,
            'val_RMSE': rmse_dev,
            'val_R2': r2_dev,
            'computation_time': computation_time,
            'CNN_layer_sizes': CNN_layer_sizes,
            'kernel_sizes': kernel_size,
            'pool_sizes': pool_size,
            'dense_units': dense_units,
            'epochs': epochs,
            'batch_size': batch_size,
            'sequence_length': sequence_length,
            'l2_param': l2_param if l2_param != None else "no",
            'activation_conv': conv_activation_function,
            'activation_dense': dense_activation_function,
            'scaling': scaling_method if apply_scaling != 0 else "no",
            'clipping_threshold': clipping_threshold if apply_clipping != 0 else "no"
        }

        UT.save_results('outputs/results_cnn1d', res)
           
    return rmse_train, r2_train, rmse_dev, r2_dev, computation_time


def run_random_forest(dataset_name, train_set, test_set, y_test, train_size,  sensor_names, seed, 
                    n_estimators, max_depth = None, min_samples_leaf = None, add_lagged_vars = 0, 
                    apply_scaling = 0, apply_condition_scaling = 0,scaling_method = 'z-score', apply_clipping = 0, clipping_threshold = 125, 
                    print_results = 0, drop_useless_sensors = 0, export_data = 0): 
    start_time = time.time()

    train_set_copy = train_set.copy()
    test_set_copy = test_set.copy()
    
    if drop_useless_sensors != 0:
        train_set_copy, selected_sensors, excluded_sensors = FE.drop_sensors(train_set_copy, sensor_names)
        test_set_copy = test_set_copy.drop(excluded_sensors, axis=1)
    else:
        selected_sensors = sensor_names

    if add_lagged_vars != 0:
        X_train = FE.add_specific_lags(train_set, selected_sensors)
        X_test = FE.add_specific_lags(test_set, selected_sensors)
    
    test_set_copy = test_set.groupby('unit_number').last().reset_index()
    
    if apply_clipping!=0:
        train_set_copy = FE.clip_rul_values(train_set_copy, upper_threshold=clipping_threshold) 

    train_set_copy = DP.add_operating_condition(train_set_copy)
    test_set_copy = DP.add_operating_condition(test_set_copy)

    if apply_scaling != 0:
        train_set_copy, test_set_copy = FE.scale_features(train_set_copy, test_set_copy, selected_sensors,method = scaling_method) 
    elif apply_condition_scaling != 0:
        train_set_copy, test_set_copy = FE.scale_features_condition(train_set_copy, test_set_copy, selected_sensors, method = scaling_method)
        apply_condition_scaling = 1

    gss = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=seed)
    train_set_copy, y_train, dev_set, _ = DP.split_devSet_trainSet(train_set_copy, test_set_copy, gss, groups = train_set['unit_number'], print_groups=False) 
    dev_set_copy = dev_set.copy()

    X_train, X_test, X_dev, y_train, y_dev = DP.cleaning_data(train_set_copy, test_set_copy, selected_sensors, dataset_name, df_dev = dev_set_copy)

    rf_params = {
    'n_estimators': n_estimators,
    'max_features': "sqrt",
    'random_state': seed,
    **({'max_depth': max_depth} if max_depth is not None else {}),
    **({'min_samples_leaf': min_samples_leaf} if min_samples_leaf is not None else {})
    }
    rf = RandomForestRegressor(**rf_params)
    
    rf.fit(X_train, y_train)

    y_hat_train = rf.predict(X_train)
    y_hat_test = rf.predict(X_test)
    y_hat_dev = rf.predict(X_dev)

    computation_time = time.time() - start_time

    if print_results:
        plt.figure(figsize=(7,7))
        plt.scatter(y_test, y_hat_test, alpha=0.5, edgecolors='k')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)  
        plt.xlabel("Valori Reali (y_test)")
        plt.ylabel("Valori Predetti (y_hat_test)")
        plt.title(f"Scatter plot: Valori Reali vs Predetti - {dataset_name}")
        plt.grid(True)
        plt.show()

        print(f"Results of a random forest regressor applied to dataset {dataset_name} | Time required to complete the computation: {computation_time}")
        print("\n") 
    signed_error_train, rmse_train, r2_train = UT.evaluate(y_train, y_hat_train, "Train", print_results)
    signed_error_test, rmse_test, r2_test = UT.evaluate(y_test.values.flatten(), y_hat_test, "Test", print_results)
    signed_error_dev, rmse_dev, r2_dev = UT.evaluate(y_dev, y_hat_dev, "Cross Validation", print_results)  

    if export_data:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        res = {
            'timestamp': timestamp,
            'dataset': dataset_name,
            'train_RMSE': rmse_train, 
            'train_R2': r2_train,
            'val_RMSE': rmse_dev,
            'val_R2': r2_dev,
            'computation_time': computation_time,
            'n_estimators': n_estimators,
            'max_depth': max_depth if max_depth != None else "no",
            'min_samples_leaf': min_samples_leaf if min_samples_leaf != None else "no",
            'lags': add_lagged_vars,
            'scaling': scaling_method if apply_scaling != 0 else "no", 
            'clipping_threshold': clipping_threshold if apply_clipping != 0 else "no"
        }

        UT.save_results('outputs/results_rf', res) 

    return rmse_train, r2_train, rmse_dev, r2_dev, computation_time

def run_XGboost(dataset_name, train_set, test_set, y_test, train_size, sensor_names, seed,
                n_estimators, max_depth = None, min_samples_leaf = None, add_lagged_vars = 0, 
                apply_scaling = 0, apply_condition_scaling = 0, scaling_method = 'z-score', apply_clipping = 0, clipping_threshold = 125, 
                print_results = 0, drop_useless_sensors = 0, export_data = 0):
   
    start_time = time.time()
    train_set_copy = train_set.copy()
    test_set_copy = test_set.copy()
    
    if drop_useless_sensors != 0:
        train_set_copy, selected_sensors, excluded_sensors = FE.drop_sensors(train_set_copy, sensor_names)
        test_set_copy = test_set_copy.drop(excluded_sensors, axis=1)
    else:
        selected_sensors = sensor_names

    if add_lagged_vars != 0:
        X_train = FE.add_specific_lags(train_set, selected_sensors)
        X_test = FE.add_specific_lags(test_set, selected_sensors)
    
    test_set_copy = test_set.groupby('unit_number').last().reset_index()

    if apply_clipping != 0:
        train_set_copy = FE.clip_rul_values(train_set_copy, upper_threshold=clipping_threshold)

    train_set_copy = DP.add_operating_condition(train_set_copy)
    test_set_copy = DP.add_operating_condition(test_set_copy)

    if apply_scaling != 0:
        train_set_copy, test_set_copy = FE.scale_features(train_set_copy, test_set_copy, selected_sensors, method=scaling_method)

    

    elif apply_condition_scaling != 0:
        train_set_copy, test_set_copy = FE.scale_features_condition(train_set_copy, test_set_copy, selected_sensors, method = scaling_method)
        apply_condition_scaling = 1

    gss = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=seed)
    train_set_copy, y_train, dev_set, _ = DP.split_devSet_trainSet(train_set_copy, test_set_copy, gss, groups=train_set['unit_number'], print_groups=False)
    dev_set_copy = dev_set.copy()
    X_train, X_test, X_dev, y_train, y_dev = DP.cleaning_data(train_set_copy, test_set_copy, selected_sensors, dataset_name, df_dev=dev_set_copy)

    xgb_params = {
        'n_estimators': n_estimators,
        'objective': 'reg:squarederror',
        'random_state': seed,
        'verbosity': 0,
        **({'max_depth': max_depth} if max_depth is not None else {}),
        **({'min_child_weight': min_samples_leaf} if min_samples_leaf is not None else {})
    }

    xg_reg = xgb.XGBRegressor(**xgb_params)
    xg_reg.fit(X_train, y_train)

    y_hat_train = xg_reg.predict(X_train)
    y_hat_test = xg_reg.predict(X_test)
    y_hat_dev = xg_reg.predict(X_dev)

    computation_time = time.time() - start_time

    if print_results:
        plt.figure(figsize=(7,7))
        plt.scatter(y_test, y_hat_test, alpha=0.5, edgecolors='k')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2) 
        plt.xlabel("Valori Reali (y_test)")
        plt.ylabel("Valori Predetti (y_hat_test)")
        plt.title(f"Scatter plot: Valori Reali vs Predetti - {dataset_name}")
        plt.grid(True)
        plt.show()
        print(f"Results of XGBoost applied to dataset {dataset_name} | Time required to complete the computation: {computation_time}")
        print("\n")
    signed_error_train, rmse_train, r2_train = UT.evaluate(y_train, y_hat_train, "Train", print_results)
    signed_error_test, rmse_test, r2_test = UT.evaluate(y_test.values.flatten(), y_hat_test, "Test", print_results)
    signed_error_dev, rmse_dev, r2_dev = UT.evaluate(y_dev, y_hat_dev, "Cross Validation", print_results)
        
    if export_data:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        res = {
            'timestamp': timestamp,
            'dataset': dataset_name,
            'train_RMSE': rmse_train, 
            'train_R2': r2_train,
            'val_RMSE': rmse_dev,
            'val_R2': r2_dev,
            'computation_time': computation_time,
            'n_estimators': n_estimators, 
            'max_depth': max_depth if max_depth != None else "no",
            'min_samples_leaf': min_samples_leaf if min_samples_leaf != None else "no",
            'lags': add_lagged_vars,
            'scaling': scaling_method if apply_scaling != 0 else "no", 
            'clipping_threshold': clipping_threshold if apply_clipping != 0 else "no"
        }

        UT.save_results('outputs/results_xgb', res)  
    
    return rmse_train, r2_train, rmse_dev, r2_dev, computation_time