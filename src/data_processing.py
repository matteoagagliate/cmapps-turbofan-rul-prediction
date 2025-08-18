# --- LIBRARIES ---
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter


# --- DATA PROCESSING FUNCTIONS ---
def reading_data(dir_path, dataset_name):

    index_names = ['unit_number', 'current_cycle']
    setting_names = ['setting_1', 'setting_2', 'setting_3']
    sensor_names = ['s_{}'.format(i) for i in range(1, 22)]
    column_names = index_names + setting_names + sensor_names

    train_set = pd.read_csv(dir_path + 'train_'+dataset_name+'.txt', sep=r'\s+', header=None, names=column_names)
    test_set = pd.read_csv(dir_path+'test_'+dataset_name+'.txt', sep=r'\s+', header=None, names=column_names)

    y_test = pd.read_csv((dir_path+'RUL_'+dataset_name+'.txt'), sep=r'\s+', header=None, names=['RUL'])

    train_set = add_remaining_useful_life(train_set)
    y_train = train_set['RUL']

    train_set[sensor_names] = train_set[sensor_names].astype('float64')
    test_set[sensor_names] = test_set[sensor_names].astype('float64')

    return dir_path, index_names, setting_names, sensor_names, column_names, train_set, test_set, y_test, y_train

def savitzky_golay_filter(df, sensor_names, window_length=11, polyorder=2):
    df_filtered = df.copy()
    for sensor in sensor_names:
        if sensor in df.columns:
            wl = window_length if window_length % 2 == 1 else window_length + 1
            if len(df[sensor]) >= wl:
                try:
                    df_filtered[sensor] = savgol_filter(df[sensor], wl, polyorder)
                except Exception as e:
                    print(f"Error applying Savitzky-Golay to {sensor}: {e}")
            else:
                print(f"Warning: column '{sensor}' too short for Savitzky-Golay (len < {wl})")
        else:
            print(f"Warning: column '{sensor}' not found in DataFrame")
    return df_filtered

def moving_average_filter(df, sensor_names, window=5):
    df_filtered = df.copy()
    for sensor in sensor_names:
        if sensor in df.columns:
            df_filtered[sensor] = df[sensor].rolling(window=window, center=True, min_periods=1).mean()
        else:
            print(f"Warning: Column '{sensor}' not found in DataFrame.")
    return df_filtered

def ema_filter(df, sensor_names, span=10):
    df_filtered = df.copy()
    for sensor in sensor_names:
        if sensor in df.columns:
            df_filtered[sensor] = df[sensor].ewm(span=span, adjust=False).mean()
        else:
            print(f"Warning: Column '{sensor}' not found in DataFrame.")
    return df_filtered

def apply_filtering(df_train, df_test, filter_name, sensor_names): 
    if filter_name == 'Moving Average':
            df_train = moving_average_filter(df_train, sensor_names)
            df_test = moving_average_filter(df_test, sensor_names)
    elif filter_name == 'EMA':
            df_train = ema_filter(df_train, sensor_names)
            df_test = ema_filter(df_test, sensor_names)
    elif filter_name == 'Savitzky-Golay':
            df_train = savitzky_golay_filter(df_train, sensor_names)
            df_test = savitzky_golay_filter(df_test, sensor_names)
    else: 
            df_train = df_train.copy()
            df_test = df_test.copy()
    return df_train, df_test

def add_remaining_useful_life(df):
    grouped_by_unit = df.groupby("unit_number")
    max_cycle = grouped_by_unit["current_cycle"].transform('max')
    df = df.copy()
    df["RUL"] = max_cycle - df["current_cycle"]
    return df

def add_operating_condition(df):

    df_op_cond = df.copy()
    df_op_cond['setting_1'] = df_op_cond['setting_1'].round()
    df_op_cond['setting_2'] = df_op_cond['setting_2'].round(decimals=2)
    
    df_op_cond['op_cond'] = df_op_cond['setting_1'].astype(str) + '_' + \
                        df_op_cond['setting_2'].astype(str) + '_' + \
                        df_op_cond['setting_3'].astype(str)
    return df_op_cond

def cleaning_data(df_train, df_test, sensor_names, dataset_name, df_dev = None): 

    sensor_names = list(sensor_names)
    y_train =  df_train['RUL']

    if dataset_name in ['FD001', 'FD003']:
        X_train = df_train[sensor_names]
        X_test = df_test[sensor_names]
    else:
        setting_names = ['setting_1', 'setting_2', 'setting_3']
        X_train = df_train[sensor_names + setting_names]
        X_test = df_test[sensor_names + setting_names]

    if df_dev is not None:
        y_dev = df_dev['RUL']
        if dataset_name in ['FD001', 'FD003']:
            X_dev = df_dev[sensor_names]
        else:
            X_dev =  df_dev[sensor_names + setting_names]
        return X_train, X_test, X_dev, y_train, y_dev

    return X_train, X_test, y_train

def split_devSet_trainSet(df_train, df_test, gss, groups, print_groups=False): 

    y_train = df_train['RUL']

    for idx_train, idx_dev in gss.split(df_train, y_train, groups=groups):
        if print_groups:
            print('train_split_engines', df_train.iloc[idx_train]['unit_number'].unique(), '\n')
            print('validate_split_engines', df_train.iloc[idx_dev]['unit_number'].unique(), '\n')

        df_train_split = df_train.iloc[idx_train]
        y_train_split = y_train.iloc[idx_train]
        df_dev_split = df_train.iloc[idx_dev]
        y_dev_split = y_train.iloc[idx_dev]
    return df_train_split, y_train_split, df_dev_split, y_dev_split

def generate_train_sequence(df, sequence_length, columns, unit_numbers=np.array([])):
    if unit_numbers.size <= 0:
        unit_numbers = df['unit_number'].unique()
    
    def generate_unit(df, sequence_length, columns):
        data = df[columns].values
        elements_number = data.shape[0]

        for start, stop in zip(range(0, elements_number-(sequence_length-1)), range(sequence_length, elements_number+1)):
            yield data[start:stop, :]
        
    generated_sequence = (list(generate_unit(df[df['unit_number']==unit_number], sequence_length, columns))
               for unit_number in unit_numbers)
    train_sequence_array = np.concatenate(list(generated_sequence)).astype(np.float32)
    return train_sequence_array

def generate_lables(df, sequence_length, label, unit_numbers=np.array([])):
    if unit_numbers.size <= 0:
        unit_numbers = df['unit_number'].unique()
    
    def generate_unit_lable(df, sequence_length, label):
        data_matrix = df[label].values
        num_elements = data_matrix.shape[0]
        return data_matrix[sequence_length-1:num_elements, :] 

    generated_labels = [generate_unit_lable(df[df['unit_number']==unit_number], sequence_length, label) 
                for unit_number in unit_numbers]
    label_array = np.concatenate(generated_labels).astype(np.float32)
    return label_array

def generate_test_data(df, sequence_length, columns, mask_value):
    if df.shape[0] < sequence_length:
        data_matrix = np.full(shape=(sequence_length, len(columns)), fill_value=mask_value)
        starting_index = data_matrix.shape[0] - df.shape[0]
        data_matrix[starting_index:,:] = df[columns].values
    else:
        data_matrix = df[columns].values
    
    stop = elements_number = data_matrix.shape[0]
    start = stop - sequence_length
    for i in list(range(1)):
        yield data_matrix[start:stop, :] 