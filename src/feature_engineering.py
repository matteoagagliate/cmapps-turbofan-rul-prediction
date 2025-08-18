# --- LIBRARIES ---
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# --- FEATURE ENGINEERING FUNCTIONS---

def scale_features(df_train, df_test, sensor_names, method = 'z-score'):

    scaler = StandardScaler() if method == 'z-score' else MinMaxScaler()
    df_train[sensor_names] = scaler.fit_transform(df_train[sensor_names])
    df_test[sensor_names] = scaler.transform(df_test[sensor_names])
    return df_train, df_test
   
def scale_features_condition(df_train, df_test, sensor_names, method = 'z-score'):
    scaler = StandardScaler() if method == 'z-score' else MinMaxScaler()
    for condition in df_train['op_cond'].unique():
        scaler.fit(df_train.loc[df_train['op_cond']==condition, sensor_names])
        df_train.loc[df_train['op_cond']==condition, sensor_names] = scaler.transform(df_train.loc[df_train['op_cond']==condition, sensor_names])
        df_test.loc[df_test['op_cond']==condition, sensor_names] = scaler.transform(df_test.loc[df_test['op_cond']==condition, sensor_names])
    return df_train, df_test


def create_polynomial_features(df_train,df_test, sensor_names,degree=2):

    X_train_temp = df_train[sensor_names].values
    X_test_temp = df_test[sensor_names].values
    
    poly = PolynomialFeatures(degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_temp)
    X_test_poly = poly.transform(X_test_temp)
    
    feature_names = poly.get_feature_names_out(input_features=sensor_names)
    
    df_train_poly = pd.DataFrame(X_train_poly, columns=feature_names, index=df_train.index)
    df_test_poly = pd.DataFrame(X_test_poly, columns=feature_names, index=df_test.index)
    
    df_train = df_train.drop(columns=sensor_names)
    df_test = df_test.drop(columns=sensor_names)
    df_train = pd.concat([df_train, df_train_poly], axis=1)
    df_test = pd.concat([df_test, df_test_poly], axis=1)
    
    return df_train, df_test, feature_names

def clip_rul_values(df, y_dev = None, upper_threshold=125):
    df['RUL'] = df['RUL'].clip(upper=upper_threshold)
    return df

def add_specific_lags(df, sensor_names):
    list_of_lags = [1, 2, 3, 4, 5, 10, 20]
    lagged_data = {} 
    
    for i in list_of_lags:
        lagged_columns = [f"{col}_lag_{i}" for col in sensor_names]
        lagged_data.update(
            zip(lagged_columns, df.groupby('unit_number')[sensor_names].shift(i).T.values)
        )

    df_lagged = pd.DataFrame(lagged_data, index=df.index)
    df = pd.concat([df, df_lagged], axis=1)
    
    df.dropna(inplace=True)
    return df


def drop_sensors(df, sensor_names): 

    df_temp = df[sensor_names]
    print(df_temp.head())

    var_thresh = 1e-5
    selector = VarianceThreshold(threshold=var_thresh)
    selector.fit(df_temp)
    selected_cols_var = df_temp.columns[selector.get_support()]

    print("selected var col: ", selected_cols_var)

    # Correlation with RUL
    corr_df = df[selected_cols_var.tolist() + ['RUL']]
    corr_matrix = corr_df.corr()
    corr_with_rul = corr_matrix['RUL'].drop('RUL').sort_values(ascending=False)

    #plt.figure(figsize=(12, 5))
    #sns.barplot(
    #    x=corr_with_rul.values,
    #    y=corr_with_rul.index,
    #    hue=corr_with_rul.values,
    #    palette='coolwarm',
    #    dodge=False,
    #    legend=False
    #)
    #plt.title("Correlation between features and RUL")
    #plt.xlabel("Correlation")
    #plt.ylabel("Features")
    #plt.grid(True)
    #plt.tight_layout()
    #plt.show()

    correlation_threshold = 0.2 
    selected_cols_corr = corr_with_rul[abs(corr_with_rul) > correlation_threshold].index.tolist()

    print("selected corr col: ", selected_cols_corr)

    # Average behavior w.r.t RUL
    avg_by_rul = df.groupby('RUL')[selected_cols_corr].mean()

    #plt.figure(figsize=(12, 6))
    #for col in selected_cols_corr:
    #    plt.plot(avg_by_rul.index, avg_by_rul[col], label=col)
    #plt.title("Average behaviour of features w.r.t. RUL")
    #plt.xlabel("RUL")
    #plt.ylabel("Mean value")
    #plt.legend(loc='upper right', fontsize='small')
    #plt.grid(True)
    #plt.tight_layout()
    #plt.show()

    # Feature importance with Random Forest
    #X = df[selected_cols_corr]

    #model = RandomForestRegressor(n_estimators=10, random_state=42)
    #model.fit(X, y_train)

    #importances = pd.Series(model.feature_importances_, index=selected_cols_corr).sort_values(ascending=False)

    #plt.figure(figsize=(12, 4))
    #importances.plot(kind='bar', color='teal')
    #plt.title("Feature importance (Random Forest)")
    #plt.ylabel("Importance")
    #plt.grid(True)
    #plt.tight_layout()
    #plt.show()

    #final_features = importances[importances > 0.01].index.tolist()
    #print("Final sensors: ", final_features)
    #return final_features

    excluded_sensors = [col for col in df[sensor_names] if col not in selected_cols_corr]
    print(excluded_sensors)

    df = df.drop(excluded_sensors,axis=1)
    print(df.head())

    return df, selected_cols_corr, excluded_sensors 