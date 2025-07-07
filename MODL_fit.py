# -------------------------------------------------------------------------------------------------------------
# Title: Diameter Distribution Percentile Models using Multi-Output Deep Learning (MODL) - R Script
# Author: Albert Ciceu
# Affiliation: Austrian Research Center For Forests (BFW)
# Corresponding Author: Albert Ciceu, Email: albert.ciceu@bfw.gv.at
#
# Description: This script implements a multi-output deep learning model to predict diameter distribution percentiles
# across multiple datasets. It performs data preprocessing, predictor selection based on dataset type, 
# scaling of input and output variables, and stratified train-test splitting. The script applies randomized hyperparameter 
# search with cross-validation and early stopping to optimize the neural network architecture and training parameters. 
# Final models are trained and saved per dataset, with predictions back-transformed and exported for evaluation.
#
# The workflow supports analysis presented in the paper:
# "Multi-output deep learning outperforms parametric models in predicting diameter percentiles, while only 
# multi-output random forest ensures monotonicity across all percentile estimates"
# DOI: 
#
# Date: 07/07/2025
# -------------------------------------------------------------------------------------------------------------


import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import train_test_split, RepeatedKFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# For reproducibility
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

# Load dataset containing diameter distribution percentiles and predictors
df = pd.read_csv("Output/Datasets_diameter_percentiles.csv")

# Define datasets
datasets = df['dataset'].unique()

# Create output folders
os.makedirs("Models", exist_ok=True)
os.makedirs("Output/Predictions", exist_ok=True)
os.makedirs("Output/Best_Params", exist_ok=True)

# Build Keras model function
def build_model(neurons_1=64, neurons_2=64, learning_rate=0.001):
    model = keras.Sequential([
        layers.Input(shape=(X_train.shape[1],)),
        layers.Dense(neurons_1, activation='relu'),
        layers.Dense(neurons_2, activation='relu'),
        layers.Dense(6, activation='linear')  # Assuming 6 quantile outputs
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mse']
    )
    return model

# Evaluation function
def evaluate(y_true, y_pred, label):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"label": label, "RMSE": rmse, "MAE": mae, "R2": r2}

results = []

# Loop through datasets
for dataset_name in datasets:
    subset = df[df['dataset'] == dataset_name].copy()
    subset = subset.set_index(["Plot"])

    # Select predictor variables depending on dataset characteristics
    if dataset_name in ['Monoculture_Fin-Picea_abies', 'Monoculture_Fin-Pinus_sylvestris']:
        predictors = ['Dg', 'Do', 'N', 'Thinned']
    elif dataset_name == 'Monoculture_Aut-Picea_abies':
        predictors = ['Dg', 'Do', 'N', 'Thinned', 'Age']
    elif dataset_name in ['Temperate_mixed_stands', 'Tropical_mixed_stands', 'Natural_Tur-Pinus_sylvestris']:
        predictors = ['Dg', 'Do', 'N']
    else:
        predictors = ['Dg', 'Do', 'N', 'Age']

    X = subset[predictors]
    y = subset[['q0', 'q0.2', 'q0.4', 'q0.6', 'q0.8', 'q1']]

    # Separate 'Thinned' (do not scale it)
    if 'Thinned' in X.columns:
        X_to_scale = X.drop(columns='Thinned')
        X_thinned = X[['Thinned']]
    else:
        X_to_scale = X
        X_thinned = None

    # Scale only the selected part
    X_scaler = StandardScaler()
    X_scaled_part = pd.DataFrame(X_scaler.fit_transform(X_to_scale), index=X.index, columns=X_to_scale.columns)

    # Combine scaled and unscaled parts
    if X_thinned is not None:
        X_scaled = pd.concat([X_scaled_part, X_thinned], axis=1)
    else:
        X_scaled = X_scaled_part

    # Scale outputs
    y_scaler = StandardScaler()
    y_scaled = pd.DataFrame(y_scaler.fit_transform(y), index=y.index, columns=y.columns)

    # Create bins from unscaled Dg to stratify
    Y_bins = pd.qcut(X['Dg'], q=5, labels=False)
    train_idx, test_idx = train_test_split(X.index, test_size=0.2, stratify=Y_bins, random_state=seed)

    X_train = X_scaled.loc[train_idx]
    X_test = X_scaled.loc[test_idx]
    y_train = y_scaled.loc[train_idx]
    y_test = y_scaled.loc[test_idx]

    model = KerasRegressor(build_fn=build_model, verbose=0)

    param_dist = {
        'neurons_1': list(range(32, 513, 5)),
        'neurons_2': list(range(32, 513, 5)),
        'learning_rate': np.round(np.arange(0.0001, 0.0111, 0.001), 5),
        'batch_size': list(range(32, 65, 2)),
        'epochs': list(range(500, 5001, 500))
    }

    cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=seed)
    early_stopping = keras.callbacks.EarlyStopping(monitor='loss', patience=250, restore_best_weights=True)

    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=500,
        cv=cv,
        random_state=seed,
        verbose=1,
        n_jobs=-1
    )

    random_search.fit(X_train, y_train, callbacks=[early_stopping])
    best_params = random_search.best_params_
    print(f"[{dataset_name}] Best Hyperparameters:", best_params)

    final_model = build_model(
        neurons_1=best_params['neurons_1'],
        neurons_2=best_params['neurons_2'],
        learning_rate=best_params['learning_rate']
    )

    history = final_model.fit(
        X_train, y_train,
        epochs=best_params['epochs'],
        batch_size=best_params['batch_size'],
        callbacks=[early_stopping],
        verbose=1
    )

    final_model.save(f"Models/MODL_{dataset_name}.h5")
    pd.Series(best_params).to_csv(f"Output/Best_Params/Best_Params_MODL_{dataset_name}.csv")

    train_preds = y_scaler.inverse_transform(final_model.predict(X_train))
    test_preds = y_scaler.inverse_transform(final_model.predict(X_test))
    y_train_original = y_scaler.inverse_transform(y_train)
    y_test_original = y_scaler.inverse_transform(y_test)

    # Back-transform X except 'Thinned'
    if X_thinned is not None:
        X_train_part = pd.DataFrame(X_scaler.inverse_transform(X_train.drop(columns='Thinned')),
                                    index=X_train.index,
                                    columns=X_to_scale.columns)
        X_train_original = pd.concat([X_train_part, X_train[['Thinned']]], axis=1)

        X_test_part = pd.DataFrame(X_scaler.inverse_transform(X_test.drop(columns='Thinned')),
                                   index=X_test.index,
                                   columns=X_to_scale.columns)
        X_test_original = pd.concat([X_test_part, X_test[['Thinned']]], axis=1)
    else:
        X_train_original = pd.DataFrame(X_scaler.inverse_transform(X_train),
                                        index=X_train.index, columns=X_to_scale.columns)
        X_test_original = pd.DataFrame(X_scaler.inverse_transform(X_test),
                                       index=X_test.index, columns=X_to_scale.columns)

    y_columns = ['q0', 'q0.2', 'q0.4', 'q0.6', 'q0.8', 'q1']

    train_output = pd.DataFrame(X_train_original, columns=X.columns)
    train_output[y_columns] = pd.DataFrame(y_train_original, columns=y_columns, index = X_train_original.index)
    train_output[[f"pred_{col}" for col in y_columns]] = pd.DataFrame(train_preds, columns=[f"pred_{col}" for col in y_columns], index=X_train_original.index)
    train_output['dataset'] = dataset_name
    train_output['set'] = 'train'
    train_output['model'] = 'MODL'
    train_output['Plot'] = train_idx

    test_output = pd.DataFrame(X_test_original, columns=X.columns)
    test_output[y_columns] = pd.DataFrame(y_test_original, columns=y_columns, index = X_test_original.index)
    test_output[[f"pred_{col}" for col in y_columns]] = pd.DataFrame(test_preds, columns=[f"pred_{col}" for col in y_columns], index = X_test_original.index)
    test_output['dataset'] = dataset_name
    test_output['set'] = 'test'
    test_output['model'] = 'MODL'
    test_output['Plot'] = test_idx

    combined_output = pd.concat([train_output, test_output])
    combined_output.to_csv(f"Output/Predictions/MODL_predictions_{dataset_name}.csv", index=False)
