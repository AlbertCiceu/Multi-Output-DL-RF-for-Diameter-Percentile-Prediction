# -------------------------------------------------------------------------------------------------------------
# Title: Diameter Distribution Percentile Models using Multi-Output Random Forest (MORF) - R Script
# Author: Albert Ciceu
# Affiliation: Austrian Research Center For Forests (BFW)
# Corresponding Author: Albert Ciceu, Email: albert.ciceu@bfw.gv.at
#
# Description: This script implements multi-output Random Forest regression models to predict diameter distribution
# percentiles across multiple forest datasets. It includes dataset-specific predictor selection, stratified train-test
# splitting, and extensive randomized hyperparameter search with cross-validation. Trained models and best parameters
# are saved, and predictions are generated and exported.
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RepeatedKFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# For reproducibility
seed = 42
np.random.seed(seed)

# Load dataset containing diameter distribution percentiles and predictors
df = pd.read_csv("Output/Datasets_diameter_percentiles.csv")

# Define datasets
datasets = df['dataset'].unique()

# Define hyperparameter search space for Multi-Output Random Forest
param_dist = {
    'n_estimators': list(np.linspace(50, 500, 46, dtype=int)),
    'max_features': ['sqrt', 'log2'],
    'max_depth': list(np.linspace(1, 10, 10, dtype=int)) + [None],
    'min_samples_split': list(np.linspace(5, 25, 21, dtype=int)),
    'min_samples_leaf': list(np.linspace(3, 10, 8, dtype=int)),
    'bootstrap': [True, False]
}

# Create output folders if they don't exist
os.makedirs("Models", exist_ok=True)
os.makedirs("Output/Predictions", exist_ok=True)
os.makedirs("Output/Best_Params", exist_ok=True)

# Define evaluation function

def evaluate(y_true, y_pred, label):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"label": label, "RMSE": rmse, "MAE": mae, "R2": r2}

results = []

# Loop through selected datasets
for dataset_name in datasets:
    subset = df[df['dataset'] == dataset_name].copy()
    subset = subset.set_index("Plot")

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
    y_columns = ['q0', 'q0.2', 'q0.4', 'q0.6', 'q0.8', 'q1']
    y = subset[y_columns]

    # Stratify by Dg to ensure consistent training/test split
    Y_bins = pd.qcut(X['Dg'], q=5, labels=False)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=Y_bins, random_state=seed
    )

    # Setup Random Forest Regressor
    rf = RandomForestRegressor(random_state=seed)
    cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=seed)
    random_search = RandomizedSearchCV(
        rf,
        param_distributions=param_dist,
        n_iter=500,
        cv=cv,
        verbose=2,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        random_state=seed
    )

    # Fit the model using random search
    random_search.fit(X_train, y_train)
    best_rf = random_search.best_estimator_
    print(f"Best Hyperparameters for {dataset_name}:", random_search.best_params_)

    # Save model and parameters
    joblib.dump(best_rf, f"Models/MORF_{dataset_name}.joblib")
    pd.Series(random_search.best_params_).to_csv(f"Output/Best_Params/Best_Params_{dataset_name}.csv")

    # Generate predictions
    train_preds = best_rf.predict(X_train)
    test_preds = best_rf.predict(X_test)

    # Combine and label outputs
    train_output = X_train.copy()
    train_output[y_columns] = y_train
    train_output[[f"pred_{col}" for col in y_columns]] = train_preds
    train_output['dataset'] = dataset_name
    train_output['set'] = 'train'
    train_output['model'] = 'MORF'

    test_output = X_test.copy()
    test_output[y_columns] = y_test
    test_output[[f"pred_{col}" for col in y_columns]] = test_preds
    test_output['dataset'] = dataset_name
    test_output['set'] = 'test'
    test_output['model'] = 'MORF'

    # Combine and save all outputs
    combined_output = pd.concat([train_output, test_output])
    combined_output.to_csv(f"Output/Predictions/MORF_predictions_{dataset_name}.csv", index=True)
