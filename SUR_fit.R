# -------------------------------------------------------------------------------------------------------------
# Title: Diameter Distribution Percentile Models using Seemingly Unrelated Regression (SUR) - R Script
# Author: Albert Ciceu
# Affiliation: Austrian Research Center For Forests (BFW)
# Corresponding Author: Albert Ciceu, Email: albert.ciceu@bfw.gv.at
#
# This script performs Seemingly Unrelated Regression (SUR) fitting for six percentiles
# across nine datasets. It processes the input data, selects predictors based on dataset characteristics,
# splits data into training and test sets using prior model predictions, 
# applies log-transformation to variables, calculates model accuracy, and saves the results.
#
# The workflow supports analysis presented in the paper:
# "Multi-output deep learning outperforms parametric models in predicting diameter percentiles, while only 
# multi-output random forest ensures monotonicity across all percentile estimates"
# DOI: 
#
# Date: 07/07/2025
# -------------------------------------------------------------------------------------------------------------


# Load required packages for system of equations, data manipulation, and model tidying
library(systemfit)   # For Seemingly Unrelated Regression (SUR) system fitting
library(tidyverse)   # For data manipulation and piping (%>%)
library(broom)       # For tidying model outputs into data frames


# Load dataset containing diameter distribution percentiles and predictors
df <- read.csv("Output/Datasets_diameter_percentiles.csv")  

# Initialize empty data frame to collect regression coefficients and statistics for all datasets
q_basket <- data.frame()

# Loop over each unique dataset to fit and evaluate models separately
for (dfset in unique(df$dataset)) {
  
  # Load previously generated MORF model predictions to identify training and test splits for this dataset
  file_name_pred <- list.files('Output/Predictions/', pattern = paste0('MORF_predictions_', dfset), full.names = TRUE)
  pred_morf <- read.csv(file_name_pred)
  
  # Extract plot IDs belonging to training and test sets from MORF predictions
  train_plots <- pred_morf %>% filter(set == 'train') %>% pull(Plot)
  test_plots  <- pred_morf %>% filter(set == 'test') %>% pull(Plot)
  
  # Select predictor variables depending on dataset characteristics
  if (dfset %in% c('Monoculture_Fin-Picea_abies', 'Monoculture_Fin-Pinus_sylvestris')) {
    predictors <- c('Dg', 'Do', 'N', 'Thinned')
  } else if (dfset == 'Monoculture_Aut-Picea_abies') {
    predictors <- c('Dg', 'Do', 'N', 'Thinned', 'Age')
  } else if (dfset %in% c('Temperate_mixed_stands', 'Tropical_mixed_stands', 'Natural_Tur-Pinus_sylvestris')) {
    predictors <- c('Dg', 'Do', 'N')
  } else {
    predictors <- c('Dg', 'Do', 'N', 'Age')
  }
  
  # Filter dataset to include only relevant rows and columns for this dataset and selected predictors
  df_selected <- df %>%
    filter(dataset == dfset) %>%
    select(Plot, q0, q0.2, q0.4, q0.6, q0.8, q1, all_of(predictors))
  
  # Split data into training and test sets, set Plot IDs as row names for modeling convenience
  df_train <- df_selected %>% filter(Plot %in% train_plots) %>% column_to_rownames("Plot")
  df_test  <- df_selected %>% filter(Plot %in% test_plots) %>% column_to_rownames("Plot")
  
  # Define the response variables — percentiles of diameter distribution
  response_vars <- c("q0", "q0.2", "q0.4", "q0.6", "q0.8", "q1")
  
  # Log-transform response and predictor variables
  vars_to_log <- setdiff(c(response_vars, predictors), "Thinned")
  
  df_train_log <- df_train %>%
    mutate(across(all_of(vars_to_log), ~ log(.)))
  
  df_test_log <- df_test %>%
    mutate(across(all_of(vars_to_log), ~ log(.)))
  
  # Construct a system of linear regression formulas for SUR fitting,
  # one formula per percentile with the chosen predictors
  equations <- map(
    response_vars,
    ~ as.formula(paste(.x, "~", paste(predictors, collapse = " + ")))
  )
  names(equations) <- response_vars
  
  # Fit the system of equations using Seemingly Unrelated Regression (SUR) method
  # This accounts for correlated residuals across different percentile equations
  modSUR_1 <- systemfit(equations, method = "SUR", data = df_train_log, maxit = 100)
  s1 <- summary(modSUR_1)
  
  # Extract residual covariance matrix diagonal to estimate mean squared errors (MSE)
  mse_df <- data.frame(mse = diag(s1$residCovEst)) %>%
    rownames_to_column('quantile')
  
  # Generate predictions for both training and test sets using the fitted SUR model
  df_pred_1_raw <- bind_rows(
    predict(modSUR_1, df_test_log) %>% mutate(set = 'test', Plot = rownames(df_test_log)),
    predict(modSUR_1, df_train_log) %>% mutate(set = 'train', Plot = rownames(df_train_log))
  ) %>%
    mutate(model = 'SUR', dataset = dfset) %>%
    rename_with(~ gsub("(q[0-9\\.]+)\\.pred", "pred_\\1", .), everything())
  
  # Reshape predictions to long format, apply bias correction (exp(prediction + 0.5 * MSE)) to revert log transform,
  # then reshape back to wide format and join with original observed data for evaluation
  df_pred_1 <- df_pred_1_raw %>%
    pivot_longer(cols = starts_with("pred_"), names_to = "quantile", values_to = "predicted") %>%
    mutate(quantile = str_remove(quantile, "pred_")) %>%
    left_join(mse_df, by = "quantile") %>%
    mutate(predicted = exp(predicted + 0.5 * mse)) %>%
    select(-mse) %>%
    pivot_wider(names_from = quantile, values_from = predicted, names_prefix = "pred_") %>%
    mutate(model = 'SUR') %>%
    left_join(df_selected, by = "Plot")
  
  # Prepare actual and predicted data in long format to calculate RMSE for each percentile and data split
  actuals <- df_pred_1 %>%
    select(Plot, set, starts_with("q")) %>%
    pivot_longer(cols = starts_with("q"), names_to = "quantile", values_to = "actual")
  
  predictions <- df_pred_1 %>%
    select(Plot, set, starts_with("pred_")) %>%
    pivot_longer(cols = starts_with("pred_"), names_to = "quantile", values_to = "predicted") %>%
    mutate(quantile = str_remove(quantile, "pred_"))
  
  # Calculate RMSE per quantile and data set (train/test) to assess prediction accuracy
  model_ranking <- left_join(actuals, predictions, by = c("Plot", "set", "quantile")) %>%
    group_by(quantile, set) %>%
    summarise(rmse = sqrt(mean((actual - predicted)^2, na.rm = TRUE)), .groups = 'drop') %>%
    arrange(rmse)
  
  # Print dataset name and RMSE results for monitoring
  print(paste("Dataset:", dfset))
  print(model_ranking)
  
  # Save predictions for further analysis or validation
  write.csv(df_pred_1, paste0('Output/Predictions/SUR_predictions_', dfset, '.csv'), row.names = FALSE)
  
  # Extract and accumulate regression coefficients and statistics for all datasets
  coeffs <- tidy(modSUR_1) %>%
    mutate(dataset = dfset, model = 'SUR') %>%
    select(dataset, model, term, estimate, std.error, statistic, p.value)
  q_basket <- bind_rows(q_basket, coeffs)
}

# Export all collected coefficients and stats to a CSV file for reporting or meta-analysis
write.csv(q_basket, "Output/SUR_datasets_parameters.csv", row.names = FALSE)

