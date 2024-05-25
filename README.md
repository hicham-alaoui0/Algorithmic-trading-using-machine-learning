# Algorithmic-trading-using-machine-learning

# Introduction

This repository contains Python code for training a time series prediction model using XGBoost. The model is designed to predict the target variable (Ticks) based on the features in the pred2017l.csv dataset.

# Dependencies

This code requires the following Python libraries:

pandas
scikit-learn
xgboost
You can install them using pip:

Bash
pip install pandas scikit-learn xgboost
Use code with caution.


# Instructions

Download the data: Obtain the pred2017l.csv dataset and place it in the same directory as this code.
Run the script: Execute the Python script (e.g., main.py) to train the model and evaluate its performance.
Code Explanation

The code is divided into several sections, each commented with # %% [markdown] to indicate descriptive text:

Import libraries (pandas, scikit-learn, xgboost)

Read the CSV data using pandas.read_csv()

Display the first few rows of the data using data.head()

# Train-test-split:

Separate the target variable (Ticks) from the features
Split the data into training, validation, and test sets using scikit-learn's train_test_split()
Print the shapes of each dataset

# Convert data into XGB format:

Create XGBoost DMatrix objects for training, validation, and test data
XGB parameters:
Define parameters for the XGBoost model, including:
n_trees: Number of decision trees to use
eta: Learning rate
max_depth: Maximum depth of each tree
subsample: Subsample ratio for each tree
objective: Objective function (regression: linear in this case)
eval_metric: Evaluation metric (root mean squared error)
silent: Verbosity level

# Train the model:

Train the XGBoost model using xgb.train()

# Predict the test data:

Make predictions on the test data using the trained model

# Evaluate model quality:

Calculate mean squared error (MSE), root mean squared error (RMSE), and mean absolute error (MAE) using scikit-learn's metrics functions

# Further Considerations

You might want to adjust the XGBoost parameters for better performance on your specific dataset.
Consider evaluating the model on unseen data to assess its generalizability.
Explore more advanced techniques like hyperparameter tuning and feature engineering.

# Contributing

Feel free to fork this repository, make changes, and create pull requests with your improvements.
