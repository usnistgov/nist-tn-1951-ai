# -*- coding: utf-8 -*-
# NIST-TN-1951-AI: clouddata_randforest.py
# Description: Python script for applying random forest-based AI techniques to analyze NIST 1951 industrial
#              wireless propagation measurements in a cloud-based environment. This script handles data
#              preprocessing, random forest model training, and evaluation for industrial wireless scenarios.
# Author: Rick Candell, NIST
# Contact: For inquiries, visit https://www.nist.gov/programs-projects/wireless-systems-industrial-environments
# Dependencies: Requires code from https://github.com/rcandell/IndustrialWirelessAnalysis
# Citation: If you use or extend this code, please cite https://doi.org/10.18434/T4359D
# Disclaimer: Certain tools, equipment, or materials identified do not imply NIST endorsement.
# Created: July 2025
# License: GNU GENERAL PUBLIC LICENSE, Version 3, 29 June 2007

# Import necessary libraries for data processing, modeling, and visualization
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

# Loading and inspecting the dataset
# Read the dataset from a CSV file
df = pd.read_csv('../dataCloud/stats/aimetrics.csv')
# Display the first few rows of the dataframe
print(df.head())
# Display information about the dataframe (e.g., data types, non-null counts)
print(df.info())
# Display summary statistics of numerical columns
print(df.describe())

# Handling missing values
# Check for missing values in each column
print(df.isnull().sum())
# Remove rows with missing values
df = df.dropna()
# Verify that no missing values remain
print(df.isnull().sum())
# TODO: Investigate the source of missing values for future improvement

# Encoding categorical variables
# Define columns to be converted to categorical
cat_columns = ['Site', 'Polarization', 'Obstructed', 'Waveguided', 'LOS']
# Convert categorical columns to dummy variables, dropping the first category to avoid multicollinearity
df_encoded = pd.get_dummies(df, columns=cat_columns, drop_first=True)
# Rename the 'LOS_1' column to 'LOS' for clarity
df_encoded = df_encoded.rename(columns={'LOS_1': 'LOS'})

# Scaling numerical features
# Convert time-based columns to nanoseconds for consistency
df_encoded['DelaySpread'] = df_encoded['DelaySpread'] * 1e9
df_encoded['MaxDelay'] = df_encoded['MaxDelay'] * 1e9
df_encoded['MeanDelay'] = df_encoded['MeanDelay'] * 1e9

# Define the list of encoded column names, including numerical and dummy variables
encoded_col_names = ['TxPos', 'TxHeight', 'Freq', 'Range_m', 'Range_l', 'CoordX', 'CoordY', 'DelaySpread', 'MeanDelay', 'MaxDelay', 'LOS'] + \
           [col for col in df_encoded.columns if col.startswith(('Site_', 'Polarization_', 'Obstructed_', 'Waveguided_', 'LOS_'))]

# Selecting features and target
# Specify the target variable (uncomment to choose a different target)
# yName = 'LOS'
yName = 'DelaySpread'
# yName = 'MaxDelay'
# yName = 'MeanDelay'
# yName = 'PathGain'
# Extract the target variable
y = df_encoded[yName]
# Select features by excluding the target variable from the encoded columns
features = [col for col in encoded_col_names if col != yName]
X = df_encoded[features].copy()

# Scaling numerical features
# Import StandardScaler for feature scaling
from sklearn.preprocessing import StandardScaler
# Initialize the scaler
scaler = StandardScaler()
# List of numerical columns to scale (excluding the target if it's numerical)
numerical_cols_raw = ['TxPos', 'TxHeight', 'Freq', 'Range_m', 'Range_l', 'CoordX', 'CoordY', 'DelaySpread', 'MeanDelay', 'MaxDelay']
numerical_cols = [col for col in numerical_cols_raw if col != yName]
# Apply scaling to numerical features
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# Selecting and initializing the model
# Check if the target variable is categorical or numerical to choose the appropriate model
if yName in cat_columns:
    # Import RandomForestClassifier for categorical target
    from sklearn.ensemble import RandomForestClassifier
    # Initialize the classifier with a fixed random state for reproducibility
    #model = RandomForestClassifier(random_state=42)
    model = RandomForestClassifier(
        n_estimators=100,
        criterion="gini",
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        bootstrap=True,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )
else:
    # Import RandomForestRegressor for numerical target
    from sklearn.ensemble import RandomForestRegressor
    # Initialize the regressor with 100 trees and a fixed random state
    model = RandomForestRegressor(n_estimators=100, random_state=42)

# Splitting the data
# Split the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning (disabled by default)
# Optional: Perform grid search to optimize model parameters
if 0:
    # Import GridSearchCV for hyperparameter tuning
    from sklearn.model_selection import GridSearchCV
    # Define the parameter grid for tuning
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    # Initialize GridSearchCV with the regressor and parameter grid
    grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5)
    # Fit the grid search to the training data
    grid_search.fit(X_train, y_train)
    # Print the best parameters found
    print(f'Best Parameters: {grid_search.best_params_}')
    # Update the model to use the best estimator
    model = grid_search.best_estimator_

# Training the model
# Fit the model to the training data
model.fit(X_train, y_train)

# Evaluating the model
# Make predictions on the test set
y_pred = model.predict(X_test)
# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
# Calculate R^2 score
r2 = r2_score(y_test, y_pred)
# Print evaluation metrics
print(f'Mean Squared Error: {mse:.4f}')
print(f'R^2 Score: {r2:.4f}')

# Analyzing feature importance
# Create a dataframe to store feature names and their importance scores
importances = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
# Print feature importances sorted in descending order
print(importances.sort_values('Importance', ascending=False))

# Visualizing results
# Create a scatter plot to compare actual vs. predicted values
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual ' + yName)
plt.ylabel('Predicted ' + yName)
plt.title('Actual vs Predicted ' + yName + ' Using Random Forest')
plt.grid(True)
plt.show()

# Saving the model
# Save the trained model to a file for future use
joblib.dump(model, yName+'_randfor_model.pkl')

