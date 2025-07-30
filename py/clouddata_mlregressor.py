# -*- coding: utf-8 -*-
# NIST-TN-1951-AI: clouddata_mlregressor.py
# Description: Python script for applying machine learning regression techniques to analyze NIST 1951 industrial
#              wireless propagation measurements in a cloud-based environment. This script handles data
#              preprocessing, training of machine learning regressors, and evaluation for industrial wireless scenarios.
# Author: Rick Candell, NIST
# Contact: For inquiries, visit https://www.nist.gov/programs-projects/wireless-systems-industrial-environments
# Dependencies: Requires code from https://github.com/rcandell/IndustrialWirelessAnalysis
# Citation: If you use or extend this code, please cite https://doi.org/10.18434/T4359D
# Disclaimer: Certain tools, equipment, or materials identified do not imply NIST endorsement.
# Created: July 2025
# License: GNU General Public License, Version 3, 29 June 2007

# Import necessary libraries for data processing, modeling, and visualization
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import math
# import seaborn as sns  # Commented out, not used in the script

# Loading and inspecting the dataset
# Read the dataset from a CSV file
df = pd.read_csv('../dataCloud/stats/aimetrics.csv')
# Display the first few rows of the dataframe
print(df.head())
# Display information about the dataframe (e.g., data types, non-null counts)
print(df.info())
# Display summary statistics of numerical columns
print(df.describe())

# Scaling specific numerical features
# Convert time-based columns to nanoseconds for consistency
df['DelaySpread'] = df['DelaySpread'] * 1e9
df['MaxDelay'] = df['MaxDelay'] * 1e9
df['MeanDelay'] = df['MeanDelay'] * 1e9

# Handling missing values
# Check for missing values in each column before dropping
print("Missing Values Before Dropping:")
print(df.isnull().sum())
# Remove rows with missing values
df = df.dropna()
# Verify that no missing values remain
print("Missing Values After Dropping:")
print(df.isnull().sum())

# Verifying LOS column (categorical input feature)
# Print the data type of the LOS column
print("LOS Data Type:", df['LOS'].dtype)
# Print unique values in the LOS column
print("LOS Unique Values:", df['LOS'].unique())
# Check for missing or non-finite values in LOS
print("LOS Missing or Non-Finite Values:", df['LOS'].isna().sum(), "NaN,", 
      (df['LOS'].isin([np.inf, -np.inf])).sum(), "Inf")

# Handling non-finite and missing values in LOS (categorical)
# Calculate the mode of LOS, default to '-1' if all values are missing
los_mode = df['LOS'].mode()[0] if not df['LOS'].isna().all() else '-1'
# Replace infinite values with the mode
df['LOS'] = df['LOS'].replace([np.inf, -np.inf], los_mode)
# Fill missing values with the mode
df['LOS'] = df['LOS'].fillna(los_mode)

# Converting LOS to standardized categorical values
# Map LOS values to a consistent format ('1' or '-1')
df['LOS'] = df['LOS'].astype(str).str.lower().map({
    '-1': '-1', '1': '1', '+1': '1', 'true': '1', 'false': '-1', 'yes': '1', 'no': '-1', 'nan': los_mode
}).fillna(los_mode)

# Defining target variables and selecting one
# List of possible numerical target variables
yNames = ['DelaySpread', 'MeanDelay', 'MaxDelay', 'PathGain']
# Corresponding types for the target variables (all numerical)
yTypes = ['num', 'num', 'num', 'num']
# Select the target variable (index 3 corresponds to PathGain)
which_target = 3  # 0: DelaySpread, 1: MeanDelay, 2: MaxDelay, 3: PathGain
yName = yNames[which_target]

# Verifying target variable
# Print the data type of the target variable
print(f"{yName} Data Type:", df[yName].dtype)
# Print unique values in the target variable
print(f"{yName} Unique Values:", df[yName].unique())
# Check for missing or non-finite values in the target
print(f"{yName} Missing or Non-Finite Values:", df[yName].isna().sum(), "NaN,", 
      (df[yName].isin([np.inf, -np.inf])).sum(), "Inf")

# Handling non-finite and missing values in the target
# Calculate the mode of the target, default to 0.0 if all values are missing
target_mode = df[yName].mode()[0] if not df[yName].isna().all() else 0.0
# Replace infinite values with the mode
df[yName] = df[yName].replace([np.inf, -np.inf], target_mode)
# Fill missing values with the mode and ensure numeric type
df[yName] = pd.to_numeric(df[yName], errors='coerce').fillna(target_mode).astype(float)

# Encoding categorical variables
# Define categorical columns to convert to dummy variables
cat_columns = ['Site', 'Polarization', 'Obstructed', 'Waveguided', 'LOS']
# Convert categorical columns to dummy variables, dropping the first category to avoid multicollinearity
df_encoded = pd.get_dummies(df, columns=cat_columns, drop_first=True)

# Defining features and target
# Initialize the list of features with core numerical columns
features = ['TxPos', 'TxHeight', 'Freq', 'Range_m', 'Range_l', 'CoordX', 'CoordY']
# Include other numerical features, excluding the target
other_numerical = ['DelaySpread', 'MeanDelay', 'MaxDelay', 'PathGain']
features += [col for col in other_numerical if col != yName]
# Include dummy variables from categorical columns
features += [col for col in df_encoded.columns if col.startswith(('Site_', 'Polarization_', 'Obstructed_', 'Waveguided_', 'LOS_'))]
# Define numerical columns for scaling
numerical_cols = ['TxPos', 'TxHeight', 'Freq', 'Range_m', 'Range_l', 'CoordX', 'CoordY'] + \
                 [col for col in other_numerical if col != yName]
# Extract features and target from the encoded dataframe
X = df_encoded[features].copy()  # Explicit copy to avoid SettingWithCopyWarning
y = df_encoded[yName].astype(float)  # Ensure target is float

# Scaling numerical features
# Initialize the StandardScaler
scaler = StandardScaler()
# Apply scaling to numerical columns
X.loc[:, numerical_cols] = scaler.fit_transform(X[numerical_cols])

# Splitting the data
# Split the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the Neural Network model (MLPRegressor)
# Define possible activation and solver options
activ_opts = ['relu', 'tanh', 'logistic', 'identity']
solver_opts = ['adam', 'sgd', 'lbfgs']
# Initialize the MLPRegressor with specified parameters
model = MLPRegressor(hidden_layer_sizes=(100, 50), activation=activ_opts[1], solver=solver_opts[0], 
                     max_iter=1000, random_state=42, early_stopping=True, validation_fraction=0.1)
# Fit the model to the training data
model.fit(X_train, y_train)

# Making predictions
# Predict on the test set
y_pred = model.predict(X_test)

# Debugging output
# Print sample values from actual and predicted outputs for inspection
print("y_test Sample Values:", y_test[:5].values)
print("y_pred Sample Values:", y_pred[:5])
# Print data types of actual and predicted outputs
print("y_test Type:", y_test.dtype)
print("y_pred Type:", y_pred.dtype)

# Evaluating the model
# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
# Calculate Root Mean Squared Error
rmse = math.sqrt(mse)
# Calculate R² score
r2 = r2_score(y_test, y_pred)
# Print evaluation metrics
print(f'Mean Squared Error: {mse:.4f}')
print(f'Root Mean Squared Error: {rmse:.4f}')
print(f'R² Score: {r2:.4f}')

# Visualizing results
# Create a scatter plot to compare actual vs. predicted values
plt.figure(figsize=(6, 4))
plt.scatter(y_test, y_pred, alpha=0.5, c='blue')
# Add a diagonal line for reference (perfect prediction)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel(f'Actual {yName}')
plt.ylabel(f'Predicted {yName}')
plt.title(f'Actual vs Predicted {yName} Using MLPRegressor')
plt.grid(True)
plt.show()

# Saving the model
# Save the trained model to a file for future use
joblib.dump(model, f'{yName}_mlpnn_model.pkl')

