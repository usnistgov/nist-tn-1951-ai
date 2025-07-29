# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 11:59:03 2025

@author: rnc4
"""

# Import necessary libraries for data processing, modeling, and visualization
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Loading and inspecting the dataset
# Read the dataset from a CSV file
df = pd.read_csv('../dataCloud/stats/aimetrics.csv')
# Display the first few rows of the dataframe
print(df.head())
# Display information about the dataframe (e.g., data types, non-null counts)
print(df.info())
# Display summary statistics of numerical columns
print(df.describe())

# Verifying LOS column (target variable)
# Print the data type of the LOS column
print("LOS Data Type:", df['LOS'].dtype)
# Print unique values in the LOS column
print("LOS Unique Values:", df['LOS'].unique())
# Check for missing or non-finite values in LOS
print("LOS Missing or Non-Finite Values:", df['LOS'].isna().sum(), "NaN,", 
      (df['LOS'].isin([np.inf, -np.inf])).sum(), "Inf")

# Handling non-finite and missing values in LOS (target)
# Calculate the mode of LOS, default to -1 if all values are missing
los_mode = df['LOS'].mode()[0] if not df['LOS'].isna().all() else -1
# Replace infinite values with the mode
df['LOS'] = df['LOS'].replace([np.inf, -np.inf], los_mode)
# Fill missing values with the mode
df['LOS'] = df['LOS'].fillna(los_mode)

# Scaling numerical features
# Convert time-based columns to nanoseconds for consistency
df['DelaySpread'] = df['DelaySpread'] * 1e9 
df['MaxDelay'] = df['MaxDelay'] * 1e9 
df['MeanDelay'] = df['MeanDelay'] * 1e9 

# Handling missing values
# Check for missing values in each column
print(df.isnull().sum())
# Remove rows with missing values
df = df.dropna()
# Verify that no missing values remain
print(df.isnull().sum())
# TODO: Investigate the source of missing values for future improvement

# Converting LOS to numeric format for classification
# Map LOS values to -1 or 1 for binary classification
df['LOS'] = df['LOS'].astype(str).str.lower().map({
    '-1': -1, '1': 1, '+1': 1, 'true': 1, 'false': -1, 'yes': 1, 'no': -1, 'nan': los_mode
}).fillna(los_mode).astype(int)

# Encoding categorical variables
# Define columns to be converted to categorical
cat_columns = ['Site', 'Polarization', 'Obstructed', 'Waveguided']
# Convert categorical columns to dummy variables, dropping the first category to avoid multicollinearity
df_encoded = pd.get_dummies(df, columns=cat_columns, drop_first=True)
# Note: LOS is not included in dummy variables as it's the target

# Defining column names for features
# List all feature columns, including numerical and dummy variables
encoded_col_names = ['TxPos', 'TxHeight', 'Freq', 'Range_m', 'Range_l', 'CoordX', 'CoordY', 'DelaySpread', 'MeanDelay', 'MaxDelay'] + \
           [col for col in df_encoded.columns if col.startswith(('Site_', 'Polarization_', 'Obstructed_', 'Waveguided_', 'LOS_'))]

# Selecting features and target
# Select the target variable (LOS for classification)
which_target = 0
yNames = ['LOS', 'DelaySpread', 'MaxDelay', 'MeanDelay', 'PathGain']
yTypes = ['cat', 'num', 'num', 'num', 'num']
yName = yNames[which_target]
# Extract the target variable
y = df_encoded[yName]
# Select features by excluding the target variable
features = [col for col in encoded_col_names if col != yName]  
# Define numerical columns for scaling
numerical_cols = ['TxPos', 'TxHeight', 'Freq', 'Range_m', 'Range_l', 'CoordX', 'CoordY']
# Extract features from the encoded dataframe
X = df_encoded[features].copy()  # Explicit copy to avoid SettingWithCopyWarning
y = df_encoded[yName]

# Scaling numerical features
# Initialize the StandardScaler
scaler = StandardScaler()
# List numerical columns to scale, excluding the target if it's numerical
numerical_cols_raw = ['TxPos', 'TxHeight', 'Freq', 'Range_m', 'Range_l', 'CoordX', 'CoordY', 'DelaySpread', 'MeanDelay', 'MaxDelay']
numerical_cols = [col for col in numerical_cols_raw if col != yName]
# Apply scaling to numerical columns
X.loc[:, numerical_cols] = scaler.fit_transform(X[numerical_cols])

# Splitting the data
# Split the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the Neural Network model
# Define possible activation and solver options
activ_opts = ['relu', 'tanh', 'logistic', 'identity']
solver_opts = ['adam', 'sgd', 'lbfgs']

# Check if the target is categorical to use MLPClassifier
if yTypes[which_target] == 'cat':
    # Initialize the MLPClassifier with specified parameters
    model = MLPClassifier(hidden_layer_sizes=(500, 50), activation=activ_opts[1], solver=solver_opts[0], 
                         max_iter=1000, random_state=42, early_stopping=True, validation_fraction=0.1)
    # Fit the model to the training data
    model.fit(X_train, y_train)
    # Predict on the test set
    y_pred = model.predict(X_test)
    # Print unique values in actual and predicted outputs for inspection
    print("y_test Unique Values:", np.unique(y_test))
    print("y_pred Unique Values:", np.unique(y_pred))
    # Print data types of actual and predicted outputs
    print("y_test Type:", y_test.dtype)
    print("y_pred Type:", y_pred.dtype)
    # Calculate and print accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.4f}')
    # Print detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['LOS=-1', 'LOS=+1'])) 
    
    # Visualizing confusion matrix
    # Compute the confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=[-1, 1])
    # Create a heatmap for the confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['LOS=-1', 'LOS=+1'], 
                yticklabels=['LOS=-1', 'LOS=+1'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix for LOS Prediction')
    plt.show()

# Visualizing predictions (Note: scatter plot may not be ideal for classification)
if 1:
    # Create a scatter plot to compare actual vs. predicted values
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel('Actual ' + yName)
    plt.ylabel('Predicted ' + yName)
    plt.title('Actual vs Predicted ' + yName + ' Using MLP Classifier')  # Note: Title incorrectly references Random Forest
    plt.grid(True)
    plt.show()

# Saving the model
# Save the trained model to a file for future use
joblib.dump(model, yName+'_mlpnn_model.pkl')

# import pandas as pd
# import numpy as np
# import joblib
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.neural_network import MLPRegressor
# from sklearn.neural_network import MLPClassifier
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# from sklearn.metrics import mean_squared_error, r2_score
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load the dataset
# df = pd.read_csv('../dataCloud/stats/aimetrics.csv')
# print(df.head())
# print(df.info())
# print(df.describe())

# # Verify LOS values and type
# print("LOS Data Type:", df['LOS'].dtype)
# print("LOS Unique Values:", df['LOS'].unique())
# print("LOS Missing or Non-Finite Values:", df['LOS'].isna().sum(), "NaN,", 
#       (df['LOS'].isin([np.inf, -np.inf])).sum(), "Inf")

# # Handle non-finite and missing values in LOS (target)
# los_mode = df['LOS'].mode()[0] if not df['LOS'].isna().all() else -1  # Default to -1 if all NaN
# df['LOS'] = df['LOS'].replace([np.inf, -np.inf], los_mode)
# df['LOS'] = df['LOS'].fillna(los_mode)

# # scale the numerical values
# df['DelaySpread'] = df['DelaySpread'] * 1e9 
# df['MaxDelay'] = df['MaxDelay'] * 1e9 
# df['MeanDelay'] = df['MeanDelay'] * 1e9 

# # checking for missing values
# print(df.isnull().sum())
# df=df.dropna()
# print(df.isnull().sum())
# # todo: figure why I have missing values

# # Convert LOS to numeric (-1 or +1) for target
# df['LOS'] = df['LOS'].astype(str).str.lower().map({
#     '-1': -1, '1': 1, '+1': 1, 'true': 1, 'false': -1, 'yes': 1, 'no': -1, 'nan': los_mode
# }).fillna(los_mode).astype(int)

# # convert some columns to categorical variables
# cat_columns = ['Site', 'Polarization', 'Obstructed', 'Waveguided']
# df_encoded = pd.get_dummies(df, columns=cat_columns, drop_first=True)
# #df_encoded = df_encoded.rename(columns={'LOS_1': 'LOS'})

# # specify the column names
# encoded_col_names = ['TxPos', 'TxHeight', 'Freq', 'Range_m', 'Range_l', 'CoordX', 'CoordY', 'DelaySpread', 'MeanDelay', 'MaxDelay'] + \
#            [col for col in df_encoded.columns if col.startswith(('Site_', 'Polarization_', 'Obstructed_', 'Waveguided_', 'LOS_'))]

# # Selecting features and target
# which_target = 0
# yNames = ['LOS','DelaySpread','MaxDelay','MeanDelay','PathGain']
# yTypes = ['cat','num','num','num','num']
# yName = yNames[which_target]
# y = df_encoded[yName]
# features = [col for col in encoded_col_names if col != yName]  
# numerical_cols = ['TxPos', 'TxHeight', 'Freq', 'Range_m', 'Range_l', 'CoordX', 'CoordY']  # Exclude LOS
# X = df_encoded[features].copy()  # Explicit copy to avoid SettingWithCopyWarning
# y = df_encoded[yName]

# # Scale numerical features
# scaler = StandardScaler()
# numerical_cols_raw = ['TxPos', 'TxHeight', 'Freq', 'Range_m', 'Range_l', 'CoordX', 'CoordY', 'DelaySpread', 'MeanDelay', 'MaxDelay']
# numerical_cols = [col for col in numerical_cols_raw if col != yName]
# X.loc[:, numerical_cols] = scaler.fit_transform(X[numerical_cols])

# # Split data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train Neural Network model
# activ_opts = ['relu','tanh','logistic','identity']
# solver_opts = ['adam','sgd','lbfgs']

# if yTypes[which_target] == 'cat':
#     model = MLPClassifier(hidden_layer_sizes=(5000, 500), activation=activ_opts[0], solver=solver_opts[0], 
#                          max_iter=1000, random_state=42, early_stopping=True, validation_fraction=0.1)
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     y_pred = np.where(y_pred == 0, -1, 1)
#     print("y_test Unique Values:", np.unique(y_test))
#     print("y_pred Unique Values:", np.unique(y_pred))
#     print("y_test Type:", y_test.dtype)
#     print("y_pred Type:", y_pred.dtype)
#     accuracy = accuracy_score(y_test, y_pred)
#     print(f'Accuracy: {accuracy:.4f}')
#     print("\nClassification Report:")
#     print(classification_report(y_test, y_pred, target_names=['LOS=-1', 'LOS=+1'])) 
    
#     # Confusion matrix
#     cm = confusion_matrix(y_test, y_pred, labels=[-1, 1])
#     plt.figure(figsize=(6, 4))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['LOS=-1', 'LOS=+1'], 
#                 yticklabels=['LOS=-1', 'LOS=+1'])
#     plt.xlabel('Predicted')
#     plt.ylabel('Actual')
#     plt.title('Confusion Matrix for LOS Prediction')
#     plt.show()
    

# if 1:
#     # Visualize predictions
#     plt.scatter(y_test, y_pred, alpha=0.5)
#     plt.xlabel('Actual ' + yName)
#     plt.ylabel('Predicted ' + yName)
#     plt.title('Actual vs Predicted ' + yName + ' Using Random Forest')
#     plt.grid(True)
#     plt.show()

# #save the model
# joblib.dump(model, yName+'_mlpnn_model.pkl')