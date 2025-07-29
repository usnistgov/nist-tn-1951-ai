# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 11:59:03 2025

@author: rnc4
"""

import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

# %%
#
# Importing the data
#

df = pd.read_csv('../dataCloud/stats/aimetrics.csv')
print(df.head())
print(df.info())
print(df.describe())

# checking for missing values
print(df.isnull().sum())
df=df.dropna()
print(df.isnull().sum())
# todo: figure why I have missing values

# convert some columns to categorical variables
cat_columns = ['Site', 'Polarization', 'Obstructed', 'Waveguided']
df_encoded = pd.get_dummies(df, columns=cat_columns, drop_first=True)

# scale the numerical values
df_encoded['DelaySpread'] = df_encoded['DelaySpread'] * 1e9 

# Selecting features and target
features = ['TxPos', 'TxHeight', 'Freq', 'Range_m', 'Range_l', 'CoordX', 'CoordY', 'LOS'] + \
           [col for col in df_encoded.columns if col.startswith(('Site_', 'Polarization_', 'Obstructed_', 'Waveguided_'))]
X = df_encoded[features].copy()
#yName = 'LOS'
yName = 'DelaySpread'
#yName = 'PathGain'
y = df_encoded[yName]

# scaling numerical parameters
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
numerical_cols = ['TxPos', 'TxHeight', 'Freq', 'Range_m', 'Range_l', 'CoordX', 'CoordY']
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# %%
#

# try using random forest regressor
#
if yName in cat_columns:
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(random_state=42)
else:        
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
#
# Hyperparameter Tuning
# Optimize the model:
#
if 0:
    from sklearn.model_selection import GridSearchCV
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    print(f'Best Parameters: {grid_search.best_params_}')
    model = grid_search.best_estimator_

# %%
# Train model

model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse:.4f}')
print(f'R^2 Score: {r2:.4f}')

# %%
#
# feature importance
#
importances = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
print(importances.sort_values('Importance', ascending=False))

#
#Visualize Results
#To compare predicted vs. actual PathGain, create a scatter plot. Without sample data, I can’t generate a chart, but here’s the code:
#
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Actual ' + yName)
plt.ylabel('Predicted ' + yName)
plt.title('Actual vs Predicted ' + yName + ' Using Random Forest')
plt.grid(True)
plt.show()

# %%
#save the model
joblib.dump(model, yName+'_randfor_model.pkl')






