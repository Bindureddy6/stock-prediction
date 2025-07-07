# Import required libraries
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Download stock data
ticker = 'AAPL'
data = yf.download(ticker, start='2010-01-01', end='2023-12-31')
data.reset_index(inplace=True)

# Feature Engineering - Moving Averages and Target
data['MA50'] = data['Close'].rolling(window=50).mean()
data['MA200'] = data['Close'].rolling(window=200).mean()
data['Target'] = data['Close'].shift(-1)

# Drop missing values
data.dropna(inplace=True)

# Prepare features and labels
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA50', 'MA200']
X = data[features]
y = data['Target']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)

# Linear Regression Evaluation
lr_mse = mean_squared_error(y_test, lr_predictions)
lr_mae = mean_absolute_error(y_test, lr_predictions)

# Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

# Random Forest Evaluation
rf_mse = mean_squared_error(y_test, rf_predictions)
rf_mae = mean_absolute_error(y_test, rf_predictions)

# Visualization: Actual vs Predicted
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_test, lr_predictions, edgecolor='k', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Linear Regression Predictions')

plt.subplot(1, 2, 2)
plt.scatter(y_test, rf_predictions, edgecolor='k', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Random Forest Predictions')

plt.tight_layout()
plt.show()

# Save models
joblib.dump(lr_model, 'linear_regression_model.pkl')
joblib.dump(rf_model, 'random_forest_model.pkl')

# Additional Feature Engineering
data['Price_Change_Percentage'] = ((data['Close'] - data['Open']) / data['Open']) * 100
data['Price_Direction'] = np.where(data['Price_Change_Percentage'] > 0, 1, 0)

# Visualization: Price Change & Direction
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.histplot(data['Price_Change_Percentage'], bins=30, kde=True, color='skyblue')
plt.title('Price Change Percentage Distribution')
plt.xlabel('Price Change Percentage')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
sns.countplot(x='Price_Direction', data=data, palette='coolwarm')
plt.title('Price Direction Distribution')
plt.xlabel('Price Direction (1 = Up, 0 = Down)')
plt.ylabel('Count')

plt.tight_layout()
plt.show()

# Additional Evaluation Metrics
lr_r2 = lr_model.score(X_test, y_test)
rf_r2 = rf_model.score(X_test, y_test)

lr_adj_r2 = 1 - (1 - lr_r2) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)
rf_adj_r2 = 1 - (1 - rf_r2) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)

print(f'Linear Regression R2 Score: {lr_r2:.4f}')
print(f'Random Forest R2 Score: {rf_r2:.4f}')
print(f'Linear Regression Adjusted R2 Score: {lr_adj_r2:.4f}')
print(f'Random Forest Adjusted R2 Score: {rf_adj_r2:.4f}')
