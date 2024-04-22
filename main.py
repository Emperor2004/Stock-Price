# Step 1: Data Collection
import pandas as pd
import numpy as np
import yfinance as yf

# Define the ticker symbol
ticker_symbol = "AAPL"

# Download historical data
data = yf.download(ticker_symbol, start="2020-01-01", end="2023-01-01")

# Step 2: Data Preprocessing
# We'll use the adjusted closing price as our target variable
data = data[['Adj Close']]
data.dropna(inplace=True)

# Step 3: Model Training
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Define features (X) and target variable (y)
X = np.array(range(1, len(data) + 1)).reshape(-1, 1)
y = data.values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 4: Model Evaluation
# Make predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Evaluate the model
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
r2 = r2_score(y_test, y_test_pred)

print(f"Train RMSE: {train_rmse:.2f}")
print(f"Test RMSE: {test_rmse:.2f}")
print(f"R^2 Score: {r2:.2f}")

# Step 5: Prediction
# Predict the stock price for the next day
next_day = np.array([[len(data) + 1]])
predicted_price = model.predict(next_day)
print(f"Predicted price for the next day: {predicted_price[0][0]:.2f}")

import matplotlib.pyplot as plt

# Plot the training data
plt.figure(figsize=(12, 6))
plt.plot(X_train, y_train, color='blue', label='Training Data')

# Plot the testing data
plt.plot(X_test, y_test, color='green', label='Testing Data')

# Plot the regression line
plt.plot(X_train, y_train_pred, color='red', linewidth=3, label='Regression Line')

plt.title('Stock Price Prediction using Linear Regression')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()
