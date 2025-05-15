#
# Linear Regression ML (Zara)
# Daven
# 2025/5/14
#

# Import Libraries \
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load Data
za = pd.read_csv('zara.csv')
X = za[['price']]  
y = za['Sales Volume']              

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

# Train the model
reg = LinearRegression()
reg.fit(X_train, y_train)

# Make predictions
y_pred = reg.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MSE: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Visualize the results
plt.figure(figsize = (10, 6))
plt.scatter(X_test, y_test, color = 'blue', label = 'Actual')
plt.plot(X_test, y_pred, color = 'red', linewidth=2, label = 'Predicted')
plt.title('Daven_ML')
plt.xlabel('Price') 
plt.ylabel('Sales Volume')
plt.legend()
plt.show()