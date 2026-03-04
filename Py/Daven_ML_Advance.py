#
# Linear Regression ML (Zara)
# Daven
# 2025/5/14
#


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv('zara.csv')

# Select features and target
features = ['Promotion', 'Seasonal', 'Product Position', 'Product Category', 'section']
target = 'Sales Volume'

# Drop rows with missing target or selected features
df = df[features + [target]].dropna()

# One-hot encode categorical variables
df_encoded = pd.get_dummies(df, columns=['Promotion', 'Seasonal', 'Product Position', 'Product Category', 'section'], drop_first=True)

# Split into features and target
X = df_encoded.drop(columns=[target])
y = df_encoded[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train linear regression model
reg = LinearRegression()
reg.fit(X_train, y_train)

# Predict
y_pred = reg.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MSE: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Visualize the results
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  
plt.xlabel('Actual Sales Volume')
plt.ylabel('Predicted Sales Volume')
plt.title('Daven ML')
plt.legend()
plt.show()