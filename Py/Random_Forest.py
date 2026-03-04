#
# RandomForest ML (Zara)
# Daven
# 2025/5/14
#

# Import Libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
za = pd.read_csv('zara.csv') 

# Select features and target
features = ['price', 'Promotion', 'Seasonal', 'Product Position', 'Product Category', 'section']
target = 'Sales Volume'

# Drop rows with missing target or selected features
za = za[features + [target]].dropna() 

# One-hot encode categorical variables
za_encoded = pd.get_dummies(za, columns=['Promotion', 'Seasonal', 'Product Position', 'Product Category', 'section'], drop_first=True)

# Split into features and target
x = za_encoded.drop(columns=[target])
y = za_encoded[target]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Train the model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MSE: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Visualize the results
plt.figure(figsize=(10, 6))
plt.scatter(x=y_test, y=y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
plt.xlabel('Actual Sales Volume')
plt.ylabel('Predicted Sales Volume')
plt.title('Random Forest: Actual vs Predicted Sales Volume')
plt.grid(True)
plt.tight_layout()
plt.show()