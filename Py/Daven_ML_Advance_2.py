#
# Gradient Boosting (Zara)
# Daven
# 2025/5/27
#


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Load data
data = pd.read_csv('zara.csv')

# Select relevant features
features = ['price', 'Promotion', 'Product Category', 'Seasonal', 'Product Position']
label = 'Sales Volume'

# Handle missing values (if any)
data = data.dropna(subset=features + [label])

# Split data into features (X) and target (y)
X = data[features]
y = data[label]

# Preprocess categorical and numerical data
categorical_features = ['Promotion', 'Product Category', 'Seasonal', 'Product Position']
numerical_features = ['price']

# Create transformers
categorical_transformer = OneHotEncoder(drop='first')
numerical_transformer = StandardScaler()

# Combine transformers into a preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Create pipeline with Gradient Boosting Regressor
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42))
])

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Output results
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2): {r2:.2f}")

