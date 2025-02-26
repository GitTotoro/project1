import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Simulated sales and weather data
data = {
    'Temperature': np.random.uniform(0, 40, 100),
    'Humidity': np.random.uniform(20, 90, 100),
    'Rainfall': np.random.uniform(0, 50, 100),
    'Sales': np.random.uniform(500, 5000, 100)
}
df = pd.DataFrame(data)

# Split data into train and test sets
X = df[['Temperature', 'Humidity', 'Rainfall']]
y = df['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print results
print(f'MAE: {mae:.2f}')
print(f'MSE: {mse:.2f}')
print(f'R2 Score: {r2:.2f}')

# Visualizing feature importance
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
feature_importances.sort_values().plot(kind='barh', title='Feature Importance')
plt.show()
