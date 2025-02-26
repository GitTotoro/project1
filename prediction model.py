from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

df = "final_data.csv"
X = df[['tavg', 'prcp']]  # Weather features
y = df['total']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)


