from meteostat import Daily, Point, Stations
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Load your data
weather_data = pd.read_csv("seoul_weather.csv")
sales_data = pd.read_csv("Bakery Sales.csv")

# Convert index (date) to column for merging
weather_data.reset_index(inplace=True)

sales_data["datetime"] = pd.to_datetime(sales_data["datetime"], format="%Y.%m.%d %H:%M")
sales_data["date"] = sales_data["datetime"].dt.date

# Aggregate sales per day (sum, mean, etc.)
daily_sales = sales_data.groupby("date").agg({
    "total": "sum"  # You can use "mean" if needed
}).reset_index()

weather_data["date"] = pd.to_datetime(weather_data["time"], format="%Y.%m.%d").dt.date

# Merge datasets on date
df = pd.merge(daily_sales, weather_data, on="date", how="left")

# Drop unnecessary columns
df.drop(columns=["time"], inplace=True)
# Fill NaN values in weather columns with 0
df.fillna(0, inplace=True)

X = df[['tavg', 'prcp']]  # Weather features
y = df['total']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)


plt.scatter(df['tavg'], df['total'], alpha=0.5)
plt.xlabel("Temperature")
plt.ylabel("Sales")
plt.title("Temperature vs Sales")
plt.show()