from meteostat import Daily, Point, Stations
from datetime import datetime
import pandas as pd
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

#print(df.head())
df.to_csv("final_data.csv")