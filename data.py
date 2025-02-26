from meteostat import Daily, Point, Stations
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

# Define the location (latitude, longitude) and date range
start = datetime(2019, 7, 11)
end = datetime(2020, 5, 2)

# Get historical weather data (Example: New York City)
seoul = Point(37.532600, 127.024612)
weather_data = Daily(seoul,start, end)
weather_data = weather_data.fetch()

# Convert index (date) to column for merging
#weather_data.reset_index(inplace=True)

# Load your sales data (example CSV file)
#sales_data = pd.read_csv("sales_data.csv")

# Ensure both datasets have a 'date' column for merging
#sales_data["date"] = pd.to_datetime(sales_data["date"])
#weather_data["time"] = pd.to_datetime(weather_data["time"])

# Merge datasets on date
#df = pd.merge(sales_data, weather_data, left_on="date", right_on="time", how="inner")

# Drop unnecessary columns
#df.drop(columns=["time"], inplace=True)

#print(df.head())
