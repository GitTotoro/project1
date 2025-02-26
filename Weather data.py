from meteostat import Daily, Point, Stations
from datetime import datetime
import pandas as pd

# Define the location (latitude, longitude) and date range
start = datetime(2019, 7, 11)
end = datetime(2020, 5, 2)

# Get historical weather data (Example: New York City)
seoul = Point(37.532600, 127.024612)
weather_data = Daily(seoul,start, end)
weather_data = weather_data.fetch()

# Convert index (date) to column for merging
weather_data.reset_index(inplace=True)

weather_data["time"] = pd.to_datetime(weather_data["time"])

#get weather data in csv
weather_data.to_csv("seoul_weather.csv")
