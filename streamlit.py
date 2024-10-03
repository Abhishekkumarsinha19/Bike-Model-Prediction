import pandas as pd
import numpy as np
import joblib
import streamlit as st

# Load the model
bikel = r"C:\Users\hp\ML file\bike_model\bike_model.pkl"
loaded_model = joblib.load(bikel)

st.header("Bike Rental Prediction")

# Input fields for the model (adjusted to match 9 features)
season = st.selectbox("Enter the Season", ("Spring", "Summer", "Autumn", "Winter"))
season_dict = {"Spring": 1, "Summer": 2, "Autumn": 3, "Winter": 4}
season = season_dict[season]

month = st.selectbox("Enter the Month", ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"))
month_dict = {"January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6, "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12}
month = month_dict[month]

holiday = st.selectbox("Enter the Holiday", ("Public Holiday", "Not a Holiday"))
holiday_dict = {"Public Holiday": 1, "Not a Holiday": 0}
holiday = holiday_dict[holiday]

weekday = st.selectbox("Enter the Weekday", ("Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"))
weekday_dict = {"Sunday": 0, "Monday": 1, "Tuesday": 2, "Wednesday": 3, "Thursday": 4, "Friday": 5, "Saturday": 6}
weekday = weekday_dict[weekday]

workingday = st.selectbox("Enter the Working Day", ("Working Day", "Not a Working Day"))
workingday_dict = {"Working Day": 1, "Not a Working Day": 0}
workingday = workingday_dict[workingday]

weathersit = st.selectbox("Enter the Weathersit", ("Clear", "Mist/Cloudy", "Light Rainfall/Snowfall", "Hailstorm"))
weathersit_dict = {"Clear": 1, "Mist/Cloudy": 2, "Light Rainfall/Snowfall": 3, "Hailstorm": 4}
weathersit = weathersit_dict[weathersit]

# Keep only temperature, excluding 'apparent temperature'
temp = st.number_input("Enter the Temperature") / 75

# Keep other input variables
hum = st.number_input("Enter the Humidity") / 75
windspeed = st.number_input("Enter the Windspeed") / 75

# Input array for prediction with 9 features (excluding 'atemp')
X_new = np.array([[season, month, holiday, weekday, workingday, weathersit, temp, hum, windspeed]])

button = st.button("Submit")

if button:
    try:
        result = loaded_model.predict(X_new)
        result = np.round(result[0])
        st.info("Total Number of Rentals: " + str(result))
    except Exception as e:
        st.error(f"Error during prediction: {e}")
