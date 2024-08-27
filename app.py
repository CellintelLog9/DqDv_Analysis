import streamlit as st
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
import plotly.express as px

# Title of the app
st.title("SOH Test Cyclewise Analysis")

# Automatically load the CSV file from the GitHub repository
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/CellintelLog9/DqDv_Analysis/main/dqdv_2w.csv"
    return pd.read_csv(url)

# Load the data
charging_data = load_data()
charging_data.info()
# Data Preprocessing
charging_data['Timestamp'] = pd.to_datetime(charging_data['Timestamp'], format='%d-%m-%y %H:%M')
charging_data = charging_data.dropna(subset=['Timestamp'])

charging_data = charging_data.sort_values(by='Timestamp').reset_index(drop=True)

# Calculate capacity and timestamps for each zone
zone_capacity = []
zone_start_time = []
zone_end_time = []

# Calculate capacity and timestamps for each zone
for zone in charging_data['zone'].unique():
    zone_data = charging_data[charging_data['zone'] == zone]
    
    # Calculate time difference between consecutive rows in hours
    time_diff = zone_data['Timestamp'].diff().dt.total_seconds() / 3600  # Convert to hours
    
    # Calculate the capacity for the zone
    capacity = (zone_data['Current'] * time_diff).sum()
    
    # Determine the start and end timestamps for the zone
    start_time = zone_data['Timestamp'].min()
    end_time = zone_data['Timestamp'].max()
    
    # Append the results to the lists
    zone_capacity.append(capacity)
    zone_start_time.append(start_time)
    zone_end_time.append(end_time)

# Create a DataFrame to store the results
zone_summary = pd.DataFrame({
    'Zone': charging_data['zone'].unique(),
    'Capacity(Ah)': zone_capacity,
    'Start Timestamp': zone_start_time,
    'End Timestamp': zone_end_time
})

# Show the zone summary sorted by capacity
st.subheader("Zone Summary")
zone_summary = zone_summary.sort_values(by='Capacity(Ah)', ascending=False)
st.dataframe(zone_summary)

# Gaussian Smoothing function
# def gaussian_smooth(data, sigma, truncate):
#     return gaussian_filter1d(data, sigma=sigma, truncate=truncate)

# Process data function
# def process_data(df, matched_array):
#     voltage_array = df['Voltage'].values
#     matching_indices = []
#     i = 0
#     while i < len(voltage_array):
#         diffs = np.round(voltage_array[i+1:] - voltage_array[i], 4)
#         matches = np.isin(diffs, matched_array)
#         if np.any(matches):
#             matching_indices.append(i)
#             i += np.argmax(matches) + 1
#         else:
#             i += 1

#     new_df = df.iloc[matching_indices]
#     new_df['dqdv'] = (new_df['Capacity(Ah)'].diff() / new_df['Voltage'].diff()).shift(-1)
#     new_df['Gaussian_smooth'] = gaussian_smooth(new_df['dqdv'], 3, 1)
#     return new_df

# Apply process_data for each zone
# def process_data_for_zones(df, matched_array):
#     processed_zones = df.groupby('zone').apply(lambda x: process_data(x, matched_array))
#     processed_zones = processed_zones.reset_index(drop=True)
#     return processed_zones

# Define your matched array
# charge_matched_array = np.array([0.2, 0.3])
# processed_df = process_data_for_zones(charging_data, charge_matched_array)


processed_df=charging_data.copy()
# Zone selection
st.sidebar.subheader("Select Zones to Plot")
all_zones = processed_df['zone'].unique()
selected_zones = st.sidebar.multiselect("Select Zones", options=all_zones, default=all_zones[:3])

# Filter the processed_df based on the selected zones
filtered_df = processed_df[processed_df['zone'].isin(selected_zones)]

# Plotting
st.subheader("Voltage vs dQ/dV by Selected Zones")
if not filtered_df.empty:
    fig = px.line(filtered_df, x='Voltage', y='dqdv', color='zone', title='Voltage vs dQ/dV by Selected Zones')
    st.plotly_chart(fig)
else:
    st.write("No data available for the selected zones.")
