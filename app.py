import streamlit as st
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
import plotly.express as px

# Title of the app
st.title("DQDV Test Zone Analysis")

# Automatically load the CSV file from the GitHub repository
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/CellintelLog9/DqDv_Analysis/main/2w_jayesh.csv"
    return pd.read_csv(url)

# Load the data
charging_data = load_data()

# Data Preprocessing
charging_data['Timestamp'] = pd.to_datetime(charging_data['Timestamp'], format='%d-%m-%y %H:%M')
charging_data = charging_data.dropna(subset=['Timestamp'])
charging_data = charging_data.sort_values(by='Timestamp').reset_index(drop=True)

# Calculate capacity and timestamps for each zone
zone_capacity = []
zone_start_time = []
zone_end_time = []

for zone in charging_data['zone'].unique():
    zone_data = charging_data[charging_data['zone'] == zone]
    time_diff = zone_data['Timestamp'].diff().dt.total_seconds() / 3600  # Convert to hours
    capacity = (zone_data['Current'] * time_diff).sum()
    start_time = zone_data['Timestamp'].min()
    end_time = zone_data['Timestamp'].max()
    zone_capacity.append(capacity)
    zone_start_time.append(start_time)
    zone_end_time.append(end_time)

# Create a DataFrame to store the results
zone_summary = pd.DataFrame({
    'Zone': charging_data['zone'].unique(),
    'Capacity(Ah) added in each charging session': zone_capacity,
    'Start Timestamp': zone_start_time,
    'End Timestamp': zone_end_time
})

# Sort the zone summary by capacity and filter zones with capacity less than 40 Ah
zone_summary = zone_summary.sort_values(by='Capacity(Ah)', ascending=False)
zone_summary = zone_summary[zone_summary['Capacity(Ah)'] < 39]

st.subheader("Zone Summary")
zone_summary = zone_summary.sort_values(by='Capacity(Ah)', ascending=False)
st.dataframe(zone_summary)

# Filter charging_data to include only the filtered zones
charging_data = charging_data[charging_data['zone'].isin(zone_summary['Zone'])]

# Gaussian Smoothing function
def gaussian_smooth(data, sigma, truncate):
    return gaussian_filter1d(data, sigma=sigma, truncate=truncate)

# Process data function
def process_data(df, matched_array):
    voltage_array = df['Voltage'].values
    matching_indices = []
    i = 0
    while i < len(voltage_array):
        diffs = np.round(voltage_array[i+1:] - voltage_array[i], 4)
        matches = np.isin(diffs, matched_array)
        if np.any(matches):
            matching_indices.append(i)
            i += np.argmax(matches) + 1
        else:
            i += 1

    new_df = df.iloc[matching_indices]
    new_df['dqdv'] = (new_df['Capacity(Ah)'].diff() / new_df['Voltage'].diff()).shift(-1)
    new_df['Gaussian_smooth'] = gaussian_smooth(new_df['dqdv'], 3, 1)
    return new_df

# Apply process_data for each zone
def process_data_for_zones(df, matched_array):
    processed_zones = df.groupby('zone').apply(lambda x: process_data(x, matched_array))
    processed_zones = processed_zones.reset_index(drop=True)
    return processed_zones

# Define your matched array
charge_matched_array = np.array([0.2, 0.3])

# Apply the processing to each zone
processed_df = process_data_for_zones(charging_data, charge_matched_array)

# Zone selection
st.sidebar.subheader("Select Zones to Plot")
all_zones = processed_df['zone'].unique()
selected_zones = st.sidebar.multiselect("Select Zones", options=all_zones, default=all_zones[:3])

# Filter the processed_df based on the selected zones
filtered_df = processed_df[processed_df['zone'].isin(selected_zones)]

# Plotting
st.subheader("Voltage vs dQ/dV by Selected Zones")
csv_data = charging_data.to_csv(index=False)

# Add a download button for the original CSV file
st.download_button(
    label="Download Original CSV",
    data=csv_data,
    file_name='2w_jayesh.csv',
    mime='text/csv'
)

if not filtered_df.empty:
    fig = px.line(filtered_df, x='Voltage', y='dqdv', color='zone', title='Voltage vs dQ/dV by Selected Zones')
    st.plotly_chart(fig)
else:
    st.write("No data available for the selected zones.")
