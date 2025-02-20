# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 17:00:07 2024
Add a deopt 12/18/2024
@author: xfwang

This is the final version. 
Last updated 12/22/2024

Patient depot fig in the manuscript. 
"""


import folium
import pandas as pd
from geopy.distance import geodesic
import numpy as np

# Load patient data
df_patients = pd.read_csv('Data_30.csv')

# Calculate the center of all patient locations
center_lat = df_patients['Latitude'].mean()
center_lon = df_patients['Longitude'].mean()

# Create a map centered around the average location of all patients
patient_map = folium.Map(location=[center_lat, center_lon], zoom_start=12)

# Add markers for each patient
for i in range(len(df_patients['PatientID'])):
    folium.Marker(
        location=[df_patients['Latitude'][i], df_patients['Longitude'][i]],
        popup=f"PatientID: {df_patients['PatientID'][i]}",
        icon=folium.Icon(color='blue', icon='info-sign')
    ).add_to(patient_map)

# Add a depot marker at the center location
folium.Marker(
    location=[center_lat, center_lon],
    popup="Depot (Center of Patients)",
    icon=folium.Icon(color='red', icon='home')
).add_to(patient_map)

# Save the map to an HTML file
patient_map.save('patient_map_with_depot.html')

