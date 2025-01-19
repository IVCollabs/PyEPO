'''Script with coordenate plots'''

import pandas as pd
import folium
import os

coordinates = pd.read_csv(os.path.join('input_data', 'coordinates.csv'))
coordinates.dropna(axis=1, inplace=True)

# Creating map

m = folium.Map(location=coordinates[['Latitude', 'Longitude']].mean().to_list(), zoom_start=13)

# Adding markers using apply
add_marker = lambda row: folium.Marker(
    location=[row['Latitude'], row['Longitude']],
    popup=f"Patient {int(row['PatientID'])}",
    icon=folium.Icon(color='blue', icon='info-sign')).add_to(m)
coordinates.apply(add_marker, axis=1)

# Adding lines using apply
# add_line = lambda row: folium.PolyLine(
#     locations=[[row['Latitude'], row['Longitude']], [row['Latitude2'], row['Longitude2']]], 
#     color='red', weight=2.5, opacity=1).add_to(m)
line_coordinates = [
    coordinates[['Latitude', 'Longitude']].iloc[0].to_list(),
    coordinates[['Latitude', 'Longitude']].iloc[1].to_list()
]

# Create a PolyLine object
line = folium.PolyLine(locations=line_coordinates, color='blue', weight=5, opacity=0.8, tooltip='info text')

# Add the line to the map
line.add_to(m)

m.show_in_browser()