'''Script with coordenate plots'''

import folium
import numpy as np
import matplotlib.pyplot as plt

def salemen_routes_plot(coordinate_dict:dict, routes: dict):
    """Function to plot salemmen routes.

    Args:
        coordinate_dict (dict): Dictionary where keys are node indices and values are (x, y) coordinates.
        routes (dict): Dictionary where keys are salesmen (int) and values are lists of edges [(i, j), ...].
    """        
    # Creating map
    lat = np.mean([x[0] for x in list(coordinate_dict.values())])
    long = np.mean([x[1] for x in list(coordinate_dict.values())])
    m = folium.Map(location=[lat, long], zoom_start=13)

    # Adding markers in each city
    for idx, city in coordinate_dict.items():
        folium.Marker(location=[city[0], city[1]], popup=f"City {idx}",  icon=folium.Icon(color='blue', icon='info-sign')).add_to(m)


    # Adding routes with diffferent random colors per saleman
    colors = plt.get_cmap('tab20').colors + plt.get_cmap('tab20b').colors + plt.get_cmap('tab20c').colors
    colors = [
        (int(255*rgb[0]*0.7),int(255*rgb[1]*0.7),int(255*rgb[2]*0.7)) 
        for rgb in colors]

    for salesman, edges in routes.items():
        for edge in edges:
            folium.PolyLine(
                locations=[coordinate_dict[edge[0]], coordinate_dict[edge[1]]],
                color='#{:02x}{:02x}{:02x}'.format(*colors[salesman]), weight=2.5, opacity=1).add_to(m)

    m.show_in_browser()