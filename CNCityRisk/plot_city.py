# plot building footprints of a city

import Utilities

import folium
import folium.plugins
import branca
import webbrowser
import os
import matplotlib.pyplot as plt
from pathlib import Path
import tqdm
import geopandas as gpd

Palette = Utilities.FACECOLOR_HEX # color palette: green blue yellow red


def plot_BldFootprints_properties(CityName, DistrictName = None, prop = None, Center = None, Radius = None, BaseMap = 'Cartodb Positron', savedir = './') -> folium.Map:
    '''
    Plot the building footprints of a city.

    Usage:
        plot_BldFootprints_properties(CityName = '武汉', Radius = 5000): plot a rectangular part of the city.
        plot_BldFootprints_properties(CityName = '武汉'): plot the whole city.
        plot_BldFootprints_properties(CityName = '武汉', DistrictName = '武昌区'): plot the whole district.

    Parameters:
        CityName (str): Name of the city.
        DistrictName (str): Name of the district.
        prop (str): Property to plot. Options: 'story', 'structure', 'occupancy', 'area'
        Center (tuple): Center of the plot.
        Radius (float): Radius of the plot. Unit: meter.
        BaseMap (str): Base map. Options: None, 'Cartodb Positron', 'Amap'. Link to other basemaps: https://leaflet-extras.github.io/leaflet-providers/preview/
        savedir (str): Directory to save the map.
    
    Returns:
        m (folium.Map): The map object.
    '''

    city_polygon = Utilities.get_city_polygon(CityName)
    Center = (city_polygon.centroid.y, city_polygon.centroid.x)

    if Radius is not None and DistrictName is not None:
        raise ValueError("Radius and DistrictName cannot be used together.")
    elif Radius is None and DistrictName is None:
        # plot the whole city
        city_gdf = Utilities.get_CityBldData(CityName)
        rectangle = city_polygon.envelope
    elif Radius is not None and DistrictName is None:
        city_gdf = Utilities.get_CityBldData(CityName)
        rectangle = Utilities.CreateRectangle(Center, Radius, Radius)
    elif Radius is None and DistrictName is not None:
        city_gdf = Utilities.get_DistrictBldData(CityName, DistrictName)
    
    # print('Simplifying building footprints...')
    # city_gdf['geometry'] = city_gdf['geometry'].simplify(tolerance=0.0001) # simplify the geometry so that the map can be loaded faster. 0.0001 is about 10 meters.

    # Create a map
    if BaseMap is None:
        m = folium.Map(tiles=None, location=[Center[1], Center[0]], zoom_start=16)
    elif BaseMap == 'Cartodb Positron':
        m = folium.Map(location=[Center[1], Center[0]], zoom_start=16, tiles='Cartodb Positron')
    elif BaseMap == 'Amap':
        m = folium.Map(location=[Center[1], Center[0]], zoom_start=16, tiles= 'https://wprd01.is.autonavi.com/appmaptile?x={x}&y={y}&z={z}&lang=zh_cn&size=1&scl=1&style=7',attr='高德', max_zoom=19)
    
    # building footprint
    for i, row in tqdm.tqdm(city_gdf.iterrows(), total=city_gdf.shape[0], desc="Loading buildings..."):
        poly = row['geometry']
        coord_bld = list(poly.exterior.coords)
        coord_bld = [(lat, lon) for lon, lat in coord_bld]
        bld_prop = row[prop]
        if prop == 'story':
            bld_prop = int(bld_prop)
            if bld_prop == 1:
                color = Utilities.FACECOLOR_HEX[0]
            elif bld_prop < 6:
                color = Utilities.FACECOLOR_HEX[1]
            elif bld_prop < 10:
                color = Utilities.FACECOLOR_HEX[2]
            else:
                color = Utilities.FACECOLOR_HEX[3]
        elif prop == 'area':
            bld_prop = float(bld_prop)
            if bld_prop < 2000:
                color = Utilities.FACECOLOR_HEX[0]
            elif bld_prop < 5000:
                color = Utilities.FACECOLOR_HEX[1]
            elif bld_prop < 10000:
                color = Utilities.FACECOLOR_HEX[2]
            else:
                color = Utilities.FACECOLOR_HEX[3]
        elif prop == 'structure':
            if bld_prop.startswith('S'):
                # steel
                color = Utilities.FACECOLOR_HEX[0]
            elif bld_prop.startswith('C'):
                # concrete
                color = Utilities.FACECOLOR_HEX[1]
            elif bld_prop.startswith('RM'):
                # reinforced masonry
                color = Utilities.FACECOLOR_HEX[2]
            else:
                color = Utilities.FACECOLOR_HEX[3]
        elif prop == 'occupancy':
            if bld_prop.startswith('RES'):
                # residential
                color = Utilities.FACECOLOR_HEX[0]
            elif bld_prop.startswith('COM'):
                # commercial
                color = color = Utilities.FACECOLOR_HEX[1]
            elif bld_prop.startswith('IND'):
                # industrial
                color = Utilities.FACECOLOR_HEX[2]
            else:
                color = Utilities.FACECOLOR_HEX[3]
        else:
            color = 'gray'
        bld = folium.Polygon(coord_bld, weight = 0, fill_color=color, fill_opacity=1)
        # add a marker for each building
        # bld.add_child(folium.Popup(f'{prop}: {row[prop]}'))
        bld.add_to(m)
    
    # Create a legend
    if prop == 'story':
        legend_html = f'''
            <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 150px; height: 120px; 
                    border:2px solid grey; z-index:9999; font-size:14px;
                    background-color:white;
                    opacity: 0.8;
                    ">
            &nbsp; <b>{prop}</b> <br>
            &nbsp; 1 story &nbsp; <div style="width: 10px; height: 10px; background-color: {Utilities.FACECOLOR_HEX[0]}; display: inline-block;"></div><br>
            &nbsp; 2-5 stories &nbsp; <div style="width: 10px; height: 10px; background-color: {Utilities.FACECOLOR_HEX[1]}; display: inline-block;"></div><br>
            &nbsp; 6-9 stories &nbsp; <div style="width: 10px; height: 10px; background-color: {Utilities.FACECOLOR_HEX[2]}; display: inline-block;"></div><br>
            &nbsp; 10+ stories &nbsp; <div style="width: 10px; height: 10px; background-color: {Utilities.FACECOLOR_HEX[3]}; display: inline-block;"></div>
            </div>
            '''
        m.get_root().html.add_child(branca.element.Element(legend_html))
    elif prop == 'area':
        legend_html = f'''
            <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 150px; height: 120px; 
                    border:2px solid grey; z-index:9999; font-size:14px;
                    background-color:white;
                    opacity: 0.8;
                    ">
            &nbsp; <b>{prop}</b> <br>
            &nbsp; < 2000 m<sup>2</sup> &nbsp; <div style="width: 10px; height: 10px; background-color: {Utilities.FACECOLOR_HEX[0]}; display: inline-block;"></div><br>
            &nbsp; 2000-5000 m<sup>2</sup> &nbsp; <div style="width: 10px; height: 10px; background-color: {Utilities.FACECOLOR_HEX[1]}; display: inline-block;"></div><br>
            &nbsp; 5000-10000 m<sup>2</sup> &nbsp; <div style="width: 10px; height: 10px; background-color: {Utilities.FACECOLOR_HEX[2]}; display: inline-block;"></div><br>
            &nbsp; 10000+ m<sup>2</sup> &nbsp; <div style="width: 10px; height: 10px; background-color: {Utilities.FACECOLOR_HEX[3]}; display: inline-block;"></div>
            </div>
            '''
        m.get_root().html.add_child(branca.element.Element(legend_html))
    elif prop == 'structure':
        legend_html = f'''
            <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 150px; height: 120px; 
                    border:2px solid grey; z-index:9999; font-size:14px;
                    background-color:white;
                    opacity: 0.8;
                    ">
            &nbsp; <b>{prop}</b> <br>
            &nbsp; Steel &nbsp; <div style="width: 10px; height: 10px; background-color: {Utilities.FACECOLOR_HEX[0]}; display: inline-block;"></div><br>
            &nbsp; Concrete &nbsp; <div style="width: 10px; height: 10px; background-color: {Utilities.FACECOLOR_HEX[1]}; display: inline-block;"></div><br>
            &nbsp; Masonry &nbsp; <div style="width: 10px; height: 10px; background-color: {Utilities.FACECOLOR_HEX[2]}; display: inline-block;"></div><br>
            &nbsp; Others &nbsp; <div style="width: 10px; height: 10px; background-color: {Utilities.FACECOLOR_HEX[3]}; display: inline-block;"></div>
            </div>
            '''
        m.get_root().html.add_child(branca.element.Element(legend_html))
    elif prop == 'occupancy':
        legend_html = f'''
            <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 150px; height: 120px; 
                    border:2px solid grey; z-index:9999; font-size:14px;
                    background-color:white;
                    opacity: 0.8;
                    ">
            &nbsp; <b>{prop}</b> <br>
            &nbsp; Residential &nbsp; <div style="width: 10px; height: 10px; background-color: {Utilities.FACECOLOR_HEX[0]}; display: inline-block;"></div><br>
            &nbsp; Commercial &nbsp; <div style="width: 10px; height: 10px; background-color: {Utilities.FACECOLOR_HEX[1]}; display: inline-block;"></div><br>
            &nbsp; Industrial &nbsp; <div style="width: 10px; height: 10px; background-color: {Utilities.FACECOLOR_HEX[2]}; display: inline-block;"></div><br>
            &nbsp; Others &nbsp; <div style="width: 10px; height: 10px; background-color: {Utilities.FACECOLOR_HEX[3]}; display: inline-block;"></div>
            </div>
            '''
        m.get_root().html.add_child(branca.element.Element(legend_html))

    # boundary of the city
    Add_CityBoundary_to_FoliumMap(m, CityName)

    if DistrictName is None:
        # boundary of the rectangle
        coord_rect = list(rectangle.exterior.coords)
        coord_rect = [(lat, lon) for lon, lat in coord_rect]
        rect = folium.PolyLine(coord_rect, color='red', weight=1, dash_array='5')
        rect.add_to(m)

    # map control
    m.add_child(folium.plugins.MeasureControl())
    # m.add_child(folium.LatLngPopup())
    m.add_child(folium.ClickForLatLng(format_str='"[" + lng + "," + lat + "]"', alert=True))

    # Adjust the map bounds
    m.fit_bounds(m.get_bounds())

    if DistrictName is None:
        mapname = f'{CityName}_{prop}'
    else:
        mapname = f'{CityName}_{DistrictName}_{prop}'

    mapfilepath = Path(savedir) / f'{mapname}.html'

    print(f'Saving the map to {mapfilepath}')
    m.save(mapfilepath)
    
    webbrowser.open('file://' + os.path.abspath(mapfilepath))

    return m

def plot_histogram_bld(CityName: str, DistrictName: str = None, prop: str = 'story'):
    '''
    Plot the histogram of building properties.
    Parameters:
        CityName (str): Name of the city.
        prop (str): Property to plot. Options: 'story', 'structure', 'occupancy', 'area'
    '''
    if DistrictName is None:
        city_gdf = Utilities.get_CityBldData(CityName, columns = [prop])
    else:
        city_gdf = Utilities.get_DistrictBldData(CityName, DistrictName, columns = [prop])
    if prop == 'story':
        city_gdf[prop] = city_gdf[prop].astype(int).sort_values()
    elif prop == 'area':
        city_gdf[prop] = city_gdf[prop].astype(float).sort_values()
    T = city_gdf[prop]
    plt.figure()
    plt.hist(T, bins=20, color=Palette[1], edgecolor='black')
    plt.xlabel(prop)
    plt.ylabel('Number of buildings')
    plt.title(f'Histogram of {prop}')
    plt.show()

def Add_CityBoundary_to_FoliumMap(m: folium.Map, CityName: str):
    '''
    Add the boundary of a city to a folium map.

    Parameters:
        m (folium.Map): The map object.
        CityName (str): Name of the city.
    '''
    city_polygon = Utilities.get_city_polygon(CityName)
    for polygon in city_polygon.geoms:
        coord_city = list(polygon.exterior.coords)
        coord_city = [(lat, lon) for lon, lat in coord_city]
        city = folium.PolyLine(coord_city, color='blue', weight=1, dash_array='5')
        city.add_to(m)

def Add_DistrictBoundary_to_FoliumMap(m: folium.Map, CityName: str, DistrictName: str):
    '''
    Add the boundary of a district to a folium map.

    Parameters:
        m (folium.Map): The map object.
        CityName (str): Name of the city.
        DistrictName (str): Name of the district.
    '''
    district_polygon = Utilities.get_district_polygon(CityName, DistrictName)
    for polygon in district_polygon.geoms:
        coord_district = list(polygon.exterior.coords)
        coord_district = [(lat, lon) for lon, lat in coord_district]
        district = folium.PolyLine(coord_district, color='red', weight=1, dash_array='5')
        district.add_to(m)
