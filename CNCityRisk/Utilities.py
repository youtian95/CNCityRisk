from pathlib import Path
import math
import geopandas as gpd
from shapely.geometry import Polygon, Point
import addressparser
from pypinyin import lazy_pinyin
from joblib import load
import pandas as pd
from pyproj import Transformer
from tqdm import tqdm
import requests
import configparser
import rasterio
import h5py
import sys
import BldStructPred
import numpy as np
from packaging import version
import re


# API key
config = configparser.ConfigParser()
config_file = Path('config.ini')
if not config_file.exists():
    raise FileNotFoundError(f"Configuration file {config_file} not found.")
config.read(config_file)
try:
    API_KEY_OPENTOPOGRAPHY = config['API']['api_key_OpenTopography']
except KeyError:
    raise KeyError("API key 'api_key_OpenTopography' not found in the configuration file.")

# Path to the data
CHINA_ADMINDIVISON_SHP = (Path(__file__).parent / './Data/ChinaAdminDivisonSHP-24.02.06').resolve()
BLD_FOOTPRINT_PATH = (Path(__file__).parent / './Data/CNBldFootprintHeight_Che2024').resolve()
POI_AMAP_PATH = (Path(__file__).parent / './Data/POI_AMAP_2020').resolve()
BLD_DATA_PATH = (Path(__file__).parent / './Data/BldData').resolve()
np_version_obj = version.parse(np.__version__)
if np_version_obj < version.parse('1.27.0'):
    TRAINED_RF_BLDSTRUCT = Path(BldStructPred.__file__).parent / 'data/TrainedRF_numpy_v_1_26.joblib'
else:
    TRAINED_RF_BLDSTRUCT = Path(BldStructPred.__file__).parent / 'data/TrainedRF.joblib'
TRAINED_RF_BLDSTRUCT = TRAINED_RF_BLDSTRUCT.resolve()
MAP_POI_OCCUPANCY_PATH = (Path(__file__).parent / './Data/高德POI分类与HazusOccupancy对照表.xlsx').resolve()
GB50011_A_FILE = (Path(__file__).parent / './Data/GB50011-2010(2016)-Appendix-A.csv').resolve()
VS30FILE = (Path(__file__).parent / './Data/China_Mainland_SCK_Vs30.xlsx').resolve()   


# Colors for the plots
# green blue yellow red
FACECOLOR_RGB = [[0.4660,0.6740,0.1880], [0,0.4470,0.7410], [0.9290,0.6940,0.1250], [0.6350,0.0780,0.1840]]
FACECOLOR_HEX = ['#77ac30', '#005ea3', '#edb100', '#a2002c']

# radius of the earth
R_EARTH_M = 6378137

def CreateRectangle(center, length, width) -> Polygon:
    '''
    Create a rectangle polygon using center coordinates, length, and width.
    Parameters:
        Center ((lon, lat)): Center of the rectangle.
        length (float): Length of the rectangle. Unit: meter.
        width (float): Width of the rectangle. Unit: meter.
    Returns:
        rectangle (shapely.geometry.polygon.Polygon): The rectangle polygon. lon, lat in WGS84.
    '''
    center_lon, center_lat = center
    half_side_length_lon = length / 2 / math.cos(center_lat * math.pi / 180) / 6371393 * 180 / math.pi
    half_side_length_lat = width / 2 / 6371393 * 180 / math.pi
    min_lon = center_lon - half_side_length_lon
    max_lon = center_lon + half_side_length_lon
    min_lat = center_lat - half_side_length_lat
    max_lat = center_lat + half_side_length_lat
    rectangle = Polygon([(min_lon, min_lat), (min_lon, max_lat), (max_lon, max_lat), (max_lon, min_lat)])
    return rectangle

def height2story(height, Round = True):
    '''
    Convert the height of a building to the number of stories.
    Parameters:
        height (float): Height of a building in meters.
        Round (bool): If True, round the number of stories to the nearest integer.
    Returns:
        story (int): Number of stories.
    '''
    if Round:
        story = round(height/3.0)

        if story < 0:
            raise ValueError('The height of the building is negative.')
        elif story == 0:
            story = 1
    else:
        story = height/3.0

    return story

def get_DistrictsOfCity_polygons(CityName) -> gpd.GeoDataFrame:
    '''
    Get the polygons of districts of a city from the shapefile.
    Parameters:
        CityName (str): Name of the city.
    Returns:
        districts_gdf (geopandas.geodataframe.GeoDataFrame): The polygons of districts of the city.
    '''
    district_shapefile = gpd.read_file(CHINA_ADMINDIVISON_SHP/'4. District'/'district.shp')

    districts_gdf = district_shapefile[district_shapefile['ct_name'].str.contains(CityName)]
    if districts_gdf.empty:
        raise ValueError(f"City {CityName} not found in the shapefile.")

    return districts_gdf

def get_DistrictsOfCity_names(CityName) -> list:
    '''
    Get the names of districts of a city from the shapefile.
    Parameters:
        CityName (str): Name of the city.
    Returns:
        DistrictNames (list): Names of districts of the city.
    '''
    district_shapefile = gpd.read_file(CHINA_ADMINDIVISON_SHP/'4. District'/'district.shp')

    DistrictNames = district_shapefile[district_shapefile['ct_name'].str.contains(CityName)]['dt_name'].tolist()
    if not DistrictNames:
        raise ValueError(f"City {CityName} not found in the shapefile.")

    return DistrictNames

def get_district_polygon(CityName, DistrictName):
    '''
    Get the polygon of a district from the shapefile.
    Parameters:
        CityName (str): Name of the city.
        DistrictName (str): Name of the district.
    Returns:
        district_polygon (shapely.geometry.polygon.Polygon or shapely.geometry.multipolygon.MultiPolygon): The polygon of the district.
    '''
    district_shapefile = gpd.read_file(CHINA_ADMINDIVISON_SHP/'4. District'/'district.shp')

    district_boundary = district_shapefile[(district_shapefile['ct_name'].str.contains(CityName)) & (district_shapefile['dt_name'].str.contains(DistrictName))]

    if district_boundary.empty:
        raise ValueError(f"District {DistrictName} not found in the shapefile.")

    district_polygon = district_boundary.geometry.iloc[0]

    return district_polygon

def get_city_polygon(CityName):
    '''
    Get the polygon of a city from the shapefile.
    Parameters:
        CityName (str): Name of the city.
    Returns:
        city_polygon (shapely.geometry.polygon.Polygon or shapely.geometry.multipolygon.MultiPolygon): The polygon of the city.
    '''
    
    city_shapefile = gpd.read_file(CHINA_ADMINDIVISON_SHP/'3. City'/'city.shp')
    
    city_boundary = city_shapefile[city_shapefile['ct_name'].str.contains(CityName)]
    
    if city_boundary.empty:
        raise ValueError(f"City {CityName} not found in the shapefile.")
    
    city_polygon = city_boundary.geometry.iloc[0]
    
    return city_polygon

def get_DistrictBldData(CityName, DistrictName, columns = None):    
    '''
    Get the building data of a district.
    Parameters: 
        CityName (str): Name of the city.
        DistrictName (str): Name of the district.
        columns (list): Columns to read from the csv file. Default is None, which reads all columns.
    Returns:
        district_gdf (geopandas.geodataframe.GeoDataFrame): The building data of the district. Columns include: longitude, latitude, story, area, structure, occupancy, designlevel.
    '''
    try:
        district_gdf = ReadDistrictBldGDF(CityName, DistrictName)
        if columns is not None:
            district_gdf = district_gdf[columns]
        return district_gdf
    except FileNotFoundError:
        raise ValueError(f"District {DistrictName} data not found for city {CityName}.")
    
def get_rows_CityBldData(CityName) -> int:
    '''
    Get the number of rows of the building data of a city.
    Parameters:
        CityName (str): Name of the city.
    Returns:
        row_count (int): Number of rows in the building data of the city.
    '''
    row_count = 0

    # read DistrictNames
    CityDir = BLD_DATA_PATH / CityName
    if not CityDir.exists():
        raise ValueError(f"City {CityName} data not found.")
    DistrictNames = [f.stem for f in CityDir.glob('*.csv')]

    for DistrictName in DistrictNames:
        row_district_gdf = Get_rows_DistrictBldGDF(CityName, DistrictName)
        row_count += len(row_district_gdf)

    return row_count

def get_CityBldData(CityName, columns = None) -> gpd.GeoDataFrame:
    '''
    Get the building data of a city.
    Parameters:
        CityName (str): Name of the city.
        columns (list): Columns to read from the csv file. Default is None, which reads all columns.
    Returns:
        city_gdf (geopandas.geodataframe.GeoDataFrame): The building data of the city. Columns include: longitude, latitude, story, area, structure, occupancy, designlevel.
    '''
    city_gdf = gpd.GeoDataFrame()

    # read DistrictNames
    CityDir = BLD_DATA_PATH / CityName
    if not CityDir.exists():
        raise ValueError(f"City {CityName} data not found.")
    DistrictNames = [f.stem for f in CityDir.glob('*.csv')]

    for DistrictName in DistrictNames:
        district_gdf = get_DistrictBldData(CityName, DistrictName, columns)
        city_gdf = pd.concat([city_gdf, district_gdf], ignore_index=True)

    return city_gdf

def _GetCityBldData_FromSHP(CityName, Center = None, Radius = 2500):
    '''
    Get the building data of a city area.
    Parameters:
        CityName (str): Name of the city.
        Center ((lon, lat)): Center of the building portfolios.
        Radius (float): Radius of the building portfolios. Unit: meter.
    Returns:
        city_gdf (geopandas.geodataframe.GeoDataFrame): The building data of the city area.
        city_polygon (shapely.geometry.polygon.Polygon or shapely.geometry.multipolygon.MultiPolygon): The polygon of the city.
        rectangle (shapely.geometry.polygon.Polygon): The rectangle mask of the city.
        Center ((lon, lat)): Center of the building portfolios.
    '''

    city_polygon = get_city_polygon(CityName)
    if Center is None:
        Center = city_polygon.centroid.coords[0]

    # Get the province name
    df = addressparser.transform([CityName])
    ProfName = df['省'][0].replace('省', '')
    ProfName = ''.join(lazy_pinyin(ProfName))

    # Create a rectangle mask using center coordinates and radius
    rectangle = CreateRectangle(Center, Radius * 2, Radius * 2)

    # Read the building data
    prof_file = list(BLD_FOOTPRINT_PATH.glob(f'**/{ProfName}.shp'))[0]
    city_gdf: gpd.GeoDataFrame = gpd.read_file(prof_file, bbox=rectangle)

    # make city_gdf.columns lower case
    city_gdf.columns = city_gdf.columns.str.lower()

    # building height to story
    city_gdf['story'] = city_gdf['height'].apply(height2story)

    # change the crs to UTM
    city_gdf.set_crs(epsg=4326, inplace=True)
    utm_crs = city_gdf.estimate_utm_crs()
    city_gdf.to_crs(utm_crs, inplace=True)
    # building centroid
    city_gdf['centroid'] = city_gdf['geometry'].centroid
    city_gdf['centroid'] = city_gdf['centroid'].to_crs(epsg=4326)
    # building area
    city_gdf['area'] = city_gdf['geometry'].area
    # change the crs back to WGS84
    city_gdf.to_crs(epsg=4326, inplace=True)

    return city_gdf, city_polygon, rectangle, Center

def __SearchPOIForBlds(Bld_gdf: gpd.GeoDataFrame, POI_gdf: gpd.GeoDataFrame, radius=2000, types='商务住宅|住宿服务', maxPOI=20):
    '''
    Get POI for buildings.
    Parameters:
        Bld_gdf (geopandas.geodataframe.GeoDataFrame): The building data of the city. Columns include: longitude, latitude, geometry
        POI_gdf (geopandas.geodataframe.GeoDataFrame): The POI data of the city. Columns include: geometry, 大类, 中类, 小类
        radius: float
        types: str. '|' separated POI types. e.g. '商务住宅|住宿服务'
        maxPOI: int. Maximum number of POI to return.
    Returns:
        Bld_gdf (geopandas.geodataframe.GeoDataFrame): The building data of the city. Columns include: longitude, latitude, POI, geometry
            POI: list, the POI data of the building. [n_poi, 4] . e.g. [[Dist1, Cat1, Cat2, Cat3], ...]
    '''

    # create POI column if not exist
    if 'POI' not in Bld_gdf.columns:
        Bld_gdf['POI'] = None

    if types is not None:
        types = types.split('|')
    
    # change the crs to UTM so that the distance can be calculated in meters
    POI_gdf.set_crs(epsg=4326, inplace=True)
    utm_crs = POI_gdf.estimate_utm_crs()
    POI_gdf.to_crs(utm_crs, inplace=True)
    def filter_POI(row, POI_gdf: gpd.GeoDataFrame, radius, types, maxPOI):
        # center of the building
        center = Point(row['longitude'], row['latitude'])
        transformer = Transformer.from_crs("EPSG:4326", utm_crs.srs, always_xy=True)
        center = Point(transformer.transform(center.x, center.y))
        # a rectangle mask centered at the building
        rectangle = Polygon([(center.x - radius, center.y - radius), (center.x - radius, center.y + radius), (center.x + radius, center.y + radius), (center.x + radius, center.y - radius)])
        # filter POI by rectangle
        POI_gdf_filter = POI_gdf[POI_gdf.within(rectangle)].copy(deep=True)
        # distance to the building
        POI_gdf_filter.loc[:, 'distance'] = POI_gdf_filter.distance(center)
        # filter POI by distance
        POI_gdf_filter = POI_gdf_filter[POI_gdf_filter['distance'] < radius]
        # filter POI by types
        if types is not None:
            POI_gdf_filter = POI_gdf_filter[POI_gdf_filter['大类'].isin(types)]
        # reogranize the POI data
        if POI_gdf_filter.shape[0] > 0:
            POI_gdf_filter = POI_gdf_filter.sort_values('distance')
        if POI_gdf_filter.shape[0] > maxPOI:
            POI_gdf_filter = POI_gdf_filter.head(maxPOI)
        # assign the found POI_gdf_filter to Bld_gdf
        POI = POI_gdf_filter[['distance', '大类', '中类', '小类']].values.tolist()
        return POI

    tqdm.pandas(desc='Searching POI...')
    Bld_gdf['POI'] = Bld_gdf.progress_apply(filter_POI, axis=1, POI_gdf=POI_gdf, radius=radius, types=types, maxPOI=maxPOI)

    return Bld_gdf 

def SearchBldPOI_Local(CityName, Bld_gdf, radius=2000, types='商务住宅|住宿服务', maxPOI=20):
    '''
    Search POI around a building
    Parameters:
        CityName (str): Name of the city. This is for searching for the POI data file.
        Bld_gdf (geopandas.geodataframe.GeoDataFrame): The building data of the city. Columns include: longitude, latitude, geometry
        radius: float
        types: str. '|' separated POI types. e.g. '商务住宅|住宿服务'
        maxPOI: int. Maximum number of POI to return.
    Returns:
        Bld_gdf (geopandas.geodataframe.GeoDataFrame): The building data of the city. Columns include: longitude, latitude, POI, geometry
    '''

    CityPOIDir = list(POI_AMAP_PATH.glob(f'**/*{CityName}*'))
    if not CityPOIDir:
        raise Exception(f"POI data not found for city {CityName}.")
    elif len(CityPOIDir) > 1:
        raise Exception(f"Multiple POI data found for city {CityName}.")
    else:
        CityPOIDir = CityPOIDir[0]

    # Merge all POI data within CityPOIDir for the city
    POI_files = list(CityPOIDir.glob('*.xlsx')) + list(CityPOIDir.glob('*.csv'))
    POI_gdf = gpd.GeoDataFrame()
    for POI_file in POI_files:
        if POI_file.suffix == '.xlsx':
            df = pd.read_excel(POI_file)
        elif POI_file.suffix == '.csv':
            df = pd.read_csv(POI_file)
        else:
            raise Exception(f"Unsupported file format: {POI_file.suffix}")
        if 'wgs84lon' in df.columns and 'wgs84lat' in df.columns:
            gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['wgs84lon'], df['wgs84lat']))
        elif 'WGS84_经度' in df.columns and 'WGS84_纬度' in df.columns:
            gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['WGS84_经度'], df['WGS84_纬度']))
        elif '经度' in df.columns and '纬度' in df.columns:
            gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['经度'], df['纬度']))
        else:
            raise Exception(f"Columns 'wgs84lon' and 'wgs84lat' or 'WGS84_经度' and 'WGS84_纬度' not found in {POI_file}.")
        retained_keys = ['大类', '中类', '小类', 'geometry']
        gdf = gdf[retained_keys]
        POI_gdf = pd.concat([POI_gdf, gdf], ignore_index=True)

    # for each building, search for POI
    Bld_gdf = __SearchPOIForBlds(Bld_gdf, POI_gdf, radius=radius, types=types, maxPOI=maxPOI)

    return Bld_gdf

def PredictBldStructuralType(Bld_gdf, UsePOI = False):
    '''
    Predict the structural type of buildings in a city.
    Parameters:
        Bld_gdf (geopandas.geodataframe.GeoDataFrame): The building data of the city. Columns include: longitude, latitude, story, area, geometry, POI
        UsePOI (bool): If True, POI data are used as well to predict the building structure type.
    Returns:
        Bld_gdf (geopandas.geodataframe.GeoDataFrame): The building data of the city. Columns include: longitude, latitude, story, area, structure, geometry, POI
    '''
    Area = Bld_gdf['area'].astype(float).values.tolist()
    Floor = Bld_gdf['height'].astype(float).apply(height2story, Round=False).values.tolist()
    # footprint in utm. Unit: meter
    Footprint = Bld_gdf['geometry']
    Footprint.set_crs(epsg=4326, inplace=True)
    utm_crs = Footprint.estimate_utm_crs()
    Footprint = Footprint.to_crs(utm_crs)
    
    # 修复处理MultiPolygon的问题
    def get_coords(geom):
        from shapely.geometry import MultiPolygon, Polygon
        if isinstance(geom, Polygon):
            return list(geom.exterior.coords)
        elif isinstance(geom, MultiPolygon):
            # 对于MultiPolygon，我们取面积最大的Polygon的坐标
            largest_polygon = max(geom.geoms, key=lambda p: p.area)
            return list(largest_polygon.exterior.coords)
        else:
            # 其他几何类型的处理
            return []
    
    Footprint = Footprint.apply(get_coords)
    Footprint = Footprint.tolist()
    
    # POI
    if UsePOI:
        POI = Bld_gdf['POI'].values.tolist()
    # predict building structure type
    clf = load(TRAINED_RF_BLDSTRUCT)
    if UsePOI:
        Y_struct = clf.predict(Area, Floor, Footprint, POI)
    else:
        Y_struct = clf.predict(Area, Floor, Footprint)
    Bld_gdf['structure'] = Y_struct

    return Bld_gdf

def SearchBldNearestPOI_Local(CityName, Bld_gdf, radius=500):
    '''
    Search the nearest POI for each building.
    Parameters:
        CityName (str): Name of the city.
        Bld_gdf (geopandas.geodataframe.GeoDataFrame): The building data of the city. Columns include: longitude, latitude, story, area, structure, geometry, POI
        radius (float): Radius for searching the nearest POI. Unit: meter.
    Returns:
        Bld_gdf (geopandas.geodataframe.GeoDataFrame): The building data of the city. Columns include: longitude, latitude, story, area, structure, geometry, POI, POI_Nearest
    '''
    print('Searching the nearest POI...')

    pd_POI_to_OCPC = pd.read_excel(MAP_POI_OCCUPANCY_PATH)

    # find all the Cat1 types for each the column 'Hazus Occupancy Class' is not str 'None'
    AllCa1Types = pd_POI_to_OCPC[pd_POI_to_OCPC['Hazus Occupancy Class'] != 'None']['大类'].unique().tolist()

    # find the nearest POI for each building
    Bld_gdf.rename(columns={'POI': 'POI_ForStruct'}, inplace=True)
    Bld_gdf = SearchBldPOI_Local(CityName, Bld_gdf, radius=radius, types='|'.join(AllCa1Types), maxPOI=1)
    Bld_gdf.rename(columns={'POI': 'POI_Nearest'}, inplace=True)
    Bld_gdf.rename(columns={'POI_ForStruct': 'POI'}, inplace=True)

    return Bld_gdf

def DeriveBldOccupancyType(CityName: str, Bld_gdf: gpd.GeoDataFrame):
    '''
    Derive the occupancy type of buildings from POI data.
    Parameters:
        CityName (str): Name of the city.
        Bld_gdf (geopandas.geodataframe.GeoDataFrame): The building data of the city. Columns include: longitude, latitude, story, area, geometry, POI, POI_Nearest
    Returns:
        Bld_gdf (geopandas.geodataframe.GeoDataFrame): The building data of the city. Columns include: longitude, latitude, story, area, structure, occupancy, geometry, POI, POI_Nearest
    '''
    print('Deriving occupancy type...')

    pd_POI_to_OCPC = pd.read_excel(MAP_POI_OCCUPANCY_PATH)

    # construct a dict for mapping POI to occupancy type
    POI_to_OCPC = {}
    for index, row in pd_POI_to_OCPC.iterrows():
        if row['Hazus Occupancy Class'] != 'None':
            if row['大类'] in POI_to_OCPC:
                if row['中类'] in POI_to_OCPC[row['大类']]:
                    POI_to_OCPC[row['大类']][row['中类']][row['小类']] = row['Hazus Occupancy Class']
                else:
                    POI_to_OCPC[row['大类']][row['中类']] = {row['小类']: row['Hazus Occupancy Class']}
            else:
                POI_to_OCPC[row['大类']] = {row['中类']: {row['小类']: row['Hazus Occupancy Class']}}

    # derive the occupancy type
    def get_occupancy_type(poi_nearest: list, poi_to_ocpc: dict):
        if not poi_nearest:
            return None
        elif not poi_nearest[0]:
            return None
        else:
            poi = poi_nearest[0][1:] # [cat1, cat2, cat3]
            if poi[0] in poi_to_ocpc:
                if poi[1] in poi_to_ocpc[poi[0]]:
                    if poi[2] in poi_to_ocpc[poi[0]][poi[1]]:
                        return poi_to_ocpc[poi[0]][poi[1]][poi[2]]
                    else:
                        # return the first deepest value of poi_to_ocpc[poi[0]][poi[1]]
                        cat3_1st = list(poi_to_ocpc[poi[0]][poi[1]].keys())[0]
                        return poi_to_ocpc[poi[0]][poi[1]][cat3_1st]
                else:
                    # return the first deepest value of poi_to_ocpc[poi[0]]
                    cat2_1st = list(poi_to_ocpc[poi[0]].keys())[0]
                    cat3_1st = list(poi_to_ocpc[poi[0]][cat2_1st].keys())[0]
                    return poi_to_ocpc[poi[0]][cat2_1st][cat3_1st]
            else:
                return None
    Bld_gdf['occupancy'] = Bld_gdf['POI_Nearest'].apply(get_occupancy_type, poi_to_ocpc=POI_to_OCPC)

    return Bld_gdf

def Get_rows_DistrictBldGDF(CityName: str, DistrictName: str) -> int:
    '''
    Read the building data of a district from a csv file in the BLD_DATA_PATH.
    Parameters:
        CityName (str): Name of the city.
        DistrictName (str): Name of the district.
    Returns:
        row_count (int): Number of rows in the building data of the district.
    '''
    CityDir = BLD_DATA_PATH / CityName
    DistrictFile = CityDir / f'{DistrictName}.csv'
    if not DistrictFile.exists():
        raise FileNotFoundError(f"District file {DistrictFile} not found.")
    row_count = sum(1 for _ in pd.read_csv(DistrictFile, chunksize=10000))
    return row_count

def ReadDistrictBldGDF(CityName: str, DistrictName: str) -> gpd.GeoDataFrame:
    '''
    Read the building data of a district from a csv file in the BLD_DATA_PATH.
    Parameters:
        CityName (str): Name of the city.
        DistrictName (str): Name of the district.
    Returns:
        Bld_gdf (geopandas.geodataframe.GeoDataFrame): The building data of the district. Columns include: longitude, latitude, story, area, structure, occupancy, designlevel.
    '''
    try:
        Bld_gdf = gpd.read_file(BLD_DATA_PATH / CityName / f'{DistrictName}.csv', GEOM_POSSIBLE_NAMES="geometry", KEEP_GEOM_COLUMNS="NO", encoding='utf-8-sig')
        if 'longitude' in Bld_gdf.columns:
            Bld_gdf['longitude'] = Bld_gdf['longitude'].astype(float)
        if 'latitude' in Bld_gdf.columns:
            Bld_gdf['latitude'] = Bld_gdf['latitude'].astype(float)
        if 'story' in Bld_gdf.columns:
            Bld_gdf['story'] = Bld_gdf['story'].astype(int)
        if 'area' in Bld_gdf.columns:
            Bld_gdf['area'] = Bld_gdf['area'].astype(float)
        if 'POI' in Bld_gdf.columns:
            Bld_gdf['POI'] = Bld_gdf['POI'].apply(eval)
        if 'POI_Nearest' in Bld_gdf.columns:
            Bld_gdf['POI_Nearest'] = Bld_gdf['POI_Nearest'].apply(eval)
        if 'ReprBldID' in Bld_gdf.columns:
            Bld_gdf['ReprBldID'] = Bld_gdf['ReprBldID'].astype(int)
        Bld_gdf['DistrictName'] = DistrictName
        return Bld_gdf
    except:
        return None

def SaveDistrictBldGDF(CityName: str, DistrictName: str, Bld_gdf: gpd.GeoDataFrame):
    CityDir = BLD_DATA_PATH / CityName
    # save to csv
    Bld_gdf.to_csv(CityDir / f'{DistrictName}.csv', index=False, encoding='utf-8-sig')

        
class SearchVs30:
    '''
    A class to search Vs30 data from the VS30FILE
    '''
    def __init__(self):
        self.Vs30DataFile = VS30FILE
        self.T = pd.read_excel(self.Vs30DataFile, header=1)

    def GetVs30(self, lon, lat):
        Dist = (self.T["Longitude (°)"] - lon) ** 2 + (self.T["Latitude (°)"] - lat) ** 2
        row = Dist.idxmin()
        Vs30_mpers = self.T.iloc[row, -1]
        return Vs30_mpers

class DEMFetcher:

    output_dir = (Path(__file__).parent / 'Data/DEM/').resolve()
    

    def __init__(self, api_key=API_KEY_OPENTOPOGRAPHY, dem_type='SRTMGL3', output_format='GTiff'):
        """
        Initialize the DEMFetcher class.
        
        Parameters:
            api_key: str - Your API key from OpenTopography
            dem_type: str - Type of DEM data to fetch (default is 'SRTMGL3')
            output_format: str - Output format of the data (default is 'GTiff')
        """
        self.api_key = api_key
        self.dem_type = dem_type
        self.output_format = output_format
        self.base_url = "https://portal.opentopography.org/API/globaldem"

        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def fetch_dem_data(self, south, north, west, east, Edge=0.01):
        """
        Fetches DEM data from OpenTopography API. 
        Note:
            If the data is already downloaded, it will not be downloaded again, instead it will be read from the file.
            A little bigger bounding box (0.01 degrees) is used to avoid missing data at the edges.
        
        Parameters:
            south: float - South coordinate of the bounding box. Round to 2 decimal places
            north: float - North coordinate of the bounding box. Round to 2 decimal places
            west: float - West coordinate of the bounding box. Round to 2 decimal places
            east: float - East coordinate of the bounding box. Round to 2 decimal places
            Edge: float - Buffer around the bounding box to avoid missing data at the edges (default is 0.01)
        
        Returns:
            DEMdata: rasterio.io.DatasetReader - DEM data as a rasterio dataset
        """

        filename = f'dem_data_{south:.2f}_{north:.2f}_{west:.2f}_{east:.2f}.tif'
        file_path = self.output_dir / filename

        # Check if the file already exists
        if file_path.exists():
            DEMdata = rasterio.open(file_path)
            return DEMdata

        # Parameters for the API request
        params = {
            'demtype': self.dem_type,
            'south': round(south-Edge, 2),
            'north': round(north+Edge, 2),
            'west': round(west-Edge, 2),
            'east': round(east+Edge, 2),
            'outputFormat': self.output_format,
            'API_Key': self.api_key
        }
        
        # Make the GET request to the API
        response = requests.get(self.base_url, params=params)
        
        # Check if the request was successful
        params_str = f"south={south:.2f}, north={north:.2f}, west={west:.2f}, east={east:.2f}"
        if response.status_code == 200:
            print(f"Downloaded DEM data for the given parameters: {params_str}.")
        elif response.status_code == 204:
            raise ValueError(f"No data available for the given parameters: {params_str}.")
        elif response.status_code == 400:
            raise ValueError(f"Bad request. Please check the parameters: {params_str}.")
        elif response.status_code == 401:
            raise PermissionError("Unauthorized. Please check your API key.")
        elif response.status_code == 404:
            raise FileNotFoundError(f"Resource not found for the given parameters: {params_str}.")
        elif response.status_code == 500:
            raise ConnectionError("Internal server error. Please try again later.")
        else:
            raise Exception(f"An unexpected error occurred. Status code: {response.status_code}, Parameters: {params_str}.")
        
        # Save the data to a file
        with open(file_path, 'wb') as file:
            file.write(response.content)

        DEMdata = rasterio.open(file_path)
        return DEMdata
    
    def get_elevation(self, latitude, longitude, buffer = 0.5):
        """
        Get the elevation data for a specific coordinate point.
        
        Parameters:
            latitude: float or list - Latitude of the point
            longitude: float or list - Longitude of the point
            buffer: float - Buffer around the point to create a bounding box (default is 0.5)
        Returns:
            elevation: float or list - Elevation of the point in kilometers
        """
        south = math.floor(latitude / buffer) * buffer
        north = math.ceil(latitude / buffer) * buffer
        west = math.floor(longitude / buffer) * buffer
        east = math.ceil(longitude / buffer) * buffer
        
        DEMcontent = self.fetch_dem_data(south, north, west, east)

        # Get the elevation data
        row, col = DEMcontent.index(longitude, latitude)
        elevation = DEMcontent.read(1)[row, col]
        return elevation

def Calc4CornersEQRupt(W_km, length_km, lat_Hypo, lon_Hypo, RuptureNormal_x, RuptureNormal_y, RuptureNormal_z) -> list:
    """
    Calculate the four corners of the rupture rectangle
    Parameters:
        W_km: float - Width of the rupture in kilometers
        length_km: float - Length of the rupture in kilometers
        lat_Hypo: float - Latitude of the hypocenter
        lon_Hypo: float - Longitude of the hypocenter
        RuptureNormal_x: float - x component of the rupture normal vector
        RuptureNormal_y: float - y component of the rupture normal vector
        RuptureNormal_z: float - z component of the rupture normal vector
    Returns:
        corners: list - List of four corner coordinates. Each corner is a list of [latitude, longitude]
    """

    # Convert rupture normal vector to strike and dip
    strike, dip = RuptureNormal2StrikeDip(RuptureNormal_x, RuptureNormal_y, RuptureNormal_z)

    # Calculate the coordinates of the four corners
    half_length = length_km / 2
    half_width = W_km / 2 * math.cos(math.radians(dip))

    # Calculate the coordinates of the four corners
    corners = []
    for dx, dy in [(-half_length, -half_width), (half_length, -half_width), (half_length, half_width), (-half_length, half_width)]:
        d_lon = dx * math.cos(math.radians(strike)) - dy * math.sin(math.radians(strike))
        d_lat = dx * math.sin(math.radians(strike)) + dy * math.cos(math.radians(strike))
        # from UTM to WGS84
        d_lon = d_lon *1000 / math.cos(math.radians(lat_Hypo)) / R_EARTH_M * 180 / math.pi
        d_lat = d_lat *1000 / R_EARTH_M * 180 / math.pi
        corners.append([lat_Hypo + d_lat, lon_Hypo + d_lon])

    return corners

def CreateMeshedSiteFile(lonMin, lonMax, latMin, latMax, SiteFile = 'SiteFile.txt', MeshDist = 500):
    """
    Generate a meshed site file

    Parameters:
        MeshDist: float - mesh distance, m
        lonMin: float - minimum longitude, degrees
        lonMax: float - maximum longitude, degrees
        latMin: float - minimum latitude, degrees
        latMax: float - maximum latitude, degrees

    Returns:
        IDvec: list - ID of each site
        lonVec: list - corresponding longitude
        latVec: list - corresponding latitude

    The site file format is as follows:
    Each row contains data for one site, with the following fields separated by spaces:
        ID - site ID
        lon - longitude
        lat - latitude
        elevation_km - elevation in km
        period1 - foundamental period
        Vs30_mpers - shear wave velocity
        Z25_km - depth to the 2.5 km/s shear wave velocity horizon in km (if unknown, input 999)
    """

    SiteFile = Path(SiteFile).resolve()

    if not SiteFile.parent.exists():
        SiteFile.parent.mkdir(parents=True, exist_ok=True)

    if SiteFile.exists():
        SiteFile.unlink()

    # Calculate grid points
    R = 6374000
    lonStep = MeshDist / R / math.cos(math.radians(latMin)) * 180 / math.pi
    latStep = MeshDist / R * 180 / math.pi
    Nlon = math.ceil((lonMax - lonMin) / lonStep)
    Nlat = math.ceil((latMax - latMin) / latStep)

    # Generate site file
    IDvec = []
    lonVec = []
    latVec = []
    TVec = []
    ID = 0

    obj = SearchVs30()
    obj_DEM = DEMFetcher()
    for row in range(Nlon):
        for col in range(Nlat):
            lon = lonMin + row * lonStep
            lat = latMin + col * latStep
            elevation_km = obj_DEM.get_elevation(lat, lon) / 1000
            Vs30_mpers = obj.GetVs30(lon, lat)
            Z25_km = 999  # unknown
            ID += 1
            period1 = 0
            with open(SiteFile, 'a') as f:
                f.write(f"{ID} {lon} {lat} {elevation_km} {period1} {Vs30_mpers} {Z25_km}\n")
            IDvec.append(ID)
            lonVec.append(lon)
            latVec.append(lat)


    return IDvec, lonVec, latVec

def ReadIDofMeshedSiteFile(SiteFile):
    """
    Read the meshed site file

    Parameters:
        SiteFile: str - site file name

    Returns:
        IDvec: list - ID of each site
        lonVec: list - corresponding longitude
        latVec: list - corresponding latitude
    """

    SiteFile = Path(SiteFile).resolve()
    
    if not SiteFile.exists():
        raise FileNotFoundError(f"Site file {SiteFile} not found.")
    
    df = pd.read_csv(SiteFile, sep=' ', header=None, usecols=[0, 1, 2], dtype={0: int, 1: float, 2: float})
    IDvec = df[0].tolist()
    lonVec = df[1].tolist()
    latVec = df[2].tolist()

    return IDvec, lonVec, latVec

def CalcRuptureNormal(strike, dip):
    """
    Calculate the rupture normal vector

    Parameters:
        strike: float, strike angle in degrees (0-360). The strike direction correspond to the angle between the north and the direction you take so that when you walk along the Fault Trace the fault dips on your right.
        dip: float, dip angle in degrees. The dip is the steepest angle of descent of the fault plane relative to a horizontal plane; it is measured in degrees [0,90]

    Returns:
        RuptureNormal_x: float - x component of the rupture normal vector. East
        RuptureNormal_y: float - y component of the rupture normal vector. North
        RuptureNormal_z: float - z component of the rupture normal vector. Up
    """

    strike = math.radians(strike)
    dip = math.radians(dip)

    RuptureNormal_x = math.cos(strike) * math.sin(dip)
    RuptureNormal_y = math.sin(strike) * math.sin(dip)
    RuptureNormal_z = math.cos(dip)

    return RuptureNormal_x, RuptureNormal_y, RuptureNormal_z

def RuptureNormal2StrikeDip(RuptureNormal_x, RuptureNormal_y, RuptureNormal_z):
    """
    Convert the rupture normal vector to strike and dip angles

    Parameters:
        RuptureNormal_x: float - x component of the rupture normal vector. East
        RuptureNormal_y: float - y component of the rupture normal vector. North
        RuptureNormal_z: float - z component of the rupture normal vector. Up

    Returns:
        strike: float - strike angle in degrees (0-360). The strike direction correspond to the angle between the north and the direction you take so that when you walk along the Fault Trace the fault dips on your right.
        dip: float - dip angle in degrees. The dip is the steepest angle of descent of the fault plane relative to a horizontal plane; it is measured in degrees [0,90]
    """

    one = math.sqrt(RuptureNormal_x**2 + RuptureNormal_y**2 + RuptureNormal_z**2)
    strike = math.degrees(math.atan2(RuptureNormal_y/one, RuptureNormal_x/one))
    dip = math.degrees(math.acos(RuptureNormal_z/one))

    return strike, dip

def load_h5py_group_to_memory(group):
    data = {}
    for name, item in group.items():
        if isinstance(item, h5py.Group):
            data[name] = load_h5py_group_to_memory(item)
        elif isinstance(item, h5py.Dataset):
            if item.shape == ():  
                data[name] = item[()] 
            else:
                data[name] = item[:] 
    return data

def minify_html(html_path):
    """
    Compress HTML file content, remove unnecessary whitespace and comments
    
    Parameters:
        html_path: HTML file path
    """
    try:
        with open(html_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Remove comments (<!-- ... -->)
        content = re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL)
        # Remove excess whitespace
        content = re.sub(r'\s+', ' ', content)
        # Remove whitespace between tags
        content = re.sub(r'>\s+<', '><', content)
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return True
    except Exception as e:
        print(f"Error compressing HTML: {e}")
        return False
