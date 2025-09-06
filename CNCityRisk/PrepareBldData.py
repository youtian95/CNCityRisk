import geopandas as gpd
from pypinyin import lazy_pinyin
import addressparser
from tqdm import tqdm
import re
import pandas as pd
import numpy as np
import time

from .Utilities import *
from .Logger import get_logger
from MDOFModel.MDOF_CN import MDOF_CN

# 获取模块日志记录器
logger = get_logger('PrepareBldData')


def prepare_building_data(CityName, MinBldArea = None):
    '''
    Prepare building data, including structural type, occupancy type, design level.
    Parameters:
        CityName (str): Name of the city.
        MinBldArea (float): Minimum area of buildings to be considered. Unit: square meter.
    Returns:
        None. A folder containing csv files. Each csv file contains the building data of the city. Columns include: longitude, latitude, story, area, structure, occupancy, designlevel.
    '''
    logger.info(f'Starting to prepare building data for {CityName}...')
    start_time = time.time()
    
    # Get the province name
    df = addressparser.transform([CityName])
    ProfName = df['省'][0].replace('省', '').replace('市', '')
    ProfName = ''.join(lazy_pinyin(ProfName))
    logger.debug(f'Corresponding province: {ProfName}')

    # get all the districts
    districts_gdf = get_DistrictsOfCity_polygons(CityName)
    logger.info(f'Found {len(districts_gdf)} districts')   
    # create a new folder for the city if not exist
    CityDir = BLD_DATA_PATH / CityName
    CityDir.mkdir(parents=True, exist_ok=True)
    logger.debug(f'Created city data directory: {CityDir}')

    # Read the building data district by district
    prof_file= list(BLD_FOOTPRINT_PATH.glob(f'**/{ProfName}.shp'))[0]
    logger.debug(f'Found province building footprint file: {prof_file}')

    for i in range(len(districts_gdf)):
        # district name
        districtname = districts_gdf.iloc[i]["dt_name"]

        logger.info(f'Processing district: {CityName}-{districtname}')
        district_start_time = time.time()

        # read district building data if exists
        district_gdf = ReadDistrictBldGDF(CityName, districtname)

        # read district building data from shapefile
        if district_gdf is None:
            logger.info(f'District {districtname} data does not exist, reading from province shapefile')
            district_polygon = districts_gdf.iloc[i].geometry
            district_gdf = gpd.read_file(prof_file, mask=district_polygon)
            district_gdf.columns = district_gdf.columns.str.lower()
            
            # 处理建筑数据
            logger.debug('Converting building height to stories')
            district_gdf['story'] = district_gdf['height'].apply(height2story)
            
            logger.debug('Converting coordinate system to UTM')
            district_gdf.set_crs(epsg=4326, inplace=True)
            utm_crs = district_gdf.estimate_utm_crs()
            district_gdf.to_crs(utm_crs, inplace=True)
            
            logger.debug('Calculating building centroid coordinates')
            district_gdf['centroid'] = district_gdf['geometry'].centroid
            district_gdf['centroid'] = district_gdf['centroid'].to_crs(epsg=4326)
            district_gdf['longitude'] = district_gdf['centroid'].x
            district_gdf['latitude'] = district_gdf['centroid'].y
            district_gdf.drop(columns=['centroid'], inplace=True)
            
            logger.debug('Calculating building area')
            district_gdf['area'] = district_gdf['geometry'].area
            
            logger.debug('Converting coordinate system back to WGS84')
            district_gdf.to_crs(epsg=4326, inplace=True)
            
            # 筛选建筑
            if MinBldArea is not None:
                logger.debug(f'Filtering buildings with area greater than {MinBldArea} square meters')
                old_count = len(district_gdf)
                district_gdf = district_gdf[district_gdf['area'].astype(float) >= MinBldArea]
                logger.debug(f'Filtered buildings count: {len(district_gdf)}/{old_count}')
        else:
            logger.info(f'District {districtname} data already exists')
            if MinBldArea is not None:
                logger.debug(f'Filtering buildings with area greater than {MinBldArea} square meters')
                old_count = len(district_gdf)
                district_gdf = district_gdf[district_gdf['area'].astype(float) >= MinBldArea]
                logger.debug(f'Filtered buildings count: {len(district_gdf)}/{old_count}')
                SaveDistrictBldGDF(CityName, districtname, district_gdf)        # get POI
        if 'POI' not in district_gdf.columns:
            logger.info(f'Getting POI data for district {districtname}')
            district_gdf = SearchBldPOI_Local(CityName, district_gdf)
            SaveDistrictBldGDF(CityName, districtname, district_gdf)

        # predict building structure type
        if 'structure' not in district_gdf.columns:
            logger.info(f'Predicting building structure types for district {districtname}')
            district_gdf = PredictBldStructuralType(district_gdf, UsePOI=True)
            SaveDistrictBldGDF(CityName, districtname, district_gdf)

        # get nearest POI
        if 'POI_Nearest' not in district_gdf.columns:
            logger.info(f'Getting nearest POI for buildings in district {districtname}')
            district_gdf = SearchBldNearestPOI_Local(CityName, district_gdf)
            SaveDistrictBldGDF(CityName, districtname, district_gdf)

        # derive building occupancy type from POI data
        if 'occupancy' not in district_gdf.columns:
            logger.info(f'Deriving building occupancy types for district {districtname}')
            district_gdf = DeriveBldOccupancyType(CityName, district_gdf)
            SaveDistrictBldGDF(CityName, districtname, district_gdf)
            
        district_end_time = time.time()
        logger.info(f'District {districtname} processing completed, time: {district_end_time-district_start_time:.2f} seconds')

    end_time = time.time()
    logger.info(f'Building data preparation for {CityName} completed, total time: {end_time-start_time:.2f} seconds')

def GetDesignLevelAndEQGroupbyLoc(CityName: str, DistrictName: str):
    '''
    Get the seismic design level of buildings based on city location.
    Parameters:
        CityName (str): Name of the city.
        DistrictName (str): Name of the district.
    Returns:
        SeismicDesignLevel (str): Seismic design level of the city.
        EQGroup (str): Earthquake group of the city.
    '''
    GBApp_A = pd.read_csv(GB50011_A_FILE, na_values='-')
    GBApp_A['City'] = GBApp_A['City'].ffill()    
    Rows = GBApp_A[GBApp_A['City'].str.contains(CityName,na=False)]
    if Rows.empty:
        raise Exception(f'City {CityName} not found in {GB50011_A_FILE}')
    else:
        Row = Rows[Rows['District'].str.contains(DistrictName,na=False)]
        if Row.empty:
            logger.warning(f'District {DistrictName} not found in {CityName} of {GB50011_A_FILE}, using the first district as default')
            Row = Rows.iloc[0:1]  # 使用该城市的第一个区域作为默认值
        
        SDL = Row['Design Level'].values[-1]
        SDL = re.findall(r"\d+\.?\d*", SDL)[0]
        PGA = Row['PGA'].values[-1]
        PGA = re.findall(r"\d+\.?\d*", PGA)[0]
        PGA = float(PGA)
        alphaMax = PGA*2.4
        if SDL == '8' and alphaMax == 0.3:
            SDL = '8.5'
        elif SDL == '7' and alphaMax == 0.15:
            SDL = '7.5'
        SeismicDesignLevel = SDL
        EQGroup = Row['EQgroup'].values[-1]
        if EQGroup[1] == '一':
            EQGroup = '1'
        elif EQGroup[1] == '二':
            EQGroup = '2'
        elif EQGroup[1] == '三':
            EQGroup = '3'

        return SeismicDesignLevel, EQGroup

def GetSiteClassbyLoc(Longitudes: list[float], Latitudes: list[float]):
    '''
    Set site class according to location per GB 50011-2010(2016) Table 4.1.6
    [1] GB 50011-2010(2016) Table 4.1.6
    [2] Zhou J, Li X, Tian X, Xu G. New Framework of Combining Observations with Topographic Slope to Estimate VS30 and Its Application on Building a VS30 Map for Mainland China. Bulletin of the Seismological Society of America, 2022, 112(4): 2049-2069.
    Parameters:
        Longitudes (list[float]): Longitude of the location.
        Latitudes (list[float]): Latitude of the location.
    Returns:   
        SiteClass (list[str]): Site class of the location.
    '''
    VS30Table = pd.read_excel(VS30FILE, header=1)
    SiteClass_list = []
    for Longitude, Latitude in zip(Longitudes, Latitudes):
        distances = np.sqrt((VS30Table['Longitude (°)'] - Longitude)**2 \
            + (VS30Table['Latitude (°)'] - Latitude)**2)
        closest_index = distances.idxmin()
        VS30 = VS30Table.iloc[closest_index-1, 4]
        if VS30 > 800:
            SiteClass = '1_0'
        elif VS30 > 500:
            SiteClass = '1_1'
        elif VS30 > 250:
            SiteClass = '2'
        elif VS30 > 150:
            SiteClass = '3'
        else:
            SiteClass = '4'
        SiteClass_list.append(SiteClass)
    return SiteClass

def _EstimateSeismicDesignLevel(CityName, DistrictName, overwrite=False):
    '''
    Estimate the seismic design level of buildings based on city location and building properties. 'SeismicDesignLevel', 'EQGroup', 'SiteClass' columns will be added to the building data.
    Parameters:
        CityName (str): Name of the city.
        DistrictName (str): Name of the district.
        overwrite (bool): Whether to overwrite the existing data. Default is False.
    '''    
    logger.debug(f'Reading building data for {CityName}-{DistrictName}')
    Bld_gdf: gpd.GeoDataFrame = ReadDistrictBldGDF(CityName, DistrictName)

    # check if the 'SeismicDesignLevel', 'EQGroup', 'SiteClass' columns exist
    if not overwrite and 'SeismicDesignLevel' in Bld_gdf.columns and 'EQGroup' in Bld_gdf.columns and 'SiteClass' in Bld_gdf.columns:
        logger.debug(f'Building data for {CityName}-{DistrictName} already contains seismic design level, earthquake group and site class. Skipping...')
        return

    # get seismic design level and EQ group
    logger.debug(f'Getting seismic design level and earthquake group')
    SeismicDesignLevel, EQGroup = GetDesignLevelAndEQGroupbyLoc(CityName, DistrictName)
    logger.debug(f'Seismic design level: {SeismicDesignLevel}, Earthquake group: {EQGroup}')
    Bld_gdf['SeismicDesignLevel'] = SeismicDesignLevel
    Bld_gdf['EQGroup'] = EQGroup

    # get site class
    logger.debug(f'Calculating site class')
    SiteClass = GetSiteClassbyLoc(Bld_gdf['longitude'].to_list(), Bld_gdf['latitude'].to_list())
    Bld_gdf['SiteClass'] = SiteClass
    
    logger.debug(f'Saving building data for {CityName}-{DistrictName}')
    SaveDistrictBldGDF(CityName, DistrictName, Bld_gdf)

def EstimateSeismicDesignLevel(CityName):
    '''
    Estimate the seismic design level of buildings based on city location and building properties. 'SeismicDesignLevel', 'EQGroup', 'SiteClass' columns will be added to the building data.
    Parameters:
        CityName (str): Name of the city.
    '''    
    logger.info(f'Starting to estimate seismic design level, earthquake group and site class for {CityName}...')
    start_time = time.time()

    # get all the districts
    districts_gdf = get_DistrictsOfCity_polygons(CityName)
    logger.info(f'Found {len(districts_gdf)} districts')
    
    for i in tqdm(range(len(districts_gdf))):
        # district name
        districtname = districts_gdf.iloc[i]["dt_name"]
        logger.debug(f'Processing district: {districtname}')
        district_start = time.time()
        
        _EstimateSeismicDesignLevel(CityName, districtname)
        
        district_end = time.time()
        logger.debug(f'District {districtname} processing completed, time: {district_end-district_start:.2f} seconds')
    
    end_time = time.time()
    logger.info(f'Seismic design level estimation for {CityName} completed, total time: {end_time-start_time:.2f} seconds')