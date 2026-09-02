"""
城市建筑基础数据准备模块。

使用流程：按城市或经纬度范围读取建筑轮廓，计算层数、面积与位置，
补充 POI、结构类型和使用类型，并保存区县级 Parquet 数据。

输入：城市名称或空间范围、最小建筑面积及数据读写函数。
输出：标准化的区县建筑 GeoDataFrame 文件及处理统计。
"""

import geopandas as gpd
from pypinyin import lazy_pinyin
import addressparser
from tqdm import tqdm
import re
import pandas as pd
import numpy as np
import time
from shapely.geometry import box

from .Utilities import *
from .Logger import get_logger
from MDOFModel.MDOF_CN import MDOF_CN

# 获取模块日志记录器
logger = get_logger('PrepareBldData')

def prepare_building_data(CityName=None, min_lon=None, max_lon=None, min_lat=None, max_lat=None, MinBldArea=None, output_data_dir=BLD_DATA_PATH, read_func=ReadDistrictBldGDF, save_func=SaveDistrictBldGDF, storage_type='HDF5'):
    '''
    Prepare building data, including structural type, occupancy type, design level.
    
    This function supports two modes of operation:
    1. City mode: Provide CityName to process all districts in a specific city
    2. Coordinate mode: Provide min_lon, max_lon, min_lat, max_lat to process districts within specified coordinates (automatically rounded to 0.01 precision)
    
    Parameters:
        CityName (str, optional): Name of the city. If provided, will use city mode.
        min_lon (float, optional): Minimum longitude of the region. Required for coordinate mode.
        max_lon (float, optional): Maximum longitude of the region. Required for coordinate mode.
        min_lat (float, optional): Minimum latitude of the region. Required for coordinate mode.
        max_lat (float, optional): Maximum latitude of the region. Required for coordinate mode.
        MinBldArea (float, optional): Minimum area of buildings to be considered. Unit: square meter.
    Returns:
        dict: A dictionary with processing results containing:
            - 'total_districts': Total number of districts processed
            - 'successful_districts': Number of districts with buildings found and processed
            - 'empty_districts': List of district names where no buildings were found

    Resulting Data:
        A folder named output_data_dir.
    '''
    return _prepare_building_data_common(
        CityName=CityName,
        min_lon=min_lon,
        max_lon=max_lon,
        min_lat=min_lat,
        max_lat=max_lat,
        MinBldArea=MinBldArea,
        data_dir=output_data_dir,
        read_func=read_func,
        save_func=save_func,
        storage_label=storage_type
    )


def _prepare_building_data_common(*, CityName=None, min_lon=None, max_lon=None, min_lat=None, max_lat=None, MinBldArea=None, data_dir, read_func, save_func, storage_label: str):
    '''Internal helper shared by CSV and HDF5 workflows.'''
    if CityName is not None:
        logger.info(f'Starting to prepare building data for city {CityName} (storage={storage_label})...')
        start_time = time.time()

        df = addressparser.transform([CityName])
        ProfName = df['省'][0].replace('省', '').replace('市', '')
        ProfName = ''.join(lazy_pinyin(ProfName))
        logger.debug(f'Corresponding province: {ProfName}')

        districts_gdf = get_DistrictsOfCity_polygons(CityName)
        logger.info(f'Found {len(districts_gdf)} districts')

        CityDir = data_dir / CityName
        CityDir.mkdir(parents=True, exist_ok=True)
        logger.debug(f'Created city data directory ({storage_label}): {CityDir}')

        # Read the building data district by district
        prof_file = list(BLD_FOOTPRINT_PATH.glob(f'**/{ProfName}.shp'))[0]
        logger.debug(f'Found province building footprint file: {prof_file}')
        
        # Use city mode
        is_coordinate_mode = False

    elif all(param is not None for param in [min_lon, max_lon, min_lat, max_lat]):
        # Coordinate mode - round to 0.01 precision
        region_name, (min_lon, max_lon, min_lat, max_lat) = generate_region_name(min_lon, max_lon, min_lat, max_lat)

        logger.info(f'Starting to prepare building data for coordinate region {region_name} (storage={storage_label})...')
        start_time = time.time()

        districts_gdf = get_DistrictsOfRegion_polygons(min_lat, max_lat, min_lon, max_lon)
        logger.info(f'Found {len(districts_gdf)} districts in the specified coordinate region')

        # Get all unique provinces in the region to determine which shapefile to use
        provinces_in_region = districts_gdf['pr_name'].unique()
        logger.debug(f'Provinces in region: {list(provinces_in_region)}')
        
        # Use coordinate mode
        is_coordinate_mode = True
        prof_file = None

        RegionDir = data_dir / region_name
        RegionDir.mkdir(parents=True, exist_ok=True)

    else:
        raise ValueError("Invalid parameters. Either provide 'CityName' for city mode, or provide 'min_lon', 'max_lon', 'min_lat', 'max_lat' for coordinate mode.")

    # Initialize statistics tracking
    total_districts = len(districts_gdf)
    successful_districts = 0
    empty_districts = []

    # Process each district
    for i in range(len(districts_gdf)):
        # district name and info
        district_row = districts_gdf.iloc[i]
        districtname = district_row["dt_name"]

        if is_coordinate_mode:
            # Coordinate mode: Use coordinate-based region folder
            city_name = district_row["ct_name"]
            province_name = district_row["pr_name"]
            
            # Create coordinate region directory if not exists
            RegionDir = data_dir / region_name
            RegionDir.mkdir(parents=True, exist_ok=True)
            
            # Use combined city-district name as file identifier
            file_identifier = f"{city_name}_{districtname}"

            logger.info(f'Processing district: {city_name}-{districtname} (coordinate mode)')
            district_start_time = time.time()

            district_gdf = read_func(region_name, file_identifier)
        else:
            # City mode: use original logic
            city_name = CityName
            file_identifier = districtname

            logger.info(f'Processing district: {CityName}-{districtname}')
            district_start_time = time.time()

            district_gdf = read_func(CityName, file_identifier)

        # read district building data from shapefile
        if district_gdf is None:
            logger.info(f'District {districtname} data does not exist, reading from province shapefile')

            if is_coordinate_mode:
                # Coordinate mode: get province shapefile dynamically
                prov_name_clean = province_name.replace('省', '').replace('市', '').replace('自治区', '').replace('特别行政区', '')
                prov_name_pinyin = ''.join(lazy_pinyin(prov_name_clean))
                logger.debug(f'Province name for shapefile: {prov_name_pinyin}')
                
                # Find the corresponding province shapefile
                prof_files = list(BLD_FOOTPRINT_PATH.glob(f'**/{prov_name_pinyin}.shp'))
                if not prof_files:
                    logger.warning(f'Province shapefile not found for {prov_name_pinyin}, skipping district {districtname}')
                    continue

                current_prof_file = prof_files[0]
                logger.debug(f'Found province building footprint file: {current_prof_file}')
            else:
                # City mode: use pre-determined shapefile
                current_prof_file = prof_file

            district_polygon = district_row.geometry
            district_gdf = gpd.read_file(current_prof_file, mask=district_polygon)
            district_gdf.columns = district_gdf.columns.str.lower()

            # Check if any buildings are found
            if len(district_gdf) == 0:
                logger.warning(f'No buildings found in district {districtname} from province building footprint. Skipping')
                empty_districts.append(districtname)
                continue
            
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
                
                # Check if any buildings remain after filtering
                if len(district_gdf) == 0:
                    logger.warning(f'No buildings remain in district {districtname} after area filtering (min area: {MinBldArea} m²). Skipping')
                    empty_districts.append(districtname)
                    continue
        else:
            logger.info(f'District {districtname} data already exists')
            if MinBldArea is not None:
                logger.debug(f'Filtering buildings with area greater than {MinBldArea} square meters')
                old_count = len(district_gdf)
                district_gdf = district_gdf[district_gdf['area'].astype(float) >= MinBldArea]
                filtered_count = len(district_gdf)
                logger.debug(f'Filtered buildings count: {filtered_count}/{old_count}')

                # 仅在确实删除建筑时写回，避免无变化数据覆盖原 Parquet。
                if filtered_count == 0:
                    logger.warning(f'No buildings remain in district {districtname} after area filtering (min area: {MinBldArea} m²). Skipping')
                    empty_districts.append(districtname)
                    continue

                if filtered_count < old_count:
                    if is_coordinate_mode:
                        save_func(region_name, file_identifier, district_gdf)
                    else:
                        save_func(city_name, file_identifier, district_gdf)
                else:
                    logger.debug(f'No buildings removed in district {districtname}; skipping Parquet rewrite')

        POI_available = False
        if 'POI' not in district_gdf.columns:
            logger.info(f'Getting POI data for district {districtname}')
            district_gdf_new = SearchBldPOI_Local(city_name, district_gdf)
            if district_gdf_new is not None:
                district_gdf = district_gdf_new
                POI_available = True
            if is_coordinate_mode:
                save_func(region_name, file_identifier, district_gdf)
            else:
                save_func(city_name, file_identifier, district_gdf)

        # predict building structure type
        if 'structure' not in district_gdf.columns:
            logger.info(f'Predicting building structure types for district {districtname}')
            UsePOI = POI_available
            district_gdf = PredictBldStructuralType(district_gdf, UsePOI=UsePOI)
            if is_coordinate_mode:
                save_func(region_name, file_identifier, district_gdf)
            else:
                save_func(city_name, file_identifier, district_gdf)

        # get nearest POI
        if POI_available and ('POI_Nearest' not in district_gdf.columns):
            logger.info(f'Getting nearest POI for buildings in district {districtname}')
            district_gdf = SearchBldNearestPOI_Local(city_name, district_gdf)
            if is_coordinate_mode:
                save_func(region_name, file_identifier, district_gdf)
            else:
                save_func(city_name, file_identifier, district_gdf)

        # derive building occupancy type from POI data
        if 'occupancy' not in district_gdf.columns:
            logger.info(f'Deriving building occupancy types for district {districtname}')
            district_gdf = DeriveBldOccupancyType(district_gdf)
            if is_coordinate_mode:
                save_func(region_name, file_identifier, district_gdf)
            else:
                save_func(city_name, file_identifier, district_gdf)

        district_end_time = time.time()
        logger.info(f'District {districtname} processing completed, time: {district_end_time-district_start_time:.2f} seconds')
        successful_districts += 1

    end_time = time.time()
    
    # Prepare result statistics
    result = {
        'total_districts': total_districts,
        'successful_districts': successful_districts,
        'empty_districts': empty_districts
    }

    if is_coordinate_mode:
        logger.info(f'Building data preparation for coordinate region {region_name} completed, total time: {end_time-start_time:.2f} seconds')
        logger.info(f'Processing summary: {successful_districts}/{total_districts} districts processed successfully')
        if empty_districts:
            logger.warning(f'Districts with no buildings: {", ".join(empty_districts)}')
    else:
        logger.info(f'Building data preparation for city {CityName} completed, total time: {end_time-start_time:.2f} seconds')
        logger.info(f'Processing summary: {successful_districts}/{total_districts} districts processed successfully')
        if empty_districts:
            logger.warning(f'Districts with no buildings: {", ".join(empty_districts)}')

    return result

def _convert_CSV_to_Parquet(CityName: str):
    '''
    Convert building data from CSV to Parquet format for a specific city.
    Parameters:
        CityName (str): Name of the city.
    '''    
    logger.info(f'Starting conversion of building data for city {CityName} from CSV to Parquet...')

    CityDir_Parquet = BLD_DATA_PATH / CityName
    districtnames = CityDir_Parquet.glob('*.csv')

    # Process each district
    for district_csv in districtnames:
        districtname = district_csv.stem

        logger.info(f'Converting district: {CityName}-{districtname}')

        # read district building data from CSV
        district_gdf = ReadDistrictBldGDF_CSV(CityName, districtname)
        if district_gdf is None:
            logger.warning(f'District {districtname} data does not exist in CSV format. Skipping')
            continue

        # save as Parquet
        SaveDistrictBldGDF(CityName, districtname, district_gdf)

        logger.info(f'District {districtname} conversion completed')

    logger.info(f'Building data conversion for city {CityName} completed')

# 现行行政区名称与 GB 50011-2010（2016 年版）附录 A 中名称的对应关系。
DESIGN_CODE_LOCATION_ALIASES = {
    ('成都', '双流区'): ('成都', '双流县'),
    ('成都', '郫都区'): ('成都', '郫县'),
    ('成都', '新津区'): ('成都', '新津县'),
    ('成都', '简阳市'): ('资阳', '简阳市'),
}


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
    LookupCityName, LookupDistrictName = DESIGN_CODE_LOCATION_ALIASES.get((CityName, DistrictName), (CityName, DistrictName))
    Rows = GBApp_A[GBApp_A['City'].str.contains(LookupCityName, na=False, regex=False)]
    if Rows.empty:
        raise Exception(f'City {LookupCityName} not found in {GB50011_A_FILE}')
    else:
        Row = Rows[Rows['District'].str.contains(LookupDistrictName, na=False, regex=False)]
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

    # 根据规范表取得当前行政区应使用的设防参数。
    SeismicDesignLevel, EQGroup = GetDesignLevelAndEQGroupbyLoc(CityName, DistrictName)

    # 已有参数与规范表一致时直接复用；行政区名称调整后则更新已有数据。
    required_columns_exist = 'SeismicDesignLevel' in Bld_gdf.columns and 'EQGroup' in Bld_gdf.columns and 'SiteClass' in Bld_gdf.columns
    design_level_matches = required_columns_exist and pd.to_numeric(Bld_gdf['SeismicDesignLevel'], errors='coerce').eq(float(SeismicDesignLevel)).all()
    eq_group_matches = required_columns_exist and Bld_gdf['EQGroup'].astype(str).eq(str(EQGroup)).all()
    if not overwrite and required_columns_exist and design_level_matches and eq_group_matches:
        logger.debug(f'Building data for {CityName}-{DistrictName} already contains seismic design level, earthquake group and site class. Skipping...')
        return

    # 更新抗震设防烈度和设计地震分组。
    logger.debug(f'Seismic design level: {SeismicDesignLevel}, Earthquake group: {EQGroup}')
    Bld_gdf['SeismicDesignLevel'] = SeismicDesignLevel
    Bld_gdf['EQGroup'] = EQGroup

    # 场地类别已经存在时无需因行政区名称调整而重新计算。
    if overwrite or 'SiteClass' not in Bld_gdf.columns:
        logger.debug(f'Calculating site class')
        SiteClass = GetSiteClassbyLoc(Bld_gdf['longitude'].to_list(), Bld_gdf['latitude'].to_list())
        Bld_gdf['SiteClass'] = SiteClass
    
    logger.debug(f'Saving building data for {CityName}-{DistrictName}')
    SaveDistrictBldGDF(CityName, DistrictName, Bld_gdf)

def _EstimateSeismicDesignLevel_Coordinate(RegionName, FileIdentifier, CityName, DistrictName, overwrite=False):
    '''
    Estimate the seismic design level of buildings for coordinate mode. 'SeismicDesignLevel', 'EQGroup', 'SiteClass' columns will be added to the building data.
    Parameters:
        RegionName (str): Name of the coordinate region folder.
        FileIdentifier (str): File identifier (city_district format).
        CityName (str): Name of the city.
        DistrictName (str): Name of the district.
        overwrite (bool): Whether to overwrite the existing data. Default is False.
    '''    
    logger.debug(f'Reading building data for {RegionName}/{FileIdentifier}')
    Bld_gdf: gpd.GeoDataFrame = ReadDistrictBldGDF(RegionName, FileIdentifier)

    # check if the 'SeismicDesignLevel', 'EQGroup', 'SiteClass' columns exist
    if not overwrite and 'SeismicDesignLevel' in Bld_gdf.columns and 'EQGroup' in Bld_gdf.columns and 'SiteClass' in Bld_gdf.columns:
        logger.debug(f'Building data for {RegionName}/{FileIdentifier} already contains seismic design level, earthquake group and site class. Skipping...')
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
    
    logger.debug(f'Saving building data for {RegionName}/{FileIdentifier}')
    SaveDistrictBldGDF(RegionName, FileIdentifier, Bld_gdf)

def EstimateSeismicDesignLevel(CityName=None, min_lon=None, max_lon=None, min_lat=None, max_lat=None):
    '''
    Estimate the seismic design level of buildings based on city location and building properties. 'SeismicDesignLevel', 'EQGroup', 'SiteClass' columns will be added to the building data.
    This function supports two modes:
    1. City mode: Provide CityName to process all districts in a specific city
    2. Coordinate mode: Provide min_lon, max_lon, min_lat, max_lat to process districts within specified coordinates (automatically rounded to 0.01 precision)
    
    Parameters:
        CityName (str, optional): Name of the city. If provided, will use city mode.
        min_lon (float, optional): Minimum longitude of the region. Required for coordinate mode.
        max_lon (float, optional): Maximum longitude of the region. Required for coordinate mode.
        min_lat (float, optional): Minimum latitude of the region. Required for coordinate mode.
        max_lat (float, optional): Maximum latitude of the region. Required for coordinate mode.
    '''    
    if CityName is not None:
        # City mode - original logic
        logger.info(f'Starting to estimate seismic design level, earthquake group and site class for city {CityName}...')
        start_time = time.time()

        # get all the districts
        districts_gdf = get_DistrictsOfCity_polygons(CityName)
        logger.info(f'Found {len(districts_gdf)} districts')
        
        for i in tqdm(range(len(districts_gdf))):
            # district name
            districtname = districts_gdf.iloc[i]["dt_name"]
            
            _EstimateSeismicDesignLevel(CityName, districtname)
        
        end_time = time.time()
        logger.info(f'Seismic design level estimation for city {CityName} completed, total time: {end_time-start_time:.2f} seconds')
        
    elif all(param is not None for param in [min_lon, max_lon, min_lat, max_lat]):
        # Coordinate mode - round to 0.01 precision
        region_name, (min_lon, max_lon, min_lat, max_lat) = generate_region_name(min_lon, max_lon, min_lat, max_lat)
        
        logger.info(f'Starting to estimate seismic design level, earthquake group and site class for coordinate region {region_name}...')
        start_time = time.time()

        districts_gdf = get_DistrictsOfRegion_polygons(min_lat, max_lat, min_lon, max_lon)

        logger.info(f'Found {len(districts_gdf)} districts in the coordinate region')
        
        for i in tqdm(range(len(districts_gdf))):
            district_row = districts_gdf.iloc[i]
            city_name = district_row["ct_name"]
            district_name = district_row["dt_name"]
            
            # Use combined city-district name as file identifier
            file_identifier = f"{city_name}_{district_name}"
            
            # Check if the region directory and district file exist
            region_dir = BLD_DATA_PATH / region_name
            if not region_dir.exists() or not (region_dir / f'{file_identifier}.csv').exists():
                logger.warning(f'District data not found for {region_name}/{file_identifier}, skipping')
                continue
            
            # Use a modified version of _EstimateSeismicDesignLevel for coordinate mode
            _EstimateSeismicDesignLevel_Coordinate(region_name, file_identifier, city_name, district_name)
        
        end_time = time.time()
        logger.info(f'Seismic design level estimation for coordinate region {region_name} completed, total time: {end_time-start_time:.2f} seconds')
    else:
        raise ValueError("Either provide 'CityName' for city mode, or provide 'min_lon', 'max_lon', 'min_lat', 'max_lat' for coordinate mode")

