import geopandas as gpd
from xpinyin import Pinyin
import addressparser
from tqdm import tqdm
import re
import pandas as pd
import numpy as np

from .Utilities import *
from MDOFModel.MDOF_CN import MDOF_CN


def prepare_building_data(CityName, MinBldArea = None):
    '''
    Prepare building data, including structural type, occupancy type, design level.
    Parameters:
        CityName (str): Name of the city.
        MinBldArea (float): Minimum area of buildings to be considered. Unit: square meter.
    Returns:
        None. A folder containing csv files. Each csv file contains the building data of the city. Columns include: longitude, latitude, story, area, structure, occupancy, designlevel.
    '''
    
    # Get the province name
    df = addressparser.transform([CityName])
    ProfName = df['省'][0].replace('省', '')
    ProfName = Pinyin().get_pinyin(ProfName, '')

    # get all the districts
    districts_gdf = get_DistrictsOfCity_polygons(CityName)

    # create a new folder for the city if not exist
    CityDir = BLD_DATA_PATH / CityName
    CityDir.mkdir(parents=True, exist_ok=True)

    # Read the building data district by district
    prof_file= list(BLD_FOOTPRINT_PATH.glob(f'**/{ProfName}.shp'))[0]
    for i in range(len(districts_gdf)):
        # district name
        districtname = districts_gdf.iloc[i]["dt_name"]

        print(f'{CityName}-{districtname}: Preparing building data...')

        # read district building data if exists
        district_gdf = ReadDistrictBldGDF(CityName, districtname)

        # read district building data from shapefile
        if district_gdf is None:
            district_polygon = districts_gdf.iloc[i].geometry
            district_gdf = gpd.read_file(prof_file, mask=district_polygon)
            district_gdf.columns = district_gdf.columns.str.lower()

            # building height to story
            district_gdf['story'] = district_gdf['height'].apply(height2story)
            # change the crs to UTM
            district_gdf.set_crs(epsg=4326, inplace=True)
            utm_crs = district_gdf.estimate_utm_crs()
            district_gdf.to_crs(utm_crs, inplace=True)
            # building coordinates 
            district_gdf['centroid'] = district_gdf['geometry'].centroid
            district_gdf['centroid'] = district_gdf['centroid'].to_crs(epsg=4326)
            district_gdf['longitude'] = district_gdf['centroid'].x
            district_gdf['latitude'] = district_gdf['centroid'].y
            district_gdf.drop(columns=['centroid'], inplace=True)
            # building area
            district_gdf['area'] = district_gdf['geometry'].area
            # change the crs back to WGS84
            district_gdf.to_crs(epsg=4326, inplace=True)
            # filter buildings by area
            if MinBldArea is not None:
                district_gdf = district_gdf[district_gdf['area'].astype(float) >= MinBldArea]
        else:
            if MinBldArea is not None:
                district_gdf = district_gdf[district_gdf['area'].astype(float) >= MinBldArea]
                SaveDistrictBldGDF(CityName, districtname, district_gdf)


        # get POI
        if 'POI' not in district_gdf.columns:
            district_gdf = SearchBldPOI_Local(CityName, district_gdf)
            SaveDistrictBldGDF(CityName, districtname, district_gdf)

        # predict building structure type
        if 'structure' not in district_gdf.columns:
            district_gdf = PredictBldStructuralType(district_gdf, UsePOI=True)
            SaveDistrictBldGDF(CityName, districtname, district_gdf)

        # get nearest POI
        if 'POI_Nearest' not in district_gdf.columns:
            district_gdf = SearchBldNearestPOI_Local(CityName, district_gdf)
            SaveDistrictBldGDF(CityName, districtname, district_gdf)

        # derive building occupancy type from POI data
        if 'occupancy' not in district_gdf.columns:
            district_gdf = DeriveBldOccupancyType(CityName, district_gdf)
            SaveDistrictBldGDF(CityName, districtname, district_gdf)

    print(f'{CityName}: Done')

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
            raise Exception(f'District {DistrictName} not found in {CityName} of {GB50011_A_FILE}')
        else:
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

def _EstimateSeismicDesignLevel(CityName, DistrictName):
    '''
    Estimate the seismic design level of buildings based on city location and building properties. 'SeismicDesignLevel', 'EQGroup', 'SiteClass' columns will be added to the building data.
    Parameters:
        CityName (str): Name of the city.
        DistrictName (str): Name of the district.
    '''
    Bld_gdf: gpd.GeoDataFrame = ReadDistrictBldGDF(CityName, DistrictName)

    # get seismic design level and EQ group
    SeismicDesignLevel, EQGroup = GetDesignLevelAndEQGroupbyLoc(CityName, DistrictName)
    Bld_gdf['SeismicDesignLevel'] = SeismicDesignLevel
    Bld_gdf['EQGroup'] = EQGroup

    # get site class
    SiteClass = GetSiteClassbyLoc(Bld_gdf['longitude'].to_list(), Bld_gdf['latitude'].to_list())
    Bld_gdf['SiteClass'] = SiteClass

    SaveDistrictBldGDF(CityName, DistrictName, Bld_gdf)

def EstimateSeismicDesignLevel(CityName):
    '''
    Estimate the seismic design level of buildings based on city location and building properties. 'SeismicDesignLevel', 'EQGroup', 'SiteClass' columns will be added to the building data.
    Parameters:
        CityName (str): Name of the city.
    '''
    print(f'{CityName}: Estimating seismic design level, EQ Group, and Site Class...')

    # get all the districts
    districts_gdf = get_DistrictsOfCity_polygons(CityName)
    for i in tqdm(range(len(districts_gdf))):
        # district name
        districtname = districts_gdf.iloc[i]["dt_name"]
        _EstimateSeismicDesignLevel(CityName, districtname)