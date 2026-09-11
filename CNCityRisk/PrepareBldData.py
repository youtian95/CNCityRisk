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

        # 已保存 POI 的中断任务仍需继续补充最近 POI，不能只在本次新搜索后才标记为可用
        POI_available = 'POI' in district_gdf.columns
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
# 附录 A 沿用 2016 年前的区划名称，此后的县改区、县改市、跨市划转和新设区都查不到对应行，
# 原逻辑会静默回退到该市第一行，导致设计水平或地震分组取错。
# 键为（城市名，现行区县名），值为（附录中的市名，附录中的区县名）。
# 城市名写简称即可（如“天津”），查表时按前缀匹配，兼容 shapefile 的全称（如“天津市”“凉山彝族自治州”）。
DESIGN_CODE_LOCATION_ALIASES = {
    # ---- 一、县改区 / 县改市：附录沿用 2016 年前的旧县名（99 条）----

    ('上海', '崇明区'): ('上海', '崇明县'),  # 2016 年撤县设区
    ('东营', '垦利区'): ('东营', '垦利县'),
    ('伊春', '南岔县'): ('伊春', '南岔区'),
    ('佳木斯', '抚远市'): ('佳木斯', '抚远县'),
    ('六盘水', '水城区'): ('六盘水', '水城县'),
    ('内江', '隆昌市'): ('内江', '隆昌县'),
    ('凉山彝族自治州', '会理市'): ('凉山彝族自治州', '会理县'),
    ('南通', '海安市'): ('南通', '海安县'),
    ('南通', '海门区'): ('南通', '海门市'),
    ('台州', '玉环市'): ('台州', '玉环县'),
    ('周口', '淮阳区'): ('周口', '淮阳县'),
    ('唐山', '滦州市'): ('唐山', '滦县'),  # 2018 年撤县设市
    ('塔城地区', '沙湾市'): ('塔城地区', '沙湾县'),
    ('大兴安岭地区', '漠河市'): ('大兴安岭地区', '漠河县'),
    ('大同', '云州区'): ('大同', '大同县'),  # 2018 年撤县设区
    ('大连', '普兰店区'): ('大连', '普兰店市'),
    ('天津', '蓟州区'): ('天津', '蓟县'),  # 2016 年撤县设区
    ('宁波', '奉化区'): ('宁波', '奉化市'),
    ('安庆', '潜山市'): ('安庆', '潜山县'),
    ('安康', '旬阳市'): ('安康', '旬阳县'),
    ('宝鸡', '凤翔区'): ('宝鸡', '凤翔县'),
    ('宣城', '广德市'): ('宣城', '广德县'),
    ('山南', '乃东区'): ('山南', '乃东县'),
    ('平凉', '华亭市'): ('平凉', '华亭县'),
    ('延安', '子长市'): ('延安', '子长县'),
    ('延安', '安塞区'): ('延安', '安塞县'),
    ('张家口', '万全区'): ('张家口', '万全县'),
    ('张家口', '崇礼区'): ('张家口', '崇礼县'),
    ('德阳', '罗江区'): ('德阳', '罗江县'),
    ('怒江傈僳族自治州', '泸水市'): ('怒江傈僳族自治州', '泸水县'),
    ('成都', '双流区'): ('成都', '双流县'),
    ('成都', '新津区'): ('成都', '新津县'),
    ('成都', '郫都区'): ('成都', '郫县'),  # 2016 年撤县设区
    ('承德', '平泉市'): ('承德', '平泉县'),
    ('抚州', '东乡区'): ('抚州', '东乡县'),
    ('拉萨', '堆龙德庆区'): ('拉萨', '堆龙德庆县'),
    ('拉萨', '达孜区'): ('拉萨', '达孜县'),
    ('新乡', '长垣市'): ('新乡', '长垣县'),
    ('昆明', '晋宁区'): ('昆明', '晋宁县'),
    ('昭通', '水富市'): ('昭通', '水富县'),
    ('晋中', '太谷区'): ('晋中', '太谷县'),
    ('曲靖', '沾益区'): ('曲靖', '沾益县'),
    ('曲靖', '马龙区'): ('曲靖', '马龙县'),
    ('朔州', '怀仁市'): ('朔州', '怀仁县'),
    ('杭州', '临安区'): ('杭州', '临安市'),
    ('柳州', '柳江区'): ('柳州', '柳江县'),
    ('桂林', '荔浦市'): ('桂林', '荔浦县'),
    ('楚雄彝族自治州', '禄丰市'): ('楚雄彝族自治州', '禄丰县'),
    ('榆林', '横山区'): ('榆林', '横山县'),
    ('榆林', '神木市'): ('榆林', '神木县'),
    ('毕节', '黔西市'): ('毕节', '黔西县'),
    ('永州', '祁阳市'): ('永州', '祁阳县'),
    ('汉中', '南郑区'): ('汉中', '南郑县'),
    ('沈阳', '辽中区'): ('沈阳', '辽中县'),
    ('河池', '宜州区'): ('河池', '宜州市'),
    ('洛阳', '偃师区'): ('洛阳', '偃师市'),
    ('洛阳', '孟津区'): ('洛阳', '孟津县'),
    ('济南', '济阳区'): ('济南', '济阳县'),
    ('济南', '章丘区'): ('济南', '章丘市'),
    ('淮安', '洪泽区'): ('淮安', '洪泽县'),
    ('滨州', '邹平市'): ('滨州', '邹平县'),
    ('漳州', '长泰区'): ('漳州', '长泰县'),
    ('漳州', '龙海区'): ('漳州', '龙海市'),
    ('烟台', '蓬莱区'): ('烟台', '蓬莱市'),
    ('牡丹江', '东宁市'): ('牡丹江', '东宁县'),
    ('玉溪', '江川区'): ('玉溪', '江川县'),
    ('玉溪', '澄江市'): ('玉溪', '澄江县'),
    ('百色', '平果市'): ('百色', '平果县'),
    ('百色', '田阳区'): ('百色', '田阳县'),
    ('盘锦', '大洼区'): ('盘锦', '大洼县'),
    ('福州', '长乐区'): ('福州', '长乐市'),
    ('聊城', '茌平区'): ('聊城', '茌平县'),
    ('芜湖', '无为市'): ('芜湖', '无为县'),
    ('芜湖', '繁昌区'): ('芜湖', '繁昌县'),
    ('荆州', '监利市'): ('荆州', '监利县'),
    ('荆门', '京山市'): ('荆门', '京山县'),
    ('菏泽', '定陶区'): ('菏泽', '定陶县'),
    ('衡水', '冀州区'): ('衡水', '冀州市'),
    ('西宁', '湟中区'): ('西宁', '湟中县'),
    ('赣州', '龙南市'): ('赣州', '龙南县'),
    ('遂宁', '射洪市'): ('遂宁', '射洪县'),
    ('邢台', '任泽区'): ('邢台', '任县'),  # 2020 年撤县设区
    ('邢台', '南和区'): ('邢台', '南和县'),
    ('邯郸', '永年区'): ('邯郸', '永年县'),
    ('邯郸', '肥乡区'): ('邯郸', '肥乡县'),
    ('邵阳', '邵东市'): ('邵阳', '邵东县'),
    ('重庆', '开州区'): ('重庆', '开县'),  # 2016 年撤县设区
    ('重庆', '梁平区'): ('重庆', '梁平县'),  # 2016 年撤县设区
    ('重庆', '武隆区'): ('重庆', '武隆县'),  # 2016 年撤县设区
    ('长沙', '宁乡市'): ('长沙', '宁乡县'),
    ('长治', '上党区'): ('长治', '长治县'),  # 2018 年撤县设区
    ('长治', '屯留区'): ('长治', '屯留县'),
    ('长治', '潞城区'): ('长治', '潞城市'),
    ('阿克苏地区', '库车市'): ('阿克苏地区', '库车县'),
    ('阿坝藏族羌族自治州', '马尔康市'): ('阿坝藏族羌族自治州', '马尔康县'),
    ('青岛', '即墨区'): ('青岛', '即墨市'),
    ('鹰潭', '余江区'): ('鹰潭', '余江县'),
    ('黄南藏族自治州', '同仁市'): ('黄南藏族自治州', '同仁县'),
    ('黑河', '嫩江市'): ('黑河', '嫩江县'),
    ('黔西南布依族苗族自治州', '兴仁市'): ('黔西南布依族苗族自治州', '兴仁县'),

    # ---- 二、跨市整体划转：附录仍挂在原所属市下（8 条）----

    ('保山', '腾冲市'): ('省直辖县级行政单位', '腾冲市'),  # 2015 年撤县设市，附录挂在云南省直辖县级单位下
    ('成都', '简阳市'): ('资阳', '简阳市'),  # 2016 年由资阳划归成都
    ('济南', '莱芜区'): ('莱芜', '莱城区'),  # 2019 年莱芜市并入济南，莱城区改莱芜区
    ('济南', '钢城区'): ('莱芜', '钢城区'),  # 2019 年莱芜市并入济南
    ('淮南', '寿县'): ('六安', '寿县'),  # 2015 年由六安划归淮南
    ('百色', '靖西市'): ('自治区直辖县级行政单位', '靖西市'),  # 2015 年撤县设市，附录挂在广西直辖县级单位下
    ('铜陵', '枞阳县'): ('安庆', '枞阳县'),  # 2016 年由安庆划归铜陵
    ('长春', '公主岭市'): ('四平', '公主岭市'),  # 2020 年由四平代管改由长春代管

    # ---- 三、区划调整（合并、析出、新设区）：新旧名词根不同，按主要前身取值（47 条）----

    ('三明', '沙县区'): ('三明', '沙县'),  # 2021 年撤县设区
    ('三沙', '南沙区'): ('三沙', '三沙市'),  # 2020 年新设区，附录仅到三沙市
    ('三沙', '西沙区'): ('三沙', '三沙市'),  # 2020 年新设区，附录仅到三沙市
    ('上饶', '广信区'): ('上饶', '上饶县'),  # 2019 年撤县设区
    ('九江', '庐山市'): ('九江', '星子县'),  # 2016 年星子县改设庐山市，与旧庐山区（今濂溪区）无关
    ('九江', '柴桑区'): ('九江', '九江县'),  # 2017 年撤县设区
    ('九江', '濂溪区'): ('九江', '庐山区'),  # 2016 年庐山区改濂溪区
    ('伊春', '丰林县'): ('伊春', '新青区'),  # 2019 年新青区、五营区、红星区合并设县
    ('伊春', '乌翠区'): ('伊春', '翠峦区'),  # 2019 年乌马河区、翠峦区合并
    ('伊春', '伊美区'): ('伊春', '伊春区'),  # 2019 年伊春区、美溪区合并
    ('伊春', '大箐山县'): ('伊春', '带岭区'),  # 2019 年带岭区、朗乡镇合并设县
    ('伊春', '汤旺县'): ('伊春', '汤旺河区'),  # 2019 年撤区设县
    ('伊春', '金林区'): ('伊春', '金山屯区'),  # 2019 年金山屯区、西林区合并
    ('六安', '叶集区'): ('六安', '霍邱县'),  # 2015 年由霍邱县析出设区
    ('六盘水', '盘州市'): ('六盘水', '盘县'),  # 2017 年撤县设市
    ('南宁', '横州市'): ('南宁', '横县'),  # 2021 年撤县设市
    ('南昌', '红谷滩区'): ('南昌', '青山湖区'),  # 2020 年由青山湖区、湾里区析出设区
    ('呼伦贝尔', '新巴尔虎右旗'): ('呼伦贝尔', '新巴尔虎左旗'),  # 附录未列右旗，按相邻左旗取值，待人工确认
    ('咸阳', '彬州市'): ('咸阳', '彬县'),  # 2018 年撤县设市
    ('哈密', '伊州区'): ('哈密', '哈密市'),  # 2016 年撤地设市，原哈密市改伊州区
    ('大同', '云冈区'): ('大同', '矿区'),  # 2018 年矿区、南郊区合并为云冈区
    ('大同', '平城区'): ('大同', '城区'),  # 2018 年城区、矿区合并为平城区
    ('宜宾', '叙州区'): ('宜宾', '宜宾县'),  # 2018 年撤县设区
    ('无锡', '新吴区'): ('无锡', '滨湖区'),  # 2015 年新区改设，取值与原属滨湖区一致
    ('无锡', '梁溪区'): ('无锡', '崇安区'),  # 2015 年崇安、南长、北塘三区合并
    ('杭州', '临平区'): ('杭州', '余杭区'),  # 2021 年余杭区拆分
    ('杭州', '钱塘区'): ('杭州', '江干区'),  # 2021 年江干区、萧山区部分合并
    ('株洲', '渌口区'): ('株洲', '株洲县'),  # 2018 年撤县设区
    ('淮安', '清江浦区'): ('淮安', '清河区'),  # 2016 年清河区、清浦区合并
    ('深圳', '光明区'): ('深圳', '宝安区'),  # 2018 年由宝安区析出设区
    ('深圳', '坪山区'): ('深圳', '龙岗区'),  # 2016 年由龙岗区析出设区
    ('深圳', '龙华区'): ('深圳', '宝安区'),  # 2016 年由宝安区析出设区
    ('温州', '龙港市'): ('温州', '苍南县'),  # 2019 年由苍南县析出设市
    ('渭南', '华州区'): ('渭南', '华县'),  # 2015 年撤县设区
    ('绵阳', '安州区'): ('绵阳', '安县'),  # 2016 年撤县设区
    ('芜湖', '湾沚区'): ('芜湖', '芜湖县'),  # 2020 年撤县设区
    ('西安', '鄠邑区'): ('西安', '户县'),  # 2016 年撤县设区
    ('许昌', '建安区'): ('许昌', '许昌县'),  # 2016 年撤县设区
    ('贺州', '平桂区'): ('贺州', '钟山县'),  # 2007 年由八步区、钟山县析出设立平桂管理区
    ('赣州', '赣县区'): ('赣州', '赣县'),  # 2016 年撤县设区
    ('遵义', '播州区'): ('遵义', '遵义县'),  # 2016 年撤县设区
    ('邢台', '信都区'): ('邢台', '桥西区'),  # 2020 年桥西区改信都区
    ('邢台', '襄都区'): ('邢台', '桥东区'),  # 2020 年桥东区并入襄都区
    ('那曲', '色尼区'): ('那曲', '那曲县'),  # 2017 年撤地设市、撤县设区
    ('鄂尔多斯', '康巴什区'): ('鄂尔多斯', '东胜区'),  # 2016 年由东胜区析出设区
    ('铜陵', '义安区'): ('铜陵', '铜陵县'),  # 2015 年撤县设区
    ('铜陵', '铜官区'): ('铜陵', '铜官山区'),  # 2015 年铜官山区、狮子山区合并
    ('长治', '潞州区'): ('长治', '城区'),  # 2018 年城区、郊区合并为潞州区

    # ---- 四、附录用字与现行名不一致（5 条）----

    ('乌鲁木齐', '达坂城区'): ('乌鲁木齐', '达阪城区'),  # 附录作“达阪城区”
    ('巴音郭楞', '和静县'): ('巴音郭楞', '和静镇'),  # 附录作“和静镇”
    ('普洱', '宁洱哈尼族彝族自治县'): ('普洱', '宁洱哈尼族彝族自县'),  # 附录漏“治”字
    ('玉溪', '元江哈尼族彝族傣族自治县'): ('玉溪', '元江哈尼族彝族傣自治县'),  # 附录漏“族”字
    ('齐齐哈尔', '铁锋区'): ('齐齐哈尔', '铁峰区'),  # 附录作“铁峰区”
}

def _resolve_alias(CityName: str, DistrictName: str) -> tuple:
    """在别名表中查找现行区县名对应的附录旧名，城市名允许简称与全称互为前缀。

    参数：
        CityName: 城市名，简称（如“天津”）或 shapefile 全称（如“天津市”）均可。
        DistrictName: 现行区县名。

    返回：
        tuple: 用于查 GB50011 附录的（市名，区县名）；无别名时原样返回。
    """
    for (city, district), target in DESIGN_CODE_LOCATION_ALIASES.items():
        if district == DistrictName and (city == CityName or CityName.startswith(city) or city.startswith(CityName)):
            return target
    return CityName, DistrictName


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
    LookupCityName, LookupDistrictName = _resolve_alias(CityName, DistrictName)
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

