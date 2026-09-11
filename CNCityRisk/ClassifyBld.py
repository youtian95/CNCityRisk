"""
城市建筑代表类型分类模块。

使用流程：读取城市各区县的 Parquet 建筑数据，按指定属性分组，并写回代表建筑编号。
IDA 分类不区分层面积，PSPA 分类仍保留层面积和建筑用途。

输入：城市名称、分类标签、面积分组步长及代表建筑结果目录。
输出：区县建筑与代表建筑的映射，以及代表建筑属性表。
"""

import pandas as pd
from pathlib import Path
import tempfile
import numpy as np
from tqdm import tqdm
import time

from BldCluster import BldCluster
from . import Utilities
from .Logger import get_logger

# Get module logger
logger = get_logger('ClassifyBld')

# parameters
STRUCT_CLASS_LABELS = ['story','structure','SeismicDesignLevel','EQGroup','SiteClass']
PSPA_CLASS_LABELS = ['story','area','structure','SeismicDesignLevel','EQGroup','SiteClass','occupancy']
INCR_LABELS = {'area': 100} # buildings that have similar plan area are grouped together
REPR_BLD_DIR = (Path(__file__).parent / './results/ReprBld').resolve()
REPR_PSPA_BLD_DIR = (Path(__file__).parent / './results/ReprBld_PSPA').resolve()
REPR_BLD_FILE = 'ReprBld.csv'

# intermediate files
_ReprBldFile = Path.cwd() / 'Representative Buildings.csv'
_Bld2ReprBldFile = Path.cwd() / 'Bld2ReprBld.csv'

def BldClassify(CityName: str=None, min_lon=None, max_lon=None, min_lat=None, max_lat=None, SaveDir: Path = REPR_BLD_DIR, TargetLabels: list = STRUCT_CLASS_LABELS, IncrLabels: dict[str,float] = INCR_LABELS, StartFromScratch: bool = False, ReprColName: str = 'ReprBldID'):
    '''
    Classify buildings based on their properties, including structural type, occupancy type, design level, etc.

    Two modes are supported:
        1. City mode: If CityName is provided, the buildings within the city will be classified.
        2. Coordinate mode: If min_lon, max_lon, min_lat, max_lat are provided, the buildings within the specified coordinate range will be classified (automatically rounded to 0.01 precision).

    Parameters:
        CityName (str, optional): Name of the city. If provided, will use city mode.
        min_lon (float, optional): Minimum longitude of the region. Required for coordinate mode.
        max_lon (float, optional): Maximum longitude of the region. Required for coordinate mode.
        min_lat (float, optional): Minimum latitude of the region. Required for coordinate mode.
        max_lat (float, optional): Maximum latitude of the region. Required for coordinate mode.
        SaveDir (Path): Directory to save the representative buildings
        TargetLabels (list): Labels that are used for classification
        IncrLabels (dict): Labels that are used to group buildings together
        StartFromScratch (bool): If True, the classification will be started from scratch. Otherwise, the classification will be based on the existing representative buildings.
        ReprColName (str): Name of the column that stores the representative building ID
    Returns:
        Bld2ReprBld (dict): Building to representative building mapping
        ReprBld (pd.DataFrame): Representative building properties. 
    '''    

    if CityName is None and all(param is not None for param in [min_lon, max_lon, min_lat, max_lat]):
        # Coordinate mode - round to 0.01 precision
        region_name, _ = Utilities.generate_region_name(min_lon, max_lon, min_lat, max_lat)
    else:
        region_name = CityName

    logger.info(f'Starting to classify buildings for {region_name}...')
    start_time = time.time()

    
    # create SaveDir if not exists
    if not SaveDir.exists():
        logger.debug(f'Creating directory: {SaveDir}')
        SaveDir.mkdir(parents=True, exist_ok=True)

    # existing district file names
    RegionBldDir = Utilities.BLD_DATA_PATH / region_name
    DistrictNames = [file.stem for file in RegionBldDir.glob('*.parquet')]

    if not RegionBldDir.exists():
        logger.error(f'{RegionBldDir} does not exist. Please run prepare_building_data first.')
        return None

    if StartFromScratch:
        # remove the existing representative buildings
        if (SaveDir/REPR_BLD_FILE).exists():
            (SaveDir/REPR_BLD_FILE).unlink()
        # remove the existing representative buildings in the district files
        for district in DistrictNames:
            Bld_gdf = Utilities.ReadDistrictBldGDF(CityName, district)
            if ReprColName in Bld_gdf.columns:
                Bld_gdf.drop(columns=[ReprColName], inplace=True)
                Utilities.SaveDistrictBldGDF(RegionBldDir, district, Bld_gdf)

    # classify district by district
    for district in tqdm(DistrictNames):
        BldClassify_District(region_name, district, SaveDir, TargetLabels, IncrLabels, ReprColName)

    end_time = time.time()
    logger.info(f'Building classification for {region_name} completed, total time: {end_time-start_time:.2f} seconds')

def _CleanIntermidiateFiles():
    '''
    Clean the intermediate files
    '''
    if _ReprBldFile.exists():
        _ReprBldFile.unlink()
    if _Bld2ReprBldFile.exists():
        _Bld2ReprBldFile.unlink() 

def BldClassify_District(CityName: str, DistrictName: str, SaveDir: Path = REPR_BLD_DIR, TargetLabels: list = STRUCT_CLASS_LABELS, IncrLabels: dict[str,float] = INCR_LABELS, ReprColName: str = 'ReprBldID', overwrite: bool = True):
    '''
    Classify buildings based on their properties, including structural type, occupancy type, design level, etc. For buildings that have been classified, a representative building is assigned to the BldDirtFile.
    Parameters:
        CityName (str): Name of the city
        DistrictName (str): Name of the district
        SaveDir (Path): Directory to save the representative buildings
        TargetLabels (list): Labels that are used for classification
        IncrLabels (dict): Labels that are used to group buildings together
        ReprColName (str): Name of the column that stores the representative building ID
        overwrite (bool): If True, the representative buildings will be overwritten. 
    Returns:
        Bld2ReprBld (dict): Building to representative building mapping
        ReprBld (pd.DataFrame): Representative building properties. 
    '''
    
    # if the representative buildings have been generated, skip the classification
    Bld_gdf = Utilities.ReadDistrictBldGDF(CityName, DistrictName)
    if not overwrite:
        if ReprColName in Bld_gdf.columns:
            return None

    # create a directory to save the representative buildings
    SaveDir.mkdir(parents=True, exist_ok=True)

    # BldCluster 目前只接受 CSV 路径，因此使用临时 CSV 兼容 Parquet 主数据。
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp_file:
        temp_csv_path = Path(temp_file.name)
    try:
        Bld_gdf.to_csv(temp_csv_path, index=False)
        obj = BldCluster.BldCluster(temp_csv_path)
        all_columns = Bld_gdf.columns.tolist()
        IgnoredLabels = [col for col in all_columns if col not in TargetLabels]
        # 只向 BldCluster 传入本次分类实际使用的连续变量分组步长。
        active_increments = {label: increment for label, increment in IncrLabels.items() if label in TargetLabels}
        obj.ClassifyBld(IgnoredLabels=IgnoredLabels, **active_increments)
        Bld2ReprBld = obj.Bld2ReprBld
        ReprBld = obj.ReprBld
    finally:
        temp_csv_path.unlink(missing_ok=True)

    # Use the function to align datatypes
    ReprBld = _align_dataframe_dtypes(
        source_df=ReprBld,
        target_df=Bld_gdf
    )

    # remove the intermediate files
    _CleanIntermidiateFiles()

    # 将本区新生成的代表建筑与既有代表建筑合并，并保留既有的稀疏 ID。
    if (SaveDir/REPR_BLD_FILE).exists():
        ReprBld_old = ReadReprBld(ReprDir=SaveDir)
        headers = ReprBld_old.columns.to_list()
        ReprBld_compare = ReprBld[headers].copy()
        ReprBld_compare['id_new'] = ReprBld_compare.index
        ReprBld_old_compare = ReprBld_old.copy()
        ReprBld_old_compare['id_old'] = ReprBld_old_compare.index

        # 与既有逻辑一致，将连续分组标签按整数值比较。
        for col in headers:
            if ReprBld_old[col].dtype == 'float64':
                ReprBld_old_compare[col] = ReprBld_old_compare[col].round(0)
                ReprBld_compare[col] = ReprBld_compare[col].round(0)

        matched = ReprBld_compare.merge(ReprBld_old_compare, on=headers, how='left', validate='one_to_one')
        missing = matched['id_old'].isna()
        next_id = int(ReprBld_old.index.max()) + 1 if len(ReprBld_old) else 0
        matched.loc[missing, 'id_old'] = np.arange(next_id, next_id + int(missing.sum()))
        matched['id_old'] = matched['id_old'].astype(int)

        # 新组合从历史最大 ID 后继续编号，避免与迁移后保留的稀疏 ID 冲突。
        if missing.any():
            new_local_ids = matched.loc[missing, 'id_new'].astype(int).to_numpy()
            new_global_ids = matched.loc[missing, 'id_old'].astype(int).to_numpy()
            new_rows = ReprBld.loc[new_local_ids, headers].copy()
            new_rows.index = new_global_ids
            ReprBld = pd.concat([ReprBld_old, new_rows]).sort_index()
        else:
            ReprBld = ReprBld_old

        local_to_global = dict(zip(matched['id_new'].astype(int), matched['id_old'].astype(int)))
        Bld2ReprBld['Representative Buildings index'] = Bld2ReprBld['Representative Buildings index'].map(local_to_global).astype(int)

    # assign the representative building to the BldDirtFile
    Bld_df = Bld_gdf.copy()
    Bld2ReprBld = dict(zip(Bld2ReprBld['Original Buildings index'].to_list(), Bld2ReprBld['Representative Buildings index'].to_list()))

    # BldCluster 会把临时 CSV 重编号为 0...N-1；面积筛选后的 Parquet 索引可能不连续，
    # 因此必须按当前行位置写回映射，不能使用原 Parquet 索引直接查找。
    row_positions = pd.Series(np.arange(len(Bld_df)), index=Bld_df.index)
    mapped_repr_ids = row_positions.map(Bld2ReprBld)
    if mapped_repr_ids.isna().any():
        missing_count = int(mapped_repr_ids.isna().sum())
        raise ValueError(
            f'{CityName}-{DistrictName} 有 {missing_count} 栋建筑未匹配到 {ReprColName}，停止写入。'
        )
    Bld_df[ReprColName] = mapped_repr_ids.astype(int)

    # save the representative building and the mapping
    ReprBld.to_csv(SaveDir / REPR_BLD_FILE, index=True)
    Utilities.SaveDistrictBldGDF(CityName, DistrictName, Bld_df)

    return Bld2ReprBld, ReprBld

def ReadReprBld(ReprDir: Path = REPR_BLD_DIR) -> pd.DataFrame:

    # basic properties of representative buildings
    ReprBld = pd.read_csv(ReprDir / REPR_BLD_FILE, index_col=0, dtype={'story': int, 'area': float, 'structure': str, 'SeismicDesignLevel': str, 'EQGroup': str, 'SiteClass': str, 'occupancy': str})

    return ReprBld

def ReadReprBldStructPara(property: str, ReprDir: Path = REPR_BLD_DIR) -> dict:
    '''
    Read structural parameters of representative buildings.
    Parameters:
        properties (str): Name of the properties. e.g. 'T1 (s)'
        ReprDir (Path): Directory of the representative buildings. Default is REPR_BLD_DIR.
    Returns:
        repr_blds_properties (dict): The properties of representative buildings.
    '''
    # basic properties of representative buildings
    ReprBld = ReadReprBld(ReprDir = ReprDir)
    
    # Read structural parameters of representative buildings
    repr_blds_properties = {}
    for index, row in tqdm(ReprBld.iterrows(), desc="Reading representative buildings properties", total=ReprBld.shape[0]):
        file_path = ReprDir / f'StructPara_ReprBldID_{index}.csv'
        if file_path.exists():
            repr_bld_data = pd.read_csv(file_path, header=0, nrows=1).to_dict(orient='records')[0]
            repr_bld_data_second = pd.read_csv(file_path, header=0, skiprows=2).to_dict(orient='records')[0]
            repr_bld_data.update(repr_bld_data_second)
            repr_blds_properties[index] = repr_bld_data[property]

    return repr_blds_properties

    
# modify the column types ReprBld to be consistent with Bld_gdf
def _align_dataframe_dtypes(source_df: pd.DataFrame, target_df: pd.DataFrame, type_mapping=None, logger=logger):
    """
    Align the data types of columns in source_df to match those in target_df.
    
    Parameters:
        source_df (pd.DataFrame): DataFrame whose data types need to be modified
        target_df (pd.DataFrame): DataFrame with reference data types
        type_mapping (dict): Dictionary of column names and their desired types
        logger: Logger to record warnings
    
    Returns:
        pd.DataFrame: DataFrame with aligned data types
    """
    # Make a copy to avoid modifying the original
    result_df = source_df.copy()
    
    # Get dtypes from target_df for columns that exist in both dataframes
    common_columns = set(result_df.columns) & set(target_df.columns)
    for col in common_columns:
        try:
            # 方法1: 使用 pd.api.types.infer_dtype 推断实际类型
            if str(target_df[col].dtype) == 'object':
                inferred_type = pd.api.types.infer_dtype(target_df[col], skipna=True)
                if inferred_type == 'string':
                    dtype = 'str'
                elif inferred_type == 'integer':
                    dtype = 'int64'
                elif inferred_type == 'floating':
                    dtype = 'float64'
                elif inferred_type == 'boolean':
                    dtype = 'bool'
                else:
                    dtype = target_df[col].dtype
            else:
                dtype = target_df[col].dtype
            
            # 方法2: 或者尝试使用 pd.to_numeric 检测数值类型
            # if target_df[col].dtype == 'object':
            #     # 尝试转换为数值类型
            #     numeric_series = pd.to_numeric(target_df[col], errors='coerce')
            #     if not numeric_series.isna().all():
            #         # 如果能转换为数值，检查是否为整数
            #         if numeric_series.dropna().apply(lambda x: x.is_integer()).all():
            #             dtype = 'int64'
            #         else:
            #             dtype = 'float64'
            #     else:
            #         dtype = 'str'  # 保持为字符串类型
            # else:
            #     dtype = target_df[col].dtype
            
            result_df[col] = result_df[col].astype(dtype)
        except (ValueError, TypeError):
            logger.warning(f"Could not convert column '{col}' to dtype {dtype}")

    # Apply specific type mappings if provided
    if type_mapping:
        for col, dtype in type_mapping.items():
            if col in result_df.columns:
                try:
                    result_df[col] = result_df[col].astype(dtype)
                except (ValueError, TypeError):
                    logger.warning(f"Could not convert column '{col}' to {dtype}")
    
    return result_df
