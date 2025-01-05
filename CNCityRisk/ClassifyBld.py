import pandas as pd
from pathlib import Path
import numpy as np
from tqdm import tqdm

from .PythonLib.BldCluster import BldCluster
import Utilities

# parameters
STRUCT_CLASS_LABELS = ['story','area','structure','SeismicDesignLevel','EQGroup','SiteClass']
PSPA_CLASS_LABELS = ['story','area','structure','SeismicDesignLevel','EQGroup','SiteClass','occupancy']
INCR_LABELS = {'area': 100} # buildings that have similar plan area are grouped together
REPR_BLD_DIR = (Path(__file__).parent / './results/ReprBld').resolve()
REPR_PSPA_BLD_DIR = (Path(__file__).parent / './results/ReprBld_PSPA').resolve()
REPR_BLD_FILE = 'ReprBld.csv'

# intermediate files
_ReprBldFile = Path.cwd() / 'Representative Buildings.csv'
_Bld2ReprBldFile = Path.cwd() / 'Bld2ReprBld.csv'

def BldClassify(CityName: str, SaveDir: Path = REPR_BLD_DIR, TargetLabels: list = STRUCT_CLASS_LABELS, IncrLabels: dict[str,float] = INCR_LABELS, StartFromScratch: bool = False, ReprColName: str = 'ReprBldID'):
    '''
    Classify buildings based on their properties, including structural type, occupancy type, design level, etc.
    Parameters:
        CityName (str). Name of the city
        SaveDir (Path). Directory to save the representative buildings
        IgnoredLabels (list). Labels that are ignored during classification
        IncrLabels (dict). Labels that are used to group buildings together
        StartFromScratch (bool). If True, the classification will be started from scratch. Otherwise, the classification will be based on the existing representative buildings.
    Returns:
        Bld2ReprBld (dict). Building to representative building mapping
        ReprBld (pd.DataFrame). Representative building properties. 
    '''
    print(f'{CityName}: Classify buildings...')

    # create a directory to save the representative buildings
    SaveDir.mkdir(parents=True, exist_ok=True)

    # existing district file names
    CityBldDir = Utilities.BLD_DATA_PATH / CityName
    DistrictNames = [file.stem for file in CityBldDir.glob('*.csv')]

    if StartFromScratch:
        # remove the existing representative buildings
        if (SaveDir/REPR_BLD_FILE).exists():
            (SaveDir/REPR_BLD_FILE).unlink()
        # remove the existing representative buildings in the district files
        for district in DistrictNames:
            Bld_gdf = Utilities.ReadDistrictBldGDF(CityName, district)
            if ReprColName in Bld_gdf.columns:
                Bld_gdf.drop(columns=[ReprColName], inplace=True)
                Utilities.SaveDistrictBldGDF(CityName, district, Bld_gdf)

    # classify district by district
    for district in tqdm(DistrictNames):
        BldClassify_District(CityName, district, SaveDir, TargetLabels, IncrLabels, ReprColName)

def _CleanIntermidiateFiles():
    '''
    Clean the intermediate files
    '''
    if _ReprBldFile.exists():
        _ReprBldFile.unlink()
    if _Bld2ReprBldFile.exists():
        _Bld2ReprBldFile.unlink() 

def BldClassify_District(CityName: str, DistrictName: str, SaveDir: Path = REPR_BLD_DIR, TargetLabels: list = STRUCT_CLASS_LABELS, IncrLabels: dict[str,float] = INCR_LABELS, ReprColName: str = 'ReprBldID'):
    '''
    Classify buildings based on their properties, including structural type, occupancy type, design level, etc. For buildings that have been classified, a representative building is assigned to the BldDirtFile.
    Parameters:
        CityName (str). Name of the city
        DistrictName (str). Name of the district
        SaveDir (Path). Directory to save the representative buildings
        TargetLabels (list). Labels that are used for classification
        IncrLabels (dict). Labels that are used to group buildings together
        ReprColName (str). Name of the column that stores the representative building ID
    Returns:
        Bld2ReprBld (dict). Building to representative building mapping
        ReprBld (pd.DataFrame). Representative building properties. 
    '''
    
    # if the representative buildings have been generated, skip the classification
    Bld_gdf = Utilities.ReadDistrictBldGDF(CityName, DistrictName)
    if ReprColName in Bld_gdf.columns:
        return None

    # create a directory to save the representative buildings
    SaveDir.mkdir(parents=True, exist_ok=True)

    # use python library, BldCluster, to classify buildings
    BldDirtFile = Utilities.BLD_DATA_PATH / CityName / f'{DistrictName}.csv'
    obj = BldCluster.BldCluster(BldDirtFile)
    obj.ClassifyBld(IgnoredLabels=None, TargetLabels=TargetLabels, **IncrLabels)
    Bld2ReprBld = obj.Bld2ReprBld
    ReprBld = obj.ReprBld

    # remove the intermediate files
    _CleanIntermidiateFiles()

    # merge the new generated representative buildings with the existing representative buildings
    if (SaveDir/REPR_BLD_FILE).exists():
        # read the existing representative buildings
        ReprBld_old = pd.read_csv(SaveDir / REPR_BLD_FILE, index_col=0)
        headers = ReprBld_old.columns.to_list()
        ReprBld_old['id_old'] = ReprBld_old.index
        ReprBld['id_new'] = ReprBld.index
        # find the new representative buildings that are not in the existing representative buildings
        df_temp = ReprBld_old.merge(ReprBld, on=headers, how='right')
        row_not_in_old = df_temp[df_temp['id_old'].isna()]
        if row_not_in_old.empty:
            ReprBld_new = ReprBld_old.copy(deep=True)
            ReprBld_new['id_new'] = np.nan
        else:
            ReprBld_new = pd.concat([ReprBld_old, row_not_in_old], ignore_index=True)
        # update the representative buildings
        ReprBld = ReprBld_new[headers]
        # update the mapping
        def find_new_id(x):
            if x in row_not_in_old.index:
                return ReprBld_new[ReprBld_new['id_new']==x].index[0].astype(int)
            else:
                return df_temp[df_temp['id_new']==x]['id_old'].values[0].astype(int)
        Bld2ReprBld['Representative Buildings index'] = Bld2ReprBld['Representative Buildings index'].apply(find_new_id)

    # assign the representative building to the BldDirtFile
    Bld_df = pd.read_csv(BldDirtFile)
    Bld2ReprBld = dict(zip(Bld2ReprBld['Original Buildings index'].to_list(), Bld2ReprBld['Representative Buildings index'].to_list()))
    Bld_df[ReprColName] = Bld_df.index.map(Bld2ReprBld)

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
    ReprBld = pd.read_csv(ReprDir / REPR_BLD_FILE, index_col=0)
    
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

    