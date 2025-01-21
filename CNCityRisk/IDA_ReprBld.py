from . import ClassifyBld
from .PythonLib.MDOFModel.Tool_IDA import main_IDA

from pathlib import Path
import pandas as pd
from tqdm import tqdm

# Parameters
ReprBldFile = ClassifyBld.REPR_BLD_DIR / ClassifyBld.REPR_BLD_FILE
IM_D_list = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 5, 10] # Unit: times of Sa(T1) with 475-year return period
OutputDir = (Path(__file__).parent / './results/IDA_Results').resolve()
EQInputDir = (Path(__file__).parent / './Data/FEMA_P-695_far-field_ground_motions').resolve()
FEMTempDir = (Path(__file__).parent / './results/temp').resolve()

def _IDA_1Bld(row, EQMetaDataFile: Path, IM_D_list: list, NumPool: int = 1):
    '''
    Incremental dynamic analysis for one representative building
    Parameters:
        row (pd.Series). Row of representative building data
        IM_D_list (list). List of intensity measures. Unit: times of Sa(T1) with 475-year return period
        EQMetaDataFile (Path). Meta info file of earthquake input
        NumPool (int). Number of parallel processes
    '''
    NumofStories = row['story']
    FloorArea = row['area']
    StructuralType = row['structure']
    DesignInfo = {'Code': 'CN', 'SeismicDesignLevel': str(row['SeismicDesignLevel']), 'EQgroup': str(row['EQGroup']), 'SiteClass': str(row['SiteClass'])}
    ReprBldID = row['ReprBldID']
    OutputCSVFile = OutputDir / f'IDA_result_ReprBldID_{ReprBldID}.csv'
    WriteStructParaFile = ClassifyBld.REPR_BLD_DIR / f'StructPara_ReprBldID_{ReprBldID}.csv'

    main_IDA(IM_list = IM_D_list, NumofStories = NumofStories, FloorArea = FloorArea, StructuralType = StructuralType, EQMetaDataFile = EQMetaDataFile, OutputCSVFile = OutputCSVFile, DesignInfo = DesignInfo, NumPool = NumPool, TempDir = FEMTempDir, UseRelativeIM = True, WriteStructParaFile = WriteStructParaFile)

def IDA_analysis(ReprBldFile: Path = ReprBldFile, IM_D_list: list = IM_D_list, EQInputDir: Path = EQInputDir, OutputDir: Path = OutputDir, NumPool: int = 1, StartOver: bool = False):
    '''
    Incremental dynamic analysis for individual representative buildings
    Parameters:
        ReprBldFile (Path). File of representative buildings
        IM_D_list (list). List of intensity measures. Unit: times of Sa(T1) with 475-year return period
        EQInputDir (Path). Directory of earthquake input
        OutputDir (Path). Directory of IDA results
        NumPool (int). Number of parallel
        StartOver (bool). Whether to start over. if True, IDA will be performed for all representative buildings from ID=0. If False, IDA will be performed for representative buildings that have not been analyzed yet.
    '''
    OutputDir.mkdir(parents=True, exist_ok=True)

    # Read representative building data
    ReprBld_df = pd.read_csv(ReprBldFile, index_col=0)
    ReprBld_df['ReprBldID'] = ReprBld_df.index

    # Get the list of building ID that have already been analyzed
    Existing_IDA_ReprBldID = _get_existing_IDA_ReprBldID(OutputDir)

    # Filter out the buildings that have already been analyzed
    if not StartOver:
        ReprBld_df = ReprBld_df[~ReprBld_df['ReprBldID'].isin(Existing_IDA_ReprBldID)]

    # Perform IDA for each representative building
    EQMetaDataFile = EQInputDir / 'MetaData.txt'
    print('Performing IDA...')
    tqdm.pandas()
    ReprBld_df.progress_apply(_IDA_1Bld, EQMetaDataFile = EQMetaDataFile, IM_D_list = IM_D_list, NumPool = NumPool, axis = 1)


def _get_existing_IDA_ReprBldID(ResultsDir: Path = OutputDir) -> list:
    '''
    Get the list of building ID that have already been analyzed
    '''
    return [int(f.stem.split('_')[-1]) for f in ResultsDir.glob('IDA_result_ReprBldID_*.csv')]
