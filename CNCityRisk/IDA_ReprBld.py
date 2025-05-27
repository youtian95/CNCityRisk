from . import ClassifyBld
from .Logger import get_logger
logger = get_logger('IDA_ReprBld')
from MDOFModel.Tool_IDA import main_IDA

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
    ReprBldID = row['ReprBldID']
    
    try:
        NumofStories = row['story']
        FloorArea = row['area']
        StructuralType = row['structure']
        DesignInfo = {'Code': 'CN', 'SeismicDesignLevel': str(row['SeismicDesignLevel']), 'EQgroup': str(row['EQGroup']), 'SiteClass': str(row['SiteClass'])}
        OutputCSVFile = OutputDir / f'IDA_result_ReprBldID_{ReprBldID}.csv'
        WriteStructParaFile = ClassifyBld.REPR_BLD_DIR / f'StructPara_ReprBldID_{ReprBldID}.csv'
        
        main_IDA(IM_list = IM_D_list, NumofStories = NumofStories, FloorArea = FloorArea, StructuralType = StructuralType, EQMetaDataFile = EQMetaDataFile, OutputCSVFile = OutputCSVFile, DesignInfo = DesignInfo, NumPool = NumPool, TempDir = FEMTempDir, UseRelativeIM = True, WriteStructParaFile = WriteStructParaFile)
        
    except Exception as e:
        logger.error(f"Failed to complete IDA analysis for building ID {ReprBldID}: {str(e)}")
        raise

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
    logger.info(f"Starting IDA analysis with parameters: NumPool={NumPool}, StartOver={StartOver}")
    logger.info(f"Input file: {ReprBldFile}")
    logger.info(f"Output directory: {OutputDir}")
    
    OutputDir.mkdir(parents=True, exist_ok=True)

    # Read representative building data
    ReprBld_df = pd.read_csv(ReprBldFile, index_col=0)
    ReprBld_df['ReprBldID'] = ReprBld_df.index
    logger.info(f"Loaded {len(ReprBld_df)} representative buildings from file")

    # Get the list of building ID that have already been analyzed
    Existing_IDA_ReprBldID = _get_existing_IDA_ReprBldID(OutputDir)
    logger.info(f"Found {len(Existing_IDA_ReprBldID)} buildings with existing IDA results")

    # Filter out the buildings that have already been analyzed
    if not StartOver:
        original_count = len(ReprBld_df)
        ReprBld_df = ReprBld_df[~ReprBld_df['ReprBldID'].isin(Existing_IDA_ReprBldID)]
        filtered_count = len(ReprBld_df)
        logger.info(f"Filtered out {original_count - filtered_count} buildings with existing results")
        logger.info(f"Will analyze {filtered_count} buildings")
    else:
        logger.info("StartOver=True, will analyze all buildings regardless of existing results")

    if len(ReprBld_df) == 0:
        logger.warning("No buildings to analyze after filtering")
        return

    # Perform IDA for each representative building
    EQMetaDataFile = EQInputDir / 'MetaData.txt'
    logger.info(f"Using earthquake metadata file: {EQMetaDataFile}")
    
    print('Performing IDA...')
    tqdm.pandas()
    
    logger.info("Starting IDA analysis for all filtered buildings...")
    ReprBld_df.progress_apply(_IDA_1Bld, EQMetaDataFile = EQMetaDataFile, IM_D_list = IM_D_list, NumPool = NumPool, axis = 1)
    
    logger.info("Completed IDA analysis for all buildings")


def _get_existing_IDA_ReprBldID(ResultsDir: Path = OutputDir) -> list:
    '''
    Get the list of building ID that have already been analyzed
    '''
    existing_files = list(ResultsDir.glob('IDA_result_ReprBldID_*.csv'))
    existing_ids = [int(f.stem.split('_')[-1]) for f in existing_files]
    
    return existing_ids
