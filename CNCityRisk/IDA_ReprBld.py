from . import ClassifyBld
from .Logger import get_logger
logger = get_logger('IDA_ReprBld')
from MDOFModel.Tool_IDA import main_IDA

from pathlib import Path
import pandas as pd
from tqdm import tqdm
import os

# Parameters
ReprBldFile = ClassifyBld.REPR_BLD_DIR / ClassifyBld.REPR_BLD_FILE
IM_D_list = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 5, 10] # Unit: times of Sa(T1) with 475-year return period
OutputDir = (Path(__file__).parent / './results/IDA_Results').resolve()
EQInputDir = (Path(__file__).parent / './Data/FEMA_P-695_far-field_ground_motions').resolve()
FEMTempDir = (Path(__file__).parent / './results/temp').resolve()

def _IDA_1Bld(row, EQMetaDataFile: Path, IM_D_list: list, NumPool: int = 1, IDAOutputFormat: list = ['csv']):
    '''
    Incremental dynamic analysis for one representative building
    Parameters:
        row (pd.Series). Row of representative building data
        IM_D_list (list). List of intensity measures. Unit: times of Sa(T1) with 475-year return period
        EQMetaDataFile (Path). Meta info file of earthquake input
        NumPool (int). Number of parallel processes
        IDAOutputFormat (list). List of output formats. ['csv', 'hdf5']
    Resulting files:
        OutputDir/IDA_result_ReprBldID_{ReprBldID}.csv (if 'csv' in IDAOutputFormat)
        OutputDir/IDA_Results.hdf5 (if 'hdf5' in IDAOutputFormat)
        ClassifyBld.REPR_BLD_DIR/StructPara_ReprBldID_{ReprBldID}.csv
    '''
    ReprBldID = row['ReprBldID']
    
    try:
        NumofStories = row['story']
        FloorArea = row['area']
        StructuralType = row['structure']
        DesignInfo = {'Code': 'CN', 'SeismicDesignLevel': str(row['SeismicDesignLevel']), 'EQgroup': str(row['EQGroup']), 'SiteClass': str(row['SiteClass'])}
        
        if 'csv' in IDAOutputFormat:
            OutputCSVFile = OutputDir / f'IDA_result_ReprBldID_{ReprBldID}.csv'
        else:
            OutputCSVFile = FEMTempDir / f'IDA_result_ReprBldID_{ReprBldID}_temp.csv'
            FEMTempDir.mkdir(parents=True, exist_ok=True)

        WriteStructParaFile = ClassifyBld.REPR_BLD_DIR / f'StructPara_ReprBldID_{ReprBldID}.csv'
        
        main_IDA(IM_list = IM_D_list, NumofStories = NumofStories, FloorArea = FloorArea, StructuralType = StructuralType, EQMetaDataFile = EQMetaDataFile, OutputCSVFile = OutputCSVFile, DesignInfo = DesignInfo, NumPool = NumPool, TempDir = FEMTempDir, UseRelativeIM = True, WriteStructParaFile = WriteStructParaFile)
        
        # Save to HDF5 if required
        if 'hdf5' in IDAOutputFormat:
            HDF5File = OutputDir / 'IDA_Results.hdf5'
            if OutputCSVFile.exists():
                try:
                    df_res = pd.read_csv(OutputCSVFile)
                    key = f'IDA_result_ReprBldID_{ReprBldID}'
                    df_res.to_hdf(HDF5File, key=key, mode='a', complib='zlib', complevel=5)
                except Exception as e:
                    logger.error(f"Failed to write HDF5 for building {ReprBldID}: {e}")
            else:
                logger.warning(f"Output CSV not found for building {ReprBldID}, cannot save to HDF5")

        # Remove temp CSV file if not needed
        if 'csv' not in IDAOutputFormat and OutputCSVFile.exists():
            try:
                os.remove(OutputCSVFile)
            except Exception as e:
                logger.warning(f"Failed to remove temp file {OutputCSVFile}: {e}")
        
    except Exception as e:
        logger.error(f"Failed to complete IDA analysis for building ID {ReprBldID}: {str(e)}")
        raise

def IDA_analysis(ReprBldFile: Path = ReprBldFile, IM_D_list: list = IM_D_list, EQInputDir: Path = EQInputDir, OutputDir: Path = OutputDir, NumPool: int = 1, StartOver: bool = False, IDAOutputFormat: list = ['csv']):
    '''
    Incremental dynamic analysis for individual representative buildings
    Parameters:
        ReprBldFile (Path). File of representative buildings
        IM_D_list (list). List of intensity measures. Unit: times of Sa(T1) with 475-year return period
        EQInputDir (Path). Directory of earthquake input
        OutputDir (Path). Directory of IDA results
        NumPool (int). Number of parallel
        StartOver (bool). Whether to start over. if True, IDA will be performed for all representative buildings from ID=0. If False, IDA will be performed for representative buildings that have not been analyzed yet.
        IDAOutputFormat (list). List of output formats. ['csv', 'hdf5']
    '''
    logger.info(f"Starting IDA analysis with parameters: NumPool={NumPool}, StartOver={StartOver}, IDAOutputFormat={IDAOutputFormat}")
    logger.info(f"Input file: {ReprBldFile}")
    logger.info(f"Output directory: {OutputDir}")
    
    OutputDir.mkdir(parents=True, exist_ok=True)

    # Read representative building data
    ReprBld_df = pd.read_csv(ReprBldFile, index_col=0)
    ReprBld_df['ReprBldID'] = ReprBld_df.index
    logger.info(f"Loaded {len(ReprBld_df)} representative buildings from file")

    # Get the list of building ID that have already been analyzed
    Existing_IDA_ReprBldID = _get_existing_IDA_ReprBldID(OutputDir, IDAOutputFormat)
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
    ReprBld_df.progress_apply(_IDA_1Bld, EQMetaDataFile = EQMetaDataFile, IM_D_list = IM_D_list, NumPool = NumPool, IDAOutputFormat=IDAOutputFormat, axis = 1)
    
    logger.info("Completed IDA analysis for all buildings")


def _get_existing_IDA_ReprBldID(ResultsDir: Path, IDAOutputFormat: list = ['csv']) -> list:
    '''
    Get the list of building ID that have already been analyzed
    '''
    existing_ids = set()
    
    if 'csv' in IDAOutputFormat:
        existing_files = list(ResultsDir.glob('IDA_result_ReprBldID_*.csv'))
        existing_ids.update([int(f.stem.split('_')[-1]) for f in existing_files])
        
    if 'hdf5' in IDAOutputFormat:
        HDF5File = ResultsDir / 'IDA_Results.hdf5'
        try:
            with pd.HDFStore(HDF5File, mode='r') as store:
                keys = store.keys()
                for key in keys:
                    try:
                        # key format: /IDA_result_ReprBldID_1
                        bid = int(key.split('_')[-1])
                        existing_ids.add(bid)
                    except:
                        pass
        except Exception as e:
            logger.warning(f"Failed to read HDF5 file {HDF5File}: {e}")
    
    return list(existing_ids)
