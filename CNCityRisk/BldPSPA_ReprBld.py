# Perform probabilistic seismic loss assessment for individual representative buildings. Culmulative distribution functions (CDF) of the losses given any IM are provided. 

from . import ClassifyBld
from MDOFModel import Tool_LossAssess
from . import IDA_ReprBld
from . import Utilities
from . import Logger

from pathlib import Path
import pandas as pd
import shutil
from tqdm import tqdm
from scipy.stats import ecdf
import h5py
import numpy as np

# Initialize logger
logger = Logger.get_logger(__name__)

# Parameters
ReprBldDir = ClassifyBld.REPR_BLD_DIR
ReprBldPSPADir = ClassifyBld.REPR_PSPA_BLD_DIR
IDA_results_dir = IDA_ReprBld.OutputDir
OutputDir = (Path(__file__).parent / './results/IndBldLoss').resolve()
hdf5File = OutputDir / 'LossSimReprBld.hdf5'

DS_types = ['None', 'Slight', 'Moderate', 'Extensive', 'Complete']
LossTypes1 = ['DS_Struct', 'DS_NonStruct_DriftSen', 'DS_NonStruct_AccelSen']
LossTypes2 = ['RepairCost_Total', 'RepairCost_Struct', 'RepairCost_NonStruct_DriftSen','RepairCost_NonStruct_AccelSen', 'RepairTime', 'RecoveryTime', 'FunctionLossTime']


def _LossSim_1Bld(row, h5pyFile: h5py.File, IDA_result_dir: Path, OutputDir: Path = None, N_Sim: int = 1000, BetaM: float = 0.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Simulate losses for one representative building

    Parameters:
        row (pd.Series). Row of representative building data. It should contain the following columns: 'ReprBldID', 'ReprPSPAID', 'story', 'area', 'structure', 'SeismicDesignLevel', 'occupancy'
        h5pyFile (h5py.File). HDF5 file to store the results
        IDA_result_dir (Path). Directory containing IDA results
        OutputDir (Path). Directory to store the output files. If None, no output files will be generated
        N_Sim (int). the number of simulations for each IM within IM_list
        BetaM (float). Beta value for structural modeling uncertainty

    Returns:
        SimEDP (pd.DataFrame). Simulated EDPs
        BldLoss (pd.DataFrame). Building loss results
    
    Results:
        BldLoss_ReprPSPAID_{ID_repr_PSPA}.csv: Building loss results
        SimEDP_ReprPSPAID_{ID_repr_PSPA}.csv: Simulated EDPs
    '''

    ID_repr_PSPA = int(row['ReprPSPAID'])
    ID_repr_IDA = int(row['ReprBldID'])
    logger.debug(f"Starting loss simulation for ReprPSPAID: {ID_repr_PSPA}, ReprBldID: {ID_repr_IDA}")
    
    NumofStories = int(row['story'])
    FloorArea = float(row['area'])
    StructuralType = row['structure']
    DesignInfo = {}
    DesignInfo['SeismicDesignLevel'] = row['SeismicDesignLevel']
    DesignInfo['Code'] = 'CN'
    OccupancyClass = row['occupancy'].strip() if pd.notna(row['occupancy']) and row['occupancy'].strip() and row['occupancy'].strip().lower() != 'unknown' else 'RES3'

    IDA_result_file = IDA_result_dir / f'IDA_result_ReprBldID_{ID_repr_IDA}.csv'
    if not IDA_result_file.exists():
        logger.error(f"IDA result file not found: {IDA_result_file}")
        raise FileNotFoundError(f"IDA result file not found: {IDA_result_file}")
    
    IDA_result = pd.read_csv(IDA_result_file)
    # read IM list
    unique_IMs = IDA_result['IM'].unique()
    IM_list = sorted(unique_IMs)
    logger.debug(f"Processing {len(IM_list)} IM values for building {ID_repr_PSPA}")

    N_Sim = [N_Sim] * len(IM_list)

    # Simulate losses for the building
    try:
        SimEDP, BldLoss = Tool_LossAssess.Simulate_losses_given_IM_basedon_IDA(
            IDA_result_file, IM_list, N_Sim, BetaM,
            OutputDir, NumofStories, FloorArea, StructuralType, DesignInfo, OccupancyClass
        )
        logger.debug(f"Loss simulation completed for building {ID_repr_PSPA}")
    except Exception as e:
        logger.error(f"Error in loss simulation for building {ID_repr_PSPA}: {str(e)}")
        raise

    # save the results to the hdf5 file
    try:
        grp = h5pyFile.create_group(f'ReprBldID_{ID_repr_PSPA}')
        IM_list = sorted(BldLoss['IM'].unique())
        i_IM = -1
        for im in IM_list:
            i_IM += 1
            grp_im = grp.create_group(f'IM{i_IM}')
            grp_im.attrs['IM'] = im
            for losstype in ['DS_Struct', 'DS_NonStruct_DriftSen', 'DS_NonStruct_AccelSen', 'RepairCost_Total', 'RepairCost_Struct', 'RepairCost_NonStruct_DriftSen','RepairCost_NonStruct_AccelSen', 'RepairTime', 'RecoveryTime', 'FunctionLossTime']:
                if losstype in BldLoss.columns:
                    grp_im_loss = grp_im.create_group(losstype)
                    BldLoss_IM: pd.Series = BldLoss.loc[BldLoss['IM'] == im, losstype]
                    if losstype.startswith('DS'):
                        # calculate probability of each damage state
                        DS_prob = BldLoss_IM.value_counts(normalize=True)
                        for DS, prob in DS_prob.items():
                            # 损伤状态概率是标量；HDF5 标量数据集不支持分块和压缩。
                            grp_im_loss.create_dataset(DS, data=np.float32(prob), dtype=np.float32)
                    else:
                        # save CDF of the loss
                        quantiles = np.linspace(0, 1, 101)
                        ecdf_values = np.quantile(BldLoss_IM, quantiles)
                        grp_im_loss.create_dataset('quantiles', data=ecdf_values, dtype=np.float32, compression="gzip", compression_opts=6)
        logger.debug(f"Results saved to HDF5 for building {ID_repr_PSPA}")
    except Exception as e:
        logger.error(f"Error saving results to HDF5 for building {ID_repr_PSPA}: {str(e)}")
        raise

    # move the result files to the output directory
    if OutputDir is not None:
        try:
            shutil.move(OutputDir / 'BldLoss.csv', OutputDir / f'BldLoss_ReprPSPAID_{ID_repr_PSPA}.csv')
            shutil.move(OutputDir / 'SimEDP.csv', OutputDir / f'SimEDP_ReprPSPAID_{ID_repr_PSPA}.csv')
            logger.debug(f"Result files moved for building {ID_repr_PSPA}")
        except Exception as e:
            logger.warning(f"Error moving result files for building {ID_repr_PSPA}: {str(e)}")

    return SimEDP, BldLoss

def LossAssess(ReprBldPSPADir: Path = ReprBldPSPADir, ReprBldDir: Path = ReprBldDir, IDA_results_dir: Path = IDA_results_dir, OutputDir: Path = OutputDir, StartOver: bool = False) -> None:
    '''
    Probabilistic seismic loss assessment for individual representative buildings

    Parameters:
        ReprBldPSPADir (Path). Directory containing representative building data for PSPA
        ReprBldDir (Path). Directory containing representative building data for IDA
        IDA_results_dir (Path). Directory containing IDA results
        OutputDir (Path). Directory to store the output files
        StartOver (bool). If True, the function will re-run the loss simulations for the representative buildings in ReprBldPSPADir. If False, the function will skip the buildings that have already been simulated.

    Results:
        LossSimReprBld.hdf5: HDF5 file containing the results of loss simulations. 
            /ReprBldID_{ID_repr_PSPA}
                /IM{i_IM}
                    attrs:
                        IM: IM values
                    /DS_Struct
                        /None: probability of DS = None
                        /Slight
                        /Moderate
                        /Extensive
                        /Complete
                    /DS_NonStruct_DriftSen
                    /DS_NonStruct_AccelSen
                    /RepairCost_Total
                        /quantiles: loss values corresponding to 0-100 quantiles
                    /RepairCost_Struct
                        /quantiles
                    /RepairCost_NonStruct_DriftSen
                        /quantiles
                    /RepairCost_NonStruct_AccelSen
                        /quantiles
                    /RepairTime
                        /quantiles
                    /RecoveryTime
                        /quantiles
                    /FunctionLossTime
                        /quantiles
    '''

    logger.info("Starting probabilistic seismic loss assessment for representative buildings")
    logger.info(f"ReprBldPSPADir: {ReprBldPSPADir}")
    logger.info(f"IDA_results_dir: {IDA_results_dir}")
    logger.info(f"OutputDir: {OutputDir}")
    logger.info(f"StartOver: {StartOver}")

    OutputDir.mkdir(parents=True, exist_ok=True)

    # Read representative building data for PSPA
    logger.info("Reading representative building data for PSPA")
    ReprBld_df = ClassifyBld.ReadReprBld(ReprBldPSPADir)
    ReprBld_df['ReprPSPAID'] = ReprBld_df.index
    logger.info(f"Found {len(ReprBld_df)} representative buildings for PSPA")

    # determine repr ID of IDA for each building in PSPA
    logger.info("Mapping PSPA buildings to IDA buildings")
    ReprBld_df_IDA = ClassifyBld.ReadReprBld(ReprBldDir)
    ReprBld_df_IDA['ReprBldID'] = ReprBld_df_IDA.index
    common_columns = list(set(ReprBld_df.columns) & set(ReprBld_df_IDA.columns))
    num_rows_before_merge = len(ReprBld_df)
    ReprBld_df = ReprBld_df.merge(ReprBld_df_IDA, on=common_columns, how='left')

    # Check if the number of rows in ReprBld_df remains the same before and after merging
    if len(ReprBld_df) != num_rows_before_merge:
        logger.error("The number of rows in ReprBld_df has changed after merging")
        raise ValueError("The number of rows in ReprBld_df has changed after merging.")
    
    logger.info(f"Successfully mapped {len(ReprBld_df)} buildings")

    with h5py.File(hdf5File, "a") as f:
        if not StartOver:
            # skip buildings that have already been simulated
            ReprBldIDs = [key.split('_')[-1] for key in f.keys()]
            buildings_to_skip = ReprBld_df['ReprPSPAID'].astype(str).isin(ReprBldIDs)
            skipped_count = buildings_to_skip.sum()
            ReprBld_df = ReprBld_df[~buildings_to_skip]
            logger.info(f"Skipped {skipped_count} buildings that have already been simulated")
        else:
            # remove the repr buildings in hdf5File that are included in ReprBldPSPADir, so that the loss simulations can be re-run
            removed_count = 0
            for repr_bld_id in ReprBld_df['ReprPSPAID'].astype(str):
                if repr_bld_id in f:
                    del f[f'ReprBldID_{repr_bld_id}']
                    removed_count += 1
            logger.info(f"Removed {removed_count} existing building results from HDF5 file")

        # Perform loss simulations for each representative building
        buildings_to_process = len(ReprBld_df)
        logger.info(f'Performing loss simulations for {buildings_to_process} representative buildings...')
        if buildings_to_process == 0:
            logger.info("No buildings to process")
            return
            
        tqdm.pandas()
        try:
            ReprBld_df.progress_apply(
                lambda row: _LossSim_1Bld(row, h5pyFile=f, IDA_result_dir = IDA_results_dir),
                axis=1
            )
            logger.info(f"Successfully completed loss simulations for all {buildings_to_process} buildings")
        except Exception as e:
            logger.error(f"Error during loss simulations: {str(e)}")
            raise

def GetProbDS(ReprPSPAID: int, IM: list[float], losstype: str) -> list[dict]:
    '''
    Get the probability of each damage state given any IM

    Parameters:
        ReprPSPAID (int). ID of the representative building for PSPA
        IM (list[float]). List of IMs
        losstype (str). Type of loss. Options: 'DS_Struct', 'DS_NonStruct_DriftSen', 'DS_NonStruct_AccelSen'

    Returns:
        DS_prob_All (list[dict]). List of dictionaries. Each dictionary contains the probability of each damage state. DS values include 'None', 'Slight', 'Moderate', 'Extensive', 'Complete'
    '''

    logger.debug(f"Getting damage state probabilities for ReprPSPAID: {ReprPSPAID}, losstype: {losstype}")
    
    # check losstype is valid
    if losstype not in LossTypes1:
        logger.error(f"Invalid losstype: {losstype}. Options: {', '.join(LossTypes1)}")
        raise ValueError(f"Invalid losstype: {losstype}. Options: {', '.join(LossTypes1)}")

    DS_prob_All = []
    with h5py.File(hdf5File, 'r') as f:
        grp_h5py = f[f'ReprBldID_{ReprPSPAID}']
        # load into memory
        grp = Utilities.load_h5py_group_to_memory(grp_h5py)
        IM_list = [float(grp_h5py[grp_im].attrs['IM']) for grp_im in grp_h5py]

        for im in IM:
            # Find the two closest indices in IM_list to the given IM
            idx = np.searchsorted(IM_list, im)
            if idx == 0:
                closest_indices = [0, 1]
            elif idx == len(IM_list):
                closest_indices = [len(IM_list) - 2, len(IM_list) - 1]
            else:
                closest_indices = [idx - 1, idx]
            # Calculate the probability of each damage state for the two closest IMs. Then interpolate the probabilities for the given im
            DS_prob = []
            for i in closest_indices:
                grp_im = grp[f'IM{i}']
                grp_im_ds = grp_im[losstype]
                DS_prob_im = {key: 0 for key in DS_types} # initialize the probabilities
                for DS_type in DS_types:
                    if DS_type in grp_im_ds:
                        DS_prob_im[DS_type] = grp_im_ds[DS_type][()]
                DS_prob.append(DS_prob_im)
            # Interpolate the probabilities
            DS_prob_interp = {key: 0 for key in DS_types}
            for DS_type in DS_types:
                DS_prob_interp[DS_type] = float(np.interp(im, [IM_list[closest_indices[0]], IM_list[closest_indices[1]]], [DS_prob[0][DS_type], DS_prob[1][DS_type]]))

            DS_prob_All.append(DS_prob_interp)

    return DS_prob_All
                
def GetCDFLoss(ReprPSPAID: int, IM: list[float], losstype: str, loss: list[float]) -> list[float]:
    '''
    Get the probability of being smaller than a loss value given any IM

    Parameters:
        ReprPSPAID (int). ID of the representative building for PSPA
        IM (list[float]). List of IMs
        losstype (str). Type of loss. Options: 'RepairCost_Total', 'RepairCost_Struct', 'RepairCost_NonStruct_DriftSen','RepairCost_NonStruct_AccelSen', 'RepairTime', 'RecoveryTime', 'FunctionLossTime'
        loss (list[float]). List of loss values corresponding to the IMs
    
    Returns:
        prob_loss (list[float]). List of probabilities of the loss value for each IM
    '''
    
    logger.debug(f"Getting CDF loss for ReprPSPAID: {ReprPSPAID}, losstype: {losstype}")
    
    prob_loss = []

    # number of loss values should be the same as the number of IMs
    if len(IM) != len(loss):
        logger.error("The number of loss values should be the same as the number of IMs")
        raise ValueError("The number of loss values should be the same as the number of IMs")

    # check losstype is valid
    if losstype not in LossTypes2:
        logger.error(f"Invalid losstype: {losstype}. Options: {', '.join(LossTypes2)}")
        raise ValueError(f"Invalid losstype: {losstype}. Options: {', '.join(LossTypes2)}")
    
    with h5py.File(hdf5File, 'r') as f:
        grp_h5py = f[f'ReprBldID_{ReprPSPAID}']
        # load into memory
        grp = Utilities.load_h5py_group_to_memory(grp_h5py)
        IM_list = [float(grp_h5py[grp_im].attrs['IM']) for grp_im in grp_h5py]
        for i_im, im in enumerate(IM):
            # Find the two closest indices in IM_list to the given IM
            idx = np.searchsorted(IM_list, im)
            if idx == 0:
                closest_indices = [0, 1]
            elif idx == len(IM_list):
                closest_indices = [len(IM_list) - 2, len(IM_list) - 1]
            else:
                closest_indices = [idx - 1, idx]

            # Calculate the probability of the loss value for the two closest IMs. 
            prob_loss_2im = []  # probabilities of the loss value for the two closest IMs
            for i in closest_indices:
                grp_im = grp[f'IM{i}']
                grp_im_loss = grp_im[losstype]
                # calculate CDF of the loss
                ecdf_values = grp_im_loss['quantiles'][()]
                quantiles = np.linspace(0, 1, 101)
                res = np.interp(loss[i_im], ecdf_values, quantiles)
                prob_loss_2im.append(res)

            # Interpolate the probabilities
            prob_loss_interp = float(np.interp(im, [IM_list[closest_indices[0]], IM_list[closest_indices[1]]], prob_loss_2im))
            prob_loss.append(prob_loss_interp)

    return prob_loss

def get_reconstruction_cost_and_time(ReprPSPAID: int | list[int] | None = None) -> dict[int, tuple[float, float]]:
    '''
    Get the reconstruction cost and time of a representative building for PSPA

    Parameters:
        ReprPSPAID (int or list or None). ID of the representative building for PSPA. If None, return for all buildings.
    Returns:
        results (dict). Dictionary with ReprPSPAID as key and (reconstruction_cost, reconstruction_time) as value.
    '''
    
    results = {}
    
    with h5py.File(hdf5File, 'r') as f:
        if ReprPSPAID is None:
            target_ids = []
            for key in f.keys():
                if key.startswith('ReprBldID_'):
                    try:
                        pid = int(key.split('_')[-1])
                        target_ids.append(pid)
                    except ValueError:
                        continue
        elif isinstance(ReprPSPAID, list):
            target_ids = ReprPSPAID
        else:
            target_ids = [ReprPSPAID]

        for pid in target_ids:
            cost = 0.0
            time = 0.0
            bld_key = f'ReprBldID_{pid}'
            if bld_key in f:
                grp_bld = f[bld_key]
                # Iterate over all IMs to find the maximum loss and time
                for grp_im_name in grp_bld:
                    grp_im = grp_bld[grp_im_name]
                    
                    if 'RepairCost_Total' in grp_im:
                        grp_loss = grp_im['RepairCost_Total']
                        if 'quantiles' in grp_loss:
                            # The last element of quantiles is the maximum value (100th percentile)
                            max_loss = grp_loss['quantiles'][-1]
                            if max_loss > cost:
                                cost = max_loss
                                
                    if 'RepairTime' in grp_im:
                        grp_time = grp_im['RepairTime']
                        if 'quantiles' in grp_time:
                            max_time = grp_time['quantiles'][-1]
                            if max_time > time:
                                time = max_time
            
            results[pid] = (float(cost), float(time))
            
    return results


def SimLossRandomly(ReprPSPAID: int, IM: list[float], losstype: str, Nsim: list[int]) -> list[list]:
    '''
    Simulate random loss values given any IM

    Parameters:
        ReprPSPAID (int). ID of the representative building for PSPA
        IM (list[float]). List of IMs
        losstype (str). Type of loss. Can be DS_types or LossTypes
        Nsim (list[int]). Number of simulations for each IM
    
    Returns:
        sim_loss (list[list]). List of simulated loss values for each IM. Each element is a list of Nsim values.
    '''
    
    logger.debug(f"Simulating random losses for ReprPSPAID: {ReprPSPAID}, losstype: {losstype}")
    
    sim_loss = []

    if losstype in LossTypes1:
        logger.debug(f"Processing damage state type: {losstype}")
        ds_prob = GetProbDS(ReprPSPAID, IM, losstype)
        for i, im in enumerate(IM):
            # 消除 float32 存储和插值产生的概率和舍入误差。
            probabilities = np.array([ds_prob[i][key] for key in DS_types], dtype=float)
            probabilities /= probabilities.sum()
            sim_loss_im = np.random.choice(DS_types, size=Nsim[i], p=probabilities).tolist()
            sim_loss.append(sim_loss_im)

    elif losstype in LossTypes2:
        logger.debug(f"Processing loss type: {losstype}")
        with h5py.File(hdf5File, 'r') as f:
            grp_h5py = f[f'ReprBldID_{ReprPSPAID}']
            grp = Utilities.load_h5py_group_to_memory(grp_h5py)
            IM_list = [float(grp_h5py[grp_im].attrs['IM']) for grp_im in grp_h5py]
            for i_im, im in enumerate(IM):
                # Find the two closest indices in IM_list to the given IM
                idx = np.searchsorted(IM_list, im)
                if idx == 0:
                    closest_indices = [0, 1]
                elif idx == len(IM_list):
                    closest_indices = [len(IM_list) - 2, len(IM_list) - 1]
                else:
                    closest_indices = [idx - 1, idx]
                # Simulate loss values for the two closest IMs. Then interpolate the loss values for the given im
                loss_2im = []
                for i in closest_indices:
                    grp_im = grp[f'IM{i}']
                    grp_im_loss = grp_im[losstype]
                    # calculate CDF of the loss
                    ecdf_values = grp_im_loss['quantiles'][()]
                    quantiles = np.linspace(0, 1, 101)
                    res = np.interp(np.random.rand(Nsim[i_im]), quantiles, ecdf_values)
                    loss_2im.append(res)
                # Interpolate the loss values for each simulation
                loss_interp = []
                for j in range(Nsim[i_im]):
                    loss_interp.append(float(np.interp(im, [IM_list[closest_indices[0]], IM_list[closest_indices[1]]], [loss_2im[0][j], loss_2im[1][j]])))
                sim_loss.append(loss_interp)
    else:  
        logger.error(f"Invalid losstype: {losstype}. Options: {', '.join(LossTypes1 + LossTypes2)}")
        raise ValueError(f"Invalid losstype: {losstype}. Options: {', '.join(LossTypes1 + LossTypes2)}")
    
    logger.debug(f"Completed simulation for {len(IM)} IM values")
    return sim_loss
