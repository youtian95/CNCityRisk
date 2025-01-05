# Perform probabilistic seismic loss assessment for individual representative buildings. Culmulative distribution functions (CDF) of the losses given any IM are provided. 

from . import ClassifyBld
from .PythonLib.MDOFModel import Tool_LossAssess
from . import IDA_ReprBld
from . import Utilities

from pathlib import Path
import pandas as pd
import shutil
from tqdm import tqdm
from scipy.stats import ecdf
import h5py
import numpy as np

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
    NumofStories = int(row['story'])
    FloorArea = float(row['area'])
    StructuralType = row['structure']
    DesignInfo = {}
    DesignInfo['SeismicDesignLevel'] = row['SeismicDesignLevel']
    DesignInfo['Code'] = 'CN'
    OccupancyClass = row['occupancy'].strip() if pd.notna(row['occupancy']) and row['occupancy'].strip() and row['occupancy'].strip().lower() != 'unknown' else 'RES3'

    IDA_result_file = IDA_result_dir / f'IDA_result_ReprBldID_{ID_repr_IDA}.csv'
    IDA_result = pd.read_csv(IDA_result_file)
    # read IM list
    unique_IMs = IDA_result['IM'].unique()
    IM_list = sorted(unique_IMs)

    N_Sim = [N_Sim] * len(IM_list)

    # Simulate losses for the building
    SimEDP, BldLoss = Tool_LossAssess.Simulate_losses_given_IM_basedon_IDA(
        IDA_result_file, IM_list, N_Sim, BetaM,
        OutputDir, NumofStories, FloorArea, StructuralType, DesignInfo, OccupancyClass
    )

    # save the results to the hdf5 file
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
                        grp_im_loss.create_dataset(DS, data=prob)
                else:
                    # save CDF of the loss
                    quantiles = np.linspace(0, 1, 101)
                    ecdf_values = np.quantile(BldLoss_IM, quantiles)
                    grp_im_loss.create_dataset('quantiles', data=ecdf_values)

    # move the result files to the output directory
    if OutputDir is not None:
        shutil.move(OutputDir / 'BldLoss.csv', OutputDir / f'BldLoss_ReprPSPAID_{ID_repr_PSPA}.csv')
        shutil.move(OutputDir / 'SimEDP.csv', OutputDir / f'SimEDP_ReprPSPAID_{ID_repr_PSPA}.csv')

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

    OutputDir.mkdir(parents=True, exist_ok=True)

    # Read representative building data for PSPA
    ReprBld_df = ClassifyBld.ReadReprBld(ReprBldPSPADir)
    ReprBld_df['ReprPSPAID'] = ReprBld_df.index

    # determine repr ID of IDA for each building in PSPA
    ReprBld_df_IDA = ClassifyBld.ReadReprBld(ReprBldDir)
    ReprBld_df_IDA['ReprBldID'] = ReprBld_df_IDA.index
    common_columns = list(set(ReprBld_df.columns) & set(ReprBld_df_IDA.columns))
    num_rows_before_merge = len(ReprBld_df)
    ReprBld_df = ReprBld_df.merge(ReprBld_df_IDA, on=common_columns, how='left')

    # Check if the number of rows in ReprBld_df remains the same before and after merging
    if len(ReprBld_df) != num_rows_before_merge:
        raise ValueError("The number of rows in ReprBld_df has changed after merging.")

    with h5py.File(hdf5File, "a") as f:
        if not StartOver:
            # skip buildings that have already been simulated
            ReprBldIDs = f.keys()
            ReprBld_df = ReprBld_df[~ReprBld_df['ReprPSPAID'].astype(str).isin(ReprBldIDs)]
        else:
            # remove the repr buildings in hdf5File that are included in ReprBldPSPADir, so that the loss simulations can be re-run
            for repr_bld_id in ReprBld_df['ReprPSPAID'].astype(str):
                if repr_bld_id in f:
                    del f[f'ReprBldID_{repr_bld_id}']

        # Perform loss simulations for each representative building
        print('Performing loss simulations for each representative building...')
        tqdm.pandas()
        ReprBld_df.progress_apply(
            lambda row: _LossSim_1Bld(row, h5pyFile=f, IDA_result_dir = IDA_results_dir),
            axis=1
        )

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

    # check losstype is valid
    if losstype not in LossTypes1:
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
    prob_loss = []

    # number of loss values should be the same as the number of IMs
    if len(IM) != len(loss):
        raise ValueError("The number of loss values should be the same as the number of IMs")

    # check losstype is valid
    if losstype not in LossTypes2:
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
    
    sim_loss = []

    if losstype in LossTypes1:
        ds_prob = GetProbDS(ReprPSPAID, IM, losstype)
        for i, im in enumerate(IM):
            sim_loss_im = np.random.choice(DS_types, size=Nsim[i], p=[ds_prob[i][key] for key in DS_types]).tolist()
            sim_loss.append(sim_loss_im)

    elif losstype in LossTypes2:
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
        raise ValueError(f"Invalid losstype: {losstype}. Options: {', '.join(LossTypes1 + LossTypes2)}")
    
    return sim_loss
