
from . import Utilities
from . import BldPSPA_ReprBld
from . import EQsources
from . import IMSim

from pathlib import Path
import pandas as pd
from tqdm import tqdm
import geopandas as gpd
import numpy as np
import pandas as pd
import h5py
import geopandas as gpd
import dask.array as da
from typing import Union
import matplotlib.pyplot as plt
import py7zr

RESULTDIR = Path(__file__).resolve().parent / 'results/RegionalLossSim/'
MAP_ZIP = RESULTDIR / 'maps.7z'

def simulate_regional_losses(IM_sim: Union[np.ndarray, da.Array], Bld_gdf: gpd.GeoDataFrame, losstype: str, OutputDir: Path = RESULTDIR, ChunkSize = 100000) -> None:
    '''
    Simulate regional seismic losses based on probabilistic seismic performance assessment (PSPA) results.

    Parameters:
        IM_sim (np.ndarray | da.Array): Array containing IM simulation results.
        Bld_gdf (gpd.GeoDataFrame): GeoDataFrame containing building data.
        losstype (str): Type of loss to simulate. Options are 'DS_Struct', 'DS_NonStruct_DriftSen', 'DS_NonStruct_AccelSen', 'RepairCost_Total', 'RepairCost_Struct', 'RepairCost_NonStruct_DriftSen','RepairCost_NonStruct_AccelSen', 'RepairTime', 'RecoveryTime', 'FunctionLossTime'
        OutputDir (Path): Directory to save the output files. Default is 'results/RegionalLossSim/'.
        ChunkSize (int): Number of buildings to process in each chunk. Default is 10000.
    
    Results:
        BldLoss_{losstype}.hdf5: HDF5 file containing the simulated regional losses.
            BldID_i: Dataset containing the simulated loss values for building i. Each element is a list of Nsim values.
    '''
    # check if the losstype is valid
    if losstype not in BldPSPA_ReprBld.LossTypes1 + BldPSPA_ReprBld.LossTypes2:
        raise ValueError(f"losstype must be one of {BldPSPA_ReprBld.LossTypes1 + BldPSPA_ReprBld.LossTypes2}.")
    
    # Get the full path of the OutputDir
    OutputDir = Path(OutputDir).resolve()
    if not OutputDir.exists():
        OutputDir.mkdir(parents=True)

    # Read the table
    T = Bld_gdf

    # check if the row number of IM_sim is equal to the row number of T
    if len(IM_sim) != len(T):
        raise ValueError("The number of rows in IM_sim must be equal to the number of rows in BldsFile.")
    
    # Progress bar
    with tqdm(total=len(T), desc="Simulating regional seismic losses...") as pbar:
        with h5py.File(OutputDir / f'BldLoss_{losstype}.hdf5', 'w') as f:

            # Split T into chunks based on ChunkSize
            for chunk_start in range(0, len(T), ChunkSize):
                chunk_end = min(chunk_start + ChunkSize, len(T))
                T_chunk = T.iloc[chunk_start:chunk_end]

                # find unique ReprBldID_PSPA in the chunk
                unique_ReprBldID_PSPA = T_chunk['ReprBldID_PSPA'].unique()
                # iterate based on the ReprBldID_PSPA
                for ReprPSPAID in unique_ReprBldID_PSPA:
                    # find the index of the buildings with the same ReprBldID_PSPA.
                    idx = T_chunk[T_chunk['ReprBldID_PSPA'] == ReprPSPAID].index
                    N_same_ReprBldID_PSPA = len(idx)
                    N_sim_1_bld = IM_sim.shape[1]
                    # get the IM list
                    if isinstance(IM_sim, np.ndarray):
                        IMlist = IM_sim[idx, :].tolist()
                    elif isinstance(IM_sim, da.Array):
                        IMlist = IM_sim[idx, :].compute().tolist()
                    # convert IMlist (N_same_ReprBldID_PSPA x N_sim_1_bld) to one row
                    IMlist = [item for sublist in IMlist for item in sublist]
                    # sim_loss: List of simulated loss values for each IM. Each element is a list of Nsim values.
                    sim_loss = BldPSPA_ReprBld.SimLossRandomly(
                        ReprPSPAID=ReprPSPAID, IM=IMlist, losstype=losstype, Nsim=[1]*len(IMlist)
                    )
                    # since each element in sim_loss has only one value, flatten it
                    sim_loss = list(np.array(sim_loss).flatten())
                    # sim_loss has only one row. convert it to N_same_ReprBldID_PSPA x N_sim_1_bld
                    sim_loss = [sim_loss[i*N_sim_1_bld:(i+1)*N_sim_1_bld] for i in range(N_same_ReprBldID_PSPA)]

                    # save the results
                    for i, idx_i in enumerate(idx):
                        if losstype in BldPSPA_ReprBld.LossTypes1:
                            # Convert to categorical type to save space
                            sim_loss_cat = pd.Categorical(np.array(sim_loss[i]), categories=BldPSPA_ReprBld.DS_types)
                            f.create_dataset(f'BldID_{idx_i}', data=sim_loss_cat.codes, dtype='int8')
                        else:
                            f.create_dataset(f'BldID_{idx_i}', data=sim_loss[i], dtype='float32')
                    pbar.update(len(idx))



def get_statistics_regional_losses(CityName: str, ResultsDir: str = RESULTDIR, LossType: str = 'DS_Struct', i_rup: int = 0, iSim: int = None, LossDataInMemory = None):
    '''
    Get the statistics of the regional seismic losses, given an earthquake event.

    Parameters:
        CityName (str): Name of the city.
        ResultsDir (str): Directory containing the loss results.
        LossType (str): Type of loss to plot. Default is 'DS_Struct' (structural damage state).
        i_rup (int): Index of the rupture. Default is 0.
        iSim (int): Index of the simulation for a specific rupture. Default is None.
        LossDataInMemory: 预加载的损失数据，如果提供则直接使用，否则从文件加载

    Returns:
        total_loss_sim (np.ndarray): Cumulative distribution function of the regional seismic losses. 
            (N_DS x N_sim)
                if LossType is 'DS_Struct' ..., N_DS = 5 (damage states). Each element is the numbers of the building to be in the corresponding damage state.
                if LossType is 'RepairCost_Total' ..., N_DS = 1. Each element is a total repair cost.
        total_loss_iSim (np.ndarray): regional total losses for the iSim-th simulation. (N_DS)
        percentiles_iSim (np.ndarray): Percentiles of the iSim-th simulation.
            (N_DS)
                if LossType is 'DS_Struct' ..., N_DS = 5 (damage states). 
                if LossType is 'RepairCost_Total' ..., N_DS = 1.
    '''
    print(f'Calculating statistics of regional losses...')

    Bld_gdf = Utilities.get_CityBldData(CityName)
    N_bld = len(Bld_gdf)

    ResultsDir = Path(ResultsDir).resolve()

    # rupid 2 i_sim
    rupid_2_iSim = pd.read_csv(IMSim.IM_SIM_DIR / CityName / IMSim.RUPID_2_ISIM_FILENAME, index_col=0, header=0, dtype={0: str})
    rupid_2_iSim['iSim'] = rupid_2_iSim['iSim'].apply(lambda x: list(map(int, x.strip('[]').split())) if isinstance(x, str) else x)
    rupid_2_iSim = rupid_2_iSim.to_dict(orient='index')

    # read the rupture dataframe
    rup_df = EQsources.ReadCityRuptures(CityName)

    # get the rupture id
    rupid = rup_df.iloc[i_rup]['rup_id']

    # get the iSim index in all ruptures
    iSim_list = rupid_2_iSim[rupid]['iSim']
    iSim_allRup = iSim_list[iSim]
    N_sim = len(iSim_list) # simulations for the specific rupture

    if LossType in BldPSPA_ReprBld.LossTypes1:
        N_DS = len(BldPSPA_ReprBld.DS_types)
    elif LossType in BldPSPA_ReprBld.LossTypes2:
        N_DS = 1
    else:
        raise ValueError(f"LossType must be one of {BldPSPA_ReprBld.LossTypes1 + BldPSPA_ReprBld.LossTypes2}.")

    total_loss_sim = np.zeros((N_DS, N_sim))
    if iSim is not None:
        total_loss_iSim = np.zeros(N_DS)
    else:
        total_loss_iSim = None

    # calculate the numbers of buildings in each damage state or the total repair cost
    if LossDataInMemory is not None:
        # 使用预加载的数据
        f_in_memory = LossDataInMemory
    else:
        # 如果没有提供预加载数据，则从文件加载
        with h5py.File(ResultsDir / CityName / f'BldLoss_{LossType}.hdf5', 'r') as f:
            f_in_memory = Utilities.load_h5py_group_to_memory(f)
            
    for i in range(N_bld):
        loss_sim = f_in_memory[f'BldID_{i}'][:]
        # i-th simulation for the specific rupture
        if iSim is not None:
            loss = loss_sim[iSim_allRup]
            if LossType in BldPSPA_ReprBld.LossTypes1:
                total_loss_iSim[loss] += 1
            elif LossType in BldPSPA_ReprBld.LossTypes2:
                total_loss_iSim[0] += loss
        # total losses for each simulation
        for j in range(N_sim):
            loss = loss_sim[iSim_list[j]]
            if LossType in BldPSPA_ReprBld.LossTypes1:
                total_loss_sim[loss, j] += 1
            elif LossType in BldPSPA_ReprBld.LossTypes2:
                total_loss_sim[0, j] += loss
    
    # check sum(total_loss_iSim) = N
    if LossType in BldPSPA_ReprBld.LossTypes1:
        if not np.isclose(np.sum(total_loss_iSim), N_bld):
            raise ValueError("Sum of total_loss_iSim must be equal to the number of buildings.")

    # calculate the percentiles of the iSim-th simulation
    if iSim is not None:
        percentiles_iSim = np.zeros(N_DS)
        for ds in range(N_DS):
            sorted_losses = np.sort(total_loss_sim[ds, :])
            percentiles_iSim[ds] = np.interp(total_loss_iSim[ds], sorted_losses, np.linspace(0, 1, N_sim))
    else:
        percentiles_iSim = None
            
    return total_loss_sim, total_loss_iSim, percentiles_iSim


def plot_CDF_regional_losses(CityName: str, ResultsDir: str = RESULTDIR, LossType: str = 'DS_Struct', i_rup: int = 0, iSim: int = None, SavePath: Path = None, PlotNow: bool = True, Overwrite: bool = False, LossDataInMemory = None):
    '''
    plot the cumulative distribution function of total regional seismic losses.
    '''
    
    if SavePath is None:
        SavePath = ResultsDir / f'CDF_{CityName}_{LossType}_{i_rup}_{iSim}.jpg'

    # Check if the file already exists in the archive
    if MAP_ZIP.exists():
        with py7zr.SevenZipFile(MAP_ZIP, 'r') as archive:
            existing_files = archive.getnames()
            if SavePath.name in existing_files:
                if not Overwrite:
                    print(f'{SavePath.name} already exists in the archive. Skipping...')                    
                    return None

    
    total_loss_sim, total_loss_iSim, percentiles_iSim = get_statistics_regional_losses(CityName, ResultsDir, LossType, i_rup, iSim, LossDataInMemory)

    # print the percentiles
    print(f'Percentiles of the iSim-th simulation: {percentiles_iSim}')

    # Plot the CDF
    fig, ax = plt.subplots()
    if LossType in BldPSPA_ReprBld.LossTypes1:
        for ds, ds_name in enumerate(BldPSPA_ReprBld.DS_types):
            sorted_losses = np.sort(total_loss_sim[ds, :])
            cdf = np.linspace(0, 1, len(sorted_losses))
            ax.plot(sorted_losses, cdf, label=ds_name)
        ax.set_xlabel('Number of Buildings')
        ax.set_ylabel('Cumulative Probability')
        ax.set_title(f'Cumulative Distribution Function of Damage state {LossType}')
        ax.legend()
    elif LossType in BldPSPA_ReprBld.LossTypes2:
        sorted_losses = np.sort(total_loss_sim[0, :])
        cdf = np.linspace(0, 1, len(sorted_losses))
        ax.plot(sorted_losses, cdf, label=LossType)
        ax.set_xlabel('Total Loss')
        ax.set_ylabel('Cumulative Probability')
        ax.set_title(f'Cumulative Distribution Function of {LossType}')
        ax.legend()

    # Highlight the iSim-th simulation if provided
    if iSim is not None:
        if LossType in BldPSPA_ReprBld.LossTypes1:
            for ds in range(len(BldPSPA_ReprBld.DS_types)):
                ax.scatter(total_loss_iSim[ds], percentiles_iSim[ds], color='r', zorder=5)
        elif LossType in BldPSPA_ReprBld.LossTypes2:
            ax.scatter(total_loss_iSim[0], percentiles_iSim[0], color='r', zorder=5)
        ax.scatter([], [], color='r', label='The displayed simulation')
        ax.legend()

    fig.savefig(SavePath)    # Compress the map file into MAP_ZIP
    if not MAP_ZIP.exists():
        with py7zr.SevenZipFile(MAP_ZIP, 'w', filters=[{'id': py7zr.FILTER_LZMA2, 'preset': 9}]) as archive:
            archive.write(SavePath, arcname=SavePath.name)
    else:
        with py7zr.SevenZipFile(MAP_ZIP, 'a', filters=[{'id': py7zr.FILTER_LZMA2, 'preset': 9}]) as archive:
            archive.write(SavePath, arcname=SavePath.name)

    # delete the html file
    SavePath.unlink()
    
    if PlotNow:
        plt.show()
    else:
        plt.close('all')

def create_geojson_regional_losses(CityName, DistrictName=None, ResultsDir=RESULTDIR, LossType='DS_Struct', i_rup=0, iSim=0):
    """生成区域损失的GeoJSON数据"""
    """
    Parameters:
        CityName (str): 城市名称
        DistrictName (str): 区域名称
        ResultsDir (str): 结果目录
        LossType (str): 损失类型
        i_rup (int or None): 震源索引，如果为None则处理所有震源
        iSim (int): 模拟索引, 从0开始
    Returns:
        geojson_data (str): GeoJSON格式的字符串数据，包含以下内容：
            - geometry: 建筑物的几何形状信息（点、多边形等）
            - loss_value: 指定模拟场景下建筑物的损失值
                如果i_rup不为None: 单个数值（单个rupture的损失值）
                如果i_rup为None: 数组（所有rupture的损失值列表）
            - structure: 建筑物结构类型
            - story: 建筑物层数
            - DistrictName: 建筑物所属区域名称
    """
    # 读取建筑数据
    if DistrictName is not None:
        Bld_gdf = Utilities.get_DistrictBldData(CityName, DistrictName=DistrictName)
    else:
        Bld_gdf = Utilities.get_CityBldData(CityName)
    
    # 获取震源和模拟索引
    rupid_2_iSim = pd.read_csv(IMSim.IM_SIM_DIR / CityName / IMSim.RUPID_2_ISIM_FILENAME, index_col=0, header=0, dtype={0: str})
    rupid_2_iSim['iSim'] = rupid_2_iSim['iSim'].apply(
        lambda x: list(map(int, x.strip('[]').split())) if isinstance(x, str) else x)
    rupid_2_iSim = rupid_2_iSim.to_dict(orient='index')
    
    # 读取震源信息
    rup_df = EQsources.ReadCityRuptures(CityName)
    
    # 创建结果GeoDataFrame
    result_gdf = Bld_gdf.copy()
    
    # 添加损失数据
    with h5py.File(ResultsDir / CityName / f'BldLoss_{LossType}.hdf5', 'r') as f:
        # 先将数据加载到内存一次，避免在循环中重复访问
        f_in_memory = Utilities.load_h5py_group_to_memory(f)
        
        if i_rup is None:
            # 处理所有rupture
            loss_values_all_ruptures = []
            
            for bld_idx in result_gdf.index:
                loss_sim = f_in_memory[f'BldID_{bld_idx}'][:]
                bld_losses = []
                
                # 遍历所有rupture
                for rup_idx in range(len(rup_df)):
                    rupid = rup_df.iloc[rup_idx]['rup_id']
                    iSim_list = rupid_2_iSim[rupid]['iSim']
                    iSim_allRup = iSim_list[iSim]
                    bld_losses.append(int(loss_sim[iSim_allRup]))
                
                loss_values_all_ruptures.append(bld_losses)
            
            result_gdf['loss_value'] = loss_values_all_ruptures
        else:
            # 处理单个rupture（原来的逻辑）
            rupid = rup_df.iloc[i_rup]['rup_id']
            iSim_list = rupid_2_iSim[rupid]['iSim']
            iSim_allRup = iSim_list[iSim]
            
            loss_values = []
            for bld_idx in result_gdf.index:
                loss_sim = f_in_memory[f'BldID_{bld_idx}'][:]
                loss_values.append(int(loss_sim[iSim_allRup]))
            
            result_gdf['loss_value'] = loss_values
    
    # 只保留必要的列
    keep_columns = ['geometry', 'loss_value', 'structure', 'story', 'DistrictName']
    result_gdf = result_gdf[keep_columns]
    
    # 转换为GeoJSON
    geojson_data = result_gdf.to_json()
    
    return geojson_data

