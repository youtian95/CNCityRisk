from . import Utilities
from . import BldPSPA_ReprBld
from . import EQsources
from . import IMSim
from . import Logger

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
import json
import multiprocessing as mp



RESULTDIR = Path(__file__).resolve().parent / 'results/RegionalLossSim/'
MAP_ZIP = RESULTDIR / 'maps.7z'

# Create module logger
logger = Logger.get_logger('RegionalLossSim')

def _calculate_rupture_loss(args):
    """计算单个震源的损失分布"""
    i_rup_single, rupid_2_iSim, rup_df, N_DS, N_bld, loss_data, LossType_inner = args
    
    rupid = rup_df.iloc[i_rup_single]['rup_id']
    iSim_single = rupid_2_iSim[rupid]['iSim']
    
    N_sim_single = len(iSim_single)
    total_loss_single = np.zeros((N_DS, N_sim_single))
    
    # 计算当前震源的损失分布
    # loss_data 可能是普通字典(单进程)或Manager.dict(多进程共享)
    for i in range(N_bld):
        loss_sim = loss_data[f'BldID_{i}']
        for j in range(N_sim_single):
            loss = loss_sim[iSim_single[j]]
            if LossType_inner in BldPSPA_ReprBld.LossTypes1:
                total_loss_single[int(loss), j] += 1
            elif LossType_inner in BldPSPA_ReprBld.LossTypes2:
                total_loss_single[0, j] += loss
    
    return i_rup_single, total_loss_single

def simulate_regional_losses_single_process(CityName: str, Bld_gdf: gpd.GeoDataFrame, losstype: str, OutputDir: Path = RESULTDIR, ChunkSize = 10000) -> None:
    '''
    单进程版本：基于概率性地震性能评估(PSPA)结果模拟区域地震损失。

    Parameters:
        CityName (str): 城市名称，用于获取IM模拟数据
        Bld_gdf (gpd.GeoDataFrame): 包含建筑数据的GeoDataFrame
        losstype (str): 要模拟的损失类型。选项包括 'DS_Struct', 'DS_NonStruct_DriftSen', 'DS_NonStruct_AccelSen', 'RepairCost_Total', 'RepairCost_Struct', 'RepairCost_NonStruct_DriftSen','RepairCost_NonStruct_AccelSen', 'RepairTime', 'RecoveryTime', 'FunctionLossTime'
        OutputDir (Path): 保存输出文件的目录。默认为 'results/RegionalLossSim/'
        ChunkSize (int): 每个chunk中要处理的建筑数量。默认为10000
    
    Results:
        BldLoss_{losstype}.hdf5: 包含模拟区域损失的HDF5文件
            BldID_i: 包含建筑i的模拟损失值的数据集。每个元素是Nsim个值的列表
    '''
    logger.info(f"Starting single-process regional loss simulation for {len(Bld_gdf)} buildings with loss type: {losstype}")
    
    # 获取IM模拟数据
    logger.info(f"Loading IM simulation data for city: {CityName}")
    _, IM_sim = IMSim.get_all_IM_mat_Bld(CityName)
    
    # check if the losstype is valid
    if losstype not in BldPSPA_ReprBld.LossTypes1 + BldPSPA_ReprBld.LossTypes2:
        error_msg = f"losstype must be one of {BldPSPA_ReprBld.LossTypes1 + BldPSPA_ReprBld.LossTypes2}."
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Get the full path of the OutputDir
    OutputDir = Path(OutputDir).resolve()
    if not OutputDir.exists():
        OutputDir.mkdir(parents=True)
        logger.info(f"Created output directory: {OutputDir}")

    # Read the table
    T = Bld_gdf

    # check if the row number of IM_sim is equal to the row number of T
    if len(IM_sim) != len(T):
        error_msg = "The number of rows in IM_sim must be equal to the number of rows in BldsFile."
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.info(f"Processing {len(T)} buildings in chunks of {ChunkSize}")
    
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
                            f.create_dataset(f'BldID_{idx_i}', data=sim_loss[i], dtype='float16')
                    pbar.update(len(idx))
    
    output_file = OutputDir / f'BldLoss_{losstype}.hdf5'
    logger.info(f"Single-process regional loss simulation completed. Results saved to: {output_file}")

def load_loss_data_to_memory(CityName: str, ResultsDir: str = RESULTDIR, LossType: str = 'DS_Struct') -> dict:
    """
    Load loss data from an HDF5 file into memory.
    
    Parameters:
        file_path (Path): Path to the HDF5 file containing loss data.
        
    Returns:
        dict: A dictionary where keys are building IDs and values are lists of simulated losses.
    """
    file_path = ResultsDir / CityName / f'BldLoss_{LossType}.hdf5'
    with h5py.File(file_path, 'r') as f:
        loss_data = Utilities.load_h5py_group_to_memory(f)
    return loss_data

def get_total_loss_sim(CityName: str, ResultsDir: str = RESULTDIR, LossType: str = 'DS_Struct', i_rup: Union[int, list] = 0, LossDataInMemory = None) -> np.ndarray:
    '''
    计算所有模拟的区域总损失分布（优化版本）
    
    Parameters:
        CityName (str): 城市名称
        ResultsDir (str): 结果目录
        LossType (str): 损失类型
        i_rup (Union[int, list]): 震源索引，可以是单个整数或整数列表
        LossDataInMemory: 预加载的损失数据，如果提供则直接使用，否则从文件加载
        
    Returns:
        total_loss_sim (np.ndarray): 所有模拟的损失分布
            - 如果i_rup是int: (N_DS x N_sim) 
            - 如果i_rup是list: (N_DS x N_sim_per_rup x N_rup)
            其中 N_DS 可以通过 total_loss_sim.shape[0] 获取
    '''
    # 标准化 i_rup 为列表并记录日志
    is_single_rupture = isinstance(i_rup, int)
    if is_single_rupture:
        i_rup_list = [i_rup]
        logger.info(f'Calculating total loss distribution for {CityName}, rupture {i_rup}, loss type: {LossType}')
    else:
        i_rup_list = i_rup
        logger.info(f'Calculating total loss distribution for {CityName}, ruptures {i_rup}, loss type: {LossType}')
    
    Bld_gdf = Utilities.get_CityBldData(CityName)
    N_bld = len(Bld_gdf)
    
    ResultsDir = Path(ResultsDir).resolve()
    
    # 获取震源映射信息
    rupid_2_iSim = IMSim.get_rupid_2_iSim(CityName)
    rup_df = EQsources.ReadCityRuptures(CityName)
    
    # 确定损失状态数量
    if LossType in BldPSPA_ReprBld.LossTypes1:
        N_DS = len(BldPSPA_ReprBld.DS_types)
    elif LossType in BldPSPA_ReprBld.LossTypes2:
        N_DS = 1
    else:
        raise ValueError(f"LossType must be one of {BldPSPA_ReprBld.LossTypes1 + BldPSPA_ReprBld.LossTypes2}.")
    
    # 加载损失数据
    if LossDataInMemory is not None:
        f_in_memory = LossDataInMemory
    else:
        f_in_memory = load_loss_data_to_memory(CityName, ResultsDir, LossType)

    # 计算每个震源的损失分布
    # 使用多进程池计算不同震源
    if len(i_rup_list) > 1:
        logger.info(f'Using multiprocessing for {len(i_rup_list)} ruptures')
        # 限制进程数以避免过度消耗资源，同时考虑任务数量
        n_processes = min(mp.cpu_count(), len(i_rup_list), 4)  # 最多4个进程
        
        # 创建共享字典来避免内存重复占用
        # 每个进程将共享同一份损失数据，而不是复制到每个进程
        manager = mp.Manager()
        shared_loss_data = manager.dict(f_in_memory)
        logger.info(f'Created shared loss data dictionary with {len(shared_loss_data)} building datasets')
        
        # 准备多进程任务
        rupture_tasks = []
        for i_rup_single in i_rup_list:
            rupture_tasks.append((i_rup_single, rupid_2_iSim, rup_df, N_DS, N_bld, shared_loss_data, LossType))
        
        with mp.Pool(processes=n_processes) as pool:
            # 使用imap显示进度
            rupture_results_with_idx = []
            with tqdm(total=len(rupture_tasks), desc=f"Calculating losses for {len(rupture_tasks)} ruptures", unit="rupture") as pbar:
                for result in pool.imap(_calculate_rupture_loss, rupture_tasks):
                    rupture_results_with_idx.append(result)
                    pbar.update(1)
        
        # 按震源索引排序结果
        rupture_results_with_idx.sort(key=lambda x: x[0])
        rupture_results = [result for _, result in rupture_results_with_idx]
    else:
        # 单个震源时直接计算，避免多进程开销
        logger.info('Single rupture, using direct calculation')
        # 为单震源情况使用相同的数据结构，确保一致性
        rupture_tasks = [(i_rup_list[0], rupid_2_iSim, rup_df, N_DS, N_bld, f_in_memory, LossType)]
        _, result = _calculate_rupture_loss(rupture_tasks[0])
        rupture_results = [result]
    
    # 根据输入类型组织返回结果
    if is_single_rupture:
        # 单个震源：返回 (N_DS x N_sim)
        total_loss_sim = rupture_results[0]
    else:
        # 多个震源：返回 (N_DS x N_sim_per_rup x N_rup)
        # 检查所有震源的模拟数量是否相同
        sim_counts = [result.shape[1] for result in rupture_results]
        if len(set(sim_counts)) > 1:
            raise ValueError(f"All ruptures must have the same number of simulations. Found simulation counts: {sim_counts} for ruptures: {i_rup_list}")
        
        N_sim_per_rup = sim_counts[0]
        N_rup = len(i_rup_list)
        
        total_loss_sim = np.zeros((N_DS, N_sim_per_rup, N_rup))
        
        for rup_idx, result in enumerate(rupture_results):
            total_loss_sim[:, :, rup_idx] = result
    
    return total_loss_sim

def get_statistics_regional_losses(CityName: str, ResultsDir: str = RESULTDIR, LossType: str = 'DS_Struct', i_rup: Union[int, list] = 0, iSim: int = None, LossDataInMemory = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Get the statistics of the regional seismic losses, given an earthquake event.
    优化版本：先计算总分布，然后通过索引获取特定模拟结果

    Parameters:
        CityName (str): Name of the city.
        ResultsDir (str): Directory containing the loss results.
        LossType (str): Type of loss to plot. Default is 'DS_Struct' (structural damage state).
        i_rup (Union[int, list]): Index of the rupture or list of rupture indices. Default is 0.
        iSim (int): Index of the simulation for a specific rupture. Default is None.
        LossDataInMemory: 预加载的损失数据，如果提供则直接使用，否则从文件加载

    Returns:
        total_loss_sim (np.ndarray): Cumulative distribution function of the regional seismic losses.
            - 如果i_rup是int: (N_DS x N_sim)
            - 如果i_rup是list: (N_DS x N_sim_per_rup x N_rup)
                if LossType is 'DS_Struct' ..., N_DS = 5 (damage states). Each element is the numbers of the building to be in the corresponding damage state.
                if LossType is 'RepairCost_Total' ..., N_DS = 1. Each element is a total repair cost.
        total_loss_iSim (np.ndarray): regional total losses for the iSim-th simulation.
            - 如果i_rup是int: (N_DS,)
            - 如果i_rup是list: (N_DS, N_rup)        
        percentiles_iSim (np.ndarray): Percentiles of the iSim-th simulation.
            - 如果i_rup是int: (N_DS,)
            - 如果i_rup是list: (N_DS, N_rup)
                if LossType is 'DS_Struct' ..., N_DS = 5 (damage states). 
                if LossType is 'RepairCost_Total' ..., N_DS = 1.
    '''
    # 第一步：计算所有模拟的损失分布
    total_loss_sim = get_total_loss_sim(
        CityName, ResultsDir, LossType, i_rup, LossDataInMemory
    )
    
    # 从total_loss_sim的形状确定损失状态数量
    N_DS = total_loss_sim.shape[0]
    
    # 第二步：如果需要特定模拟的结果，直接从总分布中索引获取
    if iSim is not None:
        # 根据i_rup类型处理不同的数据结构
        if isinstance(i_rup, int):
            # 单个震源：total_loss_sim是 (N_DS x N_sim)
            total_loss_iSim = total_loss_sim[:, iSim]
        else:
            # 多个震源：total_loss_sim是 (N_DS x N_sim_per_rup x N_rup)
            # 对于多震源情况，我们需要为每个震源计算iSim对应的损失
            N_rup = len(i_rup)
            total_loss_iSim = np.zeros((N_DS, N_rup))
            
            # 获取每个震源的模拟索引信息
            rupid_2_iSim = IMSim.get_rupid_2_iSim(CityName)
            rup_df = EQsources.ReadCityRuptures(CityName)
            
            for rup_idx, i_rup_single in enumerate(i_rup):
                rupid = rup_df.iloc[i_rup_single]['rup_id']
                iSim_list = rupid_2_iSim[rupid]['iSim']
                
                # 确保iSim不超出该震源的模拟数量
                current_iSim = iSim if iSim < len(iSim_list) else 0
                total_loss_iSim[:, rup_idx] = total_loss_sim[:, current_iSim, rup_idx]
        
        # 验证损伤状态类型的总和
        if LossType in BldPSPA_ReprBld.LossTypes1:
            N_bld = len(Utilities.get_CityBldData(CityName))
            if isinstance(i_rup, int):
                if not np.isclose(np.sum(total_loss_iSim), N_bld):
                    raise ValueError("Sum of total_loss_iSim must be equal to the number of buildings.")
            else:
                # 对于多震源，检查每个震源的总和
                for rup_idx in range(len(i_rup)):
                    if not np.isclose(np.sum(total_loss_iSim[:, rup_idx]), N_bld):
                        raise ValueError(f"Sum of total_loss_iSim for rupture {rup_idx} must be equal to the number of buildings.")
        
        # 计算百分位数
        if isinstance(i_rup, int):
            # 单个震源的百分位数计算
            percentiles_iSim = np.zeros(N_DS)
            N_sim = total_loss_sim.shape[1]
            for ds in range(N_DS):
                sorted_losses = np.sort(total_loss_sim[ds, :])
                percentiles_iSim[ds] = np.interp(total_loss_iSim[ds], sorted_losses, np.linspace(0, 1, N_sim))
        else:
            # 多个震源的百分位数计算
            N_rup = len(i_rup)
            percentiles_iSim = np.zeros((N_DS, N_rup))
            for rup_idx in range(N_rup):
                N_sim_this_rup = np.count_nonzero(total_loss_sim[0, :, rup_idx] != 0)  # 统计有效模拟数
                if N_sim_this_rup == 0:  # 如果没有有效模拟，使用所有可能的模拟
                    N_sim_this_rup = total_loss_sim.shape[1]
                
                for ds in range(N_DS):
                    valid_losses = total_loss_sim[ds, :N_sim_this_rup, rup_idx]
                    sorted_losses = np.sort(valid_losses)
                    percentiles_iSim[ds, rup_idx] = np.interp(
                        total_loss_iSim[ds, rup_idx], 
                        sorted_losses, 
                        np.linspace(0, 1, N_sim_this_rup)
                    )
    else:
        total_loss_iSim = None
        percentiles_iSim = None
            
    return total_loss_sim, total_loss_iSim, percentiles_iSim

# deprecated
def plot_CDF_regional_losses(CityName: str, ResultsDir: str = RESULTDIR, LossType: str = 'DS_Struct', i_rup: int = 0, iSim: int = None, SavePath: Path = None, PlotNow: bool = True, Overwrite: bool = False, LossDataInMemory = None):
    '''
    plot the cumulative distribution function of total regional seismic losses.
    '''
    logger.info(f'Plotting CDF for regional losses: {CityName}, rupture {i_rup}, loss type: {LossType}')
    
    if SavePath is None:
        SavePath = ResultsDir / f'CDF_{CityName}_{LossType}_{i_rup}_{iSim}.jpg'    
    # Check if the file already exists in the archive
    if MAP_ZIP.exists():
        with py7zr.SevenZipFile(MAP_ZIP, 'r') as archive:
            existing_files = archive.getnames()
            if SavePath.name in existing_files:
                if not Overwrite:
                    logger.info(f'CDF plot {SavePath.name} already exists in archive, skipping')
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
    logger.info(f'Creating GeoJSON for regional losses: {CityName}, district: {DistrictName}, loss type: {LossType}')
    
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


def create_statistics_regional_losses(CityName: str, ResultsDir: str = RESULTDIR, OutputDir: str = None, LossType: str = 'DS_Struct', iSim: int = 0) -> str:
    '''
    生成某城市所有地震事件的地震损失统计信息JSON文件
    
    该函数遍历指定城市的所有地震事件，调用get_statistics_regional_losses函数获取每个地震事件的
    损失统计信息，并将结果保存为结构化的JSON文件。

    Parameters:
        CityName (str): 城市名称
        ResultsDir (str): 读取损失数据的目录路径，默认为RESULTDIR
        OutputDir (str): 保存统计结果JSON文件的目录路径，如果为None则使用ResultsDir
        LossType (str): 损失类型，默认为'DS_Struct'
            - 'DS_Struct', 'DS_NonStruct_DriftSen', 'DS_NonStruct_AccelSen': 结构损伤状态
            - 'RepairCost_Total', 'RepairCost_Struct', 'RepairCost_NonStruct_DriftSen', 
              'RepairCost_NonStruct_AccelSen': 修复成本
            - 'RepairTime', 'RecoveryTime', 'FunctionLossTime': 时间损失
        iSim (int): 指定模拟索引，默认为0。如果指定震源的模拟数量不足，将自动使用第0个模拟

    Returns:
        str: 保存的JSON文件路径

    生成的JSON文件结构:
        {
            "CityName": "城市名称",
            "LossType": "损失类型",
            "generated_at": "生成时间戳 (ISO格式)",
            "total_earthquakes": "地震事件总数 (int)",
            "earthquake_events": [
                {
                    "i_rup": "震源索引 (int)",
                    "rup_id": "震源ID (str)",
                    "earthquake_info": {
                        "magnitude": "震级 (float)",
                        "centroid_lat": "震中纬度 (float)",
                        "centroid_lon": "震中经度 (float)", 
                        "hypo_depth": "震源深度 (float)",
                        "strike": "走向角 (float)",
                        "dip": "倾角 (float)",
                        "rake": "滑动角 (float)",
                        "trt": "构造类型 (str)",
                        "source_id": "震源ID (str)"
                    },
                    "simulation_info": {
                        "total_simulations": "该震源的总模拟次数 (int)",
                        "selected_iSim": "选择的模拟索引 (int)",
                        "simulation_indices": "所有模拟索引列表 (list[int])"
                    },
                    "loss_statistics": {
                        "total_loss_sim": "该震源所有模拟的损失分布 (N_DS x N_sim 数组)",
                        "total_loss_iSim": "指定模拟的总损失 (N_DS 数组)",
                        "percentiles_iSim": "指定模拟在分布中的百分位数 (N_DS 数组)",
                        "damage_state_names": "损伤状态名称列表 (仅损伤状态类型)",
                        "loss_description": "损失描述 (str)"
                    }
                }
            ]
        }

    损失统计数据说明:
        - 对于损伤状态类型 (DS_*):
            * N_DS = 5 (对应5个损伤状态: 'DS0', 'DS1', 'DS2', 'DS3', 'DS4')
            * total_loss_sim[i,j]: 第j次模拟中处于第i个损伤状态的建筑物数量
            * total_loss_iSim[i]: 指定模拟中处于第i个损伤状态的建筑物数量
            * percentiles_iSim[i]: 指定模拟中第i个损伤状态建筑数量在所有模拟中的百分位数
        
        - 对于连续损失类型 (RepairCost_*, RepairTime等):
            * N_DS = 1
            * total_loss_sim[0,j]: 第j次模拟的总损失值
            * total_loss_iSim[0]: 指定模拟的总损失值
            * percentiles_iSim[0]: 指定模拟的总损失值在所有模拟中的百分位数

    
    文件保存位置:
        {OutputDir}/RegionalLossStatistics/{CityName}/RegionalLossStatistics_{LossType}.json
        如果OutputDir为None，则保存到{ResultsDir}/RegionalLossStatistics/{CityName}/RegionalLossStatistics_{LossType}.json

    使用示例:
        # 生成武汉市结构损伤状态统计，保存到默认位置
        file_path = create_statistics_regional_losses('Wuhan', LossType='DS_Struct')
        
        # 生成总修复成本统计，指定输出目录
        file_path = create_statistics_regional_losses('Wuhan', ResultsDir='/path/to/results', OutputDir='/path/to/output', LossType='RepairCost_Total', iSim=0)
    '''
    logger.info(f'Starting regional loss statistics creation for {CityName}, loss type: {LossType}')
    
    # 设置输入和输出目录
    ResultsDir = Path(ResultsDir).resolve()
    if OutputDir is None:
        OutputDir = ResultsDir
    else:
        OutputDir = Path(OutputDir).resolve()

    output_file = OutputDir / 'RegionalLossStatistics' / CityName / f'RegionalLossStatistics_{LossType}.json'
    logger.info(f'Reading loss data from: {ResultsDir}')
    logger.info(f'Saving statistics to: {output_file}')
    
    # 确保输出目录存在
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 读取地震事件数据
    rup_df = EQsources.ReadCityRuptures(CityName)
    logger.info(f'Found {len(rup_df)} earthquake events for {CityName}')
    
    # 读取rupture到模拟索引的映射
    rupid_2_iSim = IMSim.get_rupid_2_iSim(CityName)
    
    # 预加载损失数据到内存，避免重复IO操作
    loss_data_file_path = ResultsDir / CityName / f'BldLoss_{LossType}.hdf5'
    with h5py.File(loss_data_file_path, 'r') as f:
        LossDataInMemory = Utilities.load_h5py_group_to_memory(f)
    
    # 准备所有震源索引列表
    all_rup_indices = list(range(len(rup_df)))
    logger.info(f'Processing all {len(all_rup_indices)} earthquake events using get_statistics_regional_losses with internal multiprocessing')
    
    # 使用get_statistics_regional_losses一次性处理所有震源
    # 这将利用其内部的多进程优化，避免额外的进程间通信开销
    total_loss_sim_all, total_loss_iSim_all, percentiles_iSim_all = get_statistics_regional_losses(
        CityName=CityName,
        ResultsDir=str(ResultsDir),
        LossType=LossType,
        i_rup=all_rup_indices,
        iSim=iSim,
        LossDataInMemory=LossDataInMemory
    )
    
    logger.info(f'Completed batch processing of all earthquake events')
    
    # 收集所有地震事件的统计信息
    all_statistics = {
        'CityName': CityName,
        'LossType': LossType,
        'generated_at': pd.Timestamp.now().isoformat(),
        'total_earthquakes': len(rup_df),
        'earthquake_events': []
    }
    
    # 为每个震源构建详细信息
    for rup_idx in range(len(rup_df)):
        rup_data = rup_df.iloc[rup_idx]
        rupid = rup_data['rup_id']
        
        # 获取该地震事件的模拟索引列表
        iSim_list = rupid_2_iSim[rupid]['iSim']
        n_simulations = len(iSim_list)
        
        # 确保iSim不超出该震源的模拟数量
        current_iSim = iSim if iSim < n_simulations else 0
        
        # 从批量结果中提取当前震源的数据
        # total_loss_sim_all 形状: (N_DS, N_sim_per_rup, N_rup)
        # total_loss_iSim_all 形状: (N_DS, N_rup)  
        # percentiles_iSim_all 形状: (N_DS, N_rup)
        current_total_loss_sim = total_loss_sim_all[:, :, rup_idx]  # (N_DS, N_sim_per_rup)
        current_total_loss_iSim = total_loss_iSim_all[:, rup_idx] if total_loss_iSim_all is not None else None  # (N_DS,)
        current_percentiles_iSim = percentiles_iSim_all[:, rup_idx] if percentiles_iSim_all is not None else None  # (N_DS,)
        
        # 构建地震事件信息
        earthquake_info = {
            'i_rup': int(rup_idx),
            'rup_id': str(rupid),
            'earthquake_info': {
                'magnitude': float(rup_data['mag']) if 'mag' in rup_data else None,
                'centroid_lat': float(rup_data['centroid_lat']) if 'centroid_lat' in rup_data else None,
                'centroid_lon': float(rup_data['centroid_lon']) if 'centroid_lon' in rup_data else None,
                'hypo_depth': float(rup_data['hypo_depth']) if 'hypo_depth' in rup_data else None,
                'strike': float(rup_data['strike']) if 'strike' in rup_data else None,
                'dip': float(rup_data['dip']) if 'dip' in rup_data else None,
                'rake': float(rup_data['rake']) if 'rake' in rup_data else None,
                'trt': str(rup_data['trt']) if 'trt' in rup_data else None,
                'source_id': str(rup_data['source_id']) if 'source_id' in rup_data else None
            },
            'simulation_info': {
                'total_simulations': n_simulations,
                'selected_iSim': current_iSim,
                'simulation_indices': iSim_list
            },
            'loss_statistics': {
                'total_loss_sim': current_total_loss_sim.tolist(),  # 该震源所有模拟的损失分布
                'total_loss_iSim': current_total_loss_iSim.tolist() if current_total_loss_iSim is not None else None,  # 指定模拟的损失
                'percentiles_iSim': current_percentiles_iSim.tolist() if current_percentiles_iSim is not None else None  # 指定模拟的百分位数
            }
        }
        
        # 添加损失类型特定的统计信息
        if LossType in BldPSPA_ReprBld.LossTypes1:
            # 损伤状态类型的统计
            earthquake_info['loss_statistics']['damage_state_names'] = BldPSPA_ReprBld.DS_types
            earthquake_info['loss_statistics']['loss_description'] = 'Number of buildings in each damage state'
        elif LossType in BldPSPA_ReprBld.LossTypes2:
            # 连续损失类型的统计
            earthquake_info['loss_statistics']['loss_description'] = f'Total {LossType} value'
        
        all_statistics['earthquake_events'].append(earthquake_info)

    logger.info(f'Processing completed for {len(all_statistics["earthquake_events"])} earthquake events')    
    # 保存JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_statistics, f, indent=2, ensure_ascii=False)    
    logger.info(f'Regional loss statistics saved to: {output_file}')
    
    logger.info(f'Regional loss statistics generation completed successfully for {CityName}')
    return str(output_file)

def _process_BldLoss_chunk_worker(args):
    """
    多进程工作函数，处理单个chunk的损失模拟
    每个进程内部从HDF5文件读取所需的数据片段，并直接写入HDF5文件
    
    Parameters:
        args: 包含以下元素的元组
            - bld_chunk: DataFrame chunk containing building data for this chunk
            - IM_h5_file_path: HDF5文件路径，包含IM模拟数据
            - IM_is_h5_file: 布尔值，指示IM数据是否在HDF5文件中
            - IM_data: 如果不是HDF5文件，则是numpy数组数据
            - losstype: Type of loss to simulate
            - chunk_start: Starting index of this chunk
            - chunk_end: Ending index of this chunk
            - output_file: HDF5输出文件路径
            - lock: 多进程锁对象
    
    Returns:
        int: 处理的建筑数量
    """
    bld_chunk, IM_h5_file_path, IM_is_h5_file, IM_data, losstype, chunk_start, chunk_end, output_file, lock = args
    
    # 在每个进程内部按需读取IM数据片段
    if IM_is_h5_file:
        # 从HDF5文件读取当前chunk需要的数据
        with h5py.File(IM_h5_file_path, 'r') as f:
            IM_chunk = f['IM_mat_Bld'][chunk_start:chunk_end, :]
    else:
        # numpy数组直接切片
        IM_chunk = IM_data[chunk_start:chunk_end, :]
    
    # 临时存储当前chunk的结果
    chunk_results = {}
    
    # find unique ReprBldID_PSPA in the chunk
    unique_ReprBldID_PSPA = bld_chunk['ReprBldID_PSPA'].unique()
    
    # iterate based on the ReprBldID_PSPA
    for ReprPSPAID in unique_ReprBldID_PSPA:
        # find the index of the buildings with the same ReprBldID_PSPA.
        idx = bld_chunk[bld_chunk['ReprBldID_PSPA'] == ReprPSPAID].index
        N_same_ReprBldID_PSPA = len(idx)
        N_sim_1_bld = IM_chunk.shape[1]
        
        # get the IM list for this ReprBldID_PSPA
        # 调整索引以匹配chunk内的相对位置
        relative_idx = [i - chunk_start for i in idx]
        IMlist = IM_chunk[relative_idx, :].tolist()
        
        # convert IMlist (N_same_ReprBldID_PSPA x N_sim_1_bld) to one row
        IMlist = [item for sublist in IMlist for item in sublist]
        
        # sim_loss: List of simulated loss values for each IM
        sim_loss = BldPSPA_ReprBld.SimLossRandomly(
            ReprPSPAID=ReprPSPAID, IM=IMlist, losstype=losstype, Nsim=[1]*len(IMlist)
        )
        
        # since each element in sim_loss has only one value, flatten it
        sim_loss = list(np.array(sim_loss).flatten())
        # sim_loss has only one row. convert it to N_same_ReprBldID_PSPA x N_sim_1_bld
        sim_loss = [sim_loss[i*N_sim_1_bld:(i+1)*N_sim_1_bld] for i in range(N_same_ReprBldID_PSPA)]
        
        # 准备当前ReprPSPAID的结果
        for i, idx_i in enumerate(idx):
            if losstype in BldPSPA_ReprBld.LossTypes1:
                # Convert to categorical type to save space
                sim_loss_cat = pd.Categorical(np.array(sim_loss[i]), categories=BldPSPA_ReprBld.DS_types)
                chunk_results[idx_i] = {'data': sim_loss_cat.codes, 'dtype': 'int8'}
            else:
                chunk_results[idx_i] = {'data': sim_loss[i], 'dtype': 'float16'}
    
    # 使用传入的锁进行多进程安全写入
    with lock:
        try:
            with h5py.File(output_file, 'a') as f:
                # 写入数据
                for bld_idx, bld_data in chunk_results.items():
                    dataset_name = f'BldID_{bld_idx}'
                    if dataset_name not in f:  # 避免重复写入
                        f.create_dataset(dataset_name, data=bld_data['data'], dtype=bld_data['dtype'])
        except Exception as e:
            print(f"Warning: Failed to write chunk results: {e}")
            print(f"Chunk processed {len(chunk_results)} buildings but could not write to file")
    
    return len(chunk_results)

def simulate_regional_losses_multiprocess(CityName: str, Bld_gdf: gpd.GeoDataFrame, losstype: str, OutputDir: Path = RESULTDIR, ChunkSize: int = 10000, n_processes: int = None) -> None:
    '''
    使用多进程模拟区域地震损失，基于概率性地震性能评估(PSPA)结果。
    每个进程内部从dask数组读取所需的数据片段，避免大数组的完整复制。

    Parameters:
        CityName (str): 城市名称，用于获取IM模拟数据和确定HDF5文件路径
        Bld_gdf (gpd.GeoDataFrame): 包含建筑数据的GeoDataFrame
        losstype (str): 要模拟的损失类型。选项包括 'DS_Struct', 'DS_NonStruct_DriftSen', 'DS_NonStruct_AccelSen', 'RepairCost_Total', 'RepairCost_Struct', 'RepairCost_NonStruct_DriftSen','RepairCost_NonStruct_AccelSen', 'RepairTime', 'RecoveryTime', 'FunctionLossTime'
        OutputDir (Path): 保存输出文件的目录。默认为 'results/RegionalLossSim/'
        ChunkSize (int): 每个chunk中要处理的建筑数量。默认为10000
        n_processes (int): 使用的进程数。如果为None，则使用CPU核心数
    
    Results:
        BldLoss_{losstype}.hdf5: 包含模拟区域损失的HDF5文件
            BldID_i: 包含建筑i的模拟损失值的数据集。每个元素是Nsim个值的列表
    '''
    logger.info(f"Starting multiprocessing regional loss simulation for {len(Bld_gdf)} buildings with loss type: {losstype}")
    
    # 获取IM模拟数据
    logger.info(f"Loading IM simulation data for city: {CityName}")
    _, IM_sim = IMSim.get_all_IM_mat_Bld(CityName)
    
    # check if the losstype is valid
    if losstype not in BldPSPA_ReprBld.LossTypes1 + BldPSPA_ReprBld.LossTypes2:
        error_msg = f"losstype must be one of {BldPSPA_ReprBld.LossTypes1 + BldPSPA_ReprBld.LossTypes2}."
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Get the full path of the OutputDir
    OutputDir = Path(OutputDir).resolve()
    if not OutputDir.exists():
        OutputDir.mkdir(parents=True)
        logger.info(f"Created output directory: {OutputDir}")

    # Read the table
    T = Bld_gdf

    # check if the row number of IM_sim is equal to the row number of T
    if len(IM_sim) != len(T):
        error_msg = "The number of rows in IM_sim must be equal to the number of rows in BldsFile."
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # 设置进程数
    if n_processes is None:
        n_processes = min(mp.cpu_count(), 4)  # 限制最大进程数避免过度消耗资源
    
    logger.info(f"Processing {len(T)} buildings in chunks of {ChunkSize} using {n_processes} processes")
    
    # 检查数据类型并准备参数
    if isinstance(IM_sim, da.Array):
        # 对于dask数组，尝试直接使用HDF5文件路径
        IM_h5_file_path = str(IMSim.IM_SIM_DIR / CityName / IMSim.IM_MAT_BLD_FILENAME)
        IM_is_h5_file = True
        IM_data = None
        logger.info(f"Using HDF5 file for multiprocessing: {IM_h5_file_path}")
    else:
        # 对于numpy数组的情况
        IM_data = IM_sim
        estimated_size_gb = (IM_sim.nbytes / (1024**3))
        logger.info(f"Detected numpy array with size: {estimated_size_gb:.2f} GB")
        
        IM_is_h5_file = False
        IM_h5_file_path = None
    
    # 创建输出文件
    output_file = OutputDir / f'BldLoss_{losstype}.hdf5'
    
    # 先创建空的HDF5文件
    with h5py.File(output_file, 'w') as f:
        # 创建一个临时数据集以确保文件正确创建
        f.attrs['created'] = pd.Timestamp.now().isoformat()
        f.attrs['total_buildings'] = len(T)
        f.attrs['losstype'] = losstype
    
    logger.info(f"Created output file: {output_file}")
    
    # 创建多进程管理器和锁
    manager = mp.Manager()
    lock = manager.Lock()
    
    # 准备chunk数据
    chunks = []
    for chunk_start in range(0, len(T), ChunkSize):
        chunk_end = min(chunk_start + ChunkSize, len(T))
        bld_chunk = T.iloc[chunk_start:chunk_end].copy()  # 复制避免索引问题
        
        # 传递参数给工作进程（包含锁对象）
        chunks.append((bld_chunk, IM_h5_file_path, IM_is_h5_file, IM_data, losstype, chunk_start, chunk_end, str(output_file), lock))
    
    logger.info(f"Created {len(chunks)} chunks for processing")
    
    # 使用进程池处理chunks
    try:
        with mp.Pool(processes=n_processes) as pool:
            # 使用imap显示进度
            processed_counts = []
            with tqdm(total=len(chunks), desc=f"Processing {len(T)} buildings in {len(chunks)} chunks", unit="chunk") as pbar:
                for count in pool.imap(_process_BldLoss_chunk_worker, chunks):
                    processed_counts.append(count)
                    pbar.update(1)
    except Exception as e:
        logger.error(f"Error in multiprocessing: {str(e)}")
        raise
    
    # 统计处理结果
    total_buildings_processed = sum(processed_counts)
    logger.info(f"Multiprocessing regional loss simulation completed. Processed {total_buildings_processed} buildings. Results saved to: {output_file}")

def get_rupture_simulation_indices(CityName: str, i_rup: Union[int, list]) -> list:
    '''
    获取一个或多个震源的所有模拟索引
    
    Parameters:
        CityName (str): 城市名称
        i_rup (Union[int, list]): 震源索引，可以是单个整数或整数列表
        
    Returns:
        list: 所有相关模拟在全局模拟中的索引
    '''
    # 标准化 i_rup 为列表
    if isinstance(i_rup, int):
        i_rup_list = [i_rup]
    else:
        i_rup_list = i_rup
    
    # 获取震源映射信息
    rupid_2_iSim = IMSim.get_rupid_2_iSim(CityName)
    rup_df = EQsources.ReadCityRuptures(CityName)
    
    # 收集所有震源的模拟索引
    iSim_all = []
    for i_rup_single in i_rup_list:
        rupid = rup_df.iloc[i_rup_single]['rup_id']
        iSim_single = rupid_2_iSim[rupid]['iSim']
        iSim_all.extend(iSim_single)
    
    return iSim_all





