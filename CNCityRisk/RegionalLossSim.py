
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

RESULTDIR = Path(__file__).resolve().parent / 'results/RegionalLossSim/'
MAP_ZIP = RESULTDIR / 'maps.7z'

# Create module logger
logger = Logger.get_logger('RegionalLossSim')

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
    logger.info(f"Starting regional loss simulation for {len(Bld_gdf)} buildings with loss type: {losstype}")
    
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
                            f.create_dataset(f'BldID_{idx_i}', data=sim_loss[i], dtype='float32')
                    pbar.update(len(idx))
    
    output_file = OutputDir / f'BldLoss_{losstype}.hdf5'
    logger.info(f"Regional loss simulation completed. Results saved to: {output_file}")


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
        total_loss_iSim (np.ndarray): regional total losses for the iSim-th simulation. 
            (N_DS)        
        percentiles_iSim (np.ndarray): Percentiles of the iSim-th simulation.
            (N_DS)
                if LossType is 'DS_Struct' ..., N_DS = 5 (damage states). 
                if LossType is 'RepairCost_Total' ..., N_DS = 1.
    '''
    logger.info(f'Calculating regional loss statistics for {CityName}, rupture {i_rup}, loss type: {LossType}')
    
    print(f'Calculating statistics of regional losses...')
    
    Bld_gdf = Utilities.get_CityBldData(CityName)
    N_bld = len(Bld_gdf)
    logger.info(f'Processing {N_bld} buildings for city: {CityName}')

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
                        "total_loss_sim": "所有模拟的损失分布 (N_DS x N_sim 数组)",
                        "total_loss_iSim": "指定模拟的总损失 (N_DS 数组)",
                        "percentiles_iSim": "指定模拟在分布中的百分位数 (N_DS 数组)",
                        "damage_state_names": "损伤状态名称列表 (仅损伤状态类型)",
                        "loss_description": "损失描述 (str)"
                    }
                }
            ],
            "summary": {
                "successful_events": "成功处理的事件数 (int)",
                "failed_events": "处理失败的事件数 (int)",
                "success_rate": "成功率 (float, 0-1)"
            }
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

    错误处理:
        - 如果某个地震事件处理失败，会在结果中记录错误信息而不是终止整个过程
        - 错误事件包含基本的地震信息和错误描述
        - 最终统计中会显示成功和失败的事件数量    
    
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
    rupid_2_iSim = pd.read_csv(IMSim.IM_SIM_DIR / CityName / IMSim.RUPID_2_ISIM_FILENAME, index_col=0, header=0, dtype={0: str})
    rupid_2_iSim['iSim'] = rupid_2_iSim['iSim'].apply(
        lambda x: list(map(int, x.strip('[]').split())) if isinstance(x, str) else x)
    rupid_2_iSim = rupid_2_iSim.to_dict(orient='index')
    # 预加载损失数据到内存以提高效率
    with h5py.File(ResultsDir / CityName / f'BldLoss_{LossType}.hdf5', 'r') as f:
        LossDataInMemory = Utilities.load_h5py_group_to_memory(f)
    logger.info(f'Loss data preloaded into memory for efficient processing')
    
    # 收集所有地震事件的统计信息
    all_statistics = {
        'CityName': CityName,
        'LossType': LossType,
        'generated_at': pd.Timestamp.now().isoformat(),
        'total_earthquakes': len(rup_df),
        'earthquake_events': []
    }
    
    # 遍历所有地震事件
    for i_rup in tqdm(range(len(rup_df)), desc="Processing earthquake events"):
        rup_data = rup_df.iloc[i_rup]
        rupid = rup_data['rup_id']
        
        # 获取该地震事件的模拟索引列表
        iSim_list = rupid_2_iSim[rupid]['iSim']
        n_simulations = len(iSim_list)
        
        # 如果未指定iSim，使用第0个模拟
        current_iSim = iSim if iSim is not None else 0
        if current_iSim >= n_simulations:
            print(f"Warning: iSim {current_iSim} exceeds available simulations ({n_simulations}) for rupture {rupid}. Using iSim=0.")
            current_iSim = 0
        
        try:
            # 调用get_statistics_regional_losses获取统计信息
            total_loss_sim, total_loss_iSim, percentiles_iSim = get_statistics_regional_losses(
                CityName=CityName,
                ResultsDir=ResultsDir,
                LossType=LossType,
                i_rup=i_rup,
                iSim=current_iSim,
                LossDataInMemory=LossDataInMemory
            )
            
            # 构建地震事件信息
            earthquake_info = {
                'i_rup': int(i_rup),
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
                    'total_loss_sim': total_loss_sim.tolist(),  # 所有模拟的损失分布
                    'total_loss_iSim': total_loss_iSim.tolist() if total_loss_iSim is not None else None,  # 指定模拟的损失
                    'percentiles_iSim': percentiles_iSim.tolist() if percentiles_iSim is not None else None  # 指定模拟的百分位数
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
            
        except Exception as e:
            logger.error(f"Error processing rupture {i_rup} (ID: {rupid}): {str(e)}")
            # 添加错误信息到结果中
            error_info = {
                'i_rup': int(i_rup),
                'rup_id': str(rupid),
                'error': str(e),
                'earthquake_info': {
                    'magnitude': float(rup_data['mag']) if 'mag' in rup_data else None,
                    'centroid_lat': float(rup_data['centroid_lat']) if 'centroid_lat' in rup_data else None,
                    'centroid_lon': float(rup_data['centroid_lon']) if 'centroid_lon' in rup_data else None
                }
            }
            all_statistics['earthquake_events'].append(error_info)
    # 添加汇总统计信息
    successful_events = [e for e in all_statistics['earthquake_events'] if 'error' not in e]
    all_statistics['summary'] = {
        'successful_events': len(successful_events),
        'failed_events': len(all_statistics['earthquake_events']) - len(successful_events),
        'success_rate': len(successful_events) / len(all_statistics['earthquake_events']) if all_statistics['earthquake_events'] else 0
    }
    
    logger.info(f'Processing completed: {len(successful_events)} successful, {all_statistics["summary"]["failed_events"]} failed')    
    # 保存JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_statistics, f, indent=2, ensure_ascii=False)    
    logger.info(f'Regional loss statistics saved to: {output_file}')
    
    logger.info(f'Regional loss statistics generation completed successfully for {CityName}')
    return str(output_file)

