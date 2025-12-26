'''
模块功能说明：
==============

本模块 (RegionalLossSim) 提供区域地震损失模拟和分析的完整工具链，包括：

------------------
主要分析函数：
simulate_regional_losses()
    - 多进程版本的区域地震损失模拟
    - 直接按震源（i_rup）分组存储数据，无需后续重组操作
    - 提高大规模计算效率并优化存储结构
    [内部调用关系]:
      -> _simulate_regional_losses_multiprocess() (City Mode): 执行多进程的城市区域损失模拟
         -> _process_BldLoss_chunk_worker(): 多进程工作函数，处理单个建筑数据块的损失模拟
      -> _simulate_regional_losses_event() (Region Mode): 执行单事件的区域损失模拟
      -> _reorganize_RegionBldLoss_by_rupture() (Region Mode): 重组单事件模拟结果为按震源存储的格式

reorganize_BldLoss_by_rupture()
    - 重新组织已生成的BldLoss文件，按震源分组存储
    [内部调用关系]:
      -> _process_reorganize_chunk_worker(): 多进程工作函数，处理单个数据块的重组

------------------
分析后查询结果函数：
load_BldLoss_by_rupture()
    - 从重组后的文件中加载指定震源的损失数据
    - 支持单个或多个震源的快速读取

get_total_loss_sim()
    - 通用的区域总损失分布计算接口
    - 自动选择最优的数据源和计算方法
    [内部调用关系]:
      -> _get_total_loss_sim_by_rupture(): 带缓存的计算函数，从重组后的文件中获取总损失分布
         -> _calculate_total_loss_sim_by_rupture_core(): 计算总损失分布的核心逻辑，负责调度多进程计算
            -> _calculate_rupture_loss_by_rupture(): 多进程工作函数，计算单个震源的损失分布

get_total_loss_and_collapse_ratio_sim()
    - 计算所有模拟的区域总损失比率，支持按区域分解
    [内部调用关系]:
      -> _calculate_grouped_loss_sim_core(): 计算分组（如按行政区）的损失统计
         -> _calculate_rupture_grouped_loss_worker(): 多进程工作函数，计算单个震源的分组损失

get_statistics_regional_losses()
    - 计算区域地震损失的详细统计信息
    - 包括损失分布、百分位数等统计指标

------------------
生成地图显示数据：

create_geojson_regional_losses()
    - 生成区域损失的GeoJSON格式数据
    - 支持Web可视化和GIS分析
create_statistics_regional_losses()
    - 批量生成所有地震事件的损失统计JSON文件
    - 提供结构化的统计数据输出

------------------
其他功能函数：
load_loss_data_to_memory()
     - 将损失数据从HDF5文件加载到内存
     - 便于后续快速访问和分析
get_rupture_simulation_indices()
     - 获取指定震源的模拟索引信息
     - 辅助函数，用于数据索引和验证

------------------
备用/未使用的内部函数:
_simulate_regional_losses()
    - 备用实现：直接按震源分组存储的多进程模拟
    - 调用: _process_BldLoss_chunk_worker_by_rupture()
_simulate_regional_losses_single_process()
    - 备用实现：单进程版本的区域损失模拟
'''



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
from typing import Optional, Union
import matplotlib.pyplot as plt
import py7zr
import json
import multiprocessing as mp
import hashlib
import pickle
import time
import os


RESULTDIR = Path(__file__).resolve().parent / 'results/RegionalLossSim/'
MAP_ZIP = RESULTDIR / 'maps.7z'

# Create module logger
logger = Logger.get_logger('RegionalLossSim')


def simulate_regional_losses(CityNameorRegionName: str, Bld_gdf: gpd.GeoDataFrame, Event: dict = None, losstype: str = 'DS_Struct', OutputDir: Path = RESULTDIR, ChunkSize: int = 5000, n_processes: int = None) -> None:
    '''
    使用多进程模拟区域地震损失，基于概率性地震性能评估(PSPA)结果。

    Parameters:
        CityName (str): 城市名称，用于获取IM模拟数据和确定HDF5文件路径.
        Bld_gdf (gpd.GeoDataFrame): 包含建筑数据的GeoDataFrame
        Event (dict): 如果为None，则在城市模式下运行；否则在区域模式下运行
        losstype (str): 要模拟的损失类型。选项包括 'DS_Struct', 'DS_NonStruct_DriftSen', 'DS_NonStruct_AccelSen', 'RepairCost_Total', 'RepairCost_Struct', 'RepairCost_NonStruct_DriftSen','RepairCost_NonStruct_AccelSen', 'RepairTime', 'RecoveryTime', 'FunctionLossTime'
        OutputDir (Path): 保存输出文件的目录。默认为 'results/RegionalLossSim/'
        ChunkSize (int): 每个chunk中要处理的建筑数量。默认为10000
        n_processes (int): 使用的进程数。如果为None，则使用4个
    
    Results:
        城市模式 (City mode):
            results/RegionalLossSim/<CityName>/BldLoss_{losstype}_ByRupture.hdf5
                group_rup_{i}: 震源 i 的数据组
                    loss_matrix: (N_bld x N_sim) 损失矩阵，行为建筑，列为该震源的模拟
                    rupid: 震源 ID 字符串
                    iSim_indices: 该震源在全局模拟中的索引列表
                    rup_info: 震源的详细信息（震级、位置等）
        区域模式 (Region mode):
            results/RegionalLossSim/<RegionName>/<event_name>/BldLoss_{losstype}_ByRupture.hdf5
                - ``loss_matrix``: 单事件的损失矩阵 (N_bld × N_sim)
                - ``building_indices``、``simulation_indices``: 行列索引映射
                - ``event_info``: 事件名称与参数元数据
    '''
    
    if Event is None:
        # city mode
        _simulate_regional_losses_multiprocess(CityNameorRegionName, Bld_gdf, losstype, OutputDir, ChunkSize, n_processes)
        reorganize_BldLoss_by_rupture(CityNameorRegionName, losstype, OutputDir, OutputDir, chunk_size=ChunkSize)

        # clean up the intermediate file
        interm_file = OutputDir / CityNameorRegionName / f'BldLoss_{losstype}.hdf5'
        if interm_file.exists():
            interm_file.unlink()
            logger.info(f"Deleted intermediate file: {interm_file}")
    else:
        # region mode
        _simulate_regional_losses_event(
            region_name=CityNameorRegionName,
            Bld_gdf=Bld_gdf,
            event=Event,
            losstype=losstype,
            output_dir=OutputDir,
            chunk_size=ChunkSize
        )
        reorganized_file = _reorganize_RegionBldLoss_by_rupture(
            region_name=CityNameorRegionName,
            event=Event,
            losstype=losstype,
            output_dir=OutputDir,
            Bld_gdf=Bld_gdf
        )
        logger.info(f"Reorganized regional loss matrix saved to: {reorganized_file}")

        # clean up the intermediate file
        event_name = IMSim.get_event_name(Event)
        interm_file = OutputDir / CityNameorRegionName / event_name / f'BldLoss_{losstype}.hdf5'
        if interm_file.exists():
            interm_file.unlink()
            logger.info(f"Deleted intermediate file: {interm_file}")


def _simulate_regional_losses_event(region_name: str, Bld_gdf: gpd.GeoDataFrame, event: dict, losstype: str, output_dir: Path, chunk_size: int) -> None:
    """
    Simulate losses for a single-event regional run using precomputed IM fields.

    Results:
        results/RegionalLossSim/<RegionName>/<event_name>/BldLoss_{losstype}.hdf5
        文件属性: RegionName, event_name, losstype, total_buildings, total_simulations, event_parameters
        数据集 BldID_{idx}: 针对每栋建筑的模拟结果，维度为 (N_sim,)
            - 对离散损伤类型，存储为整型编码（映射至 BldPSPA_ReprBld.DS_types）
            - 对连续损失指标，存储为 float32 数组
    """

    if losstype not in BldPSPA_ReprBld.LossTypes1 + BldPSPA_ReprBld.LossTypes2:
        error_msg = f"losstype must be one of {BldPSPA_ReprBld.LossTypes1 + BldPSPA_ReprBld.LossTypes2}."
        logger.error(error_msg)
        raise ValueError(error_msg)

    event_name = IMSim.get_event_name(event)
    logger.info(
        f"Starting regional loss simulation for event '{event_name}' with {len(Bld_gdf)} buildings and loss type: {losstype}"
    )

    event_dir = IMSim.IM_SIM_DIR / region_name / event_name
    im_file = event_dir / 'IM_mat_Bld.txt'
    if not im_file.exists():
        error_msg = f"IM results file not found for event '{event_name}': {im_file}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    try:
        IM_sim_df = pd.read_csv(im_file, sep=r'\s+', header=None, dtype=np.float32)
        IM_sim = IM_sim_df.values
    except Exception as exc:
        logger.error(f"Failed to read IM_mat_Bld.txt for event '{event_name}': {exc}")
        raise

    if IM_sim.ndim == 1:
        try:
            IM_sim = IM_sim.reshape(len(Bld_gdf), -1)
        except ValueError as exc:
            error_msg = (
                "Unable to reshape IM simulation data for single-column event results. "
                "Ensure the IM_mat_Bld.txt layout matches the building count."
            )
            logger.error(error_msg)
            raise ValueError(error_msg) from exc

    if IM_sim.shape[0] != len(Bld_gdf):
        error_msg = (
            f"IM simulation row count ({IM_sim.shape[0]}) does not match building count ({len(Bld_gdf)})."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    output_dir = Path(output_dir).resolve() / region_name / event_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'BldLoss_{losstype}.hdf5'

    logger.info(f"Writing regional loss results to: {output_file}")

    with tqdm(total=len(Bld_gdf), desc=f"Simulating losses for {event_name}", unit="building") as pbar:
        with h5py.File(output_file, 'w') as f:
            f.attrs['RegionName'] = region_name
            f.attrs['event_name'] = event_name
            f.attrs['losstype'] = losstype
            f.attrs['created'] = pd.Timestamp.now().isoformat()
            f.attrs['total_buildings'] = len(Bld_gdf)
            f.attrs['total_simulations'] = int(IM_sim.shape[1])
            try:
                f.attrs['event_parameters'] = json.dumps(event)
            except TypeError:
                f.attrs['event_parameters'] = str(event)

            for chunk_start in range(0, len(Bld_gdf), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(Bld_gdf))
                T_chunk = Bld_gdf.iloc[chunk_start:chunk_end]
                IM_chunk = IM_sim[chunk_start:chunk_end, :]

                unique_reprs = T_chunk['ReprBldID_PSPA'].unique()
                for repr_id in unique_reprs:
                    mask = T_chunk['ReprBldID_PSPA'] == repr_id
                    idx = T_chunk.index[mask]
                    local_positions = np.flatnonzero(mask.to_numpy())
                    if not len(idx):
                        continue

                    N_sim_per_bld = IM_chunk.shape[1]
                    IMlist = IM_chunk[local_positions, :].reshape(-1).tolist()

                    sim_loss = BldPSPA_ReprBld.SimLossRandomly(
                        ReprPSPAID=repr_id,
                        IM=IMlist,
                        losstype=losstype,
                        Nsim=[1] * len(IMlist)
                    )

                    sim_loss = np.asarray(sim_loss).flatten()
                    if sim_loss.size != len(idx) * N_sim_per_bld:
                        error_msg = (
                            f"Unexpected loss array size for representative building {repr_id}: "
                            f"expected {len(idx) * N_sim_per_bld}, got {sim_loss.size}."
                        )
                        logger.error(error_msg)
                        raise ValueError(error_msg)

                    sim_loss = sim_loss.reshape(len(idx), N_sim_per_bld)

                    for i, idx_i in enumerate(idx):
                        dataset_name = f'BldID_{idx_i}'
                        if losstype in BldPSPA_ReprBld.LossTypes1:
                            sim_loss_cat = pd.Categorical(
                                np.array(sim_loss[i]),
                                categories=BldPSPA_ReprBld.DS_types
                            )
                            data = sim_loss_cat.codes.astype('int8')
                            f.create_dataset(
                                dataset_name,
                                data=data,
                                dtype='int8',
                                compression='gzip',
                                compression_opts=9,
                                shuffle=True
                            )
                        else:
                            data = np.array(sim_loss[i], dtype='float32')
                            f.create_dataset(
                                dataset_name,
                                data=data,
                                dtype='float32',
                                compression='gzip',
                                compression_opts=9,
                                shuffle=True
                            )
                        pbar.update(1)

    logger.info(
        f"Regional loss simulation for event '{event_name}' completed. Results saved to: {output_file}"
    )


def _reorganize_RegionBldLoss_by_rupture(
    region_name: str,
    event: Union[dict, str],
    losstype: str,
    output_dir: Path = RESULTDIR,
    chunk_size: int = 10000,
    Bld_gdf: Optional[gpd.GeoDataFrame] = None
) -> Path:
    """Reorganize single-event regional losses into a loss_matrix dataset.

    Parameters:
        region_name (str): 区域名称，对应结果输出目录。
        event (dict | str): 事件字典或事件名称。
        losstype (str): 损失类型，必须与模拟步骤保持一致。
        output_dir (Path): 区域结果根目录。
        chunk_size (int): 每次批量写入的建筑数量，避免一次性占用过多内存。
        Bld_gdf (GeoDataFrame | None): 建筑数据，用于建立建筑索引顺序。

    Returns:
        Path: 所生成的 ``BldLoss_{losstype}_ByRupture.hdf5`` 文件路径。
    """

    if losstype not in BldPSPA_ReprBld.LossTypes1 + BldPSPA_ReprBld.LossTypes2:
        error_msg = f"losstype must be one of {BldPSPA_ReprBld.LossTypes1 + BldPSPA_ReprBld.LossTypes2}."
        logger.error(error_msg)
        raise ValueError(error_msg)

    event_name = IMSim.get_event_name(event) if isinstance(event, dict) else str(event)
    if not event_name:
        raise ValueError("Event name cannot be empty when reorganizing regional losses")

    base_dir = Path(output_dir).resolve() / region_name / event_name
    input_file = base_dir / f'BldLoss_{losstype}.hdf5'
    output_file = base_dir / f'BldLoss_{losstype}_ByRupture.hdf5'

    if not input_file.exists():
        error_msg = f"Input file not found for event '{event_name}': {input_file}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    logger.info(f"Reorganizing regional loss file into matrix form: {input_file}")

    building_indices: Optional[np.ndarray] = None
    if Bld_gdf is not None and not Bld_gdf.empty:
        try:
            building_indices = Bld_gdf.index.to_numpy(dtype=np.int64, copy=True)
        except Exception:
            logger.debug("Failed to convert building GeoDataFrame indices directly; falling back to file datasets")
            building_indices = None

    with h5py.File(input_file, 'r') as f_in:
        dataset_names = [name for name in f_in.keys() if name.startswith('BldID_')]
        if not dataset_names:
            raise ValueError(f"No building datasets found in {input_file}")

        dataset_map = {int(name.split('_', 1)[1]): name for name in dataset_names}

        if building_indices is None or building_indices.size == 0:
            building_indices = np.array(sorted(dataset_map.keys()), dtype=np.int64)
        else:
            indices_list = np.array(building_indices, dtype=np.int64)
            dataset_ids = np.array(sorted(dataset_map.keys()), dtype=np.int64)
            if indices_list.size != dataset_ids.size or set(indices_list.tolist()) != set(dataset_ids.tolist()):
                raise ValueError(
                    "Building indices from geometry data do not match the loss file datasets"
                )
            building_indices = indices_list

        if building_indices.size == 0:
            raise ValueError("No building indices available to reorganize event losses")

        # Determine simulation count and data type using the first available dataset
        sample_dataset = f_in[dataset_map[int(building_indices[0])]]
        sample_data = sample_dataset[:]
        if sample_data.size == 0:
            raise ValueError("Sample dataset contains no simulations; cannot determine matrix dimensions")
        n_sim = sample_data.size
        dtype = 'int8' if losstype in BldPSPA_ReprBld.LossTypes1 else 'float32'

    base_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(input_file, 'r') as f_in, h5py.File(output_file, 'w') as f_out:
        # copy high-level metadata
        for key, value in f_in.attrs.items():
            f_out.attrs[key] = value
        f_out.attrs['RegionName'] = region_name
        f_out.attrs['event_name'] = event_name
        f_out.attrs['losstype'] = losstype
        f_out.attrs['created'] = pd.Timestamp.now().isoformat()
        f_out.attrs['total_buildings'] = int(building_indices.size)
        f_out.attrs['total_simulations'] = int(n_sim)
        f_out.attrs['original_file'] = str(input_file)
        f_out.attrs['organized_mode'] = 'event'

        fill_value = -1 if dtype == 'int8' else np.nan
        loss_ds = f_out.create_dataset(
            'loss_matrix',
            shape=(building_indices.size, n_sim),
            dtype=dtype,
            compression='gzip',
            compression_opts=9,
            shuffle=True,
            chunks=True,
            fillvalue=fill_value
        )

        f_out.create_dataset('building_indices', data=building_indices, compression='gzip')
        f_out.create_dataset('simulation_indices', data=np.arange(n_sim, dtype=np.int32), compression='gzip')

        event_info_group = f_out.create_group('event_info')
        event_info_group.attrs['event_name'] = event_name
        if 'event_parameters' in f_in.attrs:
            event_info_group.attrs['event_parameters'] = f_in.attrs['event_parameters']

        logger.info(f"Writing loss matrix with shape ({building_indices.size}, {n_sim})")

        safe_chunk = max(int(chunk_size), 1)
        total_buildings = building_indices.size

        with tqdm(total=total_buildings, desc="Reorganizing region event", unit="building") as pbar:
            for start in range(0, total_buildings, safe_chunk):
                end = min(start + safe_chunk, total_buildings)
                chunk_indices = building_indices[start:end]
                chunk_rows = end - start
                chunk_buffer = np.empty((chunk_rows, n_sim), dtype=dtype)

                for row_offset, bld_idx in enumerate(chunk_indices):
                    dataset_name = dataset_map.get(int(bld_idx))
                    if dataset_name is None:
                        raise KeyError(f"Building ID {bld_idx} missing from loss datasets")

                    data = f_in[dataset_name][:]

                    if data.size == 0:
                        chunk_buffer[row_offset, :] = fill_value
                    elif data.size != n_sim:
                        raise ValueError(
                            f"Unexpected simulation count for building {bld_idx}: expected {n_sim}, got {data.size}"
                        )
                    else:
                        chunk_buffer[row_offset, :] = data.astype(dtype, copy=False)

                loss_ds[start:end, :] = chunk_buffer
                pbar.update(chunk_rows)

    logger.info(f"Reorganized event loss matrix saved to: {output_file}")
    return output_file


def _simulate_regional_losses(CityName: str, Bld_gdf: gpd.GeoDataFrame, losstype: str, OutputDir: Path = RESULTDIR, ChunkSize: int = 5000, n_processes: int = None) -> None:
    '''
    使用多进程模拟区域地震损失，基于概率性地震性能评估(PSPA)结果。
    每个进程内部从dask数组读取所需的数据片段，避免大数组的完整复制。
    直接按震源（i_rup）分组存储数据，无需后续重组操作。

    Parameters:
        CityName (str): 城市名称，用于获取IM模拟数据和确定HDF5文件路径
        Bld_gdf (gpd.GeoDataFrame): 包含建筑数据的GeoDataFrame
        losstype (str): 要模拟的损失类型。选项包括 'DS_Struct', 'DS_NonStruct_DriftSen', 'DS_NonStruct_AccelSen', 'RepairCost_Total', 'RepairCost_Struct', 'RepairCost_NonStruct_DriftSen','RepairCost_NonStruct_AccelSen', 'RepairTime', 'RecoveryTime', 'FunctionLossTime'
        OutputDir (Path): 保存输出文件的目录。默认为 'results/RegionalLossSim/'
        ChunkSize (int): 每个chunk中要处理的建筑数量。默认为10000
        n_processes (int): 使用的进程数。如果为None，则使用4个
    
    Results:
        BldLoss_{losstype}_ByRupture.hdf5: 包含按震源分组的模拟区域损失的HDF5文件
            group_rup_{i}: 震源i的数据组
                loss_matrix: (N_bld x N_sim) 损失矩阵，行为建筑，列为该震源的模拟
                rupid: 震源ID字符串
                iSim_indices: 该震源在全局模拟中的索引列表
                rup_info: 震源的详细信息（震级、位置等）
    '''
    logger.info(f"Starting multiprocessing regional loss simulation for {len(Bld_gdf)} buildings with loss type: {losstype}")
    
    # 获取IM模拟数据
    logger.info(f"Loading IM simulation data for city: {CityName}")
    rupid_2_iSim, IM_sim = IMSim.get_all_IM_mat_Bld(CityName)
    
    # check if the losstype is valid
    if losstype not in BldPSPA_ReprBld.LossTypes1 + BldPSPA_ReprBld.LossTypes2:
        error_msg = f"losstype must be one of {BldPSPA_ReprBld.LossTypes1 + BldPSPA_ReprBld.LossTypes2}."
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Get the full path of the OutputDir with CityName subdirectory
    OutputDir = Path(OutputDir).resolve() / CityName
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
    
    # 获取震源信息
    rup_df = EQsources.ReadCityRuptures(CityName)
    N_bld = len(T)
    N_rup = len(rup_df)
    
    logger.info(f"Found {N_rup} ruptures and {N_bld} buildings")
    
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
    
    # 创建输出文件，直接按震源分组
    output_file = OutputDir / f'BldLoss_{losstype}_ByRupture.hdf5'
    
    # 创建HDF5文件并设置按震源分组的结构
    with h5py.File(output_file, 'w') as f:
        # 添加文件级别的元数据
        f.attrs['CityName'] = CityName
        f.attrs['losstype'] = losstype
        f.attrs['created'] = pd.Timestamp.now().isoformat()
        f.attrs['total_buildings'] = N_bld
        f.attrs['total_ruptures'] = N_rup
        f.attrs['organized_by_rupture'] = True
        
        # 为每个震源创建组和数据集
        logger.info("Creating groups and datasets for each rupture...")
        iSim_mappings = {}
        
        with tqdm(total=N_rup, desc="Creating rupture groups...", unit="rupture") as pbar:
            for i_rup in range(N_rup):
                rup_data = rup_df.iloc[i_rup]
                rupid = rup_data['rup_id']
            
                # 获取该震源的模拟索引
                iSim_list = rupid_2_iSim[rupid]['iSim']
                N_sim_rup = len(iSim_list)
                iSim_mappings[i_rup] = iSim_list
                
                # 创建该震源的组
                rup_group = f.create_group(f'group_rup_{i_rup}')
                
                # 预先创建损失矩阵数据集
                if losstype in BldPSPA_ReprBld.LossTypes1:
                    # 损伤状态类型使用int8
                    rup_group.create_dataset(
                        'loss_matrix', 
                        shape=(N_bld, N_sim_rup),
                        dtype='int8',
                        compression='gzip',
                        compression_opts=9,
                        shuffle=True,
                        chunks=True,
                        fillvalue=-1  # 用-1表示未处理的数据
                    )
                else:
                    # 连续损失类型使用float32
                    rup_group.create_dataset(
                        'loss_matrix', 
                        shape=(N_bld, N_sim_rup),
                        dtype='float32',
                        compression='gzip',
                        compression_opts=9,
                        shuffle=True,
                        chunks=True,
                        fillvalue=-1.0  # 用-1.0表示未处理的数据
                    )
                
                # 保存震源元数据
                rup_group.attrs['rupid'] = str(rupid)
                rup_group.attrs['i_rup'] = i_rup
                rup_group.attrs['N_sim'] = N_sim_rup
                
                # 保存模拟索引
                rup_group.create_dataset('iSim_indices', data=iSim_list, compression='gzip')
                
                # 保存震源详细信息
                rup_info_group = rup_group.create_group('rup_info')
                for col in rup_df.columns:
                    if col in rup_data and pd.notna(rup_data[col]):
                        value = rup_data[col]
                        if isinstance(value, (int, float, str)):
                            rup_info_group.attrs[col] = value
                        else:
                            rup_info_group.attrs[col] = str(value)
                
                # 更新进度条
                pbar.update(1)
    
    logger.info(f"Created output file with rupture-organized structure: {output_file}")
    
    # 创建多进程管理器和锁
    manager = mp.Manager()
    lock = manager.Lock()
    
    # 准备chunk数据
    chunks = []
    for chunk_start in range(0, len(T), ChunkSize):
        chunk_end = min(chunk_start + ChunkSize, len(T))
        bld_chunk = T.iloc[chunk_start:chunk_end].copy()  # 复制避免索引问题
        
        # 传递参数给工作进程（包含锁对象和震源映射信息）
        chunks.append((bld_chunk, IM_h5_file_path, IM_is_h5_file, IM_data, losstype, chunk_start, chunk_end, str(output_file), lock, iSim_mappings, N_rup))
    
    logger.info(f"Created {len(chunks)} chunks for processing")
    
    # 使用进程池处理chunks
    try:
        with mp.Pool(processes=n_processes) as pool:
            # 使用imap显示进度
            processed_counts = []
            with tqdm(total=len(chunks), desc=f"Processing {len(T)} buildings in {len(chunks)} chunks", unit="chunk") as pbar:
                for count in pool.imap(_process_BldLoss_chunk_worker_by_rupture, chunks):
                    processed_counts.append(count)
                    pbar.update(1)
    except Exception as e:
        logger.error(f"Error in multiprocessing: {str(e)}")
        raise
    
    # 统计处理结果
    total_buildings_processed = sum(processed_counts)
    logger.info(f"Multiprocessing regional loss simulation completed (organized by rupture). Processed {total_buildings_processed} buildings. Results saved to: {output_file}")


def _process_BldLoss_chunk_worker_by_rupture(args):
    """
    多进程工作函数，处理单个chunk的损失模拟，直接按震源分组写入数据
    每个进程内部从HDF5文件读取所需的数据片段，并按震源分组直接写入HDF5文件
    
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
            - iSim_mappings: 震源到模拟索引的映射字典 {i_rup: [iSim_list]}
            - N_rup: 震源总数
    
    Returns:
        int: 处理的建筑数量
    """
    bld_chunk, IM_h5_file_path, IM_is_h5_file, IM_data, losstype, chunk_start, chunk_end, output_file, lock, iSim_mappings, N_rup = args
    
    # 在每个进程内部按需读取IM数据片段
    if IM_is_h5_file:
        # 从HDF5文件读取当前chunk需要的数据
        with h5py.File(IM_h5_file_path, 'r') as f:
            IM_chunk = f['IM_mat_Bld'][chunk_start:chunk_end, :]
    else:
        # numpy数组直接切片
        IM_chunk = IM_data[chunk_start:chunk_end, :]
    
    # 存储按震源分组的结果
    rupture_results = {}  # {i_rup: {bld_idx: loss_values}}
    
    # find unique ReprBldID_PSPA in the chunk
    unique_ReprBldID_PSPA = bld_chunk['ReprBldID_PSPA'].unique()
    
    # iterate based on the ReprBldID_PSPA
    for ReprPSPAID in unique_ReprBldID_PSPA:
        # find the index of the buildings with the same ReprBldID_PSPA.
        idx = bld_chunk[bld_chunk['ReprBldID_PSPA'] == ReprPSPAID].index
        N_same_ReprBldID_PSPA = len(idx)
        N_sim_total = IM_chunk.shape[1]
        
        # get the IM list for this ReprBldID_PSPA
        # 调整索引以匹配chunk内的相对位置
        relative_idx = [i - chunk_start for i in idx]
        IMlist = IM_chunk[relative_idx, :].tolist()
        
        # convert IMlist (N_same_ReprBldID_PSPA x N_sim_total) to one row
        IMlist = [item for sublist in IMlist for item in sublist]
        
        # sim_loss: List of simulated loss values for each IM
        sim_loss = BldPSPA_ReprBld.SimLossRandomly(
            ReprPSPAID=ReprPSPAID, IM=IMlist, losstype=losstype, Nsim=[1]*len(IMlist)
        )
        
        # since each element in sim_loss has only one value, flatten it
        sim_loss = list(np.array(sim_loss).flatten())
        # sim_loss has only one row. convert it to N_same_ReprBldID_PSPA x N_sim_total
        sim_loss = [sim_loss[i*N_sim_total:(i+1)*N_sim_total] for i in range(N_same_ReprBldID_PSPA)]
        
        # 按震源分组处理损失数据
        for i_rup in range(N_rup):
            if i_rup not in rupture_results:
                rupture_results[i_rup] = {}
            
            # 获取该震源的模拟索引
            iSim_list = iSim_mappings[i_rup]
            
            # 为当前ReprPSPAID的每个建筑提取该震源的损失数据
            for i, idx_i in enumerate(idx):
                # 提取该震源的损失值
                rup_loss_values = [sim_loss[i][j] for j in iSim_list]
                
                # 转换数据类型
                if losstype in BldPSPA_ReprBld.LossTypes1:
                    # Convert to categorical type to save space
                    sim_loss_cat = pd.Categorical(np.array(rup_loss_values), categories=BldPSPA_ReprBld.DS_types)
                    rupture_results[i_rup][idx_i] = sim_loss_cat.codes.astype('int8')
                else:
                    rupture_results[i_rup][idx_i] = np.array(rup_loss_values, dtype='float32')
    
    # 使用传入的锁进行多进程安全写入
    with lock:
        try:
            with h5py.File(output_file, 'a') as f:
                # 按震源写入数据
                for i_rup, bld_data in rupture_results.items():
                    rup_group_name = f'group_rup_{i_rup}'
                    if rup_group_name in f:
                        rup_group = f[rup_group_name]
                        loss_matrix = rup_group['loss_matrix']
                        
                        # 为该震源写入建筑损失数据
                        for bld_idx, loss_values in bld_data.items():
                            # 写入损失矩阵的对应行
                            loss_matrix[bld_idx, :] = loss_values
        except Exception as e:
            print(f"Warning: Failed to write chunk results by rupture: {e}")
            print(f"Chunk processed buildings but could not write to file")
    
    return len(bld_chunk)


def _simulate_regional_losses_single_process(CityName: str, Bld_gdf: gpd.GeoDataFrame, losstype: str, OutputDir: Path = RESULTDIR, ChunkSize = 10000) -> None:
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
    
    # Get the full path of the OutputDir with CityName subdirectory
    OutputDir = Path(OutputDir).resolve() / CityName
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
                            f.create_dataset(f'BldID_{idx_i}', data=sim_loss_cat.codes, dtype='int8', compression='gzip', compression_opts=9, shuffle=True)
                        else:
                            f.create_dataset(f'BldID_{idx_i}', data=sim_loss[i], dtype='float32', compression='gzip', compression_opts=9, shuffle=True)
                    pbar.update(len(idx))
    
    output_file = OutputDir / f'BldLoss_{losstype}.hdf5'
    logger.info(f"Single-process regional loss simulation completed. Results saved to: {output_file}")


def _simulate_regional_losses_multiprocess(CityName: str, Bld_gdf: gpd.GeoDataFrame, losstype: str, OutputDir: Path = RESULTDIR, ChunkSize: int = 10000, n_processes: int = None) -> None:
    '''
    使用多进程模拟区域地震损失，基于概率性地震性能评估(PSPA)结果。
    每个进程内部从dask数组读取所需的数据片段，避免大数组的完整复制。

    Parameters:
        CityName (str): 城市名称，用于获取IM模拟数据和确定HDF5文件路径
        Bld_gdf (gpd.GeoDataFrame): 包含建筑数据的GeoDataFrame
        losstype (str): 要模拟的损失类型。选项包括 'DS_Struct', 'DS_NonStruct_DriftSen', 'DS_NonStruct_AccelSen', 'RepairCost_Total', 'RepairCost_Struct', 'RepairCost_NonStruct_DriftSen','RepairCost_NonStruct_AccelSen', 'RepairTime', 'RecoveryTime', 'FunctionLossTime'
        OutputDir (Path): 保存输出文件的目录。默认为 'results/RegionalLossSim/'
        ChunkSize (int): 每个chunk中要处理的建筑数量。默认为10000
        n_processes (int): 使用的进程数。如果为None，则使用4个
    
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
    
    # Get the full path of the OutputDir with CityName subdirectory
    OutputDir = Path(OutputDir).resolve() / CityName
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
                chunk_results[idx_i] = {'data': sim_loss_cat.codes, 'dtype': 'int8', 'compression': 'gzip', 'compression_opts': 9, 'shuffle': True}
            else:
                chunk_results[idx_i] = {'data': sim_loss[i], 'dtype': 'float32', 'compression': 'gzip', 'compression_opts': 9, 'shuffle': True}
    
    # 使用传入的锁进行多进程安全写入
    with lock:
        try:
            with h5py.File(output_file, 'a') as f:
                # 写入数据
                for bld_idx, bld_data in chunk_results.items():
                    dataset_name = f'BldID_{bld_idx}'
                    if dataset_name not in f:  # 避免重复写入
                        f.create_dataset(dataset_name, data=bld_data['data'], dtype=bld_data['dtype'], compression=bld_data['compression'], compression_opts=bld_data['compression_opts'], shuffle=bld_data['shuffle'])
        except Exception as e:
            print(f"Warning: Failed to write chunk results: {e}")
            print(f"Chunk processed {len(chunk_results)} buildings but could not write to file")
    
    return len(chunk_results)



def reorganize_BldLoss_by_rupture(CityName: str, losstype: str, InputDir: Path = RESULTDIR, OutputDir: Path = RESULTDIR, chunk_size: int = 10000) -> None:
    '''
    重新组织已生成的BldLoss_{losstype}.hdf5文件，按震源（i_rup）分组存储数据
    使用流式处理避免大文件的内存溢出问题
    
    原始文件结构：
        BldID_0: [sim1_loss, sim2_loss, ...]  # 建筑0的所有模拟损失
        BldID_1: [sim1_loss, sim2_loss, ...]  # 建筑1的所有模拟损失
        ...
    
    重组后的文件结构：
        group_rup_0/
            loss_matrix: (N_bld x N_sim_rup_0)  # 震源0的损失矩阵
            metadata: {rupid, rup_info, iSim_indices}
        group_rup_1/
            loss_matrix: (N_bld x N_sim_rup_1)  # 震源1的损失矩阵
            metadata: {rupid, rup_info, iSim_indices}
        ...
    
    Parameters:
        CityName (str): 城市名称
        losstype (str): 损失类型，必须与原始文件一致
        InputDir (Path): 包含原始BldLoss文件的目录，默认为RESULTDIR
        OutputDir (Path): 输出重组文件的目录，默认为RESULTDIR
        chunk_size (int): 每次处理的建筑数量，用于控制内存使用，默认为10000
    
    Results:
        BldLoss_{losstype}_ByRupture.hdf5: 按震源重组的损失文件
            group_rup_{i}: 震源i的数据组
                loss_matrix: (N_bld x N_sim) 损失矩阵，行为建筑，列为该震源的模拟
                rupid: 震源ID字符串
                iSim_indices: 该震源在全局模拟中的索引列表
                rup_info: 震源的详细信息（震级、位置等）
    '''
    logger.info(f"Starting BldLoss file reorganization by rupture for {CityName}, loss type: {losstype}")
    
    # 检查损失类型是否有效
    if losstype not in BldPSPA_ReprBld.LossTypes1 + BldPSPA_ReprBld.LossTypes2:
        error_msg = f"losstype must be one of {BldPSPA_ReprBld.LossTypes1 + BldPSPA_ReprBld.LossTypes2}."
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # 设置输入和输出目录
    InputDir = Path(InputDir).resolve()
    OutputDir = Path(OutputDir).resolve()
    if not OutputDir.exists():
        OutputDir.mkdir(parents=True)
        logger.info(f"Created output directory: {OutputDir}")
    
    # 原始文件和输出文件路径
    input_file = InputDir / CityName / f'BldLoss_{losstype}.hdf5'
    output_file = OutputDir / CityName / f'BldLoss_{losstype}_ByRupture.hdf5'
    
    # 检查输入文件是否存在
    if not input_file.exists():
        error_msg = f"Input file not found: {input_file}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    logger.info(f"Reading from: {input_file}")
    logger.info(f"Writing to: {output_file}")
    
    # 确保输出目录存在
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 获取震源信息和映射
    rup_df = EQsources.ReadCityRuptures(CityName)
    rupid_2_iSim = IMSim.get_rupid_2_iSim(CityName)
    Bld_gdf = Utilities.get_CityBldData(CityName)
    N_bld = len(Bld_gdf)
    
    logger.info(f"Found {len(rup_df)} ruptures and {N_bld} buildings")
    
    # 采用流式处理
    logger.info("Using streaming processing to avoid memory overflow for large files")
    
    # 创建输出文件并按震源重组数据
    with h5py.File(input_file, 'r') as f_in, h5py.File(output_file, 'w') as f_out:
        # 添加文件级别的元数据
        f_out.attrs['CityName'] = CityName
        f_out.attrs['losstype'] = losstype
        f_out.attrs['created'] = pd.Timestamp.now().isoformat()
        f_out.attrs['total_buildings'] = N_bld
        f_out.attrs['total_ruptures'] = len(rup_df)
        f_out.attrs['original_file'] = str(input_file)
        
        # 先创建所有震源的组和数据集
        logger.info("Creating groups and datasets for each rupture...")
        iSim_mappings = {}
        
        with tqdm(total=len(rup_df), desc="Creating...", unit="rupture") as pbar:
            for i_rup in range(len(rup_df)):
                rup_data = rup_df.iloc[i_rup]
                rupid = rup_data['rup_id']
            
                # 获取该震源的模拟索引
                iSim_list = rupid_2_iSim[rupid]['iSim']
                N_sim_rup = len(iSim_list)
                iSim_mappings[i_rup] = iSim_list
                
                # 创建该震源的组
                rup_group = f_out.create_group(f'group_rup_{i_rup}')
                
                # 预先创建损失矩阵数据集
                if losstype in BldPSPA_ReprBld.LossTypes1:
                    # 损伤状态类型使用int8
                    rup_group.create_dataset(
                        'loss_matrix', 
                        shape=(N_bld, N_sim_rup),
                        dtype='int8',
                        compression='gzip',
                        compression_opts=9,
                        shuffle=True,
                        chunks=True
                    )
                else:
                    # 连续损失类型使用float32
                    rup_group.create_dataset(
                        'loss_matrix', 
                        shape=(N_bld, N_sim_rup),
                        dtype='float32',
                        compression='gzip',
                        compression_opts=9,
                        shuffle=True,
                        chunks=True
                    )
                
                # 保存震源元数据
                rup_group.attrs['rupid'] = str(rupid)
                rup_group.attrs['i_rup'] = i_rup
                rup_group.attrs['N_sim'] = N_sim_rup
                
                # 保存模拟索引
                rup_group.create_dataset('iSim_indices', data=iSim_list, compression='gzip')
                
                # 保存震源详细信息
                rup_info_group = rup_group.create_group('rup_info')
                for col in rup_df.columns:
                    if col in rup_data and pd.notna(rup_data[col]):
                        value = rup_data[col]
                        if isinstance(value, (int, float, str)):
                            rup_info_group.attrs[col] = value
                        else:
                            rup_info_group.attrs[col] = str(value)
                
                # 更新进度条
                pbar.update(1)
        
        # 分块处理建筑数据以避免内存溢出
        logger.info(f"Processing {N_bld} buildings in chunks of {chunk_size}...")
        n_chunks = (N_bld + chunk_size - 1) // chunk_size
        
        # 使用多进程处理
        n_processes = min(mp.cpu_count(), n_chunks, 4)  # 限制最大进程数
        logger.info(f"Using multiprocessing with {n_processes} processes for {n_chunks} chunks")
        
        # 准备chunk任务参数
        chunk_tasks = []
        for chunk_start in range(0, N_bld, chunk_size):
            chunk_end = min(chunk_start + chunk_size, N_bld)
            chunk_tasks.append((
                str(input_file), chunk_start, chunk_end, 
                iSim_mappings, losstype
            ))
        
        # 使用进程池处理chunks，流式写入避免内存溢出
        with mp.Pool(processes=n_processes) as pool:
            with tqdm(total=len(chunk_tasks), desc="Processing building chunks (multiprocessing)", unit="chunk") as pbar:
                for result in pool.imap(_process_reorganize_chunk_worker, chunk_tasks):
                    chunk_start, chunk_end, chunk_data = result
                    
                    # 直接通过新的文件句柄写入，避免保持dataset引用占用内存
                    with h5py.File(output_file, 'a') as f_write:
                        for i_rup, chunk_matrix in chunk_data.items():
                            dataset_path = f'group_rup_{i_rup}/loss_matrix'
                            f_write[dataset_path][chunk_start:chunk_end, :] = chunk_matrix
                    
                    # 手动释放内存
                    del chunk_data
                    pbar.update(1)
    
    # 验证重组结果
    logger.info("Verifying reorganized data...")
    with h5py.File(output_file, 'r') as f:
        total_groups = len([key for key in f.keys() if key.startswith('group_rup_')])
        logger.info(f"Created {total_groups} rupture groups")
        
        # 检查第一个组的数据结构
        if 'group_rup_0' in f:
            first_group = f['group_rup_0']
            loss_matrix_shape = first_group['loss_matrix'].shape
            logger.info(f"Example loss matrix shape: {loss_matrix_shape} (buildings x simulations)")
    
    logger.info(f"BldLoss file reorganization completed successfully. Output saved to: {output_file}")
    
    return str(output_file)

def _process_reorganize_chunk_worker(args):
    """
    多进程工作函数，用于重组BldLoss文件的单个chunk
    
    Parameters:
        args: 包含以下元素的元组
            - input_file_path: 输入HDF5文件路径
            - chunk_start: chunk开始索引
            - chunk_end: chunk结束索引
            - iSim_mappings: 震源到模拟索引的映射
            - losstype: 损失类型
    
    Returns:
        tuple: (chunk_start, chunk_end, chunk_data)
            chunk_data: 字典，键为i_rup，值为该震源的损失矩阵
    """
    input_file_path, chunk_start, chunk_end, iSim_mappings, losstype = args
    
    current_chunk_size = chunk_end - chunk_start
    
    # 为当前chunk的所有震源准备数据
    chunk_data = {}
    
    # 在子进程中打开输入文件
    with h5py.File(input_file_path, 'r') as f_in:
        # 读取当前chunk的建筑数据
        for bld_idx in range(chunk_start, chunk_end):
            bld_key = f'BldID_{bld_idx}'
            if bld_key in f_in:
                bld_losses = f_in[bld_key][:]
                
                # 为每个震源提取对应的模拟结果
                for i_rup, iSim_list in iSim_mappings.items():
                    if i_rup not in chunk_data:
                        N_sim_rup = len(iSim_list)
                        if losstype in BldPSPA_ReprBld.LossTypes1:
                            chunk_data[i_rup] = np.zeros((current_chunk_size, N_sim_rup), dtype='int8')
                        else:
                            chunk_data[i_rup] = np.zeros((current_chunk_size, N_sim_rup), dtype='float32')
                    
                    rup_losses = bld_losses[iSim_list]
                    chunk_data[i_rup][bld_idx - chunk_start, :] = rup_losses
            else:
                # 如果缺失数据，chunk_data中对应位置会保持为0（已初始化）
                pass
    
    return chunk_start, chunk_end, chunk_data


def load_BldLoss_by_rupture(CityName: str, losstype: str, i_rup: Union[int, list], InputDir: Path = RESULTDIR) -> Union[np.ndarray, dict]:
    '''
    从按震源重组的BldLoss文件中加载指定震源的损失数据
    
    Parameters:
        CityName (str): 城市名称
        losstype (str): 损失类型
        i_rup (Union[int, list]): 震源索引，可以是单个整数或整数列表
        InputDir (Path): 包含重组文件的目录，默认为RESULTDIR
    
    Returns:
        如果i_rup是int: np.ndarray, shape为(N_bld, N_sim_rup)的损失矩阵
        如果i_rup是list: dict, 键为震源索引，值为对应的损失矩阵
    '''
    InputDir = Path(InputDir).resolve()
    input_file = InputDir / CityName / f'BldLoss_{losstype}_ByRupture.hdf5'
    
    if not input_file.exists():
        error_msg = f"Reorganized file not found: {input_file}. Please run reorganize_BldLoss_by_rupture() first."
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    is_single_rupture = isinstance(i_rup, int)
    i_rup_list = [i_rup] if is_single_rupture else i_rup
    
    results = {}
    
    with h5py.File(input_file, 'r') as f:
        for i_rup_single in i_rup_list:
            group_name = f'group_rup_{i_rup_single}'
            if group_name not in f:
                raise ValueError(f"Rupture group {group_name} not found in file")
            
            rup_group = f[group_name]
            loss_matrix = rup_group['loss_matrix'][:]
            results[i_rup_single] = loss_matrix
            
            logger.info(f"Loaded loss matrix for rupture {i_rup_single}: shape {loss_matrix.shape}")
    
    return results[i_rup] if is_single_rupture else results


def _get_total_loss_sim_by_rupture(CityName: str, ResultsDir: Path = RESULTDIR, losstype: str = 'DS_Struct', i_rup: Union[int, list] = 0, n_processes: int = None) -> np.ndarray:
    '''
    基于按震源重组的损失文件计算所有模拟的区域总损失分布（带缓存多进程版本）
    读取BldLoss_{losstype}_ByRupture.hdf5文件格式
    
    Parameters:
        CityName (str): 城市名称
        losstype (str): 损失类型，如 'DS_Struct', 'RepairCost_Total' 等
        i_rup (Union[int, list]): 震源索引，可以是单个整数或整数列表。默认为0
        ResultsDir (Path): 结果目录，默认为RESULTDIR
        n_processes (int): 使用的进程数。如果为None，则使用CPU核心数（最大4个）
        
    Returns:
        total_loss_sim (np.ndarray): 所有模拟的损失分布
            - 如果i_rup是int: (N_DS x N_sim) 
            - 如果i_rup是list: (N_DS x N_sim_per_rup x N_rup)
            其中 N_DS 可以通过 total_loss_sim.shape[0] 获取
    '''
    
    # 创建缓存文件路径
    ResultsDir = Path(ResultsDir).resolve()
    temp_dir = ResultsDir / 'temp'
    temp_dir.mkdir(exist_ok=True, parents=True)

    # 创建缓存键（基于参数生成唯一标识）
    cache_key_data = {
        'CityName': CityName,
        'losstype': losstype,
        'i_rup': i_rup,
        'function': '_get_total_loss_sim_by_rupture'
    }
    cache_key = hashlib.md5(str(cache_key_data).encode()).hexdigest()
    cache_file = temp_dir / f'total_loss_sim_by_rupture_{cache_key}.pkl'
    
    # 确定源文件路径
    input_file = ResultsDir / CityName / f'BldLoss_{losstype}_ByRupture.hdf5'
    
    # 检查源文件是否存在
    if not input_file.exists():
        error_msg = f"Reorganized file not found: {input_file}. Please run reorganize_BldLoss_by_rupture() first."
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    # 检查缓存是否有效
    cache_valid = False
    if cache_file.exists():
        try:
            cache_mtime = cache_file.stat().st_mtime
            source_mtime = input_file.stat().st_mtime
            if cache_mtime > source_mtime:
                cache_valid = True
                logger.info(f'Found valid cache file: {cache_file}')
        except Exception as e:
            logger.warning(f'Error checking cache validity: {e}')
    
    # 如果缓存有效，直接读取返回
    if cache_valid:
        try:
            with open(cache_file, 'rb') as f:
                cached_result = pickle.load(f)
            logger.info(f'Loaded result from cache: {cache_file}')
            return cached_result
        except Exception as e:
            logger.warning(f'Error loading cache file ({e}). Recalculating...')
    
    # 缓存无效或不存在，重新计算
    logger.info(f'Cache not found or invalid. Calculating total loss distribution...')
    
    # 执行原始计算逻辑
    total_loss_sim = _calculate_total_loss_sim_by_rupture_core(
        CityName, ResultsDir, losstype, i_rup, n_processes
    )
    
    # 保存结果到缓存
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(total_loss_sim, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f'Saved result to cache: {cache_file}')
    except Exception as e:
        logger.warning(f'Error saving to cache ({e}). Continuing without cache.')
    
    return total_loss_sim


def _calculate_total_loss_sim_by_rupture_core(CityName: str, ResultsDir: Path, losstype: str, i_rup: Union[int, list], n_processes: int) -> np.ndarray:
    '''
    按震源重组的损失文件计算总损失分布的核心逻辑
    '''
    # 标准化 i_rup 为列表并记录日志
    is_single_rupture = isinstance(i_rup, int)
    if is_single_rupture:
        i_rup_list = [i_rup]
        logger.info(f'Calculating total loss distribution for {CityName}, rupture {i_rup}, loss type: {losstype}')
    else:
        i_rup_list = i_rup
        logger.info(f'Calculating total loss distribution for {CityName}, ruptures {i_rup}, loss type: {losstype}')
    
    # 检查损失类型是否有效
    if losstype not in BldPSPA_ReprBld.LossTypes1 + BldPSPA_ReprBld.LossTypes2:
        error_msg = f"losstype must be one of {BldPSPA_ReprBld.LossTypes1 + BldPSPA_ReprBld.LossTypes2}."
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # 确定损失状态数量
    if losstype in BldPSPA_ReprBld.LossTypes1:
        N_DS = len(BldPSPA_ReprBld.DS_types)
    elif losstype in BldPSPA_ReprBld.LossTypes2:
        N_DS = 1
    else:
        raise ValueError(f"losstype must be one of {BldPSPA_ReprBld.LossTypes1 + BldPSPA_ReprBld.LossTypes2}.")
    
    ResultsDir = Path(ResultsDir).resolve()
    input_file = ResultsDir / CityName / f'BldLoss_{losstype}_ByRupture.hdf5'
    
    # 设置进程数
    if n_processes is None:
        n_processes = min(mp.cpu_count(), len(i_rup_list), 4)  # 限制最大进程数避免过度消耗资源
    
    # 根据震源数量决定是否使用多进程
    if len(i_rup_list) > 1 and n_processes > 1:
        logger.info(f'Using multiprocessing with {n_processes} processes for {len(i_rup_list)} ruptures')
        
        # 准备多进程任务
        rupture_tasks = []
        for i_rup_single in i_rup_list:
            rupture_tasks.append((i_rup_single, str(input_file), losstype, N_DS))
        
        with mp.Pool(processes=n_processes) as pool:
            # 使用imap显示进度
            rupture_results_with_idx = []
            with tqdm(total=len(rupture_tasks), desc=f"Calculating losses for {len(rupture_tasks)} ruptures", unit="rupture") as pbar:
                for result in pool.imap(_calculate_rupture_loss_by_rupture, rupture_tasks):
                    rupture_results_with_idx.append(result)
                    pbar.update(1)
        
        # 按震源索引排序结果
        rupture_results_with_idx.sort(key=lambda x: x[0])
        rupture_results = [result for _, result in rupture_results_with_idx]
    else:
        # 单个震源或进程数为1时直接计算，避免多进程开销
        logger.info(f'Single rupture or single process, using direct calculation')
        rupture_results = []
        for i_rup_single in i_rup_list:
            _, result = _calculate_rupture_loss_by_rupture((i_rup_single, str(input_file), losstype, N_DS))
            rupture_results.append(result)
    
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
    
    logger.info(f'Total loss simulation completed. Result shape: {total_loss_sim.shape}')
    return total_loss_sim

def _calculate_rupture_loss_by_rupture(args):
    """基于按震源重组文件的震源损失计算函数（多进程工作函数）"""
    i_rup_single, input_file_path, losstype, N_DS = args
    
    with h5py.File(input_file_path, 'r') as f:
        group_name = f'group_rup_{i_rup_single}'
        if group_name not in f:
            raise ValueError(f"Rupture group {group_name} not found in file {input_file_path}")
        
        rup_group = f[group_name]
        # 读取损失矩阵：shape为(N_bld, N_sim)
        loss_matrix = rup_group['loss_matrix'][:]
    
    N_bld, N_sim = loss_matrix.shape
    total_loss_single = np.zeros((N_DS, N_sim))
    
    if losstype in BldPSPA_ReprBld.LossTypes1:
        # 对于损伤状态类型，统计每个损伤状态的建筑数量
        for j in range(N_sim):
            loss_sim_column = loss_matrix[:, j]
            # 统计每个损伤状态的建筑数量
            for ds in range(N_DS):
                total_loss_single[ds, j] = np.sum(loss_sim_column == ds)
    elif losstype in BldPSPA_ReprBld.LossTypes2:
        # 对于连续损失类型，直接求和
        total_loss_single[0, :] = np.sum(loss_matrix, axis=0)
    
    return i_rup_single, total_loss_single

def get_total_loss_sim(CityName: str, ResultsDir: str = RESULTDIR, LossType: str = 'DS_Struct', i_rup: Union[int, list] = 0) -> np.ndarray:
    '''
    计算所有模拟的区域总损失分布
    
    Parameters:
        CityName (str): 城市名称
        ResultsDir (str): 结果目录
        LossType (str): 损失类型
        i_rup (Union[int, list]): 震源索引，可以是单个整数或整数列表
        
    Returns:
        np.ndarray: 总损失分布
            - 如果i_rup是int: (N_DS x N_sim) 
            - 如果i_rup是list: (N_DS x N_sim_per_rup x N_rup)
    '''
    
    ResultsDir_path = Path(ResultsDir)
    
    # 检查ByRupture文件是否存在
    by_rupture_file = ResultsDir_path / CityName / f'BldLoss_{LossType}_ByRupture.hdf5'
    if not by_rupture_file.exists():
        logger.error(f'ByRupture file not found: {by_rupture_file}. Please run reorganize_BldLoss_by_rupture() first.')
        raise FileNotFoundError(f'ByRupture file not found: {by_rupture_file}. Please run reorganize_BldLoss_by_rupture() first.')
    
    # 使用按震源重组的文件计算
    total_loss_sim = _get_total_loss_sim_by_rupture(
        CityName=CityName,
        ResultsDir=ResultsDir_path,
        losstype=LossType,
        i_rup=i_rup,
        n_processes=None  # 使用默认进程数
    )
    
    return total_loss_sim



def _calculate_rupture_grouped_loss_worker(args):
    """Worker function to calculate grouped losses for a single rupture."""
    i_rup, CityName, ResultsDir, indices_dict = args
    ResultsDir = Path(ResultsDir)
    
    # Define file paths
    file_cost = ResultsDir / CityName / 'BldLoss_RepairCost_Total_ByRupture.hdf5'
    file_ds = ResultsDir / CityName / 'BldLoss_DS_Struct_ByRupture.hdf5'
    
    results = {}
    
    try:
        with h5py.File(file_cost, 'r') as f_cost, h5py.File(file_ds, 'r') as f_ds:
            grp_name = f'group_rup_{i_rup}'
            
            if grp_name not in f_cost or grp_name not in f_ds:
                return i_rup, None

            # Read data
            cost_matrix = f_cost[grp_name]['loss_matrix'][:]
            ds_matrix = f_ds[grp_name]['loss_matrix'][:]
            
            # Determine max DS index (Collapse state)
            max_ds_index = len(BldPSPA_ReprBld.DS_types) - 1
            
            for group_name, indices in indices_dict.items():
                group_cost = cost_matrix[indices, :]
                group_ds = ds_matrix[indices, :]
                
                # Calculate Total Loss
                total_loss = np.sum(group_cost, axis=0) # Shape: (N_sim,)
                
                # Calculate Collapse Count (DS == max_ds_index)
                collapse_count = np.sum(group_ds == max_ds_index, axis=0) # Shape: (N_sim,)
                
                results[group_name] = {
                    'TotalLoss': total_loss,
                    'CollapseCount': collapse_count
                }
                
    except Exception as e:
        logger.error(f"Error in worker for rupture {i_rup}: {e}")
        return i_rup, None
        
    return i_rup, results

def _calculate_grouped_loss_sim_core(CityName, ResultsDir, i_rup_list, indices_dict, n_processes=None):
    """
    Core function to calculate grouped losses for multiple ruptures.
    """
    
    if n_processes is None:
        n_processes = min(mp.cpu_count(), len(i_rup_list), 4)
        
    tasks = []
    for i_rup in i_rup_list:
        tasks.append((i_rup, CityName, str(ResultsDir), indices_dict))
        
    results_all = {}
    
    if len(tasks) > 1 and n_processes > 1:
        logger.info(f"Calculating district-grouped losses for {len(tasks)} ruptures using {n_processes} processes")
        with mp.Pool(processes=n_processes) as pool:
            with tqdm(total=len(tasks), desc="Processing Ruptures") as pbar:
                for i_rup, res in pool.imap_unordered(_calculate_rupture_grouped_loss_worker, tasks):
                    if res is not None:
                        results_all[i_rup] = res
                    pbar.update(1)
    else:
        logger.info(f"Calculating district-grouped losses for {len(tasks)} ruptures sequentially")
        for task in tqdm(tasks, desc="Processing Ruptures"):
            i_rup, res = _calculate_rupture_grouped_loss_worker(task)
            if res is not None:
                results_all[i_rup] = res
                
    return results_all

def get_total_loss_and_collapse_ratio_sim(CityName: str, ResultsDir: str = RESULTDIR) -> pd.DataFrame:
    '''
    计算所有模拟的区域总损失比率

    Returns:
        df_ratio_sim (pd.DataFrame): DataFrame containing total loss ratios for each simulation
            - Columns: ['CityName', 'DistrictName', 'TotalLossRatio', 'CollapseRatio']. TotalLossRatio and CollapseRatio are numpy arrays of shape (N_sim,N_rup)
    
    Resulting file saved at:
        ResultsDir / CityName / 'TotalLossAndCollapseRatio_Sim.csv'
    '''
    logger.info(f"Calculating total loss and collapse ratio for {CityName}")
    ResultsDir = Path(ResultsDir)
    
    # 1. Get City Data
    df_bld = Utilities.get_CityBldData(CityName, ['DistrictName', 'ReprBldID_PSPA'])
    districts = sorted(df_bld['DistrictName'].unique())
    
    # 2. Calculate Reconstruction Cost
    logger.info("Calculating reconstruction costs for buildings...")
    unique_repr_ids = [int(x) for x in df_bld['ReprBldID_PSPA'].unique() if not pd.isna(x)]
    cost_time_dict = BldPSPA_ReprBld.get_reconstruction_cost_and_time(unique_repr_ids)
    
    # Map cost
    df_bld['ReconstructionCost'] = df_bld['ReprBldID_PSPA'].map(lambda x: cost_time_dict.get(x, (0,0))[0])
    
    total_cost_district = df_bld.groupby('DistrictName')['ReconstructionCost'].sum()
    total_cost_city = df_bld['ReconstructionCost'].sum()
    
    count_bld_district = df_bld.groupby('DistrictName').size()
    count_bld_city = len(df_bld)
    
    # 3. Prepare Indices for Groups
    indices_dict = {}
    for district in districts:
        indices_dict[district] = df_bld[df_bld['DistrictName'] == district].index.values
    indices_dict['All'] = df_bld.index.values
    
    # 4. Get Ruptures
    ruptures = EQsources.ReadCityRuptures(CityName)
    i_rup_list = list(range(len(ruptures)))
    N_rup = len(ruptures)
    
    # 5. Calculate Losses
    # results_all: Dict[i_rup, Dict[DistrictName, Dict['TotalLoss', 'CollapseCount']]]
    logger.info("Calculating district-grouped losses for each rupture...")
    results_all = _calculate_grouped_loss_sim_core(CityName, ResultsDir, i_rup_list, indices_dict)
    
    if not results_all:
        logger.error("No results calculated.")
        return pd.DataFrame()
        
    # Check N_sim from first result
    first_res = next(iter(results_all.values()))
    first_group = next(iter(first_res.values()))
    N_sim = len(first_group['TotalLoss'])
    
    # 6. Aggregate Results into Arrays (N_sim, N_rup)
    data_arrays = {}
    groups = list(indices_dict.keys())
    
    for group in groups:
        data_arrays[group] = {
            'TotalLoss': np.zeros((N_sim, N_rup)),
            'CollapseCount': np.zeros((N_sim, N_rup))
        }
    
    logger.info("Aggregating results into arrays...")
    for i_rup in i_rup_list:
        if i_rup in results_all:
            res_rup = results_all[i_rup]
            for district in groups:
                if district in res_rup:
                    if len(res_rup[district]['TotalLoss']) == N_sim:
                        data_arrays[district]['TotalLoss'][:, i_rup] = res_rup[district]['TotalLoss']
                        data_arrays[district]['CollapseCount'][:, i_rup] = res_rup[district]['CollapseCount']
                        
    # 7. Calculate Ratios and Build DataFrame
    rows = []
    
    for district in groups:
        if district == 'All':
            cost = total_cost_city
            count = count_bld_city
        else:
            cost = total_cost_district.get(district, 0)
            count = count_bld_district.get(district, 0)
            
        loss_array = data_arrays[district]['TotalLoss']
        collapse_array = data_arrays[district]['CollapseCount']
        
        loss_ratio = loss_array / cost if cost > 0 else np.zeros_like(loss_array)
        collapse_ratio = collapse_array / count if count > 0 else np.zeros_like(collapse_array)
        
        rows.append({
            'CityName': CityName,
            'DistrictName': district,
            'TotalLossRatio': loss_ratio,
            'CollapseRatio': collapse_ratio
        })
    
    df_ratio_sim = pd.DataFrame(rows)
    
    # 8. Save to CSV
    output_file = ResultsDir / CityName / 'TotalLossAndCollapseRatio_Sim.csv'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    df_ratio_sim.to_csv(output_file, index=False, encoding='utf-8-sig')
    logger.info(f"Saved TotalLossAndCollapseRatio_Sim to {output_file}")
    
    return df_ratio_sim

def get_statistics_regional_losses(CityName: str, ResultsDir: str = RESULTDIR, LossType: str = 'DS_Struct', i_rup: Union[int, list] = 0, iSim: int = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Get the statistics of the regional seismic losses, given an earthquake event.

    Parameters:
        CityName (str): Name of the city.
        ResultsDir (str): Directory containing the loss results.
        LossType (str): Type of loss to plot. Default is 'DS_Struct' (structural damage state).
        i_rup (Union[int, list]): Index of the rupture or list of rupture indices. Default is 0.
        iSim (int): Index of the simulation for a specific rupture. Default is None.

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
        CityName, ResultsDir, LossType, i_rup
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

def create_geojson_regional_losses(CityName, DistrictName=None, ResultsDir=RESULTDIR, LossType='DS_Struct', i_rup=0, iSim=0):
    """
    根据已经计算的结果，生成区域损失的GeoJSON数据
    
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
    
    # 读取震源信息
    rup_df = EQsources.ReadCityRuptures(CityName)
    
    # 创建结果GeoDataFrame
    result_gdf = Bld_gdf.copy()
    
    # 检查ByRupture文件是否存在
    by_rupture_file = ResultsDir / CityName / f'BldLoss_{LossType}_ByRupture.hdf5'
    
    if by_rupture_file.exists():
        
        with h5py.File(by_rupture_file, 'r') as f:
            if i_rup is None:
                # 处理所有rupture - 从所有震源读取指定模拟的数据
                # 初始化损失值矩阵：[建筑数量 x 震源数量]
                n_buildings = len(result_gdf)
                n_ruptures = len(rup_df)
                # 根据损失类型选择合适的数据类型
                if LossType in BldPSPA_ReprBld.LossTypes1:
                    loss_matrix_all = np.zeros((n_buildings, n_ruptures), dtype=int)
                else:
                    loss_matrix_all = np.zeros((n_buildings, n_ruptures), dtype=float)
                
                # 先遍历震源，再遍历建筑
                for rup_idx in tqdm(range(n_ruptures), desc="Loading loss data"):
                    group_name = f'group_rup_{rup_idx}'
                    if group_name in f:
                        loss_matrix = f[group_name]['loss_matrix']
                        # 一次性读取所有建筑在当前震源和指定模拟中的损失值
                        if LossType in BldPSPA_ReprBld.LossTypes1:
                            # 对于损伤状态类型，转换为整数
                            loss_matrix_all[:, rup_idx] = loss_matrix[:, iSim].astype(int)
                        else:
                            # 对于其他类型（如经济损失），保持原始数值
                            loss_matrix_all[:, rup_idx] = loss_matrix[:, iSim]
                    # 如果震源组不存在，默认值已经是0，无需额外处理
                
                # 将损失值矩阵转换为每个建筑的损失列表
                loss_values_all_ruptures = [loss_matrix_all[i].tolist() for i in range(n_buildings)]
                result_gdf['loss_value'] = loss_values_all_ruptures
            else:
                # 处理单个rupture - 直接从指定震源组读取
                group_name = f'group_rup_{i_rup}'
                if group_name in f:
                    loss_matrix = f[group_name]['loss_matrix']
                    # 直接读取指定震源的所有建筑在指定模拟中的损失值
                    loss_values = loss_matrix[:, iSim].tolist()
                    # 转换为整数（对于损伤状态类型）
                    if LossType in BldPSPA_ReprBld.LossTypes1:
                        loss_values = [int(val) for val in loss_values]
                    
                    result_gdf['loss_value'] = loss_values
                else:
                    logger.warning(f'Rupture group {group_name} not found, using zeros')
                    result_gdf['loss_value'] = [0] * len(result_gdf)
    else:
        # 如果ByRupture文件不存在，报错
        logger.error(f'ByRupture file not found: {by_rupture_file}. Please run reorganize_BldLoss_by_rupture() first.')
        raise FileNotFoundError(f'ByRupture file not found: {by_rupture_file}. Please run reorganize_BldLoss_by_rupture() first.')
        
    
    # 只保留必要的列
    keep_columns = ['geometry', 'loss_value', 'structure', 'story', 'DistrictName']
    result_gdf = result_gdf[keep_columns]
    
    # 转换为GeoJSON
    geojson_data = result_gdf.to_json()
    
    return geojson_data


def create_statistics_regional_losses(CityName: str, ResultsDir: str = RESULTDIR, OutputDir: str = None, LossType: str = 'DS_Struct', iSim: int = 0) -> str:
    '''
    根据已经计算的结果，生成某城市所有地震事件的地震损失统计信息JSON文件
    
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
    
    # 检查ByRupture文件是否存在，必须使用更高效的文件
    by_rupture_file = ResultsDir / CityName / f'BldLoss_{LossType}_ByRupture.hdf5'
    
    if not by_rupture_file.exists():
        # ByRupture文件不存在，直接报错退出
        error_msg = f'ByRupture file not found: {by_rupture_file}. Please run reorganize_BldLoss_by_rupture() first to create the optimized file.'
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
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
        iSim=iSim
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




