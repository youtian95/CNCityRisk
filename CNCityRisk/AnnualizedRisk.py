from . import Utilities
from . import EQsources
from . import RegionalLossSim
from . import Logger
from . import BldPSPA_ReprBld

from pathlib import Path
import numpy as np
import pandas as pd

# Initialize logger for this module
logger = Logger.get_logger('AnnualizedRisk')

RESULTDIR = Path(__file__).resolve().parent / 'results/AnnualizedRisk'


def calculate_annualized_loss(CityName, OutputDir=RESULTDIR, LossType: tuple = ('RepairCost_Total', 'RepairTime')):
    logger.info(f"Calculating annualized loss for city: {CityName} with output directory: {OutputDir}")
    _calculate_mean_annual_freq_mag(CityName, OutputDir)
    for loss_type in LossType:
        # 优化：预先计算一次get_total_loss_sim，避免重复计算
        logger.info(f"Pre-calculating total loss simulation for {CityName}, loss type: {loss_type}")
        precomputed_data = _precompute_total_loss_data(CityName, LossType=loss_type)
        _calculate_mean_annual_freq_loss(CityName, OutputDir, LossType=loss_type, precomputed_data=precomputed_data)
        _calculate_mean_annual_loss_with_year(CityName, OutputDir, LossType=loss_type, precomputed_data=precomputed_data)
    logger.info(f"Annualized loss calculation completed for city: {CityName}. Results saved in: {OutputDir}")

def _calculate_mean_annual_freq_mag(CityName, OutputDir=RESULTDIR):
    """
    Calculate the mean annual frequency of exceeding a certain magnitude for a given city.

    Writes a CSV file:
        pd.DataFrame: A DataFrame containing the annualized frequency of exceeding each magnitude. Two columns: 'Magnitude' and 'Frequency'.
    """
    # Create city-specific output directory
    city_output_dir = OutputDir / CityName
    city_output_dir.mkdir(parents=True, exist_ok=True)
    
    freq_mag_df = EQsources.get_mean_annual_freq_mag(CityName)
    if freq_mag_df is not None:
        freq_mag_df.to_csv(city_output_dir / "mean_annual_freq_mag.csv", index=False)
    else:
        logger.error(f"Failed to retrieve mean annual frequency of magnitude for city: {CityName}")

def _precompute_total_loss_data(CityName, LossType: str = 'RepairCost_Total'):
    """
    预计算总损失数据，避免在后续函数中重复计算get_total_loss_sim
    
    Parameters:
        CityName (str): 城市名称
        LossType (str): 损失类型，必须是 BldPSPA_ReprBld.LossTypes2 中的一个
        
    Returns:
        dict: 包含预计算数据的字典，包含以下键：
            - 'total_loss_mean': 总损失均值. shape: (1 x N_rup) 其中 N_rup 是震源数量
            - 'rup_multiplicity': 震源倍数. shape: (N_rup,)
            - 'rup_df': 震源数据框
            - 'years': 持续年数
    """
    # 检查损失类型
    if LossType not in BldPSPA_ReprBld.LossTypes2:
        raise ValueError(f"Invalid LossType: {LossType}. Must be one of {BldPSPA_ReprBld.LossTypes2}")
    
    logger.info(f"Precomputing total loss data for city: {CityName}, loss type: {LossType}")
    
    # 加载损失数据到内存
    loss_data = RegionalLossSim.load_loss_data_to_memory(CityName, LossType=LossType)
    
    # 读取震源数据
    rup_df = EQsources.ReadCityRuptures(CityName)
    i_rup_list = [i_rup for i_rup in range(len(rup_df))]
    rup_multiplicity = rup_df['multiplicity'].values.astype(float)
    
    # 计算总损失模拟结果（耗时操作，只执行一次）
    total_loss_sim = RegionalLossSim.get_total_loss_sim(CityName, LossType=LossType, i_rup=i_rup_list, LossDataInMemory=loss_data)
    total_loss_mean = total_loss_sim.mean(axis=1)  # 对每个震源的模拟结果求平均
    
    # 获取持续年数
    years = EQsources.get_duration_years(CityName)
    
    logger.info(f"Total loss data precomputed for city: {CityName}")
    
    # 只返回实际需要的数据
    return {
        'total_loss_mean': total_loss_mean,
        'rup_multiplicity': rup_multiplicity,
        'rup_df': rup_df,
        'years': years
    }

def _calculate_mean_annual_freq_loss(CityName, OutputDir=RESULTDIR, LossType: str = 'RepairCost_Total', precomputed_data=None):
    """
    Calculate the mean annual frequency of exceeding a certain loss value for a given city.

    Parameters:
        CityName (str): The name of the city for which to calculate the annualized loss.
        OutputDir (Path): The directory where the results will be saved. Defaults to RESULTDIR.
        LossType (str): The type of loss to analyze, e.g., 'RepairCost_Total', 'RepairTime', etc. LossType can only be BldPSPA_ReprBld.LossTypes2
        precomputed_data (dict): 预计算的数据，必须提供

    Writes a CSV file:
        pd.DataFrame: A DataFrame containing the annualized frequency of exceeding each loss value. Two columns for example: 'RepairCost_Total' and 'Frequency'. 
    """
    # 检查预计算数据
    if precomputed_data is None:
        raise ValueError("precomputed_data is required and cannot be None")
    
    # Create city-specific output directory
    city_output_dir = OutputDir / CityName
    city_output_dir.mkdir(parents=True, exist_ok=True)

    # 使用预计算的数据
    logger.info(f"Using precomputed data for mean annual frequency calculation: {CityName}, {LossType}")
    total_loss_mean = precomputed_data['total_loss_mean']
    rup_multiplicity = precomputed_data['rup_multiplicity']
    years = precomputed_data['years']

    # Calculate the annualized frequency of exceeding each loss value
    # total_loss_mean shape: (1 x N_rup) since LossType is from BldPSPA_ReprBld.LossTypes2
    loss_values = total_loss_mean[0, :]  # shape: (N_rup,) - only one design state
    
    # Get sorted indices to maintain consistency between loss values and multiplicity
    sorted_indices = np.argsort(loss_values)
    lossvalue_sorted = loss_values[sorted_indices]
    lossvalue_multiplicity = rup_multiplicity[sorted_indices]
    
    annualized_freq = []
    for i, loss_value in enumerate(lossvalue_sorted):
        exceedances = lossvalue_multiplicity[i:].sum()  # Count exceedances for this design state
        annualized_freq.append(exceedances / years)  # Annualized frequency
    
    # Save results to CSV
    results_df = pd.DataFrame({
        LossType: lossvalue_sorted,
        'Frequency': annualized_freq
    })
    results_df.to_csv(city_output_dir / f"mean_annual_freq_{LossType}.csv", index=False)

def _calculate_mean_annual_loss_with_year(CityName, OutputDir=RESULTDIR, LossType: str = 'RepairCost_Total', precomputed_data=None):
    """
    Calculate the mean annual loss for a given city with respect to simulated time.

    Parameters:
        CityName (str): The name of the city for which to calculate the annualized loss.
        OutputDir (Path): The directory where the results will be saved. Defaults to RESULTDIR.
        LossType (str): The type of loss to analyze, e.g., 'RepairCost_Total', 'RepairTime', etc. LossType can only be BldPSPA_ReprBld.LossTypes2
        precomputed_data (dict): 预计算的数据，必须提供

    Writes a CSV file:
        pd.DataFrame: A DataFrame containing the mean annual loss, which gradually stabilizes over time. Two columns: 'Year' and 'MeanAnnualLoss'.
    """
    # 检查预计算数据
    if precomputed_data is None:
        raise ValueError("precomputed_data is required and cannot be None")
    
    # Create city-specific output directory
    city_output_dir = OutputDir / CityName
    city_output_dir.mkdir(parents=True, exist_ok=True)

    # 使用预计算的数据
    logger.info(f"Using precomputed data for mean annual loss with year calculation: {CityName}, {LossType}")
    rup_df = precomputed_data['rup_df']
    total_loss_mean = precomputed_data['total_loss_mean']
    
    # read events data
    events_df = EQsources.ReadCityEvents(CityName)
    Year = events_df['year'].values.astype(int)  # 确保年份是整数类型
    event_id = events_df['event_id'].values
    rup_id = events_df['rup_id'].values
    
    # 创建rup_id到i_rup索引的映射
    rup_id_to_i_rup = {rup_df.iloc[i]['rup_id']: i for i in range(len(rup_df))}
    
    # 创建DataFrame并按年份排序
    events_loss_df = pd.DataFrame({
        'Year': Year,
        'event_id': event_id,
        'rup_id': rup_id
    })
    events_loss_df = events_loss_df.sort_values('Year')
    
    # 计算每个年份的平均年损失
    MeanAnnualLoss = []
    Years = []
    
    unique_years = sorted(events_loss_df['Year'].unique())
    
    for year in unique_years:
        # 获取到当前年份为止的所有事件
        events_up_to_year = events_loss_df[events_loss_df['Year'] <= year]
        
        # 计算累积损失
        cumulative_loss = 0
        for _, event_row in events_up_to_year.iterrows():
            rup_id_str = event_row['rup_id']
            if rup_id_str in rup_id_to_i_rup:
                i_rup = rup_id_to_i_rup[rup_id_str]
                # total_loss_mean shape: (N_DS x N_rup), 取第一个设计状态 [0, i_rup]
                loss_value = total_loss_mean[0, i_rup]
                cumulative_loss += loss_value
        
        # 计算平均年损失 = 累积损失 / 年份
        mean_annual_loss = cumulative_loss / year if year > 0 else 0
        
        MeanAnnualLoss.append(mean_annual_loss)
        Years.append(year)
    
    # 保存结果到CSV
    results_df = pd.DataFrame({
        'Year': Years,
        'MeanAnnualLoss': MeanAnnualLoss
    })
    results_df.to_csv(city_output_dir / f"mean_annual_loss_with_year_{LossType}.csv", index=False)