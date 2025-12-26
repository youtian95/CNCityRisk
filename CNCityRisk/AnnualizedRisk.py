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

    _calculate_mean_annual_freq_loss_collapse_ratio(CityName, OutputDir)
    
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
    使用流式处理避免内存溢出问题
    
    Parameters:
        CityName (str): 城市名称
        LossType (str): 损失类型，必须是 BldPSPA_ReprBld.LossTypes2 中的一个
        
    Returns:
        dict: 包含预计算数据的字典，包含以下键：
            - 'total_loss_sim': 总损失. shape: (1 x N_sim_per_rup x N_rup) 其中 N_rup 是震源数量
            - 'rup_multiplicity': 震源倍数. shape: (N_rup,)
            - 'rup_df': 震源数据框
            - 'years': 持续年数
    """
    # 检查损失类型
    if LossType not in BldPSPA_ReprBld.LossTypes2:
        raise ValueError(f"Invalid LossType: {LossType}. Must be one of {BldPSPA_ReprBld.LossTypes2}")
    
    logger.info(f"Precomputing total loss data for city: {CityName}, loss type: {LossType}")
    
    # 读取震源数据
    rup_df = EQsources.ReadCityRuptures(CityName)
    i_rup_list = [i_rup for i_rup in range(len(rup_df))]
    rup_multiplicity = rup_df['multiplicity'].values.astype(float)
    
    # 使用流式处理计算总损失模拟结果，避免内存溢出
    total_loss_sim = RegionalLossSim.get_total_loss_sim(CityName, LossType=LossType, i_rup=i_rup_list)
    
    # 获取持续年数
    years = EQsources.get_duration_years(CityName)
    
    logger.info(f"Total loss data precomputed for city: {CityName}")
    
    # 只返回实际需要的数据
    return {
        'total_loss_sim': total_loss_sim,
        'rup_multiplicity': rup_multiplicity,
        'rup_df': rup_df,
        'years': years
    }

def _calculate_mean_annual_freq_loss_collapse_ratio(CityName, OutputDir=RESULTDIR):
    """
    Calculate the mean annual frequency of exceeding a certain loss ratio and collapse ratio for a given city.

    Parameters:
        CityName (str): The name of the city for which to calculate the annualized loss.
        OutputDir (Path): The directory where the results will be saved. Defaults to RESULTDIR.

    Resulting CSV files:
         - mean_annual_freq_loss_ratio.csv: A DataFrame containing the annualized frequency of exceeding each loss ratio value. Two columns: 'LossRatio' and 'Frequency'.
         - mean_annual_freq_collapse_ratio.csv: A DataFrame containing the annualized frequency of exceeding each collapse ratio value. Two columns: 'CollapseRatio' and 'Frequency'.
         - mean_annual_freq_loss_ratio_{DistrictName}.csv: Similar to above, but for each district.
         - mean_annual_freq_collapse_ratio_{DistrictName}.csv: Similar to above, but for each district.
    """
    
    # Create city-specific output directory
    city_output_dir = OutputDir / CityName
    city_output_dir.mkdir(parents=True, exist_ok=True)

    # Get rupture info for multiplicity and duration
    rup_df = EQsources.ReadCityRuptures(CityName)
    rup_multiplicity = rup_df['multiplicity'].values.astype(float)
    years = EQsources.get_duration_years(CityName)

    # df_ratio_sim (pd.DataFrame): DataFrame containing total loss ratios for each simulation
    # - Columns: ['CityName', 'DistrictName', 'TotalLossRatio', 'CollapseRatio']. 
    # TotalLossRatio and CollapseRatio are numpy arrays of shape (N_sim,N_rup)
    df_ratio_sim = RegionalLossSim.get_total_loss_and_collapse_ratio_sim(CityName)

    if df_ratio_sim.empty:
        logger.warning(f"No ratio simulation data found for {CityName}")
        return

    for _, row in df_ratio_sim.iterrows():
        district_name = row['DistrictName']
        loss_ratio_sim = row['TotalLossRatio'] # (N_sim, N_rup)
        collapse_ratio_sim = row['CollapseRatio'] # (N_sim, N_rup)
        
        # Define file names
        if district_name == 'All':
            loss_file = city_output_dir / "mean_annual_freq_loss_ratio.csv"
            collapse_file = city_output_dir / "mean_annual_freq_collapse_ratio.csv"
        else:
            loss_file = city_output_dir / f"mean_annual_freq_loss_ratio_{district_name}.csv"
            collapse_file = city_output_dir / f"mean_annual_freq_collapse_ratio_{district_name}.csv"
            
        # Calculate and save Loss Ratio Freq
        _calc_and_save_freq_curve(loss_ratio_sim, rup_multiplicity, years, loss_file, 'LossRatio')
        
        # Calculate and save Collapse Ratio Freq
        _calc_and_save_freq_curve(collapse_ratio_sim, rup_multiplicity, years, collapse_file, 'CollapseRatio')

def _calc_and_save_freq_curve(values, multiplicity, years, filepath, col_name):
    """Helper function to calculate and save frequency curve using optimized sorting."""
    # values: (N_sim, N_rup)
    # multiplicity: (N_rup,)
    
    N_sim, N_rup = values.shape
    
    # Flatten values
    flat_values = values.flatten() # (N_sim * N_rup,)
    
    # Create weights
    # Each simulation for rupture i has weight multiplicity[i] / N_sim
    weights = np.repeat(multiplicity / N_sim, N_sim) # (N_sim * N_rup,)
    
    # Sort values and weights
    sort_idx = np.argsort(flat_values)
    sorted_values = flat_values[sort_idx]
    sorted_weights = weights[sort_idx]
    
    # Calculate cumulative frequency (exceedance)
    # We want sum of weights for v >= x
    # cumsum from right to left
    cum_freq = np.cumsum(sorted_weights[::-1])[::-1]
    
    # Annualize
    cum_freq = cum_freq / years
    
    # Save original values without interpolation
    pd.DataFrame({col_name: sorted_values, 'Frequency': cum_freq}).to_csv(filepath, index=False)

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
    total_loss_sim = precomputed_data['total_loss_sim']
    rup_multiplicity = precomputed_data['rup_multiplicity']
    years = precomputed_data['years']

    # Calculate the annualized frequency of exceeding each loss value
    # total_loss_sim shape: (1 x N_sim_per_rup x N_rup) since LossType is from BldPSPA_ReprBld.LossTypes2
    loss_values = total_loss_sim[0, :, :]  # shape: (N_sim_per_rup, N_rup) - only one design state

    min_loss = np.min(loss_values)
    max_loss = np.max(loss_values)
    N_points = 1001
    loss_X = np.linspace(min_loss, max_loss, N_points)
    
    annualized_freq = []
    for lossvalue in loss_X:
        # Count exceedances across all simulations and ruptures
        P_greater = np.sum(loss_values >= lossvalue, axis=0)/loss_values.shape[0] 
        exceedances = np.sum(P_greater * rup_multiplicity)
        annualized_freq.append(exceedances / years)  # Annualized frequency
    
    # Save results to CSV
    results_df = pd.DataFrame({
        LossType: loss_X,
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
    total_loss_sim = precomputed_data['total_loss_sim']
    
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
                # total_loss_sim shape: (N_DS x N_sim_per_rup x N_rup), 取第一个设计状态 [0, i_rup]
                loss_value = np.mean(total_loss_sim[0, :, i_rup]) 
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