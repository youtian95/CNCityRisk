"""
中国主要城市建筑区域地震风险评估主程序。

使用流程：设置城市名称后，依次准备建筑数据、补充增量代表建筑分析、生成地震事件与地震动场，并计算区域损失、年化风险和地图数据。

输入：城市名称及项目配置文件中的分析参数。
输出：建筑、震源、地震动、区域损失、年化风险和网页地图结果。

当前标准流程的重复运行策略：

1. 按需更新或增量计算
   - 建筑准备：复用已有区县数据，仅在面积筛选删除建筑或缺少属性时写回。
   - 抗震设计水平：三列结果均已存在时跳过，否则补充后写回。
   - 建筑分类：重新计算当前区县映射；沿用已有代表建筑编号，只追加新类型。
   - 代表建筑 IDA：默认 ``StartOver=False``，跳过已有结果，只计算新增代表建筑。
   - 代表建筑损失：默认 ``StartOver=False``，跳过已有结果，只计算新增代表建筑。

2. 每次强制重算
   - 城市 EQsources：初始化时清空该城市旧震源目录，然后重新运行 OpenQuake。
   - 城市地震动：使用 ``overwrite=True``，按新震源覆盖旧地震动并重建合并矩阵。
   - 三类区域损失、年化风险、地震动地图、损失地图和损失统计：全部重新生成。
   - 网页静态地图数据：使用 ``Overwrite=True`` 覆盖旧打包结果。

3. 默认不运行
   - 所有位于 ``if 0`` 分支中的诊断图和单事件示例。
"""

from CNCityRisk import plot_city
from CNCityRisk import PrepareBldData
from CNCityRisk import IDA_ReprBld
from CNCityRisk import ClassifyBld
from CNCityRisk import IMSim
from CNCityRisk import RegionalLossSim
from CNCityRisk import BldPSPA_ReprBld
from CNCityRisk import Utilities
from CNCityRisk import CreateRegionalMaps
from CNCityRisk import AnnualizedRisk
from CNCityRisk.EQsources import EQsource, plot_earthquake_events_folium
from CNCityRisk.Logger import init_root_logger
init_root_logger()
from CNCityRisk.Logger import get_logger

from MDOFModel.IDA import IDA
from pathlib import Path
import pandas as pd
import numpy as np
import cProfile
import pstats
import time

# Get main program logger
logger = get_logger('Main')

def reproduce_wenchuan2008_event():
    '''
    Reproduce the building losses of the 2008 Wenchuan earthquake event
    '''

    # Prepare building data, including structural type, occupancy type...
    lat_min, lat_max = 30.7678, 32.7578
    lon_min, lon_max = 102.8883, 105.7345

    if 0:
        PrepareBldData.prepare_building_data(min_lon=lon_min, max_lon=lon_max, min_lat=lat_min, max_lat=lat_max, MinBldArea=50)
    if 0:        
        PrepareBldData.EstimateSeismicDesignLevel(min_lon=lon_min, max_lon=lon_max, min_lat=lat_min, max_lat=lat_max)
    if 0:
        # classify per structural type
        ClassifyBld.BldClassify(min_lon=lon_min, max_lon=lon_max, min_lat=lat_min, max_lat=lat_max)
    if 0:
        # classify per PSPA
        ClassifyBld.BldClassify(min_lon=lon_min, max_lon=lon_max, min_lat=lat_min, max_lat=lat_max, TargetLabels=ClassifyBld.PSPA_CLASS_LABELS, SaveDir=ClassifyBld.REPR_PSPA_BLD_DIR, ReprColName='ReprBldID_PSPA')

    # Figures: plot city buildings
    if 0:
        plot_city.plot_BldFootprints_properties(min_lon=lon_min, max_lon=lon_max, min_lat=lat_min, max_lat=lat_max, prop = 'structure')

    # IDA analysis for individual representative buildings
    if 0:
        IDA_ReprBld.IDA_analysis(NumPool=4)

    # Seismic loss simulation for individual representative buildings
    if 0:
        BldPSPA_ReprBld.LossAssess()

    # IM field simulation
    # 汶川地震北川-映秀断裂带（张勇等，2008）
    # 'M', 'N_sim', 'lon_Hypo', 'lat_Hypo', 'W_km', 'length_km', 'strike', 'dip', 'lambda_degree', 'Zhyp_km'
    event = {'M':7.9, 'N_sim':100, 'lon_Hypo':104.1055, 'lat_Hypo':31.647, 'Zhyp_km':15.5, 'length_km': 240, 'W_km': 999, 'strike': 225, 'dip': 39, 'lambda_degree': 120}
    if 0:
        IMSim.RegionBldIMSimFromEvents(lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max, event=event)
    if 0:
        IMSim.Plot_IM_Spatial_City(lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max, median=True)

    # Regional seismic loss simulation
    if 0:
        Bld_gdf = Utilities.get_RegionBldData(lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max)
        region_name, _ = Utilities.generate_region_name(lon_min=lon_min, lon_max=lon_max, lat_min=lat_min, lat_max=lat_max)
        RegionalLossSim.simulate_regional_losses(CityNameorRegionName=region_name, Bld_gdf=Bld_gdf, Event=event, losstype='DS_Struct')

    # results\Maps\CollapseRateContours: collapse rate contour plot
    if 1:
        region_name, _ = Utilities.generate_region_name(lon_min=lon_min, lon_max=lon_max, lat_min=lat_min, lat_max=lat_max)
        CreateRegionalMaps.plot_collapse_rate_contour(CityNameorRegionName=region_name, event=event)


def analyze_1_city(CityName):

    # 三类城市损失结果需要同步重算并生成地图。
    loss_types = ('DS_Struct', 'RepairCost_Total', 'RepairTime')

    # Prepare building data, including structural type, occupancy type...
    if 1:
        PrepareBldData.prepare_building_data(CityName, MinBldArea=50)
    if 1:
        PrepareBldData.EstimateSeismicDesignLevel(CityName)
    if 1:
        # classify per structural type
        ClassifyBld.BldClassify(CityName)
    if 1:
        # classify per PSPA
        ClassifyBld.BldClassify(CityName, TargetLabels=ClassifyBld.PSPA_CLASS_LABELS, SaveDir=ClassifyBld.REPR_PSPA_BLD_DIR, ReprColName='ReprBldID_PSPA')

    # Figures: plot city buildings
    if 0:
        plot_city.plot_BldFootprints_properties(CityName = CityName, DistrictName='包河区', prop = 'structure')
    if 0:
        plot_city.plot_histogram_bld(CityName = CityName, DistrictName='包河区', prop = 'story')

    # IDA analysis for individual representative buildings
    if 1:
        IDA_ReprBld.IDA_analysis(NumPool=6)

    # Figures: plot IDA results for one buiding
    if 0:
        ReprBld = 20000
        IDA_results = (IDA_ReprBld.OutputDir / ('IDA_result_ReprBldID_'+str(ReprBld)+'.csv')).as_posix()
        IDA.plot_IDA_results(IDA_results, Stat=True)

    # Seismic loss simulation for individual representative buildings
    if 1:
        BldPSPA_ReprBld.LossAssess()

    # Earthquake events generation
    if 1:
        eqs = EQsource(CityName = CityName)
        eqs.generate_earthquake_events(CityName)
    # Figures: plot earthquake events
    if 0:
        plot_earthquake_events_folium(CityName)
        
    # IM field simulation
    if 1:
        # 震源重新生成后必须覆盖旧地震动，避免复用旧震源目录中的结果。
        IMSim.CityBldIMSimFromEvents(CityName, overwrite=True)
    if 0:
        IMSim.Plot_IM_Spatial_City(CityName, i_rup=13, median=True)

    # Regional seismic loss simulation
    if 1:
        Bld_gdf = Utilities.get_CityBldData(CityName)
        for loss_type in loss_types:
            RegionalLossSim.simulate_regional_losses(
                CityNameorRegionName=CityName,
                Bld_gdf=Bld_gdf,
                losstype=loss_type,
            )

    # results\AnnualizedRisk: annualized loss calculation
    if 1:
        AnnualizedRisk.calculate_annualized_loss(CityName)

    # results\Maps\IMmap: Create maps for displaying IM fields
    if 1:
        CreateRegionalMaps.create_mapdata_IM(CityName, Overwrite=True)
    # results\Maps\mbtiles: Create maps for displaying building losses
    if 1:
        for loss_type in loss_types:
            CreateRegionalMaps.create_mbtiles_city_regional_losses(
                CityName,
                LossType=loss_type,
                Overwrite=True,
            )

    # results\Maps\RegionalLossStatistics: Create statistics for each event
    if 1:
        for loss_type in loss_types:
            RegionalLossSim.create_statistics_regional_losses(
                CityName,
                OutputDir=CreateRegionalMaps.RESULTDIR,
                LossType=loss_type,
            )

if __name__ == '__main__':

    CityName = '重庆'

    analyze_1_city(CityName)

    Utilities.package_web_static_map_data(
        CityName=CityName,
        WebStaticMapsDir=str(Path(r'H:\2025 - City Resilience Assessment\tmp') / CityName),
        LossTypes=['DS_Struct', 'RepairCost_Total', 'RepairTime'],
        Overwrite=True,
    )
