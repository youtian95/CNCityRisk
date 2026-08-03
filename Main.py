'''
Regional seismic risk assessment of buildings of major cities in China

Author: Tian You 
Date: 2024-09-01
'''

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
        RegionalLossSim.simulate_regional_losses(CityName=region_name, Bld_gdf=Bld_gdf, Event=event, losstype='DS_Struct')

    # results\Maps\CollapseRateContours: collapse rate contour plot
    if 1:
        region_name, _ = Utilities.generate_region_name(lon_min=lon_min, lon_max=lon_max, lat_min=lat_min, lat_max=lat_max)
        CreateRegionalMaps.plot_collapse_rate_contour(CityNameorRegionName=region_name, event=event)


def analyze_1_city(CityName):

    # Prepare building data, including structural type, occupancy type...    
    if 0:
        PrepareBldData.prepare_building_data(CityName, MinBldArea=50)
    if 0:        
        PrepareBldData.EstimateSeismicDesignLevel(CityName)
    if 0:
        # classify per structural type
        ClassifyBld.BldClassify(CityName)
    if 0:
        # classify per PSPA
        ClassifyBld.BldClassify(CityName, TargetLabels=ClassifyBld.PSPA_CLASS_LABELS, SaveDir=ClassifyBld.REPR_PSPA_BLD_DIR, ReprColName='ReprBldID_PSPA')

    # Figures: plot city buildings
    if 0:
        plot_city.plot_BldFootprints_properties(CityName = CityName, DistrictName='包河区', prop = 'structure')
    if 0:
        plot_city.plot_histogram_bld(CityName = CityName, DistrictName='包河区', prop = 'story')

    # IDA analysis for individual representative buildings
    if 0:
        IDA_ReprBld.IDA_analysis(NumPool=6)

    # Figures: plot IDA results for one buiding
    if 0:
        ReprBld = 20000
        IDA_results = (IDA_ReprBld.OutputDir / ('IDA_result_ReprBldID_'+str(ReprBld)+'.csv')).as_posix()
        IDA.plot_IDA_results(IDA_results, Stat=True)

    # Seismic loss simulation for individual representative buildings
    if 0:
        BldPSPA_ReprBld.LossAssess(StartOver=True)

    # Earthquake events generation
    if 0:
        eqs = EQsource(CityName = CityName)
        eqs.generate_earthquake_events(CityName)
    # Figures: plot earthquake events
    if 0:
        plot_earthquake_events_folium(CityName)
        
    # IM field simulation
    if 0:
        IMSim.CityBldIMSimFromEvents(CityName)
    if 0:
        IMSim.Plot_IM_Spatial_City(CityName, i_rup=13, median=True)

    # Regional seismic loss simulation
    if 0:
        Bld_gdf = Utilities.get_CityBldData(CityName)
        RegionalLossSim.simulate_regional_losses(CityName=CityName, Bld_gdf=Bld_gdf, losstype='DS_Struct')
        RegionalLossSim.simulate_regional_losses(CityName=CityName, Bld_gdf=Bld_gdf, losstype='RepairCost_Total')
        RegionalLossSim.simulate_regional_losses(CityName=CityName, Bld_gdf=Bld_gdf, losstype='RepairTime')

    # results\AnnualizedRisk: annualized loss calculation
    if 0:
        AnnualizedRisk.calculate_annualized_loss(CityName)

    # results\Maps\IMmap: Create maps for displaying IM fields
    if 0:
        CreateRegionalMaps.create_mapdata_IM(CityName, Overwrite=True)
    # results\Maps\mbtiles: Create maps for displaying building losses
    if 0:
        CreateRegionalMaps.create_mbtiles_city_regional_losses(CityName, LossType='DS_Struct', Overwrite=True)

    # results\Maps\RegionalLossStatistics: Create statistics for each event
    if 0:
        RegionalLossSim.create_statistics_regional_losses(CityName, OutputDir = CreateRegionalMaps.RESULTDIR, LossType='DS_Struct')
    if 0:
        RegionalLossSim.create_statistics_regional_losses(CityName, OutputDir = CreateRegionalMaps.RESULTDIR, LossType='RepairCost_Total')
    if 0:
        RegionalLossSim.create_statistics_regional_losses(CityName, OutputDir = CreateRegionalMaps.RESULTDIR, LossType='RepairTime')

if __name__ == '__main__':    

    # analyze_1_city('武汉')
    # analyze_1_city('合肥')
    # analyze_1_city('重庆')

    Utilities.package_web_static_map_data(CityName='重庆', WebStaticMapsDir=r'H:\2025 - City Resilience Assessment\temp')

    # reproduce_wenchuan2008_event()