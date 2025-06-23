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

# The target cities 
CityName = '合肥'

if __name__ == '__main__':    

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
        IDA_ReprBld.IDA_analysis(NumPool=4)

    # Figures: plot IDA results for one buiding
    if 0:
        ReprBld = 20000
        IDA_results = (IDA_ReprBld.OutputDir / ('IDA_result_ReprBldID_'+str(ReprBld)+'.csv')).as_posix()
        IDA.plot_IDA_results(IDA_results, Stat=True)

    # Seismic loss simulation for individual representative buildings
    if 0:
        BldPSPA_ReprBld.LossAssess()

    # Earthquake events generation
    if 0:
        eqs = EQsource(CityName = CityName)
        eqs.generate_earthquake_events(CityName)
    # Figures: plot earthquake events
    if 0:
        plot_earthquake_events_folium(CityName)
        
    # IM field simulation
    if 0:
        IMSim.CityBldIMSimFromEvents(CityName, overwrite=True)
    if 0:
        IMSim.Plot_IM_Spatial_City(CityName, i_rup=13, median=True)

    # Regional seismic loss simulation
    if 0:
        Bld_gdf = Utilities.get_CityBldData(CityName)
        rupid_2_iSim, all_IM_mat_Bld = IMSim.get_all_IM_mat_Bld(CityName)
        RegionalLossSim.simulate_regional_losses(all_IM_mat_Bld, Bld_gdf, losstype='DS_Struct', OutputDir=RegionalLossSim.RESULTDIR/CityName)
        RegionalLossSim.simulate_regional_losses(all_IM_mat_Bld, Bld_gdf, losstype='RepairCost_Total', OutputDir=RegionalLossSim.RESULTDIR/CityName)

    # Create maps for displaying building losses
    if 0:
        CreateRegionalMaps.create_mapdata_IM(CityName)
    if 0:
        CreateRegionalMaps.create_mbtiles_city_regional_losses(CityName, LossType='DS_Struct', Overwrite=True)
    if 0:
        RegionalLossSim.create_statistics_regional_losses(CityName, OutputDir = CreateRegionalMaps.RESULTDIR, LossType='DS_Struct')
    if 1:
        RegionalLossSim.create_statistics_regional_losses(CityName, OutputDir = CreateRegionalMaps.RESULTDIR, LossType='RepairCost_Total')
