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
from CNCityRisk.PythonLib.MDOFModel.IDA import IDA
from CNCityRisk import Utilities
from CNCityRisk.EQsources import EQsource, plot_earthquake_events_folium

from pathlib import Path
import pandas as pd
import numpy as np
import cProfile
import pstats

# The target cities 
CityName = '合肥'

if __name__ == '__main__':

    # Prepare building data, including structural type, occupancy type...
    if 0:
        PrepareBldData.prepare_building_data(CityName)
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
        plot_city.plot_BldFootprints_properties(CityName = CityName, prop = 'structure')
    if 0:
        plot_city.plot_histogram_bld(CityName = CityName, prop = 'story')

    # IDA analysis for individual representative buildings
    if 0:
        IDA_ReprBld.IDA_analysis(NumPool=4)

    # Figures: plot IDA results for one buiding
    if 0:
        ReprBld = 100
        IDA_results = 'IDA_result_ReprBldID_'+str(ReprBld)+'.csv'
        FigName = 'IDA.jpg'
        IDA.plot_IDA_results(IDA_results, Stat=True, FigName=FigName)

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
        IMSim.CityBldIMSimFromEvents(CityName)
    if 0:
        IMSim.Plot_IM_Spatial_City(CityName, i_rup=10)

    # Regional seismic loss simulation
    if 0:
        Bld_gdf = Utilities.get_CityBldData(CityName)
        rupid_2_iSim, all_IM_mat_Bld = IMSim.get_all_IM_mat_Bld(CityName)
        RegionalLossSim.simulate_regional_losses(all_IM_mat_Bld, Bld_gdf, losstype='DS_Struct', OutputDir=RegionalLossSim.RESULTDIR/CityName)
        RegionalLossSim.simulate_regional_losses(all_IM_mat_Bld, Bld_gdf, losstype='RepairCost_Total', OutputDir=RegionalLossSim.RESULTDIR/CityName)

    # Figures: plot regional seismic loss
    if 1:
        RegionalLossSim.create_maps_city_regional_losses(CityName, LossType='DS_Struct', iSim = 0)
        # RegionalLossSim.plot_regional_losses(CityName, DistrictName = '武昌区', LossType='DS_Struct', iSim = 0)

    