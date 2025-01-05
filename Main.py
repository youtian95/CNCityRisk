'''
Regional seismic risk assessment of buildings of major cities in China

Author: Tian You 
Date: 2024-09-01
'''

from src import plot_city
from src import PrepareBldData
from src import IDA_ReprBld
from src import ClassifyBld
from src import IMSim
from src import RegionalLossSim
from src import BldPSPA_ReprBld
from src.PythonLib.MDOFModel.IDA import IDA
from src import Utilities
from src.EQsources import EQsource, plot_earthquake_events_folium

from pathlib import Path
import pandas as pd
import numpy as np
import cProfile
import pstats

# The target cities 
CityName = '武汉'

if __name__ == '__main__':

    # Prepare building data, including structural type, occupancy type...
    if 0:
        PrepareBldData.prepare_building_data(CityName)
    if 0:
        PrepareBldData.EstimateSeismicDesignLevel(CityName)
    if 0:
        ClassifyBld.BldClassify(CityName)
    if 0:
        ClassifyBld.BldClassify(CityName, TargetLabels=ClassifyBld.PSPA_CLASS_LABELS, SaveDir=ClassifyBld.REPR_PSPA_BLD_DIR, StartFromScratch=True, ReprColName='ReprBldID_PSPA')

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
    if 0:
        RegionalLossSim.plot_regional_losses(CityName, DistrictName='武昌区', LossType='DS_Struct')
    # Figures: plot CDF of regional seismic loss
    if 1:
        RegionalLossSim.plot_CDF_regional_losses(CityName, LossType='RepairCost_Total', i_rup = 0, iSim = 0)

    