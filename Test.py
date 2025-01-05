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

from pathlib import Path
import pandas as pd
import numpy as np
import cProfile
import pstats

# The target cities 
i_city = 0
CityName = ['武汉']


if __name__ == '__main__':

    # Prepare building data, including structural type, occupancy type...
    if 0:
        PrepareBldData.prepare_building_data(CityName[i_city])
    if 0:
        PrepareBldData.EstimateSeismicDesignLevel(CityName[i_city])
    if 0:
        ClassifyBld.BldClassify(CityName[i_city])
    if 0:
        ClassifyBld.BldClassify(CityName[i_city], TargetLabels=ClassifyBld.PSPA_CLASS_LABELS, SaveDir=ClassifyBld.REPR_PSPA_BLD_DIR, StartFromScratch=True, ReprColName='ReprBldID_PSPA')

    # Figures: plot city buildings
    if 0:
        plot_city.plot_BldFootprints_properties(CityName = CityName[i_city], prop = 'structure')
    if 0:
        plot_city.plot_histogram_bld(CityName = CityName[i_city], prop = 'story')

    # IDA analysis for individual representative buildings
    if 0:
        IDA_ReprBld.IDA_analysis(NumPool=4)

    # Figures: plot IDA results for one buiding
    if 0:
        ReprBld = 100
        IDA_results = 'IDA_result_ReprBldID_'+str(ReprBld)+'.csv'
        FigName = 'temp/IDA.jpg'
        IDA.plot_IDA_results(IDA_results, Stat=True, FigName=FigName)

    # Seismic loss simulation for individual representative buildings
    if 0:
        BldPSPA_ReprBld.LossAssess()
    if 0:
        ReprPSPAID = 0
        IM = [0.2, 0.3]
        Nsim = [10, 20]
        DS_prob_All = BldPSPA_ReprBld.GetProbDS(ReprPSPAID = ReprPSPAID, IM = IM, losstype = 'DS_Struct')
        print("DS_prob_All:", DS_prob_All)
        loss_prob = BldPSPA_ReprBld.GetCDFLoss(ReprPSPAID = ReprPSPAID, IM = IM, losstype = 'RepairCost_Total', loss = [0, 1e6])
        print("loss_prob:", loss_prob)
        DS_sim = BldPSPA_ReprBld.SimLossRandomly(ReprPSPAID = ReprPSPAID, IM = IM, losstype = 'DS_Struct', Nsim = Nsim)
        print("DS_sim:", DS_sim)
        Loss_sim = BldPSPA_ReprBld.SimLossRandomly(ReprPSPAID = ReprPSPAID, IM = IM, losstype = 'RepairCost_Total', Nsim = Nsim)
        print("Loss_sim:", Loss_sim)
        

    # IM field simulation
    if 0:
        DistrictName = '武昌区'
        sim = IMSim.SpatialIM()
        sim.CreateAExampleSourceFile('results/temp/SourceFile.txt')
        Bld_gdf = Utilities.ReadDistrictBldGDF(CityName[i_city], DistrictName)
        minx, miny, maxx, maxy = Bld_gdf.total_bounds
        buffer = 0.01  # Increase the bounds by a small buffer
        minx -= buffer
        miny -= buffer
        maxx += buffer
        maxy += buffer
        sim.CreateMeshedSiteFile(minx,maxx,miny,maxy,SiteFile='results/temp/SiteFile.txt', MeshDist=1000)
        sim.IMFieldSim('results/IMsim/')
        sim.Plot_IM_Spatial_folium()
        # interpolate
        IM_mat_Bld = sim.InterpIMfromMeshForCity(CityName[i_city],DistrictName) # numpy array (i_bld, i_Sim)

    # Regional seismic loss simulation
    if 0:
        DistrictName = '武昌区'
        IM_mat_Bld = np.loadtxt(f'results/IMsim/IM_mat_Bld_{CityName[i_city]}_{DistrictName}.txt')
        BldsFile = Path('./CNCityRisk/Data/BldData/') / CityName[i_city] / f'{DistrictName}.csv'
        RegionalLossSim.simulate_regional_losses(IM_mat_Bld, BldsFile, 'results/RegionalLossSim/', losstype='DS_Struct')

    # Figures: plot regional seismic loss
    if 0:
        DistrictName = '武昌区'
        Bld_gdf = Utilities.ReadDistrictBldGDF(CityName[i_city], DistrictName)
        RegionalLossSim.plot_regional_losses(Bld_gdf, ResultsDir='results/RegionalLossSim/')

    