import folium.raster_layers
from . import Utilities
from . import BldPSPA_ReprBld
from . import EQsources
from . import IMSim
from . import plot_city

from pathlib import Path
import pandas as pd
from tqdm import tqdm
import folium
import geopandas as gpd
import webbrowser
import numpy as np
import pandas as pd
import h5py
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import geopandas as gpd
import dask.array as da
from typing import Union
import xyzservices
import branca.colormap as cm
import py7zr

Palette = Utilities.FACECOLOR_HEX # color palette: green blue yellow red
RESULTDIR = Path(__file__).resolve().parent / 'results/RegionalLossSim/'
MAP_ZIP = RESULTDIR / 'maps.7z'

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
    # check if the losstype is valid
    if losstype not in BldPSPA_ReprBld.LossTypes1 + BldPSPA_ReprBld.LossTypes2:
        raise ValueError(f"losstype must be one of {BldPSPA_ReprBld.LossTypes1 + BldPSPA_ReprBld.LossTypes2}.")
    
    # Get the full path of the OutputDir
    OutputDir = Path(OutputDir).resolve()
    if not OutputDir.exists():
        OutputDir.mkdir(parents=True)

    # Read the table
    T = Bld_gdf

    # check if the row number of IM_sim is equal to the row number of T
    if len(IM_sim) != len(T):
        raise ValueError("The number of rows in IM_sim must be equal to the number of rows in BldsFile.")
    
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
  
def plot_regional_losses(CityName: str, DistrictName: str = None, ResultsDir: str = RESULTDIR, LossType: str = 'DS_Struct', i_rup: int = 0, iSim: int = 0, BaseMap: str = 'Cartodb Positron', savedir = None, ShowInBrowser: bool = True) -> folium.Map:
    """
    Plot the regional seismic losses. Each row index in Bld_gdf must correspond to each building ID in the files in ResultsDir.
    
    Parameters:
        CityName (str): Name of the city.
        DistrictName (str): Name of the district.
        Bld_gdf (gpd.GeoDataFrame): GeoDataFrame containing building data.
        ResultsDir (str): Directory containing the loss results.
        LossType (str): Type of loss to plot. Default is 'DS_Struct' (structural damage state).
        i_rup (int): Index of the rupture. Default is 0.
        iSim (int): Index of the simulation for a specific rupture. Default is 0.
        BaseMap (str): Type of basemap. Options are 'Cartodb Positron', 'Amap', 'OpenStreetMap.HOT'. Default is 'Cartodb Positron'.
        savedir (str): Directory to save the output files.
    
    Returns:
        folium.Map: Folium map object.
    """

    print(f'Reading building properties for plotting regional losses...')
    Bld_gdf = Utilities.get_CityBldData(CityName)
    if DistrictName is not None:
        Bld_gdf = Bld_gdf[Bld_gdf['DistrictName'] == DistrictName]

    ResultsDir = Path(ResultsDir).resolve()
    
    if savedir is None:
        savedir = ResultsDir
    
    # Create a basemap
    Center = Bld_gdf.geometry.centroid.iloc[0].coords[0]
    if BaseMap is None:
        m = folium.Map(tiles=None, location=[Center[1], Center[0]], zoom_start=16)
    elif BaseMap == 'Cartodb Positron':
        m = folium.Map(tiles=BaseMap, location=[Center[1], Center[0]], zoom_start=12)
    elif BaseMap == 'Amap':
        m = folium.Map(location=[Center[1], Center[0]], zoom_start=12, tiles= 'https://wprd01.is.autonavi.com/appmaptile?x={x}&y={y}&z={z}&lang=zh_cn&size=1&scl=1&style=7',attr='高德', max_zoom=19)
    elif BaseMap == 'OpenStreetMap.HOT':
        tiles = xyzservices.providers.OpenStreetMap.HOT
        m = folium.Map(location=[Center[1], Center[0]], tiles=tiles, max_zoom=19)

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
    
    # plot the building loss polygons
    print('Adding building polygons to the map...')
    with h5py.File(ResultsDir / CityName / f'BldLoss_{LossType}.hdf5', 'r') as f:
        Add_BldLoss_Polygon_to_FoliumMap(m, Bld_gdf, f, LossType, iSim_allRup, savedir=savedir)

    # add the IM field to map
    IMSim.Add_City_IMfield_to_FoliumMap(m, CityName, i_rup, iTplot = 1, iSim = iSim)

    # rupture polygon
    rupture = rup_df.loc[rup_df['rup_id'] == rupid].iloc[0]
    EQsources.Create_a_Rupture_Polygon(m, rupture)

    # plot the city boundary and district boundary
    plot_city.Add_CityBoundary_to_FoliumMap(m, CityName)
    if DistrictName is not None:
        plot_city.Add_DistrictBoundary_to_FoliumMap(m, CityName, DistrictName)

    # save the map
    filepath = Path(savedir) / f'RegionalLoss_{CityName}_{DistrictName}_{LossType}_{i_rup}_{iSim}.html'
    m.save(filepath)

    if ShowInBrowser:
        webbrowser.open(filepath)

    # Compress the map file into MAP_ZIP
    if not MAP_ZIP.exists():
        with py7zr.SevenZipFile(MAP_ZIP, 'w') as archive:
            archive.write(filepath, arcname=filepath.name)
    else:
        with py7zr.SevenZipFile(MAP_ZIP, 'a') as archive:
            archive.write(filepath, arcname=filepath.name)

    # delete the html file
    filepath.unlink()

    return m

def Add_BldLoss_Polygon_to_FoliumMap(m: folium.Map, Bld_gdf: gpd.GeoDataFrame, Lossh5pyFile: h5py.File, LossType: str = 'DS_Struct', iSim: int = 0, savedir: Path = RESULTDIR) -> None:
    """
    Add building loss polygons to a Folium map.

    Parameters:
        m (folium.Map): Folium map object.
        Bld_gdf (gpd.GeoDataFrame): GeoDataFrame containing building data.
        Lossh5pyFile (h5py.File): HDF5 file containing the simulated regional losses. Each dataset is named as 'BldID_i'. ID should be the same as the index of Bld_gdf.
        LossType (str): Type of loss to plot. Options are 'DS_Struct', 'DS_NonStruct_DriftSen', 'DS_NonStruct_AccelSen', 'RepairCost_Total', 'RepairCost_Struct', 'RepairCost_NonStruct_DriftSen','RepairCost_NonStruct_AccelSen', 'RepairTime', 'RecoveryTime', 'FunctionLossTime'.
        iSim (int): Index of the simulation in the HDF5 file.
        savedir (Path): Directory to save the intermediate legend images. Default is 'results/RegionalLossSim/'.
    """

    f = Lossh5pyFile
    f = Utilities.load_h5py_group_to_memory(f)
    
    # plot each building
    with tqdm(total=len(Bld_gdf), desc="Loading regional losses...") as pbar:
        for i, row in Bld_gdf.iterrows():
            pbar.update(1)
                
            # read the loss
            loss_sim = f[f'BldID_{i}'][:]

            # footprint
            poly = row['geometry']
            coord_bld = list(poly.exterior.coords)
            coord_bld = [(round(lat, 6), round(lon, 6)) for lon, lat in coord_bld]

            # color
            if LossType == 'DS_Struct':
                loss = loss_sim[iSim]
                if loss == 0:
                    color = Palette[0]
                elif loss == 1:
                    color = Palette[1]
                elif loss == 2:
                    color = Palette[2]
                elif loss == 3:
                    color = Palette[3]
                elif loss == 4:
                    color = 'black'
                else:
                    color = 'gray'

            # plot
            bld = folium.Polygon(coord_bld, weight = 0, fill_color=color, fill_opacity=1)
            # add a marker for each building
            bld.add_child(folium.Popup(f'Building ID: {i}; {LossType}: {loss}'))
            bld.add_to(m)


    # Create a legend image
    legend_elements = [
        Patch(facecolor=Palette[0], edgecolor='black', label='None'),
        Patch(facecolor=Palette[1], edgecolor='black', label='Slight'),
        Patch(facecolor=Palette[2], edgecolor='black', label='Moderate'),
        Patch(facecolor=Palette[3], edgecolor='black', label='Extensive'),
        Patch(facecolor='black', edgecolor='black', label='Complete'),
        Patch(facecolor='gray', edgecolor='black', label='Unknown'),
    ]

    # Create a figure for the legend
    fig, ax = plt.subplots()
    ax.legend(handles=legend_elements, loc='upper right', title=LossType)
    ax.axis('off')
    fig.patch.set_alpha(0.0)  # Set the background to be transparent
    legend_path = savedir / 'legend_Bld.png'
    fig.set_size_inches(1, 1)
    fig.savefig(legend_path, bbox_inches='tight', pad_inches=0.0, transparent=True)
    # Add custom CSS to position the legend in the top right corner
    css = """
    <style>
        .legend_Bld {
            position: absolute;
            bottom: 10px;
            right: 10px;
            z-index: 1000;
            padding: 10px;
        }
    </style>
    """
    legend_html = f"""
    {css}
    <div class='legend_Bld'>
        <img src='./legend_Bld.png' alt='legend_Bld'>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

def get_statistics_regional_losses(CityName: str, ResultsDir: str = RESULTDIR, LossType: str = 'DS_Struct', i_rup: int = 0, iSim: int = None):
    '''
    Get the statistics of the regional seismic losses, given an earthquake event.

    Parameters:
        CityName (str): Name of the city.
        ResultsDir (str): Directory containing the loss results.
        LossType (str): Type of loss to plot. Default is 'DS_Struct' (structural damage state).
        i_rup (int): Index of the rupture. Default is 0.
        iSim (int): Index of the simulation for a specific rupture. Default is None.

    Returns:
        total_loss_sim (np.ndarray): Cumulative distribution function of the regional seismic losses. 
            (N_DS x N_sim)
                if LossType is 'DS_Struct' ..., N_DS = 5 (damage states). Each element is the numbers of the building to be in the corresponding damage state.
                if LossType is 'RepairCost_Total' ..., N_DS = 1. Each element is a total repair cost.
        total_loss_iSim (np.ndarray): regional total losses for the iSim-th simulation. (N_DS)
        percentiles_iSim (np.ndarray): Percentiles of the iSim-th simulation.
            (N_DS)
                if LossType is 'DS_Struct' ..., N_DS = 5 (damage states). 
                if LossType is 'RepairCost_Total' ..., N_DS = 1.
    '''
    print(f'Calculating statistics of regional losses...')

    Bld_gdf = Utilities.get_CityBldData(CityName)
    N_bld = len(Bld_gdf)

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
    with h5py.File(ResultsDir / CityName / f'BldLoss_{LossType}.hdf5', 'r') as f:
        f = Utilities.load_h5py_group_to_memory(f)
        for i in range(N_bld):
            loss_sim = f[f'BldID_{i}'][:]
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

def plot_CDF_regional_losses(CityName: str, ResultsDir: str = RESULTDIR, LossType: str = 'DS_Struct', i_rup: int = 0, iSim: int = None, SavePath: Path = None, PlotNow: bool = True):
    '''
    plot the cumulative distribution function of total regional seismic losses.
    '''
    
    total_loss_sim, total_loss_iSim, percentiles_iSim = get_statistics_regional_losses(CityName, ResultsDir, LossType, i_rup, iSim)

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

    if SavePath is None:
        SavePath = ResultsDir / f'CDF_{CityName}_{LossType}_{i_rup}_{iSim}.jpg'
    fig.savefig(SavePath)

    # Compress the map file into MAP_ZIP
    if not MAP_ZIP.exists():
        with py7zr.SevenZipFile(MAP_ZIP, 'w') as archive:
            archive.write(SavePath, arcname=SavePath.name)
    else:
        with py7zr.SevenZipFile(MAP_ZIP, 'a') as archive:
            archive.write(SavePath, arcname=SavePath.name)

    # delete the html file
    SavePath.unlink()
    
    if PlotNow:
        plt.show()

