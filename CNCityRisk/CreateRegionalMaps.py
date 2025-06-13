
from . import Utilities
from . import EQsources
from . import plot_city
from . import IMSim
from .Utilities import minify_html
from . import BldPSPA_ReprBld
from .RegionalLossSim import get_statistics_regional_losses
from . import RegionalLossSim
from .Logger import get_logger

from pathlib import Path
import h5py
import folium
import xyzservices
import branca.colormap as cm
import webbrowser
import py7zr
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import base64
from io import BytesIO
import webbrowser
import folium
from tqdm import tqdm
import numpy as np

# Initialize logger for this module
logger = get_logger('CreateRegionalMaps')

Palette = Utilities.FACECOLOR_HEX # color palette: green blue yellow red
RESULTDIR = Path(__file__).resolve().parent / 'results/Maps/'
MAP_ZIP = RESULTDIR / 'maps.7z'
# MAP_ZIP includes: 'RegionalLoss_{CityName}_{DistrictName}_{LossType}_{i_rup}_{iSim}.html', 'IM_median_{T}_{CityName}_{i_rup}.html'


def Add_BldLoss_Polygon_to_FoliumMap(m: folium.Map, Bld_gdf: gpd.GeoDataFrame, Lossh5pyFile: h5py.File, LossType: str = 'DS_Struct', iSim: int = 0) -> None:
    """
    Add building loss polygons to a Folium map.

    Parameters:
        m (folium.Map): Folium map object.
        Bld_gdf (gpd.GeoDataFrame): GeoDataFrame containing building data.
        Lossh5pyFile (h5py.File): HDF5 file containing the simulated regional losses. Each dataset is named as 'BldID_i'. ID should be the same as the index of Bld_gdf.
        LossType (str): Type of loss to plot. Options are 'DS_Struct', 'DS_NonStruct_DriftSen', 'DS_NonStruct_AccelSen', 'RepairCost_Total', 'RepairCost_Struct', 'RepairCost_NonStruct_DriftSen','RepairCost_NonStruct_AccelSen', 'RepairTime', 'RecoveryTime', 'FunctionLossTime'.
        iSim (int): Index of the simulation in the HDF5 file.
    """
    
    logger.debug(f"Adding building loss polygons to map: LossType={LossType}, iSim={iSim}, Buildings={len(Bld_gdf)}")

    # 直接使用传入的数据，不再重复加载
    f = Lossh5pyFile
    
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
    fig.set_size_inches(1, 1)

    # Save the figure to a BytesIO object
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.0, transparent=True)
    buf.seek(0)

    # Encode the image in base64
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

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
        <img src='data:image/png;base64,{image_base64}' alt='legend_Bld'>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    plt.close('all')


def plot_regional_losses(CityName: str, DistrictName: str = None, Bld_gdf: gpd.GeoDataFrame = None, ResultsDir: str = RegionalLossSim.RESULTDIR, LossType: str = 'DS_Struct', i_rup: int = 0, iSim: int = 0, BaseMap: str = 'Cartodb Positron', savedir = RESULTDIR, ShowInBrowser: bool = True, Overwrite: bool = False, LossDataInMemory = None) -> folium.Map:
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
        LossDataInMemory: 预加载的损失数据，如果提供则直接使用，否则从文件加载
    
    Returns:
        folium.Map: Folium map object.
    """
    
    logger.info(f"Creating regional loss map: {CityName}-{DistrictName}, LossType={LossType}, Rupture={i_rup}, Simulation={iSim}")

    if savedir is None:
        savedir = ResultsDir

    # where to save the map
    filepath = Path(savedir) / f'RegionalLoss_{CityName}_{DistrictName}_{LossType}_{i_rup}_{iSim}.html'

    # Check if the file already exists in the archive
    if MAP_ZIP.exists():
        with py7zr.SevenZipFile(MAP_ZIP, 'r') as archive:
            existing_files = archive.getnames()
            if filepath.name in existing_files:
                if not Overwrite:
                    logger.debug(f'{filepath.name} already exists in the archive. Skipping...')
                    return None

    if Bld_gdf is None:
        logger.debug(f'Reading building properties for plotting regional losses...')
        if DistrictName is not None:
            Bld_gdf = Utilities.get_DistrictBldData(CityName, DistrictName=DistrictName)
        else:
            Bld_gdf = Utilities.get_CityBldData(CityName)

    ResultsDir = Path(ResultsDir).resolve()
    
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
    logger.debug('Adding building polygons to the map...')
    if LossDataInMemory is not None:
        # 使用预加载的数据
        Add_BldLoss_Polygon_to_FoliumMap(m, Bld_gdf, LossDataInMemory, LossType, iSim_allRup)
    else:
        # 如果没有提供预加载数据，则从文件加载
        logger.debug(f'Loading loss data from HDF5 file: BldLoss_{LossType}.hdf5')
        with h5py.File(ResultsDir / CityName / f'BldLoss_{LossType}.hdf5', 'r') as f:
            # 先将数据加载到内存一次，避免在函数中重复加载
            f_in_memory = Utilities.load_h5py_group_to_memory(f)
            Add_BldLoss_Polygon_to_FoliumMap(m, Bld_gdf, f_in_memory, LossType, iSim_allRup)

    # add the IM field to map
    IMSim.Add_City_IMfield_to_FoliumMap(m, CityName, i_rup, iTplot = 1, iSim = iSim)

    # rupture polygon
    rupture = rup_df.loc[rup_df['rup_id'] == rupid].iloc[0]
    EQsources.Create_a_Rupture_Polygon(m, rupture)    # plot the city boundary and district boundary
    plot_city.Add_CityBoundary_to_FoliumMap(m, CityName)
    if DistrictName is not None:
        plot_city.Add_DistrictBoundary_to_FoliumMap(m, CityName, DistrictName)
    
    m.save(filepath)
    
    # 压缩HTML内容，减小文件体积
    logger.debug(f'Compressing HTML file: {filepath.name}')
    minify_html(filepath)

    if ShowInBrowser:
        webbrowser.open(filepath)

    # 使用更高压缩等级压缩地图文件
    # 压缩设置：LZMA2算法，最高压缩级别9
    filters = [{'id': py7zr.FILTER_LZMA2, 'preset': 9}]
    
    logger.debug(f'Compressing map file to archive: {MAP_ZIP.name}')
    # Compress the map file into MAP_ZIP
    if not MAP_ZIP.exists():
        with py7zr.SevenZipFile(MAP_ZIP, 'w', filters=filters) as archive:
            archive.write(filepath, arcname=filepath.name)
    else:
        with py7zr.SevenZipFile(MAP_ZIP, 'a', filters=filters) as archive:
            archive.write(filepath, arcname=filepath.name)

    # delete the html file
    filepath.unlink()

    return m

def create_maps_median_IM(CityName: str, BaseMap: str = 'Cartodb Positron', savedir = RESULTDIR):
    """
    Create maps for the median IM field of a specific rupture.
    
    Parameters:
        CityName (str): Name of the city.
        BaseMap (str): Type of basemap. Options are 'Cartodb Positron', 'Amap', 'OpenStreetMap.HOT'. Default is 'Cartodb Positron'.
        savedir (str): Directory to save the output files.
    """

    logger.info(f"Starting creation of median IM map for city: {CityName}")

    # Create save directory if it doesn't exist
    if savedir:
        logger.info(f"Creating save directory: {savedir}")
        Path(savedir).mkdir(parents=True, exist_ok=True)

    # Create the map
    rup_df = EQsources.ReadCityRuptures(CityName)

    for i_rup in range(len(rup_df)):
        filepath, Sa_period = IMSim.Plot_IM_Spatial_City(CityName=CityName, i_rup=i_rup, BaseMap=BaseMap, median=True, SaveFigDir=savedir, PlotNow=False)

        # rename the file to include the Sa_period
        new_filepath = Path(savedir) / f'IM_median_{Sa_period}_{CityName}_{i_rup}.html'
        filepath = filepath.rename(new_filepath)
        logger.info(f"Saved median IM map to {filepath}")

        # 压缩HTML内容，减小文件体积
        logger.debug(f'Compressing HTML file: {filepath.name}')
        minify_html(filepath)
        # 压缩设置：LZMA2算法，最高压缩级别9
        filters = [{'id': py7zr.FILTER_LZMA2, 'preset': 9}]
        logger.debug(f'Compressing map file to archive: {MAP_ZIP.name}')
        # Compress the map file into MAP_ZIP
        if not MAP_ZIP.exists():
            with py7zr.SevenZipFile(MAP_ZIP, 'w', filters=filters) as archive:
                archive.write(filepath, arcname=filepath.name)
        else:
            with py7zr.SevenZipFile(MAP_ZIP, 'a', filters=filters) as archive:
                archive.write(filepath, arcname=filepath.name)

        # delete the html file
        filepath.unlink()

    logger.info(f"Completed creation of median IM map for city: {CityName}, Rupture: {i_rup}")


def create_maps_city_regional_losses(CityName: str, ResultsDir = RegionalLossSim.RESULTDIR, LossType: str = 'DS_Struct', iSim: int = 0, BaseMap: str = 'Cartodb Positron', savedir = RESULTDIR, Overwrite: bool = False):
    '''
    Create maps for the regional seismic losses. Each rupture has a random map.
    Parameters:
        CityName (str): Name of the city.
        ResultsDir (str): Directory containing the loss results.
        LossType (str): Type of loss to plot. Default is 'DS_Struct' (structural damage state).
        iSim (int): Index of the simulation for a specific rupture. Default is 0.
        BaseMap (str): Type of basemap. Options are 'Cartodb Positron', 'Amap', 'OpenStreetMap.HOT'. Default is 'Cartodb Positron'.
        savedir (str): Directory to save the output files.
        Overwrite (bool): Whether to overwrite existing files. Default is False.
    '''
    
    logger.info(f"Starting creation of regional loss maps for city: {CityName}, LossType: {LossType}")
    
    # find the district names
    DistrictNames = Utilities.get_DistrictsOfCity_names(CityName)
    logger.info(f"Found {len(DistrictNames)} districts to process: {', '.join(DistrictNames)}")

    rup_df = EQsources.ReadCityRuptures(CityName)
    logger.info(f"Processing {len(rup_df)} ruptures for each district")

    # Create save directory if it doesn't exist
    if savedir:
        logger.info(f"Creating save directory: {savedir}")
        Path(savedir).mkdir(parents=True, exist_ok=True)
    
    # load HDF5 to memory to avoid repeated loading
    RegionalLossSim_path = Path(RegionalLossSim.RESULTDIR).resolve()
    logger.info(f'Loading HDF5 file to memory to avoid repeated loading: BldLoss_{LossType}.hdf5')
    with h5py.File(RegionalLossSim_path / CityName / f'BldLoss_{LossType}.hdf5', 'r') as f:
        loss_data_in_memory = Utilities.load_h5py_group_to_memory(f)
        
        # iterate over the district names
        for DistrictName in DistrictNames:
            logger.info(f'Creating maps for {CityName}, {DistrictName}...')
            Bld_gdf = Utilities.get_DistrictBldData(CityName, DistrictName=DistrictName)
            # iterate over the ruptures
            for i_rup in range(len(rup_df)):
                plot_regional_losses(CityName, DistrictName=DistrictName, Bld_gdf=Bld_gdf, ResultsDir=ResultsDir, LossType=LossType, i_rup=i_rup, iSim=iSim, BaseMap=BaseMap, savedir=savedir, ShowInBrowser=False, Overwrite=Overwrite, LossDataInMemory=loss_data_in_memory)  # 传递内存中的数据

    logger.info(f"Completed creation of regional loss maps for city: {CityName}")

    # # 对于CDF图，我们只需要统计数据，也可以使用相同的内存数据
    # logger.info('Creating CDF plots for regional losses...')
    # with h5py.File(RegionalLossSim_path / CityName / f'BldLoss_{LossType}.hdf5', 'r') as f:
    #     loss_data_in_memory = Utilities.load_h5py_group_to_memory(f)
    #     for i_rup in range(len(rup_df)):
    #         RegionalLossSim.plot_CDF_regional_losses(CityName, ResultsDir=ResultsDir, LossType=LossType,i_rup=i_rup, iSim=iSim, PlotNow=False, Overwrite=Overwrite, LossDataInMemory=loss_data_in_memory)  # 传递内存中的数据
    


