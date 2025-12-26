from . import Utilities
from . import EQsources
from . import plot_city
from . import IMSim
from . import BldPSPA_ReprBld
from .Utilities import minify_html
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
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from scipy.interpolate import griddata
import geojsoncontour
import base64
from io import BytesIO
import webbrowser
import folium
from tqdm import tqdm
import json
import sqlite3
from datetime import datetime
import subprocess
import shutil
import os
from typing import Tuple


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


def plot_collapse_rate_contour(
    CityNameorRegionName: str,
    event: dict = None,
    i_rup: int = 0,
    ResultsDir: str = RegionalLossSim.RESULTDIR,
    grid_km: float = 10.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Figure]:
    """Plot the mean collapse probability for a selected rupture (city) or event (region).

    The routine first identifies collapse outcomes in the damage-state results (``DS_Struct``) and converts them into per-building collapse probabilities by averaging across the available simulations for the requested rupture or event. The resulting probabilities are then mapped at building centroids, aggregated onto a projected square grid to smooth the field, and finally rendered as filled contours in geographic coordinates. The aggregated grid used for plotting is also written to CSV so that other workflows can reuse the collapse surface directly.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, matplotlib.figure.Figure]: Arrays of grid-centroid longitude, latitude, the averaged collapse probability, and the rendered Matplotlib figure so callers can further customise or reuse the plot.
    """

    logger.info(f"Plotting collapse-rate contour for {CityNameorRegionName}")

    try:
        Bld_gdf = Utilities.get_CityBldData(CityNameorRegionName)
    except Exception as exc:
        logger.error(f"Failed to load building data for {CityNameorRegionName}: {exc}")
        raise

    if Bld_gdf.empty:
        raise ValueError(f"No building data found for {CityNameorRegionName}")

    collapse_code = BldPSPA_ReprBld.DS_types.index('Complete')
    results_dir = Path(ResultsDir).resolve()
    city_dir = results_dir / CityNameorRegionName

    if not city_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {city_dir}")

    collapse_rates: np.ndarray  # collapse rate for each building
    title_suffix = ''

    ds_struct_by_rupture = city_dir / 'BldLoss_DS_Struct_ByRupture.hdf5'

    if event is None:
        if not ds_struct_by_rupture.exists():
            raise FileNotFoundError(f"City-mode collapse results not found: {ds_struct_by_rupture}")
        rup_df = EQsources.ReadCityRuptures(CityNameorRegionName)
        if not (0 <= i_rup < len(rup_df)):
            raise IndexError(f"i_rup {i_rup} out of range (0-{len(rup_df)-1}) for {CityNameorRegionName}")
        collapse_rates = _compute_collapse_rate_city(ds_struct_by_rupture, collapse_code, i_rup)
        title_suffix = f'Rupture #{i_rup} (ID {rup_df.iloc[i_rup]["rup_id"]})'
    else:
        event_name = IMSim.get_event_name(event)
        event_dir = city_dir / event_name
        if not event_dir.exists():
            raise FileNotFoundError(f"Event results directory not found: {event_dir}")

        reorganized_file = event_dir / 'BldLoss_DS_Struct_ByRupture.hdf5'

        if not reorganized_file.exists():
            raise FileNotFoundError(
                f"Reorganized collapse results not found for event '{event_name}'"
            )

        collapse_rates = _compute_collapse_rate_event(reorganized_file, Bld_gdf.index.to_numpy(), collapse_code)
        title_suffix = event_name

    lon = Bld_gdf.geometry.centroid.x.to_numpy()
    lat = Bld_gdf.geometry.centroid.y.to_numpy()

    valid_mask = np.isfinite(collapse_rates)
    if not np.any(valid_mask):
        raise ValueError("Collapse-rate computation produced no finite values")

    lon_valid = lon[valid_mask]
    lat_valid = lat[valid_mask]
    collapse_valid = collapse_rates[valid_mask]

    grid_lon, grid_lat, grid_collapse = _aggregate_collapse_to_grid(
        lon_valid,
        lat_valid,
        collapse_valid,
        grid_km=grid_km,
        min_points=1
    )

    if grid_lon.size < 3 or grid_lat.size < 3:
        raise ValueError("At least three grid cells with collapse data are required to draw a contour plot")

    levels = np.linspace(0.0, 1, 101)

    # file output name
    output_dir = RESULTDIR / 'CollapseRateContours'
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_suffix = title_suffix.replace(' ', '_').replace('/', '-').replace(':', '-') if title_suffix else ''
    suffix = f'_{safe_suffix}' if safe_suffix else ''
    base_filename = f'CollapseRate_Contour{suffix}'
    fig_path = output_dir / f'{base_filename}.pdf'
    csv_path = output_dir / f'{base_filename}.csv'

    fig, ax = plt.subplots(figsize=(10, 8))
    contour = ax.tricontourf(
        grid_lon,
        grid_lat,
        grid_collapse,
        levels=levels,
        cmap='Oranges',
        alpha=0.85,
        vmin=0.0,
        vmax=1.0,
    )

    plot_bounds = _draw_county_boundaries(ax, grid_lon, grid_lat)

    cbar = fig.colorbar(contour, ax=ax, shrink=0.85)
    cbar.set_label('Mean Collapse Probability')
    cbar.set_ticks(np.linspace(0, 1, 11))

    ax.set_xlabel('Longitude (°)')
    ax.set_ylabel('Latitude (°)')
    ax.set_aspect('equal', adjustable='datalim')
    if plot_bounds is not None:
        lon_min, lon_max, lat_min, lat_max = plot_bounds
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
    ax.grid(True, linestyle='--', linewidth=0.3, alpha=0.4)

    fig.tight_layout()
    grid_df = pd.DataFrame({
        'longitude': grid_lon,
        'latitude': grid_lat,
        'mean_collapse_probability': grid_collapse
    })
    grid_df.to_csv(csv_path, index=False)
    fig.savefig(fig_path, format='pdf', bbox_inches='tight')

    logger.info(f"Collapse-rate contour saved to {fig_path}")
    logger.info(f"Collapse-rate grid data saved to {csv_path}")
    return grid_lon, grid_lat, grid_collapse, fig


def _compute_collapse_rate_city(h5_file: Path, collapse_code: int, i_rup: int) -> np.ndarray:
    """Calculate collapse rates from a single rupture in city-mode ByRupture results.

    Returns:
        numpy.ndarray: 1D array of mean collapse probabilities per building, ordered to align with the building indices stored in ``h5_file``. Values fall within ``[0, 1]``; entries remain ``nan`` if the corresponding building has no simulations."""

    if i_rup is None:
        raise ValueError("i_rup must be provided for city collapse contour plots")

    group_name = f'group_rup_{i_rup}'
    with h5py.File(h5_file, 'r') as f:
        if group_name not in f:
            available = sorted(int(g.split('_')[-1]) for g in f.keys() if g.startswith('group_rup_'))
            raise KeyError(f"Rupture group {group_name} not found. Available groups: {available}")
        loss_dataset = f[group_name]['loss_matrix']
        collapse_counts = np.sum(loss_dataset == collapse_code, axis=1)
        total_simulations = loss_dataset.shape[1]

    if total_simulations == 0:
        logger.warning("Selected rupture contains no simulations; returning zeros")
        return np.zeros(loss_dataset.shape[0], dtype=float)

    return collapse_counts / total_simulations


def _mean_collapse_from_matrix(loss_dataset: h5py.Dataset, collapse_code: int) -> np.ndarray:
    """Return per-building collapse probabilities from a loss_matrix dataset."""

    data = loss_dataset[:]
    n_buildings, n_sim = data.shape

    if n_sim == 0:
        logger.warning("Selected event contains no simulations; returning zeros")
        return np.zeros(n_buildings, dtype=float)

    missing_mask = None
    if np.issubdtype(data.dtype, np.floating):
        missing_mask = np.isnan(data)
    else:
        fillvalue = loss_dataset.fillvalue
        if fillvalue is not None:
            missing_mask = data == fillvalue

    collapse_matrix = data == collapse_code

    if missing_mask is not None:
        collapse_matrix = collapse_matrix.astype(float)
        collapse_matrix[missing_mask] = np.nan
        with np.errstate(invalid='ignore'):
            collapse_prob = np.nanmean(collapse_matrix, axis=1)
    else:
        collapse_prob = collapse_matrix.mean(axis=1, dtype=np.float64)

    return collapse_prob


def _compute_collapse_rate_event(h5_file: Path, bld_indices: np.ndarray, collapse_code: int) -> np.ndarray:
    """Calculate collapse rates from single-event regional results.

    Returns:
        numpy.ndarray: 1D array of mean collapse probabilities aligned with ``bld_indices``. Values are within ``[0, 1]`` when simulations exist; entries remain ``nan`` if a building has no simulations."""

    logger.info(f"Computing collapse rates from event-mode results: {h5_file}")
    bld_indices = np.asarray(bld_indices, dtype=np.int64)
    index_to_pos = {int(idx): pos for pos, idx in enumerate(bld_indices)}
    collapse_rates = np.full(len(bld_indices), np.nan, dtype=float)

    with h5py.File(h5_file, 'r') as f:
        if 'loss_matrix' in f and isinstance(f['loss_matrix'], h5py.Dataset):
            loss_dataset = f['loss_matrix']
            collapse_prob = _mean_collapse_from_matrix(loss_dataset, collapse_code)

            if 'building_indices' in f:
                file_indices = np.asarray(f['building_indices'][:], dtype=np.int64)
            else:
                file_indices = np.arange(collapse_prob.size, dtype=np.int64)

            for src_pos, idx in enumerate(file_indices):
                pos = index_to_pos.get(int(idx))
                if pos is not None:
                    collapse_rates[pos] = collapse_prob[src_pos]
            return collapse_rates
        else:
            raise KeyError("Expected 'loss_matrix' dataset not found in HDF5 file")


def _aggregate_collapse_to_grid(
    lon: np.ndarray,
    lat: np.ndarray,
    collapse: np.ndarray,
    grid_km: float = 10.0,
    min_points: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average collapse probabilities within square grids.

    Parameters:
        lon, lat (np.ndarray): Longitude and latitude arrays of valid building centroids.
        collapse (np.ndarray): Collapse probabilities aligned with ``lon``/``lat``.
        grid_km (float): Grid size in kilometres.
        min_points (int): Minimum number of buildings required for a grid cell to be retained.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Arrays of grid-centroid longitude, latitude, and averaged collapse
        probability. Empty arrays are returned when no cells satisfy the criteria.
    """

    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    collapse = np.asarray(collapse, dtype=float)

    finite_mask = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(collapse)
    lon = lon[finite_mask]
    lat = lat[finite_mask]
    collapse = collapse[finite_mask]

    empty = np.array([], dtype=float)

    if lon.size == 0:
        return empty, empty, empty

    if grid_km <= 0:
        raise ValueError("grid_km must be a positive number")

    points_gdf = gpd.GeoDataFrame(
        {'collapse': collapse},
        geometry=gpd.points_from_xy(lon, lat),
        crs='EPSG:4326'
    )

    try:
        target_crs = points_gdf.estimate_utm_crs()
        points_proj = points_gdf.to_crs(target_crs)
    except Exception:
        logger.debug("Failed to estimate local UTM CRS, falling back to EPSG:3857 for grid aggregation")
        points_proj = points_gdf.to_crs(epsg=3857)

    x = points_proj.geometry.x.to_numpy()
    y = points_proj.geometry.y.to_numpy()
    collapse_proj = points_proj['collapse'].to_numpy()

    grid_m = grid_km * 1000.0
    if grid_m <= 0:
        raise ValueError("Computed grid size must be positive")

    x0 = float(x.min())
    y0 = float(y.min())
    ix = np.floor((x - x0) / grid_m).astype(int)
    iy = np.floor((y - y0) / grid_m).astype(int)

    grid_df = pd.DataFrame({
        'ix': ix,
        'iy': iy,
        'x': x,
        'y': y,
        'collapse': collapse_proj
    })

    agg = grid_df.groupby(['ix', 'iy']).agg(
        x=('x', 'mean'),
        y=('y', 'mean'),
        collapse=('collapse', 'mean'),
        count=('collapse', 'size')
    ).reset_index()

    agg = agg[agg['count'] >= max(int(min_points), 1)]
    if agg.empty:
        return empty, empty, empty

    agg_gdf = gpd.GeoDataFrame(
        agg,
        geometry=gpd.points_from_xy(agg['x'], agg['y']),
        crs=points_proj.crs
    ).to_crs(epsg=4326)

    lon_grid = agg_gdf.geometry.x.to_numpy()
    lat_grid = agg_gdf.geometry.y.to_numpy()
    collapse_grid = agg_gdf['collapse'].to_numpy(dtype=float)

    return lon_grid, lat_grid, collapse_grid


def _draw_county_boundaries(ax: plt.Axes, lon: np.ndarray, lat: np.ndarray):
    """Overlay county-level boundaries using internal region polygons.

    Returns:
        tuple | None: Axis limits (lon_min, lon_max, lat_min, lat_max) including buffers if inputs are valid; ``None`` when insufficient data.
    """

    if lon.size == 0 or lat.size == 0:
        return None

    lon_min = float(np.min(lon))
    lon_max = float(np.max(lon))
    lat_min = float(np.min(lat))
    lat_max = float(np.max(lat))

    lon_buffer = max((lon_max - lon_min) * 0.05, 0.01)
    lat_buffer = max((lat_max - lat_min) * 0.05, 0.01)

    lims = (lon_min - lon_buffer, lon_max + lon_buffer, lat_min - lat_buffer, lat_max + lat_buffer)
    ax.set_xlim(lims[0], lims[1])
    ax.set_ylim(lims[2], lims[3])

    try:
        region_polygon = Utilities.get_region_polygon(lat_min, lat_max, lon_min, lon_max)
        if region_polygon is not None:
            gpd.GeoSeries([region_polygon]).plot(
                ax=ax,
                facecolor='none',
                edgecolor='black',
                linewidth=0.8,
                zorder=5
            )
    except Exception as exc:
        logger.debug(f"Failed to draw region boundaries': {exc}")

    return lims


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


def get_mbtiles_info(mbtiles_path: str):
    '''
    Get information about a MBTiles file.
    
    Parameters:
        mbtiles_path (str): Path to the MBTiles file.
    
    Returns:
        dict: Metadata information about the MBTiles file.
    '''
    
    if not Path(mbtiles_path).exists():
        logger.error(f"MBTiles file not found: {mbtiles_path}")
        return None
    
    try:
        conn = sqlite3.connect(mbtiles_path)
        cursor = conn.cursor()
        
        # Get metadata
        cursor.execute('SELECT name, value FROM metadata')
        metadata = dict(cursor.fetchall())
        
        # Get tile count
        cursor.execute('SELECT COUNT(*) FROM tiles')
        tile_count = cursor.fetchone()[0]
        
        metadata['tile_count'] = tile_count
        metadata['file_size_mb'] = Path(mbtiles_path).stat().st_size / (1024 * 1024)
        
        conn.close()
        
        return metadata
        
    except Exception as e:
        logger.error(f"Error reading MBTiles file: {e}")
        return None

def create_mbtiles_median_IM(CityName: str, savedir = RESULTDIR / 'mbtiles', Overwrite: bool = False, minzoom: int = 8, maxzoom: int = 16):
    """
    Create a single MBTiles file containing the median IM data for all ruptures and all periods for a city.
    The data structure is optimized to store one feature per site, with IM values organized as a 2D array 
    [rupture_idx][period_idx] = im_value, eliminating redundant coordinate storage.
    
    Parameters:
        CityName (str): Name of the city.
        savedir (str): Directory to save the output files.
        Overwrite (bool): Whether to overwrite existing files. Default is False.
        minzoom (int): Minimum zoom level for MBTiles. Default is 8.
        maxzoom (int): Maximum zoom level for MBTiles. Default is 16.
        
    Returns:
        str: Path to the created MBTiles file, or None if failed. The file is named as 'IM_median_all_{CityName}.mbtiles'. 
    Data Structure:
        MBTiles文件包含优化的IM站点数据，采用以下结构：
        
        Vector Tiles (MVT格式):
        - 每个站点为一个Point几何要素
        - 属性包含完整的2D强度数据矩阵
        
        Feature Properties:
            - site_id: 站点唯一标识符
            - lon, lat: 站点经纬度坐标
            - im_matrix: 2D数组 [rupture_idx][period_idx] = IM值
            - n_ruptures: 地震破裂情景数量
            - n_periods: 周期数量
        
        MBTiles Metadata表:
            - city: 城市名称
            - data_type: 数据类型标识 
            - total_sites: 站点总数
            - total_ruptures: 破裂情景总数
            - total_periods: 周期总数
            - ruptures_metadata: JSON格式的破裂情景元数据
            - periods_metadata: JSON格式的周期元数据
            - data_structure: 数据结构说明
            - data_reduction_info: 数据压缩信息
        
        数据访问方式:
        通过ruptures_metadata和periods_metadata解析matrix索引对应的物理参数。
    """
    
    logger.info(f"Starting creation of IM scatter points MBTiles for city: {CityName}")
    
    # Create save directory if it doesn't exist
    if not Path(savedir).exists():
        logger.info(f"Creating save directory: {savedir}")
        Path(savedir).mkdir(parents=True, exist_ok=True)
    
    # Create unified MBTiles filename for scatter points
    mbtiles_filename = f'IM_median_all_{CityName}.mbtiles'
    mbtiles_path = Path(savedir) / mbtiles_filename
    
    # Skip if file already exists and not overwriting
    if mbtiles_path.exists() and not Overwrite:
        logger.info(f'IM median MBTiles file already exists: {mbtiles_filename}. Skipping...')
        return str(mbtiles_path)
    
    try:
        # Read earthquake ruptures
        rup_df = EQsources.ReadCityRuptures(CityName)
        logger.info(f"Processing {len(rup_df)} earthquake ruptures for {CityName}")
        
        # Get site file for coordinates
        sitefile = IMSim.IM_SIM_DIR / CityName / 'SiteFile.txt'
        if not sitefile.exists():
            logger.error(f"Site file not found: {sitefile}")
            return None
            
        IDvec, lonVec, latVec = Utilities.ReadIDofMeshedSiteFile(sitefile)
        logger.info(f"Loaded {len(IDvec)} sites from site file")        
        # Initialize data structures for optimized storage
        site_features = {}  
        # Dictionary to store one feature per site
        ruptures_metadata = {}
        periods_metadata = {}
        
        # First pass: collect all rupture and period metadata
        all_IM_data = {}  # {site_idx: {rup_idx: {period_idx: im_value}}}
        
        # Process each rupture
        for i_rup in tqdm(range(len(rup_df)), desc="Processing ruptures for IM median MBTiles"):
            rup_id = rup_df.iloc[i_rup]['rup_id']
            IMresultsDir = IMSim.IM_SIM_DIR / CityName / f'rup_id_{rup_id}/'
            
            if not IMresultsDir.exists():
                logger.warning(f"IM results directory not found: {IMresultsDir}")
                continue
            
            # Read IM simulation results (median)
            IM_mat, Tvec = IMSim._Read_IMSim_results(IMresultsDir, median=True)
            logger.debug(f"Loaded IM data: {IM_mat.shape[0]} sites, {IM_mat.shape[1]} simulations, {IM_mat.shape[2]} periods")
            
            # Store rupture metadata
            ruptures_metadata[i_rup] = {
                'rupture_id': str(rup_id),
                'magnitude': float(rup_df.iloc[i_rup]['mag']),
                'index': int(i_rup)
            }
            
            # Store period metadata (only need to do this once)
            if not periods_metadata:
                for iTplot in range(len(Tvec)):
                    Sa_period = Tvec[iTplot]
                    periods_metadata[iTplot] = {
                        'period_value': float(Sa_period),
                        'index': int(iTplot)
                    }
            
            # Process each period for this rupture
            for iTplot in range(len(Tvec)):
                try:
                    # Extract IM values for this period (median is always iSim=0)
                    im_values = IM_mat[:, 0, iTplot]  # median values for this period

                    # Store IM values for each site
                    for site_idx in range(len(IDvec)):
                        im_value = float(im_values[site_idx])
                        
                        # Skip invalid values
                        if np.isnan(im_value) or np.isinf(im_value):
                            continue
                        
                        # Initialize nested dictionaries if needed
                        if site_idx not in all_IM_data:
                            all_IM_data[site_idx] = {}
                        if i_rup not in all_IM_data[site_idx]:
                            all_IM_data[site_idx][i_rup] = {}
                        
                        # Store the IM value
                        all_IM_data[site_idx][i_rup][iTplot] = round(im_value, 6)
                    
                    logger.debug(f"Processed period {Tvec[iTplot]:.2f}s for rupture {i_rup}")
                    
                except Exception as e:
                    logger.error(f"Error processing period {Tvec[iTplot]:.2f}s, rupture {i_rup}: {e}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    continue
        
        # Second pass: create optimized features with 2D arrays for each site
        all_features = []
        for site_idx in tqdm(all_IM_data.keys(), desc="Creating site features"):
            try:
                lon = float(lonVec[site_idx])
                lat = float(latVec[site_idx])
                
                # Build 2D array of IM values [n_ruptures x n_periods]
                n_ruptures = len(ruptures_metadata)
                n_periods = len(periods_metadata)
                im_array = [[None for _ in range(n_periods)] for _ in range(n_ruptures)]
                
                # Fill the array with available data
                for rup_idx in all_IM_data[site_idx]:
                    for period_idx in all_IM_data[site_idx][rup_idx]:
                        im_array[rup_idx][period_idx] = all_IM_data[site_idx][rup_idx][period_idx]
                
                # Create optimized feature with 2D array storage
                feature = {
                    'type': 'Feature',
                    'geometry': {
                        'type': 'Point',
                        'coordinates': [lon, lat]
                    },
                    'properties': {
                        'site_id': int(IDvec[site_idx]),
                        'im_matrix': im_array,  # 2D array: [rupture_idx][period_idx] = im_value
                    }
                }
                
                all_features.append(feature)
                
            except Exception as e:
                logger.error(f"Error creating feature for site {site_idx}: {e}")
                continue
        
        logger.info(f"Created {len(all_features)} optimized site features with 2D IM arrays")
          # Create the unified GeoJSON file
        if not all_features:
            logger.warning(f"No optimized site features were generated for {CityName}")
            return None
        
        logger.info(f"Generated {len(all_features)} optimized site features (one per site) with 2D IM arrays")
        logger.info(f"Each site contains IM data for {len(ruptures_metadata)} ruptures and {len(periods_metadata)} periods")
        
        # Create temporary GeoJSON file for point data
        temp_geojson = Path(savedir) / f'temp_IM_median_all_{CityName}.geojson'

        try:
            # Create GeoJSON collection with metadata
            geojson_collection = {
                'type': 'FeatureCollection',
                'crs': {
                    'type': 'name',
                    'properties': {
                        'name': 'urn:ogc:def:crs:OGC:1.3:CRS84'  # WGS84
                    }
                },                
                # Add global metadata
                'metadata': {
                    'city': CityName,
                    'data_type': 'optimized_median_im_sites',
                    'total_sites': len(all_features),
                    'total_ruptures': len(ruptures_metadata),
                    'total_periods': len(periods_metadata),
                    'ruptures': ruptures_metadata,
                    'periods': periods_metadata,
                    'description': 'Optimized IM site data with 2D arrays for cloud map visualization',
                    'data_structure': 'Each site feature contains im_matrix[rupture_idx][period_idx] = im_value'
                },
                'features': all_features
            }
            
            # Write temporary GeoJSON file
            with open(temp_geojson, 'w', encoding='utf-8') as f:
                json.dump(geojson_collection, f, ensure_ascii=False, separators=(',', ':'))
            
            logger.info(f"Created temporary GeoJSON with point data: {temp_geojson.name}")
            
            # Check ogr2ogr availability
            ogr2ogr_path = shutil.which('ogr2ogr')
            if not ogr2ogr_path:
                logger.error("ogr2ogr command not found in PATH. Please ensure GDAL is installed.")
                logger.error("Install with: conda install -c conda-forge gdal")
                return None            
            # Convert to MBTiles using ogr2ogr
            cmd = [
                ogr2ogr_path,
                '-f', 'MBTiles',
                str(mbtiles_path),
                str(temp_geojson),
                '-s_srs', 'EPSG:4326',  # Source: WGS84
                '-t_srs', 'EPSG:3857',  # Target: Web Mercator
                '-dsco', f'MINZOOM={minzoom}',
                '-dsco', f'MAXZOOM={maxzoom}',
                '-dsco', 'TYPE=overlay',
                '-dsco', f'NAME=IM Sites {CityName}',
                '-dsco', f'DESCRIPTION=IM Site Data for {CityName} with 2D arrays',                
                '-dsco', 'COMPRESS=YES',  
                '-lco', f'NAME=im_sites_optimized_{CityName}',
                '-nln', f'im_sites_optimized_{CityName}'
            ]
            
            logger.debug(f"Executing: {' '.join(cmd)}")
            
            # Execute ogr2ogr command
            env = os.environ.copy()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                env=env
            )
            # Check result
            if result.stdout:
                logger.debug(f"ogr2ogr stdout: {result.stdout}")
            if result.stderr:
                logger.warning(f"ogr2ogr stderr: {result.stderr}")
            if result.returncode != 0:
                logger.error(f"ogr2ogr failed with return code: {result.returncode}")
                return None            
            # Verify MBTiles file was created
            if mbtiles_path.exists():
                logger.info(f"Successfully created optimized IM sites MBTiles: {mbtiles_filename}")
                logger.info(f"File size: {mbtiles_path.stat().st_size / (1024*1024):.1f} MB")
                logger.info(f"Total site features: {len(all_features)}")
                logger.info(f"Ruptures processed: {len(ruptures_metadata)}")
                logger.info(f"Periods per rupture: {len(periods_metadata)}")
                logger.info(f"Data reduction: {len(IDvec) * len(ruptures_metadata) * len(periods_metadata)} points -> {len(all_features)} sites")
                
                # 手动添加metadata，因为-mo参数可能不被MBTiles驱动程序完全支持
                try:
                    import sqlite3
                    conn = sqlite3.connect(str(mbtiles_path))
                    cursor = conn.cursor()
                    
                    # Add custom metadata fields
                    metadata_items = [
                        ('city', CityName),
                        ('data_type', 'optimized_median_im_sites'),
                        ('total_sites', str(len(all_features))),
                        ('total_ruptures', str(len(ruptures_metadata))),
                        ('total_periods', str(len(periods_metadata))),
                        ('ruptures_metadata', json.dumps(ruptures_metadata, ensure_ascii=False)),
                        ('periods_metadata', json.dumps(periods_metadata, ensure_ascii=False)),
                        ('data_structure', 'Each site feature contains im_matrix[rupture_idx][period_idx] = im_value'),
                        ('data_reduction_info', f'{len(IDvec) * len(ruptures_metadata) * len(periods_metadata)} points -> {len(all_features)} sites')
                    ]
                    
                    for name, value in metadata_items:
                        cursor.execute('INSERT OR REPLACE INTO metadata (name, value) VALUES (?, ?)', (name, value))
                    
                    conn.commit()
                    conn.close()
                    
                    logger.info(f"Successfully added custom metadata to MBTiles file")
                    
                except Exception as e:
                    logger.warning(f"Failed to add custom metadata to MBTiles: {e}")
                
                # 验证metadata是否正确写入
                expected_keys = ['city', 'data_type', 'total_sites', 'total_ruptures', 'total_periods', 'ruptures_metadata', 'periods_metadata', 'data_structure']
                verify_mbtiles_metadata(str(mbtiles_path), expected_keys)
                
                return str(mbtiles_path)
            else:
                logger.error(f"Optimized IM sites MBTiles file was not created: {mbtiles_filename}")
                return None
                
        except Exception as e:
            logger.error(f"Error creating optimized IM sites MBTiles: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
            
        finally:
            # Clean up temporary file
            if temp_geojson.exists():
                temp_geojson.unlink()
            
    except Exception as e:
        logger.error(f"Unexpected error creating IM points MBTiles: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

def create_mbtiles_city_regional_losses(CityName: str, ResultsDir = RegionalLossSim.RESULTDIR, LossType: str = 'DS_Struct', iSim: int = 0, savedir = RESULTDIR / 'mbtiles', Overwrite: bool = False, minzoom: int = 8, maxzoom: int = 16):
    '''
    使用 GDAL/ogr2ogr 命令行工具创建 MBTiles 文件。
    
    Parameters:
        CityName (str): Name of the city.
        ResultsDir (str): Directory containing the loss results.
        LossType (str): Type of loss to plot. Default is 'DS_Struct'.
        iSim (int): Index of the simulation. Default is 0.
        savedir (str): Directory to save the output files.
        Overwrite (bool): Whether to overwrite existing files. Default is False.
        minzoom (int): Minimum zoom level. Default is 8.
        maxzoom (int): Maximum zoom level. Default is 16.
    
    Returns:
        str: Path to the generated MBTiles file, or None if failed  
          
    Data Structure:
        MBTiles文件包含区域地震损失数据，采用以下结构：
        
        Vector Tiles (MVT格式):
        - 每个建筑物为一个Polygon几何要素
        - 包含所有地震破裂情景的损失值
        
        Feature Properties:
            - BldID: 建筑物唯一标识符
            - geometry: 建筑物多边形坐标
            - loss_values: 包含所有破裂情景损失值的数组/对象
            - 具体损失字段根据LossType变化:
              * DS_Struct: 结构损伤状态
              * DS_NStruct: 非结构损伤状态  
              * DS_Content: 内容物损伤状态
              * Loss_*: 对应的经济损失值
        
        MBTiles Metadata表:
            - ruptures_metadata: JSON格式的破裂情景元数据
            - periods_metadata: JSON格式的周期元数据 (如适用)
            - geometry_type: 几何类型 (Polygon)
            - data_type: 损失类型标识
            - timestamp: 创建时间戳
            - custom_description: 数据集描述
        
        使用场景:
        适用于WebGIS平台展示区域地震损失分布，支持多层级缩放和交互查询。
    '''
    
    logger.info(f"Creating MBTiles with ogr2ogr for city: {CityName}, LossType: {LossType}")

    # Create save directory if it doesn't exist
    if not Path(savedir).exists():
        logger.info(f"Creating save directory: {savedir}")
        Path(savedir).mkdir(parents=True, exist_ok=True)
    
    # 创建临时 GeoJSON 文件
    temp_geojson = Path(savedir) / f'temp_{CityName}_{LossType}_{iSim}.geojson'
    mbtiles_path = Path(savedir) / f'RegionalLoss_{CityName}_{LossType}_{iSim}_ogr2ogr.mbtiles'
    
    if mbtiles_path.exists() and not Overwrite:
        logger.info(f"MBTiles file already exists: {mbtiles_path.name}")
        return str(mbtiles_path)
    
    try:
        
        # 首先生成完整的 GeoJSON 数据
        rup_df = EQsources.ReadCityRuptures(CityName)
        logger.info(f"Processing {len(rup_df)} ruptures")
        
        all_features = []
        
        # 一次性获取所有rupture的数据，避免重复循环
        logger.info("Generating GeoJSON data for all ruptures at once...")
        geojson_data = RegionalLossSim.create_geojson_regional_losses(
            CityName=CityName,
            DistrictName=None,
            ResultsDir=ResultsDir,
            LossType=LossType,
            i_rup=None,  # 处理所有rupture
            iSim=iSim        
            )
        
        if geojson_data:
            data = json.loads(geojson_data)
            features = data.get('features', [])
            
            # 直接使用包含所有rupture损失值的特征，不展开
            all_features = features
            
            logger.info(f"Generated {len(all_features)} features with {len(rup_df)} rupture values per building")
        else:
            logger.error("Failed to generate GeoJSON data")
            return None
 
        # 写入临时 GeoJSON 文件，确保包含坐标系信息
        geojson_collection = {
            'type': 'FeatureCollection',
            'crs': {
                'type': 'name',
                'properties': {
                    'name': 'urn:ogc:def:crs:OGC:1.3:CRS84'  # 明确指定WGS84经纬度坐标系
                }
            },
            'features': all_features
        }
        # 写入临时 GeoJSON 文件
        with open(temp_geojson, 'w', encoding='utf-8') as f:
            json.dump(geojson_collection, f, ensure_ascii=False, separators=(',', ':'))
        
        logger.info(f"Created temporary GeoJSON with {len(all_features)} features")
        
        # 创建metadata信息
        metadata = {
            'ruptures_metadata': [{'rupture_id': int(row['rup_id']), 'magnitude': float(row['mag'])} for _, row in rup_df.iterrows()],
            'data_type': LossType,
            'timestamp': datetime.now().isoformat(),
            'description': f'Regional seismic losses for {CityName}, {LossType}, simulation {iSim}'
        }
        
        # 检查 ogr2ogr 是否可用
        ogr2ogr_path = shutil.which('ogr2ogr')
        if not ogr2ogr_path:
            logger.error("ogr2ogr command not found in PATH. Please ensure GDAL is installed.")
            logger.error("Install with: conda install -c conda-forge gdal")
            return None
        
        logger.info(f"Found ogr2ogr at: {ogr2ogr_path}")            
        # 使用 ogr2ogr 转换为 MBTiles，添加更多兼容性参数
        cmd = [
            ogr2ogr_path,  # 使用完整路径
            '-f', 'MBTiles',
            str(mbtiles_path),
            str(temp_geojson),
            '-s_srs', 'EPSG:4326',  # 明确指定源坐标系为WGS84
            '-t_srs', 'EPSG:3857',  # 明确指定目标坐标系为Web Mercator
            '-dsco', f'MINZOOM={minzoom}',
            '-dsco', f'MAXZOOM={maxzoom}',
            '-dsco', 'TYPE=overlay',
            '-dsco', f'NAME=Regional Loss {CityName}',
            '-dsco', f'DESCRIPTION=Regional seismic losses for {CityName}, {LossType}',
            '-dsco', 'COMPRESS=YES',
            '-dsco', 'MAX_FEATURES=1500',    
            '-lco', f'NAME=temp_{CityName}_{LossType}_{iSim}',  # 明确指定层名称
            '-nln', f'temp_{CityName}_{LossType}_{iSim}'  # 新图层名称
        ]
        
        logger.info(f"Executing: {' '.join(cmd)}")
        
        # 执行 ogr2ogr 命令，添加更多环境信息
        env = os.environ.copy()
        
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            check=False,  # 不自动抛出异常，手动检查返回码
            env=env  # 明确传递环境变量
        )
        
        # 记录输出信息
        if result.stdout:
            logger.info(f"ogr2ogr stdout: {result.stdout}")
        if result.stderr:
            logger.warning(f"ogr2ogr stderr: {result.stderr}")
        if result.returncode != 0:
            logger.error(f"ogr2ogr failed with return code: {result.returncode}")
            return None        
        # 清理临时文件
        if temp_geojson.exists():
            temp_geojson.unlink()
        
        if mbtiles_path.exists():
            logger.info(f"Successfully created MBTiles with ogr2ogr: {mbtiles_path.name}")
            logger.info(f"File size: {mbtiles_path.stat().st_size / (1024*1024):.1f} MB")
            logger.info(f"Total features: {len(all_features)}")
            
            # 手动添加metadata，因为-mo参数可能不被MBTiles驱动程序完全支持
            try:
                import sqlite3
                conn = sqlite3.connect(str(mbtiles_path))
                cursor = conn.cursor()
                
                # Add custom metadata fields
                metadata_items = [
                    ('ruptures_metadata', json.dumps(metadata["ruptures_metadata"])),
                    ('periods_metadata', json.dumps(metadata.get("periods_metadata", []))),
                    ('geometry_type', metadata.get("geometry_type", "Polygon")),
                    ('data_type', metadata["data_type"]),
                    ('timestamp', metadata["timestamp"]),
                    ('custom_description', metadata["description"])
                ]
                
                for name, value in metadata_items:
                    cursor.execute('INSERT OR REPLACE INTO metadata (name, value) VALUES (?, ?)', (name, value))
                
                conn.commit()
                conn.close()
                
                logger.info(f"Successfully added custom metadata to regional losses MBTiles file")
                
            except Exception as e:
                logger.warning(f"Failed to add custom metadata to regional losses MBTiles: {e}")
            
            return str(mbtiles_path)
        else:
            logger.error("MBTiles file was not created")
            return None
        
    except Exception as e:
        logger.error(f"Unexpected error creating MBTiles with ogr2ogr: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None
    finally:
        # 确保清理临时文件
        if temp_geojson.exists():
            temp_geojson.unlink()

def verify_mbtiles_metadata(mbtiles_path, expected_metadata_keys=None):
    """
    验证MBTiles文件中的metadata是否正确写入
    
    Parameters:
        mbtiles_path (str): MBTiles文件路径
        expected_metadata_keys (list): 期望的metadata键列表
    
    Returns:
        dict: 验证结果，包含存在的metadata和缺失的keys
    """
    if not Path(mbtiles_path).exists():
        logger.error(f"MBTiles file not found: {mbtiles_path}")
        return None
    
    try:
        conn = sqlite3.connect(mbtiles_path)
        cursor = conn.cursor()
        
        # 获取所有metadata
        cursor.execute('SELECT name, value FROM metadata')
        metadata = dict(cursor.fetchall())
        
        conn.close()
        
        result = {
            'metadata': metadata,
            'metadata_count': len(metadata)
        }
        
        if expected_metadata_keys:
            missing_keys = [key for key in expected_metadata_keys if key not in metadata]
            result['missing_keys'] = missing_keys
            result['has_all_expected'] = len(missing_keys) == 0
        
        logger.info(f"MBTiles metadata verification:")
        logger.info(f"  Total metadata entries: {len(metadata)}")
        for key, value in metadata.items():
            # 限制长值的显示长度
            display_value = value[:100] + "..." if len(str(value)) > 100 else value
            logger.info(f"  {key}: {display_value}")
        
        if expected_metadata_keys and result.get('missing_keys'):
            logger.warning(f"  Missing expected keys: {result['missing_keys']}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error verifying MBTiles metadata: {e}")
        return None

def create_mapdata_IM(CityName: str, savedir = RESULTDIR / 'IMmap', iSim: int = 0, Overwrite: bool = False):
    """
    Create map data containing both median and random IM data for all ruptures and all periods for a city.
    Data is saved as HDF5 format, organized by rupture groups.
    
    Parameters:
        CityName (str): Name of the city.
        savedir (str): Directory to save the output files.
        iSim (int): Index of the simulation for random data. Default is 0.
        Overwrite (bool): Whether to overwrite existing files. Default is False.
        
    Returns:
        str: Path to the HDF5 file if successful. Returns None if failed.
        
    HDF5 Data Structure:
        HDF5文件按rupture为group组织数据，采用以下结构：
        
        IM_mapdata_{CityName}.hdf5
        ├── metadata (attributes)
        │   ├── city: 城市名称
        │   ├── data_type: 'complete_im_mapdata_hdf5'
        │   ├── total_sites: 站点总数
        │   ├── total_ruptures: 破裂情景总数
        │   ├── total_periods: 周期总数
        │   ├── total_simulations: 总模拟次数
        │   ├── selected_iSim: 选择的模拟索引
        │   └── created_at: 创建时间
        ├── site_coordinates
        │   ├── site_id: 1D array of site IDs
        │   ├── longitude: 1D array of site longitudes
        │   └── latitude: 1D array of site latitudes
        ├── periods: 1D array of Sa periods
        ├── rupture_0/
        │   ├── metadata (attributes): rupture_id, magnitude
        │   ├── im_median: 2D array [n_sites, n_periods]
        │   └── im_random: 2D array [n_sites, n_periods] (selected iSim)
        ├── rupture_1/
        │   ├── metadata (attributes): rupture_id, magnitude
        │   ├── im_median: 2D array [n_sites, n_periods]
        │   └── im_random: 2D array [n_sites, n_periods] (selected iSim)
        └── ...
    """
    logger.info(f"Starting creation of IM map data in HDF5 format (median + random iSim={iSim}) for city: {CityName}")
    
    # Create save directory if it doesn't exist
    if not Path(savedir).exists():
        logger.info(f"Creating save directory: {savedir}")
        Path(savedir).mkdir(parents=True, exist_ok=True)
    
    # Create output filename for HDF5 format
    output_filename = f'IM_mapdata_{CityName}.hdf5'
    output_path = Path(savedir) / output_filename
    
    # Skip if file already exists and not overwriting
    if output_path.exists() and not Overwrite:
        logger.info(f'IM map data file already exists: {output_filename}. Skipping...')
        return str(output_path)
    
    try:
        # Read earthquake ruptures
        rup_df = EQsources.ReadCityRuptures(CityName)
        logger.info(f"Processing {len(rup_df)} earthquake ruptures for {CityName}")
        
        # Get site file for coordinates
        sitefile = IMSim.IM_SIM_DIR / CityName / 'SiteFile.txt'
        if not sitefile.exists():
            logger.error(f"Site file not found: {sitefile}")
            return None
            
        IDvec, lonVec, latVec = Utilities.ReadIDofMeshedSiteFile(sitefile)
        logger.info(f"Loaded {len(IDvec)} sites from site file")
        
        # Initialize data structures for HDF5 storage
        ruptures_metadata = {}
        periods_metadata = {}
        n_simulations = 0  # Will be set after reading first rupture data
        
        # 首先获取第一个破裂情景的数据来确定数组维度和创建HDF5文件结构
        first_rup_processed = False
        
        # Create and initialize HDF5 file with streaming processing
        logger.info(f"Creating HDF5 file with streaming processing: {output_filename}")
        
        try:
            with h5py.File(output_path, 'w') as hf:
                # Process each rupture and stream data directly to HDF5 file
                for i_rup in tqdm(range(len(rup_df)), desc="Writting IM map data (streaming)"):
                    rup_id = rup_df.iloc[i_rup]['rup_id']
                    IMresultsDir = IMSim.IM_SIM_DIR / CityName / f'rup_id_{rup_id}/'
                    
                    if not IMresultsDir.exists():
                        logger.warning(f"IM results directory not found: {IMresultsDir}")
                        continue
                    
                    # Read IM simulation results (median)
                    IM_mat_median, Tvec = IMSim._Read_IMSim_results(IMresultsDir, median=True)
                    logger.debug(f"Loaded IM median data: {IM_mat_median.shape[0]} sites, {IM_mat_median.shape[1]} simulations, {IM_mat_median.shape[2]} periods")
                    
                    # Read IM simulation results (random)
                    IM_mat_random, _ = IMSim._Read_IMSim_results(IMresultsDir, median=False)
                    logger.debug(f"Loaded IM random data: {IM_mat_random.shape[0]} sites, {IM_mat_random.shape[1]} simulations, {IM_mat_random.shape[2]} periods")
                    
                    # Set n_simulations from the first rupture
                    if n_simulations == 0:
                        n_simulations = IM_mat_random.shape[1]
                    
                    # Store rupture metadata
                    ruptures_metadata[i_rup] = {
                        'rupture_id': str(rup_id),
                        'magnitude': float(rup_df.iloc[i_rup]['mag']),
                        'index': int(i_rup)
                    }
                    
                    # Store period metadata (only need to do this once)
                    if not periods_metadata:
                        for iTplot in range(len(Tvec)):
                            Sa_period = Tvec[iTplot]
                            periods_metadata[iTplot] = {
                                'period_value': float(Sa_period),
                                'index': int(iTplot)
                            }
                    
                    # Initialize HDF5 file structure after first rupture data is loaded
                    if not first_rup_processed:
                        # Create global metadata attributes
                        hf.attrs['city'] = CityName
                        hf.attrs['data_type'] = 'complete_im_mapdata_hdf5'
                        hf.attrs['total_sites'] = len(IDvec)
                        hf.attrs['total_ruptures'] = len(rup_df)  # Use full rupture count
                        hf.attrs['total_periods'] = len(periods_metadata)
                        hf.attrs['total_simulations'] = n_simulations
                        hf.attrs['selected_iSim'] = iSim
                        hf.attrs['created_at'] = datetime.now().isoformat()
                        hf.attrs['description'] = f'IM data organized by ruptures for {CityName} (median + random iSim={iSim})'
                        
                        # Create site coordinates group
                        site_group = hf.create_group('site_coordinates')
                        site_group.create_dataset('site_id', data=np.array(IDvec, dtype=np.int32))
                        site_group.create_dataset('longitude', data=np.array(lonVec, dtype=np.float64))
                        site_group.create_dataset('latitude', data=np.array(latVec, dtype=np.float64))
                        
                        # Create periods dataset
                        period_values = [periods_metadata[i]['period_value'] for i in sorted(periods_metadata.keys())]
                        hf.create_dataset('periods', data=np.array(period_values, dtype=np.float64))
                        
                        first_rup_processed = True
                        logger.debug("Initialized HDF5 file structure")
                    
                    # Process current rupture data
                    try:
                        # Extract median IM values (iSim=0 for median)
                        median_data = IM_mat_median[:, 0, :]  # Shape: [n_sites, n_periods]
                        
                        # Extract random IM values for selected iSim
                        if iSim < n_simulations:
                            random_data = IM_mat_random[:, iSim, :]  # Shape: [n_sites, n_periods]
                        else:
                            logger.warning(f"iSim={iSim} exceeds available simulations ({n_simulations}), using NaN")
                            random_data = np.full((len(IDvec), len(Tvec)), np.nan, dtype=np.float32)
                        
                        # Clean data, replacing invalid values with NaN
                        median_data = np.where(np.isfinite(median_data), median_data, np.nan)
                        random_data = np.where(np.isfinite(random_data), random_data, np.nan)
                        
                        # Create rupture group and immediately write data to HDF5
                        rup_group = hf.create_group(f'rupture_{i_rup}')
                        
                        # Add rupture metadata as attributes
                        rup_group.attrs['rupture_id'] = str(rup_id)
                        rup_group.attrs['magnitude'] = float(rup_df.iloc[i_rup]['mag'])
                        rup_group.attrs['index'] = i_rup
                        
                        # Create datasets for IM data and write directly to disk
                        # im_median: 2D array [n_sites, n_periods]
                        rup_group.create_dataset('im_median', data=median_data.astype(np.float32), compression='gzip', compression_opts=6, dtype=np.float32)
                        
                        # im_random: 2D array [n_sites, n_periods] for selected iSim
                        rup_group.create_dataset('im_random', data=random_data.astype(np.float32), compression='gzip', compression_opts=6, dtype=np.float32)
                        
                        logger.debug(f"Streamed rupture {i_rup} data to HDF5: {rup_id}, median shape: {median_data.shape}, random shape: {random_data.shape}")
                        
                        # Clear variables to free memory immediately after writing
                        del median_data, random_data, IM_mat_median, IM_mat_random
                        
                    except Exception as e:
                        logger.error(f"Error processing rupture {i_rup}: {e}")
                        import traceback
                        logger.error(f"Traceback: {traceback.format_exc()}")
                        continue
                
                # Check if we processed any data
                if not first_rup_processed:
                    logger.warning(f"No rupture data was processed for {CityName}")
                    return None
            
            # File is automatically closed when exiting with block
            logger.info(f"Successfully saved IM map data as HDF5 with streaming: {output_filename}")
            logger.info(f"File size: {output_path.stat().st_size / (1024*1024):.1f} MB")
            logger.info(f"Total sites: {len(IDvec)}")
            logger.info(f"Ruptures processed: {len(ruptures_metadata)}")
            logger.info(f"Periods per rupture: {len(periods_metadata)}")
            logger.info(f"Selected simulation index: {iSim}")
            logger.info(f"Total simulations available: {n_simulations}")
            logger.info(f"Data structure: HDF5 with rupture groups, each containing 2D arrays [sites x periods]")
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error creating/writing HDF5 file: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
            
    except Exception as e:
        logger.error(f"Unexpected error creating IM map data: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None
