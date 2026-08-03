"""
地震源与随机地震事件处理模块。

使用流程：读取 GEM China 震源模型，通过 OpenQuake 生成城市随机地震目录，
再读取震源、事件和震级频率结果供地震动模拟及年均风险计算使用。

输入：城市名称、城市空间范围、OpenQuake 震源与事件结果。
输出：震源/事件表、地图，以及震级年均超越频率曲线。
"""

from . import Utilities
from . import plot_city
from . import Logger

from pathlib import Path
from openquake.commands import engine, reset, purge
import pandas as pd
import numpy as np
import shutil
import folium
import webbrowser
import xyzservices
import math
import re

CHN_EQS_DIR = Path(__file__).parent / 'Data/GEM-CHN-v2021.1.0'
EQ_SOURCES_OUTPUT_DIR = Path(__file__).parent / 'results/EQsources'

# Create module logger
logger = Logger.get_logger('EQsources')

def plot_earthquake_events_folium(CityName: str, BaseMap='Cartodb Positron', opacity=0.5):
    # Plot the earthquake events
    logger.info(f"Starting to plot earthquake events for city: {CityName}")

    
    results_dir = EQ_SOURCES_OUTPUT_DIR / CityName

    rup_df = ReadCityRuptures(CityName)

    for file in results_dir.glob('*events*.csv'):
        events_d = pd.read_csv(file, header=1)
    
    # create a map
    if BaseMap == 'Cartodb Positron':
        tiles = BaseMap
        m = folium.Map(location=[rup_df['centroid_lat'].mean(), rup_df['centroid_lon'].mean()], zoom_start=10, tiles=tiles)
    elif BaseMap == 'OpenStreetMap.HOT':
        tiles = xyzservices.providers.OpenStreetMap.HOT
        m = folium.Map(location=[rup_df['centroid_lat'].mean(), rup_df['centroid_lon'].mean()], max_zoom=19, tiles=tiles)
        

    # add the rupture points
    for i in range(len(rup_df)):
        Create_a_Rupture_Polygon(m, rup_df.iloc[i])

    # add city bounds
    plot_city.Add_CityBoundary_to_FoliumMap(m, CityName)

    # adjust the map
    m.fit_bounds(m.get_bounds())

    # add location popup
    folium.LatLngPopup().add_to(m)    
    
    # save the map
    m.save(results_dir / 'ruptures.html')
    logger.info(f"Earthquake rupture map saved: {results_dir / 'ruptures.html'}")

    # open the map in the browser
    webbrowser.open_new( 'file://' + str((results_dir / 'ruptures.html').resolve()) )
    logger.info(f"Earthquake events plotting completed for city: {CityName}")

def Create_a_Rupture_Polygon(m: folium.Map = None, rupture: pd.Series = None) -> tuple:
    '''
    Create a rupture polygon on the map

    Parameters:
        m: folium.Map
        rupture: pd.Series. columns: mag, rake, centroid_lat, centroid_lon, strike, dip, centroid_depth, W_km (optional), length_km (optional)

    Returns:
        m: folium.Map with the rupture polygon added
        latlon_polygon: list of tuples (lat, lon) representing the corners of the rupture polygon
    '''
    if 'i_rup' not in rupture.index:
        i_rup = None
    else:
        i_rup = rupture['i_rup']
    mag = rupture['mag']
    rake = rupture['rake']
    if 'W_km' not in rupture.index:
        W_km = default_W_km(mag,rake)
    else:
        W_km = rupture['W_km']
    if 'length_km' not in rupture.index:
        length_km = default_length_km(mag, rake)
    else:
        length_km = rupture['length_km']
    lat_Hypo = rupture['centroid_lat']
    lon_Hypo = rupture['centroid_lon']
    strike = rupture['strike']
    dip = rupture['dip']
    centroid_depth = rupture['centroid_depth']
    RuptureNormal_x, RuptureNormal_y, RuptureNormal_z = Utilities.CalcRuptureNormal(strike, dip)
    latlon_polygon = Utilities.Calc4CornersEQRupt(W_km, length_km, lat_Hypo, lon_Hypo, RuptureNormal_x, RuptureNormal_y, RuptureNormal_z)
    popup_info = f'i_rup: {i_rup}<br>Magnitude: {mag}<br>Strike: {strike:.1f}<br>Dip: {dip:.1f}<br>Rake: {rake:.1f}<br>Depth: {centroid_depth:.1f}<br>Length: {length_km:.1f} km<br>Width: {W_km:.1f} km'
    latlon_polygon = [(lat, lon) for lat, lon in latlon_polygon]
    if m is not None:
        folium.Polygon(latlon_polygon, color='black', weight=1, fill=True, fill_color='red', fill_opacity=0.5, popup=popup_info).add_to(m)

    return m, latlon_polygon

# uknown parameters for EQ sources
def default_W_km(mag: float, lambda_degree: float) -> float:   
    # J Kaklamanos, L G Baise, D M Boore. Estimating Unknown Input Parameters when Implementing the NGA Ground-Motion Prediction Equations in Engineering Practice. Earthquake Spectra, 2011, 27(4): 1219-1235.
    if abs(lambda_degree) <= 45 or abs(lambda_degree) >= 135: 
        # strike-slip
        W_km = 10**(-0.76+0.27*mag)
    elif lambda_degree > 45 and lambda_degree < 135:
        # reverse
        W_km = 10**(-1.61+0.41*mag)
    else:
        # normal
        W_km = 10**(-1.14+0.35*mag)
    return W_km

def default_length_km(mag: float, lambda_degree: float = None) -> float:
    # Wells D L, Coppersmith K J. New empirical relationships among magnitude, rupture length, rupture width, rupture area, and surface displacement. Bulletin of the Seismological Society of America, 1994, 84(4): 974-1002.
    if lambda_degree is None:
        # unknown
        a = -3.22
        b = 0.69
    else:
        if abs(lambda_degree) <= 45 or abs(lambda_degree) >= 135: 
            # strike-slip
            a = -3.55
            b = 0.74
        elif lambda_degree > 45 and lambda_degree < 135:
            # reverse
            a = -2.86
            b = 0.63
        elif lambda_degree < -45 and lambda_degree > -135:
            # normal
            a = -2.01
            b = 0.50
    length_km = 10.0**(a + b*mag)
    return length_km

def default_Zhyp_km(mag: float, lambda_degree: float) -> float:
    # J Kaklamanos, L G Baise, D M Boore. Estimating Unknown Input Parameters when Implementing the NGA Ground-Motion Prediction Equations in Engineering Practice. Earthquake Spectra, 2011, 27(4): 1219-1235.
    if abs(lambda_degree) < 30:
        Zhyp_km = 5.63 + 0.68*mag
    else:
        Zhyp_km = 11.24 - 0.2*mag
    return Zhyp_km

class EQsource:

    jobpath = CHN_EQS_DIR / 'job.ini'
    results_dir = EQ_SOURCES_OUTPUT_DIR
    log_file = None

    def __init__(self, results_dir = None, CityName: str = None):
        logger.info("Initializing EQsource instance")

        if results_dir is not None:
            self.results_dir = results_dir
        elif CityName is not None:
            self.results_dir = self.results_dir / CityName

        self._clean_results()

        if not self.results_dir.exists():
            self.results_dir.mkdir(parents=True)

        self.log_file = self.results_dir / 'op_engine.log'
        logger.info(f"EQsource instance initialized, results directory: {self.results_dir}")

        purge.purge_all()
        logger.info("OpenQuake cache cleanup completed")

    def generate_earthquake_events(self, city_name: str):
        """
        job.ini
            number_of_logic_tree_samples = 2000. It means that 2000 years of earthquake events will be generated.
        """
        # Generate earthquake events for the target city
        logger.info(f"Starting to generate earthquake events for city: {city_name}")

        # Get the city bounds
        Bld_gdf = Utilities.get_CityBldData(city_name)
        minx, miny, maxx, maxy = Bld_gdf.total_bounds
        buffer = 0.1  # Increase the bounds by a small buffer
        logger.info(f"City bounds: [{minx:.4f}, {miny:.4f}, {maxx:.4f}, {maxy:.4f}], buffer: {buffer}")

        # Create a meshed site file
        self._CreateMeshedSiteFile(minx - buffer, maxx + buffer, miny - buffer, maxy + buffer)
        logger.info("Meshed site file creation completed")
        
        # Run the engine
        logger.info("Starting OpenQuake engine run")
        engine.main(run=[self.jobpath], log_file=self.log_file)
        logger.info("OpenQuake calculation completed")
        
        engine.main(export_outputs=(-1, self.results_dir))
        logger.info(f"Results export completed, saved to: {self.results_dir}")
        logger.info(f"Earthquake events generation completed for city: {city_name}")

    def _clean_results(self):
        # delete the results directory
        if self.results_dir.exists():
            logger.info(f"Cleaning existing results directory: {self.results_dir}")
            shutil.rmtree(self.results_dir)
            logger.info("Results directory cleanup completed")    
            
    def _CreateMeshedSiteFile(self, lonMin, lonMax, latMin, latMax, SiteFile = CHN_EQS_DIR / 'sites_chn.csv', MeshDist = 10000):
        logger.info(f"Creating meshed site file: {SiteFile}")
        logger.info(f"Longitude range: [{lonMin:.4f}, {lonMax:.4f}], latitude range: [{latMin:.4f}, {latMax:.4f}], mesh distance: {MeshDist}m")

        # a file is generated. Columns are speparated by space
        Utilities.CreateMeshedSiteFile(lonMin, lonMax, latMin, latMax, SiteFile = SiteFile, MeshDist = MeshDist)

        # modify the format of the file. Makecolumns separated by comma. 
        df = pd.read_csv(SiteFile, sep=' ', header=None)
        # name colomns. site_id, lon, lat, elevation_km, period1, vs30, Z25_km
        df.columns = ['site_id', 'lon', 'lat', 'elevation_km', 'period1', 'vs30', 'Z25_km']
        # drop some columns
        df.drop(columns=['elevation_km', 'period1', 'Z25_km'], inplace=True)

        df['z2pt5'] = 0.57 # depth (in km) to the 2.5 km/s shear-wave velocity
        df['z1pt0'] = 50  # depth (in meters) to the 1.0 km/s shear-wave velocity

        # change dytpe 
        df['site_id'] = df['site_id'].astype(np.uint32)
        df['lon'] = df['lon'].astype(np.float64)
        df['lat'] = df['lat'].astype(np.float64)
        df['vs30'] = df['vs30'].astype(np.float64)
        df['z2pt5'] = df['z2pt5'].astype(np.float64)
        df['z1pt0'] = df['z1pt0'].astype(np.float64)

        # reset site_id to start from 0
        df.set_index('site_id', inplace=True)
        df.reset_index(drop=True, inplace=True)

        df['vs30measured'] = True

        # output
        df.to_csv(SiteFile, index_label='site_id')
        logger.info(f"Site file processing completed, total {len(df)} sites")

def get_duration_years(CityName: str, results_dir = EQ_SOURCES_OUTPUT_DIR) -> float:
    """
    Read the duration years (num_rlzs) from the report_*.rst file for the target city
    
    Args:
        CityName (str): The name of the city for which to read the duration years.
        results_dir (Path): The directory where the results are stored. Defaults to EQ_SOURCES_OUTPUT_DIR.
    
    Returns:
        float: The number of realization years (num_rlzs) from the OpenQuake report file.
    """
    logger.info(f"Reading duration years from report file for city: {CityName}")
    
    results_dir = results_dir / CityName
    
    # Find the report file with pattern report_*.rst (stop at first match)
    try:
        report_file = next(results_dir.glob('report_*.rst'))
    except StopIteration:
        report_file = None
    
    if report_file is None:
        logger.error(f"No report_*.rst file found in {results_dir}")
        raise FileNotFoundError(f"No report_*.rst file found in {results_dir}")
    
    logger.info(f"Reading report file: {report_file}")
    
    # Read the file line by line and search for num_rlzs (stop at first match)
    pattern = r'num_rlzs\s*=\s*(\d+(?:,\d+)*)'
    
    with open(report_file, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                # Remove commas from the number and convert to float
                num_rlzs_str = match.group(1).replace(',', '')
                num_rlzs = float(num_rlzs_str)
                logger.info(f"Found num_rlzs = {num_rlzs} in report file")
                return num_rlzs
    
    # If we reach here, num_rlzs was not found
    logger.error("Could not find num_rlzs in the report file")
    raise ValueError("Could not find num_rlzs in the report file")


def ReadCityEvents(CityName: str, results_dir = EQ_SOURCES_OUTPUT_DIR) -> pd.DataFrame:
    """
    Read earthquake events for the target city

    Args:
        CityName (str): The name of the city for which to read earthquake event data.
        results_dir (Path): The directory where the results are stored. Defaults to EQ_SOURCES_OUTPUT_DIR.

    Returns:
        pd.DataFrame: A DataFrame containing the earthquake event data for the specified city. Columns include event_id, rup_id, rlz_id, year, ses_id
    """
    logger.info(f"Reading earthquake event data for city: {CityName}")

    results_dir = results_dir / CityName

    # 只匹配 events_数字.csv 格式，且只读取一个文件
    event_file = next(results_dir.glob('events_[0-9]*.csv'), None)
    if event_file is None:
        logger.error(f"No events_[0-9]*.csv file found in {results_dir}")
        raise FileNotFoundError(f"No events_[0-9]*.csv file found in {results_dir}")
    logger.info(f"Reading event file: {event_file}")
    events_df = pd.read_csv(event_file, header=1, dtype=str)

    logger.info(f"Event data reading completed, total {len(events_df)} records")
    return events_df

def ReadCityRuptures(CityName: str, results_dir = EQ_SOURCES_OUTPUT_DIR) -> pd.DataFrame:
    """
    Read unique earthquake Ruptures for the target city

    Args:
        CityName (str): The name of the city for which to read earthquake rupture data.
        results_dir (Path): The directory where the results are stored. Defaults to EQ_SOURCES_OUTPUT_DIR.

    Returns:
        pd.DataFrame: A DataFrame containing the earthquake rupture data for the specified city. Contains columns such as 'rup_id', 'trt', 'multiplicity', 'source_id', and various float columns for rupture parameters.
    """
    logger.info(f"Reading earthquake rupture data for city: {CityName}")

    results_dir = results_dir / CityName

    for file in results_dir.glob('*ruptures*.csv'):
        logger.info(f"Reading rupture file: {file}")
        rup_df = pd.read_csv(file, header=1, dtype={'rup_id': str, 'trt': str, 'multiplicity': int, 'source_id': str})
        float_columns = rup_df.columns.difference(['rup_id', 'trt', 'multiplicity', 'source_id'])
        rup_df[float_columns] = rup_df[float_columns].astype(float)
        # add i_rup
        rup_df['i_rup'] = rup_df.index

    logger.info(f"Rupture data reading completed, total {len(rup_df)} records")
    return rup_df

def _convert_incremental_mfd_to_exceedance(freq_mag_df: pd.DataFrame) -> pd.DataFrame:
    """
    将分震级档的增量发生频率转换为震级年均超越频率。

    Parameters:
        freq_mag_df: OpenQuake event_based_mfd 数据，至少包含 mag 和
            freq 两列；freq 表示对应震级档的年均发生频率。

    Returns:
        按震级升序排列的数据表。其中 Frequency 为大于等于相应震级的
        年均发生频率，因此随震级单调不增。
    """
    # 同一震级可能由多个震源贡献，先合并各震级档的增量频率。
    incremental_mfd = (
        freq_mag_df[['mag', 'freq']]
        .groupby('mag', as_index=False, sort=True)['freq']
        .sum()
    )

    # 从最大震级向最小震级累加，得到 M >= m 对应的年均超越频率。
    incremental_mfd['Frequency'] = incremental_mfd['freq'][::-1].cumsum()[::-1]
    return incremental_mfd.rename(columns={'mag': 'Magnitude'})[['Magnitude', 'Frequency']]


def get_mean_annual_freq_mag(CityName: str, results_dir=EQ_SOURCES_OUTPUT_DIR) -> pd.DataFrame:
    """
    读取 OpenQuake 增量震级频率，并计算给定城市的震级年均超越频率。

    Parameters:
        CityName: 城市名称。
        results_dir: OpenQuake 城市结果的根目录。

    Returns:
        包含 Magnitude 和 Frequency 两列的数据表。
    """
    logger.info(f"Calculating mean annual frequency of magnitude for city: {CityName}")

    city_results_dir = results_dir / CityName
    # OpenQuake 的 event_based_mfd 是分震级档的增量频率，不是超越频率。
    freq_mag_file = next(city_results_dir.glob('event_based_mfd_*.csv'), None)

    if freq_mag_file is None:
        logger.error(f"Mean annual frequency of magnitude file not found in: {city_results_dir}")
        return None

    incremental_mfd = pd.read_csv(freq_mag_file, header=1)
    exceedance_mfd = _convert_incremental_mfd_to_exceedance(incremental_mfd)
    logger.info(f"Mean annual frequency of magnitude exceedance loaded from {freq_mag_file}")
    return exceedance_mfd
