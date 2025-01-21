
from . import Utilities

from pathlib import Path
from openquake.commands import engine, reset, purge
import pandas as pd
import numpy as np
import shutil
import folium
import webbrowser
import xyzservices
import math

CHN_EQS_DIR = Path(__file__).parent / 'Data/GEM-CHN-v2021.1.0'
EQ_SOURCES_OUTPUT_DIR = Path(__file__).parent / 'results/EQsources'

def plot_earthquake_events_folium(CityName: str, BaseMap='Cartodb Positron', opacity=0.5):
    # Plot the earthquake events

    
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

    # adjust the map
    m.fit_bounds(m.get_bounds())

    # add location popup
    folium.LatLngPopup().add_to(m)

    # save the map
    m.save(results_dir / 'ruptures.html')

    # open the map in the browser
    webbrowser.open_new( 'file://' + str((results_dir / 'ruptures.html').resolve()) )

def Create_a_Rupture_Polygon(m: folium.Map, rupture: pd.Series):
    '''
    Create a rupture polygon on the map

    Parameters:
        m: folium.Map
        rupture: pd.Series. columns: mag, rake, centroid_lat, centroid_lon, strike, dip, centroid_depth, W_km (optional), length_km (optional)
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
    folium.Polygon(latlon_polygon, color='black', weight=1, fill=True, fill_color='red', fill_opacity=0.5, popup=popup_info).add_to(m)

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

        if results_dir is not None:
            self.results_dir = results_dir
        elif CityName is not None:
            self.results_dir = self.results_dir / CityName

        self._clean_results()

        if not self.results_dir.exists():
            self.results_dir.mkdir(parents=True)

        self.log_file = self.results_dir / 'op_engine.log'

        purge.purge_all()

    def generate_earthquake_events(self, city_name: str):
        # Generate earthquake events for the target city

        # Get the city bounds
        Bld_gdf = Utilities.get_CityBldData(city_name)
        minx, miny, maxx, maxy = Bld_gdf.total_bounds
        buffer = 0.1  # Increase the bounds by a small buffer

        # Create a meshed site file
        self._CreateMeshedSiteFile(minx - buffer, maxx + buffer, miny - buffer, maxy + buffer)
        
        # Run the engine
        engine.main(run=[self.jobpath], log_file=self.log_file)
        engine.main(export_outputs=(-1, self.results_dir))


    def _clean_results(self):
        # delete the results directory
        if self.results_dir.exists():
            shutil.rmtree(self.results_dir)

    def _CreateMeshedSiteFile(self, lonMin, lonMax, latMin, latMax, SiteFile = CHN_EQS_DIR / 'sites_chn.csv', MeshDist = 10000):

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


def ReadCityRuptures(CityName: str, results_dir = EQ_SOURCES_OUTPUT_DIR) -> pd.DataFrame:
    # Read the earthquake Ruptures for the target city

    results_dir = results_dir / CityName

    for file in results_dir.glob('*ruptures*.csv'):
        rup_df = pd.read_csv(file, header=1, dtype={'rup_id': str, 'trt': str, 'multiplicity': int})
        float_columns = rup_df.columns.difference(['rup_id', 'trt', 'multiplicity'])
        rup_df[float_columns] = rup_df[float_columns].astype(float)
        # add i_rup
        rup_df['i_rup'] = rup_df.index

    return rup_df