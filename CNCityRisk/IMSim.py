from . import Utilities
from . import ClassifyBld
from . import EQsources

import pandas as pd
import os
import math
import subprocess
from pathlib import Path
from pyproj import Transformer
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import griddata
import matplotlib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import folium
import webbrowser
import numpy as np
import branca.colormap as cm
import geojsoncontour
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from tqdm import tqdm
from typing import Generator, Tuple
import dask.array as da
import h5py

N_SIM_EACH_RUP = 10
IM_MAT_BLD_FILENAME = 'all_IM_mat_Bld.h5'
RUPID_2_ISIM_FILENAME = 'rupid_2_iSim.csv'
IM_SIM_DIR = Path(__file__).parent / 'results/IMsim/'

class SpatialIM():
    '''
    IM field simulation. 
    1. Create a EQ source file
    2. Create a meshed site file
    3. IM field simulation for the meshed sites
    4. Prepare building data for a city or a district
    5. Interpolate IM simulation results from mesh to building locations and periods
    *. The step 4 should be skipped if the obove steps are repeated for the same city
    '''

    # Path
    EQSourceFile: Path = None
    SiteFile: Path = None
    OutputDir: Path = None

    # Exe path
    IMSimExe: str = str((Path(__file__).parent / 'IMSim.exe').resolve())

    # building information. It is a list of dictionaries with keys 'centroid' and 'T'. 'centroid' is a list of [lon, lat], and 'T' is the period.
    BldInfo: list = None

    # Results for meshed sites
    IDvec: list = None # ID of each site
    lonVec: list = None # corresponding longitude
    latVec: list = None # corresponding latitude
    IM_mat: np.array = None # numpy array - [numel(ID_site), N_sim, i_T]
    Tvec: np.array = None # numpy array - periods corresponding to [i_T]

    def __init__(self):
        pass

    def CreateAExampleSourceFile(self, EQSourceFile='EQSource.txt'):
        '''
        Create a example EQ source file, located at Wuhan, China
        '''

        ifmedian = False  # False为随机
        M = 6.0
        N_sim = N_SIM_EACH_RUP
        seed = 2
        lon_0 = 114.278669
        lat_0 = 30.595883 
        W = 10**(-0.76 + 0.27 * M)
        length = math.exp(math.log(W * 1000 / 15) * 3 / 2) / 1000
        strike = 35
        delta = 52
        RuptureNormal_x = math.cos(math.radians(90 - delta)) * math.cos(math.radians(strike))
        RuptureNormal_y = math.cos(math.radians(90 - delta)) * math.sin(math.radians(strike))
        RuptureNormal_z = math.sin(math.radians(90 - delta))
        lambda_degree = 54  # rake angle
        Fhw = True
        Zhyp = 12
        region = 3
        nPCs = 10

        # Call the CreateSourceFile method
        self.CreateSourceFile(M, N_sim, lon_0, lat_0, W, length, RuptureNormal_x, RuptureNormal_y, RuptureNormal_z, lambda_degree, Zhyp, Fhw, ifmedian, seed, region, nPCs, EQSourceFile)

    def SetMeshedSiteFile(self, SiteFile):
        '''
        Specify a meshed site file
        '''
        if not Path(SiteFile).exists():
            raise FileNotFoundError(f"Site file {SiteFile} does not exist")
        
        self.SiteFile = Path(SiteFile).resolve()
        
        IDvec, lonVec, latVec = Utilities.ReadIDofMeshedSiteFile(SiteFile)

        self.IDvec = IDvec
        self.lonVec = lonVec
        self.latVec = latVec

        return IDvec, lonVec, latVec

    def CreateMeshedSiteFile(self, lonMin, lonMax, latMin, latMax, SiteFile = 'SiteFile.txt', MeshDist = 500):
        """
        Generate a meshed site file

        Parameters:
            MeshDist: float - mesh distance, m
            lonMin: float - minimum longitude, degrees
            lonMax: float - maximum longitude, degrees
            latMin: float - minimum latitude, degrees
            latMax: float - maximum latitude, degrees

        Returns:
            IDvec: list - ID of each site
            lonVec: list - corresponding longitude
            latVec: list - corresponding latitude

        The site file format is as follows:
        Each row contains data for one site, with the following fields separated by spaces:
            ID - site ID
            lon - longitude
            lat - latitude
            elevation_km - elevation in km
            period1 - foundamental period
            Vs30_mpers - shear wave velocity
            Z25_km - depth to the 2.5 km/s shear wave velocity horizon in km (if unknown, input 999)
        """
        self.SiteFile = Path(SiteFile).resolve()

        IDvec, lonVec, latVec = Utilities.CreateMeshedSiteFile(lonMin, lonMax, latMin, latMax, SiteFile = self.SiteFile, MeshDist = MeshDist)
        
        self.IDvec = IDvec
        self.lonVec = lonVec
        self.latVec = latVec

        return IDvec, lonVec, latVec

    def CreateSourceFile_StrikeDip(self, M, N_sim, lon_Hypo, lat_Hypo, W_km, length_km, strike, dip, lambda_degree, Zhyp_km, Fhw=True, ifmedian=False, seed=0, region=3, nPCs=10, EQSourceFile='EQSource.txt'):
        '''
        Create a EQ source file
        Parameters:
            M: float, Moment Magnitude
            SiteFile: str, Site File
            EQSourceFile: str, EQ Source File name that will be generated
            N_sim: int, number of simulations
            lon_Hypo: float, Hypocenter longitude
            lat_Hypo: float, Hypocenter latitude
            W_km: float, Down-Dip Rupture Width in km, if unknown, set to 999
            length_km: float, Rupture length in km, if unknown, set to 999
            strike: float, strike angle in degrees (0-360). The strike direction correspond to the angle between the north and the direction you take so that when you walk along the Fault Trace the fault dips on your right.
            dip: float, dip angle in degrees. The dip is the steepest angle of descent of the fault plane relative to a horizontal plane; it is measured in degrees [0,90]
            lambda_degree: float, rake angle in degrees. 0 for strike-slip, 90 for reverse, -90 for normal. The rake is the direction in which a hanging wall block moves during a rupture, measured relative to fault strike on the plane of the fault
            Zhyp_km: float, Hypocenter depth in km relative to sea level. If unknown, set to 999
            Fhw: bool, if True, consider hanging wall effect
            ifmedian: bool, if True, use median intensity
            seed: int, random seed
            region: int, region number. 0 for global (including Taiwan), 1 for California, 3 for China or Turkey, 4 for Italy
            nPCs: int, number of principal components
        '''
        # calculate the rupture normal vector
        RuptureNormal_x, RuptureNormal_y, RuptureNormal_z = Utilities.CalcRuptureNormal(strike, dip)
        
        self.CreateSourceFile(M, N_sim, lon_Hypo, lat_Hypo, W_km, length_km, RuptureNormal_x, RuptureNormal_y, RuptureNormal_z, lambda_degree, Zhyp_km, Fhw, ifmedian, seed, region, nPCs, EQSourceFile)

    def CreateSourceFile(self, M, N_sim, lon_Hypo, lat_Hypo, W_km, length_km, RuptureNormal_x, RuptureNormal_y, RuptureNormal_z, lambda_degree, Zhyp_km, Fhw=True, ifmedian=False, seed=0, region=3, nPCs=10, EQSourceFile='EQSource.txt'):
        '''
        Create a EQ source file
        Parameters:
            M: float, Moment Magnitude
            SiteFile: str, Site File
            EQSourceFile: str, EQ Source File name that will be generated
            N_sim: int, number of simulations
            lon_Hypo: float, Hypocenter longitude
            lat_Hypo: float, Hypocenter latitude
            W_km: float, Down-Dip Rupture Width in km, if unknown, set to 999
            length_km: float, Rupture length in km, if unknown, set to 999
            RuptureNormal_x: float, Rupture normal x component. East is x, North is y, Up is z
            RuptureNormal_y: float, Rupture normal y component
            RuptureNormal_z: float, Rupture normal z component
            lambda_degree: float, rake angle in degrees. 0 for strike-slip, 90 for reverse, -90 for normal
            Zhyp_km: float, Hypocenter depth in km relative to sea level. If unknown, set to 999
            Fhw: bool, if True, consider hanging wall effect
            ifmedian: bool, if True, use median intensity
            seed: int, random seed
            region: int, region number. 0 for global (including Taiwan), 1 for California, 3 for China or Turkey, 4 for Italy
            nPCs: int, number of principal components
        '''

        # uknown parameters
        if W_km == 999:
            W_km = EQsources.default_W_km(M, lambda_degree)
        if Zhyp_km == 999:
            EQsources.default_Zhyp_km(M, lambda_degree)
        if length_km == 999:
            length_km = EQsources.default_length_km(W_km, lambda_degree)

        if EQSourceFile is not None:
            self.EQSourceFile = Path(EQSourceFile).resolve()

        # Create directory if it does not exist
        if not self.EQSourceFile.parent.exists():
            self.EQSourceFile.parent.mkdir(parents=True)

        # Write a EQ source file as EQSourceFile
        with open(self.EQSourceFile, 'w') as f:
            params = [int(ifmedian), M, N_sim, seed, f"{lon_Hypo} {lat_Hypo}", W_km, length_km,
                f"{RuptureNormal_x} {RuptureNormal_y} {RuptureNormal_z}", lambda_degree,
                int(Fhw), Zhyp_km, region, nPCs]
            f.write('\n'.join(map(str, params)) + '\n')

    def IMFieldSim(self, OutputDir):
        '''
        IM field simulation given a source file and a site file

        Parameters:
            OutputDir: str, Output Directory
        '''
        self.OutputDir = Path(OutputDir).resolve()
        if not self.OutputDir.exists():
            self.OutputDir.mkdir(parents=True)

        oldFolder = os.getcwd()
        EQSourceFile = str(Path(self.EQSourceFile).resolve())
        SiteFile = str(Path(self.SiteFile).resolve())

        try:
            os.chdir(OutputDir)
            subprocess.run([self.IMSimExe, EQSourceFile, SiteFile], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            print(f"Simulations of seismic intensity fields failed: {EQSourceFile} - {SiteFile}")
        else:
            self._ReadIMSim()
            # check if there are NaN values
            if np.isnan(self.IM_mat).any():
                raise ValueError("Error: IM_mat contains NaN values")
        finally:
            os.chdir(oldFolder)

    def _ReadIMSim(self):
        """
        Read IM simulation results

        Returns:
            IM_mat: numpy array - [numel(IDvec), N_sim, i_T]
            Tvec: numpy array - periods corresponding to [i_T]
        """
        
        self.IM_mat, self.Tvec = _Read_IMSim_results(self.OutputDir)

        return self.IM_mat, self.Tvec

    def GetEQSourceInfo(self) -> dict:
        """
        Get EQ source information

        Returns:
            M: float - Moment Magnitude
            lon_Hypo: float - Hypocenter longitude
            lat_Hypo: float - Hypocenter latitude
            W_km: float - Down-Dip Rupture Width in km
            length_km: float - Rupture length in km
            RuptureNormal_x: float - Rupture normal x component
            RuptureNormal_y: float - Rupture normal y component
            RuptureNormal_z: float - Rupture normal z component
            lambda_degree: float - rake angle in degrees
            Zhyp_km: float - Hypocenter depth in km relative to sea level
        """
        return ReadEQSourceInfo(self.EQSourceFile)

    def PrepareCityBldData(self, CityName: str, DistrictName: str = None):
        '''
        Prepare building data for a city or a district

        Parameters:
            DistrictName: str - district name
            CityName: str - city name

        Returns:
            self.BldInfo: list - building information. It is a list of dictionaries with keys 'centroid' and 'T'. 'centroid' is a list of [lon, lat], and 'T' is the period.
        '''
        print(f"Loading building data for IM simulation...")

        # Building information file
        if DistrictName is None:
            BldInfo = Utilities.get_CityBldData(CityName)
        else:
            BldInfo = Utilities.ReadDistrictBldGDF(CityName, DistrictName)
        
        # create centroid column and T column
        BldInfo['centroid'] = BldInfo.apply(lambda row: [row['longitude'], row['latitude']], axis=1)
        PeriodsRepr = ClassifyBld.ReadReprBldStructPara(property='T1 (s)')
        BldInfo['T'] = BldInfo.apply(lambda row: PeriodsRepr[row['ReprBldID']], axis=1)

        # only keep the columns 'centroid' and 'T'
        BldInfo = BldInfo[['centroid', 'T']]

        # convert to list of dictionaries
        self.BldInfo = BldInfo.to_dict(orient='records')

    def InterpIMfromMeshForCity(self, ResultFileName='IM_mat_Bld.txt') -> np.array:
        """
        Interpolate IM simulation results from mesh to building locations and periods

        Returns:
            IM_mat_Bld: numpy array - interpolated IM results at building locations (i_bld, i_Sim)
        """

        IM_mat_Bld = self._InterpIMfromMesh(self.BldInfo)

        # Save to file
        if ResultFileName is not None:
            filepath = self.OutputDir / ResultFileName
            np.savetxt(filepath, IM_mat_Bld, fmt='%.4f')

        return IM_mat_Bld

    def _InterpIMfromMesh(self, BldInfo: list) -> np.array:
        """
        Interpolate IM simulation results from mesh to building locations and periods

        Parameters:
            BldInfo: list - building information. It is a list of dictionaries with keys 'centroid' and 'T'. 'centroid' is a list of [lon, lat], and 'T' is the period.

        Returns:
            IM_mat_Bld: numpy array - interpolated IM results at building locations (i_bld, i_Sim).
        """
        # Convert geographic coordinates to Web Mercator projection
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857")
        xVec, yVec = transformer.transform(self.latVec, self.lonVec)
        
        # Building locations
        centroid = np.array([bld['centroid'] for bld in BldInfo])
        TBldVec = np.array([bld['T'] for bld in BldInfo])
        xBldVec, yBldVec = transformer.transform(centroid[:, 1],centroid[:, 0])

        N_ID = self.IM_mat.shape[0]
        N_sim = self.IM_mat.shape[1]
        N_T = self.IM_mat.shape[2]

        # Expand according to TVec_Mesh
        IM_mat_Mesh = np.moveaxis(self.IM_mat, 2, 0)  # (i_T, i_ID, i_Sim)
        IM_mat_Mesh = IM_mat_Mesh.reshape((N_T * N_ID, N_sim), order='F')  # (i_T * i_ID, i_Sim) - F means the first index changes the fastest
        xVec = np.tile(xVec, (N_T, 1)).reshape(-1, order='F')  # (i_T * i_ID)
        yVec = np.tile(yVec, (N_T, 1)).reshape(-1, order='F')  # (i_T * i_ID)
        TVec_Mesh = np.tile(self.Tvec, N_ID)  # (i_T * i_ID)

        assert len(xVec) == len(yVec) == len(TVec_Mesh) == (N_T * N_ID)

        # check if there are NaN values
        if np.isnan(IM_mat_Mesh).any():
            raise ValueError("Error: IM_mat_Mesh contains NaN values")
        
        # make sure the xBldVec,yBldVec are within the range of xVec,yVec
        if np.any(xBldVec < np.min(xVec)) or np.any(xBldVec > np.max(xVec)) or np.any(yBldVec < np.min(yVec)) or np.any(yBldVec > np.max(yVec)):
            raise ValueError("Error: Building location is out of range of meshed sites")

        # Interpolate
        points = np.column_stack((xVec, yVec, TVec_Mesh))
        # in case TBldVec is out of the range of TVec_Mesh, we use the nearest value
        TBldVec = np.clip(TBldVec, np.min(TVec_Mesh), np.max(TVec_Mesh))
        F = LinearNDInterpolator(points, IM_mat_Mesh)
        IM_mat_Bld = F(xBldVec, yBldVec, TBldVec)  # (i_bld, i_Sim)

        # check if there are NaN values
        if np.isnan(IM_mat_Bld).any():
            raise ValueError("Error: IM_mat_Bld contains NaN values")

        return IM_mat_Bld
    
    def Plot_IM_Spatial_folium(self, iTplot=1, iSim=None, BaseMap='Cartodb Positron'):
        IMresultsDir = self.OutputDir
        lonVec = self.lonVec
        latVec = self.latVec
        Tvec = self.Tvec
        IM_mat = self.IM_mat
        Plot_IM_Spatial(IMresultsDir, lonVec, latVec, Tvec, IM_mat, iTplot=iTplot, iSim=iSim, BaseMap=BaseMap)

    def Plot_IM_Spatial_cartopy(self, iTplot=1, iSim=None):
        """
        Plot IM simulation results using cartopy
        Parameters:
            iTplot: int - index of the period to plot
            iSim: int - index of the simulation to plot
        """


        if iSim is None:
            iSim = np.random.randint(self.IM_mat.shape[1])

        x, y = np.array(self.lonVec), np.array(self.latVec)
        z = self.IM_mat[:, iSim, iTplot - 1]

        # 0.0001 = 10 meters
        X, Y = np.meshgrid(np.linspace(min(x), max(x), int((max(x) - min(x)) / 0.0001)),
                   np.linspace(min(y), max(y), int((max(y) - min(y)) / 0.0001)))  
        Z = griddata((x, y), z, (X, Y), method='cubic')

        fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=ccrs.PlateCarree()))

        contour = ax.contourf(X, Y, Z, levels=20, cmap='viridis', transform=ccrs.PlateCarree())
        fig.colorbar(contour, ax=ax, orientation='vertical', label=f'$S_a(T={self.Tvec[iTplot-1]:.2f}) (\\rm{{g}})$')
        ax.set_extent([min(x), max(x), min(y), max(y)])

        gl = ax.gridlines(draw_labels=True)
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.top_labels = gl.right_labels = False


        plt.show()

def _Read_IMSim_results(IMresultsDir: Path) -> Tuple[np.array, np.array]:
    """
    Read IM simulation results

    Returns:
        IM_mat: numpy array - [numel(IDvec), N_sim, i_T]
        Tvec: numpy array - periods corresponding to [i_T]
    """

    # Get list of files matching the pattern
    listing = list(Path(IMresultsDir).glob("IM sim with period*.txt"))
    Tvec = np.array([float(str(file.stem).split("period")[1]) for file in listing])
    
    # Sort by period
    sorted_indices = np.argsort(Tvec)
    Tvec = Tvec[sorted_indices]
    listing = [listing[i] for i in sorted_indices]

    IM_mat = []

    for file in listing:
        A = np.loadtxt(file)
        N_site = A.shape[0]
        N_sim = A.shape[1] - 1
        IM_mat_1T = np.zeros((N_site, N_sim))
        for i_bld in range(N_site):
            IM_mat_1T[i_bld, :] = A[i_bld, 1:]
        IM_mat.append(IM_mat_1T)

    IM_mat = np.stack(IM_mat, axis=-1)

    return IM_mat, Tvec

def ReadEQSourceInfo(EQSourceFile: Path) -> dict:
    """
    Read EQ source information

    Parameters:
        EQSourceFile: Path - EQ source file

    Returns:
        M: float - Moment Magnitude
        lon_Hypo: float - Hypocenter longitude
        lat_Hypo: float - Hypocenter latitude
        W_km: float - Down-Dip Rupture Width in km
        length_km: float - Rupture length in km
        RuptureNormal_x: float - Rupture normal x component
        RuptureNormal_y: float - Rupture normal y component
        RuptureNormal_z: float - Rupture normal z component
        lambda_degree: float - rake angle in degrees
        Zhyp_km: float - Hypocenter depth in km relative to sea level
    """
    with open(EQSourceFile, 'r') as f:
        lines = f.readlines()
        ifmedian = int(lines[0].strip())
        M = float(lines[1].strip())
        N_sim = int(lines[2].strip())
        seed = int(lines[3].strip())
        lon_Hypo, lat_Hypo = map(float, lines[4].strip().split())
        W_km = float(lines[5].strip())
        length_km = float(lines[6].strip())
        RuptureNormal_x, RuptureNormal_y, RuptureNormal_z = map(float, lines[7].strip().split())
        lambda_degree = float(lines[8].strip())
        Fhw = bool(lines[9].strip())
        Zhyp_km = float(lines[10].strip())
        region = int(lines[11].strip())
        nPCs = int(lines[12].strip())

    return {
        "M": M,
        "lon_Hypo": lon_Hypo,
        "lat_Hypo": lat_Hypo,
        "W_km": W_km,
        "length_km": length_km,
        "RuptureNormal_x": RuptureNormal_x,
        "RuptureNormal_y": RuptureNormal_y,
        "RuptureNormal_z": RuptureNormal_z,
        "lambda_degree": lambda_degree,
        "Zhyp_km": Zhyp_km
    }

def Add_IMfield_to_FoliumMap(m: folium.Map, lonVec: np.array, latVec: np.array, IMvec: np.array, caption: str = 'Spectral Acceleration (Sa) at T=0.01 seconds (g)', GridDist: float=0.01):
    '''
    Add IM field to folium map

    Parameters:
        m: folium.Map - folium map
        lonVec: np.array - longitude of each site
        latVec: np.array - latitude of each site
        IMvec: np.array - IM values
        caption: str - caption of the color bar
        GridDist: float - grid distance in degrees. 0.001 = 100 meters
    '''

    x = lonVec
    y = latVec
    z = IMvec

    # Create a grid of points
    # 0.001 = 100 meters
    X, Y = np.meshgrid(np.linspace(min(x), max(x), int((max(x) - min(x)) / GridDist)),
                np.linspace(min(y), max(y), int((max(y) - min(y)) / GridDist)))  
    Z = griddata((x, y), z, (X, Y), method='cubic')

    # Create a contour plot using matplotlib's tricontourf
    levels = np.arange(0.01, np.max(Z)+0.1, 0.02)

    c_white = matplotlib.colors.colorConverter.to_rgba('white',alpha = 0)
    c_red= matplotlib.colors.colorConverter.to_rgba('red',alpha = 1)
    cmap_rb = matplotlib.colors.LinearSegmentedColormap.from_list('rb_cmap',[c_white,c_red],512)
    contour = plt.contourf(X, Y, Z, levels=levels, cmap=cmap_rb)
    # contour = plt.contourf(X, Y, Z, levels=levels, cmap=plt.cm.viridis)
  
    # Convert the contour plot to geojson
    geojson = geojsoncontour.contourf_to_geojson(contourf=contour, ndigits=3)
    

    # Add the contour plot to the folium map
    folium.GeoJson(
        geojson,
        name='contour',
        smooth_factor=0.0, # it's important to turn off smooth factor, otherwise white lines will appear in the map
        style_function=lambda feature: {
            'fillColor': feature['properties']['fill'],
            'weight': 0,
            'fillOpacity': 0.4,
        }
    ).add_to(m)

    # Add a legend of the contour colors to the map
    colormap = cm.LinearColormap([c_white, c_red], vmin=min(levels), vmax=max(levels))
    # colormap = cm.linear.viridis.scale(min(levels), max(levels))
    colormap.caption = caption
    m.add_child(colormap)

def Add_City_IMfield_to_FoliumMap(m: folium.Map, CityName: str, i_rup: int = 0, iTplot: int = 1, iSim: int = 0):
    '''
    Add IM field to folium map for a city

    Parameters:
        m: folium.Map - folium map
        CityName: str - city name
        i_rup: int - row index of the rupture to plot. Start from 0
        iTplot: int - index of the period to plot. Start from 1
        iSim: int - index of the simulation to plot
    '''
    rup_df = EQsources.ReadCityRuptures(CityName)
    rup_id = rup_df.iloc[i_rup]['rup_id']
    IMresultsDir = IM_SIM_DIR / CityName /f'rup_id_{rup_id}/'

    sitefile = IM_SIM_DIR / CityName / 'SiteFile.txt'
    IDvec, lonVec, latVec = Utilities.ReadIDofMeshedSiteFile(sitefile)

    # read Tvec, IM_mat
    IM_mat, Tvec = _Read_IMSim_results(IMresultsDir)

    Add_IMfield_to_FoliumMap(m, lonVec, latVec, IM_mat[:, iSim, iTplot - 1], caption=f'Spectral Acceleration (Sa) at T={Tvec[iTplot-1]:.2f} seconds (g)')

def Plot_IM_Spatial_City(CityName: str, i_rup: int = 0, iTplot: int = 1, iSim = None, BaseMap='Cartodb Positron', SaveFigDir: Path = None):
    '''
    Plot IM simulation results using folium for a city

    Parameters:
        CityName: str - city name
        i_rup: int - row index of the rupture to plot. Start from 0
        iTplot: int - index of the period to plot. Start from 1
        iSim: int - index of the simulation to plot. None for random
        BaseMap: str - base map to use. None for no basemap, 'Cartodb Positron' for Cartodb Positron, 'Amap' for Amap
        SaveFigDir: Path - path to save the figure
    '''
    rup_df = EQsources.ReadCityRuptures(CityName)
    rup_id = rup_df.iloc[i_rup]['rup_id']
    IMresultsDir = IM_SIM_DIR / CityName /f'rup_id_{rup_id}/'

    sitefile = IM_SIM_DIR / CityName / 'SiteFile.txt'
    IDvec, lonVec, latVec = Utilities.ReadIDofMeshedSiteFile(sitefile)

    # read Tvec, IM_mat
    IM_mat, Tvec = _Read_IMSim_results(IMresultsDir)

    if SaveFigDir is None:
        SaveFigDir = IMresultsDir

    Plot_IM_Spatial(IMresultsDir, lonVec, latVec, Tvec, IM_mat, iTplot=iTplot, iSim=iSim, BaseMap=BaseMap, SaveFigDir=SaveFigDir)

def Plot_IM_Spatial(IMresultsDir, lonVec, latVec, Tvec, IM_mat, iTplot=1, iSim=None, BaseMap='Cartodb Positron', SaveFigDir: Path = None):
    """
    Plot IM simulation results using folium

    Parameters:
        IMresultsDir: Path - directory containing IM simulation results
        lonVec: list - longitude of each site
        latVec: list - latitude of each site
        Tvec: numpy array - periods corresponding to [i_T]
        IM_mat: numpy array - [numel(ID_site), N_sim, i_T]
        iTplot: int - index of the period to plot
        iSim: int - index of the simulation to plot
        BaseMap: str - base map to use. None for no basemap, 'Cartodb Positron' for Cartodb Positron, 'Amap' for Amap
        SaveFigDir: Path - path to save the figure. None means the same directory as IMresultsDir
    """
    if iSim is None:
        iSim = np.random.randint(IM_mat.shape[1])

    # Extract data
    x, y = np.array(lonVec), np.array(latVec)
    z = IM_mat[:, iSim, iTplot - 1]

    # Create a folium map centered around the mean location
    Center = [np.mean(x), np.mean(y)]
    if BaseMap is None:
        m = folium.Map(tiles=None, location=[Center[1], Center[0]], zoom_start=16)
    elif BaseMap == 'Cartodb Positron':
        m = folium.Map(location=[Center[1], Center[0]], zoom_start=16, tiles='Cartodb Positron')
    elif BaseMap == 'Amap':
        m = folium.Map(location=[Center[1], Center[0]], zoom_start=16, tiles= 'https://wprd01.is.autonavi.com/appmaptile?x={x}&y={y}&z={z}&lang=zh_cn&size=1&scl=1&style=7',attr='高德', max_zoom=19)

    # Add IM field to the map
    caption = f'Spectral Acceleration (Sa) at T={Tvec[iTplot-1]:.2f} seconds (g)'
    Add_IMfield_to_FoliumMap(m, x, y, z, caption=caption)

    # Add a rectangle to represent the rupture
    EQsourceInfo = ReadEQSourceInfo(IMresultsDir/'EQSource.txt')
    mag, rake, lon_Hypo, lat_Hypo, W_km, length_km, centroid_depth = EQsourceInfo['M'], EQsourceInfo['lambda_degree'], EQsourceInfo['lon_Hypo'], EQsourceInfo['lat_Hypo'], EQsourceInfo['W_km'], EQsourceInfo['length_km'], EQsourceInfo['Zhyp_km']
    RuptureNormal_x, RuptureNormal_y, RuptureNormal_z = EQsourceInfo['RuptureNormal_x'], EQsourceInfo['RuptureNormal_y'], EQsourceInfo['RuptureNormal_z']
    strike, dip = Utilities.RuptureNormal2StrikeDip(RuptureNormal_x, RuptureNormal_y, RuptureNormal_z)
    # columns: mag, rake, centroid_lat, centroid_lon, strike, dip, centroid_depth, W_km (optional), length_km (optional)
    rupture = pd.Series([mag, rake, lat_Hypo, lon_Hypo, strike, dip, centroid_depth, W_km, length_km], index=['mag', 'rake', 'centroid_lat', 'centroid_lon', 'strike', 'dip', 'centroid_depth', 'W_km', 'length_km'])
    EQsources.Create_a_Rupture_Polygon(m, rupture)

    # Save the map to an HTML file
    if SaveFigDir is None:
        SaveFigDir = IMresultsDir
    filename = SaveFigDir / f'IM_Spatial_folium_Sim{iSim}_Sa{Tvec[iTplot-1]:.2f}.html'
    m.save(filename)

    webbrowser.open_new('file://' + str(filename.resolve()))

def CityBldIMSimFromEvents(CityName: str, OutputDir=IM_SIM_DIR):
    '''
    IM field simulation for buildings in a city. A series of earthquake events are generated and simulated.
    '''
    sim = SpatialIM()
    sim.PrepareCityBldData(CityName)

    # meshed site file
    print(f"Creating meshed site file for {CityName}...")
    SiteFile = OutputDir / CityName / 'SiteFile.txt'
    Bld_gdf = Utilities.get_CityBldData(CityName)
    minx, miny, maxx, maxy = Bld_gdf.total_bounds
    buffer = 0.1  # Increase the bounds by a small buffer
    Utilities.CreateMeshedSiteFile(minx - buffer, maxx + buffer, miny - buffer, maxy + buffer, SiteFile=SiteFile, MeshDist=5000)

    # Read earthquake ruptures
    rup_df = EQsources.ReadCityRuptures(CityName)

    # Loop over each rupture
    for i, row in tqdm(rup_df.iterrows(), total=rup_df.shape[0], desc='Simulating IM fields...'):
        rup_id = row['rup_id']

        # Create a EQ source file
        EQSourceFileName = OutputDir / f'{CityName}/rup_id_{rup_id}/EQSource.txt'
        sim.CreateSourceFile_StrikeDip(M=row['mag'], N_sim=N_SIM_EACH_RUP, lon_Hypo=row['centroid_lon'], lat_Hypo=row['centroid_lat'], W_km=999, length_km=999, strike=row['strike'], dip=row['dip'], lambda_degree=row['rake'], Zhyp_km=row['centroid_depth'], Fhw=True, ifmedian=False, seed=0, region=3, nPCs=10, EQSourceFile=EQSourceFileName)

        sim.SetMeshedSiteFile(SiteFile)

        sim.IMFieldSim(OutputDir / f'{CityName}/rup_id_{rup_id}/')

        # interpolate
        sim.InterpIMfromMeshForCity() # numpy array (i_bld, i_Sim)

    _MergeIMMatBldFiles(CityName, OutputDir=OutputDir)

def _MergeIMMatBldFiles(CityName: str, OutputDir=IM_SIM_DIR, IM_mat_Bld_FileName=IM_MAT_BLD_FILENAME, rupid_2_iSim_FileName=RUPID_2_ISIM_FILENAME):
    """
    Merge all IM_mat_Bld files into a h5py file, and return a dask array
    """

    # Read earthquake ruptures
    rup_df = EQsources.ReadCityRuptures(CityName)

    rupid_2_iSim = pd.DataFrame(index=rup_df['rup_id'], columns=['iSim'])

    num_bld = Utilities.get_CityBldData(CityName).shape[0]
    total_sims = len(rup_df) * N_SIM_EACH_RUP

    # Create h5py file
    h5py_file = OutputDir / CityName / IM_mat_Bld_FileName

    with h5py.File(h5py_file, 'w') as hf:
        # Create dataset in h5py file
        dset = hf.create_dataset('IM_mat_Bld', shape=(num_bld, total_sims), dtype='float32')

        j_start = 0
        for i, row in tqdm(rup_df.iterrows(), total=rup_df.shape[0], desc='Merging IM files...'):
            rup_id = row['rup_id']
            file_name = OutputDir / CityName / f'rup_id_{rup_id}/IM_mat_Bld.txt'
            chunk = np.loadtxt(file_name)

            dset[:, j_start:j_start + chunk.shape[1]] = chunk
            rupid_2_iSim.loc[rup_id, 'iSim'] = np.arange(j_start, j_start + chunk.shape[1])
            j_start += chunk.shape[1]

    # Save rupid_2_iSim
    rupid_2_iSim.to_csv(OutputDir / CityName / rupid_2_iSim_FileName, index=True, header=True)

def read_IM_mat_Bld_iteratively(CityName: str, OutputDir=IM_SIM_DIR) -> Generator[Tuple[str, np.array], None, None]:
    """
    Generator function to read IM_mat_Bld files iteratively, one file at a time.

    Parameters:
        CityName: str - name of the city
        OutputDir: Path - output directory

    Yields:
        rup_id: str - rupture ID
        chunk: numpy array - IM_mat_Bld data from one file (i_bld, i_Sim)
    """
    # Read earthquake ruptures
    rup_df = EQsources.ReadCityRuptures(CityName)

    for i, row in rup_df.iterrows():
        rup_id = row['rup_id']

        file_name = OutputDir / CityName / f'rup_id_{rup_id}/IM_mat_Bld.txt'
        if not file_name.exists():
            raise FileNotFoundError(f"File {file_name} does not exist")

        chunk = np.loadtxt(file_name)
        yield rup_id, chunk

def get_all_IM_mat_Bld(CityName: str, IMsimDir=IM_SIM_DIR, IM_mat_Bld_FileName=IM_MAT_BLD_FILENAME, rupid_2_iSim_FileName=RUPID_2_ISIM_FILENAME):
    """
    return a dask array

    Parameters:
        CityName: str - name of the city
        IMsimDir: Path - IMsimDir directory
    
    Returns:
        rupid_2_iSim: dict - mapping from rupture ID to simulation indices. row index is rup_id, and the column is iSim
        all_IM_mat_Bld: dask array - IM_mat_Bld data (i_bld, i_Sim)
    """

    # Read rupid_2_iSim
    rupid_2_iSim = pd.read_csv(IMsimDir / CityName / rupid_2_iSim_FileName, index_col=0, header=0, dtype={0: str})
    rupid_2_iSim['iSim'] = rupid_2_iSim['iSim'].apply(lambda x: list(map(int, x.strip('[]').split())) if isinstance(x, str) else x)
    rupid_2_iSim = rupid_2_iSim.to_dict(orient='index')

    rup_df = EQsources.ReadCityRuptures(CityName)
    total_sims = len(rup_df) * N_SIM_EACH_RUP

    # Create a dask array from the h5py file
    h5py_file = IMsimDir / CityName / IM_mat_Bld_FileName
    storage = h5py.File(h5py_file, 'r')['IM_mat_Bld']
    all_IM_mat_Bld = da.from_array(storage, chunks=(1000, total_sims))

    return rupid_2_iSim, all_IM_mat_Bld

    


