from . import Utilities
from . import ClassifyBld
from . import EQsources
from . import Logger

import pandas as pd
import os
import math
import subprocess
from pathlib import Path
from pyproj import Transformer
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import RegularGridInterpolator
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

# Initialize logger for this module
logger = Logger.get_logger('IMSim')

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
        logger.info("SpatialIM instance initialized")    
        
    def CreateAExampleSourceFile(self, EQSourceFile='EQSource.txt'):
        '''
        Create a example EQ source file, located at Wuhan, China
        '''
        logger.info("Creating example earthquake source file for Wuhan, China")

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
        logger.info(f"Example earthquake source file created: {EQSourceFile}")

    def SetMeshedSiteFile(self, SiteFile):
        '''
        Specify a meshed site file
        '''
        logger.info(f"Setting meshed site file: {SiteFile}")
        
        if not Path(SiteFile).exists():
            logger.error(f"Site file not found: {SiteFile}")
            raise FileNotFoundError(f"Site file {SiteFile} does not exist")
        
        self.SiteFile = Path(SiteFile).resolve()
        
        IDvec, lonVec, latVec = Utilities.ReadIDofMeshedSiteFile(SiteFile)

        self.IDvec = IDvec
        self.lonVec = lonVec
        self.latVec = latVec

        logger.info(f"Meshed site file loaded successfully with {len(IDvec)} sites")
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
        logger.info(f"Creating meshed site file: {SiteFile}")
        logger.info(f"Mesh extent: lon({lonMin}, {lonMax}), lat({latMin}, {latMax}), distance: {MeshDist}m")
        
        self.SiteFile = Path(SiteFile).resolve()

        IDvec, lonVec, latVec = Utilities.CreateMeshedSiteFile(lonMin, lonMax, latMin, latMax, SiteFile = self.SiteFile, MeshDist = MeshDist)
        
        self.IDvec = IDvec
        self.lonVec = lonVec
        self.latVec = latVec

        logger.info(f"Meshed site file created successfully with {len(IDvec)} sites")
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
        logger.info(f"Creating earthquake source file with M={M}, strike={strike}°, dip={dip}°")
        
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
        logger.info(f"Creating earthquake source file: M={M}, {N_sim} simulations")
        logger.debug(f"Hypocenter location: ({lon_Hypo}, {lat_Hypo}), depth: {Zhyp_km}km")

        # uknown parameters
        if W_km == 999:
            W_km = EQsources.default_W_km(M, lambda_degree)
            logger.debug(f"Using default rupture width: {W_km}km")
        if Zhyp_km == 999:
            EQsources.default_Zhyp_km(M, lambda_degree)
            logger.debug(f"Using default hypocenter depth: {Zhyp_km}km")
        if length_km == 999:
            length_km = EQsources.default_length_km(W_km, lambda_degree)
            logger.debug(f"Using default rupture length: {length_km}km")

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
        
        logger.info(f"Earthquake source file created: {self.EQSourceFile}")

    def IMFieldSim(self, OutputDir, overwrite: bool = False):
        '''
        IM field simulation given a source file and a site file

        Parameters:
            OutputDir: str, Output Directory
            overwrite: bool, if True, overwrite the existing output directory
        '''
        logger.info(f"Starting IM field simulation")
        logger.info(f"Output directory: {OutputDir}")
        
        self.OutputDir = Path(OutputDir).resolve()
        if not self.OutputDir.exists():
            self.OutputDir.mkdir(parents=True)

        if (self.OutputDir / 'IM sim.txt').exists() and not overwrite:
            logger.warning(f"Output directory already exists: {self.OutputDir}. IM field simualtions are skipped. Use 'overwrite=True' to overwrite.")
            self._ReadIMSim()
            return

        oldFolder = os.getcwd()
        EQSourceFile = str(Path(self.EQSourceFile).resolve())
        SiteFile = str(Path(self.SiteFile).resolve())

        logger.debug(f"Source file: {EQSourceFile}")
        logger.debug(f"Site file: {SiteFile}")

        try:
            os.chdir(OutputDir)
            logger.info("Running IM simulation executable...")
            subprocess.run([self.IMSimExe, EQSourceFile, SiteFile], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            logger.error(f"IM field simulation failed: {EQSourceFile} - {SiteFile}")
            logger.error(f"Error details: {e.stderr.decode() if e.stderr else 'No error details'}")
            print(f"Simulations of seismic intensity fields failed: {EQSourceFile} - {SiteFile}")
        else:
            logger.info("IM simulation completed successfully")
            self._ReadIMSim()
            logger.info(f"IM simulation results loaded: {self.IM_mat.shape}")
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
        
        # check if there are NaN values
        if np.isnan(self.IM_mat).any():
            logger.error("IM simulation results contain NaN values")
            raise ValueError("Error: IM_mat contains NaN values")

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
        if DistrictName is None:
            logger.info(f"Loading building data for city: {CityName}")
        else:
            logger.info(f"Loading building data for district: {DistrictName}, {CityName}")
        
        print(f"Loading building data for IM simulation...")

        # Building information file
        if DistrictName is None:
            BldInfo = Utilities.get_CityBldData(CityName)
        else:
            BldInfo = Utilities.ReadDistrictBldGDF(CityName, DistrictName)
        
        logger.info(f"Loaded {len(BldInfo)} buildings")
        
        # create centroid column and T column
        BldInfo['centroid'] = BldInfo.apply(lambda row: [row['longitude'], row['latitude']], axis=1)
        PeriodsRepr = ClassifyBld.ReadReprBldStructPara(property='T1 (s)')
        BldInfo['T'] = BldInfo.apply(lambda row: PeriodsRepr[row['ReprBldID']], axis=1)

        # only keep the columns 'centroid' and 'T'
        BldInfo = BldInfo[['centroid', 'T']]

        # convert to list of dictionaries
        self.BldInfo = BldInfo.to_dict(orient='records')
        
        logger.info(f"Building data prepared successfully: {len(self.BldInfo)} buildings")

    def InterpIMfromMeshForCity(self, ResultFileName='IM_mat_Bld.txt', batch_size: int = 500000, use_regular_grid: bool = True, overwrite: bool = False) -> np.array:
        """
        Interpolate IM simulation results from mesh to building locations and periods

        Parameters:
            ResultFileName: str - output file name for results
            batch_size: int - number of buildings to process in each batch to avoid memory explosion
            use_regular_grid: bool - whether to use regular grid interpolation (faster) or LinearNDInterpolator
            overwrite: bool - if True, overwrite the existing result file

        Returns:
            IM_mat_Bld: numpy array - interpolated IM results at building locations (i_bld, i_Sim)
        """
        logger.info("Starting interpolation of IM results from mesh to building locations")

        # file path for results
        if ResultFileName is not None:
            filepath = self.OutputDir / ResultFileName

        if filepath.exists() and not overwrite:
            logger.warning(f"IM_mat_Bld result file already exists: {filepath}. Use 'overwrite=True' to overwrite.")
            return np.loadtxt(filepath, dtype=np.float32)

        if use_regular_grid:
            logger.info("Using regular grid interpolation method")
            IM_mat_Bld = self._InterpIMfromMesh_RegularGrid(self.BldInfo)
        else:
            logger.info("Using LinearNDInterpolator method with batching")
            IM_mat_Bld = self._InterpIMfromMesh(self.BldInfo, batch_size=batch_size)

        # Save to file
        np.savetxt(filepath, IM_mat_Bld, fmt='%.4f')
        logger.info(f"Interpolated IM results saved to: {filepath}")

        logger.info(f"Interpolation completed for {IM_mat_Bld.shape[0]} buildings")
        return IM_mat_Bld
    
    def _InterpIMfromMesh(self, BldInfo: list, batch_size: int = 500000) -> np.array:
        """
        Interpolate IM simulation results from mesh to building locations and periods

        Parameters:
            BldInfo: list - building information. It is a list of dictionaries with keys 'centroid' and 'T'. 'centroid' is a list of [lon, lat], and 'T' is the period.
            batch_size: int - number of buildings to process in each batch to avoid memory explosion

        Returns:
            IM_mat_Bld: numpy array - interpolated IM results at building locations (i_bld, i_Sim).
        """
        logger.info(f"Starting interpolation for {len(BldInfo)} buildings in batches of {batch_size}")
        
        # Convert geographic coordinates to Web Mercator projection (do this once for mesh)
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857")
        xVec, yVec = transformer.transform(self.latVec, self.lonVec)

        N_ID = self.IM_mat.shape[0]
        N_sim = self.IM_mat.shape[1]
        N_T = self.IM_mat.shape[2]

        # Prepare mesh data once (avoid repeated operations)
        # Expand according to TVec_Mesh
        IM_mat_Mesh = np.moveaxis(self.IM_mat, 2, 0)  # (i_T, i_ID, i_Sim)
        IM_mat_Mesh = IM_mat_Mesh.reshape((N_T * N_ID, N_sim), order='F')  # (i_T * i_ID, i_Sim)
        xVec_expanded = np.tile(xVec, (N_T, 1)).reshape(-1, order='F')  # (i_T * i_ID)
        yVec_expanded = np.tile(yVec, (N_T, 1)).reshape(-1, order='F')  # (i_T * i_ID)
        TVec_Mesh = np.tile(self.Tvec, N_ID)  # (i_T * i_ID)

        assert len(xVec_expanded) == len(yVec_expanded) == len(TVec_Mesh) == (N_T * N_ID)

        # check if there are NaN values
        if np.isnan(IM_mat_Mesh).any():
            raise ValueError("Error: IM_mat_Mesh contains NaN values")

        # Create interpolator once (avoid repeated operations)
        points = np.column_stack((xVec_expanded, yVec_expanded, TVec_Mesh))
        F = LinearNDInterpolator(points, IM_mat_Mesh)
        logger.info("Linear interpolator created successfully")        # Process buildings in batches
        total_buildings = len(BldInfo)
        num_batches = (total_buildings + batch_size - 1) // batch_size
        logger.info(f"Processing {total_buildings} buildings in {num_batches} batches")

        # Initialize result array
        IM_mat_Bld_list = []

        # Use tqdm for progress bar
        with tqdm(total=num_batches, desc="IM Interpolation", unit="batch") as pbar:
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, total_buildings)
                batch_BldInfo = BldInfo[start_idx:end_idx]
                
                # Update progress bar description with current batch info
                pbar.set_description(f"IM Interpolation (Batch {batch_idx + 1}/{num_batches}, Buildings {start_idx}-{end_idx-1})")
                
                logger.debug(f"Processing batch {batch_idx + 1}/{num_batches} (buildings {start_idx}-{end_idx-1})")

                # Building locations for this batch
                centroid_batch = np.array([bld['centroid'] for bld in batch_BldInfo])
                TBldVec_batch = np.array([bld['T'] for bld in batch_BldInfo])
                xBldVec_batch, yBldVec_batch = transformer.transform(centroid_batch[:, 1], centroid_batch[:, 0])

                # make sure the building locations are within the range of mesh sites
                if np.any(xBldVec_batch < np.min(xVec)) or np.any(xBldVec_batch > np.max(xVec)) or \
                   np.any(yBldVec_batch < np.min(yVec)) or np.any(yBldVec_batch > np.max(yVec)):
                    logger.error("Error: Building location is out of range of meshed sites")
                    raise ValueError("Error: Building location is out of range of meshed sites")

                # in case TBldVec is out of the range of TVec_Mesh, we use the nearest value
                TBldVec_batch = np.clip(TBldVec_batch, np.min(TVec_Mesh), np.max(TVec_Mesh))

                # Interpolate for this batch
                IM_mat_Bld_batch = F(xBldVec_batch, yBldVec_batch, TBldVec_batch)  # (i_bld_batch, i_Sim)

                # check if there are NaN values
                if np.isnan(IM_mat_Bld_batch).any():
                    # find the indices of NaN values
                    nan_indices = np.argwhere(np.isnan(IM_mat_Bld_batch))
                    for idx in nan_indices:
                        bld_idx = idx[0]
                        sim_idx = idx[1]
                        logger.error(f"Error: IM_mat_Bld contains NaN value for building {start_idx + bld_idx + 1} in simulation {sim_idx + 1}")
                    raise ValueError(f"Error: IM_mat_Bld contains NaN values in batch {batch_idx + 1}")

                IM_mat_Bld_list.append(IM_mat_Bld_batch)
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({
                    'Buildings_Processed': f"{end_idx:,}/{total_buildings:,}",
                    'Batch_Size': len(batch_BldInfo),
                    'Memory_Usage': f"{len(IM_mat_Bld_list) * batch_size * 8 / 1024**2:.1f}MB"  # Rough estimate
                })
                
                logger.debug(f"Batch {batch_idx + 1}/{num_batches} completed, processed {end_idx}/{total_buildings} buildings")

        # Concatenate all batches
        IM_mat_Bld = np.vstack(IM_mat_Bld_list)
        logger.info(f"All batches processed successfully. Final shape: {IM_mat_Bld.shape}")

        return IM_mat_Bld
    
    def _vectorized_period_interpolation(self, points_all: np.array, TBldVec_clipped: np.array, interpolators: dict, N_sim: int) -> np.array:
        """
        使用矢量化操作执行周期插值，显著提升性能
        
        Parameters:
            points_all: numpy array - 所有建筑的坐标 (N_buildings, 2) [y, x]
            TBldVec_clipped: numpy array - 限制后的建筑周期向量 (N_buildings,)
            interpolators: dict - 周期对应的插值器字典 {period: interpolator}
            N_sim: int - 模拟次数
            
        Returns:
            IM_mat_Bld: numpy array - 插值结果 (N_buildings, N_sim)
        """
        logger.debug("Starting vectorized period interpolation")
        N_buildings = len(TBldVec_clipped)
        IM_mat_Bld = np.zeros((N_buildings, N_sim))
        
        # 处理边界情况：小于最小周期的建筑
        mask_low = TBldVec_clipped <= self.Tvec[0]
        if np.any(mask_low):
            logger.debug(f"Processing {np.sum(mask_low)} buildings with T <= {self.Tvec[0]:.3f}")
            interpolator = interpolators[self.Tvec[0]]
            result_low = interpolator(points_all[mask_low])  # (N_low, N_sim)
            IM_mat_Bld[mask_low, :] = result_low
        
        # 处理边界情况：大于最大周期的建筑
        mask_high = TBldVec_clipped >= self.Tvec[-1]
        if np.any(mask_high):
            logger.debug(f"Processing {np.sum(mask_high)} buildings with T >= {self.Tvec[-1]:.3f}")
            interpolator = interpolators[self.Tvec[-1]]
            result_high = interpolator(points_all[mask_high])  # (N_high, N_sim)
            IM_mat_Bld[mask_high, :] = result_high
        
        # 处理中间范围：需要周期线性插值的建筑
        mask_middle = (TBldVec_clipped > self.Tvec[0]) & (TBldVec_clipped < self.Tvec[-1])
        if np.any(mask_middle):
            logger.debug(f"Processing {np.sum(mask_middle)} buildings with period interpolation")
            
            T_middle = TBldVec_clipped[mask_middle]
            points_middle = points_all[mask_middle]
            
            # 找到包围周期的索引
            T_indices = np.searchsorted(self.Tvec, T_middle)
            T_idx_lower = np.clip(T_indices - 1, 0, len(self.Tvec) - 1)
            T_idx_upper = np.clip(T_indices, 0, len(self.Tvec) - 1)
            
            # 处理恰好等于某个周期的情况
            mask_exact = T_idx_lower == T_idx_upper
            if np.any(mask_exact):
                exact_indices = np.where(mask_middle)[0][mask_exact]
                exact_T_idx = T_idx_lower[mask_exact]
                for i, t_idx in enumerate(exact_T_idx):
                    interpolator = interpolators[self.Tvec[t_idx]]
                    result = interpolator(points_middle[mask_exact][i:i+1])
                    IM_mat_Bld[exact_indices[i], :] = result[0]
            
            # 处理需要线性插值的情况
            mask_interp = ~mask_exact
            if np.any(mask_interp):
                interp_indices = np.where(mask_middle)[0][mask_interp]
                T_interp = T_middle[mask_interp]
                points_interp = points_middle[mask_interp]
                T_idx_lower_interp = T_idx_lower[mask_interp]
                T_idx_upper_interp = T_idx_upper[mask_interp]
                
                # 计算权重
                T_lower_vals = self.Tvec[T_idx_lower_interp]
                T_upper_vals = self.Tvec[T_idx_upper_interp]
                weight_upper = (T_interp - T_lower_vals) / (T_upper_vals - T_lower_vals)
                weight_lower = 1.0 - weight_upper
                
                # 按周期组合处理，减少插值器调用次数
                unique_pairs = np.unique(np.column_stack((T_idx_lower_interp, T_idx_upper_interp)), axis=0)
                
                for t_low_idx, t_high_idx in unique_pairs:
                    # 找到使用这对周期的建筑
                    pair_mask = (T_idx_lower_interp == t_low_idx) & (T_idx_upper_interp == t_high_idx)
                    if not np.any(pair_mask):
                        continue
                        
                    pair_indices = interp_indices[pair_mask]
                    pair_points = points_interp[pair_mask]
                    pair_weight_lower = weight_lower[pair_mask]
                    pair_weight_upper = weight_upper[pair_mask]
                    
                    # 在两个周期上插值
                    interpolator_lower = interpolators[self.Tvec[t_low_idx]]
                    interpolator_upper = interpolators[self.Tvec[t_high_idx]]
                    
                    result_lower = interpolator_lower(pair_points)  # (N_pair, N_sim)
                    result_upper = interpolator_upper(pair_points)  # (N_pair, N_sim)
                    
                    # 周期维度的线性插值
                    result_combined = (pair_weight_lower[:, np.newaxis] * result_lower + 
                                     pair_weight_upper[:, np.newaxis] * result_upper)
                    
                    IM_mat_Bld[pair_indices, :] = result_combined
        
        logger.debug("Vectorized period interpolation completed")
        return IM_mat_Bld

    def _InterpIMfromMesh_RegularGrid(self, BldInfo: list) -> np.array:
        """
        使用规则网格插值器进行IM插值，适用于规则网格的站点文件
        这个方法比LinearNDInterpolator快得多，因为利用了网格的规则结构

        Parameters:
            BldInfo: list - 建筑信息列表，每个元素是包含'centroid'和'T'键的字典
                   'centroid'是[lon, lat]列表，'T'是周期

        Returns:
            IM_mat_Bld: numpy array - 插值后的IM结果 (i_bld, i_Sim)
        """
        
        logger.info(f"Starting regular grid interpolation for {len(BldInfo)} buildings")
        
        # 转换地理坐标到Web Mercator投影
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857")
        xVec, yVec = transformer.transform(self.latVec, self.lonVec)
        
        N_ID = self.IM_mat.shape[0]
        N_sim = self.IM_mat.shape[1]
        N_T = self.IM_mat.shape[2]
        
        # 推断网格结构
        # 从CreateMeshedSiteFile的逻辑可知，网格是按 row(lon), col(lat) 的顺序创建的
        # 即外层循环是longitude (Nlon), 内层循环是latitude (Nlat)
        unique_x = np.unique(xVec)
        unique_y = np.unique(yVec)
        
        # 按照CreateMeshedSiteFile的逻辑排序
        unique_x = np.sort(unique_x)
        unique_y = np.sort(unique_y)
        
        Nlon = len(unique_x)  # 经度方向的网格数
        Nlat = len(unique_y)  # 纬度方向的网格数
        
        logger.info(f"Grid structure: {Nlon} x {Nlat} = {Nlon * Nlat} points")
        logger.info(f"Expected total points: {N_ID}")
        
        if Nlon * Nlat != N_ID:
            logger.error(f"Grid structure mismatch: {Nlon} x {Nlat} = {Nlon * Nlat} ≠ {N_ID}")
            raise ValueError("Grid is not regular or site ordering is incorrect")
        
        # 对于每个周期创建2D规则网格插值器
        interpolators = {}
        
        for i_T, T in enumerate(self.Tvec):
            logger.debug(f"Creating interpolator for period T={T:.3f}s")
            
            # 重塑IM数据为网格形状
            # CreateMeshedSiteFile的循环顺序：for row in range(Nlon): for col in range(Nlat)
            # 因此数据按照 (Nlon, Nlat) 的顺序存储，但需要转置为 (Nlat, Nlon) 给RegularGridInterpolator
            grid_values = np.zeros((Nlat, Nlon, N_sim))
            
            # 重新排列数据到正确的网格位置
            for idx in range(N_ID):
                # 根据CreateMeshedSiteFile的逻辑，第idx个点对应的网格位置是：
                row = idx // Nlat  # longitude index
                col = idx % Nlat   # latitude index
                
                # 在grid_values中，第一个维度是latitude (y)，第二个维度是longitude (x)
                grid_values[col, row, :] = self.IM_mat[idx, :, i_T]
            
            # 创建规则网格插值器
            interpolators[T] = RegularGridInterpolator(
                (unique_y, unique_x),  # 注意顺序：(y, x)
                grid_values,
                method='linear',
                bounds_error=False,
                fill_value=None  # 边界外使用最近邻
            )
        
        logger.info("All interpolators created successfully")
        
        # 准备建筑数据
        centroid_all = np.array([bld['centroid'] for bld in BldInfo])
        TBldVec_all = np.array([bld['T'] for bld in BldInfo])
        xBldVec_all, yBldVec_all = transformer.transform(centroid_all[:, 1], centroid_all[:, 0])
        
        # 检查建筑位置是否在网格范围内
        x_min, x_max = np.min(unique_x), np.max(unique_x)
        y_min, y_max = np.min(unique_y), np.max(unique_y)
        
        if (np.any(xBldVec_all < x_min) or np.any(xBldVec_all > x_max) or 
            np.any(yBldVec_all < y_min) or np.any(yBldVec_all > y_max)):
            logger.error("Some buildings are outside the grid range.")
            raise ValueError("Error: Some buildings are outside the grid range of meshed sites")
        
        # 限制建筑周期在插值范围内
        T_min, T_max = np.min(self.Tvec), np.max(self.Tvec)
        TBldVec_clipped = np.clip(TBldVec_all, T_min, T_max)
        
        # 执行插值 - 使用矢量化操作
        logger.info("Performing interpolation with vectorized operations...")
        points_all = np.column_stack((yBldVec_all, xBldVec_all))  # (N_buildings, 2)
        IM_mat_Bld = self._vectorized_period_interpolation(points_all, TBldVec_clipped, interpolators, N_sim)
        
        # 检查NaN值
        if np.isnan(IM_mat_Bld).any():
            nan_count = np.sum(np.isnan(IM_mat_Bld))
            logger.error(f"Found {nan_count} NaN values in interpolation results")
            raise ValueError("Error: IM_mat_Bld contains NaN values after interpolation")
        
        logger.info(f"Regular grid interpolation completed successfully. Final shape: {IM_mat_Bld.shape}")
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
    # if cation T=0.01, set caption to PGA
    if 'T=' in caption:
        period = float(caption.split('T=')[1].split('seconds')[0].strip())
        if period <= 0.01:
            caption = 'PGA (g)'
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

def CityBldIMSimFromEvents(CityName: str, OutputDir=IM_SIM_DIR, batch_size: int = 500000):
    '''
    IM field simulation for buildings in a city. A series of earthquake events are generated and simulated.
    
    Parameters:
        CityName: str - name of the city
        OutputDir: Path - output directory for results
        batch_size: int - number of buildings to process in each batch to avoid memory explosion
    '''
    logger.info(f"Starting IM field simulation for city: {CityName} with batch size: {batch_size}")
    
    sim = SpatialIM()
    sim.PrepareCityBldData(CityName)

    # meshed site file
    logger.info(f"Creating meshed site file for {CityName}")
    
    SiteFile = OutputDir / CityName / 'SiteFile.txt'
    Bld_gdf = Utilities.get_CityBldData(CityName)
    minx, miny, maxx, maxy = Bld_gdf.total_bounds
    buffer = 0.1  # Increase the bounds by a small buffer
    Utilities.CreateMeshedSiteFile(minx - buffer, maxx + buffer, miny - buffer, maxy + buffer, SiteFile=SiteFile, MeshDist=5000)

    # Read earthquake ruptures
    rup_df = EQsources.ReadCityRuptures(CityName)
    logger.info(f"Processing {len(rup_df)} earthquake ruptures for {CityName}")

    # Loop over each rupture
    for i, row in tqdm(rup_df.iterrows(), total=rup_df.shape[0], desc='Simulating IM fields...'):
        rup_id = row['rup_id']
        logger.debug(f"Processing rupture {i+1}/{len(rup_df)}: {rup_id} (M={row['mag']})")

        # Create a EQ source file
        EQSourceFileName = OutputDir / f'{CityName}/rup_id_{rup_id}/EQSource.txt'
        sim.CreateSourceFile_StrikeDip(M=row['mag'], N_sim=N_SIM_EACH_RUP, lon_Hypo=row['centroid_lon'], lat_Hypo=row['centroid_lat'], W_km=999, length_km=999, strike=row['strike'], dip=row['dip'], lambda_degree=row['rake'], Zhyp_km=row['centroid_depth'], Fhw=True, ifmedian=False, seed=0, region=3, nPCs=10, EQSourceFile=EQSourceFileName)

        sim.SetMeshedSiteFile(SiteFile)

        sim.IMFieldSim(OutputDir / f'{CityName}/rup_id_{rup_id}/')        
        # interpolate
        sim.InterpIMfromMeshForCity(batch_size=batch_size, use_regular_grid=True) # numpy array (i_bld, i_Sim)

    logger.info(f"IM field simulation completed for {CityName}")
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




