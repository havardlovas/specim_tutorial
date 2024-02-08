# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import os
import sys
import glob
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import spectral as sp
from spectral import envi
import configparser
import pandas as pd

# To access the custom modules of the hyperspectral_toolchain, give the path to the src folder "...hyperspectral_toolchain/src/"
module_path = 'C:/Users/haavasl/VSCodeProjects/hyperspectral_toolchain/src/'
if module_path not in sys.path:
    sys.path.append(module_path)

from lib.specim_parsing_utils import Specim
from scripts.modulate_config import prepend_data_dir_to_relative_paths

MISSION_NAME_PREFIX = '2022-08-31-060000-Remoy-Specim' # I use UTC time to avoid any timezone Bull S**t

DATE = '2022-08-31'
MISSION_DIR = 'D:/Specim/Missions/2022-08-31-Remøy/2022-08-31_0800_HSI/'
CAL_DIR = 'D:/Specim/Lab_Calibrations/'
OUT_DIR = 'D:/HyperspectralDataAll/HI/' + MISSION_NAME_PREFIX + '/'

ACTIVE_SENSOR_SPATIAL_PIXELS = 1024 # Constant for AFX10
ACTIVE_SENSOR_SPECTRAL_PIXELS = 448 # Constant for AFX10

print(CAL_DIR)
# -

# Define a configuration object for the processing.

# +



# TODO: The config_file should be better explained, and generated based on user settings. Dictionary could be good alternative
config_file_path = OUT_DIR + 'configuration.ini'

# Set the data directory for the mission (locally where the data is stored)
prepend_data_dir_to_relative_paths(config_path=config_file_path, DATA_DIR = OUT_DIR)

config = configparser.ConfigParser()
config.read(config_file_path)


specim_object = Specim(mission_path=MISSION_DIR, config=config)
# -

""" 
A mission directory is organized as follows. Missions need to be 
├───MissionDir
│   ├───configuration.ini # Configurations goes here
│   ├───<the_name_of_mat_file>.mat # Navigation data goes here
│   ├───Input # Necessary input for georeferencing
│   │   ├───H5 #The *.h5 files go here
│   │   ├───Calib #The camera calibration
│   │   └───GIS # The 3D terrain model and more goes here (not used here)
│   ├───Output
│   │   ├───3DModels # Here we get a point cloud version of the data (if we want)
│   │   └───GIS
│   │       ├───FootPrints
│   │       ├───HSIDatacubes
│   │       └───RGBComposites
│   ├───Intermediate
│   │   ├───Pickle (not relevant)
│   │   └───OrthoReshaped (not relevant)
"""

# +
# Reading envi format is achieved with spectral python.
# This cell reads the main capture and finds relevant configs from *.hdr data



PATTERN_ENVI = '*.hdr'
CAPTURE_DIR = MISSION_DIR + '/capture/'

search_path_envi = os.path.normpath(os.path.join(CAPTURE_DIR, PATTERN_ENVI))
ENVI_HDR_FILE_PATH = glob.glob(search_path_envi)[0]

spectral_image_obj = envi.open(ENVI_HDR_FILE_PATH)

# Read all meta of interest (make explicit to developer and accessible with autocomplete)
class Metadata:
    pass

metadata_obj = Metadata()
metadata_obj.autodarkstartline = int(spectral_image_obj.metadata['autodarkstartline'])
metadata_obj.n_lines = int(spectral_image_obj.metadata['lines'])
metadata_obj.n_bands = int(spectral_image_obj.metadata['bands'])
metadata_obj.n_pix = int(spectral_image_obj.metadata['samples'])
metadata_obj.t_exp_ms = float(spectral_image_obj.metadata['tint'])
metadata_obj.fps = float(spectral_image_obj.metadata['fps'])
metadata_obj.description = spectral_image_obj.metadata['description']
metadata_obj.file_type = spectral_image_obj.metadata['file type']
metadata_obj.sensor_type = spectral_image_obj.metadata['sensor type']
metadata_obj.acquisition_date = spectral_image_obj.metadata['acquisition date']
metadata_obj.sensorid = spectral_image_obj.metadata['sensorid']
metadata_obj.interleave = spectral_image_obj.metadata['interleave']
metadata_obj.data_type = spectral_image_obj.metadata['data type']
# USE FILES FROM LAB, not HEADER metadata_obj.wavelengths = np.array(spectral_image_obj.bands.centers)
#NOT CORRECT!!!!!! This is spectral sampling distance: metadata_obj.fwhm = np.array(spectral_image_obj.bands.bandwidths)
metadata_obj.binning_spatial = int(ACTIVE_SENSOR_SPATIAL_PIXELS/metadata_obj.n_pix)
metadata_obj.binning_spectral = int(ACTIVE_SENSOR_SPECTRAL_PIXELS/metadata_obj.n_bands)

# Binning solely determines which calibration files be used.

# It holds a csv like format



specim_object.metadata_obj = metadata_obj # Allow accesability for Specim Methods
# -

# Based on the binning info, we can locate relevant calibration files, including 1) spectral, 2) geometric, 3) radiometric, and dark frame (from capture).
#
# We start with the spectral calibration:

# +
"""Reads spectral calibration based on binning into a wavelength array and a fwhm array"""

# Comprehensive band info (center, fwhm) is found in "CAL_DIR/wlcal<spectral binning>b_fwhm.wls"
PATTERN_BAND_INFO = '*'+ str(metadata_obj.binning_spectral) + 'b_fwhm.wls'
# Linux-CLI search for file.
search_path_bands = os.path.normpath(os.path.join(CAL_DIR, PATTERN_BAND_INFO))
BAND_FILE_PATH = glob.glob(search_path_bands)[0]

df_bands = pd.read_csv(BAND_FILE_PATH, header=None, sep = '\s+')
df_bands.columns = ['Wavelength_nm', 'FWHM_nm']

specim_object.wavelengths = np.array(df_bands['Wavelength_nm'])
specim_object.fwhm = np.array(df_bands['FWHM_nm'])

# -

# We will here go through the loading of Specim geometric camera model. The first step is to load the angular Field-of-View file (AFOV) from the manufacturer. Then boresight angles and lever arms can be set, if relevant.

# +
# Pixel-directions is found in "CAL_DIR/FOV_****_<spatial binning>b.txt" 

PATTERN_FOV = 'FOV*' + '_' +  str(metadata_obj.binning_spatial) + 'b.txt'

# Search for fov file.
search_path_fov = os.path.normpath(os.path.join(CAL_DIR, PATTERN_FOV))
FOV_FILE_PATH = glob.glob(search_path_fov)[0]

# Calculates a camera model based on FOV file 
specim_object.read_fov_file(fov_file_path=FOV_FILE_PATH)

df_fov = pd.read_csv(FOV_FILE_PATH, header=None, sep = ',')

df_fov.columns = ['Pixel_Nr', 'View_Angle_Deg', 'Unknown']

specim_object.view_angles = np.array(df_fov['View_Angle_Deg'])
# -

binning_spatial = metadata_obj.binning_spatial
param_dict = Specim.fov_2_param(fov = specim_object.view_angles)
print(param_dict)

# The addition of boresight angles can be done by editing tx, ty, tz, and rx, ry, rz in the "OUTDIR/Input/Calib". 
#
# [tx, ty, tz] is the vector from HSI focal centre to reference origin (e.g. IMU) given in the reference frame. So if your BODY frame defines forward, right, down on the vehicle:
#
# [tx, ty, tz] = [1, 1, 1] means that HSI is 1 m behind, 1 m left of and 1 m above the IMU.
#
# Secondly, [rx, ry, rz] are Euler angles in radians with order 'ZYX'. If using another rotation convention, it's recommended to convert with scipy.spatial.Rotation. Example:
#
#     import scipy.spatial.Rotation as RotLib
#
#     # Let's say you have a rotation matrix transforming a vector from HSI frame to IMU frame
#     R_hsi_rgb = [[0, -1, 0],
#                  [1, 0, 0],
#                  [0, 0, 1]]
#
#

# +
from scripts.geometry import CalibHSI
# Before writing the calibration file, we need to think about the relative configuration of the Specim imager. 
# X: For simplicity let's assume that the x- axis of the HSI is pointing in opposite (-) direction as body y-axis, namely starboard
# Y: For simplicity let's assume that the y- axis of the HSI is pointing in same (+) direction as body x-axis, namely backward
# Z: For simplicity let's assume that the z- axis of the HSI is pointing in same direction as body z-axis (+), namely downwards
from scipy.spatial.transform import Rotation as RotLib

x_hsi = np.array([0, -1, 0]).reshape((-1,1)) # Same as statement above
y_hsi = np.array([1, 0, 0]).reshape((-1,1)) # Same as statement above
z_hsi = np.array([0, 0, 1]).reshape((-1,1)) # Same as statement above

# This knowledge allows us to assemble the rotation matrix for rotating points from HSI frame to RGB frame
R_hsi_rgb = np.concatenate((x_hsi, y_hsi, z_hsi), axis = 1)
r_zyx = RotLib.from_matrix(R_hsi_rgb).as_euler('ZYX', degrees=False)

# Lever arms are per my knowledge unknown at time (defaults to 0) being but they should be set if known
# param_dict['tz'] = 1 means that HSI origin is 1 m above body origin
# param_dict['ty'] = 1 means that HSI origin is 1 m to the left of body origin
# param_dict['tz'] = 1 means that HSI origin is 1 m behind of body origin

param_dict['rz'] = r_zyx[0]
param_dict['ry'] = r_zyx[1]
param_dict['rx'] = r_zyx[2]

param_dict['tz'] = 0
param_dict['ty'] = 0
param_dict['tz'] = 0

CAMERA_CALIB_XML_DIR = OUT_DIR + 'Input/Calib/'

file_name_xml = 'HSI_' + str(binning_spatial) + 'b.xml'
xml_cal_write_path = CAMERA_CALIB_XML_DIR + file_name_xml

CalibHSI(file_name_cal_xml= xml_cal_write_path, 
                 config = config, 
                 mode = 'w', 
                 param_dict = param_dict)


# Set value in config file:
config.set('Relative Paths', 'hsicalibfile', value = 'Input/Calib/' + file_name_xml)

with open(config_file_path, 'w') as configfile:
        config.write(configfile)
# -

""" Function added to remedy lacking byte-order entry in header files of radiometric calibration data"""
def add_byte_order_to_envi_header(header_file_path, byte_order_value):
    # Read the existing ENVI header
    with open(header_file_path, 'r') as f:
        header_lines = f.readlines()

    # Look for the line where you want to add "byte order"
    for i, line in enumerate(header_lines):
        if line.startswith('byte order'):
            # If it already exists, update the value
            header_lines[i] = f'byte order = {byte_order_value}\n'
            break
    else:
        # If it doesn't exist, add it to the end of the header
        header_lines.append(f'byte order = {byte_order_value}\n')

    # Save the updated header
    with open(header_file_path, 'w') as f:
        f.writelines(header_lines)


# The next step is to read in radiometric frame.

# +
"""Extract radiometric frame from dedicated file"""

PATTERN_ENVI_CAL = '*_' +str(metadata_obj.binning_spectral) + 'x' +  str(metadata_obj.binning_spatial) + '.hdr'

search_path_envi_cal = os.path.normpath(os.path.join(CAL_DIR, PATTERN_ENVI_CAL))
ENVI_CAL_HDR_FILE_PATH = glob.glob(search_path_envi_cal)[0]

RAD_CAL_BYTE_ORDER = 0

add_byte_order_to_envi_header(header_file_path=ENVI_CAL_HDR_FILE_PATH, byte_order_value=RAD_CAL_BYTE_ORDER)

ENVI_CAL_IMAGE_FILE_PATH = ENVI_CAL_HDR_FILE_PATH.split('.')[0] + '.cal' # SPECTRAL does not expect this suffix by default

# For some reason, the byte order

radiometric_image_obj = envi.open(ENVI_CAL_HDR_FILE_PATH, image = ENVI_CAL_IMAGE_FILE_PATH)

cal_n_lines = int(radiometric_image_obj.metadata['lines'])
cal_n_bands = int(radiometric_image_obj.metadata['bands'])
cal_n_pix = int(radiometric_image_obj.metadata['samples'])

radiometric_frame = radiometric_image_obj[:,:,:].reshape((cal_n_pix, cal_n_bands))

specim_object.radiometric_frame = radiometric_frame
# -

# The next step is to load the darkframes

# +
"""1) Crop the hyperspectral data according to the start-stop lines. 2) Write datacube to appropriate directory"""
# To ensure that the plots actually do appear in this notebook:
# %matplotlib qt

# Establish dark frame data (at end of recording)
data_dark = spectral_image_obj[metadata_obj.autodarkstartline:metadata_obj.n_lines, :, :]
dark_frame = np.median(data_dark, axis = 0)

specim_object.dark_frame = dark_frame




# -

# The navigation data is given as messages is a *.nav file. Locate the file and parse it into a suitable format. From NAVIGEOPRO we'd get a sync file giving the pose per scan. Matching against such a file makes sense.

# +
# Extract the starting/stopping lines
import pandas as pd

PATTERN_START_STOP = '*.txt'
START_STOP_DIR = MISSION_DIR + '/start_stop_lines'

search_path_lines_start_stop = os.path.normpath(os.path.join(START_STOP_DIR, PATTERN_START_STOP))
LINES_START_STOP_FILE_PATH = glob.glob(search_path_lines_start_stop)[0]

header = 0

df_start_stop = pd.read_csv(filepath_or_buffer=LINES_START_STOP_FILE_PATH, header=header, sep=' ')



# +
# Now read the *.nav file
NAV_PATTERN = '*.nav'

search_path_nav = os.path.normpath(os.path.join(CAPTURE_DIR, NAV_PATTERN))
nav_file_path = glob.glob(search_path_nav)[0]

# Parse the position/orientation messages
specim_object.read_nav_file(nav_file_path=nav_file_path, date = DATE)



# -

# Calculate the frame timestamps from sync data

# +
import pymap3d as pm
from scipy.interpolate import interp1d


df_imu = pd.DataFrame(specim_object.imu_data)
df_gnss = pd.DataFrame(specim_object.gnss_data)
df_sync_hsi = pd.DataFrame(specim_object.sync_data)
# Define the time stamps of HSI frames


# Let's consider this an interpolation problem. Every new sync means a new fps # frames:
sync_frames = df_sync_hsi['HsiFrameNum']
sync_times = df_sync_hsi['TimestampAbs']
hsi_frames = np.arange(metadata_obj.autodarkstartline)


hsi_timestamps_total = interp1d(x = sync_frames, y= sync_times, fill_value = 'extrapolate')(x = hsi_frames)


# Secondly, for ease, let us interpolate position data to imu time (avoids rotational interpolation)
imu_time = df_imu['TimestampAbs']

# Drop the specified regular clock time (as it is not needed)
df_gnss = df_gnss.drop(columns=['TimestampClock'])

# Interpolate each column in GNSS data based on 'imu_time'
interpolated_values = {
    column: np.interp(imu_time, df_gnss['TimestampAbs'], df_gnss[column])
    for column in df_gnss.columns if column != 'TimestampAbs'
}


# Create a new DataFrame with the interpolated values
df_gnss_interpolated = pd.DataFrame({'time': imu_time, **interpolated_values})

# The position defined in geodetic coordinates
lat = np.array(df_gnss_interpolated['Lat']).reshape((-1,1))
lon = np.array(df_gnss_interpolated['Lon']).reshape((-1,1))
ellipsoid_height = np.array(df_gnss_interpolated['AltMSL'] + df_gnss_interpolated['AltGeoid']).reshape((-1,1))

# Assumes WGS-84 (default GNSS frame)
x, y, z = pm.geodetic2ecef(lat = lat, lon = lon, alt = ellipsoid_height, deg=True)

# Lastly, calculate the roll, pitch, yaw
roll = np.array(df_imu['Roll']).reshape((-1,1))
pitch = np.array(df_imu['Pitch']).reshape((-1,1))
yaw = np.array(df_imu['Yaw']).reshape((-1,1))

# Roll pitch yaw are stacked with in an unintuitive attribute. The euler angles with rotation order ZYX are Yaw Pitch Roll
specim_object.eul_zyx = np.concatenate((roll, pitch, yaw), axis = 1)

# Position is stored as ECEF cartesian coordinates (mutually orthogonal axis) instead of spherioid-like lon, lat, alt
specim_object.position_ecef = np.concatenate((x,y,z), axis = 1)
specim_object.nav_timestamp = imu_time
specim_object.t_exp_ms = metadata_obj.t_exp_ms

# -

# Last preprocessing step is writing to h5 files. 

# Format the data for use in the geometric processing pipeline
h5_dict_write = {'eul_zyx' : 'raw/nav/euler_angles',
           'position_ecef' : 'raw/nav/position_ecef',
           'nav_timestamp' : 'raw/nav/timestamp',
           'radiance_cube': 'processed/radiance/radiance_cube',
           't_exp_ms': 'processed/radiance/t_exp_ms',
           'hsi_timestamps': 'processed/radiance/timestamp',
           'view_angles': 'processed/radiance/calibration/geometric/view_angles',
           'wavelengths' : 'processed/radiance/calibration/spectral/wavelengths',
           'fwhm' : 'processed/radiance/calibration/spectral/fwhm',
           'dark_frame' : 'processed/radiance/calibration/dark_frame',
           'radiometric_frame' : 'processed/radiance/calibration/radiometric_frame'}

# Time to write all the data to a h5 file

import h5py
"""Writer for the h5 file format using a dictionary. The user provides h5 hierarchy paths as values and keys are the names given to the attributes of the specim object.
A similar write process could be applied to metadata."""
def specim_object_2_h5_file(h5_filename, h5_tree_dict, specim_object):
    with h5py.File(h5_filename, 'w', libver='latest') as f:
        for attribute_name, h5_hierarchy_item_path in h5_tree_dict.items():
            print(attribute_name)
            dset = f.create_dataset(name=h5_hierarchy_item_path, 
                                            data = getattr(specim_object, attribute_name))


# # Chunking the recording to user defined sizes and writing it to disk

# +
# Define h5 file name
H5_DIR = OUT_DIR + 'Input/H5/'

# Every 1000 lines take up 0.85 GB at 8 Byte float. Therefore it could make sense to partition things that are larger than 2000 lines (sub GB for 32 float/4 byte)
dtype = np.float32
TRANSECT_CHUNK_SIZE_GB = 2
TRANSECT_CHUNK_SIZE = 2000 # The number of lines (could also make a simple calculator for this)

# It is nicer to deal with 4 byte numbers in general
n_transects = df_start_stop.shape[0]
for transect_number in range(n_transects):
    start_line = df_start_stop['line_start'][transect_number]
    stop_line = df_start_stop['line_stop'][transect_number]

    n_chunks = int(np.ceil((stop_line-start_line)/TRANSECT_CHUNK_SIZE))

    
    for chunk_number in range(n_chunks):
        chunk_start_idx = start_line + TRANSECT_CHUNK_SIZE*chunk_number

        if chunk_number == n_chunks-1:
            chunk_stop_idx = stop_line
        else:
            chunk_stop_idx = chunk_start_idx + TRANSECT_CHUNK_SIZE



        data_cube = spectral_image_obj[chunk_start_idx:chunk_stop_idx, :, :]
        # Calibration equation
        specim_object.radiance_cube = ( (data_cube - dark_frame)*radiometric_frame/(metadata_obj.t_exp_ms/1000) ).astype(dtype = dtype) # 4 Byte
        specim_object.hsi_timestamps = hsi_timestamps_total[chunk_start_idx:chunk_stop_idx]

        # Possible to name files with <PREFIX>_<time_start>_<Transect#>_<Chunk#>.h5
        h5_filename = H5_DIR + MISSION_NAME_PREFIX + '_transectnr_' + str(int(transect_number)) + '_chunknr_' + str(int(chunk_number)) + '.h5'

        specim_object_2_h5_file(h5_filename=h5_filename, h5_tree_dict=h5_dict_write, specim_object=specim_object)
