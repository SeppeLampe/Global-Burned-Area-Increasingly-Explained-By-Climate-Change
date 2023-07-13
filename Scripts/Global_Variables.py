import os
import numpy as np
import iris
import regionmask

BASE_PATH = os.path.dirname(os.getcwd()) # Get the parent directory of the directory where this file is located

DATA_PATH = os.path.join(BASE_PATH, 'Data', "")
OBSERVATIONS_PATH = os.path.join(DATA_PATH, 'Observations', "")
SUPPLEMENTARY_DATA_PATH = os.path.join(DATA_PATH, 'Supplementary_Data', "")
MODEL_PATH = os.path.join(DATA_PATH, 'ISIMIP', 'ISIMIP3a', 'OutputData', 'fire', "")

RESULTS_PATH = os.path.join(BASE_PATH, 'Results', "")
FIGURES_PATH = os.path.join(BASE_PATH, 'Visuals', 'Figures', "")
TABLES_PATH = os.path.join(BASE_PATH, 'Visuals', 'Tables', "")


SEED = int.from_bytes("They're taking the hobbits to Isengard!".encode('utf-8')) # Create a seed, which we can use to make our 'random' results repeatable.


obs_dict = {}
for obsname in [f.name for f in os.scandir(OBSERVATIONS_PATH) if f.is_dir()]:
    for dirpath, subdirs, files in os.walk(os.path.join(OBSERVATIONS_PATH, obsname)):
        for filename in files:
            if 'Burned_Percentage.nc' in filename:
                obs_dict[obsname] = os.path.join(dirpath, filename)
                

model_dict = {}
for modelname in [f.name for f in os.scandir(MODEL_PATH) if f.is_dir()]:
    model_dict[modelname] = {}
    for dirpath, subdirs, files in os.walk(os.path.join(MODEL_PATH, modelname)):
        for filename in files:
            if filename.endswith('gswp3-w5e5_obsclim_histsoc_default_burntarea-total_global_monthly_1901_2019.nc'):
                model_dict[modelname]['obsclim'] = os.path.join(dirpath, filename)
            elif filename.endswith('gswp3-w5e5_counterclim_histsoc_default_burntarea-total_global_monthly_1901_2019.nc') \
            or filename.endswith('gswp3-w5e5_counterclim_histsoc_1901co2_burntarea-total_global_monthly_1901_2019.nc'): # Files should be named via the convention above. Delete this line after all files follow ISIMIP convention.
                model_dict[modelname]['counterclim'] = os.path.join(dirpath, filename)
                
                
cs = iris.coord_systems.GeogCS(6371229)
lon = np.arange(-179.75, 180, 0.5)
lat = np.arange(-89.75, 90, 0.5)
AR6_names = ['GIC', 'NWN', 'NEN', 'WNA', 'CNA', 'ENA', 'NCA', 'SCA', 'CAR', 'NWS', 'NSA', 'NES', 'SAM', 'SWS', 'SES', 'SSA', 'NEU', 'WCE', 'EEU', 'MED', 'SAH', 'WAF', 'CAF', 'NEAF', 'SEAF', 'WSAF', 'ESAF', 'MDG', 'RAR', 'WSB', 'ESB', 'RFE', 'WCA', 'ECA', 'TIB', 'EAS', 'ARP', 'SAS', 'SEA', 'NAU', 'CAU', 'EAU', 'SAU', 'NZ']
AR6_mask = ~regionmask.defined_regions.ar6.land.mask_3D(lon, lat) # Create the mask on the 0.5 by 0.5 degree scale and reverse it (tilde) so the region of interest = False
# Conversion to np.array necessary as cube.data.data is expected to be a numpy array, not an xarray.DataArray
# All the metadata for the dim_coords is also required to match zith the obs/model dim_coords
AR6_masks = {name: 
             iris.util.reverse(
                 iris.cube.Cube(np.array(AR6_mask[idx]), 
                            dim_coords_and_dims=[(iris.coords.DimCoord(AR6_mask[idx].coords['lat'], standard_name = 'latitude', units='degrees', long_name='Latitude', var_name='lat', coord_system=cs), 0), 
                                                 (iris.coords.DimCoord(AR6_mask[idx].coords['lon'], standard_name = 'longitude', units='degrees', long_name='Longitude', var_name='lon', coord_system=cs), 1)], 
                            var_name=name),
                 'latitude')
             for idx, name in enumerate(AR6_names)}
del AR6_masks['GIC']

for mask in AR6_masks.values():
    mask.coords('latitude')[0].guess_bounds()
    mask.coords('longitude')[0].guess_bounds()