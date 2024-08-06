import os
import math
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import iris
from iris import cube
from iris import coord_categorisation
import cf_units

from Global_Variables import SUPPLEMENTARY_DATA_PATH, SEED

land_fractions = os.path.join(SUPPLEMENTARY_DATA_PATH, 'Land Fractions')

def preprocess_model(cube, modelname):    
    # Function that applies formatting that needs to be done on model data.
    # Mainly to comply with iris' strict conventions.
    # Since the data is not yet on the ISIMIP repository, many of the model data needs to undergo slightly different preprocessing.
    
    # Change the coordinates from months since to days since because cftime does not support months (it's not CF convention)
    cube.coords('time')[0].units = cf_units.Unit('days since 1901-01-01', cf_units.CALENDAR_360_DAY)
    cube.coords('time')[0].points = np.arange(1428)*30   
    cube.coords('time')[0].bounds = None
    cube.coords('time')[0].long_name = 'time'
    
    # Fix the Latitude and Longitude
    cs = iris.coord_systems.GeogCS(6371229)
    cube.coords('latitude')[0].coord_system = cs
    cube.coords('longitude')[0].coord_system = cs
    cube.coords('latitude')[0].long_name = 'Latitude'
    cube.coords('longitude')[0].long_name = 'Longitude'
    cube.coords('latitude')[0].guess_bounds()
    cube.coords('longitude')[0].guess_bounds()
    
    if modelname == 'ORCHIDEE-MICT-SPITFIRE':
            cube.remove_coord(cube.coords('time')[1]) # Remove Auxcoord (was there already)
            
            # Data is in fraction per day (on a 365 day calendar) instead of fraction per month (on a 360 day calendar), multiply by length of month (no leap years)
            # First, add the month of year as a coord, then calculate the days in each month in a non-leap year.
            # Lastly, multiply the data for each month by the number of days in that month
            month_lengths = np.array([(np.arange(f'{2005}-{str(x).rjust(2, "0")}', f'{2005+x//12}-{str(x%12+1).rjust(2, "0")}', dtype='datetime64[D]')).size for x in range(1, 13)]).repeat(cube.coords('time')[0].points.size//12)
            cube = iris.analysis.maths.multiply(cube, month_lengths[:, None, None], dim='time')
            
            land_frac = xr.open_dataset(os.path.join(land_fractions, 'ORCHIDEE.nc')).contfrac            
            cube = iris.analysis.maths.multiply(cube, land_frac)
            
    elif 'LPJ-GUESS' in modelname:        
        # Latitude is the other way around for the LPJ models
        cube = iris.util.reverse(cube, 'latitude')
        # Data is not, or not properly masked
        cube.data.mask = np.isnan(cube.data.data)
    
    elif modelname == 'CLASSIC':
        land_frac = xr.open_dataset(os.path.join(land_fractions, f'{modelname}.nc')).sftlf
        cube = iris.analysis.maths.multiply(cube, land_frac)
        
    elif modelname == 'SSiB4':
        # Data is in percentage per day instead of percentage per month, multiply by 30
        cube = iris.analysis.maths.multiply(cube, 30)
    
    elif modelname == 'JULES':
        land_frac = xr.open_dataset(os.path.join(land_fractions, f'{modelname}.nc')).LSM
        land_frac = xr.where(land_frac>1, 0, land_frac) # Ocean pixels are marked as very high values, other pixels are set to [0-1] let's set the ocean tiles to 0
        cube = iris.analysis.maths.multiply(cube, land_frac)
        
    cube.units = cf_units.Unit('%')
    cube.rename('Percentage of Area Burned')
    
    return cube



def to_timeseries(cube):
    # Transforms an iris cube into a timeseries (take the area-weighted sum over the longitude and latitude dimensions).
    # It also scales the variable by 10**12 i.e., %m^2 to Mha (*100 to remove % and * 10^10 to go from m^2 to Mha).
    coords = ('longitude', 'latitude')
    for coord in coords:
        if not cube.coord(coord).has_bounds():
            cube.coord(coord).guess_bounds()
    weights = iris.analysis.cartography.area_weights(cube) # pixel size in m^2
    return cube.collapsed(coords, iris.analysis.SUM, weights = weights)/10**12

def pixel_mean(cube):
    return cube.collapsed(('time'), iris.analysis.MEAN)

def add_year_coord(cube):
    if not any([coord.long_name == 'year' for coord in cube.coords()]):
        iris.coord_categorisation.add_year(cube, 'time', name='year')
    return cube


def add_global(df):
    df_global = df.T.groupby(level=1, sort=False, dropna=True, observed=True).sum(min_count=1).T
    df_global.columns = pd.MultiIndex.from_product([pd.CategoricalIndex(['Global'], categories=df.columns.levels[0].categories), df_global.columns], names=df.columns.names)
    return df.join(df_global)

def prepare_for_facetgrid(df):
    return df.melt(ignore_index=False).reset_index().rename(columns={'value': 'BA'})

def drop_models(df, modelnames):
    return df.drop(modelnames, axis=1, level=1)

def select_models(df, modelnames):
    return df.loc[:, (slice(None), modelnames)]

def select_model(df, modelnames):
    # Alias for select_models
    return select_models(df, modelnames)

def select_region(df, regionname):
    return df.loc[:, (regionname, slice(None))]

def to_anomaly(df):
    return df - df.mean()

def relative_anomaly(df, df2=None):
    df2 = df if df2 is None else df2
    return (df - df2.mean())/df2.mean()


def weighted_avg_and_stddev(df, weights):
    weighted_average = (df*weights.values).T.groupby(level='Region', sort=False).sum().T
    variance = df.subtract(weighted_average, axis=0)**2
    weighted_variance = variance*weights
    weighted_stddev = np.sqrt(weighted_variance.T.groupby(level='Region', sort=False).sum().T)
    return weighted_average, weighted_stddev


def to_global(df):
    if 'Observation' in df.columns.names:
        level = 'Observation'
    elif 'Model' in df.columns.names:
        level = 'Model'
    else:
        raise ValueError("Observation or Model should be in the column level names (df.column.names)")
    if 'Global' in df.columns.unique('Region'):
        return df.loc[slice(None), 'Global']
    return df.T.groupby(level=level, observed=True).sum(min_count=1).T




def constrain_time(obj, start, end):
    # Constrains the time of an iris Cube or pandas DataFrame to be between the two dates (both included).
    if type(obj) == iris.cube.Cube:
        obj = add_year_coord(obj)
        return obj.extract(iris.Constraint(time=lambda cell: start <= cell.point.year <= end))
    
    elif type(obj) in (pd.core.frame.DataFrame, pd.core.series.Series):
        return obj.loc[str(start):str(end)]
    
    raise TypeError('This function expects "cube_or_df" to be an iris Cube or a pandas DataFrame.')
         
def to_annual(obj, sum_or_mean='sum'):
    if type(obj) == iris.cube.Cube:
        obj = add_year_coord(obj) 
        if sum_or_mean == 'sum':
            return obj.aggregated_by(['year'], iris.analysis.SUM)
        elif sum_or_mean == 'mean':
            return obj.aggregated_by(['year'], iris.analysis.MEAN)
        else:
            raise ValueError('sum_or_mean should be "sum" or "mean".') 
    elif type(obj) in (pd.core.frame.DataFrame, pd.core.series.Series):
        # 1. Convert the index to year to_period('Y'), which turns it into a pandas.PeriodIndex.
        # 2. Get the year as int (.year, seaborn doesn't know how to handle a pandas.PeriodIndex)
        # 3. Group by year .groupby()
        grouped_yearly = obj.groupby(obj.index.to_period('Y').year)
        # 4. Take the sum or mean of each group. Do not use NaNs (a sum or mean of NaNs equals 0 (undesirable))
        if sum_or_mean == 'sum':
            return grouped_yearly.sum(numeric_only=True, min_count=1)
        elif sum_or_mean == 'mean':
            return grouped_yearly.mean(numeric_only=True)
        else:
            raise ValueError('sum_or_mean should be "sum" or "mean".')  
    raise TypeError('This function expects "cube_or_df" to be an iris Cube or a pandas DataFrame.')      

    
    
def mask_region(cube, region_mask):
    return iris.util.mask_cube(cube, region_mask, in_place=False, dim=0)


# Adding extra things to a seaborn Facetgrid requires you to make functions if you want to do something that can't be included easily into one lambda function. <br>
# The next two functions aren't the nicest (they're terrible), but they get the job done (somehow).

# Will be used to add CO2 to plots
def add_var(x, y, color, data, ylabel, scale_param, **kwargs):
    new_ax = plt.twinx()
    sns.lineplot(data=data, x=x, y=y, color=color, ax=new_ax,  **kwargs)
    new_ax.set_ylabel(ylabel)
    ymax = (data.max()[scale_param]/data.groupby('Date').mean().reset_index().max()[scale_param])*100 + 300
    new_ax.set(ylim=(290, ymax))
    new_ax.spines['left'].set_visible(False)    

# Will be used to add correlation value to plots
def add_corr(x, y, data, **kwargs):
    ymin, ymax = data.loc[:, x].min(), data.loc[:, x].max()
    y_text = ymin + (ymax - ymin)*0.95
    data = data.groupby(['Date', 'Region']).mean().reset_index()
    x, y = data.loc[:, x], data.loc[:, y]
    corr = pearsonr(x.values, y.values)
    plt.text(x=1900, y=y_text, s=f'r = {round(corr[0], 3)}')





def NME1(obs, model):
    obs = obs.collapsed('time', iris.analysis.MEAN)
    model = model.collapsed('time', iris.analysis.MEAN)
    area_map = iris.analysis.cartography.area_weights(model, normalize=True)
        
    numerator = iris.analysis.maths.abs(iris.analysis.maths.subtract(obs, model)).collapsed(['latitude', 'longitude'], iris.analysis.SUM, weights=area_map)
    denominator =  iris.analysis.maths.abs(iris.analysis.maths.subtract(obs, obs.collapsed(['latitude', 'longitude'], iris.analysis.MEAN))).collapsed(['latitude', 'longitude'], iris.analysis.SUM, weights=area_map)
    
    return iris.analysis.maths.divide(numerator, denominator).data.item()


def NME2(obs, model):
    obs = obs.collapsed('time', iris.analysis.MEAN)
    model = model.collapsed('time', iris.analysis.MEAN)
    area_map = iris.analysis.cartography.area_weights(model, normalize=True)
    
    obs_i = iris.analysis.maths.subtract(obs, obs.collapsed(['latitude', 'longitude'], iris.analysis.MEAN))
    model_i = iris.analysis.maths.subtract(model, model.collapsed(['latitude', 'longitude'], iris.analysis.MEAN))
        
    numerator = iris.analysis.maths.abs(iris.analysis.maths.subtract(obs_i, model_i)).collapsed(['latitude', 'longitude'], iris.analysis.SUM, weights=area_map)
    denominator =  iris.analysis.maths.abs(obs_i).collapsed(['latitude', 'longitude'], iris.analysis.SUM, weights=area_map)
    
    return iris.analysis.maths.divide(numerator, denominator).data.item()

def NME3(obs, model):
    obs = obs.collapsed('time', iris.analysis.MEAN)
    model = model.collapsed('time', iris.analysis.MEAN)
    area_map = iris.analysis.cartography.area_weights(model, normalize=True)
    
    obs_mean = obs.collapsed(['latitude', 'longitude'], iris.analysis.MEAN)
    model_mean = model.collapsed(['latitude', 'longitude'], iris.analysis.MEAN)
    
    obs_i = iris.analysis.maths.divide(iris.analysis.maths.subtract(obs, obs_mean), obs_mean)
    sim_i = iris.analysis.maths.divide(iris.analysis.maths.subtract(model, model_mean), model_mean)
        
    numerator = iris.analysis.maths.abs(iris.analysis.maths.subtract(obs_i, sim_i)).collapsed(['latitude', 'longitude'], iris.analysis.SUM, weights=area_map)
    denominator =  iris.analysis.maths.abs(obs_i).collapsed(['latitude', 'longitude'], iris.analysis.SUM, weights=area_map)
    
    return iris.analysis.maths.divide(numerator, denominator).data.item()



def NME1_temporal(obs_df, model_df):   
    numerator = np.abs(obs_df-model_df)
    denominator = np.abs(obs_df-obs_df.mean())
    return numerator.sum()/denominator.sum()

def NME2_temporal(obs_df, model_df):
    obs_anomaly = obs_df - obs_df.mean()
    sim_anomaly = model_df - model_df.mean()
        
    numerator = np.abs(obs_anomaly-sim_anomaly)
    denominator = np.abs(obs_anomaly)
    
    return numerator.sum()/denominator.sum()

def NME3_temporal(obs_df, model_df):
    obs_anomaly = obs_df - obs_df.mean()
    obs_deviation = np.abs(obs_anomaly).mean()
    obs_i = obs_anomaly/obs_deviation
    
    sim_anomaly = model_df - model_df.mean()
    sim_deviation = np.abs(sim_anomaly).mean()
    sim_i = sim_anomaly/sim_deviation
    
    numerator = np.abs(obs_i-sim_i)
    denominator = np.abs(obs_i)
    
    return numerator.sum()/denominator.sum()    

def create_rng(seed):
    return np.random.default_rng(seed)


def log_transform(df):
    return np.log(df+1+1/df.index.size)

def log_inverse(df):
    return np.exp(df) -1 -1/df.index.size


def add_error(df, seed, error):
    error_repeated = pd.concat([error]*int(df.shape[0]/error.shape[0]))
    error_matrix = pd.DataFrame(create_rng(seed).normal(loc=0, scale=error_repeated*np.sqrt(math.pi/2), size=df.shape), df.index, df.columns) #
    return log_inverse(log_transform(df) + error_matrix)
'''

def add_error(df, seed, error):
    error_repeated = pd.concat([error]*int(df.shape[0]/error.shape[0]))
    error_matrix = pd.DataFrame(create_rng(seed).normal(loc=0, scale=error_repeated*np.sqrt(math.pi/2), size=df.shape), df.index, df.columns)
    return (df + error_matrix).where(lambda val: val>-1, other=-1)
'''

def get_results(df, num_resamples, other_mean=False):
    series = df.sample(n=num_resamples, replace=True, weights='weights', random_state=create_rng(SEED))['RA']
    results = series.quantile(q=[0.025, 0.5, 0.975])
    if not other_mean:
        other_mean = series.mean()
        results['mean'] = other_mean
    results['fraction'] = (series>other_mean).mean()
    return results


def get_results_global(df, num_resamples, quantiles=[]):
    series = df.sample(n=num_resamples, replace=True, weights='weights', random_state=create_rng(SEED))['RA']
    if not any(quantiles):
        quantiles = series.quantile(q=np.round(np.arange(0.1, 1, 0.1), 1))
        quantiles = quantiles.reset_index().rename(columns={'index': 'Quantile'}).set_index('Quantile')
    for quantile in quantiles.index:
        quantiles.loc[quantile, 'fraction'] = (series>quantiles.loc[quantile, 'RA']).mean()
    return quantiles

def scale_values_fig_2(array):
    return np.where(array>1, array-1, (1/-array)+1)

