import xarray as xr
import dask
from xclim.core.units import convert_units_to
from xclim.testing import open_dataset
from xclim import sdba
from rechunker import rechunk
import glob

import matplotlib.pyplot as plt
import numpy as np
import xclim as xc

import sys
import subprocess
import time
import uuid
import json
    
from dask_jobqueue import PBSCluster
from dask.distributed import Client

import os


def standardise_dimension_names(ds):
    ds = rename_variable_in_ds(ds, 'latitude', 'lat')
    ds = rename_variable_in_ds(ds, 'longitude', 'lon')
    return(ds)
    

def standardise_latlon(ds, digits=4):
    ds = ds.assign_coords({"lat": np.round(ds.lat,digits)})
    ds = ds.assign_coords({"lon": np.round(ds.lon,digits)})
    return(ds)
    

def rename_variable_in_ds(ds, old_name, new_name):
    dict_temp = dict(zip([old_name], [new_name]))
    if old_name in ds.variables or old_name in ds.dims:
        ds = ds.rename(dict_temp)
    return(ds)


def assign_latlon(ds,ref_ds):
    ds = ds.assign_coords({"lat": ref_ds.lat.values})
    ds = ds.assign_coords({"lon": ref_ds.lon.values})
    return ds


def preprocess_gcm(ds):
    if ds.time.dtype == "O":
        ds = ds.assign_coords(time = ds.indexes['time'].to_datetimeindex()) 
    ds = standardise_latlon(ds)
    ds = ds.drop_vars([item for item in ('height', 'lat_bnds', 'lon_bnds', 'time_bnds') if item in ds.variables or item in ds.dims])
    ds = ds.assign_coords(time = ds.time.dt.floor("D") + np.timedelta64(12, 'h'))
    return ds


if __name__ == "__main__":

    mdl = sys.argv[1]
    mdl_params = sys.argv[2]
    downscaling = sys.argv[3]
    task = sys.argv[4]

    zarr_path = "/scratch/eg3/ag0738/mbcn/npcp/"
    
    dask.config.set(**{'array.slicing.split_large_chunks': False,
                       'distributed.comm.timeouts.tcp': '300s',
                       'distributed.scheduler.worker-saturation':1,
                       'distributed.scheduler.work-stealing':False,
                       'array.rechunk.algorithm':'p2p',
                       'jobqueue.pbs.scheduler_options':{'protocol':'tcp://'}})
    
    # client from scheduler
    client = Client(scheduler_file = os.environ["DASK_PBS_SCHEDULER"])

    final_chnk_size = 20
    read_chunk_time = 300
    chnk_size_tm = -1
    
    # reg_lons = [112,124]
    # reg_lats = [-44,-30]
    
    reg_lons = [112,154]
    reg_lats = [-44,-10]

    if task == "validation":
        if mdl != "ECMWF-ERA5":
            train_yr_start = 1960
            train_yr_end = 1989
    
            proj_yr_start = 1990
            proj_yr_end = 2019
        else:
            train_yr_start = 1980
            train_yr_end = 1999

            proj_yr_start = 2000
            proj_yr_end = 2019
        
    else:
        if mdl != "ECMWF-ERA5":
            train_yr_start = 1980
            train_yr_end = 2019
    
            proj_yr_start = 2060
            proj_yr_end = 2099

        else:
            train_yr_start = 1980
            train_yr_end = 2019
            
            proj_yr_start = 1980
            proj_yr_end = 2019

    vars = ["pr", "tasmax", "tasmin"]
    scaling_kind = {"pr": '*', "tasmax": '+', "tasmin": '+', "sfcWind": '*', "rsds": '*'}

    ref_path = '/g/data/ia39/npcp/data/{var}/observations/AGCD/raw/task-reference/{var}_NPCP-20i_AGCD_v1-0-1_day_{year}0101-{year}1231.nc'
    gcm_path = '/g/data/ia39/npcp/data/{var}/{mdl}/{downscaling}/raw/task-reference/{var}_NPCP-20i_{mdl}_{empat}_{params}_{downscaling}_v1_day_{year}0101-{year}1231.nc'

    ref_file_list = [ref_path.format(var=v, year=y) for v in vars for y in range(train_yr_start, train_yr_end + 1)]
    hist_file_list = [gcm_path.format(var=v, year=y, mdl = mdl, downscaling = downscaling, params = mdl_params, empat = 'evaluation' if mdl == 'ECMWF-ERA5' else ('historical' if y <= 2014 else 'ssp370')) for v in vars for y in range(train_yr_start, train_yr_end + 1)]
    fut_file_list = [gcm_path.format(var=v, year=y, mdl = mdl, downscaling = downscaling, params = mdl_params, empat = 'evaluation' if mdl == 'ECMWF-ERA5' else ('historical' if y <= 2014 else 'ssp370')) for v in vars for y in range(proj_yr_start, proj_yr_end + 1)]
    
    ref_data_full = xr.open_mfdataset(ref_file_list, preprocess = preprocess_gcm, chunks = {"time": read_chunk_time})
    hist_data_full = xr.open_mfdataset(hist_file_list, preprocess = preprocess_gcm, chunks = {"time": read_chunk_time})
    fut_data_full = xr.open_mfdataset(fut_file_list, preprocess = preprocess_gcm, chunks = {"time": read_chunk_time})
    
    # hist_data_full = xr.open_mfdataset(hist_file_list, preprocess = preprocess_cft if (mdl == "CSIRO-ACCESS-ESM1-5" and downscaling == "UQ-DES-CCAM-2105") else preprocess_gcm, chunks = {"time": read_chunk_time})
    # fut_data_full = xr.open_mfdataset(fut_file_list, preprocess = preprocess_cft if (mdl == "CSIRO-ACCESS-ESM1-5" and downscaling == "UQ-DES-CCAM-2105") else preprocess_gcm, chunks = {"time": read_chunk_time})

    ref_data = ref_data_full.sel(lon = slice(*reg_lons), lat = slice(*reg_lats), time = hist_data_full.time).chunk(chunks={"lat":final_chnk_size,"lon":final_chnk_size,"time":chnk_size_tm})
    hist_data = hist_data_full.sel(lon = slice(*reg_lons), lat = slice(*reg_lats)).chunk(chunks={"lat":final_chnk_size,"lon":final_chnk_size,"time":chnk_size_tm})
    fut_data = fut_data_full.sel(lon = slice(*reg_lons), lat = slice(*reg_lats)).chunk(chunks={"lat":final_chnk_size,"lon":final_chnk_size,"time":chnk_size_tm})

    # ref_data = ref_data.assign_coords(time = hist_data.time)

    # remove == 0 values in pr:
    if "pr" in vars:
        ref_data["pr"] = sdba.processing.jitter(ref_data.pr, lower="1 mm d-1",minimum="0 mm d-1")
        hist_data["pr"] = sdba.processing.jitter(hist_data.pr, lower="1 mm d-1",minimum="0 mm d-1")
        fut_data["pr"] = sdba.processing.jitter(fut_data.pr, lower="1 mm d-1",minimum="0 mm d-1")

    ref_data.persist()
    hist_data.persist()
    fut_data.persist()

    scenh_list = []
    scenf_list = []
    
    for var in vars:
        qdm = sdba.QuantileDeltaMapping.train(
            ref_data[var], hist_data[var], nquantiles=100, kind=scaling_kind[var], group="time"
        )
        
        # Adjust both hist and sim, we'll feed both to the Npdf transform.
        scenh_list.append(qdm.adjust(hist_data[var]))
        scenf_list.append(qdm.adjust(fut_data[var]))

    scenh = xr.Dataset({vars[i]:scenh_list[i] for i in range(len(vars))})
    scenf = xr.Dataset({vars[i]:scenf_list[i] for i in range(len(vars))})

    ref = sdba.processing.stack_variables(ref_data)
    scenh = sdba.processing.stack_variables(scenh)
    scenf = sdba.processing.stack_variables(scenf)
    
    ref, _, _ = sdba.processing.standardize(ref)

    if train_yr_start == proj_yr_start and train_yr_end == proj_yr_end:
        allsim, savg, sstd = sdba.processing.standardize(scenh)
    else:
        allsim, savg, sstd = sdba.processing.standardize(xr.concat((scenh, scenf), "time"))
    hist = allsim.sel(time=scenh.time)
    sim = allsim.sel(time=scenf.time)
    
    d = { 'ref':ref,'hist':hist,'sim':sim,'sstd':sstd,'savg':savg }

    ### https://discourse.pangeo.io/t/collecting-problematic-workloads/3683/2
    def write_wrapper(k, ds):
        path = f"{zarr_path}/{mdl}-{downscaling}-{task}-{k}"
        with dask.distributed.worker_client() as client:
            ds.to_zarr(path)
        return k
       
    futures = client.map(
        write_wrapper,
        d.keys(),
        d.values()
    )
    f=dask.distributed.wait(futures)
