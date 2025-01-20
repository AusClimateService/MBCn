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


def output_preproc(ds,fn_chnk_size_tm,outname):
    # rechunk
    ds = ds.chunk(chunks = {
        'time' : chnk_size_tm,
        'lat'  : -1,
        'lon'  : -1
    }).to_netcdf(
        path = outname,
        format = 'netcdf4',
        engine = 'netcdf4',
        unlimited_dims = ['time'],
        compute = False
    )
    return ds


if __name__ == "__main__":
    
    dask.config.set(**{'distributed.scheduler.work-stealing':True,
                       'jobqueue.pbs.scheduler_options':{'protocol':'ucx://'},
                       'distributed.scheduler.default-task-durations': {'rechunk-split': '1us', 'split-shuffle': '1us', 'npdf_transform':'15m'}})
    
    # client from scheduler
    client = Client(scheduler_file = os.environ["DASK_PBS_SCHEDULER"])

    
    mdl = sys.argv[1]
    mdl_params = sys.argv[2]
    downscaling = sys.argv[3]
    task = sys.argv[4]

    zarr_path = f'/scratch/eg3/ag0738/mbcn/npcp'
    

    final_chnk_size = 20
    read_chunk_time = 1000
    chnk_size_tm = -1


    ref = xr.open_zarr(f'{zarr_path}/{mdl}-{downscaling}-{task}-ref')
    hist = xr.open_zarr(f'{zarr_path}/{mdl}-{downscaling}-{task}-hist')
    sim = xr.open_zarr(f'{zarr_path}/{mdl}-{downscaling}-{task}-sim')
    sstd = xr.open_zarr(f'{zarr_path}/{mdl}-{downscaling}-{task}-sstd')
    savg = xr.open_zarr(f'{zarr_path}/{mdl}-{downscaling}-{task}-savg')

    ref = ref['multivariate']
    hist = hist['multivariate']
    sim = sim['multivariate']
    sstd = sstd['multivariate']
    savg = savg['multivariate']

    from xclim import set_options

    # See the advanced notebook for details on how this option work
    with set_options(sdba_extra_output=True):
        out = sdba.adjustment.NpdfTransform.adjust(
            ref,
            hist,
            sim,
            base=sdba.QuantileDeltaMapping,  # Use QDM as the univariate adjustment.
            base_kws={"nquantiles": 100, "group": "time"},
            #base_kws={"nquantiles": 20, "group": "time.month"},
            n_iter=50,  # perform 20 iteration
            #n_escore=1000,  # only send 1000 points to the escore metric (it is realy slow)
            n_escore=50,  # only send 1000 points to the escore metric (it is realy slow)
            #n_escore=-1, #Can't unstandardize AND calc escore
        )

    out = out.persist()
    # _ = dask.distributed.wait(out)
    dask.distributed.wait(out)
    
    scenh = out.scenh.rename(time_hist="time")  # Bias-adjusted historical period
    scens = out.scen  # Bias-adjusted future period
    extra = out.drop_vars(["scenh", "scen"])
    
    # Un-standardize (add the mean and the std back)
    #Impt - can'r calc escores in NPDFTransform
    scenh = sdba.processing.unstandardize(scenh, savg, sstd)
    scens = sdba.processing.unstandardize(scens, savg, sstd)
    #scens = sdba.processing.unstandardize(scens, savg, sstd)
    
    #scenh
    #scens = sdba.processing.unstandardize(scens, savg.T, sstd)
    
    
    # Maybe we don't need this?
    scenh = sdba.processing.reordering(hist, scenh, group="time")
    scens = sdba.processing.reordering(sim, scens, group="time")
    
    scenh = sdba.processing.unstack_variables(scenh)
    scens = sdba.processing.unstack_variables(scens)
    
    scenh = scenh.transpose("time","lat","lon")
    scens = scens.transpose("time","lat","lon")

    outdir = zarr_path
    
    save_attrs_pr = scenh.pr.attrs
    
    ### Remove negative ###
    scenh['pr'] = xr.where(scenh['pr'] < 0, 0, scenh['pr'],keep_attrs=True)
    
    scenh['pr'].attrs = save_attrs_pr
        
    scens = scens.persist()
    scenh = scenh.persist()

    d = {"scens": scens, "scenh": scenh}

    def write_wrapper(k, ds):
        path = f"{zarr_path}/{mdl}-{downscaling}-{task}-{k}"
        with dask.distributed.worker_client() as client:
            ds.to_zarr(path, consolidated = True)
        return k
      
    futures = client.map(
        write_wrapper,
        d.keys(),
        d.values()
    )
    
    f=dask.distributed.wait(futures)



