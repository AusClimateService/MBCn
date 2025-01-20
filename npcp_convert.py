import xarray as xr
import sys
import os

from rechunker import rechunk

import dask
from dask.distributed import Client
from dask_jobqueue import PBSCluster


# output_path = "/g/data/ia39/npcp/data"
output_path = "/scratch/eg3/ag0738/mbcn/npcp_output/"

if __name__ == "__main__":
    
    dask.config.set(**{'array.slicing.split_large_chunks': False,
                       'distributed.comm.timeouts.tcp': '300s',
                       'distributed.scheduler.worker-saturation':1,
                       'distributed.scheduler.work-stealing':False,
                       'array.rechunk.algorithm':'p2p',
                       'jobqueue.pbs.scheduler_options':{'protocol':'tcp://'}})

    client = Client(scheduler_file = os.environ["DASK_PBS_SCHEDULER"])

    write_chunk_time = 300

    
    mdl = sys.argv[1]
    mdl_params = sys.argv[2]
    downscaling = sys.argv[3]
    task = sys.argv[4]

    zarr_path = "/scratch/eg3/ag0738/mbcn/npcp"

    scenh_path = f'{zarr_path}/{mdl}-{downscaling}-{task}-scenh'
    scens_path = f'{zarr_path}/{mdl}-{downscaling}-{task}-scens'
    
    unchunked_scenh = xr.open_zarr(scenh_path)
    unchunked_scens = xr.open_zarr(scens_path)

    target_chunks = {
        "pr": {"time": write_chunk_time, "lat": unchunked_scenh.lat.size, "lon": unchunked_scenh.lon.size},
        "rsds": {"time": write_chunk_time, "lat": unchunked_scenh.lat.size, "lon": unchunked_scenh.lon.size},
        "sfcWind": {"time": write_chunk_time, "lat": unchunked_scenh.lat.size, "lon": unchunked_scenh.lon.size},
        "tasmax": {"time": write_chunk_time, "lat": unchunked_scenh.lat.size, "lon": unchunked_scenh.lon.size},
        "tasmin": {"time": write_chunk_time, "lat": unchunked_scenh.lat.size, "lon": unchunked_scenh.lon.size},
        "time": None,
        "lat": None,
        "lon": None,
    }

    mem_limit = "50GB"
    
    rechunking_scenh = rechunk(unchunked_scenh, target_chunks, mem_limit, scenh_path + "-rechunked", temp_store = scenh_path + "-temp").execute()
    rechunking_scens = rechunk(unchunked_scens, target_chunks, mem_limit, scens_path + "-rechunked", temp_store = scens_path + "-temp").execute()

    _ = dask.distributed.wait((rechunking_scenh, rechunking_scens))

    rechunked_zarr_scenh = xr.open_zarr(scenh_path + "-rechunked")
    rechunked_zarr_scens = xr.open_zarr(scens_path + "-rechunked")

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

    for var in ("pr", "tasmax", "tasmin"):
        # Write historical bias corrected data
        if task == "projection":
            xarrh = rechunked_zarr_scenh[var].chunk({'time': 'auto', 'lat': -1, 'lon': -1})
            path = f'{output_path}/{var}/{mdl}/{downscaling}/mbcn/task-historical/'
            if not os.path.exists(path):
                os.makedirs(path)
            xarrh.to_netcdf(f'{path}/{var}_NPCP-20i_{mdl}_{"evaluation" if mdl == "ECMWF-ERA5" else "ssp370"}_{mdl_params}_{downscaling}_v1_day_{train_yr_start}0101-{train_yr_end}1231_mbcn-AGCD-{train_yr_start}0101-{train_yr_end}1231.nc', 
                           unlimited_dims=['time'],
                           encoding={'time': {'dtype': 'float32'}})
            del xarrh

            if mdl != "ECMWF-ERA5":
                path = f'{output_path}/{var}/{mdl}/{downscaling}/mbcn/task-projection/'
                if not os.path.exists(path):
                    os.makedirs(path)
                xarrs = rechunked_zarr_scens[var].chunk({'time': 'auto', 'lat': -1, 'lon': -1})
                xarrs.to_netcdf(f'{path}/{var}_NPCP-20i_{mdl}_{"evaluation" if mdl == "ECMWF-ERA5" else "ssp370"}_{mdl_params}_{downscaling}_v1_day_{proj_yr_start}0101-{proj_yr_end}1231_mbcn-AGCD-{train_yr_start}0101-{train_yr_end}1231.nc', 
                               unlimited_dims=['time'],
                               encoding={'time': {'dtype': 'float32'}})
                del xarrs

        else:
            path = f'{output_path}/{var}/{mdl}/{downscaling}/mbcn/task-xvalidation/'
            if not os.path.exists(path):
                os.makedirs(path)
            xarrx = rechunked_zarr_scens[var].chunk({'time': 'auto', 'lat': -1, 'lon': -1})
            xarrx.to_netcdf(f'{path}/{var}_NPCP-20i_{mdl}_{"evaluation" if mdl == "ECMWF-ERA5" else "ssp370"}_{mdl_params}_{downscaling}_v1_day_{proj_yr_start}0101-{proj_yr_end}1231_mbcn-AGCD-{train_yr_start}0101-{train_yr_end}1231.nc', 
                               unlimited_dims=['time'],
                               encoding={'time': {'dtype': 'float32'}})

            del xarrx
        

