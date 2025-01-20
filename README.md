# MBCn

Initial version of the codebase containing the code used to execute MBCn for the purposes of [NPCP](https://github.com/AusClimateService/npcp).

Currently in the process of being adapted for more general ACS usage should the need arise.

# Usage

Running the MBCn requires 3 steps: 
1) A chunking and initial correction step (npcp_chunk.py, run using the chunk_job.pbs job script)
2) Main MBCn correction (npcp_calc.py, run using the calc_job.pbs job script)
3) Conversion to NetCDF (npcp_convert.py, run using the convert_job.pbs job script)

Steps 1 and 2 create intermediate Zarr files which should be stored in a temporary location such as a scratch directory.

