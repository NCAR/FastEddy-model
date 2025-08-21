import os, sysAA
import struct
import numpy as np
import numpy.matlib
import xarray as xr
import pandas as pd
import time 
import warnings
import gc
import json
import argparse
from mpi4py import MPI
from datetime import datetime

def field3dTranspose(fld,extents):
    fld=fld.reshape(extents)
    fldFinal=np.transpose(fld,axes=[2,1,0])
    return fldFinal[np.newaxis,Nh:-Nh,Nh:-Nh,Nh:-Nh]

def field2dTranspose(fld,extents):
    fld=fld.reshape(extents)
    fldFinal=np.transpose(fld,axes=[1,0])
    return fldFinal[np.newaxis,Nh:-Nh,Nh:-Nh]

def add_variable_attributes(ds):
    """Add CF-compliant attributes to variables"""

    # Define variable attributes based on common atmospheric variables
    var_attrs = {
        'u': {
            'standard_name': 'eastward_wind',
            'long_name': 'Eastward wind component',
            'units': 'm s-1',
            'description': 'Horizontal wind velocity component in the eastward direction'
        },
        'v': {
            'standard_name': 'northward_wind',
            'long_name': 'Northward wind component', 
            'units': 'm s-1',
            'description': 'Horizontal wind velocity component in the northward direction'
        },
        'w': {
            'standard_name': 'upward_air_velocity',
            'long_name': 'Vertical wind component',
            'units': 'm s-1',
            'description': 'Vertical wind velocity component'
        },
        'theta': {
            'standard_name': 'air_potential_temperature',
            'long_name': 'Potential temperature',
            'units': 'K',
            'description': 'Potential temperature of air'
        },
        'rho': {
            'standard_name': 'air_density',
            'long_name': 'Air density',
            'units': 'kg m-3',
            'description': 'Density of air'
        },
        'pressure': {
            'standard_name': 'air_pressure',
            'long_name': 'Air pressure',
            'units': 'Pa',
            'description': 'Atmospheric pressure'
        },
        'TKE_0': {
            'long_name': 'Turbulent kinetic energy (resolved)',
            'units': 'm2 s-2',
            'description': 'Resolved-scale turbulent kinetic energy'
        },
        'TKE_1': {
            'long_name': 'Turbulent kinetic energy (subgrid)',
            'units': 'm2 s-2', 
            'description': 'Subgrid-scale turbulent kinetic energy'
        },
        'qv': {
            'standard_name': 'specific_humidity',
            'long_name': 'Water vapor mixing ratio',
            'units': 'kg kg-1',
            'description': 'Mass mixing ratio of water vapor in air'
        },
        'ql': {
            'standard_name': 'mass_fraction_of_cloud_liquid_water_in_air',
            'long_name': 'Liquid water mixing ratio',
            'units': 'kg kg-1',
            'description': 'Mass mixing ratio of liquid water in air'
        },
        'qr': {
            'long_name': 'Rain water mixing ratio',
            'units': 'kg kg-1',
            'description': 'Mass mixing ratio of rain water in air'
        },
        'xPos': {
            'standard_name': 'projection_x_coordinate',
            'long_name': 'X position coordinate',
            'units': 'm',
            'description': 'Physical x-coordinate position in meters'
        },
        'yPos': {
            'standard_name': 'projection_y_coordinate', 
            'long_name': 'Y position coordinate',
            'units': 'm',
            'description': 'Physical y-coordinate position in meters'
        },
        'zPos': {
            'standard_name': 'height',
            'long_name': 'Z position coordinate',
            'units': 'm',
            'description': 'Physical z-coordinate position (height) in meters'
        },
        'topoPos': {
            'standard_name': 'surface_altitude',
            'long_name': 'Topographic surface height',
            'units': 'm', 
            'description': 'Height of topographic surface above reference level'
        },
        'AuxScalar_0': {
            'long_name': 'Auxiliary scalar field 0',
            'units': 'dimensionless',
            'description': 'User-defined auxiliary scalar variable (tracer or passive scalar)'
        },
        'AuxScalar_1': {
            'long_name': 'Auxiliary scalar field 1', 
            'units': 'dimensionless',
            'description': 'User-defined auxiliary scalar variable (tracer or passive scalar)'
        },
        'tskin': {
            'standard_name': 'surface_temperature',
            'long_name': 'Skin temperature',
            'units': 'K',
            'description': 'Temperature at the surface skin layer'
        },
        'fricVel': {
            'standard_name': 'friction_velocity',
            'long_name': 'Friction velocity',
            'units': 'm s-1',
            'description': 'Surface friction velocity (u*)'
        },
        'htFlux': {
            'standard_name': 'surface_upward_sensible_heat_flux',
            'long_name': 'Surface sensible heat flux',
            'units': 'W m-2',
            'description': 'Upward sensible heat flux at the surface'
        },
        'invOblen': {
            'long_name': 'Inverse Obukhov length',
            'units': 'm-1', 
            'description': 'Inverse of Obukhov length scale (1/L)'
        },
        'z0m': {
            'standard_name': 'surface_roughness_length_for_momentum_in_air',
            'long_name': 'Momentum roughness length',
            'units': 'm',
            'description': 'Surface roughness length for momentum transfer'
        },
        'z0t': {
            'standard_name': 'surface_roughness_length_for_heat_in_air',
            'long_name': 'Thermal roughness length',
            'units': 'm',
            'description': 'Surface roughness length for heat transfer'
        }
    }

    # Apply attributes to variables
    for var_name, var in ds.data_vars.items():
        if var_name in var_attrs:
            # Apply specific attributes for known variables
            for attr_name, attr_value in var_attrs[var_name].items():
                var.attrs[attr_name] = attr_value
        else:
            # Generic attributes for unknown variables
            var.attrs['long_name'] = var_name.replace('_', ' ').title()
            var.attrs['description'] = f'FastEddy model variable: {var_name}'

            # Infer units based on variable name patterns
            if any(x in var_name.lower() for x in ['temp', 'theta']):
                var.attrs['units'] = 'K'
            elif any(x in var_name.lower() for x in ['vel', 'wind', 'u', 'v', 'w']):
                var.attrs['units'] = 'm s-1'
            elif 'rho' in var_name.lower() or 'density' in var_name.lower():
                var.attrs['units'] = 'kg m-3'
            elif any(x in var_name.lower() for x in ['q', 'mixing', 'humidity']):
                var.attrs['units'] = 'kg kg-1'
            elif 'tke' in var_name.lower() or 'energy' in var_name.lower():
                var.attrs['units'] = 'm2 s-2'
            elif 'pressure' in var_name.lower():
                var.attrs['units'] = 'Pa'
            else:
                var.attrs['units'] = 'unknown'
            
        # Add common attributes to all variables (handle both float and int types)
        if var.dtype.kind == 'f':  # floating point
            var.attrs['_FillValue'] = np.nan
            var.attrs['missing_value'] = np.nan
        elif var.dtype.kind == 'i':  # integer
            var.attrs['_FillValue'] = -9999
            var.attrs['missing_value'] = -9999
        
        # Add data type information
        var.attrs['dtype'] = str(var.dtype)
    
    return ds

def add_coordinate_attributes(ds):
    """Add attributes to coordinate variables"""
    
    if 'time' in ds.coords:
        ds['time'].attrs = {
            'standard_name': 'time',
            'long_name': 'time',
            'axis': 'T'
        }

    coord_attrs = {
        'xIndex': {
            'standard_name': 'projection_x_coordinate',
            'long_name': 'x-coordinate in Cartesian system',
            'axis': 'X',
            'units': 'index'
        },
        'yIndex': {
            'standard_name': 'projection_y_coordinate', 
            'long_name': 'y-coordinate in Cartesian system',
            'axis': 'Y',
            'units': 'index'
        },
        'zIndex': {
            'standard_name': 'atmosphere_hybrid_height_coordinate',
            'long_name': 'z-coordinate in Cartesian system',
            'axis': 'Z',
            'positive': 'up',
            'units': 'index'
        }
    }

    for coord_name, attrs in coord_attrs.items():
        if coord_name in ds.coords:
            ds[coord_name].attrs.update(attrs)
    
    return ds

def readBinary(outpath,theseFiles):
    verboseLogging=False
    print(theseFiles)
    dsSet=[]
    for thatFile in theseFiles:
       print(thatFile)
       ds_fe=xr.Dataset()
       try:
          thisFile='{:s}{:s}'.format(outpath,thatFile)
          flength = os.stat(thisFile).st_size
          with open(thisFile, mode='rb') as f:
               while(f.tell() < flength):
                 nameLen=struct.unpack("i", f.read(4))
                 if verboseLogging:
                     print(len(nameLen),nameLen[0])
                 fieldName=f.read(nameLen[0]).rstrip(b'\x00').decode()
                 if verboseLogging:
                     print(fieldName)
                 typeLen=struct.unpack("i", f.read(4))
                 if verboseLogging:
                     print(len(typeLen),typeLen[0])
                 fieldType=f.read(typeLen[0]).rstrip(b'\x00').decode()
                 nDims=struct.unpack("i", f.read(4))
                 if verboseLogging:
                     print(nDims)
                 extents=np.array([],dtype=np.int32)
                 fmtStr='{:d}i'.format(nDims[0])
                 extents=np.asarray(struct.unpack(fmtStr,f.read(nDims[0]*4)),dtype=np.int32)
                 if verboseLogging:
                     print(extents)
                 if fieldType == 'float':
                    fmtStr='{:d}f'.format(np.prod(extents))
                    fld=np.asarray(struct.unpack(fmtStr,f.read(np.prod(extents)*4)),dtype=np.float32)
                 elif fieldType == 'int':
                    fmtStr='{:d}i'.format(np.prod(extents))
                    fld=np.asarray(struct.unpack(fmtStr,f.read(np.prod(extents)*4)),dtype=np.int32)
                 if(len(extents)==3):
                     fld=fld.reshape(extents)
                     fldFinal=field3dTranspose(fld,extents)
                     if verboseLogging:
                         print(fldFinal.shape)
                     ds_fe[fieldName]=xr.DataArray(fldFinal,dims=['time','zIndex','yIndex','xIndex'])
                 elif(len(extents)==2):
                     fldFinal=field2dTranspose(fld,extents)
                     if verboseLogging:
                         print(fldFinal.shape)
                     ds_fe[fieldName]=xr.DataArray(fldFinal,dims=['time','yIndex','xIndex'])
                 elif(len(extents)==1):
                     fldFinal=fld
                     ds_fe[fieldName]=xr.DataArray(fldFinal,dims=['time'])
       except IOError:
         print('Error While Opening the file: {:s}'.format(thisFile))
       dsSet.append(ds_fe)
      
    #Concatenate all the perRank dataSets into a single dataset 
    dsFull=xr.concat(dsSet,'xIndex',data_vars='minimal')

    # Add variable and coordinate attributes
    dsFull = add_variable_attributes(dsFull)
    dsFull = add_coordinate_attributes(dsFull)
        
    return dsFull

###

def parse_args():
    """ parse the command line arguments """

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", required=True, help="JSON file with converter parameter settings")
    args = parser.parse_args()
    return args

################## main()
print("Hello performing first MPI calls.")

mpi_size = MPI.COMM_WORLD.Get_size()
mpi_rank = MPI.COMM_WORLD.Get_rank()
mpi_name = MPI.Get_processor_name()

########################################
### Parse the command line arguments ###
########################################
args = parse_args()
#########################################################
### Read the json file of converter script parameters ###
#########################################################
with open(args.file) as file:
  params = json.loads(file.read())

outpath = params["outpath"]
FEoutBase = params["FEoutBase"]
numOutRanks = params["numOutRanks"] 
fileSetSize = params["fileSetSize"]
tstart = params["tstart"]
tstep = params["tstep"]
netCDFpath = params["netCDFpath"]
removeBinaries = params["removeBinaries"]

if mpi_size <= fileSetSize:
  fileBatchsize = np.int32(fileSetSize/mpi_size)
else:
  print('mpi_size of {:d} is > fileSetSize = {:d}. Please ensure mpi_size <= fileSetSize.*'.format(mpi_size,fileSetSize))
  exit()
tstop=tstart+tstep*fileSetSize   #tstart+(mpi_size*fileBatchsize+1)*tstep
print("{:d}/{:d}: Hello World! on {:s}.".format(mpi_rank, mpi_size, mpi_name))
print('Converting binary FE outputs {:s}{:s}_rank_{:d}-{:d}.*'.format(outpath,FEoutBase,0,numOutRanks))
print('In batches of {:d} files per rank beginning from timestep {:d} to timestep {:d} every {:d} timesteps.'.format(fileBatchsize,tstart,tstop,tstep))
print('Writing full netCDF files to {:s}/{:s}.*'.format(netCDFpath,FEoutBase))

Nh=3

if(mpi_rank == 0):
  if not os.path.exists(netCDFpath):
    os.makedirs(netCDFpath)

ts_list=np.arange(tstart,tstop+1,tstep,dtype=np.int32)

#setup mpi task decomposition over the set of output file timesteps
list_len = len(ts_list)
elems_perRank = np.int64(np.floor(list_len/mpi_size))
extra_elems = np.int64(list_len % elems_perRank)
if mpi_rank == 0:
       print("{:d}/{:d}: len(ts_list) = {:d}".format(mpi_rank, mpi_size,list_len))
       print("{:d}/{:d}: elems_perRank = {:d}".format(mpi_rank, mpi_size,elems_perRank))
       print("{:d}/{:d}: extra_elems = {:d}".format(mpi_rank, mpi_size,extra_elems))
for iRank in range(mpi_size):
    if mpi_rank == iRank:
       mystart = (iRank)*elems_perRank
       myend = (iRank+1)*elems_perRank
       if iRank is (mpi_size-1):
          myend = myend+(list_len-mpi_size*elems_perRank) ###Catch straggler files with the last rank
       mytslist = ts_list[mystart:myend]
       print("{:d}/{:d}: mytslist = ts_list({:d}:{:d})".format(mpi_rank, mpi_size, mystart, myend))
       print("{:d}/{:d}: Converting from {:s}.{:d} to {:s}.{:d}".format(mpi_rank, mpi_size, FEoutBase, mytslist[0], FEoutBase, mytslist[-1]))
       print("{:d}/{:d}: len(myfileslist) = {:d}".format(mpi_rank, mpi_size,len(mytslist)))
    MPI.COMM_WORLD.Barrier()

#Each rank can now loop over a subset of the timesteps to concatenate and create a single netCDF file per timestep
for timeStep in mytslist:
   theseFiles=[]
   for outRank in range(numOutRanks):
       theseFiles.append('{:s}_rank_{:d}.{:d}'.format(FEoutBase,outRank,timeStep))
   parseProceed=False
   goodCnt=0
   for thatFile in theseFiles:
       #print('Checking {:s} '.format(thatFile))
       thisFile='{:s}{:s}'.format(outpath,thatFile)
       if os.path.exists(thisFile):
           goodCnt+=1
   #print(goodCnt)
   if(goodCnt==numOutRanks):
       parseProceed=True
   else:
       print('{:d} specified binary files are missing. Skipping timestep: {:d}...'.format(numOutRanks-goodCnt,timeStep))
   if parseProceed:
     dsFull=readBinary(outpath,theseFiles)
     #write the full  domain datatset to netcdf file
     if False:
        dsFull.to_netcdf('{:s}NETCDF/{:s}.{:d}'.format(outpath,FEoutBase,timeStep),format='NETCDF4')
     else:
        dsFull.to_netcdf('{:s}/{:s}.{:d}'.format(netCDFpath,FEoutBase,timeStep),format='NETCDF4')
     del dsFull
     if os.path.exists('{:s}/{:s}.{:d}'.format(netCDFpath,FEoutBase,timeStep)):
       if removeBinaries:
         for thatFile in theseFiles:
           thisFile='{:s}{:s}'.format(outpath,thatFile)
           os.remove(thisFile)
   gc.collect()

MPI.COMM_WORLD.Barrier()
print("{:d}/{:d}: Conversions complete on {:s}.".format(mpi_rank, mpi_size, mpi_name))
print("{:d}/{:d}: Goodbye World! on {:s}.".format(mpi_rank, mpi_size, mpi_name))
MPI.Finalize()
