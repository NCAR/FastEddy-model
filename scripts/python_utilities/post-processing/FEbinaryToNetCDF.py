import os, sys
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
import re

### Load lookup tables from JSON file ###

def load_field_attributes(json_file_path):
    """
    Load field attribute lookup tables from a JSON file.
    
    Args:
        json_file_path (str): Path to the JSON file containing field attributes
    
    Returns:
        tuple: (base_attrs, jacobian_attrs, coordinate_attrs, directions)
    """

    try:
        with open(json_file_path, 'r') as f:
            attrs_data = json.load(f)
        
        # Convert lists back to tuples for consistency with original code
        base_attrs = {k: tuple(v) for k, v in attrs_data['base_attrs'].items()}
        jacobian_attrs = {k: tuple(v) for k, v in attrs_data['jacobian_attrs'].items()}
        coordinate_attrs = {k: tuple(v) for k, v in attrs_data['coordinate_attrs'].items()}
        
        # Convert string keys back to integers for directions
        directions = {int(k): v for k, v in attrs_data['directions'].items()}
        
        # Load special field mappings
        base_state_indices = {int(k): tuple(v) for k, v in attrs_data['special_field_mappings']['base_state_indices'].items()}

        return base_attrs, jacobian_attrs, coordinate_attrs, directions, base_state_indices
        
    except FileNotFoundError:
        print(f"Warning: Field attributes file '{json_file_path}' not found. Using empty lookup tables.")
        return {}, {}, {}, {}, {}
    except Exception as e:
        print(f"Error loading field attributes from '{json_file_path}': {e}")
        return {}, {}, {}, {}, {}

def field3dTranspose(fld,extents):
    fld=fld.reshape(extents)
    fldFinal=np.transpose(fld,axes=[2,1,0])
    del fld
    return fldFinal[np.newaxis,Nh:-Nh,Nh:-Nh,Nh:-Nh]

def field2dTranspose(fld,extents):
    fld=fld.reshape(extents)
    fldFinal=np.transpose(fld,axes=[1,0])
    del fld
    return fldFinal[np.newaxis,Nh:-Nh,Nh:-Nh]

def get_variable_attrs(var_name, base_attrs, jacobian_attrs, coordinate_attrs, directions, base_state_indices):
    """
    Get CF-compliant attributes for a variable name, handling special cases.   
    Args: 
        var_name (str): Variable name to get attributes for
        base_attrs (dict): Base field attributes lookup table
        jacobian_attrs (dict): Jacobian field attributes lookup table
        coordinate_attrs (dict): Coordinate field attributes lookup table
        directions (dict): Direction index to name mapping
        base_state_indices (dict): Base state field index mappings
    Returns: 
        tuple or None: (units, long_name, standard_name) or None if no match
    """

    # Handle BS_ fields with numeric identifiers
    if var_name.startswith('BS_'):
        try:
            field_index = int(var_name[3:])
            if field_index in base_state_indices:
                return base_state_indices[field_index]
            else:
                return ('1', 'Base state field', None)
        except ValueError:
            pass

    # Handle TauQv/TauQl moisture flux fields
    tau_moisture_match = re.match(r'^TauQ([vl])(\d+)$', var_name)
    if tau_moisture_match:
        species, direction_idx = tau_moisture_match.groups()
        direction_idx = int(direction_idx)
        direction_name = directions.get(direction_idx, str(direction_idx))
        
        if species == 'v':  # TauQv (water vapor)
            long_name = f'Subgrid-{direction_name} water vapor flux in {direction_name} direction'
        elif species == 'l':  # TauQl (liquid water)
            long_name = f'Subgrid-{direction_name} liquid water flux in {direction_name} direction'
        else:
            long_name = f'Subgrid-{direction_name} moisture flux in {direction_name} direction'
        
        return ('kg kg-1 m s-1', long_name, None)

    # Handle numbered versions of base fields (e.g., AuxScalar_0, etc.)
    base_name_match = re.match(r'^([A-Za-z_]+?)_?(\d+)$', var_name)
    if base_name_match:
        base_name = base_name_match.group(1)
        if base_name in base_attrs:
            return base_attrs[base_name]

    # Check specific attribute dictionaries
    for attr_dict in [jacobian_attrs, coordinate_attrs, base_attrs]:
        if var_name in attr_dict:
            return attr_dict[var_name]

    return None

def add_variable_attributes(ds, base_attrs, jacobian_attrs, coordinate_attrs, directions, base_state_indices):
    """Add attributes to variables"""

    for var_name, var in ds.data_vars.items():
        attrs_tuple = get_variable_attrs(var_name, base_attrs, jacobian_attrs, coordinate_attrs, directions, base_state_indices)
        
        if attrs_tuple:
            units, long_name, standard_name = attrs_tuple
            var.attrs['units'] = units
            var.attrs['long_name'] = long_name
            if standard_name is not None:
                var.attrs['standard_name'] = standard_name
                
    return ds
    
def add_coordinate_attributes(ds):
    """Add coordinate variables and their attributes"""

    # Create explicit coordinate variables based on dimension sizes
    coords_to_add = {}

    if 'xIndex' in ds.dims:
        coords_to_add['xIndex'] = np.arange(ds.sizes['xIndex'], dtype=np.int32)

    if 'yIndex' in ds.dims:
        coords_to_add['yIndex'] = np.arange(ds.sizes['yIndex'], dtype=np.int32)

    if 'zIndex' in ds.dims:
        coords_to_add['zIndex'] = np.arange(ds.sizes['zIndex'], dtype=np.int32)

    # Add the coordinate variables to the dataset
    if coords_to_add:
        ds = ds.assign_coords(coords_to_add)

    if 'time' in ds.coords:
        ds['time'].attrs = {
            'units': 's',
            'long_name': 'Simulation time',
            'standard_name': 'time',
            'axis': 'T'
        }

    coord_attrs = {
        'xIndex': {
            'long_name': 'x-coordinate index',
            'units': '1',
            'axis': 'X'
        },
        'yIndex': {
            'long_name': 'y-coordinate index',
            'units': '1',
            'axis': 'Y'
        },
        'zIndex': {
            'long_name': 'z-coordinate index',
            'units': '1',
            'axis': 'Z',
            'positive': 'up'
        }
    }

    for coord_name, attrs in coord_attrs.items():
        if coord_name in ds.coords:
            ds[coord_name].attrs.update(attrs)

    return ds

def reorder_dataset_variables(ds):
    """
    Reorder dataset variables to desired order:
    zIndex, yIndex, xIndex, xPos, yPos, zPos, then other variables, with time last
    """

    # Define the desired order for the first variables
    priority_order = ['zIndex', 'yIndex', 'xIndex', 'xPos', 'yPos', 'zPos']

    # Get all variable names from both data_vars and coords, preserving original order
    original_data_vars = list(ds.data_vars.keys())
    original_coords = list(ds.coords.keys())

    # Build the new order
    new_order = []
    used_vars = set()

    # Add priority variables first (if they exist)
    for var_name in priority_order:
        if var_name in ds.data_vars or var_name in ds.coords:
            new_order.append(var_name)
            used_vars.add(var_name)

    # Add remaining data variables in their original order (except time)
    for var_name in original_data_vars:
        if var_name not in used_vars and var_name != 'time':
            new_order.append(var_name)
            used_vars.add(var_name)

    # Add remaining coordinate variables in their original order (except time)
    for var_name in original_coords:
        if var_name not in used_vars and var_name != 'time':
            new_order.append(var_name)
            used_vars.add(var_name)

    # Add time last if it exists
    if 'time' in ds.data_vars or 'time' in ds.coords:
        new_order.append('time')

    # Use xarray's reindex to reorder variables
    # This preserves the distinction between coords and data_vars
    return ds[new_order]

def readBinary(numOutRanks,outpath,theseFiles):
    
    if 'fulldata_dict' in locals():
        del fulldata_dict
    if 'subdata_dict' in locals():
        del subdata_dict
    gc.collect()

    verboseLogging=False #True 
    print(theseFiles)
    fulldata_dict = {}

    for thatFile in theseFiles:
       subdata_dict = {}
       print(thatFile)
       try:
          thisFile='{:s}{:s}'.format(outpath,thatFile)
          flength = os.stat(thisFile).st_size
          with open(thisFile, mode='rb') as f:
               while(f.tell() < flength): #while the filepointer is not at the end of the binary file
                 ## Read and parse the binary representation of 
                 ## the current variable "name" (a string of integer length)
                 nameLen=struct.unpack("i", f.read(4))
                 if verboseLogging:
                     print(f"len(nameLen) = {len(nameLen)}, nameLen[0] = {nameLen[0]}")
                 fieldName=f.read(nameLen[0]).rstrip(b'\x00').decode()
                 if verboseLogging:
                     print(f"fieldName = {fieldName}")
                 ## Read and parse the binary representation of 
                 ## the current variable "type" (a string of integer length)
                 typeLen=struct.unpack("i", f.read(4))
                 if verboseLogging:
                     print(f"len(typeLen) = {len(typeLen)}, typeLen[0] = {typeLen[0]}")
                 fieldType=f.read(typeLen[0]).rstrip(b'\x00').decode()
                 if verboseLogging:
                     print(f"fieldType = {fieldType}")
                 ## Read and parse the binary representation of 
                 ## the current variable "number-of-dimensions" (integer) 
                 nDims=struct.unpack("i", f.read(4))
                 if verboseLogging:
                     print(f"nDims = {nDims}")
                 ## Read and parse the binary representation of 
                 ## the current variable "dimension extents" (1-d integer array) 
                 extents=np.array([],dtype=np.int32)
                 fmtStr='{:d}i'.format(nDims[0])
                 extents=np.asarray(struct.unpack(fmtStr,f.read(nDims[0]*4)),dtype=np.int32)
                 if verboseLogging:
                     print(f"extents = {extents}")
                 ## Read, parse, reshape, transpose the binary field based on the type and extents  
                 if fieldType == 'float':
                     fmtStr='{:d}f'.format(np.prod(extents))
                     if(len(extents)==3):
                         fld3dfloat=np.frombuffer(f.read(np.prod(extents)*4),dtype=np.float32).reshape(extents)
                         fldFinal=field3dTranspose(fld3dfloat,extents)
                         if verboseLogging:
                             print(fldFinal.shape)
                     elif(len(extents)==2):
                         fld2dfloat=np.frombuffer(f.read(np.prod(extents)*4),dtype=np.float32).reshape(extents)
                         fldFinal=field2dTranspose(fld2dfloat,extents)
                         if verboseLogging:
                             print(fldFinal.shape)
                     elif(len(extents)==1):
                         fld1dfloat=np.frombuffer(f.read(np.prod(extents)*4),dtype=np.float32)
                         if("GAD" in fieldName):
                            fldFinal=fld1dfloat[np.newaxis,:]
                         else:
                            fldFinal=fld1dfloat
                         if verboseLogging:
                             print(fldFinal.shape)
                 elif fieldType == 'int':
                     fmtStr='{:d}i'.format(np.prod(extents))
                     if(len(extents)==1):
                         fld1dint=np.frombuffer(f.read(np.prod(extents)*4),dtype=np.int32)
                         if("GAD" in fieldName):
                            fldFinal=fld1dint[np.newaxis,:]
                         else:
                            fldFinal=fld1dint
                         if verboseLogging:
                             print(fldFinal.shape)
                 ### Add the named field to a dictionary as a key-value pair 
                 subdata_dict[fieldName]=fldFinal
       except IOError:
         print('Error While Opening the file: {:s}'.format(thisFile))
       finally:
        if f:  # Check if f was successfully assigned a file object
            f.close()
            # The file is closed here
       ### If this is the first time through allocate all the full data arrays 
       ### necessary for a full data dictionary
       if len(fulldata_dict) == 0:
           rank_cnt=0
           for key in subdata_dict.keys():
               subextents = subdata_dict[key].shape
               fullextents = subextents
               if len(subextents) > 1:  ## Note tuples are immutable so using tuple(list(blah_tuple)) as workaround 
                                        ## to compute xIndex value in fullextents 
                   list_extents = list(subextents)
                   list_extents[-1] = list_extents[-1]*numOutRanks
                   fullextents = tuple(list_extents)
               if verboseLogging:
                   print(f"rank_cnt = {rank_cnt}, {key}: subextents = {subextents}, fullextents={fullextents}")
               subtype = subdata_dict[key].dtype
               fulldata_dict[key] = np.zeros(fullextents,dtype=subtype)
               if verboseLogging:
                    print(f"rank_cnt = {rank_cnt}, {key}:  fulldata_dict[key].shape= {fulldata_dict[key].shape}, fullextents={fullextents}")
               ## NOTE for now this assumes concatenation is always on the xIndex 
               ## (just like the original converter script)!!!
               if len(subextents) == 1:    #[time]  #[GADNumTurbines]
                      fulldata_dict[key] = np.copy(subdata_dict[key])
               elif len(subextents) == 2:    #[time, GADNumTurbines]
                      fulldata_dict[key] = np.copy(subdata_dict[key])
               elif len(subextents) == 3:  #[time, yIndex, xIndex]
                   fulldata_dict[key][:,:,rank_cnt*subextents[-1]:(rank_cnt+1)*subextents[-1]] = np.copy(subdata_dict[key])
               elif len(subextents) == 4:  #[time, zIndex, yIndex, xIndex]
                   if verboseLogging:
                       print(f"rank_cnt = {rank_cnt}, {key}:  slice.shape= {fulldata_dict[key][:,:,:,rank_cnt*subextents[-1]:(rank_cnt+1)*subextents[-1]].shape}, fullextents={fullextents}")
                   fulldata_dict[key][:,:,:,rank_cnt*subextents[-1]:(rank_cnt+1)*subextents[-1]] = np.copy(subdata_dict[key])
           rank_cnt += 1
       else:  ## This is the second or later file in the per-rank sequence so just fill fullData array segments
           for key in subdata_dict.keys():
               subextents = subdata_dict[key].shape
               fullextents = fulldata_dict[key].shape
               if verboseLogging:
                   print(f"rank_cnt = {rank_cnt}, {key}: subextents = {subextents}, fullextents={fullextents}")
               subtype = subdata_dict[key].dtype
               if len(subextents) == 1: 
                   if verboseLogging:
                      print(f"rank_cnt = {rank_cnt}, skipping {key} since no spatial dimensions involved...")
               elif len(subextents) == 2:
                   ### perform a reduction (by sum) across rank files 
                   fulldata_dict[key] = fulldata_dict[key]+np.copy(subdata_dict[key])
               elif len(subextents) == 3:
                   fulldata_dict[key][:,:,rank_cnt*subextents[-1]:(rank_cnt+1)*subextents[-1]] = np.copy(subdata_dict[key])
               elif len(subextents) == 4:
                   fulldata_dict[key][:,:,:,rank_cnt*subextents[-1]:(rank_cnt+1)*subextents[-1]] = np.copy(subdata_dict[key])
           rank_cnt += 1
              
    #Create the full-domain single xarray dataset
    dsFull=xr.Dataset()
    for key in fulldata_dict.keys():
        if verboseLogging:
             print(f"rank_cnt = {rank_cnt}, creating dataset field {key} with shape {fulldata_dict[key].shape}...")
        if len(fulldata_dict[key].shape) == 4:
            dsFull[key]=xr.DataArray(fulldata_dict[key],dims=['time','zIndex','yIndex','xIndex'])
        if len(fulldata_dict[key].shape) == 3:
            dsFull[key]=xr.DataArray(fulldata_dict[key],dims=['time','yIndex','xIndex'])
        if len(fulldata_dict[key].shape) == 2:
            dsFull[key]=xr.DataArray(fulldata_dict[key],dims=['time','GADNumTurbines'])
        if len(fulldata_dict[key].shape) == 1:
            dsFull[key]=xr.DataArray(fulldata_dict[key],dims=['time'])
   
    ## Clean up memory 
    del subdata_dict
    del fulldata_dict
    del fldFinal 
    gc.collect()

    return dsFull

###

def parse_args():
    """ parse the command line arguments """

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", required=True, help="JSON file with converter parameter settings")
    parser.add_argument("-a", "--attrs", required=True, help="JSON file with field attribute definitions")
    args = parser.parse_args()
    return args

################## main() ################################################################################
print("Hello performing first MPI calls.")

mpi_size = MPI.COMM_WORLD.Get_size()
mpi_rank = MPI.COMM_WORLD.Get_rank()
mpi_name = MPI.Get_processor_name()

########################################
### Parse the command line arguments ###
########################################
args = parse_args()

#########################################################
### Load field attributes from JSON file ###
#########################################################
base_attrs, jacobian_attrs, coordinate_attrs, directions, base_state_indices = load_field_attributes(args.attrs)

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

if mpi_size <= fileSetSize+1:
  fileBatchsize = np.int32(fileSetSize/mpi_size)
else:
  print('mpi_size of {:d} is > fileSetSize = {:d}. Please ensure mpi_size <= fileSetSize.*'.format(mpi_size,fileSetSize))
  exit()
tstop=tstart+tstep*fileSetSize   
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
     dsFull=readBinary(numOutRanks,outpath,theseFiles)

     # Add variable and coordinate attributes
     dsFull = add_coordinate_attributes(dsFull)
     dsFull = add_variable_attributes(dsFull, base_attrs, jacobian_attrs, coordinate_attrs, directions, base_state_indices)

     # Reorder variables to desired order
     dsFull = reorder_dataset_variables(dsFull)

     # Create encoding to prevent _FillValue for all variables AND coordinates
     encoding = {var: {'_FillValue': None} for var in list(dsFull.data_vars) + list(dsFull.coords)}

     #write the full domain datatset to netcdf file
     dsFull.to_netcdf('{:s}/{:s}.{:d}'.format(netCDFpath,FEoutBase,timeStep),format='NETCDF4',encoding=encoding)
     
     del dsFull
     gc.collect()
     if os.path.exists('{:s}/{:s}.{:d}'.format(netCDFpath,FEoutBase,timeStep)):
       if removeBinaries:
         for thatFile in theseFiles:
           thisFile='{:s}{:s}'.format(outpath,thatFile)
           os.remove(thisFile)

MPI.COMM_WORLD.Barrier()
print("{:d}/{:d}: Conversions complete on {:s}.".format(mpi_rank, mpi_size, mpi_name))
print("{:d}/{:d}: Goodbye World! on {:s}.".format(mpi_rank, mpi_size, mpi_name))
MPI.Finalize()
