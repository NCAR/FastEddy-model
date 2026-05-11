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
from pathlib import Path
def parse_args():
    """ parse the command line arguments """

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", required=True, help="JSON file with coupler parameter settings")
    args = parser.parse_args()
    return args

def get_params_FE(FE_params_file):

    FE_params_dict = {}
    n_header = 1
    f = open(FE_params_file,'r')
    data = f.readlines()
    f.close
    row_len = len(data) - n_header
    col_len = len(data[n_header].split())
    for rr in range(n_header,row_len+n_header):
        row_rr = data[rr]
        if (row_rr[0]=='#'):
            continue
        varname_rr = row_rr.split('=')[0].split()
        varval_rr = row_rr.split('=')[1].split('#')[0].split()
        FE_params_dict[varname_rr[0]] = varval_rr

    return FE_params_dict

################## main() ################################################################################

########################################
### Parse the command line arguments ###
########################################
args = parse_args()

#########################################################
### Read the json file of converter script parameters ###
#########################################################
with open(args.file) as file:
  params = json.loads(file.read())

runPath = params["runPath"]
FEparamsFile = params["FEparamsFile"]
outputFileName = params["outputFileName"]
startStep = params["startStep"]
endStep = params["endStep"]

#Open and parse the FEparamsFile if it exists, else eit
if(Path(f"{runPath}/{FEparamsFile}").exists()):
    FE_params = get_params_FE(f"{runPath}/{FEparamsFile}")
elif(Path(f"{FEparamsFile}").exists()):
    FE_params = get_params_FE(f"{FEparamsFile}")
else:
    sys.exit(f"ERROR: Could not find either {FEparamsFile} or {runPath}/{FEparamsFile}.\nExiting Now!") # Exit with error message

#Gather configuration parameters from the FEtowerSpecsFile and FEparamsFile
batchSteps=int(FE_params['NtBatch'][0])
Nz=int(FE_params['Nz'][0])
towerPath=f"{runPath}/{FE_params['towerPath'][0]}"

FEtowerSpecsFile = FE_params["towerSpecsFile"][0]
#Open the FEtowerSpecsFile if it exists, else exit
if(Path(f"{runPath}/{FEtowerSpecsFile}").exists()):
    ds_towSpecs = xr.open_dataset(f"{runPath}/{FEtowerSpecsFile}")
elif(Path(f"{FEtowerSpecsFile}").exists()):
    ds_towSpecs = xr.open_dataset(f"{FEtowerSpecsFile}")
else:
    sys.exit(f"ERROR: Could not find either {FEtowerSpecsFile} or {runPath}/{FEtowerSpecsFile}.\nExiting Now!") # Exit with error message
#Read the towerSpecs to determine the number of tower in the run
nTowers = ds_towSpecs.sizes['nProfs']

#Build the list of towerFlds and towerSurfFlds contained in the raw binary files by parsing the controlling switches in the FEparamsFile
nTowerVars = 5  #The minimum number of tower variable profiles that should have been written
towerFldNames = ['rho', 'u', 'v', 'w', 'theta']
nSurfVars = 6  #The minimum number of surf variables that should have been written
towerSurfFldNames = ['z0m', 'z0t', 'tskin', 'fricVel', 'invObLen', 'htFlux']
nTKE=int(FE_params['TKESelector'][0])*int(FE_params['turbulenceSelector'][0])
for iTKE in range(nTKE):
    nTowerVars += 1
    towerFldNames.append(f"TKE_{iTKE}")
nmoist=int(FE_params['moistureNvars'][0])*int(FE_params['moistureSelector'][0])
for imoist in range(nmoist):
    nTowerVars += 1
    if imoist == 0:
        towerFldNames.append("qv")
        nSurfVars = nSurfVars + 2
        towerSurfFldNames.extend(['qskin', 'qFlux'])
    else:
        towerFldNames.append("ql")
if "NhydroAuxScalars" in FE_params:
    NhydroAuxScalars = int(FE_params['NhydroAuxScalars'][0])
    for iAuxSc in range(NhydroAuxScalars):
        nTowerVars += 1
        towerFldNames.append(f"AuxScalar_{iAuxSc}")
if(int(FE_params['hydroSubGridWrite'][0]) > 0):
    nTaus = 9
    nTowerVars = nTowerVars + nTaus
    towerFldNames.extend(['Tau11', 'Tau21', 'Tau31', 'Tau32', 'Tau22', 'Tau33', 'TauTH1', 'TauTH2', 'TauTH3'])
    for imoist in range(nmoist):
        nTowerVars = nTowerVars+3
        if imoist == 0:
            towerFldNames.extend(['TauQv1','TauQv2','TauQv3'])
        else:
            towerFldNames.extend(['TauQl1','TauQl2','TauQl3'])

#Determine the number of batches the user requested to consolidate into a NetCDF file 
nBatches = np.int32((endStep-startStep)/batchSteps)

#Summarize the intended consolidation parameters
print(f"Attempting to read raw tower files from {towerPath}.")
print(f"Consolidating {nBatches} of {batchSteps} timestep-instances into a single timeseries.")
print(f"Expecting nTowerVars = {nTowerVars}, nSurfVars = {nSurfVars}")

#Preallocate numpy arrays for all the binary data
towerZ = np.zeros((nTowers,Nz),dtype=np.float32)
towerY = np.zeros((nTowers),dtype=np.float32)
towerX = np.zeros((nTowers),dtype=np.float32)
towerElev = np.zeros((nTowers),dtype=np.float32)
towerSeaMask = np.zeros((nTowers),dtype=np.float32)
towerYoffset = np.zeros((nTowers),dtype=np.float64)
towerXoffset = np.zeros((nTowers),dtype=np.float64)
towerTimes = np.zeros((nBatches*batchSteps+1),dtype=np.float32)
towerData = np.zeros((nTowers,nBatches*batchSteps+1,nTowerVars,Nz),dtype=np.float32)
towerSurfData = np.zeros((nTowers,nBatches*batchSteps+1,nSurfVars),dtype=np.float32)

###########################################################################
### Read the tower_ic_*.0 files  (Initial/static conditions)
###########################################################################
iStep=0
for itower in range(0,nTowers):
    thisFile=f"{towerPath}/tower_ic_{itower}.{iStep}"
    flength = os.stat(thisFile).st_size
    try:
        with open(thisFile, mode='rb') as f:
            while(f.tell() < flength): #while the filepointer is not at the end of the binary file
                #----- Tower static data
                towerNz=struct.unpack("i", f.read(4))[0]
                towerZ[itower,:]=np.frombuffer(f.read(towerNz*4),dtype=np.float32)
                towerY[itower]=np.frombuffer(f.read(4),dtype=np.float32)[0]
                towerX[itower]=np.frombuffer(f.read(4),dtype=np.float32)[0]
                towerElev[itower]=np.frombuffer(f.read(4),dtype=np.float32)[0]
                if "surflayer_offshore" not in FE_params:
                  towerSeaMask[itower]=np.frombuffer(f.read(4),dtype=np.float32)[0]
                else:
                    if(int(FE_params['surflayer_offshore'][0]) > 0):
                        towerSeaMask[itower]=np.frombuffer(f.read(4),dtype=np.float32)[0]
                towerYoffset[itower]=np.frombuffer(f.read(8),dtype=np.float64)[0]
                towerXoffset[itower]=np.frombuffer(f.read(8),dtype=np.float64)[0]

                #----- Tower profile data
                ## Read and parse the number elements per tower instance (single timestep)
                towerInstanceSize=struct.unpack("i", f.read(4))[0]
                ## Read and parse the number instances in this file (batch of timesteps)
                batchSize=struct.unpack("i", f.read(4))[0]
                ## Read the full set of batch tower time values in this file
                if itower == 0:
                   towerTimes[iStep]=np.frombuffer(f.read(4),dtype=np.float32)[0]
                else:
                   towerTmpTimes=np.frombuffer(f.read(4),dtype=np.float32)
                ## Read the full set of batch tower instances in this file
                towerData[itower,iStep,:,:]=np.frombuffer(f.read(towerInstanceSize*4),dtype=np.float32).reshape((nTowerVars,Nz))

                #---- Tower surf data
                ## Read and parse the number elements per tower instance (single timestep)
                towerSurfInstanceSize=struct.unpack("i", f.read(4))[0]
                ## Read and parse the number instances in this file (batch of timesteps)
                batchSize=struct.unpack("i", f.read(4))[0]
                ## Read the full set of batch tower time values in this file
                towerSurfTimes=np.frombuffer(f.read(batchSize*4),dtype=np.float32)
                ## Read the full set of batch tower instances in this file
                towerSurfData[itower,iStep,:]=np.frombuffer(f.read(towerSurfInstanceSize*4),dtype=np.float32).reshape((towerSurfInstanceSize))

    except IOError:
         print(f"Error While Opening the file: {thisFile}")

    finally:
        if f:  # Check if f was successfully assigned a file object
            f.close()
            # The file is closed here

###########################################################################
### Reads all of the subsequent batches of timesteps
###########################################################################
if(startStep == 0):
    iStep = (startStep+1)
else:
    iStep = (startStep)
while iStep < (endStep+1)-batchSteps+1:
  print(f"Reading at iStep = {iStep}...")
  for itower in range(nTowers):
    thisFile=f"{towerPath}/tower_{itower}.{(iStep-1)}"
    thisSurfFile=f"{towerPath}/tower_sv_{itower}.{(iStep-1)}"
    flength = os.stat(thisFile).st_size
    try:
        with open(thisFile, mode='rb') as f:
            while(f.tell() < flength): #while the filepointer is not at the end of the binary file
                ## Read and parse the number elements per tower instance (single timestep)
                towerInstanceSize=struct.unpack("i", f.read(4))[0]
                ## Read and parse the number instances in this file (batch of timesteps)
                batchSize=struct.unpack("i", f.read(4))[0]
                ## Read the full set of batch tower time values in this file
                if itower == 0:
                   towerTimes[iStep:(iStep+batchSteps)]=np.frombuffer(f.read(batchSize*4),dtype=np.float32)
                else:
                   towerTmpTimes=np.frombuffer(f.read(batchSize*4),dtype=np.float32)
                ## Read the full set of batch tower instances in this file
                towerData[itower,iStep:(iStep+batchSteps),:,:]=np.frombuffer(f.read(batchSize*towerInstanceSize*4),dtype=np.float32).reshape((batchSize,nTowerVars,Nz))

    except IOError:
         print('Error While Opening the file: {:s}'.format(thisFile))
    finally:
        if f:  # Check if f was successfully assigned a file object
            f.close()
            # The file is closed here

    flength = os.stat(thisSurfFile).st_size
    try:
        with open(thisSurfFile, mode='rb') as f:
            while(f.tell() < flength): #while the filepointer is not at the end of the binary file
                ## Read and parse the number elements per tower instance (single timestep)
                towerSurfInstanceSize=struct.unpack("i", f.read(4))[0]
                ## Read and parse the number instances in this file (batch of timesteps)
                batchSize=struct.unpack("i", f.read(4))[0]
                ## Read the full set of batch tower time values in this file
                towerSurfTimes=np.frombuffer(f.read(batchSize*4),dtype=np.float32)
                ## Read the full set of batch tower instances in this file
                towerSurfData[itower,iStep:(iStep+batchSteps),:]=np.frombuffer(f.read(batchSize*towerSurfInstanceSize*4),dtype=np.float32).reshape((batchSize,towerSurfInstanceSize))

    except IOError:
         print('Error While Opening the file: {:s}'.format(thisFile))
    finally:
        if f:  # Check if f was successfully assigned a file object
            f.close()
            # The file is closed here
  iStep = iStep + batchSteps

ds = xr.Dataset()
ds['time'] = xr.DataArray(towerTimes,dims=['time'])
for iVar in range(nTowerVars):
    ds[towerFldNames[iVar]] = xr.DataArray(towerData[:,:,iVar,:],dims=['towerID','time','zIndex'])
    if towerFldNames[iVar] in ['u', 'v', 'w', 'theta', 'TKE_0', 'TKE_1', 'qv', 'ql', 'qr',
                               'Tau11','Tau21','Tau31','Tau32','Tau22','Tau33',
                               'TauTH1','TauTH2','TauTH3',
                               'TauQv1','TauQv2','TauQv3',
                               'TauQv1','TauQl2','TauQl3',]:
        ds[towerFldNames[iVar]] = ds[towerFldNames[iVar]]/ds['rho']
for iVar in range(nSurfVars):
    ds[towerSurfFldNames[iVar]] = xr.DataArray(towerSurfData[:,:,iVar],dims=['towerID','time'])

ds['z'] = xr.DataArray(towerZ,dims=['towerID','zIndex'])
ds['y'] = xr.DataArray(towerY,dims=['towerID']) 
ds['x'] = xr.DataArray(towerX,dims=['towerID'])  
ds['elevation'] = xr.DataArray(towerElev,dims=['towerID'])  
ds['SeaMask'] = xr.DataArray(towerSeaMask,dims=['towerID'])  
ds['yOffset'] = xr.DataArray(towerYoffset,dims=['towerID'])  
ds['xOffset'] = xr.DataArray(towerXoffset,dims=['towerID'])

ds.to_netcdf(f"{towerPath}/{outputFileName}")
