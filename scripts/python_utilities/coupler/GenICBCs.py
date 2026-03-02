from mpi4py import MPI
import os, sys
import struct
import argparse
import json
import numpy as np
import numpy.matlib
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import time 
import warnings
from matplotlib.gridspec import GridSpec
from scipy import ndimage,interpolate
from netCDF4 import Dataset
import datetime as dt
#from datetime import date


from couplingUtils import *

################## main()
mpi_size = MPI.COMM_WORLD.Get_size()
mpi_rank = MPI.COMM_WORLD.Get_rank()
mpi_name = MPI.Get_processor_name()

print("{:d}/{:d}: Hello World! on {:s}.".format(mpi_rank, mpi_size, mpi_name))
DEBUG_COUPLER = False #True

######################################################
### Parse the command line arguments                ###
######################################################
args = parse_args()

########################################################
### Read the ijson file of coupler script parameters ###
########################################################
with open(args.file) as file:
    params = json.loads(file.read())

ICBC_dir = params["ICBC_dir"]
FE_simGrid = params["FE_simGrid"]
WRF_PrntDir = params["WRF_PrntDir"]
WRF_PrntOutPrefix = params["WRF_PrntOutPrefix"]
dateString = params["date0"]
timeHour0 = params["timeHour0"]
timeMinute0 = params["timeMinute0"]
timeSecond0 = params["timeSecond0"]
secMax = params["secMax"]
secInc = params["secInc"]

print(f"{mpi_rank}/{mpi_size}: Writing coupler outputs to {ICBC_dir}")
print(f"{mpi_rank}/{mpi_size}: Interpolating to FE-domain from {FE_simGrid}")
print(f"{mpi_rank}/{mpi_size}: Processing of WRF-files {WRF_PrntDir}{WRF_PrntOutPrefix}*")
print(f"{mpi_rank}/{mpi_size}: Date and times of WRF-files to process: {dateString}_{timeHour0:02}:{timeMinute0:02}:*, every {secInc} s for {secMax} total seconds.")

################################################################################################
### Create a coupler output directory for initial and boundary conditions if necessary
################################################################################################
if(mpi_rank == 0):
  if not(os.path.exists(ICBC_dir)):
    os.makedirs(ICBC_dir)

################################################################################################
### Define a master list of WRF files to process from the specified coupler parameters
################################################################################################
files_list=[]
times=[]

year0 = int(dateString[0:4])
month0 = int(dateString[5:7])
day0 = int(dateString[8:10])

date_it = dt.datetime(year0,month0,day0,timeHour0,timeMinute0,timeSecond0)
for it in range(0,secMax,secInc):
    dateString_it = str(date_it.year) + '-' + "{:02d}".format(date_it.month)  + '-' + "{:02d}".format(date_it.day) + '_'
    thistime = "{:s}{:02d}:{:02d}:{:02d}".format(dateString_it,date_it.hour,date_it.minute,date_it.second)
    file_tmp = f'{WRF_PrntDir}{WRF_PrntOutPrefix}{thistime}'
    files_list.append(file_tmp)
    if(mpi_rank == 0):
       print(file_tmp)
    date_it = date_it + dt.timedelta(seconds=secInc)

################################################################################################
### Setup mpi task decomposition over the set of files to process
################################################################################################
list_len = len(files_list)
currentExists = True
nextExists = False
listCntr=0
list_StartOffset=0
while (listCntr < list_len) and currentExists:
  bdyFileName = "{:s}FE_Bndys.{:d}".format(ICBC_dir,listCntr) 
  if (mpi_rank ==0):
    print("{:d}/{:d}: Checking for {:s}".format(mpi_rank, mpi_size,bdyFileName))
  currentExists = os.path.isfile(bdyFileName)
  if not(currentExists):
      list_StartOffset = listCntr
      print("{:d}/{:d}: {:s} is missing, setting list_StartOffset = {:d} ".format(mpi_rank, mpi_size,bdyFileName,list_StartOffset))
  else:
      listCntr+=1
MPI.COMM_WORLD.Barrier()
elems_perRank = np.int32(np.floor((list_len-list_StartOffset)/mpi_size))
extra_elems = np.int32((list_len-list_StartOffset)% elems_perRank)
if(mpi_rank == 0):
  print("{:d}/{:d}: len(files_list) = {:d}".format(mpi_rank, mpi_size,list_len))
  print("{:d}/{:d}: len(files_list)-list_StartOffset = {:d}".format(mpi_rank, mpi_size,list_len-list_StartOffset))
  print("{:d}/{:d}: elems_perRank = {:d}".format(mpi_rank, mpi_size,elems_perRank))
  print("{:d}/{:d}: extra_elems = {:d}".format(mpi_rank, mpi_size,extra_elems))
MPI.COMM_WORLD.Barrier()
for iRank in range(mpi_size):
  MPI.COMM_WORLD.Barrier()
  if mpi_rank == iRank:
     mystart = (iRank)*elems_perRank + list_StartOffset
     myend = (iRank+1)*elems_perRank + list_StartOffset
     if iRank == (mpi_size-1):
        myend = myend+((list_len-list_StartOffset)-mpi_size*elems_perRank) ###Catch straggler files with the last rank
     print("{:d}/{:d}: mylist = files_list({:d}:{:d})".format(mpi_rank, mpi_size,mystart,myend))
     mylist = files_list[mystart:myend]
     print("{:d}/{:d}: len(mylist) = {:d}".format(mpi_rank, mpi_size,len(mylist)))
  MPI.COMM_WORLD.Barrier()


MPI.COMM_WORLD.Barrier()

##############################################################################
### Ingest the target FE grid created from "BuildingMask_FEgen_SimGrid.py" ###
##############################################################################
ds_FEGrid=xr.open_dataset(FE_simGrid, engine="netcdf4")

########################################
### Load the reference WRF data file ###
########################################
ds_WRFRef=xr.open_dataset(files_list[0], engine="netcdf4")

############################################################################################
### Locate the FE-grid-bounding corners as index pairs (j,i) in the WRF-reference domain ###
############################################################################################

## Find the the WRF d02 profiler locations
itargs=[]
jtargs=[]
#print('WRF:')
#print('(j,i):')

latFE = ds_FEGrid.lat.values
lonFE = ds_FEGrid.lon.values

corners_lat = np.asarray([latFE[0,0], latFE[0,-1],latFE[-1,-1], latFE[-1,0]])
corners_lon = np.asarray([lonFE[0,0], lonFE[0,-1],lonFE[-1,-1], lonFE[-1,0]])
print('corners_lat.shape=',corners_lat.shape)
print('corners_lon.shape=',corners_lon.shape)
print('corners_lat=',corners_lat)
print('corners_lon=',corners_lon)

for indx in range(len(corners_lat)):
    blah3=np.sqrt( (ds_WRFRef['XLAT'][0,:,:].values-corners_lat[indx])**2
                  +(ds_WRFRef['XLONG'][0,:,:].values-corners_lon[indx])**2)
    locCount=0
    for jtarg, itarg in np.argwhere(blah3 == np.min(blah3,axis=(0,1))): 
        if locCount < 1:
            #print('{:d},{:d}'.format(jtarg,itarg))
            jtargs.append(jtarg)
            itargs.append(itarg)
            locCount+=1
        else:
            #skip this redundant location of minimum distance
            if(mpi_rank == 0):
              print('Skipping redundant closest corner location: ',jtarg,itarg)

#### Append the corner index pairs to the WRFref dataset dFE_jindxs and dFE_iindxs  
ds_WRFRef['dFE_jindxs']=xr.DataArray(np.asarray(jtargs,dtype=np.int32),dims=["corners"])
ds_WRFRef['dFE_iindxs']=xr.DataArray(np.asarray(itargs,dtype=np.int32),dims=["corners"])
for indx in range(len(corners_lat)):
  if(mpi_rank == 0):
    print(f"corner({indx}) @ WRF({ds_WRFRef['dFE_jindxs'][indx].values},{ds_WRFRef['dFE_iindxs'][indx].values})")
    print('[WRF,corner({:d})]: lats = [{:f},{:f}], lons = [{:f},{:f}]'.format(indx,ds_WRFRef['XLAT'][0,ds_WRFRef['dFE_jindxs'][indx],ds_WRFRef['dFE_iindxs'][indx]].values,
                                                                                 corners_lat[indx],
                                                                                 ds_WRFRef['XLONG'][0,ds_WRFRef['dFE_jindxs'][indx],ds_WRFRef['dFE_iindxs'][indx]].values,
                                                                                 corners_lon[indx]))
    
    
### Compute FE-domain corner lat/lon offsets from closest dsWRFRef cell-centered lat/lons
print('\t Pre-correction offsets:')        
for indx in range(len(corners_lat)):  
    latOff = ds_WRFRef['XLAT'][0,ds_WRFRef['dFE_jindxs'][indx].values,ds_WRFRef['dFE_iindxs'][indx].values]-corners_lat[indx]
    lonOff = ds_WRFRef['XLONG'][0,ds_WRFRef['dFE_jindxs'][indx].values,ds_WRFRef['dFE_iindxs'][indx].values]-corners_lon[indx]
    if(mpi_rank == 0):
      print('\t corner({:d}): latOff = {:f}, lonOff = {:f}'.format(indx,latOff,lonOff))
    yoffset = -(ds_WRFRef['XLAT'][0,ds_WRFRef['dFE_jindxs'][indx],ds_WRFRef['dFE_iindxs'][indx]]-corners_lat[indx])\
              *(ds_WRFRef.attrs['DY']/( ds_WRFRef['XLAT'][0,ds_WRFRef['dFE_jindxs'][indx]+1,ds_WRFRef['dFE_iindxs'][indx]]
                                       -ds_WRFRef['XLAT'][0,ds_WRFRef['dFE_jindxs'][indx],ds_WRFRef['dFE_iindxs'][indx]]))
    xoffset = -(ds_WRFRef['XLONG'][0,ds_WRFRef['dFE_jindxs'][indx],ds_WRFRef['dFE_iindxs'][indx]]-corners_lon[indx])\
              *(ds_WRFRef.attrs['DX']/( ds_WRFRef['XLONG'][0,ds_WRFRef['dFE_jindxs'][indx],ds_WRFRef['dFE_iindxs'][indx]+1]
                                       -ds_WRFRef['XLONG'][0,ds_WRFRef['dFE_jindxs'][indx],ds_WRFRef['dFE_iindxs'][indx]]))   
    if(mpi_rank == 0):
      print('\t corner({:d}): yOff = {:f}, xOff = {:f}'.format(indx,yoffset.values,xoffset.values))
    if indx<2:   ## 0=southwest, or 1=southeast corner
        if yoffset < 0.0:  #FE domain SW/SE corner is south of the closest wrf cell, decrement the bounding jindx
            ds_WRFRef['dFE_jindxs'][indx]-=1 
        if indx < 1: 
            if xoffset < 0.0: #FE domain SW corner is west of the closest wrf cell, decrement the bounding iindx
                ds_WRFRef['dFE_iindxs'][indx]-=1
        else: 
            if xoffset > 0.0: #FE domain SE corner is east of the closest wrf cell, increment the bounding iindx
                ds_WRFRef['dFE_iindxs'][indx]+=1
    else:   ## 3=northwest, or 2=northeast corner
        if yoffset > 0.0:  #FE domain NE/NW corner is north of the closest wrf cell, increment the bounding jindx
            ds_WRFRef['dFE_jindxs'][indx]+=1
        if indx < 3: 
            if xoffset > 0.0: #FE domain NE corner is east of the closest wrf cell, increment the bounding iindx
                ds_WRFRef['dFE_iindxs'][indx]+=1
        else: 
            if xoffset < 0.0: #FE domain NW corner is west of the closest wrf cell, decrement the bounding iindx
                ds_WRFRef['dFE_iindxs'][indx]-=1 
if(mpi_rank == 0):
  print('Bounding-box corrected offsets:')        
for indx in range(len(corners_lat)):  
    latOff = ds_WRFRef['XLAT'][0,ds_WRFRef['dFE_jindxs'][indx].values,ds_WRFRef['dFE_iindxs'][indx].values]-corners_lat[indx]
    lonOff = ds_WRFRef['XLONG'][0,ds_WRFRef['dFE_jindxs'][indx].values,ds_WRFRef['dFE_iindxs'][indx].values]-corners_lon[indx]
    yoffset = -(ds_WRFRef['XLAT'][0,ds_WRFRef['dFE_jindxs'][indx],ds_WRFRef['dFE_iindxs'][indx]]-corners_lat[indx])\
              *(ds_WRFRef.attrs['DY']/( ds_WRFRef['XLAT'][0,ds_WRFRef['dFE_jindxs'][indx]+1,ds_WRFRef['dFE_iindxs'][indx]]
                                       -ds_WRFRef['XLAT'][0,ds_WRFRef['dFE_jindxs'][indx],ds_WRFRef['dFE_iindxs'][indx]]))
    xoffset = -(ds_WRFRef['XLONG'][0,ds_WRFRef['dFE_jindxs'][indx],ds_WRFRef['dFE_iindxs'][indx]]-corners_lon[indx])\
              *(ds_WRFRef.attrs['DX']/( ds_WRFRef['XLONG'][0,ds_WRFRef['dFE_jindxs'][indx],ds_WRFRef['dFE_iindxs'][indx]+1]
                                       -ds_WRFRef['XLONG'][0,ds_WRFRef['dFE_jindxs'][indx],ds_WRFRef['dFE_iindxs'][indx]]))   
    if(mpi_rank == 0):
      print('corner({:d}):latOff = {:f}, lonOff = {:f} -- yOff = {:f}, xOff = {:f}'.format(indx,latOff,lonOff,yoffset.values,xoffset.values))
    if indx == 0:
        ll_yoffset=yoffset.values
        ll_xoffset=xoffset.values
#print('WRF:')
#print('(j,i):')
for indx in range(len(corners_lat)):
    if(mpi_rank == 0):
      print('{:d},{:d}'.format(ds_WRFRef['dFE_jindxs'][indx].values,ds_WRFRef['dFE_iindxs'][indx].values))
    
#Nesting configuration parameters
ll_jindx=ds_WRFRef['dFE_jindxs'].min(dim='corners').values
ll_iindx=ds_WRFRef['dFE_iindxs'].min(dim='corners').values
j_extent=ds_WRFRef['dFE_jindxs'].max(dim='corners').values-ll_jindx
i_extent=ds_WRFRef['dFE_iindxs'].max(dim='corners').values-ll_iindx

##Ensure WRF interpolation area extents will entirely encompass FE target x & y domain 
y_distWRF = (j_extent-1)*ds_WRFRef.attrs['DY']-ll_yoffset
x_distWRF = (i_extent-1)*ds_WRFRef.attrs['DX']-ll_xoffset
dxFE=(ds_FEGrid['xPos'][0,0,1]-ds_FEGrid['xPos'][0,0,0]).values
dyFE=(ds_FEGrid['yPos'][0,1,0]-ds_FEGrid['yPos'][0,0,0]).values
x_distFE = (ds_FEGrid.sizes['xIndex']-1)*dxFE
y_distFE = (ds_FEGrid.sizes['yIndex']-1)*dyFE
while x_distWRF <= x_distFE:
    i_extent += 1
    x_distWRF = (i_extent-1)*ds_WRFRef.attrs['DX']-ll_xoffset
while y_distWRF <= y_distFE:
    j_extent += 1
    y_distWRF = (j_extent-1)*ds_WRFRef.attrs['DY']-ll_yoffset

if(mpi_rank == 0):
    if (ll_jindx < 0):
        print(f"Southern FE domain boundary coordinate falls outside of the provided WRF domain, exiting.")
        exit()
    elif (ll_iindx < 0):
        print(f"Western Ft domain boundary coordinate falls outside of the provided WRF domain, exiting.")
        exit()
    elif (ll_jindx+j_extent > ds_WRFRef.sizes['south_north']):
        print(f"Northern Ft domain boundary coordinate falls outside of the provided WRF domain, exiting.")
        exit()
    elif (ll_iindx+i_extent > ds_WRFRef.sizes['west_east']):  
        print(f"Eastern Ft domain boundary coordinate falls outside of the provided WRF domain, exiting.")
        exit()
    else:  #All set to perform strictly interpolation in the horizontal of WRF outputs to FE domain 
        print('ll: ({:d},{:d})'.format(ll_jindx,ll_iindx))
        print('extents: ({:d},{:d})'.format(j_extent,i_extent))
        print('y,x offsets: ({:f},{:f})'.format(ll_yoffset,ll_xoffset))

######################################################################################################################
### Define the Cartesian southwest corner origin (x,y) WRF coordinate system for the horizontal FE-bounding domain ###
######################################################################################################################
xWRF,stepX=np.linspace((ll_iindx+0.5)*ds_WRFRef.attrs['DX'],(ll_iindx+0.5+i_extent)*ds_WRFRef.attrs['DX'],i_extent,endpoint=False,retstep=True)
yWRF,stepY=np.linspace((ll_jindx+0.5)*ds_WRFRef.attrs['DY'],(ll_jindx+0.5+j_extent)*ds_WRFRef.attrs['DY'],j_extent,endpoint=False,retstep=True)
print(xWRF,'\n',yWRF,'\n')
print(stepX,stepY)

##Given the correct x,y vectors create 2-d grids of x and y coordinates
YvWRF,XvWRF=np.meshgrid(yWRF,xWRF, sparse=False, indexing='ij')
print(XvWRF.shape,XvWRF.shape)


print(ds_WRFRef['HGT'][0,ll_jindx:ll_jindx+j_extent,ll_iindx:ll_iindx+i_extent].values)

####################################################################################
### Map the target FE domain into the WRF bounding-grid relative x,y coordinates ###
####################################################################################
dxFE=(ds_FEGrid['xPos'][0,0,1]-ds_FEGrid['xPos'][0,0,0]).values
dyFE=(ds_FEGrid['yPos'][0,1,0]-ds_FEGrid['yPos'][0,0,0]).values
if(mpi_rank == 0) and DEBUG_COUPLER:
  print("Nx,NY = ({:d},{:d}), dxFE={:f},dyFE={:f}".format(ds_FEGrid.sizes['xIndex'],ds_FEGrid.sizes['yIndex'],dxFE,dyFE))
x0FEinWRF=XvWRF[0,0]+ll_xoffset 
y0FEinWRF=YvWRF[0,0]+ll_yoffset 
if(mpi_rank == 0) and DEBUG_COUPLER:
  print("x0FEinWRF,y0FEinWRF = {:f},{:f}".format(x0FEinWRF,y0FEinWRF))
xNFEinWRF = x0FEinWRF+(ds_FEGrid.sizes['xIndex']-1)*dxFE
yNFEinWRF = y0FEinWRF+(ds_FEGrid.sizes['yIndex']-1)*dyFE
if(mpi_rank == 0) and DEBUG_COUPLER:
  print("xNFEinWRF,yNFEinWRF = {:f},{:f}".format(xNFEinWRF,yNFEinWRF))
  print("X_extentFE,Y_extentFE = {:f},{:f}".format(xNFEinWRF-x0FEinWRF,yNFEinWRF-y0FEinWRF))
xVec,stepX=np.linspace(x0FEinWRF,xNFEinWRF,ds_FEGrid.sizes['xIndex'],endpoint=True,retstep=True)
yVec,stepY=np.linspace(y0FEinWRF,yNFEinWRF,ds_FEGrid.sizes['yIndex'],endpoint=True,retstep=True)
if(mpi_rank == 0) and DEBUG_COUPLER:
  print('\n',xVec[0],yVec[0],'\n',xVec[-1],yVec[-1],'\n')
  print(stepX,stepY)
  print(xVec.shape,yVec.shape)

#####################################
### Define Eq. of State Constants ###
#####################################
refPressure = 1.0e5
accel_g = 9.81
R_gas   = 287.04
cv_gas  = 718.0
cp_gas  = R_gas+cv_gas
R_cp = R_gas/cp_gas
cp_R = cp_gas/R_gas

####################################################################################################
### Define a rectilinear vertical coordinate basis for vertical interpolation between WRF and FE ###
####################################################################################################
zBottom = 0.0
zTop = ds_FEGrid['zPos'][-1,0,0].values+90.0
NzWRFInterp = 200 #275
zRect = np.linspace(zBottom,zTop,NzWRFInterp)
if(mpi_rank == 0) and DEBUG_COUPLER:
  print(zRect[0],zRect[-1])

##############################################################################
### Create lists of relevant variable names in the WRF-FE coupling process ###
##############################################################################
varsList = ['Z','ALT','U','V','W','T','QVAPOR','QCLOUD']
surfVarsList = ['TSK','Q2','HGT','PSFC']  #Note: Q2 in absence of QVG (which is not in wrfout by deafult) from WRF
FEvarsList = ['rho','u','v','w','theta','qv','ql']
FEsurfVarsList = ['tskin','qskin','topoWRF','psfc','SeaMask']

#######################################################################
### Finally go ahead and create the initial and boundary conditions ###
#######################################################################
it0=0
it00=mystart #rank-specific starting index in the files_list 
it11=myend #rank-specific ending index in the files_list 

for Bdy_file_num in range(it00,it11):
  bdyFileName = "{:s}/FE_Bndys.{:d}".format(ICBC_dir,Bdy_file_num)
  if not(os.path.isfile(bdyFileName)):
    print('{:d}{:d}: {:s} does not exist, creating it...'.format(mpi_rank, mpi_size, bdyFileName))
    print("{:d}{:d}: Working on file {:s}".format(mpi_rank, mpi_size, files_list[Bdy_file_num]))
    ds_ref = xr.open_mfdataset(files_list[Bdy_file_num],combine='nested',concat_dim='Time')

    t0s = time.perf_counter()
    dsWRF=interpWRFToGrids(ds_ref,it0,varsList,surfVarsList,zRect,ll_iindx,i_extent,ll_jindx,j_extent)
    t0e = time.perf_counter()
    print('{:d}/{:d}: t0_elapsed = {:f} (s)'.format(mpi_rank, mpi_size, t0e-t0s))
    t1s = time.perf_counter()
    ds=copyAndTranspose(dsWRF)
    t1e = time.perf_counter()
    print('{:d}/{:d}: t1_elapsed = {:f} (s)'.format(mpi_rank, mpi_size, t1e-t1s))
    t2s = time.perf_counter()
    dsFENew=interp2DForFE(ds,ds_FEGrid,XvWRF,YvWRF,xVec,yVec)
    t2e = time.perf_counter()
    print('{:d}/{:d}: t2_elapsed = {:f} (s)'.format(mpi_rank, mpi_size, t2e-t2s))
    t3s = time.perf_counter()
    dsFEFinal=create_dsFEFinal(ds_FEGrid)
    verticalInterpFinal(ds_FEGrid,dsFENew,dsFEFinal,zRect)
    if 'BuildingMask' in list(dsFEFinal.variables):
       for var in ['u','v','w','ql','TKE_0']:
          if var in list(dsFEFinal.variables):
            dsFEFinal[var][:,:,:]=dsFEFinal[var][:,:,:]*np.where((dsFEFinal['BuildingMask'][:,:,:]>1e-3),0.0,1.0)
    t3e = time.perf_counter()
    print('{:d}/{:d}: t3_elapsed = {:f} (s)'.format(mpi_rank, mpi_size, t3e-t3s))
    addTimeDim_FEfinal(dsFEFinal)
    if Bdy_file_num == 0:
        timeLabel="{:02d}{:02d}{:02d}UTC".format(timeHour0,timeMinute0,timeSecond0)
        dsFEFinal.to_netcdf(ICBC_dir+'FE_interp_{:s}.{:d}'.format(timeLabel,0),format='NETCDF4',
                            encoding={'xIndex': {'dtype': 'i4'},'yIndex': {'dtype': 'i4'},'zIndex': {'dtype': 'i4'}})
    t4s = time.perf_counter()
    ds_Bdy=create_dsBdy(ds_FEGrid,FEvarsList,FEsurfVarsList)
    t4e = time.perf_counter()
    print('{:d}/{:d}: t4_elapsed = {:f} (s)'.format(mpi_rank, mpi_size, t4e-t4s))
    t5s = time.perf_counter()
    createBdysFrom3D(ds_Bdy,dsFEFinal,FEvarsList,FEsurfVarsList)
    new_fileName="FE_Bndys.{:d}".format(Bdy_file_num)
    writeBdyFile(ICBC_dir,new_fileName,ds_Bdy)
    t5e = time.perf_counter()
    print('{:d}/{:d}: t5_elapsed = {:f} (s)'.format(mpi_rank, mpi_size, t5e-t5s))
  else:
    print('{:d}/{:d}: {:d} exists, skipping...'.format(mpi_rank, mpi_size, Bdy_file_num))
MPI.COMM_WORLD.Barrier()
print("{:d}/{:d}: Goodbye World! on {:s}.".format(mpi_rank, mpi_size, mpi_name))
MPI.Finalize()
