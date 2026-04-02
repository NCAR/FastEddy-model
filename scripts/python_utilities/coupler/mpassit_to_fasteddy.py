import os, sys
import numpy as np
import xarray as xr
import argparse
import json
from netCDF4 import Dataset
import datetime as dt
import xarray as xr

from couplingUtils import *

# Read run parameters from json
args = parse_args()
with open(args.file) as file:    
    params = json.loads(file.read())    
    
MPASSIT_Dir = params["MPASSIT_Dir"]    
MPASSIT_Prefix = params["MPASSIT_Prefix"]    
FE_PrntOutDir = params["FE_PrntOutDir"]    
FE_PrntOutPrefix = params["FE_PrntOutPrefix"]    
dateString = params["date0"]    
timeHour0 = params["timeHour0"]    
timeMinute0 = params["timeMinute0"]    
timeSecond0 = params["timeSecond0"]    
secMax = params["secMax"]    
secInc = params["secInc"]   

# Define Constants
grav=9.80665
p1000mb=100000.0
rv=461.6
rd=287.0
cp=7.0*rd/2.0
cv=cp-rd
rvovrd=rv/rd
cvpm=-1.0*(cv/cp)

# Define a list of MPAS files to process from the specified coupler parameters    
files_list_mpassit=[]
files_list_fe=[]

year0 = int(dateString[0:4])
month0 = int(dateString[5:7])
day0 = int(dateString[8:10])
   
date_it = dt.datetime(year0,month0,day0,timeHour0,timeMinute0,timeSecond0)
for it in range(0,secMax,secInc):
    dateString_it = str(date_it.year) + '-' + "{:02d}".format(date_it.month)  + '-' + "{:02d}".format(date_it.day) + '_'
    thistime_mpassit = "{:s}{:02d}.{:02d}.{:02d}".format(dateString_it,date_it.hour,date_it.minute,date_it.second)
    thistime_fe = "{:s}{:02d}:{:02d}:{:02d}".format(dateString_it,date_it.hour,date_it.minute,date_it.second)
    file_tmp_mpassit = f'{MPASSIT_Dir}{MPASSIT_Prefix}{thistime_mpassit}.nc'
    file_tmp_fe = f'{FE_PrntOutDir}{FE_PrntOutPrefix}{thistime_fe}'
    files_list_mpassit.append(file_tmp_mpassit)
    files_list_fe.append(file_tmp_fe)
    date_it = date_it + dt.timedelta(seconds=secInc)

# Perform a set of conversions on the new netcdf files
for idx,file in enumerate(files_list_mpassit):
    print("Processing "+file)
    ds=xr.open_dataset(file)
    ds['PHB']=grav*ds['PHB']
    ds['PH']=grav*ds['PH']
    ds['ALT']=(rd/p1000mb)*(300.0+ds['T'])*(1.0+rvovrd*ds['QVAPOR'])*(((ds['P']+ds['PB'])/p1000mb)**cvpm)
    ds.to_netcdf(files_list_fe[idx]) # rewrite to netcdf
    ds.close()
