import os, sys
import numpy as np
import xarray as xr
from netCDF4 import Dataset
import datetime as dt
import xarray as xr


dateString="2024-02-16"
timeHour0=17
timeMinute0=00
timeSecond0=00
secMax=7500
secInc=300
MPASSIT_PrntDir="./"
MPASSIT_PrntOutPrefix="proc."
FE_PrntOutPrefix="mpassit_"

#Define Constants
grav=9.80665
p1000mb=100000.0
rv=461.6
rd=287.0
cp=7.0*rd/2.0
cv=cp-rd
rvovrd=rv/rd
cvpm=-1.0*(cv/cp)

################################################################################################    
### Define a list of MPAS files to process from the specified coupler parameters    
################################################################################################    
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
    file_tmp_mpassit = f'{MPASSIT_PrntDir}{MPASSIT_PrntOutPrefix}{thistime_mpassit}.nc'
    file_tmp_fe = f'{MPASSIT_PrntDir}{FE_PrntOutPrefix}{thistime_fe}'
    files_list_mpassit.append(file_tmp_mpassit)
    files_list_fe.append(file_tmp_fe)
    date_it = date_it + dt.timedelta(seconds=secInc)

for idx,file in enumerate(files_list_mpassit):
    print("Processing "+file)
    ds=xr.open_dataset(file)
    ds['PHB']=grav*ds['PHB']
    ds['PH']=grav*ds['PH']
    ds['ALT']=(rd/p1000mb)*(300.0+ds['T'])*(1.0+rvovrd*ds['QVAPOR'])*(((ds['P']+ds['PB'])/p1000mb)**cvpm)
    ds.to_netcdf(files_list_fe[idx]) # rewrite to netcdf
    ds.close()
