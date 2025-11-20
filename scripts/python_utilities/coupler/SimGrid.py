import os, sys
import numpy as np
import numpy.matlib
import xarray as xr
import argparse
import json
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import time
import matplotlib
import math
import struct
from scipy.interpolate import RectBivariateSpline, NearestNDInterpolator
from couplingUtils import *

#######################################
### read parameters from .json file ###
#######################################

args = parse_args()
with open(args.file) as file:
    params = json.loads(file.read())
#
name_dom = params["name_dom"]
FE_ref_GIS_nc = params["FE_ref_GIS_nc"]
FE_params_file = params["FE_params_file"]
center_lat = params["center_lat"]
center_lon = params["center_lon"]
urban_opt = params["urban_opt"]
FE_new_nc_path = params["FE_new_nc_path"]
name_dom_add = params["name_dom_add"]
save_plot_opt = params["save_plot_opt"]

#######################################

# Read in FE params file

FE_params = get_params_FE(FE_params_file)

# Other steps

start_code_all = time.perf_counter()

FE_new_nc = FE_new_nc_path + name_dom + name_dom_add + '.0'
FE_plot = FE_new_nc_path + name_dom + name_dom_add + '_simgrid.png'

print('FE_ref_GIS_nc: ',FE_ref_GIS_nc)
print('FE_new_nc: ',FE_new_nc)

if not os.path.exists(FE_new_nc_path):
    os.makedirs(FE_new_nc_path)

ds_GIS = xr.open_dataset(FE_ref_GIS_nc)

dx_inter = ds_GIS.dx_inter.values
dy_inter = ds_GIS.dy_inter.values

Nz = int(str(FE_params['Nz'][0]))

### determine x_s,x_e,y_s,y_e for the coarsened grid ###

xPos_2d = ds_GIS.xPos2d.values
yPos_2d = ds_GIS.yPos2d.values
lat = ds_GIS.lat.values
lon = ds_GIS.lon.values

Nx = int(str(FE_params['Nx'][0]))
Ny = int(str(FE_params['Ny'][0]))
d_xi = float(str(FE_params['d_xi'][0]))
d_eta = float(str(FE_params['d_eta'][0]))
d_zeta = float(str(FE_params['d_zeta'][0]))

print('Nx,Ny,d_xi,d_eta,d_zeta=',Nx,',',Ny,',',d_xi,',',d_eta,',',d_zeta)

ll_diff = np.sqrt(np.power(lat-center_lat,2.0)+np.power(lon-center_lon,2.0))
ind_center = np.where(ll_diff==np.min(ll_diff,axis=(1,0)))
ind_cc_y = ind_center[0][0]
ind_cc_x = ind_center[1][0]
print('ind_cc_x,ind_cc_y=',ind_cc_x,',',ind_cc_y)

x_s = ind_cc_x - int(0.5*Nx*d_xi/dx_inter)
y_s = ind_cc_y - int(0.5*Ny*d_eta/dy_inter)
print('xPos_2d[y_s,x_s]=',xPos_2d[y_s,x_s],'m')
print('yPos_2d[y_s,x_s]=',yPos_2d[y_s,x_s],'m')

##

npx_inc = int(d_xi/dx_inter)
npy_inc = int(d_eta/dy_inter)
if (npx_inc==0):
    npx_inc = d_xi/dx_inter
    npy_inc = d_eta/dy_inter
    x_e = x_s + int(np.ceil(Nx*npx_inc))
    y_e = y_s + int(np.ceil(Ny*npy_inc))
else:
    x_e = x_s + Nx*npx_inc
    y_e = y_s + Ny*npy_inc
print('x_s,x_e,y_s,y_e=',x_s,x_e,y_s,y_e)

box_indx = [x_s,x_e,x_e,x_s,x_s]
box_indy = [y_s,y_s,y_e,y_e,y_s]

x_box_corners = [xPos_2d[box_indy[0],box_indx[0]], xPos_2d[box_indy[1],box_indx[1]], xPos_2d[box_indy[2],box_indx[2]], xPos_2d[box_indy[3],box_indx[3]], xPos_2d[box_indy[4],box_indx[4]]]
y_box_corners = [yPos_2d[box_indy[0],box_indx[0]], yPos_2d[box_indy[1],box_indx[1]], yPos_2d[box_indy[2],box_indx[2]], yPos_2d[box_indy[3],box_indx[3]], yPos_2d[box_indy[4],box_indx[4]]]
print('x_box_corners=',x_box_corners,'m')
print('y_box_corners=',y_box_corners,'m')

# Decide between decimation and interpolation
if (d_xi%dx_inter==0.0):
    print('Grid is an even factor of GIS resolution, use decimation')
    interp_flag = 0
else:
    print('Grid is not an even factor of GIS resolution, use interpolation')
    interp_flag = 1
print('interp_flag=',interp_flag)

verticalDeformSwitch = int(str(FE_params['verticalDeformSwitch'][0]))
print('verticalDeformSwitch=',verticalDeformSwitch)
if (verticalDeformSwitch==1):
    c1 = float(str(FE_params['verticalDeformFactor'][0]))
    fCoeff = float(str(FE_params['verticalDeformQuadCoeff'][0]))
else:
    c1 = 0.0
    fCoeff = 0.0
print('c1,fCoeff=',c1,fCoeff)

######################
# Calculate new zPos #
######################
## Read in terrain elevation array
topo = ds_GIS.topoPos.values
if (interp_flag==0):
    data_topo0 = topo[y_s:y_e:npy_inc,x_s:x_e:npx_inc]
else:
    xPos_2d_dom_ori = xPos_2d[y_s:y_e,x_s:x_e]
    yPos_2d_dom_ori = yPos_2d[y_s:y_e,x_s:x_e]
    topo_dom_ori = topo[y_s:y_e,x_s:x_e]
    print('xPos_2d_dom_ori.shape=',xPos_2d_dom_ori.shape)
    f_topo = RectBivariateSpline(xPos_2d_dom_ori[0,:], yPos_2d_dom_ori[:,0], topo_dom_ori.T, kx=3, ky=3)

    xPos_1d_new = np.arange(x_box_corners[0],x_box_corners[1],d_xi)
    yPos_1d_new = np.arange(y_box_corners[0],y_box_corners[2],d_eta)

    data_topo0_b = f_topo(xPos_1d_new, yPos_1d_new).T
    xPos_2d_new_b, yPos_2d_new_b = np.meshgrid(xPos_1d_new, yPos_1d_new)

    data_topo0 = data_topo0_b[0:Ny,0:Nx]
    xPos_2d_new = xPos_2d_new_b[0:Ny,0:Nx]
    yPos_2d_new = yPos_2d_new_b[0:Ny,0:Nx]

data_topo = smoothTerrain(data_topo0,d_xi)

topoPos_min = np.min(data_topo,axis=(1,0))
topoPos_max = np.max(data_topo,axis=(1,0))
print('topoPos_min=',topoPos_min)
print('topoPos_max=',topoPos_max)

terrain_filename = "{:s}_Topography_{:d}x{:d}.dat".format(name_dom,Nx,Ny)
FE_new_terrain = FE_new_nc_path + terrain_filename
topoflat = bytearray(Nx*Ny*4)
struct.pack_into('{:d}f'.format(Nx*Ny),topoflat,0,*data_topo.flatten())
topoFile = open(FE_new_terrain,'wb')
topoFile.write(struct.pack('i',Nx))
topoFile.write(struct.pack('i',Ny))
topoFile.write(topoflat)
topoFile.close()

ztop = Nz*d_zeta-0.5*d_zeta # Note the rectilinear vertical resolution max
zPos_str = np.zeros(Nz)
print(zPos_str.shape)
zarr = np.zeros((Nz,Ny,Nx),dtype=np.float32)

if(np.mean(topo,axis=(0,1)) == 0.0):
  zbot = 0.0
  print('zbot,ztop=',zbot,'m ,',ztop,'m')

  for kk in range(0,Nz):
    zPos_uni = kk*d_zeta + 0.5*d_zeta
    zPos_str[kk] = zDeform(zPos_uni,zbot,ztop,c1,fCoeff)
    if (kk==0):
        print('kk,zPos_str,dz=',kk,',',zPos_str[kk],', -')
    else:
        print('kk,zPos_str,dz=',kk,',',zPos_str[kk],',',zPos_str[kk]-zPos_str[kk-1])
    zarr[kk,:,:] = zPos_str[kk]
else:
  for j in range(Ny):
    if(j%int(Ny/10)==0):
      print('{:d}% complete...'.format(10*int(j/int(Ny/10))))
    for i in range(Nx):
        zbot = data_topo[j,i]
        zPos_uni = np.linspace(0.5*d_zeta,(Nz-0.5)*d_zeta,Nz)
        zPos_str = zDeform(zPos_uni,zbot,ztop,c1,fCoeff)
        zarr[:,j,i] = zPos_str
        if (j==0) and (i==0):
         for k in range(Nz):
           if (k==0):
             print('k,zPos_str,dz=',k,',',zPos_str[k],', -')
           else:
             print('k,zPos_str,dz=',k,',',zPos_str[k],',',zPos_str[k]-zPos_str[k-1])

ind_topomin = np.where(data_topo==topoPos_min)
if isinstance(ind_topomin, tuple):
   j_topomin = ind_topomin[0][0]
   i_topomin = ind_topomin[1][0]
else:
   j_topomin = ind_topomin[0]
   i_topomin = ind_topomin[1]
z_lowTopo_v = zarr[:,j_topomin,i_topomin]
ind_topomax = np.where(data_topo==topoPos_max)
if isinstance(ind_topomax, tuple):
  j_topomax = ind_topomax[0][0]
  i_topomax = ind_topomax[1][0]
else:
  j_topomax = ind_topomax[0]
  i_topomax = ind_topomax[1]
z_highTopo_v = zarr[:,j_topomax,i_topomax]

dz_lowTopo_v = z_lowTopo_v[1:Nz]-z_lowTopo_v[0:Nz-1]
dz_highTopo_v = z_highTopo_v[1:Nz]-z_highTopo_v[0:Nz-1]

print('topoPos_min =',topoPos_min,'m')
print('topoPos_max =',topoPos_max,'m')
print('Domain top =',ztop,'m')
print('Maximum domain vertical extent =',z_lowTopo_v[-1]-z_lowTopo_v[0],'m')
print('Minimum domain vertical extent =',z_highTopo_v[-1]-z_highTopo_v[0],'m')
print('dz_lowTopo_v at the surface =',dz_lowTopo_v[0],'m')
print('dz_highTopo_v at the surface =',dz_highTopo_v[0],'m')
print('dz_lowTopo_v at the top =',dz_lowTopo_v[-1],'m')
print('dz_highTopo_v at the top =',dz_highTopo_v[-1],'m')

# building mask
if (urban_opt == 1):
    print('Computing 3-d building mask...')

    bdg_heights = ds_GIS.BuildingHeights.values
    if (interp_flag==0):
        data_bmask = bdg_heights[y_s:y_e:npy_inc,x_s:x_e:npx_inc]
    else:
        f_bdg = NearestNDInterpolator(list(zip(xPos_2d_dom_ori.flatten(), yPos_2d_dom_ori.flatten())), data_bmask[y_s:y_e,x_s:x_e].flatten())
        data_bmask = f_bdg(xPos_2d_new, yPos_2d_new)

    bdg3d_tmp = np.zeros((Nz,Ny,Nx),dtype=np.float32)

    for j in range(Ny):
        if(j%int(Ny/10)==0):
          print('{:d}% complete...'.format(10*int(j/int(Ny/10))))
        for i in range(Nx):
            z_1d = zarr[:,j,i]
            if (data_bmask[j,i]!=0.0):
                z_diff = np.abs(z_1d-(data_bmask[j,i]+data_topo[j,i]))
                ind_min = np.where(z_diff==np.amin(z_diff))
                ind_min = ind_min[0]
                ind_min = ind_min[0]
                bdg3d_tmp[0:ind_min+1,j,i] = 1.0

# xPos, yPos
xarr = np.zeros((Nz,Ny,Nx),dtype=np.float32)
if (interp_flag==0):
    xPos_2d_new = xPos_2d[y_s:y_e:npy_inc,x_s:x_e:npx_inc]
    xPos_2d_new = xPos_2d_new - xPos_2d_new[0,0] + 0.5*d_xi
for kk in range(0,Nz):
   xarr[kk,:,:] = xPos_2d_new

yarr = np.zeros((Nz,Ny,Nx),dtype=np.float32)
if (interp_flag==0):
    yPos_2d_new = yPos_2d[y_s:y_e:npy_inc,x_s:x_e:npx_inc]
    yPos_2d_new = yPos_2d_new - yPos_2d_new[0,0] + 0.5*d_xi
for kk in range(0,Nz):
   yarr[kk,:,:] = yPos_2d_new

# Surface static fields
z0m_field = ds_GIS.z0m.values
z0t_field = ds_GIS.z0t.values
SeaMask_field = ds_GIS.SeaMask.values
land_cover = ds_GIS.LandCover.values

if (interp_flag==0):
    data_z0m = z0m_field[y_s:y_e:npy_inc,x_s:x_e:npx_inc]
    data_z0t = z0t_field[y_s:y_e:npy_inc,x_s:x_e:npx_inc]
    data_SeaMask = SeaMask_field[y_s:y_e:npy_inc,x_s:x_e:npx_inc]
    data_landc = land_cover[y_s:y_e:npy_inc,x_s:x_e:npx_inc]
    lat_dom = lat[y_s:y_e:npy_inc,x_s:x_e:npx_inc]
    lon_dom = lon[y_s:y_e:npy_inc,x_s:x_e:npx_inc]
else:
    f_z0m = NearestNDInterpolator(list(zip(xPos_2d_dom_ori.flatten(), yPos_2d_dom_ori.flatten())), z0m_field[y_s:y_e,x_s:x_e].flatten())
    f_z0t = NearestNDInterpolator(list(zip(xPos_2d_dom_ori.flatten(), yPos_2d_dom_ori.flatten())), z0t_field[y_s:y_e,x_s:x_e].flatten())
    f_SeaMask = NearestNDInterpolator(list(zip(xPos_2d_dom_ori.flatten(), yPos_2d_dom_ori.flatten())), SeaMask_field[y_s:y_e,x_s:x_e].flatten())
    f_landc = NearestNDInterpolator(list(zip(xPos_2d_dom_ori.flatten(), yPos_2d_dom_ori.flatten())), land_cover[y_s:y_e,x_s:x_e].flatten())
    print('xPos_2d_dom_ori[0,:].shape=',xPos_2d_dom_ori[0,:].shape)
    print('yPos_2d_dom_ori[:,0].shape=',yPos_2d_dom_ori[:,0].shape)
    print('lat.T.shape=',lat.T.shape)
    f_lat = RectBivariateSpline(xPos_2d_dom_ori[0,:], yPos_2d_dom_ori[:,0], lat[y_s:y_e,x_s:x_e].T, kx=3, ky=3)
    f_lon = RectBivariateSpline(xPos_2d_dom_ori[0,:], yPos_2d_dom_ori[:,0], lon[y_s:y_e,x_s:x_e].T, kx=3, ky=3)
    data_z0m = f_z0m(xPos_2d_new, yPos_2d_new)
    data_z0t = f_z0t(xPos_2d_new, yPos_2d_new)
    data_SeaMask = f_SeaMask(xPos_2d_new, yPos_2d_new)
    data_landc = f_landc(xPos_2d_new, yPos_2d_new)
    lat_dom_b = f_lat(xPos_1d_new, yPos_1d_new).T
    lon_dom_b = f_lon(xPos_1d_new, yPos_1d_new).T
    lat_dom = lat_dom_b[0:Ny,0:Nx]
    lon_dom = lon_dom_b[0:Ny,0:Nx]

# Save to netCDF file

ds_data = xr.Dataset()

ds_data['xPos']= xr.DataArray(xarr,dims=(['zIndex','yIndex','xIndex']))
ds_data['yPos']= xr.DataArray(yarr,dims=(['zIndex','yIndex','xIndex']))
ds_data['zPos']= xr.DataArray(zarr,dims=(['zIndex','yIndex','xIndex']))
ds_data['topoPos']= xr.DataArray(data_topo.astype(dtype=np.float32),dims=(['yIndex','xIndex']))
ds_data['z0m']= xr.DataArray(data_z0m.astype(dtype=np.float32),dims=(['yIndex','xIndex']))
ds_data['z0t']= xr.DataArray(data_z0t.astype(dtype=np.float32),dims=(['yIndex','xIndex']))
ds_data['SeaMask']= xr.DataArray(data_SeaMask.astype(dtype=np.float32),dims=(['yIndex','xIndex']))
ds_data['LandCover']= xr.DataArray(data_landc.astype(dtype=np.int32),dims=(['yIndex','xIndex']))
if (urban_opt == 1):
    ds_data['BuildingMask']= xr.DataArray(bdg3d_tmp.astype(dtype=np.float32),dims=(['zIndex','yIndex','xIndex']))
    ds_data['BuildingHeights']= xr.DataArray(data_bmask.astype(dtype=np.float32),dims=(['yIndex','xIndex']))
ds_data['lat']= xr.DataArray(lat_dom.astype(dtype=np.float64),dims=(['yIndex','xIndex']))
ds_data['lon']= xr.DataArray(lon_dom.astype(dtype=np.float64),dims=(['yIndex','xIndex']))
ds_data['xIndex']= xr.DataArray(np.arange(0,xarr.shape[2],dtype=np.int32),dims='xIndex')
ds_data['yIndex']= xr.DataArray(np.arange(0,xarr.shape[1],dtype=np.int32),dims='yIndex')
ds_data['zIndex']= xr.DataArray(np.arange(0,xarr.shape[0],dtype=np.int32),dims='zIndex')

ds_data.to_netcdf(FE_new_nc,format='NETCDF4')

end_code_all = time.perf_counter()
print('done creating reference FastEddy grid and surface fields, elapsed time ' + str(end_code_all-start_code_all) + ' s')

# Plot

if (save_plot_opt==1):

    start_code = time.perf_counter()

    fntSize=16
    fntSize_title=22
    fntSize_legend=16
    fntSize_label=16
    plt.rcParams['xtick.labelsize']=fntSize
    plt.rcParams['ytick.labelsize']=fntSize
    plt.rcParams['axes.linewidth']=2.0

    numPlotsX = 1
    numPlotsY = 2

    sizeX = 10
    sizeY = 6

    fig,axs = plt.subplots(numPlotsX,numPlotsY,sharey=False,sharex=False,figsize=(sizeX,sizeY))

    ###############
    ### panel 0 ###
    ###############
    ax = axs[0]

    im2 = ax.plot(z_lowTopo_v,'-',color='b',linewidth=2.0,zorder=0,label='lowTopo')
    im2 = ax.plot(z_highTopo_v,'-',color='r',linewidth=2.0,zorder=0,label='highTopo')
    ax.set_ylabel(r"$z$ $[$m$]$",fontsize=fntSize)
    ax.set_xlabel(r"Vertical index $[$-$]$",fontsize=fntSize)
    ax.legend(loc=0,prop={'size': fntSize_legend},edgecolor='white')

    ###############
    ### panel 1 ###
    ###############
    ax = axs[1]

    im2 = ax.plot(dz_lowTopo_v,'-',color='b',linewidth=2.0,zorder=0)
    im2 = ax.plot(dz_highTopo_v,'-',color='r',linewidth=2.0,zorder=0)
    ax.set_ylabel(r"$\Delta z$ $[$m$]$",fontsize=fntSize)
    ax.set_xlabel(r"Vertical index $[$-$]$",fontsize=fntSize)

    fig.tight_layout()

    # save figure
    CCC = FE_plot
    print(CCC)
    plt.savefig(CCC,dpi=300,bbox_inches = "tight")
    plt.close(fig)

    end_code = time.perf_counter()
    print('done creating dataset plot, elapsed time ' + str(end_code-start_code) + ' s')
