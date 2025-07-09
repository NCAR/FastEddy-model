import numpy as np
import numpy.matlib
import xarray as xr
import csv
#import argparse
import json
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import time
import matplotlib
from couplingUtils import *

#######################################
### read parameters from .json file ###
#######################################

args = parse_args()
with open(args.file) as file:
    params = json.loads(file.read())
#
name_dom = params["name_dom"]
gis_root = params["gis_root"]
gis_file = params["gis_file"]
nlcd_name = params["nlcd_name"]
water_cats = params["water_cats"]
urban_opt = params["urban_opt"]

FE_dataset_path = params["FE_dataset_path"]
gis_opt = params["gis_opt"]
name_dom_add = params["name_dom_add"]
save_plot_opt = params["save_plot_opt"]

#######################################

# derived paths

file_nlcd = gis_root + nlcd_name
FE_new_nc = FE_dataset_path + name_dom + name_dom_add + '.nc'
FE_plot = FE_dataset_path + name_dom + name_dom_add + '_geospec.png'
print('FE_new_nc:', FE_new_nc)

# Calculate xPos2d, yPos2d

start_code = time.perf_counter()

ds_GIS = xr.open_dataset(gis_root+gis_file)
if (gis_opt==0):
    dx = ds_GIS.cellsize.values
    dy = dx
    Nx = ds_GIS.sizes['x']
    Ny = ds_GIS.sizes['y']
elif (gis_opt==1):
    dx = ds_GIS.attrs['DX']
    dy = ds_GIS.attrs['DY']
    Nx = ds_GIS.sizes['west_east']
    Ny = ds_GIS.sizes['south_north']
print('dx,dy=',dx,',',dy,'(m)')
print('Nx,Ny=',Nx,',',Ny)

xarr = np.zeros([Ny,Nx],dtype=np.float32)
yarr = np.zeros([Ny,Nx],dtype=np.float32)
xarr_1d = np.arange(0.5*dx,Nx*dx,dx)
yarr_1d = np.arange(0.5*dy,Ny*dy,dy)
xarr[:,:] = np.matlib.repmat(xarr_1d,Ny,1)
yarr[:,:] = np.transpose(np.matlib.repmat(yarr_1d,Nx,1))

# lat/lon info

vars_gis_ref_v = ['lat','lon','data_topo0','data_land']

if (gis_opt==0):
    vars_gis_v = ['lat','lon','elevation','LandCover']
elif (gis_opt==1):
    vars_gis_v = ['XLAT','XLONG','HGT','LU_INDEX']

vv = 0
for var in vars_gis_ref_v:
    var_name = vars_gis_v[vv]
    if (gis_opt==0):
        line_vv = var + ' = ds_GIS.' + var_name + '.values'
    elif (gis_opt==1):
        line_vv = var + ' = ds_GIS.' + var_name + '.isel(Time=0).values'
    exec(line_vv)
    vv = vv + 1

print('topoPos.shape=',data_topo0.shape)
print('min(data_topo0)=',np.min(data_topo0,axis=(1,0)))
print('max(data_topo0)=',np.max(data_topo0,axis=(1,0)))

# roughness length & sea mask

if (gis_opt==0):
    nlcd_class = []
    nlcd_z0 = []

    with open(file_nlcd) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            print('row[',line_count,']=',row)
            if line_count > 0:
                nlcd_class.append(int(row[0]))
                nlcd_z0.append(float(row[2]))

            line_count = line_count + 1

    z0_tmp = np.zeros([Ny,Nx],dtype=np.float32)
    for cc in range(0,len(nlcd_class)):
            class_tmp = nlcd_class[cc]
            print('cc,class_tmp=',cc,class_tmp)
            ind_cc = np.where(data_land==class_tmp)
            z0_tmp[ind_cc] = nlcd_z0[cc]

    SeaMask_tmp =np.zeros([Ny,Nx],dtype=np.float32)
    ind_water = np.isin(data_land,water_cats)
    SeaMask_tmp[ind_water] = 1.0
elif (gis_opt==1):
    z0_tmp = ds_GIS.ZNT.isel(Time=0).values
    SeaMask_tmp = ds_GIS.LANDMASK.isel(Time=0).values
    ind_land_wrf = np.where(SeaMask_tmp==1.0)
    ind_sea_wrf = np.where(SeaMask_tmp==0.0)
    SeaMask_tmp[ind_land_wrf] = 0.0
    SeaMask_tmp[ind_sea_wrf] = 1.0

# Building height information

if (gis_opt==0 and urban_opt==1):

    bdg_height = ds_GIS.BuildingHeights.values
    ind_missing = np.where(bdg_height==-9999)
    bdg_height[ind_missing] = 0.0

# Save to netcdf file

ds_data = xr.Dataset()

ds_data['xPos2d']= xr.DataArray(xarr,dims=(['yIndex','xIndex']))
ds_data['yPos2d']= xr.DataArray(yarr,dims=(['yIndex','xIndex']))
ds_data['topoPos']= xr.DataArray(data_topo0,dims=(['yIndex','xIndex']))
ds_data['LandCover']= xr.DataArray(data_land.astype(dtype=np.int32),dims=(['yIndex','xIndex']))
ds_data['z0m']= xr.DataArray(z0_tmp,dims=(['yIndex','xIndex']))
ds_data['z0t']= xr.DataArray(0.1*z0_tmp,dims=(['yIndex','xIndex']))
ds_data['SeaMask']= xr.DataArray(SeaMask_tmp,dims=(['yIndex','xIndex']))
ds_data['dx_inter']= xr.DataArray(np.array(dx,dtype=np.float32))
ds_data['dy_inter']= xr.DataArray(np.array(dy,dtype=np.float32))
if (gis_opt==0 and urban_opt==1):
    ds_data['BuildingHeights']= xr.DataArray(bdg_height,dims=(['yIndex','xIndex']))
ds_data['lat']= xr.DataArray(lat.astype(dtype=np.float64),dims=(['yIndex','xIndex']))
ds_data['lon']= xr.DataArray(lon.astype(dtype=np.float64),dims=(['yIndex','xIndex']))

ds_data.to_netcdf(FE_new_nc,format='NETCDF4')

end_code = time.perf_counter()
print('done creating GIS dataset, elapsed time ' + str(end_code-start_code) + ' s')

# Plot

if (save_plot_opt==1):

    start_code = time.perf_counter()

    _nlcd_colors_1 = [[1.0, 1.0, 1.0],
                         [0.27843137, 0.41960784, 0.67058824],
                         [0.81960784, 0.87058824, 0.98039216],
                         [0.87058824, 0.78823529, 0.78823529],
                         [0.85098039, 0.58039216, 0.50980392],
                         [0.92941176, 0.        , 0.        ],
                         [0.67058824, 0.        , 0.        ],
                         [0.70196078, 0.67843137, 0.63921569],
                         [0.41176471, 0.67058824, 0.38823529],
                         [0.10980392, 0.38823529, 0.18823529],
                         [0.70980392, 0.78823529, 0.56078431],
                         [0.8       , 0.72941176, 0.49019608],
                         [0.89019608, 0.89019608, 0.76078431],
                         [0.85882353, 0.85098039, 0.23921569],
                         [0.67058824, 0.43921569, 0.16078431],
                         [0.72941176, 0.85098039, 0.92156863],
                         [0.43921569, 0.63921569, 0.72941176]]

    nlcd_color_map = ListedColormap(_nlcd_colors_1, name='nlcd_color_map')
    matplotlib.colormaps.register(nlcd_color_map, name='nlcd_color_map', force=False)

    fntSize = 14.0
    fntSize_labels = 14.0
    fntSize_title = 16.0
    plt.rcParams['xtick.labelsize']=fntSize
    plt.rcParams['ytick.labelsize']=fntSize
    plt.rcParams['axes.linewidth']=2.0

    numPlotsX=2
    numPlotsY=2
    fig,axs = plt.subplots(numPlotsX,numPlotsY,sharey=False,sharex=False,figsize=(15,12))

    ### terrain elevation  ###
    ax=axs[0,0]

    data_topo_plot = np.zeros(data_topo0.shape)
    data_topo_plot[:,:] = data_topo0[:,:]
    ind_zero = np.where(data_topo_plot==0.0)
    data_topo_plot[ind_zero] = np.nan

    im = ax.pcolormesh(xarr/1e3,yarr/1e3,data_topo_plot,cmap='terrain',linewidth=0,rasterized=True,shading='nearest')
    cbar=fig.colorbar(im, ax=ax, orientation='vertical')
    ax.set_aspect('equal', 'box')
    ax.set_ylabel(r'$y$ $[\mathrm{km}]$',fontsize=fntSize_labels)
    ax.set_xlabel(r'$x$ $[\mathrm{km}]$',fontsize=fntSize_labels)
    ax.set_title('terrain elevation [m ASL]',fontsize=fntSize_title)

    ### land cover ###

    ax=axs[0,1]

    data_land_plot = np.zeros(data_land.shape)
    data_land_plot[:,:] = data_land[:,:]

    im = ax.pcolormesh(xarr/1e3,yarr/1e3,data_land_plot,cmap='nlcd_color_map',linewidth=0,rasterized=True,shading='nearest')
    cbar=fig.colorbar(im, ax=ax, orientation='vertical')
    ax.set_aspect('equal', 'box')
    ax.set_ylabel(r'$y$ $[\mathrm{km}]$',fontsize=fntSize_labels)
    ax.set_xlabel(r'$x$ $[\mathrm{km}]$',fontsize=fntSize_labels)
    ax.set_title('land cover [-]',fontsize=fntSize_title)

    ### roughness length z0m ###

    ax=axs[1,0]

    im = ax.pcolormesh(xarr/1e3,yarr/1e3,z0_tmp,cmap='terrain',linewidth=0,rasterized=True,shading='nearest')
    cbar=fig.colorbar(im, ax=ax, orientation='vertical')
    ax.set_aspect('equal', 'box')
    ax.set_ylabel(r'$y$ $[\mathrm{km}]$',fontsize=fntSize_labels)
    ax.set_xlabel(r'$x$ $[\mathrm{km}]$',fontsize=fntSize_labels)
    ax.set_title('roughness length, z0m [m]',fontsize=fntSize_title)

    ## SeaMask ###
    ax=axs[1,1]

    im = ax.pcolormesh(xarr/1e3,yarr/1e3,SeaMask_tmp,cmap='bwr_r',linewidth=0,rasterized=True,shading='nearest')
    cbar=fig.colorbar(im, ax=ax, orientation='vertical')
    ax.set_aspect('equal', 'box')
    ax.set_ylabel(r'$y$ $[\mathrm{km}]$',fontsize=fntSize_labels)
    ax.set_xlabel(r'$x$ $[\mathrm{km}]$',fontsize=fntSize_labels)
    ax.set_title('sea mask [-]',fontsize=fntSize_title)

    # save figure
    CCC = FE_plot
    print(CCC)
    plt.savefig(CCC,dpi=300,bbox_inches = "tight")
    plt.close(fig)

    end_code = time.perf_counter()
    print('done creating dataset plot, elapsed time ' + str(end_code-start_code) + ' s')
