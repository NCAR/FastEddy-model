import os, sys
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import math
from scipy.stats import skew
from scipy.stats import kurtosis
import matplotlib.colors as mcolors
import scipy.fftpack as fftpack
from scipy import interpolate
from math import log10, floor

def check_imports():
   import pkg_resources
   
   imports = list(set(get_imports()))

   # The only way I found to get the version of the root package
   # from only the name of the package is to cross-check the names 
   # of installed packages vs. imported packages
   requirements = []
   for m in pkg_resources.working_set:
    if m.project_name in imports and m.project_name!="pip":
        requirements.append((m.project_name, m.version))

   for r in requirements:
    print("{}=={}".format(*r))

def get_imports():
    import types
    for name, val in globals().items():
        if isinstance(val, types.ModuleType):
            # Split ensures you get root package, 
            # not just imported function
            name = val.__name__.split(".")[0]

        elif isinstance(val, type):
            name = val.__module__.split(".")[0]

        # Some packages are weird and have different
        # imported names vs. system/pip names. Unfortunately,
        # there is no systematic way to get pip names from
        # a package's imported name. You'll have to had
        # exceptions to this list manually!
        poorly_named_packages = {
            "PIL": "Pillow",
            "sklearn": "scikit-learn"
        }
        if name in poorly_named_packages.keys():
            name = poorly_named_packages[name]

        yield name

def compute_mean_profiles(FE_xr):

    var_list = list(FE_xr.data_vars.keys())
    moist_flag = 0
    if 'qv' in var_list:
        moist_flag = 1
    
    u_3d = np.squeeze(FE_xr.u.isel(time=0).values)
    v_3d = np.squeeze(FE_xr.v.isel(time=0).values)
    th_3d = np.squeeze(FE_xr.theta.isel(time=0).values)
    z_3d = np.squeeze(FE_xr.zPos.isel(time=0).values)
    if (moist_flag==1):
        qv_3d = np.squeeze(FE_xr.qv.isel(time=0).values)

    Nz = u_3d.shape[0]
    wd_1d = np.zeros([Nz])

    u_1d = np.mean(np.mean(u_3d,axis=2),axis=1)
    v_1d = np.mean(np.mean(v_3d,axis=2),axis=1)
    th_1d = np.mean(np.mean(th_3d,axis=2),axis=1)
    z_1d = np.mean(np.mean(z_3d,axis=2),axis=1)
    if (moist_flag==1):
        qv_1d = np.mean(np.mean(qv_3d,axis=2),axis=1)

    for kk in range(0,Nz):
        wd_tmp = math.atan2(-u_1d[kk],-v_1d[kk]) * 180.0 / np.pi
        if (wd_tmp<0.0):
            wd_tmp = 180.0 + (180.0 + wd_tmp)
        wd_1d[kk] = wd_tmp

    array_out = np.zeros([Nz,6])
    array_out[:,0] = z_1d
    array_out[:,1] = u_1d
    array_out[:,2] = v_1d
    array_out[:,3] = wd_1d
    array_out[:,4] = th_1d
    if (moist_flag==1):
        array_out[:,5] = qv_1d
    
    return array_out

def plot_XY_UVWTHETA(case, case_open, zChoose, save_plot_opt, path_figure, plot_qv_cont):

    zVect = case_open.zPos.isel(time=0,xIndex=0,yIndex=0).values
    z_diff = np.abs(zVect - zChoose)
    zgp = np.where(z_diff==np.amin(z_diff))
    zgp = zgp[0]
    zChoose = zgp[0]
    
    ufield = case_open.u.isel(time=0).values
    vfield = case_open.v.isel(time=0).values
    wfield = case_open.w.isel(time=0).values
    thetafield = case_open.theta.isel(time=0).values
    xPos = case_open.xPos.isel(time=0,zIndex=zChoose).values
    yPos = case_open.yPos.isel(time=0,zIndex=zChoose).values
    zPos = case_open.zPos.isel(time=0,zIndex=zChoose).values
    if (plot_qv_cont==1):
        wsfield = np.sqrt(np.power(ufield,2.0)+np.power(vfield,2.0))
        ws_min = np.amin(np.amin(wsfield[zChoose,:,:]))
        ws_max = np.amax(np.amax(wsfield[zChoose,:,:]))
        qvfield = case_open.qv.isel(time=0).values
        qv_min = np.amin(np.amin(qvfield[zChoose,:,:]))
        qv_max = np.amax(np.amax(qvfield[zChoose,:,:]))
        
    u_min = np.amin(np.amin(ufield[zChoose,:,:]))
    u_max = np.amax(np.amax(ufield[zChoose,:,:]))
    v_min = np.amin(np.amin(vfield[zChoose,:,:]))
    v_max = np.amax(np.amax(vfield[zChoose,:,:]))
    w_min = np.amin(np.amin(wfield[zChoose,:,:]))
    w_max = np.amax(np.amax(wfield[zChoose,:,:]))
    t_min = np.amin(np.amin(thetafield[zChoose,:,:]))
    t_max = np.amax(np.amax(thetafield[zChoose,:,:]))
    
    fig_name = "UVWTHETA-XY-"+case+".png"
    colormap1 = 'viridis'
    colormap2 = 'seismic'
    colormap3 = 'afmhot_r' # 'YlOrRd'
    colormap4 = 'summer'
    if (plot_qv_cont==1):
        FE_legend = [r'$U$ [m/s] at z='+str(np.amax(zPos))+' m',r'$\theta$ [K] at z='+str(np.amax(zPos))+' m',r'$w$ [m/s] at z='+str(np.amax(zPos))+' m', r'$q_v$ [g/kg] at z='+str(np.amax(zPos))+' m']  
    else:
        FE_legend = [r'$u$ [m/s] at z='+str(np.amax(zPos))+' m',r'$v$ [m/s] at z='+str(np.amax(zPos))+' m',r'$w$ [m/s] at z='+str(np.amax(zPos))+' m', r'$\theta$ [K] at z='+str(np.amax(zPos))+' m']
    
    fntSize=20
    fntSize_title=22
    plt.rcParams['xtick.labelsize']=fntSize
    plt.rcParams['ytick.labelsize']=fntSize
    plt.rcParams['axes.linewidth']=2.0
    plt.rcParams['pcolor.shading']='auto'

    numPlotsX=2
    numPlotsY=2
    fig,axs = plt.subplots(numPlotsX,numPlotsY,sharey=False,sharex=False,figsize=(26,20))

    ###############
    ### U plot ###
    ###############
    ax=axs[0][0]
    if (plot_qv_cont==1):
        im = ax.pcolormesh(xPos/1e3,yPos/1e3,wsfield[zChoose,:,:],cmap=colormap1,linewidth=0,rasterized=True,vmin=ws_min,vmax=ws_max)
    else:
        im = ax.pcolormesh(xPos/1e3,yPos/1e3,ufield[zChoose,:,:],cmap=colormap1,linewidth=0,rasterized=True,vmin=u_min,vmax=u_max)
    ax.set_ylabel(r'$y$ $[\mathrm{km}]$',fontsize=fntSize)
    ax.set_xlabel(r'$x$ $[\mathrm{km}]$',fontsize=fntSize)
    cbar=fig.colorbar(im, ax=ax)

    title_fig_0 = FE_legend[0]
    ax.set_title(title_fig_0,fontsize=fntSize)

    ###############
    ### V plot ###
    ###############
    ax=axs[1][0]
    if (plot_qv_cont==1):
        im = ax.pcolormesh(xPos/1e3,yPos/1e3,thetafield[zChoose,:,:],cmap=colormap3,linewidth=0,rasterized=True,vmin=t_min,vmax=t_max)
    else:
        im = ax.pcolormesh(xPos/1e3,yPos/1e3,vfield[zChoose,:,:],cmap=colormap1,linewidth=0,rasterized=True,vmin=v_min,vmax=v_max)
    ax.set_ylabel(r'$y$ $[\mathrm{km}]$',fontsize=fntSize)
    ax.set_xlabel(r'$x$ $[\mathrm{km}]$',fontsize=fntSize)
    cbar=fig.colorbar(im, ax=ax)

    title_fig_1 = FE_legend[1]
    ax.set_title(title_fig_1,fontsize=fntSize)

    ###############
    ### W plot ###
    ###############
    w_min=-1.0*w_max
    ax=axs[0][1]
    im = ax.pcolormesh(xPos/1e3,yPos/1e3,wfield[zChoose,:,:],cmap=colormap2,linewidth=0,rasterized=True,vmin=w_min,vmax=w_max)
    ax.set_ylabel(r'$y$ $[\mathrm{km}]$',fontsize=fntSize)
    ax.set_xlabel(r'$x$ $[\mathrm{km}]$',fontsize=fntSize)
    cbar=fig.colorbar(im, ax=ax)

    title_fig_2 = FE_legend[2]
    ax.set_title(title_fig_2,fontsize=fntSize)
    
    ###############
    ### THETA plot ###
    ###############
    ax=axs[1][1]
    if (plot_qv_cont==1):
        im = ax.pcolormesh(xPos/1e3,yPos/1e3,qvfield[zChoose,:,:],cmap=colormap4,linewidth=0,rasterized=True,vmin=qv_min,vmax=qv_max)
    else:    
        im = ax.pcolormesh(xPos/1e3,yPos/1e3,thetafield[zChoose,:,:],cmap=colormap3,linewidth=0,rasterized=True,vmin=t_min,vmax=t_max)
    ax.set_ylabel(r'$y$ $[\mathrm{km}]$',fontsize=fntSize)
    ax.set_xlabel(r'$x$ $[\mathrm{km}]$',fontsize=fntSize)
    cbar=fig.colorbar(im, ax=ax)

    title_fig_3 = FE_legend[3]
    ax.set_title(title_fig_3,fontsize=fntSize)

    if (save_plot_opt==1):
        print(path_figure + fig_name)
        plt.savefig(path_figure + fig_name,dpi=300,bbox_inches = "tight")

def plot_XZ_UVWTHETA(case, case_open, z_max, sizeX_XZ, sizeY_XZ, save_plot_opt, path_figure, zline_opt, zline_val, plot_qv_cont):
    
    ufield = case_open.u.isel(time=0).values
    vfield = case_open.v.isel(time=0).values
    wfield = case_open.w.isel(time=0).values
    thetafield = case_open.theta.isel(time=0).values
    
    [Npz,Npy,Npx] = ufield.shape
    yChoose = int(Npy/2)
    
    xPos = case_open.xPos.isel(time=0,yIndex=yChoose).values
    yPos = case_open.yPos.isel(time=0,yIndex=yChoose).values
    zPos = case_open.zPos.isel(time=0,yIndex=yChoose).values

    zPos_1d = zPos[:,0]
    diff_z = np.abs(zPos_1d - z_max)
    ind_zmax = np.where(diff_z==np.min(diff_z))[0][0]
            
    u_min = np.amin(np.amin(ufield[0:ind_zmax,yChoose,:]))
    u_max = np.amax(np.amax(ufield[0:ind_zmax,yChoose,:]))
    v_min = np.amin(np.amin(vfield[0:ind_zmax,yChoose,:]))
    v_max = np.amax(np.amax(vfield[0:ind_zmax,yChoose,:]))
    w_max = np.amax(np.amax(np.abs(wfield[0:ind_zmax,yChoose,:])))
    w_min = -w_max
    t_min = np.amin(np.amin(thetafield[0:ind_zmax,yChoose,:]))
    t_max = np.amax(np.amax(thetafield[0:ind_zmax,yChoose,:]))

    if (plot_qv_cont==1):
        wsfield = np.sqrt(np.power(ufield,2.0)+np.power(vfield,2.0))
        ws_min = np.amin(np.amin(wsfield[0:ind_zmax,yChoose,:]))
        ws_max = np.amax(np.amax(wsfield[0:ind_zmax,yChoose,:]))
        qvfield = case_open.qv.isel(time=0).values
        qv_min = np.amin(np.amin(qvfield[0:ind_zmax,yChoose,:]))
        qv_max = np.amax(np.amax(qvfield[0:ind_zmax,yChoose,:]))

    fig_name = "UVWTHETA-XZ-"+case+".png"
    colormap1 = 'viridis'
    colormap2 = 'seismic'
    colormap3 = 'afmhot_r' # 'nipy_spectral' # 'gist_ncar' # 'afmhot' # 'YlOrRd'
    colormap4 = 'summer'

    if (plot_qv_cont==1):
        FE_legend = [r'$U$ [m/s] at y='+str(yPos[0,yChoose]/1e3)+' km',r'$w$ [m/s] at y='+str(yPos[0,yChoose]/1e3)+' km', r'$\theta$ [K] at y='+str(yPos[0,yChoose]/1e3)+' km', r'$q_v$ [g/kg] at y='+str(yPos[0,yChoose]/1e3)+' km']  
    else:
        FE_legend = [r'$u$ [m/s] at y='+str(yPos[0,yChoose]/1e3)+' km',r'$v$ [m/s] at y='+str(yPos[0,yChoose]/1e3)+' km',r'$w$ [m/s] at y='+str(yPos[0,yChoose]/1e3)+' km', r'$\theta$ [K] at y='+str(yPos[0,yChoose]/1e3)+' km']
    
    numPlotsX=4
    numPlotsY=1
    fntSize=20
    fntSize_title=22
    plt.rcParams['xtick.labelsize']=fntSize
    plt.rcParams['ytick.labelsize']=fntSize
    plt.rcParams['axes.linewidth']=2.0
    plt.rcParams['pcolor.shading']='auto'
    
    fig,axs = plt.subplots(numPlotsX,numPlotsY,sharey=False,sharex=False,figsize=(sizeX_XZ,sizeY_XZ))

    ###############
    ### U plot ###
    ###############
    ax=axs[0]
    if (plot_qv_cont==1):
        im = ax.pcolormesh(xPos/1e3,zPos/1e3,wsfield[:,yChoose,:],cmap=colormap1,linewidth=0,rasterized=True,vmin=u_min,vmax=u_max)
    else:
        im = ax.pcolormesh(xPos/1e3,zPos/1e3,ufield[:,yChoose,:],cmap=colormap1,linewidth=0,rasterized=True,vmin=u_min,vmax=u_max)
    if (zline_opt==1):
        im2 = ax.plot([xPos[0,0]/1e3, xPos[0,xPos.shape[1]-1]/1e3],[zline_val/1e3, zline_val/1e3],'--',color='white',linewidth=2.5)
    ax.set_xticklabels([])
    ax.set_ylabel(r'$z$ $[\mathrm{km}]$',fontsize=fntSize)
    cbar=fig.colorbar(im, ax=ax)
    ax.set_ylim(0.0,z_max/1e3)

    title_fig_0 = FE_legend[0]
    ax.set_title(title_fig_0,fontsize=fntSize)

    ###############
    ### V plot ###
    ###############
    ax=axs[1]
    if (plot_qv_cont==1):
        im = ax.pcolormesh(xPos/1e3,zPos/1e3,wfield[:,yChoose,:],cmap=colormap2,linewidth=0,rasterized=True,vmin=w_min,vmax=w_max)
    else:
        im = ax.pcolormesh(xPos/1e3,zPos/1e3,vfield[:,yChoose,:],cmap=colormap1,linewidth=0,rasterized=True,vmin=v_min,vmax=v_max)
    if (zline_opt==1):
        im2 = ax.plot([xPos[0,0]/1e3, xPos[0,xPos.shape[1]-1]/1e3],[zline_val/1e3, zline_val/1e3],'--',color='white',linewidth=2.5)
    ax.set_xticklabels([])
    ax.set_ylabel(r'$z$ $[\mathrm{km}]$',fontsize=fntSize)
    cbar=fig.colorbar(im, ax=ax)
    ax.set_ylim(0.0,z_max/1e3)

    title_fig_1 = FE_legend[1]
    ax.set_title(title_fig_1,fontsize=fntSize)

    ###############
    ### W plot ###
    ###############
    ax=axs[2]
    if (plot_qv_cont==1):
        im = ax.pcolormesh(xPos/1e3,zPos/1e3,thetafield[:,yChoose,:],cmap=colormap3,linewidth=0,rasterized=True,vmin=t_min,vmax=t_max)
    else:
        im = ax.pcolormesh(xPos/1e3,zPos/1e3,wfield[:,yChoose,:],cmap=colormap2,linewidth=0,rasterized=True,vmin=w_min,vmax=w_max)
    if (zline_opt==1):
        im2 = ax.plot([xPos[0,0]/1e3, xPos[0,xPos.shape[1]-1]/1e3],[zline_val/1e3, zline_val/1e3],'--',color='black',linewidth=2.5)
    ax.set_xticklabels([])
    ax.set_ylabel(r'$z$ $[\mathrm{km}]$',fontsize=fntSize)
    cbar=fig.colorbar(im, ax=ax)
    ax.set_ylim(0.0,z_max/1e3)
    
    title_fig_2 = FE_legend[2]
    ax.set_title(title_fig_2,fontsize=fntSize)
    
    ###############
    ### THETA plot ###
    ###############
    ax=axs[3]
    if (plot_qv_cont==1):
        im = ax.pcolormesh(xPos/1e3,zPos/1e3,qvfield[:,yChoose,:],cmap=colormap4,linewidth=0,rasterized=True,vmin=qv_min,vmax=qv_max)
    else:
        im = ax.pcolormesh(xPos/1e3,zPos/1e3,thetafield[:,yChoose,:],cmap=colormap3,linewidth=0,rasterized=True,vmin=t_min,vmax=t_max)
    if (zline_opt==1):
        im2 = ax.plot([xPos[0,0]/1e3, xPos[0,xPos.shape[1]-1]/1e3],[zline_val/1e3, zline_val/1e3],'--',color='black',linewidth=2.5)
    ax.set_ylabel(r'$z$ $[\mathrm{km}]$',fontsize=fntSize)
    ax.set_xlabel(r'$x$ $[\mathrm{km}]$',fontsize=fntSize)
    cbar=fig.colorbar(im, ax=ax)
    ax.set_ylim(0.0,z_max/1e3)

    title_fig_3 = FE_legend[3]
    ax.set_title(title_fig_3,fontsize=fntSize)

    # save figure
    if (save_plot_opt==1):
        print(path_figure + fig_name)
        plt.savefig(path_figure + fig_name,dpi=300,bbox_inches = "tight")
        
          
def plot_figureConfigure(numPlotsX,numPlotsY,sizeX,sizeY,styleFile='./feplot.mplstyle'):
    plt.style.use(styleFile) #Must issue the style.use before creating a figure handle
    fig,axs = plt.subplots(numPlotsX,numPlotsY,sharey=True,sharex=False,figsize=(sizeX,sizeY))
  
    return fig,axs

def plot_mean_profiles(fig, axs, FE_mean, z_max, caseLabel, save_plot_opt, path_figure, caseCnt, moist_flag):

    colores_v = []
    colores_v.append('darkblue')
    colores_v.append('darkred')
    colores_v.append('dodgerblue')
    colores_v.append('orangered')

    lineas_v = []
    lineas_v.append('-')
    lineas_v.append('--')
    lineas_v.append('-.')
    lineas_v.append('.')
    
    fntSize=20
    fntSize_title=22
    fntSize_legend=16
    plt.rcParams['xtick.labelsize']=fntSize
    plt.rcParams['ytick.labelsize']=fntSize
    plt.rcParams['axes.linewidth']=2.0
  
    y_min = 0.0
    y_max = z_max
    zPos = FE_mean[:,0]/1e3
          
    fig_name = "MEAN-PROF-"+caseLabel+".png"
    
    ###############
    ### panel 0 ###
    ###############
    ax = axs[0]
    im = ax.plot(np.sqrt(FE_mean[:,1]**2+FE_mean[:,2]**2),zPos,lineas_v[caseCnt],color=colores_v[caseCnt],label=caseLabel)
    ax.set_xlabel(r"$U$ $[$m s$^{-1}]$",fontsize=fntSize_legend)
    ax.set_ylabel(r"$z$ $[$km$]$",fontsize=fntSize)
    ax.set_ylim(0.0,z_max/1e3)
    ax.legend(loc=2,edgecolor='white')
    
    ###############
    ### panel 1 ###
    ###############
    ax = axs[1]
    im = ax.plot(FE_mean[:,3],zPos,lineas_v[caseCnt],color=colores_v[caseCnt],label=caseLabel)
    ax.set_xlabel(r"$\phi$ $[^{\circ}]$",fontsize=fntSize)
    ax.set_ylim(0.0,z_max/1e3)
    
    ###############
    ### panel 2 ###
    ###############
    ax = axs[2]
    im = ax.plot(FE_mean[:,4],zPos,lineas_v[caseCnt],color=colores_v[caseCnt],label=caseLabel)
    ax.set_xlabel(r"$\theta$ [K]",fontsize=fntSize)
    ax.set_ylim(0.0,z_max/1e3)

    if (moist_flag==1):
        ###############
        ### panel 3 ###
        ###############
        ax = axs[3]
        im = ax.plot(FE_mean[:,5],zPos,lineas_v[caseCnt],color=colores_v[caseCnt],label=caseLabel)
        ax.set_xlabel(r"$q_v$ [g kg$^{-1}$]",fontsize=fntSize)
        ax.set_ylim(0.0,z_max/1e3)
    
    if (save_plot_opt):
        print(path_figure + fig_name)
        plt.savefig(path_figure + fig_name,dpi=300,bbox_inches = "tight")

def compute_turb_profiles(FE_xr, array_out, flag_taus):

    var_list = list(FE_xr.data_vars.keys())
    tke1_flag = 0
    if 'TKE_1' in var_list:
        tke1_flag = 1
    moist_flag = 0
    if 'qv' in var_list:
        moist_flag = 1
    
    u_3d = np.squeeze(FE_xr.u.isel(time=0).values)
    v_3d = np.squeeze(FE_xr.v.isel(time=0).values)
    w_3d = np.squeeze(FE_xr.w.isel(time=0).values)
    th_3d = np.squeeze(FE_xr.theta.isel(time=0).values)
    z_3d = np.squeeze(FE_xr.zPos.isel(time=0).values)
    if (flag_taus==1):
        tau13_3d = np.squeeze(FE_xr.Tau31.isel(time=0).values)
        tau23_3d = np.squeeze(FE_xr.Tau32.isel(time=0).values)
        tau33_3d = np.squeeze(FE_xr.Tau33.isel(time=0).values)
        tauTH3_3d = np.squeeze(FE_xr.TauTH3.isel(time=0).values)
        if (moist_flag==1):
            tauQV3_3d = np.squeeze(FE_xr.TauQv3.isel(time=0).values)
    sgstke_3d = np.squeeze(FE_xr.TKE_0.isel(time=0).values)
    if (tke1_flag==1):
        sgstke1_3d = np.squeeze(FE_xr.TKE_1.isel(time=0).values)
    if (moist_flag==1):
        qv_3d = np.squeeze(FE_xr.qv.isel(time=0).values)

    Nz = u_3d.shape[0]
    Ny = u_3d.shape[1]
    Nx = u_3d.shape[2]

    u_1d = np.mean(np.mean(u_3d,axis=2),axis=1)
    v_1d = np.mean(np.mean(v_3d,axis=2),axis=1)
    w_1d = np.mean(np.mean(w_3d,axis=2),axis=1)
    th_1d = np.mean(np.mean(th_3d,axis=2),axis=1)
    z_1d = np.mean(np.mean(z_3d,axis=2),axis=1)
    if (moist_flag==1):
        qv_1d = np.mean(np.mean(qv_3d,axis=2),axis=1)
    
    u_2d_mean = np.tile(u_1d,[Nx,1])
    u_3d_mean = np.tile(u_2d_mean,[Ny,1,1])
    u_3d_mean = np.swapaxes(u_3d_mean,0,2)
    u_3d_mean = np.swapaxes(u_3d_mean,1,2)
    
    v_2d_mean = np.tile(v_1d,[Nx,1])
    v_3d_mean = np.tile(v_2d_mean,[Ny,1,1])
    v_3d_mean = np.swapaxes(v_3d_mean,0,2)
    v_3d_mean = np.swapaxes(v_3d_mean,1,2)
    
    w_2d_mean = np.tile(w_1d,[Nx,1])
    w_3d_mean = np.tile(w_2d_mean,[Ny,1,1])
    w_3d_mean = np.swapaxes(w_3d_mean,0,2)
    w_3d_mean = np.swapaxes(w_3d_mean,1,2)
    
    th_2d_mean = np.tile(th_1d,[Nx,1])
    th_3d_mean = np.tile(th_2d_mean,[Ny,1,1])
    th_3d_mean = np.swapaxes(th_3d_mean,0,2)
    th_3d_mean = np.swapaxes(th_3d_mean,1,2)

    if (moist_flag==1):
        qv_2d_mean = np.tile(qv_1d,[Nx,1])
        qv_3d_mean = np.tile(qv_2d_mean,[Ny,1,1])
        qv_3d_mean = np.swapaxes(qv_3d_mean,0,2)
        qv_3d_mean = np.swapaxes(qv_3d_mean,1,2)
    
    up = u_3d - u_3d_mean
    vp = v_3d - v_3d_mean
    wp = w_3d - w_3d_mean
    thp = th_3d - th_3d_mean
    if (moist_flag==1):
        qvp = qv_3d - qv_3d_mean
    
    upup = up*up
    upwp = up*wp
    vpvp = vp*vp
    vpwp = vp*wp
    wpwp = wp*wp
    thpwp = thp*wp
    tke = 0.5*(upup+vpvp+wpwp)
    if (moist_flag==1):
        qvpwp = qvp*wp
    
    upup_1d = np.mean(np.mean(upup,axis=2),axis=1)
    upwp_1d = np.mean(np.mean(upwp,axis=2),axis=1)
    vpvp_1d = np.mean(np.mean(vpvp,axis=2),axis=1)
    vpwp_1d = np.mean(np.mean(vpwp,axis=2),axis=1)
    if (flag_taus==1):
        tau13_1d = np.mean(np.mean(tau13_3d,axis=2),axis=1)
        tau23_1d = np.mean(np.mean(tau23_3d,axis=2),axis=1)
        tau33_1d = np.mean(np.mean(tau33_3d,axis=2),axis=1)
        tauTH3_1d = np.mean(np.mean(tauTH3_3d,axis=2),axis=1)
        if (moist_flag==1):
            tauQV3_1d = np.mean(np.mean(tauQV3_3d,axis=2),axis=1)
    sgstke_1d = np.mean(np.mean(sgstke_3d,axis=2),axis=1)
    if (tke1_flag==1):
        sgstke1_1d = np.mean(np.mean(sgstke1_3d,axis=2),axis=1)
    
    Upwp_1d = np.sqrt(np.power(upwp_1d,2.0)+np.power(vpwp_1d,2.0))
    wpwp_1d = np.mean(np.mean(wpwp,axis=2),axis=1)
    tke_1d = np.mean(np.mean(tke,axis=2),axis=1)
    thpwp_1d = np.mean(np.mean(thpwp,axis=2),axis=1)
    if (flag_taus==1):
        tau1323_1d = np.sqrt(np.power(tau13_1d,2.0)+np.power(tau23_1d,2.0))
    if (moist_flag==1):
        qvpwp_1d = np.mean(np.mean(qvpwp,axis=2),axis=1)

    array_out = np.zeros([Nz,16])
    
    array_out[:,0] = z_1d
    array_out[:,1] = upup_1d
    array_out[:,2] = upwp_1d
    array_out[:,3] = vpvp_1d
    array_out[:,4] = vpwp_1d
    array_out[:,5] = Upwp_1d
    array_out[:,6] = wpwp_1d
    array_out[:,7] = tke_1d
    array_out[:,8] = thpwp_1d
    array_out[:,9] = sgstke_1d
    if (tke1_flag==1):
        array_out[:,10] = sgstke1_1d
    if (flag_taus==1):
        array_out[0:Nz-1,11] = 0.5*(tau1323_1d[0:Nz-1]+tau1323_1d[1:Nz])
        array_out[0:Nz-1,12] = 0.5*(tau33_1d[0:Nz-1]+tau33_1d[1:Nz])
        array_out[0:Nz-1,13] = 0.5*(tauTH3_1d[0:Nz-1]+tauTH3_1d[1:Nz])
        if (moist_flag==1):
            array_out[0:Nz-1,15] = 0.5*(tauQV3_1d[0:Nz-1]+tauQV3_1d[1:Nz])
    if (moist_flag==1):
        array_out[:,14] = qvpwp_1d
                               
    return array_out
        
def plot_turb_profiles(fig, axs, case, FE_turb_tmp, z_max, save_plot_opt, path_figure, moist_opt):

    colores_v = []
    colores_v.append('darkblue')
    colores_v.append('darkred')
    colores_v.append('dodgerblue')

    lineas_v = []
    lineas_v.append('-')
    lineas_v.append('--')
    lineas_v.append('-.')
  
    fntSize=20
    fntSize_title=22
    fntSize_legend=16
    plt.rcParams['xtick.labelsize']=fntSize
    plt.rcParams['ytick.labelsize']=fntSize
    plt.rcParams['axes.linewidth']=2.0
    
    y_min = 0.0
    y_max = z_max
    zPos = FE_turb_tmp[:,0]/1e3

    ###############
    ### panel 0 ###
    ###############
    ax = axs[0]
    varplot_1 = FE_turb_tmp[:,7]+FE_turb_tmp[:,9]
    varplot_2 = FE_turb_tmp[:,7]
    varplot_3 = FE_turb_tmp[:,9]
    im2 = ax.plot(varplot_1,zPos,lineas_v[0],color=colores_v[0],linewidth=2.5,markersize=8,label='Total',zorder=2)
    im2 = ax.plot(varplot_2,zPos,lineas_v[0],color=colores_v[2],linewidth=2.5,markersize=8,label='Res.',zorder=1)
    im2 = ax.plot(varplot_3,zPos,lineas_v[0],color=colores_v[1],linewidth=2.5,markersize=8,label='SGS',zorder=0)
    ax.set_xlabel(r"TKE $[$m$^2$ s$^{-2}]$",fontsize=fntSize)
    ax.set_ylabel(r"$z$ $[$km$]$",fontsize=fntSize)
    ax.legend(loc=1,prop={'size': fntSize_legend},edgecolor='white')
    ax.set_ylim(y_min/1e3,y_max/1e3)
    
    ###############
    ### panel 1 ###
    ###############
    ax = axs[1]
    if (moist_opt==1):
        varplot_1 = FE_turb_tmp[:,5]+FE_turb_tmp[:,11]
        varplot_2 = FE_turb_tmp[:,5]
        varplot_3 = FE_turb_tmp[:,11]
        im2 = ax.plot(varplot_1,zPos,lineas_v[0],color=colores_v[0],linewidth=2.5,markersize=8,zorder=2)
        im2 = ax.plot(varplot_2,zPos,lineas_v[0],color=colores_v[2],linewidth=2.5,markersize=8,zorder=1)
        im2 = ax.plot(varplot_3,zPos,lineas_v[0],color=colores_v[1],linewidth=2.5,markersize=8,zorder=0)
        ax.set_xlabel(r"$\sigma_w^2$ $[$m$^2$ s$^{-2}]$",fontsize=fntSize)
        ax.set_xlabel(r"$\langle U' w' \rangle$ $[$m$^2$ s$^{-2}]$",fontsize=fntSize)
        ax.set_ylim(y_min/1e3,y_max/1e3)
    else:
        varplot_1 = FE_turb_tmp[:,6]
        im2 = ax.plot(varplot_1,zPos,lineas_v[0],color=colores_v[2],linewidth=2.5,markersize=8,label='Res.')
        ax.set_xlabel(r"$\sigma_w^2$ $[$m$^2$ s$^{-2}]$",fontsize=fntSize)
        ax.set_ylim(y_min/1e3,y_max/1e3)
        
    ###############
    ### panel 2 ###
    ###############
    ax = axs[2]
    varplot_1 = FE_turb_tmp[:,8]+FE_turb_tmp[:,13]
    varplot_2 = FE_turb_tmp[:,8]
    varplot_3 = FE_turb_tmp[:,13]
    im2 = ax.plot(varplot_1,zPos,lineas_v[0],color=colores_v[0],linewidth=2.5,markersize=8,zorder=2)
    im2 = ax.plot(varplot_2,zPos,lineas_v[0],color=colores_v[2],linewidth=2.5,markersize=8,zorder=1)
    im2 = ax.plot(varplot_3,zPos,lineas_v[0],color=colores_v[1],linewidth=2.5,markersize=8,zorder=0)
    ax.set_xlabel(r"$\langle w' \theta' \rangle$ $[$K m s$^{-1}]$",fontsize=fntSize)
    ax.set_ylim(y_min/1e3,y_max/1e3)
    
    min_all = np.amin([varplot_1,varplot_2,varplot_3])
    max_all = np.amax([varplot_1,varplot_2,varplot_3])
    tol_plot = 1e-2
    if (np.abs(min_all)<=tol_plot and np.abs(max_all)<=tol_plot):
        max_abs = np.max([np.abs(min_all),np.abs(max_all)])
        ax.set_xlim(-max_abs*10.0,max_abs*10.0)
        
    ###############
    ### panel 3 ###
    ###############
    ax = axs[3]
    if (moist_opt==1):
        varplot_1 = FE_turb_tmp[:,14]+FE_turb_tmp[:,15]
        varplot_2 = FE_turb_tmp[:,14]
        varplot_3 = FE_turb_tmp[:,15]
        im2 = ax.plot(varplot_1,zPos,lineas_v[0],color=colores_v[0],linewidth=2.5,markersize=8,zorder=2)
        im2 = ax.plot(varplot_2,zPos,lineas_v[0],color=colores_v[2],linewidth=2.5,markersize=8,zorder=1)
        im2 = ax.plot(varplot_3,zPos,lineas_v[0],color=colores_v[1],linewidth=2.5,markersize=8,zorder=0)
        ax.set_xlabel(r"$\langle w' q_v' \rangle$ $[$g/kg m s$^{-1}]$",fontsize=fntSize)
        ax.set_ylim(y_min/1e3,y_max/1e3)
    else:
        varplot_1 = FE_turb_tmp[:,5]+FE_turb_tmp[:,11]
        varplot_2 = FE_turb_tmp[:,5]
        varplot_3 = FE_turb_tmp[:,11]
        im2 = ax.plot(varplot_1,zPos,lineas_v[0],color=colores_v[0],linewidth=2.5,markersize=8,zorder=2)
        im2 = ax.plot(varplot_2,zPos,lineas_v[0],color=colores_v[2],linewidth=2.5,markersize=8,zorder=1)
        im2 = ax.plot(varplot_3,zPos,lineas_v[0],color=colores_v[1],linewidth=2.5,markersize=8,zorder=0)
        ax.set_xlabel(r"$\langle U' w' \rangle$ $[$m$^2$ s$^{-2}]$",fontsize=fntSize)
        ax.set_ylim(y_min/1e3,y_max/1e3)
        
    fig_name = "TURB-PROF-"+case+".png"
    
    if (save_plot_opt==1):
        print(path_figure + fig_name)
        plt.savefig(path_figure + fig_name,dpi=300,bbox_inches = "tight")

def plot_turb_profiles_canopy(fig, axs, case, FE_mean, FE_turb_tmp, z_max, z_canopy, save_plot_opt, path_figure):

    colores_v = []
    # colores_v.append('dodgerblue')
    colores_v.append('darkblue')
    colores_v.append('dodgerblue')
    colores_v.append('black')
    # colores_v.append('darkblue')
    colores_v.append('orangered')
    colores_v.append('brown')

    lineas_v = []
    lineas_v.append('-')
    lineas_v.append('-')
    lineas_v.append('-')
    lineas_v.append('-')

    markers_v = []
    markers_v.append('o')

    marker_size = 6
    
    fntSize=16
    fntSize_title=22
    fntSize_legend=11
    fntSize_label=16
    plt.rcParams['xtick.labelsize']=fntSize
    plt.rcParams['ytick.labelsize']=fntSize
    plt.rcParams['axes.linewidth']=2.0
    
    y_min = 0.0/z_canopy
    y_max = z_max/z_canopy
    zPos = FE_turb_tmp[:,0]/z_canopy

    x_min = 0.0/z_canopy

    z_diff = np.abs(zPos-y_max)
    ind_zmax = np.where(z_diff==np.min(z_diff))[0][0]

    ###############
    ### panel 0 ###
    ###############
    ax = axs[0]
    varplot_1 = np.sqrt(np.power(FE_mean[:,1],2.0)+np.power(FE_mean[:,2],2.0))
    x_max = 1.05*np.max(varplot_1[0:ind_zmax+1])
    im2 = ax.plot(varplot_1,zPos,lineas_v[0],marker=markers_v[0],markersize=marker_size,color=colores_v[0],linewidth=2.5,label='Total',zorder=2)
    im2 = ax.plot([x_min,x_max],[1.0,1.0],'--',color='k',linewidth=1.0,zorder=0)
    ax.set_xlabel(r"$U$ $[$m s$^{-1}]$",fontsize=fntSize)
    ax.set_ylabel(r"$z/h_c$",fontsize=fntSize)
    # ax.legend(loc=0,prop={'size': fntSize_legend},edgecolor='white')
    ax.set_ylim(y_min,y_max)
    ax.set_xlim(x_min,x_max)
    
    ###############
    ### panel 1 ###
    ###############
    ax = axs[1]
    varplot_1 = FE_turb_tmp[:,7]
    varplot_2 = FE_turb_tmp[:,7] + FE_turb_tmp[:,9] + FE_turb_tmp[:,10]
    varplot_3 = FE_turb_tmp[:,9]
    varplot_4 = FE_turb_tmp[:,10]
    
    x_max = 10.0*np.max(varplot_2[0:ind_zmax+1])
    x_min = 0.1*np.min(varplot_4[0:ind_zmax+1])
    
    im2 = ax.plot(varplot_1,zPos,lineas_v[0],marker=markers_v[0],markersize=marker_size,color=colores_v[1],linewidth=2.5,label='Resolved',zorder=2)
    im2 = ax.plot(varplot_3,zPos,lineas_v[1],marker=markers_v[0],markersize=marker_size,color=colores_v[3],linewidth=2.5,label='TKE_0',zorder=2)
    im2 = ax.plot(varplot_4,zPos,lineas_v[2],marker=markers_v[0],markersize=marker_size,color=colores_v[4],linewidth=2.5,label='TKE_1',zorder=2)
    im2 = ax.plot(varplot_2,zPos,lineas_v[3],marker=markers_v[0],markersize=marker_size,color=colores_v[0],fillstyle='none',linewidth=2.5,label='Total',zorder=2)
    im2 = ax.plot([x_min,x_max],[1.0,1.0],'--',color='k',linewidth=1.0,zorder=0)
    ax.set_xlabel(r"TKE $[$m$^2$ s$^{-2}]$",fontsize=fntSize)
    ax.legend(loc=0,prop={'size': fntSize_legend},edgecolor='white')
    ax.set_ylim(y_min,y_max)
    ax.set_xscale('log')
    ax.set_xlim(x_min,x_max)
    
    fig_name = "TURB-PROF-"+case+".png"
    
    if (save_plot_opt==1):
        print(path_figure + fig_name)
        plt.savefig(path_figure + fig_name,dpi=300,bbox_inches = "tight")

def plot_pdfs(fig, axs, FE_xr, case, save_plot_opt, path_figure):

    colores_v = []
    colores_v.append('darkblue')

    lineas_v = []
    lineas_v.append('-')

    markers_v = []
    markers_v.append('o')

    marker_size = 6
    
    fntSize=16
    fntSize_title=22
    fntSize_legend=14
    fntSize_label=16
    plt.rcParams['xtick.labelsize']=fntSize
    plt.rcParams['ytick.labelsize']=fntSize
    plt.rcParams['axes.linewidth']=2.0
    
    var0 = np.squeeze(FE_xr.z0m.isel(time=0).values).flatten()
    var1 = np.squeeze(FE_xr.z0t.isel(time=0).values).flatten()

    var0_max = np.max(var0)
    var1_max = np.max(var1)
    exp0 = -find_exp(var0_max)
    exp1 = -find_exp(var1_max)
    var0 = var0/(10**(exp0))
    var1 = var1/(10**(exp1))

    var0_min = np.min(var0)
    var0_max = np.max(var0)
    var1_min = np.min(var1)
    var1_max = np.max(var1)

    pdf_inc_0 = (var0_max-var0_min)/60
    pdf_inc_1 = (var1_max-var1_min)/60
    
    bins_0 = np.arange(var0_min,var0_max+pdf_inc_0,pdf_inc_0)
    bins_1 = np.arange(var1_min,var1_max+pdf_inc_1,pdf_inc_1)

    ###############
    ### panel 0 ###
    ###############
    [pdf_tmp,bin_edge] = np.histogram(var0,bins_0)
    bin_c = 0.5*(bin_edge[0:len(bin_edge)-1]+bin_edge[1:len(bin_edge)])
    
    ax = axs[0]
    im2 = ax.plot(bin_c,pdf_tmp/np.sum(pdf_tmp),lineas_v[0],color=colores_v[0],linewidth=2.5,label=case)
    str_tmp = r"$z_{0,m}$ $\times$ 10$^{" + str(exp0) + "}$ $[$m$]$"
    ax.set_xlabel(str_tmp,fontsize=fntSize)
    ax.set_ylabel(r"PDF $[$-$]$",fontsize=fntSize)
    ax.legend(loc=0,prop={'size': fntSize_legend},edgecolor='white')
    
    ###############
    ### panel 1 ###
    ###############
    [pdf_tmp,bin_edge] = np.histogram(var1,bins_1)
    bin_c = 0.5*(bin_edge[0:len(bin_edge)-1]+bin_edge[1:len(bin_edge)])
    
    ax = axs[1]
    im2 = ax.plot(bin_c,pdf_tmp/np.sum(pdf_tmp),lineas_v[0],color=colores_v[0],linewidth=2.5)
    str_tmp = r"$z_{0,t}$ $\times$ 10$^{" + str(exp1) + "}$ $[$m$]$"
    ax.set_xlabel(str_tmp,fontsize=fntSize)
    # ax.set_ylabel(r"PDF $[$-$]$",fontsize=fntSize)
    
    fig_name = "PDF-"+case+".png"
    
    if (save_plot_opt==1):
        print(path_figure + fig_name)
        plt.savefig(path_figure + fig_name,dpi=300,bbox_inches = "tight")

def find_exp(number):
    base10 = log10(abs(number))
    return abs(floor(base10))

def adjustRhoQvInitial(z_prof, BS_Dict, qv_lev, zq_lev, dqdz):
    printLog = False

    # Set Constant values #
    accel_g = 9.81          # Acceleration of gravity 9.8 m/s^2 
    R_gas = 287.04          # The ideal gas constant in J/(kg*K) 
    R_vapor = 461.60        # The ideal gas constant for water vapor in J/(kg*K) 
    cv_gas = 718.0          # Specific heat of air at constant volume ****and temperature 300 K in J/(kg*K) 
    cp_gas = R_gas+cv_gas   # Specific heat of air at constant pressure ****and temperature 300 K in J/(kg*K) 
    L_v = 2.5e6             # latent heat of vaporization (J/kg) 
    
    R_cp = R_gas/cp_gas     # Ratio R/cp
    cp_R = cp_gas/R_gas     # Ratio cp/R
    cp_cv = cp_gas/cv_gas   # Ratio cp/cv
    refPressure = 1.0e5     # Reference pressure set constant to 1e5 Pascals or 1000 millibars) 
    Rv_Rg = R_vapor/R_gas   # Ratio R_vapor/R_gas

    ###
    pres_grnd = BS_Dict['pres_grnd']
    temp_grnd = BS_Dict['temp_grnd']
    if (printLog):
        print('pres_grnd=',pres_grnd,'Pa')
        print('temp_grnd=',temp_grnd,'K')

    constant_1 = R_gas/(refPressure**R_cp)
    rho_grnd = pres_grnd/(R_gas*temp_grnd)
    theta_grnd = temp_grnd*((pres_grnd/refPressure)**(-R_cp))
    qv_grnd = qv_lev[0]
    prm_grnd = (rho_grnd*(theta_grnd*(1.0+Rv_Rg*qv_grnd*1e-3))*constant_1)**(cp_cv)
    thm_grnd = theta_grnd*(1.0+Rv_Rg*qv_grnd*1e-3)
    Tm_grnd = thm_grnd*(prm_grnd/refPressure)**R_cp
    rhom_grnd = prm_grnd/(R_gas*Tm_grnd)
    if (printLog):
        print('pres_grnd = {:f}, theta_grnd = {:f}, rho_grnd = {:f}, temp_grnd = {:f}'.format(pres_grnd,theta_grnd,rho_grnd,temp_grnd))
        print('prm_grnd = {:f}, thm_grnd = {:f}, rhom_grnd = {:f}, Tm_grnd = {:f}, qv_grnd = {:f}'.format(prm_grnd,thm_grnd,rhom_grnd,Tm_grnd,qv_grnd))

    th_prof=np.zeros(z_prof.size)
    dth_prof=np.zeros(z_prof.size)
    dthm_prof=np.zeros(z_prof.size)
    thm_prof=np.zeros(z_prof.size)
    qv_prof=np.zeros(z_prof.size)
    dqv_prof=np.zeros(z_prof.size)
    Tm_prof=np.zeros(z_prof.size)
    rhom_prof=np.zeros(z_prof.size)
    prnew_prof=np.zeros(z_prof.size)

    #Given a stability scheme define the theta (dry/moist) profile that we will use assuming hydrostatic balance to obtain rho profile
    if(BS_Dict['stabilityScheme'] == 2):
        for k in range(len(z_prof)):
            if(z_prof[k] <= zq_lev[1]):
                z_inc = z_prof[k] - 0.0
                qv_prof[k] = qv_lev[0] + z_inc*dqdz[0]
                dqv_prof[k] = 0 # qv constant here so dqv/dz = 0
            elif((zq_lev[1] < z_prof[k]) and (z_prof[k] <= zq_lev[2])):
                z_inc = z_prof[k] - zq_lev[1]
                qv_prof[k] = qv_lev[0] + (zq_lev[1]-zq_lev[0])*dqdz[0] + z_inc*dqdz[1]
                dqv_prof[k] = dqdz[1] # qv is linear here so dqv/dz = dqdz[1]
            else:
                z_inc = z_prof[k] - zq_lev[2]
                qv_prof[k] = qv_lev[0] + (zq_lev[1]-zq_lev[0])*dqdz[0] + (zq_lev[2]-zq_lev[1])*dqdz[1] + z_inc*dqdz[2]
                dqv_prof[k] = dqdz[2] # qv is linear here (the first layer contribution is constant at this height) so dqv/dz = dqdz[2]
            if(qv_prof[k] < 0.0):
                qv_prof[k] = 0.0
                dqv_prof[k] = (qv_prof[k]-qv_prof[k-1])/(z_prof[k]-z_prof[k-1])
            #Now prescribe theta (dry/moist) and the corrseponding pressure
            if(z_prof[k] <= BS_Dict['zStableBottom']): 
                #This point is in a neutral lowest-layer
                th_prof[k] = theta_grnd;                               #dry-theta
                dth_prof[k] = 0                               #dry-theta is constant at this height so dth/dz = 0
                dthm_prof[k] = dth_prof[k]*(1.0+Rv_Rg*qv_prof[k]*1e-3)+dqv_prof[k]*1e-3*(Rv_Rg*th_prof[k])
                thm_prof[k] = th_prof[k]*(1.0+Rv_Rg*qv_prof[k]*1e-3)   #moist-theta  (like WRF)
                #Dry HB
                z_dep = (z_prof[k]/theta_grnd)
                frac_refp = ((-accel_g/cp_gas)*z_dep+(pres_grnd/refPressure)**(R_cp))**(cp_R)
                #Moist HB
                if k == 0:
                    zm_dep = (z_prof[k]/thm_prof[k])
                    prnew_prof[k] = refPressure*((-accel_g/cp_gas)*zm_dep+(prm_grnd/refPressure)**(R_cp))**(cp_R)
                    frac_refpm = ((-accel_g/cp_gas)*zm_dep+(prm_grnd/refPressure)**(R_cp))**(cp_R)
                else:
                    zm_dep = ((z_prof[k]-z_prof[k-1])/thm_prof[k])
                    prnew_prof[k] = refPressure*((-accel_g/cp_gas)*zm_dep+(prnew_prof[k-1]/refPressure)**(R_cp))**(cp_R)
                    frac_refpm = ((-accel_g/cp_gas)*zm_dep+(prnew_prof[k-1]/refPressure)**(R_cp))**(cp_R)
                if printLog:
                    print("@[k={:d}],zm_dep = {:f}, frac_refpm = {:f}, p = {:f}".format(k,zm_dep,frac_refpm,prnew_prof[k]))
            elif((BS_Dict['zStableBottom'] < z_prof[k]) and (z_prof[k] <= BS_Dict['zStableBottom2'])): 
                #This point is within the first stable upper-layer
                th_prof[k] = theta_grnd + BS_Dict['stableGradient']*(z_prof[k]-BS_Dict['zStableBottom'])
                dth_prof[k] = BS_Dict['stableGradient']
                dthm_prof[k] = dth_prof[k]*(1.0+Rv_Rg*qv_prof[k]*1e-3)+dqv_prof[k]*1e-3*(Rv_Rg*th_prof[k])
                dthm_dz1=dthm_prof[k]
                thm_prof[k] = th_prof[k]*(1.0+Rv_Rg*qv_prof[k]*1e-3)
                #Dry HB
                z_dep = (BS_Dict['zStableBottom']/theta_grnd+(1.0/BS_Dict['stableGradient'])*np.log(1.0+BS_Dict['stableGradient']*(z_prof[k]-BS_Dict['zStableBottom'])/theta_grnd))
                frac_refp = ((-accel_g/cp_gas)*z_dep+(pres_grnd/refPressure)**(R_cp))**(cp_R)
                #Moist HB
                zm_dep = ((z_prof[k]-z_prof[k-1])/thm_prof[k])
                prnew_prof[k] = refPressure*((-accel_g/cp_gas)*zm_dep+(prnew_prof[k-1]/refPressure)**(R_cp))**(cp_R)
                frac_refpm = ((-accel_g/cp_gas)*zm_dep+(prnew_prof[k-1]/refPressure)**(R_cp))**(cp_R)
                if printLog:
                    print("@[k={:d}], frac_refpm = {:f}, p = {:f}".format(k,frac_refpm,prnew_prof[k]))
            elif((BS_Dict['zStableBottom2'] < z_prof[k]) and (z_prof[k] <= BS_Dict['zStableBottom3'])): 
                #This point is within the third stable upper-layer
                th_prof[k] = theta_grnd + BS_Dict['stableGradient']*(BS_Dict['zStableBottom2']-BS_Dict['zStableBottom']) + BS_Dict['stableGradient2']*(z_prof[k]-BS_Dict['zStableBottom2'])
                dth_prof[k] = BS_Dict['stableGradient2']
                dthm_prof[k] = dth_prof[k]*(1.0+Rv_Rg*qv_prof[k]*1e-3)+dqv_prof[k]*1e-3*(Rv_Rg*th_prof[k])
                dthm_dz2 = dthm_prof[k]
                thm_prof[k] = th_prof[k]*(1.0+Rv_Rg*qv_prof[k]*1e-3)
                #Dry HB
                z_dep = ( BS_Dict['zStableBottom']/theta_grnd
                         +(1.0/BS_Dict['stableGradient'])*np.log(1.0+BS_Dict['stableGradient']*(BS_Dict['zStableBottom2']-BS_Dict['zStableBottom'])/theta_grnd)
                         +(1.0/BS_Dict['stableGradient2'])*np.log(1.0+BS_Dict['stableGradient2']*(z_prof[k]-BS_Dict['zStableBottom2'])
                         /(theta_grnd+BS_Dict['stableGradient']*(BS_Dict['zStableBottom2']-BS_Dict['zStableBottom']))))
                frac_refp = ((-accel_g/cp_gas)*z_dep+(pres_grnd/refPressure)**(R_cp))**(cp_R)
                #Moist HB
                zm_dep = ((z_prof[k]-z_prof[k-1])/thm_prof[k])
                prnew_prof[k] = refPressure*((-accel_g/cp_gas)*zm_dep+(prnew_prof[k-1]/refPressure)**(R_cp))**(cp_R)
                frac_refpm = ((-accel_g/cp_gas)*zm_dep+(prnew_prof[k-1]/refPressure)**(R_cp))**(cp_R)
                if printLog:
                    print("@[k={:d}], frac_refpm = {:f}, p = {:f}".format(k,frac_refpm,prnew_prof[k]))
            else: 
                #This point is within the second stable upper-layer
                th_prof[k] = theta_grnd + BS_Dict['stableGradient']*(BS_Dict['zStableBottom2']-BS_Dict['zStableBottom']) + BS_Dict['stableGradient2']*(BS_Dict['zStableBottom3']-BS_Dict['zStableBottom2']) + BS_Dict['stableGradient3']*(z_prof[k]-BS_Dict['zStableBottom3'])
                dth_prof[k] = BS_Dict['stableGradient3']
                dthm_prof[k] = dth_prof[k]*(1.0+Rv_Rg*qv_prof[k]*1e-3)+dqv_prof[k]*1e-3*(Rv_Rg*th_prof[k])
                dthm_dz3 = dthm_prof[k]
                thm_prof[k] = th_prof[k]*(1.0+Rv_Rg*qv_prof[k]*1e-3)
                #Dry HB
                z_dep = ( BS_Dict['zStableBottom']/theta_grnd
                         +(1.0/BS_Dict['stableGradient'])*np.log(1.0+BS_Dict['stableGradient']*(BS_Dict['zStableBottom2']-BS_Dict['zStableBottom'])/theta_grnd)
                         +(1.0/BS_Dict['stableGradient2'])*np.log(1.0+BS_Dict['stableGradient2']*(BS_Dict['zStableBottom3']-BS_Dict['zStableBottom2'])
                         /(theta_grnd+BS_Dict['stableGradient']*(BS_Dict['zStableBottom2']-BS_Dict['zStableBottom'])))
                         +(1.0/BS_Dict['stableGradient3'])*np.log(1.0+BS_Dict['stableGradient3']*(z_prof[k]-BS_Dict['zStableBottom3'])
                         /(theta_grnd+BS_Dict['stableGradient']*(BS_Dict['zStableBottom2']-BS_Dict['zStableBottom'])+BS_Dict['stableGradient2']*(BS_Dict['zStableBottom3']-BS_Dict['zStableBottom2']))))
                frac_refp = ((-accel_g/cp_gas)*z_dep+(pres_grnd/refPressure)**(R_cp))**(cp_R)
                #Moist HB
                zm_dep = ((z_prof[k]-z_prof[k-1])/thm_prof[k])
                prnew_prof[k] = refPressure*((-accel_g/cp_gas)*zm_dep+(prnew_prof[k-1]/refPressure)**(R_cp))**(cp_R)
                frac_refpm = ((-accel_g/cp_gas)*zm_dep+(prnew_prof[k-1]/refPressure)**(R_cp))**(cp_R)
                if printLog:
                    print("@[k={:d}], frac_refpm = {:f}, p = {:f}".format(k,frac_refpm,prnew_prof[k]))
            #Determine base state air temperature, T_prof from th_prof and pr_prof 
            Tm_prof[k] = thm_prof[k]*(prnew_prof[k]/refPressure)**(R_cp)
            #Determine base state denisty, rho_prof from Tm_prof and prnew_prof 
            rhom_prof[k] = prnew_prof[k]/(Tm_prof[k]*R_gas)

    return rhom_prof, qv_prof