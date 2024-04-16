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
    
    u_3d = np.squeeze(FE_xr.u.isel(time=0).values)
    v_3d = np.squeeze(FE_xr.v.isel(time=0).values)
    th_3d = np.squeeze(FE_xr.theta.isel(time=0).values)
    z_3d = np.squeeze(FE_xr.zPos.isel(time=0).values)

    Nz = u_3d.shape[0]
    wd_1d = np.zeros([Nz])

    u_1d = np.mean(np.mean(u_3d,axis=2),axis=1)
    v_1d = np.mean(np.mean(v_3d,axis=2),axis=1)
    th_1d = np.mean(np.mean(th_3d,axis=2),axis=1)
    z_1d = np.mean(np.mean(z_3d,axis=2),axis=1)

    for kk in range(0,Nz):
        wd_tmp = math.atan2(-u_1d[kk],-v_1d[kk]) * 180.0 / np.pi
        if (wd_tmp<0.0):
            wd_tmp = 180.0 + (180.0 + wd_tmp)
        wd_1d[kk] = wd_tmp

    array_out = np.zeros([Nz,5])
    array_out[:,0] = z_1d
    array_out[:,1] = u_1d
    array_out[:,2] = v_1d
    array_out[:,3] = wd_1d
    array_out[:,4] = th_1d
    
    return array_out

def plot_XY_UVWTHETA(case, case_open, zChoose, save_plot_opt, path_figure):

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
        
    u_min = np.amin(np.amin(ufield))
    u_max = np.amax(np.amax(ufield))
    v_min = np.amin(np.amin(vfield))
    v_max = np.amax(np.amax(vfield))
    w_min = np.amin(np.amin(wfield))
    w_max = np.amax(np.amax(wfield))
    t_min = np.amin(np.amin(thetafield[zChoose,:,:]))
    t_max = np.amax(np.amax(thetafield[zChoose,:,:]))
    
    fig_name = "UVWTHETA-XY-"+case+".png"
    colormap1 = 'viridis'
    colormap2 = 'seismic'
    colormap3 = 'YlOrRd'
    FE_legend = [r'u [m/s] at z='+str(np.amax(zPos))+' m',r'v [m/s] at z='+str(np.amax(zPos))+' m',r'w [m/s] at z='+str(np.amax(zPos))+' m', \
                 '\u03B8 [K] at z='+str(np.amax(zPos))+' m']
    
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
    im = ax.pcolormesh(xPos/1e3,yPos/1e3,thetafield[zChoose,:,:],cmap=colormap3,linewidth=0,rasterized=True,vmin=t_min,vmax=t_max)
    ax.set_ylabel(r'$y$ $[\mathrm{km}]$',fontsize=fntSize)
    ax.set_xlabel(r'$x$ $[\mathrm{km}]$',fontsize=fntSize)
    cbar=fig.colorbar(im, ax=ax)

    title_fig_3 = FE_legend[3]
    ax.set_title(title_fig_3,fontsize=fntSize)

    if (save_plot_opt==1):
        print(path_figure + fig_name)
        plt.savefig(path_figure + fig_name,dpi=300,bbox_inches = "tight")

def plot_XZ_UVWTHETA(case, case_open, z_max, sizeX_XZ, sizeY_XZ, save_plot_opt, path_figure):
    
    ufield = case_open.u.isel(time=0).values
    vfield = case_open.v.isel(time=0).values
    wfield = case_open.w.isel(time=0).values
    thetafield = case_open.theta.isel(time=0).values
    
    [Npz,Npy,Npx] = ufield.shape
    yChoose = int(Npy/2)
    
    xPos = case_open.xPos.isel(time=0,yIndex=yChoose).values
    yPos = case_open.yPos.isel(time=0,yIndex=yChoose).values
    zPos = case_open.zPos.isel(time=0,yIndex=yChoose).values
            
    u_min = np.amin(np.amin(ufield))
    u_max = np.amax(np.amax(ufield))
    v_min = np.amin(np.amin(vfield))
    v_max = np.amax(np.amax(vfield))
    w_min = np.amin(np.amin(wfield))
    w_max = np.amax(np.amax(wfield))
    t_min = np.amin(np.amin(thetafield))
    t_max = np.amax(np.amax(thetafield))

    fig_name = "UVWTHETA-XZ-"+case+".png"
    colormap1 = 'viridis'
    colormap2 = 'seismic'
    colormap3 = 'YlOrRd'
    FE_legend = [r'u [m/s] at y='+str(yPos[0,yChoose]/1e3)+' km',r'v [m/s] at y='+str(yPos[0,yChoose]/1e3)+' km',r'w [m/s] at y='+str(yPos[0,yChoose]/1e3)+' km', \
                 '\u03B8 [K] at y='+str(yPos[0,yChoose]/1e3)+' km']
    
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
    im = ax.pcolormesh(xPos/1e3,zPos/1e3,ufield[:,yChoose,:],cmap=colormap1,linewidth=0,rasterized=True,vmin=u_min,vmax=u_max)
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
    im = ax.pcolormesh(xPos/1e3,zPos/1e3,vfield[:,yChoose,:],cmap=colormap1,linewidth=0,rasterized=True,vmin=v_min,vmax=v_max)
    ax.set_xticklabels([])
    ax.set_ylabel(r'$z$ $[\mathrm{km}]$',fontsize=fntSize)
    cbar=fig.colorbar(im, ax=ax)
    ax.set_ylim(0.0,z_max/1e3)

    title_fig_1 = FE_legend[1]
    ax.set_title(title_fig_1,fontsize=fntSize)

    ###############
    ### W plot ###
    ###############
    w_min=-1.0*w_max
    ax=axs[2]
    im = ax.pcolormesh(xPos/1e3,zPos/1e3,wfield[:,yChoose,:],cmap=colormap2,linewidth=0,rasterized=True,vmin=w_min,vmax=w_max)
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
    im = ax.pcolormesh(xPos/1e3,zPos/1e3,thetafield[:,yChoose,:],cmap=colormap3,linewidth=0,rasterized=True,vmin=t_min,vmax=t_max)
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

def plot_mean_profiles(fig, axs, FE_mean, z_max, caseLabel, save_plot_opt, path_figure, caseCnt):

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
    ### panel 3 ###
    ###############
    ax = axs[2]
    im = ax.plot(FE_mean[:,4],zPos,lineas_v[caseCnt],color=colores_v[caseCnt],label=caseLabel)
    ax.set_xlabel(r"$\theta$ [K]",fontsize=fntSize)
    ax.set_ylim(0.0,z_max/1e3)
    
    if (save_plot_opt):
        print(path_figure + fig_name)
        plt.savefig(path_figure + fig_name,dpi=300,bbox_inches = "tight")

def compute_turb_profiles(FE_xr, array_out):
    
    u_3d = np.squeeze(FE_xr.u.isel(time=0).values)
    v_3d = np.squeeze(FE_xr.v.isel(time=0).values)
    w_3d = np.squeeze(FE_xr.w.isel(time=0).values)
    th_3d = np.squeeze(FE_xr.theta.isel(time=0).values)
    z_3d = np.squeeze(FE_xr.zPos.isel(time=0).values)
    tau13_3d = np.squeeze(FE_xr.Tau31.isel(time=0).values)
    tau23_3d = np.squeeze(FE_xr.Tau32.isel(time=0).values)
    tau33_3d = np.squeeze(FE_xr.Tau33.isel(time=0).values)
    tauTH3_3d = np.squeeze(FE_xr.TauTH3.isel(time=0).values)
    sgstke_3d = np.squeeze(FE_xr.TKE_0.isel(time=0).values)

    Nz = u_3d.shape[0]
    Ny = u_3d.shape[1]
    Nx = u_3d.shape[2]

    u_1d = np.mean(np.mean(u_3d,axis=2),axis=1)
    v_1d = np.mean(np.mean(v_3d,axis=2),axis=1)
    w_1d = np.mean(np.mean(w_3d,axis=2),axis=1)
    th_1d = np.mean(np.mean(th_3d,axis=2),axis=1)
    z_1d = np.mean(np.mean(z_3d,axis=2),axis=1)
    
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
    
    up = u_3d - u_3d_mean
    vp = v_3d - v_3d_mean
    wp = w_3d - w_3d_mean
    thp = th_3d - th_3d_mean
    
    upup = up*up
    upwp = up*wp
    vpvp = vp*vp
    vpwp = vp*wp
    wpwp = wp*wp
    thpwp = thp*wp
    tke = 0.5*(upup+vpvp+wpwp)
    
    upup_1d = np.mean(np.mean(upup,axis=2),axis=1)
    upwp_1d = np.mean(np.mean(upwp,axis=2),axis=1)
    vpvp_1d = np.mean(np.mean(vpvp,axis=2),axis=1)
    vpwp_1d = np.mean(np.mean(vpwp,axis=2),axis=1)
    tau13_1d = np.mean(np.mean(tau13_3d,axis=2),axis=1)
    tau23_1d = np.mean(np.mean(tau23_3d,axis=2),axis=1)
    tau33_1d = np.mean(np.mean(tau33_3d,axis=2),axis=1)
    tauTH3_1d = np.mean(np.mean(tauTH3_3d,axis=2),axis=1)
    sgstke_1d = np.mean(np.mean(sgstke_3d,axis=2),axis=1)
    
    Upwp_1d = np.sqrt(np.power(upwp_1d,2.0)+np.power(vpwp_1d,2.0))
    wpwp_1d = np.mean(np.mean(wpwp,axis=2),axis=1)
    tke_1d = np.mean(np.mean(tke,axis=2),axis=1)
    thpwp_1d = np.mean(np.mean(thpwp,axis=2),axis=1)
    tau1323_1d = np.sqrt(np.power(tau13_1d,2.0)+np.power(tau23_1d,2.0))
    
    array_out = np.zeros([Nz,15])
    array_out[:,0] = z_1d
    array_out[:,1] = upup_1d
    array_out[:,2] = upwp_1d
    array_out[:,3] = vpvp_1d
    array_out[:,4] = vpwp_1d
    array_out[:,5] = Upwp_1d
    array_out[:,6] = wpwp_1d
    array_out[:,7] = tke_1d
    array_out[:,8] = thpwp_1d
    array_out[0:Nz-1,9] = 0.5*(tau1323_1d[0:Nz-1]+tau1323_1d[1:Nz])
    array_out[:,10] = sgstke_1d
    array_out[0:Nz-1,11] = 0.5*(tau33_1d[0:Nz-1]+tau33_1d[1:Nz])
    array_out[0:Nz-1,12] = 0.5*(tauTH3_1d[0:Nz-1]+tauTH3_1d[1:Nz])
                               
    return array_out
        
def plot_turb_profiles(fig, axs, case, FE_turb_tmp, z_max, save_plot_opt, path_figure):

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
    varplot_1 = FE_turb_tmp[:,7]+FE_turb_tmp[:,10]
    varplot_2 = FE_turb_tmp[:,7]
    varplot_3 = FE_turb_tmp[:,10]
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
    varplot_1 = FE_turb_tmp[:,6]
    im2 = ax.plot(varplot_1,zPos,lineas_v[0],color=colores_v[2],linewidth=2.5,markersize=8,label='Res.')
    ax.set_xlabel(r"$\sigma_w^2$ $[$m$^2$ s$^{-2}]$",fontsize=fntSize)
    ax.set_ylim(y_min/1e3,y_max/1e3)
    
    ###############
    ### panel 2 ###
    ###############
    ax = axs[2]
    varplot_1 = FE_turb_tmp[:,8]+FE_turb_tmp[:,12]
    varplot_2 = FE_turb_tmp[:,8]
    varplot_3 = FE_turb_tmp[:,12]
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
    varplot_1 = FE_turb_tmp[:,5]+FE_turb_tmp[:,9]
    varplot_2 = FE_turb_tmp[:,5]
    varplot_3 = FE_turb_tmp[:,9]
    im2 = ax.plot(varplot_1,zPos,lineas_v[0],color=colores_v[0],linewidth=2.5,markersize=8,zorder=2)
    im2 = ax.plot(varplot_2,zPos,lineas_v[0],color=colores_v[2],linewidth=2.5,markersize=8,zorder=1)
    im2 = ax.plot(varplot_3,zPos,lineas_v[0],color=colores_v[1],linewidth=2.5,markersize=8,zorder=0)
    ax.set_xlabel(r"$\sigma_w^2$ $[$m$^2$ s$^{-2}]$",fontsize=fntSize)
    ax.set_xlabel(r"$\langle U' w' \rangle$ $[$m$^2$ s$^{-2}]$",fontsize=fntSize)
    ax.set_ylim(y_min/1e3,y_max/1e3)
    
    fig_name = "TURB-PROF-"+case+".png"
    
    if (save_plot_opt==1):
        print(path_figure + fig_name)
        plt.savefig(path_figure + fig_name,dpi=300,bbox_inches = "tight")