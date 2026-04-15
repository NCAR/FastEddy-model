import argparse
import math
import time
import scipy.ndimage as ndimage
import numpy as np
import xarray as xr
import pandas as pd
from scipy import interpolate
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import BSpline, make_interp_spline

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

def zDeform(zRect, zGround, zCeiling, c1, fCoeff):

    # FastEddy's vertical coordinate stretching formulation #
    c2 = fCoeff*(1.0-c1)/zCeiling
    c3 = (1.0-c2*zCeiling-c1)/(pow(zCeiling,2.0))
    zStretch = (c3*pow(zRect,3.0)+c2*pow(zRect,2.0)+c1*zRect)*(zCeiling-zGround)/zCeiling+zGround

    return zStretch

def block_average_topo(topo_dom_ori,dx_inter,dy_inter,d_xi,d_eta,Nx,Ny):
    ny_src, nx_src = topo_dom_ori.shape
    data_topo0 = np.zeros((Ny, Nx), dtype=float)
    x_edges_src = np.arange(nx_src + 1) * dx_inter
    y_edges_src = np.arange(ny_src + 1) * dy_inter
    x_edges_dst = np.arange(Nx + 1) * d_xi
    y_edges_dst = np.arange(Ny + 1) * d_eta
    for j in range(Ny):
        y0 = y_edges_dst[j]
        y1 = y_edges_dst[j + 1]
        iy0 = np.searchsorted(y_edges_src, y0, side="right") - 1
        iy1 = np.searchsorted(y_edges_src, y1, side="left")
        for i in range(Nx):
            x0 = x_edges_dst[i]
            x1 = x_edges_dst[i + 1]
            ix0 = np.searchsorted(x_edges_src, x0, side="right") - 1
            ix1 = np.searchsorted(x_edges_src, x1, side="left")
            val = 0.0
            wsum = 0.0
            for iy in range(max(iy0, 0), min(iy1 + 1, ny_src)):
                ys0 = y_edges_src[iy]
                ys1 = y_edges_src[iy + 1]
                overlap_y = min(y1, ys1) - max(y0, ys0)
                for ix in range(max(ix0, 0), min(ix1 + 1, nx_src)):
                    xs0 = x_edges_src[ix]
                    xs1 = x_edges_src[ix + 1]
                    overlap_x = min(x1, xs1) - max(x0, xs0)
                    if overlap_x > 0.0 and overlap_y > 0.0:
                        w = overlap_x * overlap_y
                        val += topo_dom_ori[iy, ix] * w
                        wsum += w
            if wsum == 0.0:
                raise ValueError(f'No overlap found for cell j={j}, i={i}.')
            data_topo0[j, i] = val / wsum
    return data_topo0

def smoothTerrain(tPos0,dx):
    start = time.time()
    slopeThresh=math.tan(math.radians(35.0))
    blend=0.2
    n=3
    gradTopoX,gradTopoY=np.gradient(tPos0,dx)
    gradTopoR=np.sqrt(gradTopoX**2+gradTopoY**2)
    max_dTdR=gradTopoR.max(axis=(0,1))
    problematic_points0=np.sum(gradTopoR >= slopeThresh)
    problematic_points=problematic_points0
    iterSmooth=0
    max_iter=problematic_points0*1.5
    tPos1=tPos0.copy('K')
    print(f'Problematic points: {problematic_points0}')
    if (problematic_points0!=0):
        while((max_dTdR>=slopeThresh) and (iterSmooth<=max_iter)):
            sigTry = 25
            tPos2=tPos1.copy('K')
            problematic_points_solved = 0
            while((max_dTdR>=slopeThresh) and (sigTry<=25) and (problematic_points_solved <= 0)):
                pps = np.where(gradTopoR >= slopeThresh)
                for pp in range(len(pps[0])):
                    max_idx = (pps[0][pp], pps[1][pp])
                    min_row = max(0, max_idx[0] - n//2)
                    max_row = min(tPos1.shape[0], max_idx[0] + n//2+1)
                    min_col = max(0, max_idx[1] - n//2)
                    max_col = min(tPos1.shape[1], max_idx[1] + n//2+1)
                    local_area = tPos1[min_row:max_row, min_col:max_col]
                    local_blur_center = ndimage.gaussian_filter(local_area, sigma=sigTry)
                    for i in range(min_row, max_row):
                        for j in range(min_col, max_col):
                            if ((i, j) == max_idx):
                                tPos2[i, j] = local_blur_center[i - min_row, j - min_col]
                            else:
                                tPos2[i, j] = (1-blend)*tPos1[i, j] + blend*local_blur_center[i - min_row, j - min_col]
                gradTopoX_new, gradTopoY_new = np.gradient(tPos2, dx)
                gradTopoR_new = np.sqrt(gradTopoX_new**2 + gradTopoY_new**2)
                max_dTdR_new = gradTopoR_new.max(axis=(0, 1))
                problematic_points_new=np.sum(gradTopoR_new >= slopeThresh)
                problematic_points_solved=problematic_points - problematic_points_new
                if((max_dTdR>=slopeThresh) and (problematic_points_solved <= 0)):
                    sigTry+=1
            iterSmooth+=1
            problematic_points=problematic_points_new
            tPos1=tPos2.copy('K')
            gradTopoR = gradTopoR_new
            max_dTdR = max_dTdR_new
        end = time.time()
        print(f'From {problematic_points0} to {problematic_points_new} prob points in {iterSmooth} iterations')
        print(f'Elapsed time [s]: {np.round(end-start,3)}')
    return tPos1

def interpFEToGrids(ds_ref,it0,varsList,surfVarsList,zRect,ll_iindx,i_extent,ll_jindx,j_extent,kMaxPrnt):
    verbose = False
    surfVarDict = {'tskin':'tskin','qskin':'qskin','topoPos':'topoParent'}
    dsFE = xr.Dataset()
    zPrntArray = ds_ref['zPos'][it0,:kMaxPrnt,ll_jindx:ll_jindx+j_extent,ll_iindx:ll_iindx+i_extent].values   
    for var in varsList:
        tmpArray=np.zeros((zRect.shape[0],j_extent,i_extent))
        if var !=  'zPos':
         varPrntArray=ds_ref[var][it0,:kMaxPrnt,ll_jindx:ll_jindx+j_extent,ll_iindx:ll_iindx+i_extent].values
         for i in range(i_extent):
             for j in range(j_extent):
                 f1=make_interp_spline(zPrntArray[:,j,i],varPrntArray[:,j,i],
                                       k=3, bc_type='natural')
                 tmpArray[:,j,i] = f1(zRect)
         dsFE[var] = xr.DataArray(tmpArray,dims=('zIndex','yIndex','xIndex'))
        if verbose:
           print(f"interpFEToGrids: {var}--({dsFE[var].values.shape})")
    for surfVar in surfVarsList:
        tmpArray=ds_ref[surfVar][it0,ll_jindx:ll_jindx+j_extent,ll_iindx:ll_iindx+i_extent].values
        dsFE[surfVarDict[surfVar]] = xr.DataArray(tmpArray,dims=('yIndex','xIndex'))
        if verbose:
           print(f"interpFEToGrids: {surfVarDict[surfVar]}--({dsFE[surfVarDict[surfVar]].values.shape})")

    return dsFE

def interpWRFToGrids(ds_ref,it0,varsList,surfVarsList,zRect,ll_iindx,i_extent,ll_jindx,j_extent):
    dsWRF = xr.Dataset()
    for i in range(ll_iindx,ll_iindx+i_extent):
        for j in range(ll_jindx,ll_jindx+j_extent):
            dsWRF1=get_dsWRFStandardZprof(it0,j,i,ds_ref,varsList,surfVarsList,zRect)

            if(j == ll_jindx):
                dsWRFj=dsWRF1.expand_dims(dim={'yIndex':1})
            else:
                dsWRFj=xr.concat([dsWRFj,dsWRF1.expand_dims(dim={'yIndex':1})],dim='yIndex')
        if(i == ll_iindx):
            dsWRF=dsWRFj.expand_dims(dim={'xIndex':1})
        else:
            dsWRF=xr.concat([dsWRF,dsWRFj.expand_dims(dim={'xIndex':1})],dim='xIndex')
    return dsWRF

def get_dsWRFStandardZprof(it0,j0,i0,ds_ref,varsList,surfVarsList,zProf):
    ds_ij=getFEProfileDS(getWRFProfileDS(it0,j0,i0,ds_ref,varsList,surfVarsList,zProf[-1]),varsList,surfVarsList,zProf)

    return ds_ij

def getFEProfileDS(dsWrf,varsList,surfVarsList,zFE): ##### Map (Interp/Extrap-olate) a collected set of WRF vertical profiles from
                                                     ##### the WRF vertical coordinate to a specified cartesian z-coord (zFE)
    ds_ret=xr.Dataset()
    varDict = {'Z':'zPos','U':'u','V':'v','W':'w','T':'theta','QVAPOR':'qv','QCLOUD':'ql','ALT':'rho','QKE':'TKE_0'}
    surfVarDict = {'TSK':'tskin','Q2':'qskin','HGT':'topoWRF','T2':'t2','PSFC':'psfc'} #Note using Q2 instead of QVG since QVG not default in wrfout files
    #print(f"kMaxPrnt = {dsWrf.sizes['bottom_top']}")
    for var in varsList:
        if var !=  'Z':
            f1=interpolate.interp1d(dsWrf['Z'],dsWrf[var],kind='linear',fill_value='extrapolate')
            if var != 'ALT' and var != 'ALB':
                ds_ret[varDict[var]] = xr.DataArray(f1(zFE),dims=['zIndex'])
            else:
                ds_ret[varDict[var]] = xr.DataArray(1.0/f1(zFE),dims=['zIndex'])
    for surfVar in surfVarsList:
        ds_ret[surfVarDict[surfVar]] = xr.DataArray(dsWrf[surfVar])

    return ds_ret

def getWRFProfileDS(it,j,i,dsWrf,varsList,surfVarsList,zTargTop): ##### Destagger and collect a set of required WRF vertical profiles from a given i,j location in WRF
    ds_ret=xr.Dataset()
    for var in varsList:
        if var == 'Z':
           ds_ret[var] = xr.DataArray((0.5*(dsWrf.PH[it,0:-1,j,i]+dsWrf.PH[it,1:,j,i])
                                       +0.5*(dsWrf.PHB[it,0:-1,j,i]+dsWrf.PHB[it,1:,j,i]))/9.81,
                                       dims=(['bottom_top']))
           ## Determine a kMaxPrnt for this i,j to be used for the interpolation step in the calling function 
           kMaxPrnt = np.min(np.where(ds_ret[var].values>zTargTop))+1
        elif var == 'QKE':
            tke_infile = dsWrf.get(var)
            if tke_infile is not None:
                tke_var = 1
            else: # try with TKE_PBL alternatively
                tke_infile = dsWrf.get('TKE_PBL')
                if tke_infile is not None:
                    tke_var = 2
                else:
                    tke_var = 0
            if (tke_var == 1):
                ds_ret[var] = xr.DataArray(0.5*dsWrf[var][it,:,j,i], # QKE is 2.0*TKE
                                           dims=(['bottom_top']))
            elif (tke_var == 2): # TKE_PBL is bottom_top_stag
                ds_ret[var] = xr.DataArray(0.5*(dsWrf['TKE_PBL'][it,0:-1,j,i]+dsWrf['TKE_PBL'][it,1:,j,i]),
                                       dims=(['bottom_top']))
            else: # tke_var == 0 (no tke variable present)
                ds_ret[var] = xr.DataArray(dsWrf['T'][it,:,j,i]*0.0+1e-10,
                                           dims=(['bottom_top']))
        elif 'west_east_stag' in dsWrf[var].dims:
            ds_ret[var] = xr.DataArray(0.5*(dsWrf[var][it,:,j,i]+dsWrf[var][it,:,j,i+1]),
                                       dims=(['bottom_top']))
        elif 'south_north_stag' in dsWrf[var].dims:
            ds_ret[var] = xr.DataArray(0.5*(dsWrf[var][it,:,j,i]+dsWrf[var][it,:,j+1,i]),
                                       dims=(['bottom_top']))
        elif 'bottom_top_stag' in dsWrf[var].dims:
            ds_ret[var] = xr.DataArray(0.5*(dsWrf[var][it,0:-1,j,i]+dsWrf[var][it,1:,j,i]),
                                       dims=(['bottom_top']))
        else:
            ds_ret[var] = xr.DataArray(dsWrf[var][it,:,j,i],
                                       dims=(['bottom_top']))
        if var == 'T':
            ds_ret[var] = 300.0+ds_ret[var]

    for surfVar in surfVarsList:
        ds_ret[surfVar] = xr.DataArray(dsWrf[surfVar][it,j,i])#,
    ## Determine a kMaxPrnt for this i,j to be used for the interpolation step in the calling function  

    return ds_ret.isel(bottom_top=slice(0,kMaxPrnt))


def copyAndTranspose(dsWRF):
    ds=xr.Dataset()
    for var in dsWRF.variables:
        if len(dsWRF[var].dims) == 3:
            ds[var]=xr.DataArray(np.transpose(dsWRF[var].values,(2,0,1)),dims=('zIndex','yIndex','xIndex'))
        elif len(dsWRF[var].dims) == 2:
            ds[var]=xr.DataArray(dsWRF[var].values,dims=('yIndex','xIndex'))  #Note no transpose needed here...
    return ds

#def interp2DForFE(ds,ds_FE,XvWRF,YvWRF,xVec,yVec):
def interp2DForFE(ds,ds_FE,XvWRF,YvWRF,xVec,yVec,parent_model):
    k_val = 1
    k_val_surf = 3
    dsFENew=xr.Dataset()
    for var in ds.variables:
        t1s = time.perf_counter()
        if len(ds[var].dims) == 3:
            tmpVar3d = np.zeros((ds.sizes['zIndex'],ds_FE.sizes['yIndex'],ds_FE.sizes['xIndex']))
            print(tmpVar3d.shape)
            for k in range(ds.sizes['zIndex']):
                if parent_model == 0:
                    fInterp2 = RectBivariateSpline(YvWRF[:,0],XvWRF[0,:],ds[var][k,:,:].values.transpose(), kx=k_val, ky=k_val)
                else:
                    fInterp2 = RectBivariateSpline(YvWRF[:,0],XvWRF[0,:],ds[var][k,:,:].values, kx=k_val, ky=k_val)
                tmp=fInterp2(yVec,xVec)
                tmpVar3d[k,:,:]=tmp
            dsFENew[var]=xr.DataArray(tmpVar3d,dims=('zIndex','yIndex','xIndex'))
        elif len(ds[var].sizes) == 2:
            if parent_model == 0:
               fInterp = RectBivariateSpline(YvWRF[:,0],XvWRF[0,:],ds[var].values.transpose(), kx=k_val_surf, ky=k_val_surf)
            else:
               fInterp = RectBivariateSpline(YvWRF[:,0],XvWRF[0,:],ds[var].values, kx=k_val_surf, ky=k_val_surf)

            tmp=fInterp(yVec,xVec)
            print(tmp.shape)
            dsFENew[var]=xr.DataArray(tmp,dims=('yIndex','xIndex'))
        t1e = time.perf_counter()
        print('{:s} required {:f} s for ij-interpolation'.format(var,t1e-t1s))
    return dsFENew

def create_dsFEFinal(ds_FE,parent_model):
    dsFEFinal=ds_FE.copy(deep=True)
    dsFEFinal.load()
    for var in ['rho', 'u', 'v', 'w', 'theta', 'TKE_0', 'qv', 'ql', 'pressure']:
        dsFEFinal[var]=0.0*dsFEFinal['xPos'] 
    for var in ['fricVel','htFlux','invOblen','qFlux']:
        dsFEFinal[var]=0.0*dsFEFinal['z0m']
    if (parent_model == 0):
        for var in ['XLAT','XLONG','topoWRF','t2','psfc']:
            if var in list(dsFEFinal.variables):
                if var in ['XLAT','XLONG','topoWRF','t2','psfc']:
                    dsFEFinal[var]=0.0*dsFEFinal['tskin']
    return dsFEFinal

def verticalInterpFinal(ds_FE,dsFENew,dsFEFinal,zRect,parent_model):
    z3d=ds_FE['zPos'][:,:,:].values.squeeze()
    for var in dsFENew.variables:
        print(var)
        t1s = time.perf_counter()
        print(var,list(dsFENew[var].dims))
        if 'zIndex' in list(dsFENew[var].dims):
            if 'time' in list(dsFENew[var].dims):
                fld3dRect=dsFENew[var][0,:,:,:].values.squeeze()
            else:
                fld3dRect=dsFENew[var].values
            print(z3d.shape,fld3dRect.shape)
            tmpVar3d=interpolateIrregularVertical(zRect,fld3dRect,z3d)
            if var in list(dsFEFinal.variables):
                dsFEFinal[var][:,:,:]=tmpVar3d.astype(np.float32)
            else:
                print('Omitting {:s} from dsFEFinal...'.format(var))
        else:
            if 'time' in list(dsFENew[var].dims):
                dsFEFinal[var][0,:,:]=dsFENew[var][0,:,:].values.astype(np.float32)
            else:
                if var in list(dsFEFinal.variables):
                  dsFEFinal[var][:,:]=dsFENew[var].values.astype(np.float32)
                else:
                  tmp=np.zeros((dsFEFinal.sizes['yIndex'],dsFEFinal.sizes['xIndex']))
                  tmp[:,:]=dsFENew[var].values.astype(np.float32)
                  dsFEFinal[var]=xr.DataArray(tmp,dims=('yIndex','xIndex'))
        t1e = time.perf_counter()
        print('{:s} required {:f} s for vertical interpolation of the i,j-set'.format(var,t1e-t1s))
    if (parent_model == 0): #Scale the water vapor mixing ratio from kg/kg (WRF) to g/kg (FE)
        if 'qv' in list(dsFEFinal.variables):
            dsFEFinal['qv'] = 1e3*dsFEFinal['qv']
        if 'ql' in list(dsFEFinal.variables):
            dsFEFinal['ql'] = 1e3*dsFEFinal['ql']
        if 'qskin' in list(dsFEFinal.variables):
            dsFEFinal['qskin'] = 1e3*dsFEFinal['qskin']

def interpolateIrregularVertical(zRect,fld3dRect,z3d):
    NzR,NyR,NxR = fld3dRect.shape
    Nz3,Ny3,Nx3 = z3d.shape
    tmpVar3d=np.zeros(z3d.shape)
    for j in range(NyR):
        for i in range(NxR):
            tmp=np.interp(z3d[:,j,i],zRect,fld3dRect[:,j,i])
            tmpVar3d[:,j,i]=tmp
    return tmpVar3d

def create_dsBdy(ds_FE,FEvarsList,FEsurfVarsList,parent_model):
    ds_Bdy=xr.Dataset()
    notit=0
    for var in FEvarsList:
        print(var)
        if var in list(ds_FE.variables):
            ds_Bdy[var+'_YZL']=xr.DataArray(ds_FE[var][:,:,:,0])
            ds_Bdy[var+'_YZH']=xr.DataArray(ds_FE[var][:,:,:,ds_FE.sizes['xIndex']-1])
            ds_Bdy[var+'_XZL']=xr.DataArray(ds_FE[var][:,:,0,:])
            ds_Bdy[var+'_XZH']=xr.DataArray(ds_FE[var][:,:,ds_FE.sizes['yIndex']-1,:])
            ds_Bdy[var+'_XYL']=xr.DataArray(ds_FE[var][:,0,:,:])
            ds_Bdy[var+'_XYH']=xr.DataArray(ds_FE[var][:,ds_FE.sizes['zIndex']-1,:,:])
    if (parent_model == 0): 
        for surfVar in FEsurfVarsList:
            print('{:s}: notit={:d}'.format(surfVar,notit))
            if surfVar in ['topoWRF','topoParent','t2','psfc','tskin','qskin']:
                notit +=1
            else:
                ds_Bdy[surfVar]=xr.DataArray(ds_FE[surfVar][:,:])
    elif (parent_model == 1):
        for surfVar in FEsurfVarsList:
            print('{:s}: notit={:d}'.format(surfVar,notit))
            if surfVar in ['tskin','qskin']:
                notit +=1
            else:
                ds_Bdy[surfVar]=xr.DataArray(ds_FE[surfVar][:,:])

    return ds_Bdy

def createBdysFrom3D(ds_Bdy,ds3D,FEvarsList,FEsurfVarsList):
    for var in FEvarsList:
        if var in ds3D.variables:
            ds_Bdy[var+'_YZL']=ds3D[var][0,:,:,0].expand_dims(dim={'time':ds3D.sizes['time']},axis=0)
            ds_Bdy[var+'_YZH']=ds3D[var][0,:,:,-1].expand_dims(dim={'time':ds3D.sizes['time']},axis=0)
            ds_Bdy[var+'_XZL']=ds3D[var][0,:,0,:].expand_dims(dim={'time':ds3D.sizes['time']},axis=0)
            ds_Bdy[var+'_XZH']=ds3D[var][0,:,-1,:].expand_dims(dim={'time':ds3D.sizes['time']},axis=0)
            ds_Bdy[var+'_XYL']=ds3D[var][0,0,:,:].expand_dims(dim={'time':ds3D.sizes['time']},axis=0)
            ds_Bdy[var+'_XYH']=ds3D[var][0,-1,:,:].expand_dims(dim={'time':ds3D.sizes['time']},axis=0)
        else:
            ds_Bdy[var+'_YZL']=0.0*ds3D['rho'][0,:,:,0].expand_dims(dim={'time':ds3D.sizes['time']},axis=0)
            ds_Bdy[var+'_YZH']=0.0*ds3D['rho'][0,:,:,-1].expand_dims(dim={'time':ds3D.sizes['time']},axis=0)
            ds_Bdy[var+'_XZL']=0.0*ds3D['rho'][0,:,0,:].expand_dims(dim={'time':ds3D.sizes['time']},axis=0)
            ds_Bdy[var+'_XZH']=0.0*ds3D['rho'][0,:,-1,:].expand_dims(dim={'time':ds3D.sizes['time']},axis=0)
            ds_Bdy[var+'_XYL']=0.0*ds3D['rho'][0,0,:,:].expand_dims(dim={'time':ds3D.sizes['time']},axis=0)
            ds_Bdy[var+'_XYH']=0.0*ds3D['rho'][0,-1,:,:].expand_dims(dim={'time':ds3D.sizes['time']},axis=0)
    for surfVar in FEsurfVarsList:
        if surfVar in ds3D.variables:
            ds_Bdy[surfVar]=ds3D[surfVar][0,:,:].expand_dims(dim={'time':ds3D.sizes['time']},axis=0)
        else:
            ds_Bdy[surfVar]=0.0*ds3D['tskin'][0,:,:].expand_dims(dim={'time':ds3D.sizes['time']},axis=0)
    return ds_Bdy

def writeBdyFile(path_out_analysis,fileName,ds_Bdy):
    Bdy_filename = path_out_analysis+fileName #"FE_Bndys_WRF_PDG_w-rhoALT-moist_qv-gpkg.{:d}".format(fileCounter)
    print(Bdy_filename)
    ds_Bdy.isel(time = [0]).to_netcdf(Bdy_filename,format='NETCDF4')  #note the isel(time =[#]) preserves the time dimension

    return

def addTimeDim_FEfinal(dsFEFinal):
    for var in dsFEFinal.variables:
        if len(dsFEFinal[var].values.shape) != 1:
            dsFEFinal[var] = dsFEFinal[var].expand_dims(dim={'time':1},axis=0)
    return

def read_lc_table(filepath):
    df = pd.read_csv(filepath)
    z0_original = {int(k): float(v) for k, v in zip(df.iloc[:,0], df.iloc[:,2])}
    z0_modified = {int(k): float(v) for k, v in zip(df.iloc[:,0], df.iloc[:,3])}
    return z0_original, z0_modified

def SHFR_process_polygons(landcover, buildings, z1, z0_original, z0_modified, nodata=0, N0 = 10, Nmin = 25, fmin = 0.10):
    result = np.zeros_like(landcover, dtype=np.float32)
    labels = np.zeros_like(landcover, dtype=np.int32)
    current_label = 1
    #-------
    r0 = 0
    r1 = 0
    r2 = 0
    r3 = 0
    r4 = 0
    #-------
    for category in np.unique(landcover):
        category_mask = landcover == category
        category_labels, num_labels = ndimage.label(category_mask, structure = ndimage.generate_binary_structure(2,2))
        for i in range(1, num_labels + 1):
            labels[category_labels == i] = current_label
            current_label += 1
    num_polygons = current_label - 1
    print(f'Number of land cover polygons: {num_polygons}')
    for label in range(1, num_polygons + 1):
        polygon_mask = labels == label
        total_area = np.sum(polygon_mask)
        building_mask = (buildings > nodata) & polygon_mask
        building_area = np.sum(building_mask)
        no_building_area = total_area - building_area
        Nreq = np.maximum(Nmin, total_area*fmin)
        lc = landcover[polygon_mask][0]
        if building_area == 0:
            no_building_value = 1.0
            r0 += 1
        else:
            if no_building_area <= N0:
                no_building_value = 1.0
                r1 += 1
            else:
                z = z1[polygon_mask].mean()
                z0lc = z0_original[lc]
                z0st = z0_modified[lc] if z0_modified[lc] > 0.0 else z0lc
                factor_z0 = (np.log(z/z0st+1)*np.log(z/(0.1*z0st)+1)) / (np.log(z/z0lc+1)*np.log(z/(0.1*z0lc)+1))
                if no_building_area < Nreq:
                    w = (no_building_area-N0)/(Nreq-N0)
                    no_building_value = 1.0 + w * np.minimum( 4, (building_area/no_building_area)*factor_z0 )
                    r2 += 1
                else:
                    if (building_area / no_building_area)*factor_z0 > 4:
                        no_building_value = 5.0
                        r4 += 1
                    else:
                        no_building_value = 1.0 + (building_area / no_building_area)*factor_z0
                        r3 += 1
            if (z0_modified[lc] == 0.0 and no_building_value < 1.1):
                no_building_value = 1.0
        result[polygon_mask & ~building_mask] = no_building_value
    print(f'R0 = {r0}, R1 = {r1}, R2 = {r2}, R3 = {r3}, R4 = {r4}')
    return result
