import argparse
import math
import time
from scipy.ndimage import gaussian_filter
import numpy as np

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
                    local_blur_center = gaussian_filter(local_area, sigma=sigTry)
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
