/* FastEddy®: SRC/GRID/CUDA/cuda_gridDevice_cu.h 
* ©2016 University Corporation for Atmospheric Research
* 
* This file is licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
* 
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/
#ifndef _GRID_CUDADEV_CU_H
#define _GRID_CUDADEV_CU_H

/*grid_ return codes */
#define CUDA_GRID_SUCCESS               0

/*######################------------------- GRID module variable declarations ---------------------#################*/
/* Parameters */
//#pragma once
extern __constant__ int Nh_d;     //Number of halo cells to be used (dependent on largest stencil extent
extern __constant__ int Nx_d;     //Computational domain extents in the x, y, and z directions 
extern __constant__ int Ny_d;
extern __constant__ int Nz_d;
extern __constant__ float dX_d; //Computational domain fixed resolutions (i, j, k respectively)
extern __constant__ float dY_d;
extern __constant__ float dZ_d; 
extern __constant__ float dXi_d; //Computational domain fixed inverse-resolutions (i, j, k respectively)
extern __constant__ float dYi_d;
extern __constant__ float dZi_d; 
extern __constant__ int iMin_d;//Constant min and max bounds of i-index for only non-halos cells of the cuda domain
extern __constant__ int iMax_d; 
extern __constant__ int jMin_d;//Constant min and max bounds of j-index for only non-halos cells of the cuda domain
extern __constant__ int jMax_d; 
extern __constant__ int kMin_d;//Constant min and max bounds of k-index for only non-halos cells of the cuda domain
extern __constant__ int kMax_d; 

 
/* array fields */
extern float *xPos_d;  /* Cell-center position in x (meters) */
extern float *yPos_d;  /* Cell-center position in y (meters) */
extern float *zPos_d;  /* Cell-center position in z (meters) */
extern float *topoPos_d; /*Topography elevation (z in meters) at the cell center position in x and y. */

extern float *J31_d;      // dz/d_xi
extern float *J32_d;      // dz/d_eta
extern float *J33_d;      // dz/d_zeta

extern float *D_Jac_d;    //Determinant of the Jacbian  (called scale factor i.e. if d_xi=d_eta=d_zeta=1, then cell volume)
extern float *invD_Jac_d; //inverse Determinant of the Jacbian 

/*#################------------------- GRID_CUDADEV module function declarations ---------------------##############*/

/*----->>>>> int cuda_gridDeviceSetup();       ----------------------------------------------------------------------
Used to cudaMalloc and cudaMemcpy parameters and coordinate arrays for the GRID_CUDADEV module.
*/
extern "C" int cuda_gridDeviceSetup();

/*----->>>>> int cuda_gridDeviceCleanup();       ----------------------------------------------------------------------
Used to free all malloced memory by the GRID_CUDADEV module.
*/
extern "C" int cuda_gridDeviceCleanup();

/*#########------------------- GRID_CUDADEV module device function declarations ---------------------##############*/

/*----->>>>> __global__ void  cudaDevice_calculateJacobians();  --------------------------------------------------
 * This is the cuda version of the calculateJacobians routine from the GRID module
 * */
__global__ void cudaDevice_calculateJacobians(float *J31_d, float *J32_d, float *J33_d,
                                              float *D_Jac_d, float *invD_Jac_d,
                                              float *xPos_d, float *yPos_d, float *zPos_d);

#endif // _GRID_CUDADEV_CU_H
