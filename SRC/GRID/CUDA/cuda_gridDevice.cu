/* FastEddy®: SRC/GRID/CUDA/cuda_gridDevice.cu 
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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <float.h>
#include <math.h>
#include <fempi.h>
#include <grid.h>
#include <fecuda_Device_cu.h>
#include <cuda_gridDevice_cu.h>

/*######################------------------- GRID module variable declarations ---------------------#################*/
/* Parameters */
__constant__ int Nh_d;     //Number of halo cells to be used (dependent on largest stencil extent
__constant__ int Nx_d;     //Computational domain extents in the x, y, and z directions 
__constant__ int Ny_d;
__constant__ int Nz_d;
__constant__ float dX_d; //Computational domain fixed resolutions (i, j, k respectively)
__constant__ float dY_d;
__constant__ float dZ_d;
__constant__ float dXi_d; //Computational domain fixed inverse-resolutions (i, j, k respectively)
__constant__ float dYi_d;
__constant__ float dZi_d;
__constant__ int iMin_d;//Constant min and max bounds of i-index for only non-halos cells of the cuda domain
__constant__ int iMax_d;
__constant__ int jMin_d;//Constant min and max bounds of j-index for only non-halos cells of the cuda domain
__constant__ int jMax_d;
__constant__ int kMin_d;//Constant min and max bounds of k-index for only non-halos cells of the cuda domain
__constant__ int kMax_d;

/* array fields */
float *xPos_d;  // Cell-center position in x (meters) 
float *yPos_d;  // Cell-center position in y (meters) 
float *zPos_d;  // Cell-center position in z (meters) 
float *topoPos_d; //Topography elevation (z in meters) at the cell center position in x and y. 

float *J31_d;      // dz/d_xi
float *J32_d;      // dz/d_eta
float *J33_d;      // dz/d_zeta

float *D_Jac_d;    //Determinant of the Jacbian  (called scale factor i.e. if d_xi=d_eta=d_zeta=1, then cell volume)
float *invD_Jac_d; //inverse Determinant of the Jacbian 

/*#################------------------- CUDA_GRID module function definitions ---------------------#################*/
/*----->>>>> int cuda_gridDeviceSetup();       ----------------------------------------------------------------------
 * Used to cudaMalloc and cudaMemcpy parameters and coordinate arrays, and for the GRID_CUDA module.
*/
extern "C" int cuda_gridDeviceSetup(){
   int errorCode = CUDA_GRID_SUCCESS;
   int Nelems;
#ifdef DEBUG 
   cudaEvent_t startE, stopE;
   float elapsedTime;
#endif
   /*Synchronize the Device*/
   gpuErrchk( cudaDeviceSynchronize() );
  
//#ifdef DEBUG 
#if 1 
   printf("cuda_gridDeviceSetup:-- (i,j,k)Min, ()Max = (%d,%d,%d), (%d,%d,%d)\n",
                                                    iMin,jMin,kMin,iMax,jMax,kMax);
#endif
   /*Constants*/
   /* number of halo-cell and  grid array extents */
   cudaMemcpyToSymbol(Nh_d, &Nh, sizeof(int));
   cudaMemcpyToSymbol(Nx_d, &Nxp, sizeof(int));
   cudaMemcpyToSymbol(Ny_d, &Nyp, sizeof(int));
   cudaMemcpyToSymbol(Nz_d, &Nzp, sizeof(int));
   /* cell resolutions by dimension and inverses */
   cudaMemcpyToSymbol(dX_d, &dX, sizeof(float));
   cudaMemcpyToSymbol(dY_d, &dY, sizeof(float));
   cudaMemcpyToSymbol(dZ_d, &dZ, sizeof(float));
   cudaMemcpyToSymbol(dXi_d, &dXi, sizeof(float));
   cudaMemcpyToSymbol(dYi_d, &dYi, sizeof(float));
   cudaMemcpyToSymbol(dZi_d, &dZi, sizeof(float));
   /* min and max loop indices by dimension for non-halo inclusive domain*/
   cudaMemcpyToSymbol(iMin_d, &iMin, sizeof(int));
   cudaMemcpyToSymbol(iMax_d, &iMax, sizeof(int));
   cudaMemcpyToSymbol(jMin_d, &jMin, sizeof(int));
   cudaMemcpyToSymbol(jMax_d, &jMax, sizeof(int));
   cudaMemcpyToSymbol(kMin_d, &kMin, sizeof(int));
   cudaMemcpyToSymbol(kMax_d, &kMax, sizeof(int));
   gpuErrchk( cudaPeekAtLastError() ); /*Check for errors in the cudaMemCpy calls*/

   /*Set the full memory block number of elements for grid fields*/
   Nelems = (Nxp+2*Nh)*(Nyp+2*Nh)*(Nzp+2*Nh); 
   /* Allocate the GRID arrays */
   /* Coordinate Arrays */
   fecuda_DeviceMalloc(Nelems*sizeof(float), &xPos_d);
   fecuda_DeviceMalloc(Nelems*sizeof(float), &yPos_d);
   fecuda_DeviceMalloc(Nelems*sizeof(float), &zPos_d);
   fecuda_DeviceMalloc(((Nxp+2*Nh)*(Nyp+2*Nh))*sizeof(float), &topoPos_d);
   /* Metric Tensors Fields */
   fecuda_DeviceMalloc(Nelems*sizeof(float), &J31_d);
   fecuda_DeviceMalloc(Nelems*sizeof(float), &J32_d);
   fecuda_DeviceMalloc(Nelems*sizeof(float), &J33_d);
   fecuda_DeviceMalloc(Nelems*sizeof(float), &D_Jac_d);
   fecuda_DeviceMalloc(Nelems*sizeof(float), &invD_Jac_d);
   gpuErrchk( cudaPeekAtLastError() ); /*Check for errors in the cudaMalloc calls*/

   /* cudaMemcpy the GRID arrays from Host to Device*/
   /* Coordinate Arrays */
   cudaMemcpy(xPos_d, xPos, Nelems*sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(yPos_d, yPos, Nelems*sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(zPos_d, zPos, Nelems*sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(topoPos_d, topoPos, ((Nxp+2*Nh)*(Nyp+2*Nh))*sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(J31_d, J31, Nelems*sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(J32_d, J32, Nelems*sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(J33_d, J33, Nelems*sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(D_Jac_d, D_Jac, Nelems*sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(invD_Jac_d, invD_Jac, Nelems*sizeof(float), cudaMemcpyHostToDevice);
   gpuErrchk( cudaPeekAtLastError() ); /*Check for errors in the cudaMemCpy calls*/
 
#ifdef DEBUG
   /*Launch an independent GPU calculation of the GRID arrays*/
   /*Synchronize the Device*/
   gpuErrchk( cudaDeviceSynchronize() );
   createAndStartEvent(&startE, &stopE);
   printf("Calling cudaDevice_calculateJacobians...\n");
   printf("grid = {%d, %d, %d}\n",grid.x, grid.y, grid.z);
   printf("tBlock = {%d, %d, %d}\n",tBlock.x, tBlock.y, tBlock.z);
   cudaDevice_calculateJacobians<<<grid, tBlock>>>(J31_d, J32_d, J33_d,
                                                  D_Jac_d, invD_Jac_d, xPos_d, yPos_d, zPos_d);
   gpuErrchk( cudaPeekAtLastError() );
   gpuErrchk( cudaDeviceSynchronize() );
   stopSynchReportDestroyEvent(&startE, &stopE, &elapsedTime);
   gpuErrchk( cudaPeekAtLastError() );
   gpuErrchk( cudaDeviceSynchronize() );
   printf("cuda_calculateJacobians()  Kernel execution time (ms): %12.8f\n", elapsedTime);

   /* cudaMemcpy the GPU-computed GRID arrays from Device Host*/
   /* Coordinate Arrays */
   cudaMemcpy(J31, J31_d, Nelems*sizeof(float), cudaMemcpyDeviceToHost);
   cudaMemcpy(J32, J32_d, Nelems*sizeof(float), cudaMemcpyDeviceToHost);
   cudaMemcpy(J33, J33_d, Nelems*sizeof(float), cudaMemcpyDeviceToHost);
   cudaMemcpy(D_Jac, D_Jac_d, Nelems*sizeof(float), cudaMemcpyDeviceToHost);
   cudaMemcpy(invD_Jac, invD_Jac_d, Nelems*sizeof(float), cudaMemcpyDeviceToHost);
   gpuErrchk( cudaPeekAtLastError() ); /*Check for errors in the cudaMemCpy calls*/
#endif


   /* Done */
   return(errorCode);
} //end cuda_gridDeviceSetup()

/*----->>>>> extern "C" int cuda_gridDeviceCleanup();  -----------------------------------------------------------
Used to free all malloced memory by the GRID module.
*/
extern "C" int cuda_gridDeviceCleanup(){
   int errorCode = GRID_SUCCESS;

   /* Free any GRID module arrays */
    /* metric tensor fields */
   cudaFree(J31_d); 
   cudaFree(J32_d); 
   cudaFree(J33_d); 
   cudaFree(D_Jac_d); 
   cudaFree(invD_Jac_d); 
    /* coordinate fields */
   cudaFree(xPos_d); 
   cudaFree(yPos_d); 
   cudaFree(zPos_d); 
   cudaFree(topoPos_d); 
   gpuErrchk( cudaPeekAtLastError() ); /*Check for errors in the cudaMemCpy calls*/
 
   return(errorCode);

}//end cuda_gridDeviceCleanup()

/*----->>>>> __global__ void  cudaDevice_calculateJacobians();  --------------------------------------------------
This is the cuda version of the calculateJacobians routine from the GRID module
*/
__global__ void cudaDevice_calculateJacobians(float *J31_d, float *J32_d, float *J33_d,
                                              float *D_Jac_d, float *invD_Jac_d,
                                              float *xPos_d, float *yPos_d, float *zPos_d){
  int i,j,k;
  int ijk,ip1jk,im1jk,ijp1k,ijm1k,ijkp1,ijkm1;
  int iStride,jStride,kStride;
  float T[9];
  float inv_2dxi; 
  float inv_2deta; 
  float inv_2dzeta; 

  /*Set a couple of constants*/
  inv_2dxi = 1.0/(2.0*dX_d);
  inv_2deta = 1.0/(2.0*dY_d);
  inv_2dzeta = 1.0/(2.0*dZ_d);

  /*Establish necessary indices for the stencil operation*/
  i = (blockIdx.x)*blockDim.x + threadIdx.x;
  j = (blockIdx.y)*blockDim.y + threadIdx.y;
  k = (blockIdx.z)*blockDim.z + threadIdx.z;
  
  iStride = (Ny_d+2*Nh_d)*(Nz_d+2*Nh_d);
  jStride = (Nz_d+2*Nh_d);
  kStride = 1;
  ijk =     i  *iStride +   j  *jStride +   k  *kStride;
  ip1jk = (i+1)*iStride +   j  *jStride +   k  *kStride;
  im1jk = (i-1)*iStride +   j  *jStride +   k  *kStride;
  ijp1k =   i  *iStride + (j+1)*jStride +   k  *kStride;
  ijm1k =   i  *iStride + (j-1)*jStride +   k  *kStride;
  ijkp1 =   i  *iStride +   j  *jStride + (k+1)*kStride;
  ijkm1 =   i  *iStride +   j  *jStride + (k-1)*kStride;

  /*if this thread is in the range of non-halo cells*/
  if((i >= iMin_d+1)&&(i < iMax_d-1) && 
     (j >= jMin_d+1)&&(j < jMax_d-1) && 
     (k >= kMin_d+1)&&(k < kMax_d-1) ){
     
     T[0] = (xPos_d[ip1jk]-xPos_d[im1jk])*inv_2dxi;   //T[0][0]
     T[1] = (yPos_d[ip1jk]-yPos_d[im1jk])*inv_2dxi;   //T[0][1]
     T[2] = (zPos_d[ip1jk]-zPos_d[im1jk])*inv_2dxi;   //T[0][2]

     T[3] = (xPos_d[ijp1k]-xPos_d[ijm1k])*inv_2deta;  //T[1][0]
     T[4] = (yPos_d[ijp1k]-yPos_d[ijm1k])*inv_2deta;  //T[1][1]
     T[5] = (zPos_d[ijp1k]-zPos_d[ijm1k])*inv_2deta;  //T[1][2]
     
     T[6] = (xPos_d[ijkp1]-xPos_d[ijkm1])*inv_2dzeta; //T[2][0]
     T[7] = (yPos_d[ijkp1]-yPos_d[ijkm1])*inv_2dzeta; //T[2][1]
     T[8] = (zPos_d[ijkp1]-zPos_d[ijkm1])*inv_2dzeta; //T[2][2]
         
     D_Jac_d[ijk] =  (T[4]*T[8] - T[5]*T[7])*T[0] 
                    -(T[1]*T[8] - T[2]*T[7])*T[3]
                    +(T[1]*T[5] - T[2]*T[4])*T[6];   
     /*Ensure no divide by zero*/
     if(fabsf(D_Jac_d[ijk]) < 1e-6){
            D_Jac_d[ijk] = 1e-6;
     }
     /*Set the inverse*/
     invD_Jac_d[ijk] = 1.0/D_Jac_d[ijk];
   
#ifdef DEBUG 
     //if(isnan(D_Jac_d[ijk])||isinf(D_Jac_d[ijk])||(fabsf(D_Jac_d[ijk])==0)){ 
     //if((i==iMin_d)&&(j==jMin_d)&&(k==kMin_d)){
/*     if((i<10)&&(j<10)&&(k<10)){
      printf("cudaDevice_calculateJacobians: D_Jac_d[%d,%d,%d] = %f-- %d, %d, %d, (%f, %f, %f), (%f,%f,%f), (%f,%f,%f) \n",i,j,k,D_Jac_d[ijk],Nx_d,Ny_d,Nz_d,T[0],T[1],T[2],T[3],T[4],T[5],T[6],T[7],T[8]); 
     }//finite check */
#endif
     /*Set the elements*/
     /*d(x,y,z)/d_zetaa*/
     J31_d[ijk] = (T[3]*T[7] - T[4]*T[6])*invD_Jac_d[ijk];
     J32_d[ijk] = (T[0]*T[7] - T[1]*T[6])*invD_Jac_d[ijk];
     J33_d[ijk] = (T[0]*T[4] - T[1]*T[3])*invD_Jac_d[ijk];
#ifdef DEBUG
   if((i==64)&&(j==64)&&(k==64)){
     printf("At (%d, %d, %d):\n\t\t J31=%f, J32=%f, J33=%f,\n\t\t D_Jac=%f, invD_Jac=%f\n",
             i,j,k, J31_d[ijk],J32_d[ijk],J33_d[ijk],
                    D_Jac_d[ijk],invD_Jac_d[ijk]);
   }
#endif
  }//if in the range of non-halo cells
} // end cudaDevice_calculateJacobians()
