/* FastEddy®: SRC/FECUDA/fecuda_Device.cu 
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
#include <fempi.h>
#include <fecuda.h>
#include <fecuda_Device_cu.h>

//INCLUDED SOURCE FILES
#include "fecuda_Utils.cu"
#include "fecuda_PlugIns.cu"

/*############------------------- FECUDA module internal function declarations ---------------------#############*/
dim3 tBlock; //Module Global configuration parameter for threadBlocks
dim3 grid;   //Module Global Configuration parameters for grids of threadBlocks

__constant__ int mpi_size_world_d;
__constant__ int mpi_rank_world_d;
__constant__ int numProcsX_d;
__constant__ int numProcsY_d;
__constant__ int rankXid_d;
__constant__ int rankYid_d;

/*######################------------------- FECUDA module function definitions ---------------------##############*/

/*----->>>>> int fecuda_DeviceSetup();       ----------------------------------------------------------------------
Used to broadcast and print parameters, allocate memory, and initialize configuration settings for  the FECUDA module.
*/
extern "C" int fecuda_DeviceSetup(int tBx, int tBy, int tBz){
   int errorCode = FECUDA_SUCCESS;
   int numDevs;  /* the number of devices present */
   int iDev;    /* an index for iterations over the devices */ 
   cudaDeviceProp* devProps; /*the device properties structure*/

   printf("mpi_rank_world--%d/%d Obtaining devices...\n",mpi_rank_world, mpi_size_world);
   fflush(stdout);
   numDevs = 0;
   cudaGetDeviceCount(&numDevs);
   gpuErrchk( cudaPeekAtLastError() );    //Log an issue if there was one...
   if(numDevs > 0){
      printf("Obtaining properties for %d devices...\n\n",numDevs);
      fflush(stdout);
      cudaSetDevice(mpi_rank_world % numDevs); 
      printf("mpi_rank_world--%d/%d assigned accelerator device %d/%d...\n",
              mpi_rank_world, mpi_size_world, mpi_rank_world % numDevs, numDevs);
      fflush(stdout);
      /*malloc vector of devProps*/
      devProps = (cudaDeviceProp *) malloc(sizeof(cudaDeviceProp));
      iDev = mpi_rank_world % numDevs;
      cudaGetDeviceProperties(&devProps[0], iDev);
      gpuErrchk( cudaPeekAtLastError() );    
      gpuErrchk( cudaDeviceSynchronize() );  
      fecuda_logDeviceProperties(devProps[0],iDev);
      MPI_Barrier(MPI_COMM_WORLD);  
      printf("\n");
      printf("mpi_rank_world--%d/%d done!\n",
              mpi_rank_world, mpi_size_world);
      fflush(stdout);
   }else{
      printf("No CUDA devices found...exiting now!\n");
      fflush(stdout);
      exit(0);
   } // if numDevs > 0
   MPI_Barrier(MPI_COMM_WORLD);  
   gpuErrchk( cudaDeviceSynchronize() );  
   
   MPI_Barrier(MPI_COMM_WORLD);  
   printf("mpi_rank_world--%d/%d Assigning threadPerBlock.[xyz]\n",mpi_rank_world, mpi_size_world);
   fflush(stdout);
   tBlock.x = tBx;
   tBlock.y = tBy;
   tBlock.z = tBz;
   printf("mpi_rank_world--%d/%d done!\n",mpi_rank_world, mpi_size_world);
   fflush(stdout);

   cudaMemcpyToSymbol(mpi_size_world_d, &mpi_size_world, sizeof(int));      
   cudaMemcpyToSymbol(mpi_rank_world_d, &mpi_rank_world, sizeof(int));      
   cudaMemcpyToSymbol(numProcsX_d, &numProcsX, sizeof(int));      
   cudaMemcpyToSymbol(numProcsY_d, &numProcsY, sizeof(int));      
   cudaMemcpyToSymbol(rankXid_d, &rankXid, sizeof(int));      
   cudaMemcpyToSymbol(rankYid_d, &rankYid, sizeof(int));

   //Setup for any reductions
   tBlock_red.x = tBx_red;
   tBlock_red.y = tBy_red;
   tBlock_red.z = tBz_red;

   return(errorCode);
} //end fecuda_DeviceSetup()

/*----->>>>> int fecuda_DeviceSetBlocksPerGrid(); -----------------------------------------------------------------
Used to broadcast and print parameters, allocate memory, and initialize configuration settings for  the FECUDA module.
*/
extern "C" int fecuda_DeviceSetBlocksPerGrid(int Nx, int Ny, int Nz){ //These extents are halo-inclusive (per-rank)
   int errorCode = FECUDA_SUCCESS;

   grid.x = ceil((float) Nx/(float) tBlock.x);
   grid.y = ceil((float) Ny/(float) tBlock.y);
   grid.z = ceil((float) Nz/(float) tBlock.z);

   /* log the configuration parameters */
   printf("fecuda_SetBlocksPerGrid: tBlock = {%d, %d, %d}\n",tBlock.x, tBlock.y, tBlock.z);
   printf("fecuda_SetBlocksPerGrid: grid = {%d, %d, %d}\n",grid.x, grid.y, grid.z);
   fflush(stdout);

   //Setup for any reductions out of fecuda_Utils.cu
   grid_red.x = ceil((float) Nx/(float) tBlock_red.x);
   grid_red.y = ceil((float) Ny/(float) tBlock_red.y);
   grid_red.z = ceil((float) Nz/(float) tBlock_red.z);

   return(errorCode);
} //end fecuda_DeviceSetBlocksPerGrid()

/*----->>>>> void fecuda_logDeviceProperties(cudaDeviceProp dProps, int dNum);-----------------------------------------------
This routine logs (to stdout) the CudaDeviceProperties on a given architecture
*/
void fecuda_logDeviceProperties(cudaDeviceProp dProps, int dNum){
   printf("Device Number:          %d\n", dNum);
   printf("---------------------------------------------------------------------\n");
   printf("\n__General__\n");
   printf("   Name                               : %s\n",dProps.name);
   printf("   Compute Capability                 : %d.%d\n",dProps.major,dProps.minor);
   printf("   Clock Rate                         : %d\n",dProps.clockRate);
   printf("   Number of multiprocessors          : %d\n",dProps.multiProcessorCount);
   printf("   Max Threads/multiprocessor         : %d\n",dProps.maxThreadsPerMultiProcessor);
   printf("   Concurrent Copy/Execution          : %s\n",(dProps.deviceOverlap ? "Yes" : "No"));
   printf("   Concurrent Kernels                 : %d\n",dProps.concurrentKernels);
   printf("   Compute Mode                       : %d\n",dProps.computeMode);
   printf("   Kernel Timeout Enabled             : %s\n",(dProps.kernelExecTimeoutEnabled ? "Yes" : "No"));
   printf("   Integrated                         : %s\n",(dProps.integrated ? "Yes" : "No"));
   printf("   Host Memory Mapping                : %s\n",(dProps.canMapHostMemory ? "Yes" : "No"));
   printf("\n__Configuration__\n");
   printf("   Registers/Block                    : %d\n",dProps.regsPerBlock);
   printf("   Warp Size                          : %d\n",dProps.warpSize);
   printf("   Max Threads/Block                  : %d\n",dProps.maxThreadsPerBlock);
   printf("   Max Block-extent / Dimension       : %12d %12d %12d\n",dProps.maxThreadsDim[0],dProps.maxThreadsDim[1],dProps.maxThreadsDim[2]);
   printf("   Max Grid-extent / Dimension        : %12d %12d %12d\n",dProps.maxGridSize[0],dProps.maxGridSize[1],dProps.maxGridSize[2]);
   printf("\n__Memory__\n");
   printf("   Tot. Global Memory (GB) / (B)      : %10.5f / %u\n", (((double)dProps.totalGlobalMem) / pow(1024.0,3)),dProps.totalGlobalMem);
   printf("   Shared Mem/Block   (GB) / (B)      : %10.5f / %u\n", (((double)dProps.sharedMemPerBlock) / pow(1024.0,3)),dProps.sharedMemPerBlock);
   printf("   Constant Memory    (GB) / (B)      : %10.5f / %u\n", (((double)dProps.totalConstMem) / pow(1024.0,3)),dProps.totalConstMem);
   printf("   Max 1-D Texture                    : %12d\n",dProps.maxTexture1D);
   printf("   Max 2-D Texture                    : %12d %12d\n",dProps.maxTexture2D[0], dProps.maxTexture2D[1]);
   printf("   Max 3-D Texture                    : %12d %12d %12d\n",dProps.maxTexture3D[0], dProps.maxTexture3D[1], dProps.maxTexture3D[2]);
   printf("   Texture Alignment  (B)             : %u\n",dProps.textureAlignment);
   printf("   Max Memory pitch                   : %u\n",dProps.memPitch);
   return;
} // end logDeviceProperties

/*----->>>>> int fecuda_DeviceCleanup(); -------------------------------------------------------------------
Used to free all malloced memory by the FECUDA module.
*/
extern "C" int fecuda_DeviceCleanup(){
   int errorCode = FECUDA_SUCCESS;

   /* Free any FECUDA module device-arrays */

   return(errorCode);

}//end fecuda_Cleanup()
