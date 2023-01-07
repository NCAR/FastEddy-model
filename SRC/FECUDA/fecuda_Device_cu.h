/* FastEddy®: SRC/FECUDA/fecuda_Device_cu.h 
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
#ifndef _FECUDA_DEV_CU_H
#define _FECUDA_DEV_CU_H

/*fecuda return codes */
#define FECUDA_DEV_CU_SUCCESS               0

/*fecuda includes*/
#include <cuda.h>
#include <cuda_runtime.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      //fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      printf("GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/*##################------------------- FECUDA module variable declarations ---------------------#################*/
/* Parameters */
extern dim3 tBlock;
extern dim3 grid;

extern __constant__ int mpi_size_world_d;
extern __constant__ int mpi_rank_world_d;
extern __constant__ int numProcsX_d;
extern __constant__ int numProcsY_d;
extern __constant__ int rankXid_d;
extern __constant__ int rankYid_d;

/*##################------------------- FECUDA module function declarations ---------------------#################*/

/*----->>>>> int fecuda_DeviceSetup();       ----------------------------------------------------------------------
 * Used to set the "dim3 tblock" module variable that is passed to any device kernel to specify the number of 
 * threads per block in each dimension 
*/
extern "C" int fecuda_DeviceSetup(int tBx, int tBy, int tBz);

/*----->>>>> void fecuda_DeviceMallocInt();    -----------------------------------------------------------
* Used to allocate device memory integer blocks and set the  host memory addresses of device memory pointers.
*/
extern "C" void fecuda_DeviceMallocInt(int Nelems, int** memBlock_d);

/*----->>>>> int fecuda_SetBlocksPerGrid();   ------------------------------------------------------------------
 * Used to set the "dim3 grid" module variable that is passed to any device kernel 
 * to specify the number of blocks per grid in each dimenaion
*/
extern "C" int fecuda_DeviceSetBlocksPerGrid(int Nx, int Ny, int Nz);

/*----->>>>> void fecuda_logDeviceProperties(cudaDeviceProp dProps, int dNum);-----------------------------------------------
 * This routine logs (to stdout) the CudaDeviceProperties on a given architecture
*/
extern "C" void fecuda_logDeviceProperties(cudaDeviceProp dProps, int dNum);

/*----->>>>> int fecuda_DeviceCleanup();       ----------------------------------------------------------------------
Used to free all malloced memory by the FECUDA module.
*/
extern "C" int fecuda_DeviceCleanup();

/*---UTILS*/
#include <fecuda_Utils_cu.h>

/*---PLUGINS*/
#include <fecuda_PlugIns_cu.h>

#endif // _FECUDA_DEV_CU_H
