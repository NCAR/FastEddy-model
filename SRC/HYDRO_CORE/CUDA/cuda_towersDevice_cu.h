/* FastEddy®: SRC/HYDRO_CORE/CUDA/cuda_towersDevice_cu.h 
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
#ifndef _TOWERS_CUDADEV_CU_H
#define _TOWERS_CUDADEV_CU_H

/*coriolis return codes */
#define CUDA_TOWERS_SUCCESS    0

/*##############------------------- TOWERS submodule variable declarations ---------------------#################*/
/*---TOWERS*/
extern __constant__ int rank_nTowers_d;   /*Number of towers within an mpi_rank_world subdomain*/
extern __constant__ int towerInstanceSize_d;    /*size of a timestep instance of all tower data*/
extern __constant__ int towerSurfInstanceSize_d;    /*size of a timestep instance of all tower surface data*/
extern float* towersData_d; /* Data Structure to store virtual tower data on device*/
extern float* towersSurfData_d; /* Data Structure to store virtual tower surface ata on device*/
extern int* tower_iInds_d; /*rank-centric i-indices */
extern int* tower_jInds_d; /*rank-centric j-indices */

/*##############-------------- TOWERS_CUDADEV submodule function declarations ------------------############*/

/*----->>>>> int cuda_towersDeviceSetup();       ---------------------------------------------------------
* Used to cudaMemcpy parameters and allocate arrays for the TOWERS_CUDA submodule.
*/
extern "C" int cuda_towersDeviceSetup(int NtBatch, int rank_nTowers, int towerInstanceSize, int towerSurfInstanceSize);

/*----->>>>> extern "C" int cuda_towersDeviceCleanup();  -----------------------------------------------------------
* Used to free all malloced memory by the TOWERS submodule.
*/
extern "C" int cuda_towersDeviceCleanup();

/*----->>>>> __global__ void  cudaDevice_towerAppendBuffers();  ------------------------------------------
* This is the gloabl-entry kernel routine for updating tower device-sided buffers
*/
//__global__ void cudaDevice_towerAppendBuffers(int itBatch, int NtBatch,
__global__ void cudaDevice_towerAppendBuffers(int itBatch, int batchSize,
		                              float *towersData_d, float *towersSurfData_d, int *tower_iInds_d, int *tower_jInds_d,
		                              int Nhydro, float *hydroFlds_d,
                                              int Nsgstke, float *sgstkeScalars_d,
                                              int Nmoist, float *moistScalars_d,
                                              int NauxSc, float *hydroAuxScalars_d,
                                              int Ntaus, float *hydroTauFlds_d,
					      int NtausMoist, float *moistTauFlds_d,
                                              float *z0m_d, float *z0t_d, float *tskin_d, float *qskin_d,
                                              float *fricVel_d, float *invOblen_d, float *htFlux_d, float *qFlux_d);

#endif // _TOWERS_CUDADEV_CU_H
