/* FastEddy®: SRC/EXTENSIONS/URBAN/CUDA/cuda_urbanDevice_cu.h
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
#ifndef _URBAN_CUDADEV_CU_H
#define _URBAN_CUDADEV_CU_H

/*urban_ return codes */
#define CUDA_URBAN_SUCCESS               0

/*##############------------------- URBAN submodule variable declarations ---------------------#################*/
/* Parameters */
extern __constant__ int urbanSelector_d;          /* urban selector: 0=off, 1=on */
extern __constant__ float cd_build_d;             /* c_d coefficient (m-1) used by the drag-based building formulation: -c_d|u_i|u_i */
extern __constant__ float ct_build_d;             /* c_t coefficient (s-1) used by the drag-based building formulation: -c_t(rho*theta-rho_b*theta_b) & -c_t(rho-rho_b) */
extern __constant__ float delta_aware_bdg_d;      /* scale-aware correction for building forcing and limiters */
/* array fields */
extern float* building_mask_d;                    /* Base Address of memory containing building mask field: 0 (atmosphere) or 1 (building) */
extern __constant__ int urban_heatRedis_d;        /* selector to activate surface heat redistribution */
extern float *urban_heat_redis_d;                 /* Base Address of memory containing 2d map of heat redistribution coefficient in urban areas */

/*##############-------------- URBAN_CUDADEV submodule function declarations ------------------############*/

/*----->>>>> int cuda_urbanDeviceSetup();      -----------------------------------------------------------------
* Used to cudaMalloc and cudaMemcpy parameters and coordinate arrays for the URBAN_CUDADEV submodule.
*/
extern "C" int cuda_urbanDeviceSetup();

/*----->>>>> int cuda_urbanDeviceCleanup();    ---------------------------------------------------------------
* Used to free all malloced memory by the URBAN_CUDADEV submodule.
*/
extern "C" int cuda_urbanDeviceCleanup();

__global__ void cudaDevice_URBANinter(float* z0m, float* z0t, float* hydroTauFlds, float* moistTauFlds,
                                      float* fricVel, float* htFlux, float* qFlux, float* invOblen,
                                      float* bdg_mask, float* sea_mask, float* urban_redis);
__global__ void cudaDevice_URBANfinal(float* hydroFlds_d, float* hydroFldsFrhs_d, float* hydroBaseStateFlds_d,
	                              float* hydroAuxScalars_d, float* hydroAuxScalarsFrhs_d,
                                      float* hydroFldsFrhsMoist_d,
			      	      float* building_mask_d);

/*----->>>>> __device__ void  cudaDevice_UrbanDragMethod();  --------------------------------------------------
* This cuda kerne lsets up the cells and their id in the urban drag-based approach
*/
__device__ void cudaDevice_UrbanDragMethod(float* rho, float* u, float* v, float* w, float* th, float* th_base, float* rho_base, float* Frhs_u, float* Frhs_v, float* Frhs_w, float* Frhs_th, float* Frhs_rho, float* bdg_mask);
__device__ void cudaDevice_UrbanDragMethodMoist(float* Frhs_qMoistFld, float* bdg_mask);
__device__ void cudaDevice_UrbanDragMethodAuxScalar(float* AuxScalar, float* Frhs_AuxScalar, float* bdg_mask);

#endif // _URBAN_CUDADEV_CU_H
