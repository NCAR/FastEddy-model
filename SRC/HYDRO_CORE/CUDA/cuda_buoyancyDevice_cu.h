/* FastEddy®: SRC/HYDRO_CORE/CUDA/cuda_buoyancyDevice_cu.h 
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
#ifndef _BUOYANCY_CUDADEV_CU_H
#define _BUOYANCY_CUDADEV_CU_H

/*buoyancy return codes */
#define CUDA_BUOYANCY_SUCCESS    0

/*##############------------------- BUOYANCY submodule variable declarations ---------------------#################*/
/*---BUOYANCY*/
extern __constant__ int buoyancySelector_d;   /*buoyancy Force selector: 0=off, 1=on*/

/*##############-------------- BUOYANCY_CUDADEV submodule function declarations ------------------############*/

/*----->>>>> int cuda_buoyancyDeviceSetup();       ---------------------------------------------------------
* Used to cudaMalloc and cudaMemcpy parameters and coordinate arrays, and for the BUOYANCY_CUDA submodule.
*/
extern "C" int cuda_buoyancyDeviceSetup();

/*----->>>>> extern "C" int cuda_buoyancyDeviceCleanup();  -----------------------------------------------------------
* Used to free all malloced memory by the BUOYANCY submodule.
*/
extern "C" int cuda_buoyancyDeviceCleanup();

/*----->>>>> __device__ void  cudaDevice_calcBuoyancy();  --------------------------------------------------
* This is the cuda version of the calcBuoyancy routine from the HYDRO_CORE module
*/
__device__ void cudaDevice_calcBuoyancy(float* Frhs_w, float* rho, float* rho_BS);

/*----->>>>> __device__ void  cudaDevice_calcBuoyancyMoistNvar1();  --------------------------------------------------
* Bouyancy term for single vapor only moisture species + dry air (see Klemp 2007 MWR)
*/
__device__ void cudaDevice_calcBuoyancyMoistNvar1(float* Frhs_w, float* rho, float* rho_BS, float* rho_v);

/*----->>>>> __device__ void  cudaDevice_calcBuoyancyMoistNvar2();  --------------------------------------------------
* Bouyancy term for vapor + liquid  moisture species + dry air (see Klemp 2007 MWR)
*/
__device__ void cudaDevice_calcBuoyancyMoistNvar2(float* Frhs_w, float* rho, float* rho_BS, float* rho_v, float * rho_l);

#endif // _BUOYANCY_CUDADEV_CU_H
