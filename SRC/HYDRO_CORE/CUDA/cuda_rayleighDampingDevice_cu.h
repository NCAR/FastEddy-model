/* FastEddy®: SRC/HYDRO_CORE/CUDA/cuda_rayleighDampingDevice_cu.h 
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
#ifndef _RAYLEIGHDAMPING_CUDADEV_CU_H
#define _RAYLEIGHDAMPING_CUDADEV_CU_H

/*rayleighDamping return codes */
#define CUDA_RAYLEIGHDAMPING_SUCCESS    0

/*##############------------- RAYLEIGHDAMPING submodule variable declarations ---------------------#################*/
/*---RAYLEIGH DAMPING LAYER*/
extern __constant__ int dampingLayerSelector_d;       // Rayleigh Damping Layer selector
extern __constant__ float dampingLayerDepth_d;       // Rayleigh Damping Layer Depth

/*##############------------ RAYLEIGHDAMPING_CUDADEV submodule function declarations ------------------############*/

/*----->>>>> int cuda_rayleighDampingDeviceSetup();  ---------------------------------------------------------
* Used to cudaMalloc and cudaMemcpy parameters and coordinate arrays, and for the RAYLEIGHDAMPING_CUDA submodule.
*/
extern "C" int cuda_rayleighDampingDeviceSetup();

/*----->>>>> extern "C" int cuda_rayleighDampingDeviceCleanup();  ---------------------------------------------------
* Used to free all malloced memory by the RAYLEIGHDAMPING submodule.
*/
extern "C" int cuda_rayleighDampingDeviceCleanup();

/*----->>>>> __device__ void cudaDevice_topRayleighDampingLayerForcing();  ------------------------------------------
* Rayleigh damping layer forcing term 
*/
__device__ void cudaDevice_topRayleighDampingLayerForcing(float* scalarField, float* scalarFrhs,
                                                          float* rho, float* rhoBS, float* zPos_d);

#endif // _RAYLEIGHDAMPING_CUDADEV_CU_H
