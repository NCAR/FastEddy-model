/* FastEddy®: SRC/HYDRO_CORE/CUDA/cuda_filtersDevice_cu.h 
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
#ifndef _FILTERS_CUDADEV_CU_H
#define _FILTERS_CUDADEV_CU_H

/*filters return codes */
#define CUDA_FILTERS_SUCCESS    0

/*##############------------------- FILTERS submodule variable declarations ---------------------#################*/
/* array fields */
extern __constant__ int filterSelector_d;     /* explicit filter selector: 0=off, 1=on */
extern __constant__ float filter_6th_coeff_d; /* 6th-order filter factor: 0.0=off, 1.0=full */

/*##############-------------- FILTERS_CUDADEV submodule function declarations ------------------############*/

/*----->>>>> int cuda_filtersDeviceSetup();       ---------------------------------------------------------
 * Used to cudaMalloc and cudaMemcpy parameters and coordinate arrays, and for the FILTERS_CUDA submodule.
*/
extern "C" int cuda_filtersDeviceSetup();

/*----->>>>> extern "C" int cuda_filtersDeviceCleanup();  -----------------------------------------------------------
Used to free all malloced memory by the FILTERS submodule.
*/
extern "C" int cuda_filtersDeviceCleanup();

__global__ void cudaDevice_hydroCoreUnitTestCompleteFilters(float* hydroFlds_d, float* hydroFldsFrhs_d, float dt);

/*----->>>>> __device__ void cudaDevice_filter6th();  --------------------------------------------------
*/
__device__ void cudaDevice_filter6th(float* fld, float* hydroFldsFrhs_d, float dt);

#endif // _FILTERS_CUDADEV_CU_H
