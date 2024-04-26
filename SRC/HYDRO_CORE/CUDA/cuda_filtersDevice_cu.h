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
extern __constant__ int filter_6thdiff_vert_d;          /* vertical 6th-order filter on w selector: 0=off, 1=on */
extern __constant__ float filter_6thdiff_vert_coeff_d;  /* vertical 6th-order filter w factor: 0.0=off, 1.0=full */
extern __constant__ int filter_6thdiff_hori_d;          /* horizontal 6th-order filter on rho,theta,qv selector: 0=off, 1=on */
extern __constant__ float filter_6thdiff_hori_coeff_d;  /* horizontal 6th-order filter factor: 0.0=off, 1.0=full */
extern __constant__ int filter_divdamp_d;               /* divergence damping selector: 0=off, 1=on */

/*##############-------------- FILTERS_CUDADEV submodule function declarations ------------------############*/

/*----->>>>> int cuda_filtersDeviceSetup();       ---------------------------------------------------------
 * Used to cudaMalloc and cudaMemcpy parameters and coordinate arrays, and for the FILTERS_CUDA submodule.
*/
extern "C" int cuda_filtersDeviceSetup();

/*----->>>>> extern "C" int cuda_filtersDeviceCleanup();  -----------------------------------------------------------
Used to free all malloced memory by the FILTERS submodule.
*/
extern "C" int cuda_filtersDeviceCleanup();

__global__ void cudaDevice_hydroCoreUnitTestCompleteFilters(float* hydroFlds_d, float* hydroFldsFrhs_d, float dt,
                                                            float* moistScalars_d, float* moistScalarsFrhs, float* hydroPres_d,
                                                            float* hydroBaseStatePres_d, int timeStage);

/*----->>>>> __device__ void cudaDevice_filter6th();  --------------------------------------------------
*/
__device__ void cudaDevice_filter6th(float* fld, float* hydroFldsFrhs_d, float dt);

/*----->>>>> __device__ void cudaDevice_filter6th2D();  --------------------------------------------------
*/
__device__ void cudaDevice_filter6th2D(float* fld, float* fld_Frhs, float dt);

/*----->>>>> __device__ void cudaDevice_divergenceDamping();  --------------------------------------------------
*/
__device__ void cudaDevice_divergenceDamping(float* uFrhs, float* vFrhs, float* thetaFrhs,
                                             float* theta, float* rho, float* moistScalars,
                                             float* pres, float* baseStatePres, float dt, int timeStage);

#endif // _FILTERS_CUDADEV_CU_H
