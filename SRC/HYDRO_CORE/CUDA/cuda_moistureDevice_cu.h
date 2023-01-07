/* FastEddy®: SRC/HYDRO_CORE/CUDA/cuda_moistureDevice_cu.h 
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
#ifndef _MOISTURE_CUDADEV_CU_H
#define _MOISTURE_CUDADEV_CU_H

/*moisture_ return codes */
#define CUDA_MOISTURE_SUCCESS    0

/*##############------------------- MOISTURE submodule variable declarations ---------------------#################*/
/* array fields */
extern __constant__ int moistureSelector_d;     /* moisture selector: 0=off, 1=on */
extern __constant__ int moistureNvars_d;        /* number of moisture species */
extern __constant__ int moistureAdvSelectorQv_d;  /* water vapor advection scheme selector */
extern __constant__ float moistureAdvSelectorQv_b_hyb_d; /*hybrid advection scheme parameter */
extern __constant__ int moistureSGSturb_d;      /* selector to apply sub-grid scale diffusion to moisture fields */
extern __constant__ int moistureCond_d;         /* selector to apply condensation to mositure fields */
extern __constant__ int moistureAdvSelectorQi_d;  /* moisture advection scheme selector for non-qv fields (non-oscillatory schemes) */
extern __constant__ float moistureCondTscale_d; /* relaxation time in seconds */
extern __constant__ int moistureCondBasePressure_d;  /* selector to use base pressure for microphysics */
extern __constant__ float moistureMPcallTscale_d;  /* time scale for microphysics to be called */
extern float* moistScalars_d;                   /*Base Address of memory containing moisture fields */
extern float* moistScalarsFrhs_d;               /*Base Address of memory containing RHS forcing to moisture fields */
extern float* moistTauFlds_d;                   /*Base address for moisture SGS field arrays*/
extern float* fcond_d;                          /*Base address for f_cond array*/

/*##############-------------- MOISTURE_CUDADEV submodule function declarations ------------------############*/

/*----->>>>> int cuda_moistureDeviceSetup();      -----------------------------------------------------------------
* Used to cudaMalloc and cudaMemcpy parameters and coordinate arrays for the MOISTURE_CUDADEV submodule.
*/
extern "C" int cuda_moistureDeviceSetup();

/*----->>>>> int cuda_moistureDeviceCleanup();    ---------------------------------------------------------------
* Used to free all malloced memory by the MOISTURE_CUDADEV submodule.
*/
extern "C" int cuda_moistureDeviceCleanup();

/*----->>>>> __global__ void  cudaDevice_hydroCoreUnitTestCompleteMP();  -------------------------------------------
* Global Kernel for calculating/accumulating moisture (microphysics) forcing Frhs terms   
*/
__global__ void cudaDevice_hydroCoreUnitTestCompleteMP(float* hydroFlds_d, float* hydroFldsFrhs_d, float* moistScalars_d,
                                                       float* moistScalarsFrhs_d, float* hydroRhoInv_d, 
                                                       float* hydroPres_d, float* fcond_d, float dt, float* hydroBaseStateFlds_d);

/*----->>>>> __device__ void cudaDevice_moistZerothOrder();  --------------------------------------------------
*/
__device__ void cudaDevice_moistZerothOrder(float* rho_qv, float* rho_ql, float* th, float* press, float* rhoInv, float* fcond, float dt, float* th_base);

/*----->>>>> __device__ void cudaDevice_moistCondFrhs();  --------------------------------------------------
*/
__device__ void cudaDevice_moistCondFrhs(float* fcond, float* qv_Frhs, float* ql_Frhs, float* qr_Frhs);

/*----->>>>> __device__ void cudaDevice_thetaCondFrhs();  --------------------------------------------------
*/
__device__ void cudaDevice_thetaCondFrhs(float* press, float* rhoInv, float* th, float* fcond, float* th_Frhs);


#endif // _MOISTURE_CUDADEV_CU_H
