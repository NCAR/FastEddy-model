/* FastEddy®: SRC/HYDRO_CORE/CUDA/cuda_sgstkeDevice_cu.h 
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
#ifndef _SGSTKE_CUDADEV_CU_H
#define _SGSTKE_CUDADEV_CU_H

/*sgstke_ return codes */
#define CUDA_SGSTKE_SUCCESS    0

/*##############------------------- SGSTKE submodule variable declarations ---------------------#################*/
/*Advection selectors 1 ior 2-eq TKE */
extern __constant__ int TKEAdvSelector_d;      /* SGSTKE advection scheme selector */
extern __constant__ float TKEAdvSelector_b_hyb_d;      /* hybrid advection scheme parameter */
extern __constant__ int TKEAdvSelectorWake_d;    /* SGSTKE advection scheme selector for wake scale SGSTKE */
extern __constant__ float TKEAdvSelectorWake_b_hyb_d;      /* hybrid advection scheme parameter */

/* array fields */
extern float* sgstkeScalars_d;       /*Base Address of memory containing SGSTKE fields */
extern float* sgstkeScalarsFrhs_d;   /*Base Address of memory containing RHS forcing to SGSTKE fields */
extern float* sgstke_ls_d;           /*Base address for SGSTKE length scale field arrays*/
extern float* dedxi_d; /*Base address for d(SGSTKE)/dxi field arrays*/

/*##############-------------- SGSTKE_CUDADEV submodule function declarations ------------------############*/

/*----->>>>> int cuda_sgstkeDeviceSetup();      -----------------------------------------------------------------
* Used to cudaMalloc and cudaMemcpy parameters and coordinate arrays for the SGSTKE_CUDADEV submodule.
*/
extern "C" int cuda_sgstkeDeviceSetup();

/*----->>>>> int cuda_sgstkeDeviceCleanup();    ---------------------------------------------------------------
* Used to free all malloced memory by the SGSTKE_CUDADEV submodule.
*/
extern "C" int cuda_sgstkeDeviceCleanup();

/*----->>>>> __device__ void  cudaDevice_hydroCoreUnitTestCompleteSGSTKE();  ----------------------------------------
 * Global Kernel for calculating/accumulating SGSTKE Frhs     
*/
__global__ void cudaDevice_hydroCoreUnitTestCompleteSGSTKE(float* hydroFlds_d, float* hydroRhoInv_d, float* hydroTauFlds_d,
                                                           float* hydroKappaM_d, float* dedxi_d, float* sgstke_ls_d,
                                                           float* sgstkeScalars_d, float* sgstkeScalarsFrhs_d,
                                                           float* J31_d, float* J32_d, float* J33_d, float* D_Jac_d);

/*----->>>>> __device__ void  cudaDevice_sgstkeLengthScale();  --------------------------------------------------
* This cuda kernel calculates the sub-grid length scale
*/
__device__ void cudaDevice_sgstkeLengthScale(float* th, float* rhoInv, float* sgstke, float* sgstke_ls, float* J31_d, float* J32_d, float* J33_d, float* D_Jac_d);

/*----->>>>> __device__ void  cudaDevice_sgstkeBuoyancy();  --------------------------------------------------
* This cuda kernel calculates the Buoyancy term in the SGSTKE equation
*/ 
__device__ void cudaDevice_sgstkeBuoyancy(float* th, float* rhoInv, float* STH3, float* Frhs_sgstke);

/*----->>>>> __device__ void  cudaDevice_sgstkeDissip();  --------------------------------------------------
* This cuda kernel calculates the Dissipation term in the SGSTKE equation
*/ 
__device__ void cudaDevice_sgstkeDissip(float* sgstke, float* rhoInv, float* sgstke_ls, float* Frhs_sgstke, int l_corr_ce, float* D_Jac_d);

/*----->>>>> __device__ void  cudaDevice_sgstkeShearProd();  --------------------------------------------------
* This cuda kernel calculates the Shear Production term in the SGSTKE equation
*/ 
__device__ void cudaDevice_sgstkeShearProd(float* tau_11, float* tau_12, float* tau_13, float* tau_22, float* tau_23, float* tau_33, 
                                           float* u, float* v, float* w, float* rhoInv, float* Frhs_sgstke,
                                           float* J31_d, float* J32_d, float* J33_d);

/*----->>>>> __device__ void  cudaDevice_GradScalar();  --------------------------------------------------
* This cuda kernel calculates the spatial gradient of a scalar field: 2delta, gradient located at the cell center
*/ 
__device__ void cudaDevice_GradScalar(float* scalar, float* rhoInv, float* dedx, float* dedy, float* dedz,
                                      float* J31_d, float* J32_d, float* J33_d);

/*----->>>>> __device__ void  cudaDevice_sgstkeTurbTransport();  --------------------------------------------------
* This cuda kernel calculates the Turbulent Transport term in the SGSTKE equation
*/ 
__device__ void cudaDevice_sgstkeTurbTransport(float* Km, float* dedx, float* dedy, float* dedz, float* rho, float* Frhs_sgstke,
                                               float* J31_d, float* J32_d, float* J33_d);


#endif // _SGSTKE_CUDADEV_CU_H
