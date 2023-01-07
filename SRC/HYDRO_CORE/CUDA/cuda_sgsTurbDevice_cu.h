/* FastEddy®: SRC/HYDRO_CORE/CUDA/cuda_sgsTurbDevice_cu.h 
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
#ifndef _SGSTURB_CUDADEV_CU_H
#define _SGSTURB_CUDADEV_CU_H

/*hydroCore_ return codes */
#define CUDA_SGSTURB_SUCCESS               0

#define UU_INDX                0
#define UV_INDX                1
#define UW_INDX                2
#define VW_INDX                3
#define VV_INDX                4
#define WW_INDX                5
#define THU_INDX               6
#define THV_INDX               7
#define THW_INDX               8

/*##############------------------- SGSTURB HC-Submodule variable declarations -------------------#################*/
/* Parameters */

/*HYDRO_CORE Submodule parameters*/
/*---TURBULENCE*/
extern __constant__ int turbulenceSelector_d;  /*turbulence scheme selector: 0= none, 1= Lilly/Smagorinsky */
extern __constant__ int TKESelector_d;    /* Prognostic TKE selector: 0= none, 1= Prognostic */
extern __constant__ float c_s_d;       /* Smagorinsky turbulence model constant used for turbulenceSelector = 1 with TKESelector = 0 */
extern __constant__ float c_k_d;       /* Lilly turbulence model constant used for turbulenceSelector = 1 with TKESelector > 0 */
extern float* hydroTauFlds_d;  /*Base address for 6 Tau field arrays*/
extern float* hydroKappaM_d;  /*Base address for KappaM (eddy diffusivity for momentum)*/

/*##############-------------- SGSTURB HC-Submodule function declarations ------------------############*/
/*----->>>>> int cuda_sgsTurbDeviceSetup();       ----------------------------------------------------------------
 * Used to cudaMalloc and cudaMemcpy parameters and coordinate arrays, and for the SGSTURB HC-Submodule.
*/
extern "C" int cuda_sgsTurbDeviceSetup();
/*----->>>>> extern "C" int cuda_sgsTurbDeviceCleanup();  -----------------------------------------------------------
Used to free all malloced memory by the SGSTURB HC-Submodule.
*/
extern "C" int cuda_sgsTurbDeviceCleanup();

/*#########--------------- SGSTURB HC-Submodule device function declarations ------------############*/

/*----->>>>> __device__ void  cudaDevice_hydroCoreCalcStrainRateElements();  ----------------------------------------
 * This is the cuda version of calculating strain rate elements: S_ij and S_thj
*/ 
__device__ void cudaDevice_hydroCoreCalcStrainRateElements(float* u, float* v, float* w, float* theta,
                                                           float* S11, float* S21, float* S31,
                                                           float* S32, float* S22, float* S33,
                                                           float* STH1, float* STH2, float* STH3,
                                                           float* J31_d, float* J32_d, float* J33_d,
                                                           float* rhoInv);

/*----->>>>> __device__ void  cudaDevice_hydroCoreCalcEddyDiff();  ---------------------------------------------
* This is the cuda version of calculating eddy diffusivity of momentum Km
*/
__device__ void cudaDevice_hydroCoreCalcEddyDiff(float* S11, float* S21, float* S31,
                                                 float* S32, float* S22, float* S33,
                                                 float* STH3, float* theta, float* rhoInv,
                                                 float* sgstke, float* sgstke_ls, float* Km, float* D_Jac_d);

/*----->>>>> __device__ void  cudaDevice_hydroCoreCalcTaus();  ---------------------------------------------
* This is the cuda version of calculating sub-grid scale stresses
*/
__device__ void cudaDevice_hydroCoreCalcTaus(float* S11, float* S21, float* S31,
                                             float* S32, float* S22, float* S33,
                                             float* STH1, float* STH2, float* STH3,
                                             float* rho, float* Km, float* sgstke_ls, float* D_Jac_d);

/*----->>>>> __device__ void  cudaDevice_hydroCoreCalcTaus_PrognosticTKE_DeviatoricTerm();  -------------------------
*  This is the compressible stress formulation routine with deviatoric term modeled using 
*  the prognostic TKE (i.e. TKESelector_d > 0)
*/
__device__ void cudaDevice_hydroCoreCalcTaus_PrognosticTKE_DeviatoricTerm(
                                             float* S11, float* S21, float* S31,
                                             float* S32, float* S22, float* S33,
                                             float* STH1, float* STH2, float* STH3,
                                             float* rho, float* Km, float* sgstke_ls,
                                             float* u, float* v, float* w, float* sgstke,
                                             float* J31_d, float* J32_d, float* J33_d, float* D_Jac_d);

/*----->>>>> __device__ void  cudaDevice_GradScalarToFaces();  --------------------------------------------------
* This cuda kernel calculates the spatial gradient of a scalar field: 1delta, gradient located at the cell face
*/
__device__ void cudaDevice_GradScalarToFaces(float* scalar, float* rhoInv, float* dSdx, float* dSdy, float* dSdz,
                                             float* J31_d, float* J32_d, float* J33_d);

/*----->>>>> __device__ void  cudaDevice_hydroCoreCalcTausScalar();  ---------------------------------------------
* This is the cuda version of calculating SGS stresses of a scalar field field for subgrid-scale mixing formulations
*/ 
__device__ void cudaDevice_hydroCoreCalcTausScalar(float* SM1, float* SM2, float* SM3,
                                                   float* rho, float* Km, float* sgstke_ls, float* D_Jac_d);

/*----->>>>> __device__ void  cudaDevice_hydroCoreCalcTurbMixing();  ---------------------------------------------
 * This is the cuda version of calculating forcing terms from subgrid-scale mixing
*/
__device__ void cudaDevice_hydroCoreCalcTurbMixing(float* uFrhs, float* vFrhs, float* wFrhs, float* thetaFrhs,
                                                   float* T11, float* T21, float* T31,
                                                   float* T32, float* T22, float* T33,
                                                   float* TH1, float* TH2, float* TH3,
                                                   float* J31_d, float* J32_d, float* J33_d);

/*----->>>>> __device__ void  cudaDevice_hydroCoreCalcTurbMixingScalar();  ---------------------------------------------
* This is the cuda version of calculating forcing terms from subgrid-scale mixing of a scalar field
*/ 
__device__ void cudaDevice_hydroCoreCalcTurbMixingScalar(float* mFrhs, float* M1, float* M2, float* M3,
                                                         float* J31_d, float* J32_d, float* J33_d);

#endif // _SGSTURB_CUDADEV_CU_H
