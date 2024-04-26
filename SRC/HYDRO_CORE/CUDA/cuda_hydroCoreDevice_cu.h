/* FastEddy®: SRC/HYDRO_CORE/CUDA/cuda_hydroCoreDevice_cu.h 
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
#ifndef _HYDRO_CORE_CUDADEV_CU_H
#define _HYDRO_CORE_CUDADEV_CU_H

/*hydroCore_ return codes */
#define CUDA_HYDRO_CORE_SUCCESS               0

/*##############------------------- HYDRO_CORE module variable declarations ---------------------#################*/
/* Parameters */
extern __constant__ int Nhydro_d;       // Number of hydro_core prognostic variable fields
extern __constant__ int hydroBCs_d;       // hydro_core BC set selector

/* array fields */
extern float *hydroFlds_d;     //Base Adress of memory containing all prognostic variable fields under hydro_core
extern float *hydroFldsFrhs_d; //Base Adress of memory containing variable field Frhs(s) under hydro_core
extern float *hydroRhoInv_d;   //storage for 1.0/rho

/*HYDRO_CORE Submodule parameters*/
/*---BASESTATE*/
#include <cuda_BaseStateDevice_cu.h>

/*---BUOYANCY TERM*/
#include <cuda_buoyancyDevice_cu.h>

/*---CORIOLIS TERMS*/
#include <cuda_coriolisDevice_cu.h>

/*---PRESSURE_GRADIENT_FORCE*/
#include <cuda_pressureDevice_cu.h>

/*---BCS*/
#include <cuda_BCsDevice_cu.h>

/*---RAYLEIGHDAMPING TERM*/
#include <cuda_rayleighDampingDevice_cu.h>

/*---TURBULENCE*/
#include <cuda_sgsTurbDevice_cu.h>

/*---DIFFUSION*/
#include <cuda_molecularDiffDevice_cu.h>

/*---ADVECTION*/
#include <cuda_advectionDevice_cu.h>

/*---SURFACE LAYER*/
#include <cuda_surfaceLayerDevice_cu.h>

/*---SGSTKE */
#include <cuda_sgstkeDevice_cu.h>

/*---LARGE SCALE FORCINGS */
#include <cuda_largeScaleForcingsDevice_cu.h>

/*---MOISTURE */
#include <cuda_moistureDevice_cu.h>

/*EXPLICIT FILTERS */
#include <cuda_filtersDevice_cu.h>

/*Switch for Last-RK stage physics */
extern __constant__ int physics_oneRKonly_d; /* selector to apply physics RHS forcing only at the latest RK stage: 0= off, 1= on */

/*Constants*/
extern __constant__ float R_gas_d;   /* The ideal gas constant in J/(mol*K) */
extern __constant__ float R_vapor_d; /* The ideal gas constant for water vapor in J/(mol*K) */
extern __constant__ float Rv_Rg_d;    /* Ratio R_vapor/R_gas */
extern __constant__ float cp_gas_d;  /* Specific heat of air at constant pressure */
extern __constant__ float cv_gas_d;  /* Specific heat of air at constant pressure */
extern __constant__ float accel_g_d; /* Acceleration of gravity 9.8 m/(s^2) */
extern __constant__ float R_cp_d;    /* Ratio R/cp */
extern __constant__ float cp_R_d;    /* Ratio cp/R */
extern __constant__ float cp_cv_d;   /* Ratio cp/cv */
extern __constant__ float refPressure_d;   /* Reference pressure set constant to 1e5 Pascals or 1000 millibars */
extern __constant__ float kappa_d;          /*von Karman constant*/
extern __constant__ float L_v_d;            /* latent heat of vaporization (J/kg) */

/*##############------ HYDRO_CORE_CUDADEV module C-layer (Host/CPU) function declarations -------------############*/

/*----->>>>> int cuda_hydroCoreDeviceSetup();      -----------------------------------------------------------------
* Used to cudaMalloc and cudaMemcpy parameters and coordinate arrays for the HYDRO_CORE_CUDADEV module.
*/
extern "C" int cuda_hydroCoreDeviceSetup();

/*----->>>>> int cuda_hydroCoreDeviceCleanup();    ---------------------------------------------------------------
* Used to free all malloced memory by the HYDRO_CORE_CUDADEV module.
*/
extern "C" int cuda_hydroCoreDeviceCleanup();

/*----->>>>> extern "C" int cuda_hydroCoreDeviceBuildFrhs();  --------------------------------------------------
* This routine provides the externally callable cuda-kernel call to perform a complete hydroCore build_Frhs
*/
extern "C" int cuda_hydroCoreDeviceBuildFrhs(float simTime, int simTime_it, int simTime_itRestart, 
                                             float dt,int timeStage, int numRKstages);

/*#########--------------- HYDRO_CORE_CUDADEV module device function declarations ------------############*/

/*----->>>>> __global__ void  cudaDevice_hydroCoreUnitTestCommence();  ------------------------------------------
* This is the gloabl-entry kernel routine used by the HYDRO_CORE module
*/
__global__ void cudaDevice_hydroCoreUnitTestCommence(int simTime_it, float* hydroFlds_d, float* hydroFldsFrhs_d, 
                                                     float*  hydroBaseStateFlds_d,
                                                     float* tskin_d, float* qskin_d,
                                                     float* sgstkeScalars_d, float* sgstkeScalarsFrhs_d, float* Km_d, 
                                                     float* moistScalars_d, float* moistScalarsFrhs_d, float* zPos_d);
__global__ void cudaDevice_hydroCoreUnitTestCommenceRhoInvPresPert(float* hydroFlds_d, float* hydroRhoInv_d,
                                                     float* hydroBaseStateFlds_d,
                                                     float* hydroPres_d, float* hydroBaseStatePres_d,
                                                     float* moistScalars_d, float* zPos_d); 
__global__ void cudaDevice_hydroCoreCalcFaceVelocities(float simTime, int simTime_it, int simTime_itRestart,
                                                       float dt, int timeStage, int numRKstages,
                                                       float* hydroFlds_d, float* hydroFldsFrhs_d,
                                                       float* hydroFaceVels_d, float* hydroPres_d,
                                                       float* hydroNuGradXFlds_d, float* hydroNuGradYFlds_d,
                                                       float* hydroNuGradZFlds_d, 
                                                       float* hydroTauFlds_d, 
                                                       float* cdFld_d, float* chFld_d, float* cqFld_d, float* fricVel_d,
                                                       float* htFlux_d, float* tskin_d, float* invOblen_d,
                                                       float* z0m_d, float* z0t_d, float* qFlux_d, float* qskin_d, float* sea_mask_d,
                                                       float* hydroRhoInv_d, float* hydroKappaM_d,
                                                       float* sgstkeScalars_d, float* sgstke_ls_d,
                                                       float* dedxi_d, float* moistScalars_d,
                                                       float* moistTauFlds_d, float* moistScalarsFrhs_d,
                                                       float* J31_d, float* J32_d, float* J33_d, float* D_Jac_d);
__global__ void cudaDevice_hydroCoreUnitTestComplete(float simTime, int simTime_it, float dt, int timeStage, int numRKstages,
                                                     float* hydroFlds,float* hydroFldsFrhs,
                                                     float* hydroFaceVels, float* hydroBaseStateFlds, float* hydroTauFlds, 
                                                     float* sgstkeScalars, float* sgstkeScalarsFrhs,
                                                     float* moistScalars, float* moistScalarsFrhs, float* moistTauFlds,
                                                     float* J31_d, float* J32_d, float* J33_d, float* invD_Jac_d, float* zPos_d);
/*----->>>>> __device__ void  cudaDevice_SetRhoInv();  --------------------------------------------------
* This is the cuda version of the SetRhoInv routine from the HYDRO_CORE module
*/
__device__ void cudaDevice_SetRhoInv(float* hydroFlds, float* hydroRhoInv);

/*----->>>>> __device__ void cudaDevice_setToZero();  --------------------------------------------------
* This sets every element of a device "field"-array to zero
*/
__device__ void cudaDevice_setToZero(float* fld);

#endif // _HYDRO_CORE_CUDADEV_CU_H
