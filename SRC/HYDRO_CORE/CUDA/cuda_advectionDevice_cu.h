/* FastEddy®: SRC/HYDRO_CORE/CUDA/cuda_advectionDevice_cu.h 
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
#ifndef _ADVECTION_CUDADEV_CU_H
#define _ADVECTION_CUDADEV_CU_H

/*advection_ return codes */
#define CUDA_ADVECTION_SUCCESS               0

/*##############------------------- ADVECTION submodule variable declarations ---------------------#################*/
/* Parameters */
extern float *hydroFaceVels_d; //cell face velocities
extern __constant__ int advectionSelector_d;  /*advection scheme selector: 0= 1st-order upwind, 1= 3rd-order QUICK, 2= hybrid 3rd-4th order, 3= hybrid 5th-6th order*/
extern __constant__ float b_hyb_d;            /*hybrid advection scheme parameter: 0.0= higer-order upwind, 1.0=lower-order cetered, 0.0 < b_hyb < 1.0 = hybrid*/

/*##############-------------- ADVECTION_CUDADEV submodule function declarations ------------------############*/
/*----->>>>> int cuda_advectionDeviceSetup();       ---------------------------------------------------------
 * Used to cudaMalloc and cudaMemcpy parameters and coordinate arrays, and for the ADVECTION_CUDA submodule.
*/
extern "C" int cuda_advectionDeviceSetup();

/*----->>>>> extern "C" int cuda_advectionDeviceCleanup();  -----------------------------------------------------------
Used to free all malloced memory by the ADVECTION_CUDA submodule.
*/
extern "C" int cuda_advectionDeviceCleanup();

/*----->>>>> __device__ void  cudaDevice_calcFaceVelocities();  --------------------------------------------------
* This device function calculates the cell face velocities to prepare for use in the chosen advection scheme
*/
__device__ void cudaDevice_calcFaceVelocities(float* hydroFlds_d, float* hydroFaceVels_d,
                                              float* J31_d, float* J32_d, float* J33_d, float* D_Jac_d);

/*----->>>>> __device__ void  cudaDevice_UpstreamDivAdvFlux();  --------------------------------------------------
* This is the cuda version of the UpstreamDivAdvFlux routine from the HYDRO_CORE module
*/
__device__ void cudaDevice_UpstreamDivAdvFlux(float* scalarField, float* scalarFadv,
                                              float* u_cf, float* v_cf, float* w_cf, float* invD_Jac_d);

/*----->>>>> __device__ void  cudaDevice_SecondDivAdvFlux();  -------------------------------------------------- 
*/
__device__ void cudaDevice_SecondDivAdvFlux(float* scalarField, float* scalarFadv,
                                              float* u_cf, float* v_cf, float* w_cf, float* invD_Jac_d);

/*----->>>>> __device__ void  cudaDevice_QUICKDivAdvFlux();  --------------------------------------------------
* This is the cuda version of the QUICKDivAdvFlux routine from the HYDRO_CORE module
*/
__device__ void cudaDevice_QUICKDivAdvFlux(float* scalarField, float* scalarFadv,
                                              float* u_cf, float* v_cf, float* w_cf, float* invD_Jac_d);

/*----->>>>> __device__ void  cudaDevice_HYB34DivAdvFlux();  --------------------------------------------------
* This is the cuda version of the HYB34DivAdvFlux routine from the HYDRO_CORE module
*/
__device__ void cudaDevice_HYB34DivAdvFlux(float* scalarField, float* scalarFadv,
                                              float* u_cf, float* v_cf, float* w_cf, float b_hyb_p, float* invD_Jac_d);

/*----->>>>> __device__ void  cudaDevice_HYB56DivAdvFlux();  --------------------------------------------------
* This is the cuda version of the HYB56DivAdvFlux routine from the HYDRO_CORE module
*/
__device__ void cudaDevice_HYB56DivAdvFlux(float* scalarField, float* scalarFadv,
                                              float* u_cf, float* v_cf, float* w_cf, float b_hyb_p, float* invD_Jac_d);

/*----->>>>> __device__ void  cudaDevice_WENO3DivAdvFluxX();  -------------------------------------------------- */
__device__ void cudaDevice_WENO3DivAdvFluxX(float* scalarField, float* scalarFadv,float* u_cf, float* invD_Jac_d);

/*----->>>>> __device__ void  cudaDevice_WENO3DivAdvFluxY();  -------------------------------------------------- */
__device__ void cudaDevice_WENO3DivAdvFluxY(float* scalarField, float* scalarFadv,float* v_cf, float* invD_Jac_d);

/*----->>>>> __device__ void  cudaDevice_WENO3DivAdvFluxZ();  -------------------------------------------------- */
__device__ void cudaDevice_WENO3DivAdvFluxZ(float* scalarField, float* scalarFadv,float* w_cf, float* invD_Jac_d);

/*----->>>>> __device__ void  cudaDevice_WENO5DivAdvFluxX();  -------------------------------------------------- */
__device__ void cudaDevice_WENO5DivAdvFluxX(float* scalarField, float* scalarFadv,float* u_cf, float* invD_Jac_d);

/*----->>>>> __device__ void  cudaDevice_WENO5DivAdvFluxY();  -------------------------------------------------- */
__device__ void cudaDevice_WENO5DivAdvFluxY(float* scalarField, float* scalarFadv,float* v_cf, float* invD_Jac_d);

/*----->>>>> __device__ void  cudaDevice_WENO5DivAdvFluxZ();  -------------------------------------------------- */
__device__ void cudaDevice_WENO5DivAdvFluxZ(float* scalarField, float* scalarFadv,float* w_cf, float* invD_Jac_d);

#endif // _ADVECTION_CUDADEV_CU_H
