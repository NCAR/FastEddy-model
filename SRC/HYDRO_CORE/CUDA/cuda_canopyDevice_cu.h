/* FastEddy®: SRC/HYDRO_CORE/CUDA/cuda_canopyDevice_cu.h
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
#ifndef _CANOPY_CUDADEV_CU_H
#define _CANOPY_CUDADEV_CU_H

/*canopy_ return codes */
#define CUDA_CANOPY_SUCCESS    0

/*##############------------------- CANOPY submodule variable declarations ---------------------#################*/
/* array fields */
extern __constant__ int canopySelector_d;         /* canopy selector: 0=off, 1=on */
extern __constant__ int canopySkinOpt_d;          /* canopy selector to use additional skin friction effect on drag coefficient: 0=off, 1=on */
extern __constant__ float canopy_cd_d;            /* non-dimensional canopy drag coefficient cd */
extern __constant__ float canopy_lf_d;            /* representative canopy element length scale */
extern float* canopy_lad_d;          /* Base Address of memory containing leaf area density (LAD) field [m^{-1}] */

/*##############-------------- CANOPY_CUDADEV submodule function declarations ------------------############*/

/*----->>>>> int cuda_canopyDeviceSetup();      -----------------------------------------------------------------
* Used to cudaMalloc and cudaMemcpy parameters and coordinate arrays for the CANOPY_CUDADEV submodule.
*/
extern "C" int cuda_canopyDeviceSetup();

/*----->>>>> int cuda_canopyDeviceCleanup();    ---------------------------------------------------------------
* Used to free all malloced memory by the CANOPY_CUDADEV submodule.
*/
extern "C" int cuda_canopyDeviceCleanup();

/*----->>>>> __global__ void  cudaDevice_hydroCoreCompleteCanopy();  ----------------------------------------
 * Global Kernel for Canopy model
*/
__global__ void cudaDevice_hydroCoreCompleteCanopy(float* hydroFlds_d, float* hydroRhoInv_d, float* canopy_lad_d, float* hydroFldsFrhs_d);

/*----->>>>> __device__ void  cudaDevice_canopyMomDrag();  --------------------------------------------------
*/ // This cuda kernel calculates the forcing term to the momentum equations due to canopy drag
__device__ void cudaDevice_canopyMomDrag(float* rhoInv, float* u, float* v, float* w, float* lad, float* Frhs_u, float* Frhs_v, float* Frhs_w);

/*----->>>>> __device__ void  cudaDevice_canopySGSTKEtransfer();  --------------------------------------------------
*/ // This cuda kernel calculates the forcing term to 1st SGSTKE equation due to canopy drag (transfer to wake scale)
__device__ void cudaDevice_canopySGSTKEtransfer(float* rhoInv, float* u, float* v, float* w,
                                                float* lad, float* sgstke, float* Frhs_sgstke, int sign_term);

/*----->>>>> __device__ void  cudaDevice_sgstkeLengthScaleLF();  --------------------------------------------------
*/ // This cuda kernel assigns canopy_lf representative canopy scale
__device__ void cudaDevice_sgstkeLengthScaleLF(float* sgstke_ls);

/*----->>>>> __device__ void  cudaDevice_canopySGSTKEwakeprod();  --------------------------------------------------
*/ // This cuda kernel calculates the SGSTKE wake production term
__device__ void cudaDevice_canopySGSTKEwakeprod(float* rhoInv, float* u, float* v, float* w,
                                                float* lad, float* Frhs_sgstke);

#endif // _CANOPY_CUDADEV_CU_H
