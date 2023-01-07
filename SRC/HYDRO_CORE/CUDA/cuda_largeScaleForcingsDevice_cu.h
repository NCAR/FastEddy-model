/* FastEddy®: SRC/HYDRO_CORE/CUDA/cuda_largeScaleForcingsDevice_cu.h 
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
#ifndef _LSF_CUDADEV_CU_H
#define _LSF_CUDADEV_CU_H

/*LSF_ return codes */
#define CUDA_LSF_SUCCESS    0

/*##############------------------- LSF submodule variable declarations ---------------------#################*/
/* array fields */
extern __constant__ int lsfSelector_d;         /* large-scale forcings selector: 0=off, 1=on */
extern __constant__ float lsf_w_surf_d;        /* lsf to w at the surface */
extern __constant__ float lsf_w_lev1_d;        /* lsf to w at the first specified level */
extern __constant__ float lsf_w_lev2_d;        /* lsf to w at the second specified level */
extern __constant__ float lsf_w_zlev1_d;       /* lsf to w height 1 */
extern __constant__ float lsf_w_zlev2_d;       /* lsf to w height 2 */
extern __constant__ float lsf_th_surf_d;       /* lsf to theta at the surface */
extern __constant__ float lsf_th_lev1_d;       /* lsf to theta at the first specified level */
extern __constant__ float lsf_th_lev2_d;       /* lsf to theta at the second specified level */
extern __constant__ float lsf_th_zlev1_d;      /* lsf to theta height 1 */
extern __constant__ float lsf_th_zlev2_d;      /* lsf to theta height 2 */
extern __constant__ float lsf_qv_surf_d;       /* lsf to qv at the surface */
extern __constant__ float lsf_qv_lev1_d;       /* lsf to qv at the first specified level */
extern __constant__ float lsf_qv_lev2_d;       /* lsf to qv at the second specified level */
extern __constant__ float lsf_qv_zlev1_d;      /* lsf to qv height 1 */
extern __constant__ float lsf_qv_zlev2_d;      /* lsf to qv height 2 */

extern __constant__ int lsf_horMnSubTerms_d;   /* Switch 0=off, 1=on */
extern __constant__ int lsf_numPhiVars_d;      /* number of variables in the slabMeanPhiProfiles set (e.g. rho,u,v,theta,qv=5) */
extern float* lsf_slabMeanPhiProfiles_d;       /*Base address of -w*(d_phi/dz) lsf term horz mean profiles of phi variables*/
extern float* lsf_meanPhiBlock_d;              /*Base address of work arrray for block reduction Mean */

/*##############-------------- LSF_CUDADEV submodule function declarations ------------------############*/

/*----->>>>> int cuda_lsfDeviceSetup();      -----------------------------------------------------------------
* Used to cudaMalloc and cudaMemcpy parameters and coordinate arrays for the LSF_CUDADEV submodule.
*/
extern "C" int cuda_lsfDeviceSetup();

/*----->>>>> int cuda_lsfDeviceCleanup();    ---------------------------------------------------------------
* Used to free all malloced memory by the LSF_CUDADEV submodule.
*/
extern "C" int cuda_lsfDeviceCleanup();

/*----->>>>> extern "C" int cuda_lsfSlabMeans();  -----------------------------------------------------------
*  Obtain the slab means of rho, u, v, theta, and qv
*/
extern "C" int cuda_lsfSlabMeans();

/*----->>>>> __global__ void  cudaDevice_hydroCoreUnitTestCompleteLSF();  --------------------------------------------------
* Global Kernel for calculating/accumulating large-scale forcing Frhs terms   
*/
__global__ void cudaDevice_hydroCoreUnitTestCompleteLSF(float temp_freq_fac, float* hydroFlds_d, float* lsf_slabMeanPhiProfiles_d, float* hydroFldsFrhs_d, float* moistScalarsFrhs_d, float* zPos_d);

/*----->>>>> __device__ void  cudaDevice_lsfRHS();  --------------------------------------------------
*/ // This cuda kernel calculates the large-scale forcing terms
__device__ void cudaDevice_lsfRHS(float temp_freq_fac, float* rho,
                                  float* lsf_slabMeanPhiProfiles_d, float* Frhs_HC, float* Frhs_qv, float* zPos_d);

#endif // _LSF_CUDADEV_CU_H
