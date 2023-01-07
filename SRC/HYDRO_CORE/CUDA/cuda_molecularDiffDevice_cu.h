/* FastEddy®: SRC/HYDRO_CORE/CUDA/cuda_molecularDiffDevice_cu.h 
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
#ifndef _MOLDIFF_CUDADEV_CU_H
#define _MOLDIFF_CUDADEV_CU_H

/*moldiff_ return codes */
#define CUDA_MOLDIFF_SUCCESS               0

/*##############------------------- MOLECULAR DIFFUSION submodule variable declarations ---------------------#################*/
/* Parameters */
extern __constant__ int diffusionSelector_d;      /* molecular diffusion selector: 0=off, 1=on */
extern __constant__ float nu_0_d;                 /* constant molecular diffusivity used when diffusionSelector_d == 1 */
/* array fields */
extern float* hydroNuGradXFlds_d;                 /* Base address for diffusion for nu*grad_x */
extern float* hydroNuGradYFlds_d;                 /* Base address for diffusion for nu*grad_y */
extern float* hydroNuGradZFlds_d;                 /* Base address for diffusion for nu*grad_z */

/*##############-------------- MOLECULAR DIFFUSION submodule function declarations ------------------############*/

/*----->>>>> int cuda_molecularDiffDeviceSetup();      -----------------------------------------------------------------
* Used to cudaMalloc and cudaMemcpy parameters and coordinate arrays for the MOLECULAR DIFFUSION submodule.
*/
extern "C" int cuda_molecularDiffDeviceSetup();

/*----->>>>> int cuda_molecularDiffDeviceCleanup();    ---------------------------------------------------------------
* Used to free all malloced memory by the MOLECULAR DIFFUSION submodule.
*/
extern "C" int cuda_molecularDiffDeviceCleanup();

/*----->>>>> __global__ void  cudaDevice_hydroCoreUnitTestCompleteMolecularDiffusion();  --------------------------------------------------
* Global Kernel for calculating/accumulating molecular diffusion Frhs terms   
*/
__global__ void cudaDevice_hydroCoreUnitTestCompleteMolecularDiffusion(float* hydroFlds, float* hydroFldsFrhs,
                                                                       float* hydroNuGradXFlds_d, float* hydroNuGradYFlds_d, float* hydroNuGradZFlds_d,
                                                                       float* J31_d, float* J32_d, float* J33_d,
                                                                       float* D_Jac_d, float* invD_Jac_d);

/*----->>>>> __device__ void cudaDevice_diffusionDriver();  --------------------------------------------------
* This function drives the element-wise calls to cudaDevice_calcConstNuGrad() for molecular diffusion 
* of an arbitrary field. 
*/
__device__ void cudaDevice_diffusionDriver(float* fld, float* NuGradX, float* NuGradY,float* NuGradZ, float inv_pr,
                                           float* J31_d, float* J32_d, float* J33_d,
                                           float* D_Jac_d);


/*----->>>>> __device__ void cudaDevice_calcConstNuGrad();  --------------------------------------------------
*  This is the cuda form of formulating the gradient of a field with constant molecular diffusivity
*/  
__device__ void cudaDevice_calcConstNuGrad(float* NuGradX, float* NuGradY, float* NuGradZ,
                                           float* sFld_ijk,float* sFld_im1jk, float* sFld_ijm1k, float* sFld_ijkm1,
                                           float* sFld_ip1jk, float* sFld_ijp1k, float* sFld_ijkp1,
                                           float* sFld_im1jm1k, float* sFld_im1jkm1, float* sFld_ijm1km1,
                                           float* sFld_im1jp1k, float* sFld_im1jkp1, float* sFld_ijm1kp1,
                                           float* sFld_ip1jkm1, float* sFld_ip1jm1k, float* sFld_ijp1km1, float inv_pr,
                                           float* J31_d, float* J32_d, float* J33_d,
                                           float* D_Jac_d);

/*----->>>>> __device__ void cudaDevice_calcDivNuGrad();  --------------------------------------------------
* This is the cuda version of taking the divergence of nu_0 times the gradient of a field
*/
__device__ void cudaDevice_calcDivNuGrad(float* scalarFrhs, float* rho, float* NuGradX, float* NuGradY, float* NuGradZ, int iFld,
                                         float* J31_d, float* J32_d, float* J33_d,
                                         float* D_Jac_d, float* invD_Jac_d);

#endif // _MOLDIFF_CUDADEV_CU_H
