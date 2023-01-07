/* FastEddy®: SRC/HYDRO_CORE/CUDA/cuda_coriolisDevice_cu.h 
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
#ifndef _CORIOLIS_CUDADEV_CU_H
#define _CORIOLIS_CUDADEV_CU_H

/*coriolis return codes */
#define CUDA_CORIOLIS_SUCCESS    0

/*##############------------------- CORIOLIS submodule variable declarations ---------------------#################*/
/*---CORIOLIS*/
extern __constant__ int coriolisSelector_d;   /*coriolis Force selector: 0= none, 1= horiz., 2= horiz. & vert.*/
extern __constant__ float corioConstHorz_d;          /*coriolis horizontal term constant */
extern __constant__ float corioConstVert_d;          /*coriolis vertical term constant */
extern __constant__ float corioLS_fact_d;            /*large-scale factor on Coriolis term*/

/*##############-------------- CORIOLIS_CUDADEV submodule function declarations ------------------############*/

/*----->>>>> int cuda_coriolisDeviceSetup();       ---------------------------------------------------------
* Used to cudaMalloc and cudaMemcpy parameters and coordinate arrays, and for the CORIOLIS_CUDA submodule.
*/
extern "C" int cuda_coriolisDeviceSetup();

/*----->>>>> extern "C" int cuda_coriolisDeviceCleanup();  -----------------------------------------------------------
* Used to free all malloced memory by the CORIOLIS submodule.
*/
extern "C" int cuda_coriolisDeviceCleanup();

/*----->>>>> __device__ void  cudaDevice_calcCoriolis();  --------------------------------------------------
* This is the cuda version of the calcCoriolis routine from the HYDRO_CORE module
*/
__device__ void cudaDevice_calcCoriolis(float* Frhs_u, float* Frhs_v, float* Frhs_w,
                                        float* rho, float* uMom, float* vMom, float* wMom,
                                        float* rhoBS, float* uBS, float* vBS, float* wBS);

#endif // _CORIOLIS_CUDADEV_CU_H
