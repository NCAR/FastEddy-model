/* FastEddy®: SRC/HYDRO_CORE/CUDA/cuda_pressureDevice_cu.h 
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
#ifndef _PRESSURE_CUDADEV_CU_H
#define _PRESSURE_CUDADEV_CU_H

/*pressure_ return codes */
#define CUDA_PRESSURE_SUCCESS               0

/*##############------------------- PRESSURE submodule variable declarations ---------------------#################*/
/* Parameters */
/*---PRESSURE_GRADIENT_FORCE*/
extern __constant__ int pgfSelector_d;        /*Pressure Gradient Force (pgf) selector: 0=off, 1=on*/
extern float *hydroPres_d;            /*Base Adress of memory containing the diagnostic perturbation pressure field */

/*##############-------------- PRESSURE_CUDADEV submodule function declarations ------------------############*/

/*----->>>>> __device__ void  cudaDevice_calcPerturbationPressure();  -----------------------------------------------
* This is the cuda version of the calcPerturbationPressure routine from the HYDRO_CORE module
*/
__device__ void cudaDevice_calcPerturbationPressure(float* pres, float* rhoTheta, float* rhoTheta_BS, float* zPos_d);

/*----->>>>> __device__ void  cudaDevice_calcPerturbationPressureMoist();  ------------------------------------------
*/ 
__device__ void cudaDevice_calcPerturbationPressureMoist(float* pres, float* rho, float* rhoTheta, float* rhoTheta_BS, float* moist_qv, float* zPos_d);

/*----->>>>> __device__ void  cudaDevice_calcPressureGradientForce();  ---------------------------------------------
* This is the cuda version of the calcPressureGradientForce routine from the HYDRO_CORE module
*/
__device__ void cudaDevice_calcPressureGradientForce(float* Frhs_u, float* Frhs_v, float* Frhs_w, float* pres,
                                                     float* J31_d, float* J32_d, float* J33_d);

/*----->>>>> __device__ void  cudaDevice_calcPressureGradientForceMoist();  -----------------------------------------
*/ 
__device__ void cudaDevice_calcPressureGradientForceMoist(float* Frhs_u, float* Frhs_v, float* Frhs_w, float* rho,
                                                          float* pres, float* moistScalars,
                                                          float* J31_d, float* J32_d, float* J33_d);

#endif // _PRESSURE_CUDADEV_CU_H
