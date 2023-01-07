/* FastEddy®: SRC/TIME_INTEGRATION/CUDA/cuda_RKSchemes_cu.h 
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
#ifndef _RKSCHEMES_CUDADEV_CU_H
#define _RKSCHEMES_CUDADEV_CU_H

/*rkschemes_ return codes */
#define CUDA_RKSCHEMES_SUCCESS               0

/*----->>>>> __global__ void  cudaDevice_timeIntegrationCommenceRK3_WS2002();  --------------------------------------------
* This is the gloabl-entry kernel routine used by the TIME_INTEGRATION module for the
* Runge-Kutta 3 Wicker and Skamarock (2002) MWR paper formulation.
*/
__global__ void cudaDevice_timeIntegrationCommenceRK3_WS2002(int Nphi, float* phi_Flds, float* phi_Frhs,
                                                             int Nsgstke, float* sgstkeSc_Flds, float* sgstkeSc_Frhs,
                                                             int Nmoist, float* moistSc_Flds, float* moistSc_Frhs,
                                                             float* timeFlds0, int RKstage);

/*----->>>>> __device__ void  cudaDevice_RungeKutta3WS02Stage1();  --------------------------------------------------
* This is the device function to perform stage 1 of 3 from the  Runge-Kutta-3 WS02 time_integration scheme 
*/
__device__ void cudaDevice_RungeKutta3WS02Stage1(float* currFld, float* currFrhs, float* currFld0);

/*----->>>>> __device__ void  cudaDevice_RungeKutta3WS02Stage2();  --------------------------------------------------
* This is the device function to perform stage 2 of 3 from the Runge-Kutta-3 WS02 time_integration scheme 
*/
__device__ void cudaDevice_RungeKutta3WS02Stage2(float* currFld, float* currFrhs, float* currFld0);

/*----->>>>> __device__ void  cudaDevice_RungeKutta3WS02Stage3();  --------------------------------------------------
* This is the device function to perform stage 3 of 3 from the Runge-Kutta-3 WS02 time_integration scheme 
*/
__device__ void cudaDevice_RungeKutta3WS02Stage3(float* currFld, float* currFrhs, float* currFld0);

/*----->>>>> __device__ void  cudaDevice_PositiveDef();  --------------------------------------------------
*/ // Def
__device__ void cudaDevice_PositiveDef(float* Fld, float min_threshold);

#endif // _RKSCHEMES_CUDADEV_CU_H
