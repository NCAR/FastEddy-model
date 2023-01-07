/* FastEddy®: SRC/HYDRO_CORE/CUDA/cuda_BaseStateDevice_cu.h 
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
#ifndef _BASESTATE_CUDADEV_CU_H
#define _BASESTATE_CUDADEV_CU_H

/*BaseState return codes */
#define CUDA_BASESTATE_SUCCESS    0

/*##############------------------- BASESTATE submodule variable declarations ---------------------#################*/
/*---BASESTATE*/
extern float *hydroBaseStateFlds_d;   /*Base Adress of memory containing rho/theta prognostic variable base-states */
extern float *hydroBaseStatePres_d;   /*Base Adress of memory containing the diagnostic base-state pressure field */

/*##############-------------- BASESTATE_CUDADEV submodule function declarations ------------------############*/

/*----->>>>> int cuda_BaseStateDeviceSetup();       ---------------------------------------------------------
* Used to cudaMalloc and cudaMemcpy parameters and coordinate arrays, and for the BASESTATE_CUDA submodule.
*/
extern "C" int cuda_BaseStateDeviceSetup();

/*----->>>>> extern "C" int cuda_BaseStateDeviceCleanup();  -----------------------------------------------------------
* Used to free all malloced memory by the BASESTATE submodule.
*/
extern "C" int cuda_BaseStateDeviceCleanup();

#endif // _BASESTATE_CUDADEV_CU_H
