/* FastEddy®: SRC/HYDRO_CORE/CUDA/cuda_hydroCoreDevice.h 
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
#ifndef _HYDRO_CORE_CUDADEV_H
#define _HYDRO_CORE_CUDADEV_H

/*hydroCore_ return codes */
#define CUDA_HYDRO_CORE_SUCCESS               0


/*################------------------- HYDRO_CORE module variable declarations ---------------------#################*/
/* Parameters */

/*############------------------- HYDRO_CORE_CUDADEV module function declarations ---------------------############*/

/*----->>>>> int cuda_hydroCoreDeviceSetup();  -------------------------------------------------------------------
Used to cudaMalloc and cudaMemcpy parameters and coordinate arrays for the HYDRO_CORE_CUDADEV module.
*/
int cuda_hydroCoreDeviceSetup();

/*----->>>>> int cuda_hydroCoreDeviceCleanup();  ---------------------------------------------------------------
Used to free all malloced memory by the HYDRO_CORE_CUDADEV module.
*/
int cuda_hydroCoreDeviceCleanup();

#endif // _HYDRO_CORE_CUDADEV_H
