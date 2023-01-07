/* FastEddy®: SRC/TIME_INTEGRATION/CUDA/cuda_timeIntDevice.h 
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
#ifndef _TIME_INTEGRATION_CUDADEV_H
#define _TIME_INTEGRATION_CUDADEV_H

/*timeInt_ return codes */
#define CUDA_TIME_INTEGRATION_SUCCESS               0


/*######################------------------- TIME_INTEGRATION module variable declarations ---------------------#################*/
/* Parameters */

/*###################------------------- TIME_INTEGRATION_CUDADEV module function declarations ---------------------#################*/

/*----->>>>> int cuda_timeIntDeviceSetup();       ----------------------------------------------------------------------
Used to cudaMalloc and cudaMemcpy parameters and coordinate arrays for the TIME_INTEGRATION_CUDADEV module.
*/
int cuda_timeIntDeviceSetup();

/*----->>>>> int cuda_timeIntDeviceCleanup();  ---------------------------------------------------------------------
Used to free all malloced memory by the TIME_INTEGRATION_CUDADEV module.
*/
int cuda_timeIntDeviceCleanup();

/*----->>>>> int cuda_timeIntDeviceCommence();  -----------------------------------------------------------
* This routine provides the externally callable cuda-kernel call to commence with timeIntegration
*/
int cuda_timeIntDeviceCommence(int it);

#endif // _TIME_INTEGRATION_CUDADEV_H
