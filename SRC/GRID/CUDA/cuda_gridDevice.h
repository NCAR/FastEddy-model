/* FastEddy®: SRC/GRID/CUDA/cuda_gridDevice.h 
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
#ifndef _GRID_CUDADEV_H
#define _GRID_CUDADEV_H

/*grid_ return codes */
#define CUDA_GRID_SUCCESS               0


/*######################------------------- GRID module variable declarations ---------------------#################*/
/* Parameters */

/*###################------------------- GRID_CUDADEV module function declarations ---------------------#################*/

/*----->>>>> int cuda_gridDeviceSetup();       ----------------------------------------------------------------------
Used to cudaMalloc and cudaMemcpy parameters and coordinate arrays for the GRID_CUDADEV module.
*/
int cuda_gridDeviceSetup();

/*----->>>>> int cuda_gridDeviceCleanup();       ----------------------------------------------------------------------
Used to free all malloced memory by the GRID_CUDADEV module.
*/
int cuda_gridDeviceCleanup();


#endif // _GRID_CUDADEV_H
