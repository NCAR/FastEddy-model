/* FastEddy®: SRC/HYDRO_CORE/CUDA/cuda_hydroCore.h 
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
#ifndef _CUDA_HYDRO_CORE_H
#define _CUDA_HYDRO_CORE_H

/*cuda_hydroCore_ return codes */
#define CUDA_HYDRO_CORE_SUCCESS               0

/*######################------------------- CUDA_HYDRO_CORE module variable declarations ---------------------#################*/
/* Parameters */

/*##################------------------- CUDA_HYDRO_CORE module function declarations ---------------------#################*/

/*----->>>>> int cuda_hydroCoreInit();       ----------------------------------------------------------------------
Used to broadcast and print parameters, allocate memory, and initialize configuration settings for  the CUDA_HYDRO_CORE module.
*/
int cuda_hydroCoreInit();

/*----->>>>> int cuda_hydroCoreCleanup();       ----------------------------------------------------------------------
Used to free all malloced memory by the CUDA_HYDRO_CORE module.
*/
int cuda_hydroCoreCleanup();

#endif // _CUDA_HYDRO_CORE_H
