/* FastEddy®: SRC/TIME_INTEGRATION/CUDA/cuda_timeInt.h 
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
#ifndef _CUDA_TIME_INTEGRATION_H
#define _CUDA_TIME_INTEGRATION_H

/*cuda_timeInt_ return codes */
#define CUDA_TIME_INTEGRATION_SUCCESS               0

/*######################------------------- CUDA_TIME_INTEGRATION module variable declarations ---------------------#################*/
/* Parameters */

/*##################------------------- CUDA_TIME_INTEGRATION module function declarations ---------------------#################*/

/*----->>>>> int cuda_timeIntInit();       ----------------------------------------------------------------------
Used to broadcast and print parameters, allocate memory, and initialize configuration settings for  the CUDA_TIME_INTEGRATION module.
*/
int cuda_timeIntInit();

/*----->>>>> int cuda_timeIntCleanup();       ----------------------------------------------------------------------
Used to free all malloced memory by the CUDA_TIME_INTEGRATION module.
*/
int cuda_timeIntCleanup();

/*----->>>>> int cuda_timeIntCommence(); -------------------------------------------------------------------
* Used to place the GPU in timeIntegration commence mode.
*/
int cuda_timeIntCommence(int it);

#endif // _CUDA_TIME_INTEGRATION_H
