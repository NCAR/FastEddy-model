/* FastEddy®: SRC/TIME_INTEGRATION/CUDA/cuda_timeInt.c 
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
#include <cuda_timeInt.h>
#include <cuda_timeIntDevice.h>

/*######################------------------- CUDA_TIME_INTEGRATION module function definitions ---------------------##############*/

/*----->>>>> int cuda_timeIntInit();       ----------------------------------------------------------------------
Used to broadcast and print parameters, allocate memory, and initialize configuration settings for  the CUDA_TIME_INTEGRATION module.
*/
int cuda_timeIntInit(){
   int errorCode = CUDA_TIME_INTEGRATION_SUCCESS;

   /*Setup the timeInt parameters and arrays on the device*/
   errorCode = cuda_timeIntDeviceSetup();
      
   return(errorCode);
} //end cuda_timeIntInit()

/*----->>>>> int cuda_timeIntCleanup(); -------------------------------------------------------------------
Used to free all malloced memory by the CUDA_TIME_INTEGRATION module.
*/
int cuda_timeIntCleanup(){
   int errorCode = CUDA_TIME_INTEGRATION_SUCCESS;

   /* Free any CUDA_TIME_INTEGRATION module device-arrays */
   errorCode = cuda_timeIntDeviceCleanup();

   return(errorCode);

}//end cuda_timeIntCleanup()

/*----->>>>> int cuda_timeIntCommence(); -------------------------------------------------------------------
* Used to place the GPU in timeIntegration commence mode.
*/
int cuda_timeIntCommence(int it){
   int errorCode = CUDA_TIME_INTEGRATION_SUCCESS;

   /* Launch the kernel */
   errorCode = cuda_timeIntDeviceCommence(it);

   return(errorCode);

}//end cuda_timeIntCommence()

