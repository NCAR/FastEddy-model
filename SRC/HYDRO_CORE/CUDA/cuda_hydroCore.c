/* FastEddy®: SRC/HYDRO_CORE/CUDA/cuda_hydroCore.c 
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
#include <cuda_hydroCore.h>
#include <cuda_hydroCoreDevice.h>

/*######################------------------- CUDA_HYDRO_CORE module function definitions ---------------------##############*/

/*----->>>>> int cuda_hydroCoreInit();       ----------------------------------------------------------------------
Used to broadcast and print parameters, allocate memory, and initialize configuration settings for  the CUDA_HYDRO_CORE module.
*/
int cuda_hydroCoreInit(){
   int errorCode = CUDA_HYDRO_CORE_SUCCESS;

   /*Setup the hydroCore parameters and arrays on the device*/
   errorCode = cuda_hydroCoreDeviceSetup();
      
   return(errorCode);
} //end cuda_hydroCoreInit()

/*----->>>>> int cuda_hydroCoreCleanup(); -------------------------------------------------------------------
Used to free all malloced memory by the CUDA_HYDRO_CORE module.
*/
int cuda_hydroCoreCleanup(){
   int errorCode = CUDA_HYDRO_CORE_SUCCESS;

   /* Free any CUDA_HYDRO_CORE module device-arrays */
   errorCode = cuda_hydroCoreDeviceCleanup();

   return(errorCode);

}//end cuda_hydroCoreCleanup()
