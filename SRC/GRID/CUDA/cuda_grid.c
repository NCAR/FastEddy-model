/* FastEddy®: SRC/GRID/CUDA/cuda_grid.c 
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
#include <cuda_grid.h>
#include <cuda_gridDevice.h>

/*######################------------------- CUDA_GRID module function definitions ---------------------##############*/

/*----->>>>> int cuda_gridInit();       ----------------------------------------------------------------------
Used to broadcast and print parameters, allocate memory, and initialize configuration settings for  the CUDA_GRID module.
*/
int cuda_gridInit(){
   int errorCode = CUDA_GRID_SUCCESS;

   /*Setup the grid parameters and arrays on the device*/
   errorCode = cuda_gridDeviceSetup();
      
   return(errorCode);
} //end cuda_gridInit()

/*----->>>>> int cuda_gridCleanup(); -------------------------------------------------------------------
Used to free all malloced memory by the CUDA_GRID module.
*/
int cuda_gridCleanup(){
   int errorCode = CUDA_GRID_SUCCESS;

   /* Free any CUDA_GRID module device-arrays */
   errorCode = cuda_gridDeviceCleanup();

   return(errorCode);

}//end cuda_gridCleanup()
