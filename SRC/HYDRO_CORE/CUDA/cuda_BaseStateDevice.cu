/* FastEddy®: SRC/HYDRO_CORE/CUDA/cuda_BaseStateDevice.cu 
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
/*##############------------------- BASESTATE submodule variable declarations ---------------------#################*/
/*---BASESTATE*/
float *hydroBaseStateFlds_d;   /*Base Adress of memory containing all prognostic variable fields base-states */
float *hydroBaseStatePres_d;   /*Base Adress of memory containing the diagnostic base-state pressure field */

/*#################------------ BASESTATE submodule function definitions ------------------#############*/
/*----->>>>> int cuda_BaseStateDeviceSetup();       ---------------------------------------------------------
 * Used to cudaMalloc and cudaMemcpy parameters and coordinate arrays, and for the BASESTATE_CUDA submodule.
*/
extern "C" int cuda_BaseStateDeviceSetup(){
   int errorCode = CUDA_BASESTATE_SUCCESS;
   int Nelems;

   /*Set the full memory block number of elements for base-state fields*/
   Nelems = (Nxp+2*Nh)*(Nyp+2*Nh)*(Nzp+2*Nh);
   /* Allocate the Base State arrays on the device */
   fecuda_DeviceMalloc(Nelems*2*sizeof(float), &hydroBaseStateFlds_d);  //Only rho and theta base-state variables
   fecuda_DeviceMalloc(Nelems*sizeof(float), &hydroBaseStatePres_d);  //Only base-state pressure 

   /* Send the Base State arrays down to the device */
   cudaMemcpy(hydroBaseStateFlds_d, hydroBaseStateFlds, Nelems*2*sizeof(float), cudaMemcpyHostToDevice);
   gpuErrchk( cudaPeekAtLastError() ); /*Check for errors in the cudaMemCpy calls*/
   cudaMemcpy(hydroBaseStatePres_d, hydroBaseStatePres, Nelems*sizeof(float), cudaMemcpyHostToDevice); 
   gpuErrchk( cudaPeekAtLastError() ); /*Check for errors in the cudaMemCpy calls*/

   return(errorCode);
} //end cuda_BaseStateDeviceSetup()

/*----->>>>> extern "C" int cuda_BaseStateDeviceCleanup();  -----------------------------------------------------------
Used to free all malloced memory by the BASESTATE submodule.
*/

extern "C" int cuda_BaseStateDeviceCleanup(){
   int errorCode = CUDA_BASESTATE_SUCCESS;

   /* Free any BASESTATE submodule arrays */
   cudaFree(hydroBaseStateFlds_d);
   cudaFree(hydroBaseStatePres_d);
 
   return(errorCode);

}//end cuda_BaseStateDeviceCleanup()
