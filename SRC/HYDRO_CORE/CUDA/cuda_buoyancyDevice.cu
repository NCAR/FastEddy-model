/* FastEddy®: SRC/HYDRO_CORE/CUDA/cuda_buoyancyDevice.cu 
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

/*---BUOYANCY*/
__constant__ int buoyancySelector_d;          /*buoyancy Force selector: 0=off, 1=on*/

/*#################------------ BUOYANCY submodule function definitions ------------------#############*/
/*----->>>>> int cuda_buoyancyDeviceSetup();       ---------------------------------------------------------
 * Used to cudaMalloc and cudaMemcpy parameters and coordinate arrays, and for the BUOYANCY_CUDA submodule.
*/
extern "C" int cuda_buoyancyDeviceSetup(){
   int errorCode = CUDA_BUOYANCY_SUCCESS;

   cudaMemcpyToSymbol(buoyancySelector_d, &buoyancySelector, sizeof(int));

   return(errorCode);
} //end cuda_buoyancyDeviceSetup()

/*----->>>>> extern "C" int cuda_buoyancyDeviceCleanup();  -----------------------------------------------------------
Used to free all malloced memory by the BUOYANCY submodule.
*/

extern "C" int cuda_buoyancyDeviceCleanup(){
   int errorCode = CUDA_BUOYANCY_SUCCESS;

   /* Free any BUOYANCY submodule arrays */

   return(errorCode);

}//end cuda_buoyancyDeviceCleanup()

/*----->>>>> __device__ void  cudaDevice_calcBuoyancy();  --------------------------------------------------
* This is the cuda version of the calcBuoyancy routine from the HYDRO_CORE module
*/
__device__ void cudaDevice_calcBuoyancy(float* Frhs_w, float* rho, float* rho_BS){

  *Frhs_w = *Frhs_w - accel_g_d*(*rho - *rho_BS);
} // end cudaDevice_calcBuoyancy()

/*----->>>>> __device__ void  cudaDevice_calcBuoyancyMoistNvar1();  --------------------------------------------------
* Bouyancy term for single vapor only moisture species + dry air (see Klemp 2007 MWR)
*/
__device__ void cudaDevice_calcBuoyancyMoistNvar1(float* Frhs_w, float* rho, float* rho_BS, float* rho_v){

  *Frhs_w = *Frhs_w -(*rho/(*rho+*rho_v*1e-3))*accel_g_d*((*rho+*rho_v*1e-3) - *rho_BS);
} // end cudaDevice_calcBuoyancyMoistiNvar1()

/*----->>>>> __device__ void  cudaDevice_calcBuoyancyMoistNvar2();  --------------------------------------------------
* Bouyancy term for  vapor and liquid moisture species + dry air (see Klemp 2007 MWR)
*/
__device__ void cudaDevice_calcBuoyancyMoistNvar2(float* Frhs_w, float* rho, float* rho_BS, float* rho_v, float * rho_l){

  *Frhs_w = *Frhs_w -(*rho/(*rho+*rho_v*1e-3+*rho_l*1e-3))*accel_g_d*((*rho+*rho_v*1e-3+*rho_l*1e-3) - *rho_BS);
} // end cudaDevice_calcBuoyancyMoistNvar2()
