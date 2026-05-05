/* FastEddy®: SRC/HYDRO_CORE/CUDA/cuda_coriolisDevice.cu 
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
/*---CORIOLIS*/ 
__constant__ int coriolisSelector_d;          /*coriolis Force selector: 0=off, 1=on*/
__constant__ float corioConstHorz_d;          /*coriolis horizontal term constant */
__constant__ float corioConstVert_d;          /*coriolis vertical term constant */
__constant__ float corioLS_fact_d;            /*large-scale forcing factor on Coriolis term*/
float* lat_d; /* latitude in degrees north "()" 2-d array (x by y) (m)*/
float* lon_d; /* longitude in degrees east "()" 2-d array (x by y) (m)*/


/*#################------------ CORIOLIS submodule function definitions ------------------#############*/
/*----->>>>> int cuda_coriolisDeviceSetup();       ---------------------------------------------------------
 * Used to cudaMalloc and cudaMemcpy parameters and coordinate arrays, and for the CORIOLIS_CUDA submodule.
*/
extern "C" int cuda_coriolisDeviceSetup(){
   int errorCode = CUDA_CORIOLIS_SUCCESS;
   size_t Nelems2d;

   cudaMemcpyToSymbol(coriolisSelector_d, &coriolisSelector, sizeof(int));
   cudaMemcpyToSymbol(corioConstHorz_d, &corioConstHorz, sizeof(float));
   cudaMemcpyToSymbol(corioConstVert_d, &corioConstVert, sizeof(float));
   cudaMemcpyToSymbol(corioLS_fact_d, &corioLS_fact, sizeof(float));

   Nelems2d = (size_t)((Nxp+2*Nh)*(Nyp+2*Nh));
   fecuda_DeviceMalloc(Nelems2d, &lat_d);
   cudaMemcpy(lat_d, lat, Nelems2d*sizeof(float), cudaMemcpyHostToDevice);
   fecuda_DeviceMalloc(Nelems2d, &lon_d);
   cudaMemcpy(lon_d, lon, Nelems2d*sizeof(float), cudaMemcpyHostToDevice);

   return(errorCode);
} //end cuda_coriolisDeviceSetup()

/*----->>>>> extern "C" int cuda_coriolisDeviceCleanup();  -----------------------------------------------------------
Used to free all malloced memory by the CORIOLIS submodule.
*/

extern "C" int cuda_coriolisDeviceCleanup(){
   int errorCode = CUDA_CORIOLIS_SUCCESS;

   /* Free any CORIOLIS submodule arrays */
   cudaFree(lat_d);
   cudaFree(lon_d);

   return(errorCode);

}//end cuda_coriolisDeviceCleanup()

/*----->>>>> __device__ void  cudaDevice_calcCoriolis();  --------------------------------------------------
* This is the cuda version of the calcCoriolis routine from the HYDRO_CORE module
*/
__device__ void cudaDevice_calcCoriolis(float* Frhs_u, float* Frhs_v, float* Frhs_w,
                                        float* rho, float* uMom, float* vMom, float* wMom,
                                        float* rhoBS, float* uBS, float* vBS, float* wBS,
					float* lat){
  float pi = acosf(-1.0);
  float lat_factH;
  float lat_factV;

  lat_factH = sinf(pi/180.0*(*lat));
  lat_factV = cosf(pi/180.0*(*lat));

  *Frhs_u = *Frhs_u + ( corioConstHorz_d*lat_factH*((*vMom)/(*rho)-corioLS_fact_d*(*vBS)/(*rhoBS))
                       -corioConstVert_d*lat_factV*((*wMom)/(*rho)-corioLS_fact_d*(*wBS)/(*rhoBS)) );
  *Frhs_v = *Frhs_v - ( corioConstHorz_d*lat_factH*((*uMom)/(*rho)-corioLS_fact_d*(*uBS)/(*rhoBS)) );
  *Frhs_w = *Frhs_w + ( corioConstVert_d*lat_factV*((*uMom)/(*rho)-corioLS_fact_d*(*uBS)/(*rhoBS)) );
} // end cudaDevice_calcCoriolis()
