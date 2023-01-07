/* FastEddy®: SRC/HYDRO_CORE/CUDA/cuda_filtersDevice.cu 
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
/*---EXPLICIT FILTERS*/ 
__constant__ int filterSelector_d;     /* explicit filter selector: 0=off, 1=on */
__constant__ float filter_6th_coeff_d; /* 6th-order filter factor: 0.0=off, 1.0=full */

/*#################------------ FILTERS submodule function definitions ------------------#############*/
/*----->>>>> int cuda_filtersDeviceSetup();       ---------------------------------------------------------
 * Used to cudaMalloc and cudaMemcpy parameters and coordinate arrays, and for the FILTERS_CUDA submodule.
*/
extern "C" int cuda_filtersDeviceSetup(){
   int errorCode = CUDA_FILTERS_SUCCESS;

   cudaMemcpyToSymbol(filterSelector_d, &filterSelector, sizeof(int));
   cudaMemcpyToSymbol(filter_6th_coeff_d, &filter_6th_coeff, sizeof(float));

   return(errorCode);
} //end cuda_filtersDeviceSetup()

/*----->>>>> extern "C" int cuda_filtersDeviceCleanup();  -----------------------------------------------------------
Used to free all malloced memory by the FILTERS submodule.
*/

extern "C" int cuda_filtersDeviceCleanup(){
   int errorCode = CUDA_FILTERS_SUCCESS;

   /* Free any FILTERS submodule arrays */

   return(errorCode);

}//end cuda_filtersDeviceCleanup()

__global__ void cudaDevice_hydroCoreUnitTestCompleteFilters(float* hydroFlds_d, float* hydroFldsFrhs_d, float dt){
   int i,j,k;
   int fldStride,iFld;
   int iFld_s,iFld_e;
   float* fld;
   float* fld_Frhs;

   iFld_s = W_INDX;
   iFld_e = iFld_s + 1;

   /*Establish necessary indices for spatial locality*/
   i = (blockIdx.x)*blockDim.x + threadIdx.x;
   j = (blockIdx.y)*blockDim.y + threadIdx.y;
   k = (blockIdx.z)*blockDim.z + threadIdx.z;

   fldStride = (Nx_d+2*Nh_d)*(Ny_d+2*Nh_d)*(Nz_d+2*Nh_d);

   if((i >= iMin_d)&&(i < iMax_d) &&(j >= jMin_d)&&(j < jMax_d) &&(k >= kMin_d)&&(k < kMax_d)){
     for(iFld=iFld_s; iFld < iFld_e; iFld++){
       fld = &hydroFlds_d[fldStride*iFld];
       fld_Frhs = &hydroFldsFrhs_d[fldStride*iFld];
       cudaDevice_filter6th(fld,fld_Frhs,dt);
     }
   }//end if in the range of non-halo cells

} // end cudaDevice_hydroCoreUnitTestCompleteFilters

/*----->>>>> __device__ void cudaDevice_filter6th();  --------------------------------------------------
*/ // 6th-order filter to remove 2dx noise: Xue (2000), Knievel et al. (2007)
__device__ void cudaDevice_filter6th(float* fld, float* fld_Fhrs, float dt){
  int i,j,k;
  int ijk,ijkm1,ijkm2,ijkm3,ijkp1,ijkp2,ijkp3;
  int iStride,jStride,kStride;
  float filter_coeff;
  float pases_space_dir = 1.0; // number of spatial directions over which the filter is applied
  float coeff_1 = 10.0;
  float coeff_2 = 5.0;
  float flxz_kmf,flxz_kpf;
  float grad_m,grad_p,diff_z;

  filter_coeff = 0.015625*filter_6th_coeff_d/(pases_space_dir*dt);
 
  /*Establish necessary indices for spatial locality*/
  i = (blockIdx.x)*blockDim.x + threadIdx.x;
  j = (blockIdx.y)*blockDim.y + threadIdx.y;
  k = (blockIdx.z)*blockDim.z + threadIdx.z;

  iStride = (Ny_d+2*Nh_d)*(Nz_d+2*Nh_d);
  jStride = (Nz_d+2*Nh_d);
  kStride = 1;

  ijk = i*iStride + j*jStride + k*kStride;
  ijkm1 = i*iStride + j*jStride + (k-1)*kStride;
  ijkm2 = i*iStride + j*jStride + (k-2)*kStride;
  ijkm3 = i*iStride + j*jStride + (k-3)*kStride;
  ijkp1 = i*iStride + j*jStride + (k+1)*kStride;
  ijkp2 = i*iStride + j*jStride + (k+2)*kStride;
  ijkp3 = i*iStride + j*jStride + (k+3)*kStride;
 
  grad_m = fld[ijk] - fld[ijkm1];
  grad_p = fld[ijkp1] - fld[ijk];

  //*** z-direction filtering ***//
  flxz_kmf = coeff_1*(fld[ijk] -   fld[ijkm1]) - coeff_2*(fld[ijkp1] - fld[ijkm2]) + (fld[ijkp2] - fld[ijkm3]);
  flxz_kpf = coeff_1*(fld[ijkp1] - fld[ijk]  ) - coeff_2*(fld[ijkp2] - fld[ijkm1]) + (fld[ijkp3] - fld[ijkm2]);
  grad_m = grad_m*flxz_kmf;
  grad_p = grad_p*flxz_kpf;

  // limiter to ensure downgradient diffusion //
  flxz_kmf = fmaxf(copysign(1.0,grad_m),0.0)*flxz_kmf;
  flxz_kpf = fmaxf(copysign(1.0,grad_p),0.0)*flxz_kpf;

  diff_z = filter_coeff*(flxz_kpf - flxz_kmf);
  
  fld_Fhrs[ijk] = fld_Fhrs[ijk] + diff_z;

} //end cudaDevice_filter6th
