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
__constant__ int filter_6thdiff_vert_d;          /* vertical 6th-order filter on w selector: 0=off, 1=on */
__constant__ float filter_6thdiff_vert_coeff_d;  /* vertical 6th-order filter w factor: 0.0=off, 1.0=full */
__constant__ int filter_6thdiff_hori_d;          /* horizontal 6th-order filter on rho,theta,qv selector: 0=off, 1=on */
__constant__ float filter_6thdiff_hori_coeff_d;  /* horizontal 6th-order filter factor: 0.0=off, 1.0=full */
__constant__ int filter_divdamp_d;               /* divergence damping selector: 0=off, 1=on */

/*#################------------ FILTERS submodule function definitions ------------------#############*/
/*----->>>>> int cuda_filtersDeviceSetup();       ---------------------------------------------------------
 * Used to cudaMalloc and cudaMemcpy parameters and coordinate arrays, and for the FILTERS_CUDA submodule.
*/
extern "C" int cuda_filtersDeviceSetup(){
   int errorCode = CUDA_FILTERS_SUCCESS;

   cudaMemcpyToSymbol(filter_6thdiff_vert_d, &filter_6thdiff_vert, sizeof(int));
   cudaMemcpyToSymbol(filter_6thdiff_vert_coeff_d, &filter_6thdiff_vert_coeff, sizeof(float));
   cudaMemcpyToSymbol(filter_6thdiff_hori_d, &filter_6thdiff_hori, sizeof(int));
   cudaMemcpyToSymbol(filter_6thdiff_hori_coeff_d, &filter_6thdiff_hori_coeff, sizeof(float));
   cudaMemcpyToSymbol(filter_divdamp_d, &filter_divdamp, sizeof(int));

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

__global__ void cudaDevice_hydroCoreUnitTestCompleteFilters(float* hydroFlds_d, float* hydroFldsFrhs_d, float dt,
                                                            float* moistScalars_d, float* moistScalarsFrhs, float* hydroPres_d,
                                                            float* hydroBaseStatePres_d, int timeStage){
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
     if(filter_6thdiff_vert_d==1){ // 6th-order vertical fiter on w
       for(iFld=iFld_s; iFld < iFld_e; iFld++){
         fld = &hydroFlds_d[fldStride*iFld];
         fld_Frhs = &hydroFldsFrhs_d[fldStride*iFld];
         cudaDevice_filter6th(fld,fld_Frhs,dt);
       }
     }
     if(filter_divdamp_d==1){ // divergence damping
       cudaDevice_divergenceDamping(&hydroFldsFrhs_d[fldStride*U_INDX], &hydroFldsFrhs_d[fldStride*V_INDX], &hydroFldsFrhs_d[fldStride*THETA_INDX],
                                    &hydroFlds_d[fldStride*THETA_INDX], &hydroFlds_d[fldStride*RHO_INDX], &moistScalars_d[0],
                                    &hydroPres_d[0], &hydroBaseStatePres_d[0], dt, timeStage);
     }
     if(filter_6thdiff_hori_d==1){ // 6th-order horizontal filter on rho, theta, qv
       for(iFld=0; iFld < 5; iFld=iFld+4){
         fld = &hydroFlds_d[fldStride*iFld];
         fld_Frhs = &hydroFldsFrhs_d[fldStride*iFld];
         cudaDevice_filter6th2D(fld,fld_Frhs,dt);
       }
       if (moistureSelector_d > 0){
         fld = &moistScalars_d[0];
         fld_Frhs = &moistScalarsFrhs[0];
         cudaDevice_filter6th2D(fld,fld_Frhs,dt);
       }
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

  filter_coeff = 0.015625*filter_6thdiff_vert_coeff_d/(pases_space_dir*dt);
 
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

/*----->>>>> __device__ void cudaDevice_filter6th2D();  --------------------------------------------------
*/ // 6th-order filter to remove 2dx noise: Xue (2000), Knievel et al. (2007)
__device__ void cudaDevice_filter6th2D(float* fld, float* fld_Frhs, float dt){
  int i,j,k;
  int ijk,im1jk,im2jk,im3jk,ip1jk,ip2jk,ip3jk,ijm1k,ijm2k,ijm3k,ijp1k,ijp2k,ijp3k;
  int iStride,jStride,kStride;
  float filter_coeff;
  float pases_space_dir = 2.0; // number of spatial directions over which the filter is applied
  float coeff_1 = 10.0;
  float coeff_2 = 5.0;
  float flxx_imf,flxx_ipf,flxy_jmf,flxy_jpf;
  float grad_m,grad_p,diff_x,diff_y;

  filter_coeff = 0.015625*filter_6thdiff_hori_coeff_d/(pases_space_dir*dt);

  /*Establish necessary indices for spatial locality*/
  i = (blockIdx.x)*blockDim.x + threadIdx.x;
  j = (blockIdx.y)*blockDim.y + threadIdx.y;
  k = (blockIdx.z)*blockDim.z + threadIdx.z;

  iStride = (Ny_d+2*Nh_d)*(Nz_d+2*Nh_d);
  jStride = (Nz_d+2*Nh_d);
  kStride = 1;

  ijk = i*iStride + j*jStride + k*kStride;
  im1jk = (i-1)*iStride + j*jStride + k*kStride;
  im2jk = (i-2)*iStride + j*jStride + k*kStride;
  im3jk = (i-3)*iStride + j*jStride + k*kStride;
  ip1jk = (i+1)*iStride + j*jStride + k*kStride;
  ip2jk = (i+2)*iStride + j*jStride + k*kStride;
  ip3jk = (i+3)*iStride + j*jStride + k*kStride;
  ijm1k = i*iStride + (j-1)*jStride + k*kStride;
  ijm2k = i*iStride + (j-2)*jStride + k*kStride;
  ijm3k = i*iStride + (j-3)*jStride + k*kStride;
  ijp1k = i*iStride + (j+1)*jStride + k*kStride;
  ijp2k = i*iStride + (j+2)*jStride + k*kStride;
  ijp3k = i*iStride + (j+3)*jStride + k*kStride;

  //*** x-direction filtering ***//
  grad_m = fld[ijk] - fld[im1jk];
  grad_p = fld[ip1jk] - fld[ijk];

  flxx_imf = coeff_1*(fld[ijk] -   fld[im1jk]) - coeff_2*(fld[ip1jk] - fld[im2jk]) + (fld[ip2jk] - fld[im3jk]);
  flxx_ipf = coeff_1*(fld[ip1jk] - fld[ijk]  ) - coeff_2*(fld[ip2jk] - fld[im1jk]) + (fld[ip3jk] - fld[im2jk]);
  grad_m = grad_m*flxx_imf;
  grad_p = grad_p*flxx_ipf;

  // limiter to ensure downgradient diffusion //
  flxx_imf = fmaxf(copysign(1.0,grad_m),0.0)*flxx_imf;
  flxx_ipf = fmaxf(copysign(1.0,grad_p),0.0)*flxx_ipf;

  diff_x = filter_coeff*(flxx_ipf - flxx_imf);

  //*** y-direction filtering ***//
  grad_m = fld[ijk] - fld[ijm1k];
  grad_p = fld[ijp1k] - fld[ijk];

  flxy_jmf = coeff_1*(fld[ijk] -   fld[ijm1k]) - coeff_2*(fld[ijp1k] - fld[ijm2k]) + (fld[ijp2k] - fld[ijm3k]);
  flxy_jpf = coeff_1*(fld[ijp1k] - fld[ijk]  ) - coeff_2*(fld[ijp2k] - fld[ijm1k]) + (fld[ijp3k] - fld[ijm2k]);
  grad_m = grad_m*flxy_jmf;
  grad_p = grad_p*flxy_jpf;

  // limiter to ensure downgradient diffusion //
  flxy_jmf = fmaxf(copysign(1.0,grad_m),0.0)*flxy_jmf;
  flxy_jpf = fmaxf(copysign(1.0,grad_p),0.0)*flxy_jpf;

  diff_y = filter_coeff*(flxy_jpf - flxy_jmf);

  fld_Frhs[ijk] = fld_Frhs[ijk] + diff_x + diff_y;

} //end cudaDevice_filter6th2D

/*----->>>>> __device__ void cudaDevice_divergenceDamping();  --------------------------------------------------
*/ // Divergence damping applied at the last RK step
__device__ void cudaDevice_divergenceDamping(float* uFrhs, float* vFrhs, float* thetaFrhs,
                                             float* theta, float* rho, float* moistScalars,
                                             float* pres, float* baseStatePres, float dt, int timeStage){
  int i,j,k;
  int ijk,ip1jk,im1jk,ijp1k,ijm1k;
  int iStride,jStride,kStride,fldStride;
  float rhomd_ijk,rhodm_ijk;
  int iFld;
  float beta_d; // damping coefficient
  float gpt_ip1jk,gpt_im1jk,gpt_ijp1k,gpt_ijm1k; // p*gamma/pt

  if (timeStage == 0) {
    beta_d = 0.1/3.0;
  } else if (timeStage == 1) {
    beta_d = 0.1/2.0;
  } else if (timeStage == 2) {
    beta_d = 0.1;
  }

  /*Establish necessary indices for spatial locality*/
  i = (blockIdx.x)*blockDim.x + threadIdx.x;
  j = (blockIdx.y)*blockDim.y + threadIdx.y;
  k = (blockIdx.z)*blockDim.z + threadIdx.z;

  iStride = (Ny_d+2*Nh_d)*(Nz_d+2*Nh_d);
  jStride = (Nz_d+2*Nh_d);
  kStride = 1;

  ijk = i*iStride + j*jStride + k*kStride;
  ip1jk = (i+1)*iStride + j*jStride + k*kStride;
  ijp1k = i*iStride + (j+1)*jStride + k*kStride;
  im1jk = (i-1)*iStride + j*jStride + k*kStride;
  ijm1k = i*iStride + (j-1)*jStride + k*kStride;

  if((i >= iMin_d)&&(i < iMax_d) &&
     (j >= jMin_d)&&(j < jMax_d) &&
     (k >= kMin_d)&&(k < kMax_d)){

    rhomd_ijk = 1.0;
    if (moistureSelector_d > 0){
      for (iFld=0; iFld < moistureNvars_d; iFld++){
         rhomd_ijk = rhomd_ijk + moistScalars[fldStride*iFld+ijk]/rho[ijk]*1e-3;
      }
    }
    rhodm_ijk = 1.0/rhomd_ijk;

    gpt_ip1jk = (pres[ip1jk] + baseStatePres[ip1jk])*cp_cv_d/theta[ip1jk];
    gpt_im1jk = (pres[im1jk] + baseStatePres[im1jk])*cp_cv_d/theta[im1jk];
    gpt_ijp1k = (pres[ijp1k] + baseStatePres[ijp1k])*cp_cv_d/theta[ijp1k];
    gpt_ijm1k = (pres[ijm1k] + baseStatePres[ijm1k])*cp_cv_d/theta[ijm1k];

    uFrhs[ijk] = uFrhs[ijk] -rhodm_ijk*0.5*dXi_d*beta_d*dt*
                  (gpt_ip1jk*thetaFrhs[ip1jk] - gpt_im1jk*thetaFrhs[im1jk]);
    vFrhs[ijk] = vFrhs[ijk] -rhodm_ijk*0.5*dYi_d*beta_d*dt*
                  (gpt_ijp1k*thetaFrhs[ijp1k] - gpt_ijm1k*thetaFrhs[ijm1k]);
  }

}//end cudaDevice_divergenceDamping
