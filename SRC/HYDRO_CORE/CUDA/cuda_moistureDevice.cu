/* FastEddy®: SRC/HYDRO_CORE/CUDA/cuda_moistureDevice.cu 
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
/*---MOISTURE*/ 
__constant__ int moistureSelector_d;     /* moisture selector: 0=off, 1=on */
__constant__ int moistureNvars_d;        /* number of moisture species */
__constant__ int moistureAdvSelectorQv_d;  /* water vapor advection scheme selector */
__constant__ float moistureAdvSelectorQv_b_d; /* hybrid advection scheme parameter */
__constant__ int moistureSGSturb_d;      /* selector to apply sub-grid scale diffusion to moisture fields */
__constant__ int moistureCond_d;         /* selector to apply condensation to mositure fields */
__constant__ int moistureAdvSelectorQi_d;  /* moisture advection scheme selector for non-qv fields (non-oscillatory schemes) */
__constant__ float moistureCondTscale_d; /* relaxation time in seconds */
__constant__ int moistureCondBasePres_d;  /* selector to use base pressure for microphysics */
__constant__ float moistureMPcallTscale_d;  /* time scale for microphysics to be called */
float* moistScalars_d;                   /*Base address for moisture field arrays*/
float* moistScalarsFrhs_d;               /*Base address for moisture forcing field arrays*/
float* moistTauFlds_d;                   /*Base address for moisture SGS field arrays*/
float* fcond_d;                          /*Base address for f_cond array*/

/*#################------------ MOISTURE submodule function definitions ------------------#############*/
/*----->>>>> int cuda_moistureDeviceSetup();       ---------------------------------------------------------
 * Used to cudaMalloc and cudaMemcpy parameters and coordinate arrays, and for the MOISTURE_CUDA submodule.
*/
extern "C" int cuda_moistureDeviceSetup(){
   int errorCode = CUDA_MOISTURE_SUCCESS;
   int Nelems;

   cudaMemcpyToSymbol(moistureSelector_d, &moistureSelector, sizeof(int));
   if (moistureSelector > 0){
     cudaMemcpyToSymbol(moistureNvars_d, &moistureNvars, sizeof(int));
     cudaMemcpyToSymbol(moistureAdvSelectorQv_d, &moistureAdvSelectorQv, sizeof(int));
     cudaMemcpyToSymbol(moistureAdvSelectorQv_b_d, &moistureAdvSelectorQv_b, sizeof(float));
     cudaMemcpyToSymbol(moistureSGSturb_d, &moistureSGSturb, sizeof(int));
     cudaMemcpyToSymbol(moistureCond_d, &moistureCond, sizeof(int));
     cudaMemcpyToSymbol(moistureAdvSelectorQi_d, &moistureAdvSelectorQi, sizeof(int));
     cudaMemcpyToSymbol(moistureCondTscale_d, &moistureCondTscale, sizeof(float));
     cudaMemcpyToSymbol(moistureCondBasePres_d, &moistureCondBasePres, sizeof(int));
     cudaMemcpyToSymbol(moistureMPcallTscale_d, &moistureMPcallTscale, sizeof(float));

     Nelems = (Nxp+2*Nh)*(Nyp+2*Nh)*(Nzp+2*Nh);
     fecuda_DeviceMalloc(Nelems*moistureNvars*sizeof(float), &moistScalars_d);
     fecuda_DeviceMalloc(Nelems*moistureNvars*sizeof(float), &moistScalarsFrhs_d);
     fecuda_DeviceMalloc(Nelems*moistureNvars*3*sizeof(float), &moistTauFlds_d);
     fecuda_DeviceMalloc(Nelems*sizeof(float), &fcond_d);
   }

   return(errorCode);
} //end cuda_moitureDeviceSetup()

/*----->>>>> extern "C" int cuda_moistureDeviceCleanup();  -----------------------------------------------------------
Used to free all malloced memory by the MOISTURE submodule.
*/

extern "C" int cuda_moistureDeviceCleanup(){
   int errorCode = CUDA_MOISTURE_SUCCESS;

   /* Free any moisture submodule arrays */
   cudaFree(moistScalars_d);
   cudaFree(moistScalarsFrhs_d);
   cudaFree(moistTauFlds_d);
   cudaFree(fcond_d);

   return(errorCode);

}//end cuda_moistureDeviceCleanup()

/*----->>>>> __global__ void  cudaDevice_hydroCoreUnitTestCompleteMP(); ----------------------------------------------
* Global Kernel for calculating/accumulating moisture (microphysics) forcing Frhs terms   
*/
__global__ void cudaDevice_hydroCoreUnitTestCompleteMP(float* hydroFlds_d, float* hydroFldsFrhs_d, float* moistScalars_d,
                                                       float* moistScalarsFrhs_d, float* hydroRhoInv_d, 
                                                       float* hydroPres_d, float* fcond_d, float dt, float* hydroBaseStateFlds_d){ 

   int fldStride;

   fldStride = (Nx_d+2*Nh_d)*(Ny_d+2*Nh_d)*(Nz_d+2*Nh_d);

   cudaDevice_moistZerothOrder(&moistScalars_d[0*fldStride], &moistScalars_d[1*fldStride], &hydroFlds_d[fldStride*THETA_INDX],
                               &hydroPres_d[0], &hydroRhoInv_d[0], &fcond_d[0], dt, &hydroBaseStateFlds_d[fldStride*THETA_INDX_BS]); // calculate fcond
   cudaDevice_moistCondFrhs(&fcond_d[0], &moistScalarsFrhs_d[0*fldStride], &moistScalarsFrhs_d[1*fldStride], &moistScalarsFrhs_d[2*fldStride]); // forcing to qv and ql (phase change)
   cudaDevice_thetaCondFrhs(&hydroPres_d[0], &hydroRhoInv_d[0],
                            &hydroFlds_d[fldStride*THETA_INDX], &fcond_d[0], &hydroFldsFrhs_d[fldStride*THETA_INDX]); // forcing to theta (energy exchange)

} // end cudaDevice_hydroCoreUnitTestCompleteMP

/*----->>>>> __device__ void cudaDevice_moistZerothOrder();  --------------------------------------------------
*/
__device__ void cudaDevice_moistZerothOrder(float* rho_qv, float* rho_ql, float* th, float* press, float* rhoInv, float* fcond, float dt, float* th_base){

  float Td;
  float th_d;
  float p_vs;
  float f_for,f_lim;
  float Tf = 273.15; // K, freezing temperature
  float r_v,r_vs;
  float pr_d;
  float constant_1;
  float t_cond = moistureCondTscale_d; // s, relaxation time
  float dt_mp = moistureMPcallTscale_d; // s, microphysics update time scale

  // Constants from Flatau et al. 1992 (Table 4, absolute norm)
  // Formulation from Morrison 2-moment 
  float a0 = 6.11239921;
  float a1 = 0.443987641;
  float a2 = 0.142986287e-1;
  float a3 = 0.264847430e-3;
  float a4 = 0.302950461e-5;
  float a5 = 0.206739458e-7;
  float a6 = 0.640689451e-10;
  float a7 = -0.952447341e-13;
  float a8 = -0.976195544e-15;
  
  float a0i = 6.11147274;
  float a1i = 0.503160820;
  float a2i = 0.188439774e-1;
  float a3i = 0.420895665e-3;
  float a4i = 0.615021634e-5;
  float a5i = 0.602588177e-7;
  float a6i = 0.385852041e-9;
  float a7i = 0.146898966e-11;
  float a8i = 0.252751365e-14;
  float Tc;

  int i,j,k,ijk;
  int iStride,jStride,kStride;

  i = (blockIdx.x)*blockDim.x + threadIdx.x;
  j = (blockIdx.y)*blockDim.y + threadIdx.y;
  k = (blockIdx.z)*blockDim.z + threadIdx.z;
  iStride = (Ny_d+2*Nh_d)*(Nz_d+2*Nh_d);
  jStride = (Nz_d+2*Nh_d);
  kStride = 1;
  ijk = i*iStride + j*jStride + k*kStride;

  if((i >= iMin_d)&&(i < iMax_d) && (j >= jMin_d)&&(j < jMax_d) && (k >= kMin_d)&&(k < kMax_d)){

    th_d = th[ijk]*rhoInv[ijk];

    constant_1 = R_gas_d/powf( refPressure_d, R_cp_d);
    if (moistureCondBasePres_d==1){
      pr_d = powf(th_base[ijk]*constant_1, cp_cv_d);
    }else{
      pr_d = powf(th[ijk]*constant_1, cp_cv_d);
    }
    Td = th_d*powf(pr_d/refPressure_d,R_cp_d);

    if (Td>=Tf){
      Tc = fmaxf(-80.,(Td-273.16));
      p_vs = (a0 + Tc*(a1+Tc*(a2+Tc*(a3+Tc*(a4+Tc*(a5+Tc*(a6+Tc*(a7+a8*Tc))))))))*100.;
      p_vs = fminf((0.99*pr_d),p_vs); // Do not let p_vs exceed pr_d in case of low pr_d

    }else{
      Tc = fmaxf(-80.,(Td-273.16));
      p_vs = (a0i + Tc*(a1i+Tc*(a2i+Tc*(a3i+Tc*(a4i+Tc*(a5i+Tc*(a6i+Tc*(a7i+a8i*Tc))))))))*100.;
      p_vs = fminf((0.99*pr_d),p_vs); // Do not let p_vs exceed pr_d in case of low pr_d
    }

    // saturation adjustment 0th-order closure
    if (moistureCond_d == 1){
      f_for = rho_qv[ijk]*1e-3 - (p_vs/(R_vapor_d*Td));
    }else if (moistureCond_d == 2){ // Bryan MWR2003
      r_v = rho_qv[ijk]*1e-3;
      r_vs = p_vs/(R_vapor_d*Td);
      f_for = (r_v - r_vs)/(1.0+(powf(L_v_d,2.0)*r_vs*rhoInv[ijk]/(cp_gas_d*R_vapor_d*powf(Td,2.0)))/powf(1.0+r_v*rhoInv[ijk],2.0) );
    }else if (moistureCond_d == 3){
      r_v = rho_qv[ijk]*1e-3;
      r_vs = p_vs/(pr_d-p_vs)/Rv_Rg_d;
      r_vs = r_vs/rhoInv[ijk]; // add rho_dry factor to be consistent with r_v
      f_for = (r_v - r_vs)/(1.0+(powf(L_v_d,2.0)*r_vs*rhoInv[ijk]/(cp_gas_d*R_vapor_d*powf(Td,2.0)))/powf(1.0+r_v*rhoInv[ijk],2.0) );
    }else if (moistureCond_d == 4){
      r_v = rho_qv[ijk]*1e-3;
      r_vs = p_vs/(pr_d-p_vs)/Rv_Rg_d;
      r_vs = r_vs/rhoInv[ijk]; // add rho_dry factor to be consistent with r_v
      f_for = (r_v - r_vs)/( (1.0+(powf(L_v_d,2.0)*r_vs*rhoInv[ijk]/(cp_gas_d*R_vapor_d*powf(Td,2.0)))/powf(1.0+r_v*rhoInv[ijk],2.0) )*dt_mp + t_cond);
      f_for = f_for*t_cond;
    }
    f_lim = rho_ql[ijk]*1e-3;
    fcond[ijk] = fmaxf(f_for,0.0) - fmaxf(fminf(f_lim,-f_for),0.0);
    fcond[ijk] = fcond[ijk]/t_cond;

  }

} //end cudaDevice_moistZerothOrder

/*----->>>>> __device__ void cudaDevice_moistCondFrhs();  --------------------------------------------------
*/
__device__ void cudaDevice_moistCondFrhs(float* fcond, float* qv_Frhs, float* ql_Frhs, float* qr_Frhs){

  int i,j,k,ijk;
  int iStride,jStride,kStride;

  i = (blockIdx.x)*blockDim.x + threadIdx.x;
  j = (blockIdx.y)*blockDim.y + threadIdx.y;
  k = (blockIdx.z)*blockDim.z + threadIdx.z;
  iStride = (Ny_d+2*Nh_d)*(Nz_d+2*Nh_d);
  jStride = (Nz_d+2*Nh_d);
  kStride = 1;
  ijk = i*iStride + j*jStride + k*kStride;

  if((i >= iMin_d)&&(i < iMax_d) && (j >= jMin_d)&&(j < jMax_d) && (k >= kMin_d)&&(k < kMax_d)){
    qv_Frhs[ijk] = qv_Frhs[ijk] - fcond[ijk]*1e3;
    ql_Frhs[ijk] = ql_Frhs[ijk] + fcond[ijk]*1e3;
  }

} //end cudaDevice_moistCondFrhs

/*----->>>>> __device__ void cudaDevice_thetaCondFrhs();  --------------------------------------------------
*/
__device__ void cudaDevice_thetaCondFrhs(float* press, float* rhoInv, float* th, float* fcond, float* th_Frhs){

  float Td;
  float th_d;
  float pr_d;
  float constant_1;

  int i,j,k,ijk;
  int iStride,jStride,kStride;

  /*Establish necessary indices for spatial locality*/
  i = (blockIdx.x)*blockDim.x + threadIdx.x;
  j = (blockIdx.y)*blockDim.y + threadIdx.y;
  k = (blockIdx.z)*blockDim.z + threadIdx.z;
  iStride = (Ny_d+2*Nh_d)*(Nz_d+2*Nh_d);
  jStride = (Nz_d+2*Nh_d);
  kStride = 1;
  ijk = i*iStride + j*jStride + k*kStride;

  constant_1 = R_gas_d/powf( refPressure_d, R_cp_d);

  if((i >= iMin_d)&&(i < iMax_d) && (j >= jMin_d)&&(j < jMax_d) && (k >= kMin_d)&&(k < kMax_d)){

    th_d = th[ijk]*rhoInv[ijk];
    pr_d = powf(th[ijk]*constant_1, cp_cv_d); 
    Td = th_d*powf(pr_d/refPressure_d,R_cp_d);

    th_Frhs[ijk] = th_Frhs[ijk] + ((th_d*L_v_d)/(Td*cp_gas_d))*fcond[ijk];

  }

} //end cudaDevice_thetaCondFrhs
