/* FastEddy®: SRC/HYDRO_CORE/CUDA/cuda_pressureDevice.cu 
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
/*---PRESSURE_GRADIENT_FORCE*/
__constant__ int pgfSelector_d;          /*Pressure Gradient Force (pgf) selector: 0=off, 1=on*/
float *hydroPres_d;            /*Base Adress of memory containing the diagnostic perturbation pressure field */

/*#################------------ PRESSURE submodule function definitions ------------------#############*/
/*----->>>>> int cuda_pressureDeviceSetup();       ---------------------------------------------------------
 * Used to cudaMalloc and cudaMemcpy parameters and coordinate arrays, and for the PRESSURE_CUDA submodule.
*/
extern "C" int cuda_pressureDeviceSetup(){
   int errorCode = CUDA_PRESSURE_SUCCESS;
   int Nelems;

   //Copy the pgfSelector constant to device constant-memory
   cudaMemcpyToSymbol(pgfSelector_d, &pgfSelector, sizeof(int));

   /*Set the full memory block number of elements for hydroCore fields*/
   Nelems = (Nxp+2*Nh)*(Nyp+2*Nh)*(Nzp+2*Nh);
   fecuda_DeviceMalloc(Nelems*sizeof(float), &hydroPres_d);

   return(errorCode);
} //end cuda_pressureDeviceSetup()

/*----->>>>> extern "C" int cuda_pressureDeviceCleanup();  -----------------------------------------------------------
Used to free all malloced memory by the PRESSURE_CUDA submodule.
*/
extern "C" int cuda_pressureDeviceCleanup(){
   int errorCode = CUDA_PRESSURE_SUCCESS;

   cudaFree(hydroPres_d);

   return(errorCode);

}//end cuda_pressureDeviceCleanup()

/*----->>>>> __device__ void  cudaDevice_calcPerturbationPressure();  ----------------------------------------------
* This is the cuda version of the calcPerturbationPressure routine from the HYDRO_CORE module
*/
__device__ void cudaDevice_calcPerturbationPressure(float* pres, float* rhoTheta, float* rhoTheta_BS, float* zPos_d){
  float constant_1;
  int i,j,k,ijk,iStride,jStride,kStride;
  i = (blockIdx.x)*blockDim.x + threadIdx.x;
  j = (blockIdx.y)*blockDim.y + threadIdx.y;
  k = (blockIdx.z)*blockDim.z + threadIdx.z;
  iStride = (Ny_d+2*Nh_d)*(Nz_d+2*Nh_d);
  jStride = (Nz_d+2*Nh_d);
  kStride = 1;
  constant_1 = R_gas_d/powf( refPressure_d, R_cp_d);
  ijk = i*iStride + j*jStride + k*kStride;

  float pkp2;
  float pkp1;
  float zk,zkT,zkTp1;
  int ijkTarg,ijkTargp1;
  if((i >= iMin_d-Nh_d)&&(i < iMax_d+Nh_d) &&
     (j >= jMin_d-Nh_d)&&(j < jMax_d+Nh_d) ){
   if((k>0)&&(k<kMin_d)){
    ijkTarg = i*iStride + j*jStride + kMin_d*kStride;
    ijkTargp1 = i*iStride + j*jStride + (kMin_d+1)*kStride;
    zk = zPos_d[ijk];
    zkT = zPos_d[ijkTarg];
    zkTp1 = zPos_d[ijkTargp1];
    pkp1 = powf((rhoTheta[ijkTarg])*constant_1, cp_cv_d) - powf((rhoTheta_BS[ijkTarg])*constant_1, cp_cv_d);
    pkp2 = powf((rhoTheta[ijkTargp1])*constant_1, cp_cv_d) - powf((rhoTheta_BS[ijkTargp1])*constant_1, cp_cv_d);
    pres[ijk] = (zk-zkT)*(pkp2-pkp1)/(zkTp1-zkT)+pkp1;
   }else if((k>=kMax_d)&&(k<kMax_d+Nh_d)){
    ijkTarg = i*iStride + j*jStride + (kMax_d-1)*kStride;
    ijkTargp1 = i*iStride + j*jStride + (kMax_d-2)*kStride;
    zk = zPos_d[ijk];
    zkT = zPos_d[ijkTarg];
    zkTp1 = zPos_d[ijkTargp1];
    pkp1 = powf((rhoTheta[ijkTarg])*constant_1, cp_cv_d) - powf((rhoTheta_BS[ijkTarg])*constant_1, cp_cv_d);
    pkp2 = powf((rhoTheta[ijkTargp1])*constant_1, cp_cv_d) - powf((rhoTheta_BS[ijkTargp1])*constant_1, cp_cv_d);
    pres[ijk] = (zk-zkT)*(pkp1-pkp2)/(zkT-zkTp1)+pkp1;
   }else{
    pres[ijk] = powf((rhoTheta[ijk])*constant_1, cp_cv_d) - powf((rhoTheta_BS[ijk])*constant_1, cp_cv_d);
   }//endif k < kMin_d elseif >=kMax_d else...
  }//endif i&&j...

} // end cudaDevice_calcPerturbationPressure()

/*----->>>>> __device__ void  cudaDevice_calcPerturbationPressureMoist();  ------------------------------------------
*/ 
__device__ void cudaDevice_calcPerturbationPressureMoist(float* pres, float* rho, float* rhoTheta, float* rhoTheta_BS, float* moist_qv, float* zPos_d){
  float constant_1;
  int i,j,k,ijk,iStride,jStride,kStride;
  i = (blockIdx.x)*blockDim.x + threadIdx.x;
  j = (blockIdx.y)*blockDim.y + threadIdx.y;
  k = (blockIdx.z)*blockDim.z + threadIdx.z;
  iStride = (Ny_d+2*Nh_d)*(Nz_d+2*Nh_d);
  jStride = (Nz_d+2*Nh_d);
  kStride = 1;

  constant_1 = R_gas_d/powf( refPressure_d, R_cp_d);
  ijk = i*iStride + j*jStride + k*kStride;

  float pkp2;
  float pkp1;
  float zk,zkT,zkTp1;
  int ijkTarg,ijkTargp1;
  float rhothm_ijk,rhothm_ijkTarg,rhothm_ijkTargp1;

  if((i >= iMin_d-Nh_d)&&(i < iMax_d+Nh_d) &&
     (j >= jMin_d-Nh_d)&&(j < jMax_d+Nh_d) ){
   if((k>0)&&(k<kMin_d)){
    ijkTarg = i*iStride + j*jStride + kMin_d*kStride;
    ijkTargp1 = i*iStride + j*jStride + (kMin_d+1)*kStride;
    zk = zPos_d[ijk];
    zkT = zPos_d[ijkTarg];
    zkTp1 = zPos_d[ijkTargp1];
    rhothm_ijkTarg = rhoTheta[ijkTarg]*(1.0 + Rv_Rg_d*moist_qv[ijkTarg]/rho[ijkTarg]*1e-3); // *1e-3 to convert from g/kg to kg/kg
    rhothm_ijkTargp1 = rhoTheta[ijkTargp1]*(1.0 + Rv_Rg_d*moist_qv[ijkTargp1]/rho[ijkTargp1]*1e-3); // *1e-3 to convert from g/kg to kg/kg
    pkp1 = powf(rhothm_ijkTarg*constant_1, cp_cv_d) - powf((rhoTheta_BS[ijkTarg])*constant_1, cp_cv_d);
    pkp2 = powf(rhothm_ijkTargp1*constant_1, cp_cv_d) - powf((rhoTheta_BS[ijkTargp1])*constant_1, cp_cv_d);
    pres[ijk] = (zk-zkT)*(pkp2-pkp1)/(zkTp1-zkT)+pkp1;
   }else if((k>=kMax_d)&&(k<kMax_d+Nh_d)){
    ijkTarg = i*iStride + j*jStride + (kMax_d-1)*kStride;
    ijkTargp1 = i*iStride + j*jStride + (kMax_d-2)*kStride;
    zk = zPos_d[ijk];
    zkT = zPos_d[ijkTarg];
    zkTp1 = zPos_d[ijkTargp1];
    rhothm_ijkTarg = rhoTheta[ijkTarg]*(1.0 + Rv_Rg_d*moist_qv[ijkTarg]/rho[ijkTarg]*1e-3); // *1e-3 to convert from g/kg to kg/kg
    rhothm_ijkTargp1 = rhoTheta[ijkTargp1]*(1.0 + Rv_Rg_d*moist_qv[ijkTargp1]/rho[ijkTargp1]*1e-3); // *1e-3 to convert from g/kg to kg/kg
    pkp1 = powf(rhothm_ijkTarg*constant_1, cp_cv_d) - powf((rhoTheta_BS[ijkTarg])*constant_1, cp_cv_d);
    pkp2 = powf(rhothm_ijkTargp1*constant_1, cp_cv_d) - powf((rhoTheta_BS[ijkTargp1])*constant_1, cp_cv_d);
    pres[ijk] = (zk-zkT)*(pkp1-pkp2)/(zkT-zkTp1)+pkp1;
   }else{
    rhothm_ijk = rhoTheta[ijk]*(1.0 + Rv_Rg_d*moist_qv[ijk]/rho[ijk]*1e-3);
    pres[ijk] = powf(rhothm_ijk*constant_1, cp_cv_d) - powf((rhoTheta_BS[ijk])*constant_1, cp_cv_d);
   }//endif k < kMin_d elseif >=kMax_d else...
  }//endif i&&j...

} // end cudaDevice_calcPerturbationPressureMoist()

/*----->>>>> __device__ void  cudaDevice_calcPressureGradientForce();  ---------------------------------------------
* This is the cuda version of the calcPressureGradientForce routine from the HYDRO_CORE module
*/
__device__ void cudaDevice_calcPressureGradientForce(float* Frhs_u, float* Frhs_v, float* Frhs_w, float* pres,
                                                     float* J31_d, float* J32_d, float* J33_d){
  int i,j,k,ijk,iStride,jStride,kStride;
  int ip1jk,im1jk,ijp1k,ijm1k,ijkp1,ijkm1;

  i = (blockIdx.x)*blockDim.x + threadIdx.x;
  j = (blockIdx.y)*blockDim.y + threadIdx.y;
  k = (blockIdx.z)*blockDim.z + threadIdx.z;

  if((i >= iMin_d)&&(i < iMax_d) && 
     (j >= jMin_d)&&(j < jMax_d) && 
     (k >= kMin_d)&&(k < kMax_d)){
   
    iStride = (Ny_d+2*Nh_d)*(Nz_d+2*Nh_d);
    jStride = (Nz_d+2*Nh_d);
    kStride = 1;

    ijk = i*iStride + j*jStride + k*kStride;
    ip1jk = (i+1)*iStride + j*jStride + k*kStride;
    ijp1k = i*iStride + (j+1)*jStride + k*kStride;
    ijkp1 = i*iStride + j*jStride + (k+1)*kStride;
    im1jk = (i-1)*iStride + j*jStride + k*kStride;
    ijm1k = i*iStride + (j-1)*jStride + k*kStride;
    ijkm1 = i*iStride + j*jStride + (k-1)*kStride;

    Frhs_u[ijk] = Frhs_u[ijk]-0.5*dXi_d*(pres[ip1jk] - pres[im1jk]);
    Frhs_v[ijk] = Frhs_v[ijk]-0.5*dYi_d*(pres[ijp1k] - pres[ijm1k]);
    Frhs_w[ijk] = Frhs_w[ijk]-0.5*( dXi_d*J31_d[ijk]*(pres[ip1jk] - pres[im1jk])
                                   +dYi_d*J32_d[ijk]*(pres[ijp1k] - pres[ijm1k])
                                   +dZi_d*J33_d[ijk]*(pres[ijkp1] - pres[ijkm1]) );
  }//end if in the range of non-halo cells 
} // end cudaDevice_calcPressureGradientForce()

/*----->>>>> __device__ void  cudaDevice_calcPressureGradientForceMoist();  -----------------------------------------
*/
__device__ void cudaDevice_calcPressureGradientForceMoist(float* Frhs_u, float* Frhs_v, float* Frhs_w, float* rho,
                                                          float* pres, float* moistScalars,
                                                          float* J31_d, float* J32_d, float* J33_d){

  int i,j,k,ijk,iStride,jStride,kStride,fldStride;
  int ip1jk,im1jk,ijp1k,ijm1k,ijkp1,ijkm1;
  float rhomd_ijk,rhodm_ijk;
  int iFld;

  i = (blockIdx.x)*blockDim.x + threadIdx.x;
  j = (blockIdx.y)*blockDim.y + threadIdx.y;
  k = (blockIdx.z)*blockDim.z + threadIdx.z;
  iStride = (Ny_d+2*Nh_d)*(Nz_d+2*Nh_d);
  jStride = (Nz_d+2*Nh_d);
  kStride = 1;

  fldStride = (Nx_d+2*Nh_d)*(Ny_d+2*Nh_d)*(Nz_d+2*Nh_d);
  ijk = i*iStride + j*jStride + k*kStride;
  ip1jk = (i+1)*iStride + j*jStride + k*kStride;
  ijp1k = i*iStride + (j+1)*jStride + k*kStride;
  ijkp1 = i*iStride + j*jStride + (k+1)*kStride;
  im1jk = (i-1)*iStride + j*jStride + k*kStride;
  ijm1k = i*iStride + (j-1)*jStride + k*kStride;
  ijkm1 = i*iStride + j*jStride + (k-1)*kStride;

  if((i >= iMin_d)&&(i < iMax_d) && 
     (j >= jMin_d)&&(j < jMax_d) && 
     (k >= kMin_d)&&(k < kMax_d)){

    rhomd_ijk = 1.0;
    for (iFld=0; iFld < moistureNvars_d; iFld++){
       rhomd_ijk = rhomd_ijk + moistScalars[fldStride*iFld+ijk]/rho[ijk]*1e-3; // *1e-3 to convert from g/kg to kg/kg
    }
    rhodm_ijk = 1.0/rhomd_ijk;

    Frhs_u[ijk] = Frhs_u[ijk] -rhodm_ijk*0.5*dXi_d*(pres[ip1jk] - pres[im1jk]);
    Frhs_v[ijk] = Frhs_v[ijk] -rhodm_ijk*0.5*dYi_d*(pres[ijp1k] - pres[ijm1k]);
    Frhs_w[ijk] = Frhs_w[ijk] -rhodm_ijk*0.5*( dXi_d*J31_d[ijk]*(pres[ip1jk] - pres[im1jk])
                                              +dYi_d*J32_d[ijk]*(pres[ijp1k] - pres[ijm1k])
                                              +dZi_d*J33_d[ijk]*(pres[ijkp1] - pres[ijkm1]) );
  }//end if in the range of non-halo cells 
} // end cudaDevice_calcPressureGradientForceMoist()
