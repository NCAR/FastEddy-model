/* FastEddy®: SRC/HYDRO_CORE/CUDA/cuda_sgstkeDevice.cu 
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
//#define DEBUG_TKE //for verbose logging of exemplar (imin,jmin) profiles 

/*Advection selectors 1 or 2-eq TKE */
__constant__ int TKEAdvSelector_d;     /* SGSTKE advection scheme selector */
__constant__ float TKEAdvSelector_b_hyb_d; /* hybrid advection scheme selector */

/*---PROGNOSTIC SGSTKE EQUATION*/ 
float* sgstkeScalars_d;  /*Base address for SGSTKE field arrays*/
float* sgstkeScalarsFrhs_d;  /*Base address for SGSTKE forcing field arrays*/
float* sgstke_ls_d;  /*Base address for SGSTKE length scale field arrays*/
float* dedxi_d; /*Base address for d(SGSTKE)/dxi field arrays*/

/*#################------------ SGSTKE submodule function definitions ------------------#############*/
/*----->>>>> int cuda_sgstkeDeviceSetup();       ---------------------------------------------------------
 * Used to cudaMalloc and cudaMemcpy parameters and coordinate arrays, and for the SGSTKE_CUDA submodule.
*/
extern "C" int cuda_sgstkeDeviceSetup(){
   int errorCode = CUDA_SGSTKE_SUCCESS;
   int Nelems;

   Nelems = (Nxp+2*Nh)*(Nyp+2*Nh)*(Nzp+2*Nh);
   fecuda_DeviceMalloc(Nelems*TKESelector*sizeof(float), &sgstkeScalars_d);
   fecuda_DeviceMalloc(Nelems*TKESelector*sizeof(float), &sgstkeScalarsFrhs_d);
   fecuda_DeviceMalloc(Nelems*TKESelector*sizeof(float), &sgstke_ls_d);
   fecuda_DeviceMalloc(Nelems*TKESelector*3*sizeof(float), &dedxi_d);

   cudaMemcpyToSymbol(TKEAdvSelector_d, &TKEAdvSelector, sizeof(int));
   cudaMemcpyToSymbol(TKEAdvSelector_b_hyb_d, &TKEAdvSelector_b_hyb, sizeof(float));

   return(errorCode);
} //end cuda_sgstkeDeviceSetup()

/*----->>>>> extern "C" int cuda_sgstkeDeviceCleanup();  -----------------------------------------------------------
Used to free all malloced memory by the SGSTKE submodule.
*/

extern "C" int cuda_sgstkeDeviceCleanup(){
   int errorCode = CUDA_SGSTKE_SUCCESS;

   /* Free any SGSTKE submodule arrays */
   cudaFree(sgstkeScalars_d);
   cudaFree(sgstkeScalarsFrhs_d);
   cudaFree(sgstke_ls_d);
   cudaFree(dedxi_d);

   return(errorCode);

}//end cuda_sgstkeDeviceCleanup()

/*----->>>>> __device__ void  cudaDevice_hydroCoreUnitTestCompleteSGSTKE();  ----------------------------------------
* Global Kernel for calculating/accumulating SGSTKE Frhs     
*/
__global__ void cudaDevice_hydroCoreUnitTestCompleteSGSTKE(float* hydroFlds_d, float* hydroRhoInv_d, float* hydroTauFlds_d,
                                                           float* hydroKappaM_d, float* dedxi_d, float* sgstke_ls_d,
                                                           float* sgstkeScalars_d, float* sgstkeScalarsFrhs_d, float* canopy_lad_d,
                                                           float* J31_d, float* J32_d, float* J33_d, float* D_Jac_d){ 

   int fldStride,iFld;
   fldStride = (Nx_d+2*Nh_d)*(Ny_d+2*Nh_d)*(Nz_d+2*Nh_d);

   for(iFld=0; iFld < TKESelector_d; iFld++){ // loop over SGSTKE equations
      if(iFld==0){ // 1-eq SGSTKE
        cudaDevice_sgstkeBuoyancy(&hydroFlds_d[fldStride*THETA_INDX], &hydroRhoInv_d[0],
                                  &hydroTauFlds_d[fldStride*8], &sgstkeScalarsFrhs_d[fldStride*iFld]); // buoyancy source/sink term
        cudaDevice_sgstkeDissip(&sgstkeScalars_d[fldStride*iFld], &hydroRhoInv_d[0], &sgstke_ls_d[fldStride*iFld],
                                &sgstkeScalarsFrhs_d[fldStride*iFld], 1, D_Jac_d); // dissipation term
        cudaDevice_sgstkeShearProd(&hydroTauFlds_d[fldStride*0], &hydroTauFlds_d[fldStride*1], &hydroTauFlds_d[fldStride*2],
                                   &hydroTauFlds_d[fldStride*4], &hydroTauFlds_d[fldStride*3], &hydroTauFlds_d[fldStride*5],
                                   &hydroFlds_d[fldStride*U_INDX], &hydroFlds_d[fldStride*V_INDX], &hydroFlds_d[fldStride*W_INDX],
                                   &hydroRhoInv_d[0], &sgstkeScalarsFrhs_d[fldStride*iFld],
                                   J31_d, J32_d, J33_d); // shear production term
        cudaDevice_sgstkeTurbTransport(&hydroKappaM_d[0], &dedxi_d[fldStride*(iFld*3+0)], &dedxi_d[fldStride*(iFld*3+1)],
                                       &dedxi_d[fldStride*(iFld*3+2)], &hydroFlds_d[fldStride*RHO_INDX],
                                       &sgstkeScalarsFrhs_d[fldStride*iFld],
                                       J31_d, J32_d, J33_d); // turbulent transport term
         if(canopySelector_d==1){ // 1-eq SGSTKE with canopy model
            cudaDevice_canopySGSTKEtransfer(&hydroRhoInv_d[0], &hydroFlds_d[fldStride*U_INDX], &hydroFlds_d[fldStride*V_INDX],
                                            &hydroFlds_d[fldStride*W_INDX], &canopy_lad_d[0],
                                            &sgstkeScalars_d[fldStride*iFld], &sgstkeScalarsFrhs_d[fldStride*iFld], -1.0); // transfer to wake scale
         }
      }else if((iFld==1)&&(canopySelector_d==1)&&(TKESelector_d==2)){ // 2-eq SGSTKE with canopy model (wake scale SGSTKE)
	cudaDevice_sgstkeTurbTransport(&hydroKappaM_d[0], &dedxi_d[fldStride*(iFld*3+0)], &dedxi_d[fldStride*(iFld*3+1)],
                                       &dedxi_d[fldStride*(iFld*3+2)], &hydroFlds_d[fldStride*RHO_INDX],
                                       &sgstkeScalarsFrhs_d[fldStride*iFld],
                                       J31_d, J32_d, J33_d); // turbulent transport term
        cudaDevice_sgstkeDissip(&sgstkeScalars_d[fldStride*iFld], &hydroRhoInv_d[0], &sgstke_ls_d[fldStride*iFld],
                                &sgstkeScalarsFrhs_d[fldStride*iFld], 0, D_Jac_d); // dissipation term
        cudaDevice_canopySGSTKEtransfer(&hydroRhoInv_d[0], &hydroFlds_d[fldStride*U_INDX], &hydroFlds_d[fldStride*V_INDX],
                                        &hydroFlds_d[fldStride*W_INDX], &canopy_lad_d[0],
                                        &sgstkeScalars_d[0], &sgstkeScalarsFrhs_d[fldStride*iFld], 1.0); // transfer from grid scale
        cudaDevice_canopySGSTKEwakeprod(&hydroRhoInv_d[0], &hydroFlds_d[fldStride*U_INDX], &hydroFlds_d[fldStride*V_INDX],
                                        &hydroFlds_d[fldStride*W_INDX], &canopy_lad_d[0], &sgstkeScalarsFrhs_d[fldStride*iFld]); // wake production
      }
   }

} // end cudaDevice_hydroCoreUnitTestCompleteSGSTKE()

/*----->>>>> __device__ void  cudaDevice_sgstkeLengthScale();  --------------------------------------------------
*/
__device__ void cudaDevice_sgstkeLengthScale(float* th, float* rhoInv, float* sgstke, float* sgstke_ls, float* J31_d, float* J32_d, float* J33_d, float* D_Jac_d){

  float e_ijk;
  float tke_lenscale;
  float Nsquared;
  float dthdz,term_b,len_1,len_2;
  float tke_len_min = 1e-2; // minimum allowed sub-grid length scale

  int i,j,k,ijk,im1jk,ijm1k,ijkm1,ip1jk,ijp1k,ijkp1;
  int iStride,jStride,kStride;

  i = (blockIdx.x)*blockDim.x + threadIdx.x;
  j = (blockIdx.y)*blockDim.y + threadIdx.y;
  k = (blockIdx.z)*blockDim.z + threadIdx.z;
  iStride = (Ny_d+2*Nh_d)*(Nz_d+2*Nh_d);
  jStride = (Nz_d+2*Nh_d);
  kStride = 1;
  ijk = i*iStride + j*jStride + k*kStride;
  im1jk = (i-1)*iStride + j*jStride + k*kStride;
  ip1jk = (i+1)*iStride + j*jStride + k*kStride;
  ijm1k = i*iStride + (j-1)*jStride + k*kStride;
  ijp1k = i*iStride + (j+1)*jStride + k*kStride;
  ijkm1 = i*iStride + j*jStride + (k-1)*kStride;
  ijkp1 = i*iStride + j*jStride + (k+1)*kStride;

  if((i >= iMin_d-1)&&(i < iMax_d+1) && (j >= jMin_d-1)&&(j < jMax_d+1) && (k >= kMin_d-1)&&(k < kMax_d+1)){

    dthdz = 0.5*((J31_d[ijk]*dXi_d*(th[ip1jk]*rhoInv[ip1jk]-th[im1jk]*rhoInv[im1jk]))
                +(J32_d[ijk]*dYi_d*(th[ijp1k]*rhoInv[ijp1k]-th[ijm1k]*rhoInv[ijm1k])) 
                +(J33_d[ijk]*dZi_d*(th[ijkp1]*rhoInv[ijkp1]-th[ijkm1]*rhoInv[ijkm1])) 
                );

    Nsquared = (accel_g_d/(th[ijk]*rhoInv[ijk]))*dthdz; 
    len_2 = powf(dX_d*dY_d*dZ_d*D_Jac_d[ijk],1.0/3.0);
    if(Nsquared > 0.0){
      term_b = powf(Nsquared,-0.5); // ensured (sqrt(Nsquared > 0))^-1 so safe. 
      e_ijk = sgstke[ijk]*rhoInv[ijk];
      len_1 = 0.76*powf(e_ijk,0.5)*term_b;
      tke_lenscale = fmaxf(fminf(len_1,len_2),tke_len_min); // limits the length scale to avoid dissipation term becoming infinity
    }else{
      tke_lenscale = len_2; // else not stable conditions so grid resolution is the appropriate length scale
    }

    sgstke_ls[ijk] = tke_lenscale;

    
#if DEBUG_TKE 
    if (i==iMin_d && j==jMin_d){
      printf("cudaDevice_sgstkeLengthScale(): At (%d,%d,%d) sgstke_ls,len_1,len_2,e_ijk,term_b,dthdz,th_ijk = %f,%f,%f,%12.10f,%f,%f,%f\n",i,j,k,sgstke_ls[ijk],len_1,len_2,e_ijk,term_b,dthdz,th[ijk]*rhoInv[ijk]);
      //printf("cudaDevice_sgstkeLengthScale(): At (%d,%d,%d) dthdz, th = %f,%f\n",i,j,k,dthdz,th[ijk]*rhoInv[ijk]);
    }
#endif    

  } // if (within the computational domain...) 

} //end cudaDevice_sgstkeLengthScale

/*----->>>>> __device__ void  cudaDevice_sgstkeBuoyancy();  --------------------------------------------------
*/
__device__ void cudaDevice_sgstkeBuoyancy(float* th, float* rhoInv, float* STH3, float* Frhs_sgstke){

  float tau_th3_ijk;
  float f_sgstke_buoy;

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

    tau_th3_ijk = STH3[ijk];

    f_sgstke_buoy = accel_g_d/(th[ijk]*rhoInv[ijk])*tau_th3_ijk;

    Frhs_sgstke[ijk] = Frhs_sgstke[ijk] + f_sgstke_buoy;

#if DEBUG_TKE
    if (i==iMin_d && j==jMin_d){
      printf("cudaDevice_sgstkeBuoyancy(): At (%d,%d,%d) tau_th3_ijk, f_sgstke_buoy = %f,%f\n",i,j,k,tau_th3_ijk,f_sgstke_buoy);
    }
#endif

  }

} //end cudaDevice_sgstkeBuoyancy

/*----->>>>> __device__ void  cudaDevice_sgstkeDissip();  --------------------------------------------------
*/
__device__ void cudaDevice_sgstkeDissip(float* sgstke, float* rhoInv, float* sgstke_ls, float* Frhs_sgstke, int l_corr_ce, float* D_Jac_d){

  float c_e;
  float delta_ijk;
  float e_ijk;
  float f_sgstke_diss;

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

    delta_ijk = powf(dX_d*dY_d*dZ_d*D_Jac_d[ijk],1.0/3.0); 

    if (l_corr_ce==1){ // apply correction to c_e based on length scale
      c_e = 1.9*c_k_d + (0.93-1.9*c_k_d)*sgstke_ls[ijk]/delta_ijk;
    } else{ // do not apply correction to c_e
      c_e = 0.93;
    }

    e_ijk = sgstke[ijk]*rhoInv[ijk];

    f_sgstke_diss = -c_e*sgstke[ijk]*powf(e_ijk,0.5)/sgstke_ls[ijk];

    Frhs_sgstke[ijk] = Frhs_sgstke[ijk] + f_sgstke_diss;

#if DEBUG_TKE
    if (i==iMin_d && j==jMin_d){
      printf("cudaDevice_sgstkeDissip(): At (%d,%d,%d) c_e, sgstke_ls, f_sgstke_diss = %f,%f,%f\n",i,j,k,c_e,sgstke_ls[ijk],f_sgstke_diss);
    }
#endif

  }

} //end cudaDevice_sgstkeDissip

/*----->>>>> __device__ void  cudaDevice_sgstkeShearProd();  --------------------------------------------------
*/
__device__ void cudaDevice_sgstkeShearProd(float* tau_11, float* tau_12, float* tau_13, float* tau_22, float* tau_23, float* tau_33, 
                                           float* u, float* v, float* w, float* rhoInv, float* Frhs_sgstke,
                                           float* J31_d, float* J32_d, float* J33_d){

  float term_11,term_12,term_13,term_21,term_22,term_23,term_31,term_32,term_33;
  float f_sgstke_shear;

  int i,j,k,ijk;
  int im1jk,ijm1k,ijkm1;
  int ip1jk,ijp1k,ijkp1;
  int iStride,jStride,kStride;

  i = (blockIdx.x)*blockDim.x + threadIdx.x;
  j = (blockIdx.y)*blockDim.y + threadIdx.y;
  k = (blockIdx.z)*blockDim.z + threadIdx.z;
  iStride = (Ny_d+2*Nh_d)*(Nz_d+2*Nh_d);
  jStride = (Nz_d+2*Nh_d);
  kStride = 1;
  ijk = i*iStride + j*jStride + k*kStride;
  im1jk = (i-1)*iStride + j*jStride + k*kStride;
  ijm1k = i*iStride + (j-1)*jStride + k*kStride;
  ijkm1 = i*iStride + j*jStride + (k-1)*kStride;
  ip1jk = (i+1)*iStride + j*jStride + k*kStride;
  ijp1k = i*iStride + (j+1)*jStride + k*kStride;
  ijkp1 = i*iStride + j*jStride + (k+1)*kStride;

  if((i >= iMin_d)&&(i < iMax_d) && (j >= jMin_d)&&(j < jMax_d) && (k >= kMin_d)&&(k < kMax_d)){

    // term_11 = -tau_11*du/dx
    term_11 = -tau_11[ijk]*0.5*(dXi_d*(u[ip1jk]*rhoInv[ip1jk]-u[im1jk]*rhoInv[im1jk]));
    // term_12 = -tau_12*du/dy
    term_12 = -tau_12[ijk]*0.5*(dYi_d*(u[ijp1k]*rhoInv[ijp1k]-u[ijm1k]*rhoInv[ijm1k]));
    // term_13 = -tau_13*du/dz
    term_13 = -tau_13[ijk]*0.5*(J31_d[ijk]*dXi_d*(u[ip1jk]*rhoInv[ip1jk]-u[im1jk]*rhoInv[im1jk])
                               +J32_d[ijk]*dYi_d*(u[ijp1k]*rhoInv[ijp1k]-u[ijm1k]*rhoInv[ijm1k])
                               +J33_d[ijk]*dZi_d*(u[ijkp1]*rhoInv[ijkp1]-u[ijkm1]*rhoInv[ijkm1]));

    // term_21 = -tau_21*dv/dx
    term_21 = -tau_12[ijk]*0.5*(dXi_d*(v[ip1jk]*rhoInv[ip1jk]-v[im1jk]*rhoInv[im1jk]));
    // term_22 = -tau_22*dv/dy
    term_22 = -tau_22[ijk]*0.5*(dYi_d*(v[ijp1k]*rhoInv[ijp1k]-v[ijm1k]*rhoInv[ijm1k]));
    // term_23 = -tau_23*dv/dz
    term_23 = -tau_23[ijk]*0.5*(J31_d[ijk]*dXi_d*(v[ip1jk]*rhoInv[ip1jk]-v[im1jk]*rhoInv[im1jk])
                               +J32_d[ijk]*dYi_d*(v[ijp1k]*rhoInv[ijp1k]-v[ijm1k]*rhoInv[ijm1k])
                               +J33_d[ijk]*dZi_d*(v[ijkp1]*rhoInv[ijkp1]-v[ijkm1]*rhoInv[ijkm1]));

    // term_31 = -tau_31*dw/dx
    term_31 = -tau_13[ijk]*0.5*(dXi_d*(w[ip1jk]*rhoInv[ip1jk]-w[im1jk]*rhoInv[im1jk]));
    // term_32 = -tau_32*dw/dy
    term_32 = -tau_23[ijk]*0.5*(dYi_d*(w[ijp1k]*rhoInv[ijp1k]-w[ijm1k]*rhoInv[ijm1k]));
    // term_33 = -tau_33*dw/dz
    term_33 = -tau_33[ijk]*0.5*(J31_d[ijk]*dXi_d*(w[ip1jk]*rhoInv[ip1jk]-w[im1jk]*rhoInv[im1jk])
                               +J32_d[ijk]*dYi_d*(w[ijp1k]*rhoInv[ijp1k]-w[ijm1k]*rhoInv[ijm1k])
                               +J33_d[ijk]*dZi_d*(w[ijkp1]*rhoInv[ijkp1]-w[ijkm1]*rhoInv[ijkm1]));


    f_sgstke_shear = (term_11 + term_12 + term_13 + term_21 + term_22 + term_23 + term_31 + term_32 + term_33);
    Frhs_sgstke[ijk] = Frhs_sgstke[ijk] + f_sgstke_shear;

#if DEBUG_TKE
    if (i==iMin_d && j==jMin_d){
      printf("cudaDevice_sgstkeShearProd(): At (%d,%d,%d) term_11,term_12,term_13,term_21,term_22,term_23,term_31,term_32,term_33,f_sgstke_shear = %f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n",
             i,j,k,term_11,term_12,term_13,term_21,term_22,term_23,term_31,term_32,term_33,f_sgstke_shear);
    }
#endif

  }

} //end cudaDevice_sgstkeShearProd

/*----->>>>> __device__ void  cudaDevice_GradScalar();  --------------------------------------------------
*/
__device__ void cudaDevice_GradScalar(float* scalar, float* rhoInv, float* dedx, float* dedy, float* dedz,
                                      float* J31_d, float* J32_d, float* J33_d){

  int i,j,k,ijk;
  int im1jk,ijm1k,ijkm1;
  int ip1jk,ijp1k,ijkp1;
  int iStride,jStride,kStride;

  i = (blockIdx.x)*blockDim.x + threadIdx.x;
  j = (blockIdx.y)*blockDim.y + threadIdx.y;
  k = (blockIdx.z)*blockDim.z + threadIdx.z;
  iStride = (Ny_d+2*Nh_d)*(Nz_d+2*Nh_d);
  jStride = (Nz_d+2*Nh_d);
  kStride = 1;
  ijk = i*iStride + j*jStride + k*kStride;
  im1jk = (i-1)*iStride + j*jStride + k*kStride;
  ijm1k = i*iStride + (j-1)*jStride + k*kStride;
  ijkm1 = i*iStride + j*jStride + (k-1)*kStride;
  ip1jk = (i+1)*iStride + j*jStride + k*kStride;
  ijp1k = i*iStride + (j+1)*jStride + k*kStride;
  ijkp1 = i*iStride + j*jStride + (k+1)*kStride;

  if((i >= iMin_d-1)&&(i < iMax_d+1) && (j >= jMin_d-1)&&(j < jMax_d+1) && (k >= kMin_d-1)&&(k < kMax_d+1)){

    dedx[ijk] = 0.5*(dXi_d*(scalar[ip1jk]*rhoInv[ip1jk]-scalar[im1jk]*rhoInv[im1jk]));

    dedy[ijk] = 0.5*(dYi_d*(scalar[ijp1k]*rhoInv[ijp1k]-scalar[ijm1k]*rhoInv[ijm1k]));

    dedz[ijk] = 0.5*(J31_d[ijk]*dXi_d*(scalar[ip1jk]*rhoInv[ip1jk]-scalar[im1jk]*rhoInv[im1jk])
                    +J32_d[ijk]*dYi_d*(scalar[ijp1k]*rhoInv[ijp1k]-scalar[ijm1k]*rhoInv[ijm1k])
                    +J33_d[ijk]*dZi_d*(scalar[ijkp1]*rhoInv[ijkp1]-scalar[ijkm1]*rhoInv[ijkm1]));

  }

} //end cudaDevice_GradScalar

/*----->>>>> __device__ void  cudaDevice_sgstkeTurbTransport();  --------------------------------------------------
*/
__device__ void cudaDevice_sgstkeTurbTransport(float* Km, float* dedx, float* dedy, float* dedz, float* rho, float* Frhs_sgstke,
                                               float* J31_d, float* J32_d, float* J33_d){

  float term_x;
  float term_y;
  float term_z;
  float f_sgstke_tt;

  int i,j,k,ijk;
  int im1jk,ijm1k,ijkm1;
  int ip1jk,ijp1k,ijkp1;
  int iStride,jStride,kStride;

  i = (blockIdx.x)*blockDim.x + threadIdx.x;
  j = (blockIdx.y)*blockDim.y + threadIdx.y;
  k = (blockIdx.z)*blockDim.z + threadIdx.z;
  iStride = (Ny_d+2*Nh_d)*(Nz_d+2*Nh_d);
  jStride = (Nz_d+2*Nh_d);
  kStride = 1;
  ijk = i*iStride + j*jStride + k*kStride;
  im1jk = (i-1)*iStride + j*jStride + k*kStride;
  ijm1k = i*iStride + (j-1)*jStride + k*kStride;
  ijkm1 = i*iStride + j*jStride + (k-1)*kStride;
  ip1jk = (i+1)*iStride + j*jStride + k*kStride;
  ijp1k = i*iStride + (j+1)*jStride + k*kStride;
  ijkp1 = i*iStride + j*jStride + (k+1)*kStride;

  if((i >= iMin_d)&&(i < iMax_d) && (j >= jMin_d)&&(j < jMax_d) && (k >= kMin_d)&&(k < kMax_d)){

    // -2.0*Km*de/dxi // 2.0 * 0.5 factor of the 2dx derivative cancel out
    term_x = -(dXi_d*(dedx[ip1jk]*Km[ip1jk]*rho[ip1jk]-dedx[im1jk]*Km[im1jk]*rho[im1jk]));

    term_y = -(dYi_d*(dedy[ijp1k]*Km[ijp1k]*rho[ijp1k]-dedy[ijm1k]*Km[ijm1k]*rho[ijm1k]));

    term_z = -(J31_d[ijk]*dXi_d*(dedz[ip1jk]*Km[ip1jk]*rho[ip1jk]-dedz[im1jk]*Km[im1jk]*rho[im1jk])
              +J32_d[ijk]*dYi_d*(dedz[ijp1k]*Km[ijp1k]*rho[ijp1k]-dedz[ijm1k]*Km[ijm1k]*rho[ijm1k])
              +J33_d[ijk]*dZi_d*(dedz[ijkp1]*Km[ijkp1]*rho[ijkp1]-dedz[ijkm1]*Km[ijkm1]*rho[ijkm1]));

    f_sgstke_tt = -(term_x+term_y+term_z);

    Frhs_sgstke[ijk] = Frhs_sgstke[ijk] + f_sgstke_tt;

#if DEBUG_TKE
    if (i==iMin_d && j==jMin_d){
      printf("cudaDevice_sgsTurbTransport(): At (%d,%d,%d) term_x,term_y,term_z,f_sgstke_tt = %f,%f,%f,%f\n",i,j,k,term_x,term_y,term_z,f_sgstke_tt);
    }
#endif

  }

} //end cudaDevice_sgstkeTurbTransport
