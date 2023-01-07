/* FastEddy®: SRC/HYDRO_CORE/CUDA/cuda_sgsTurbDevice.cu 
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
/*---TURBULENCE*/
__constant__ int turbulenceSelector_d;    /*turbulence scheme selector: 0= none, 1= Lilly/Smagorinsky */
__constant__ int TKESelector_d;        /* Prognostic TKE selector: 0= none, 1= Prognostic */
__constant__ float c_s_d;            /* Smagorinsky turbulence model constant used for turbulenceSelector = 1 with TKESelector = 0 */
__constant__ float c_k_d;            /* Lilly turbulence model constant used for turbulenceSelector = 1 with TKESelector > 0 */
float* hydroTauFlds_d;  /*Base address for 6 Tau field arrays*/
float* hydroKappaM_d;  /*Base address for KappaM (eddy diffusivity for momentum)*/

/*----->>>>> int cuda_sgsTurbDeviceSetup();       ----------------------------------------------------------------
 * Used to cudaMalloc and cudaMemcpy parameters and coordinate arrays, and for the SGSTURB HC-Submodule.
*/
extern "C" int cuda_sgsTurbDeviceSetup(){
   int errorCode = CUDA_SGSTURB_SUCCESS;
   int Nelems;

   cudaMemcpyToSymbol(turbulenceSelector_d, &turbulenceSelector, sizeof(int));
   cudaMemcpyToSymbol(TKESelector_d, &TKESelector, sizeof(int));
   cudaMemcpyToSymbol(c_s_d, &c_s, sizeof(float));
   cudaMemcpyToSymbol(c_k_d, &c_k, sizeof(float));

   Nelems = (Nxp+2*Nh)*(Nyp+2*Nh)*(Nzp+2*Nh);

   fecuda_DeviceMalloc(Nelems*sizeof(float), &hydroKappaM_d); 
   fecuda_DeviceMalloc(Nelems*9*sizeof(float), &hydroTauFlds_d);
  
   /* Done */
   return(errorCode);
} //end cuda_sgsTurbDeviceSetup

/*----->>>>> extern "C" int cuda_sgsTurbDeviceCleanup();  -----------------------------------------------------------
Used to free all malloced memory by the SGSTURB HC-Submodule.
*/
extern "C" int cuda_sgsTurbDeviceCleanup(){
   int errorCode = CUDA_SGSTURB_SUCCESS;

   cudaFree(hydroKappaM_d);
   cudaFree(hydroTauFlds_d);
   
   return(errorCode);
}//end cuda_sgsTurbDeviceCleanup()

/*----->>>>> __device__ void  cudaDevice_hydroCoreCalcStrainRateElements();  ---------------------------------------
* This is the cuda version of calculating strain rate tensor or S_ij fields for subgrid-scale mixing formulations 
*/
__device__ void cudaDevice_hydroCoreCalcStrainRateElements(float* u, float* v, float* w, float* theta,
                                                           float* S11, float* S21, float* S31,
                                                           float* S32, float* S22, float* S33,
                                                           float* STH1, float* STH2, float* STH3,
                                                           float* J31_d, float* J32_d, float* J33_d,
                                                           float* rhoInv){
   int i,j,k,ijk;
   int im1jk,ijm1k,ijkm1;
   int im1jm1k, im1jkm1, ijm1km1, im1jm1km1;
   int iStride,jStride,kStride;
   /*Establish necessary indices for spatial locality*/
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
   im1jm1k = (i-1)*iStride + (j-1)*jStride + k*kStride;
   im1jkm1 = (i-1)*iStride + j*jStride + (k-1)*kStride;
   ijm1km1 = i*iStride + (j-1)*jStride + (k-1)*kStride;
   im1jm1km1 = (i-1)*iStride + (j-1)*jStride + (k-1)*kStride;

   if((i >= iMin_d-1)&&(i < iMax_d+1) &&
      (j >= jMin_d-1)&&(j < jMax_d+1) &&
      (k >= kMin_d)&&(k < kMax_d+1) ){

    /* Calculate the strain rate tensor elements, Sij. 
    * Note that S12 = S21, S13 = S31, and S23 = S32 since the strain rate and stress tensors 
    * are symmetric tensors.
    */

    /*S11 = 1/2*(du/dx+du/dx) @ (i-1/2,j,k), assuming A-grid  */
    S11[ijk] = (
                dXi_d*( u[ijk]*rhoInv[ijk]   //du_dxi   
                       -u[im1jk]*rhoInv[im1jk])
               ); //Done with S11 = 1/2*2*(du_dx) 
    /*S22 = 1/2*(dv/dy+dv/dy) @ (i,j-1/2,k), assuming A-grid  */
    S22[ijk] = (
                 dYi_d*( v[ijk]*rhoInv[ijk]   //dv_deta 
                                    -v[ijm1k]*rhoInv[ijm1k])
               ); //Done with S22 = 1/2*2*(dv_dy) 
    /*S33 = 1/2*(dw/dz+dw/dz) @ (i,j,k-1/2), assuming A-grid  */
    S33[ijk] = (
                  J31_d[ijk]*dXi_d*( w[ijk]*rhoInv[ijk]   //dw_dxi   
                                    -w[im1jk]*rhoInv[im1jk])
                 +J32_d[ijk]*dYi_d*( w[ijk]*rhoInv[ijk]   //dw_deta  
                                    -w[ijm1k]*rhoInv[ijm1k])
                 +J33_d[ijk]*dZi_d*( w[ijk]*rhoInv[ijk]   //dw_dzeta 
                                    -w[ijkm1]*rhoInv[ijkm1])
               ); //Done with S33 = 1/2*2*(dw_dz) 
    /*S21 = 1/2*(dv/dx+du/dy) @ (i-1/2,j-1/2,k), assuming A-grid  */
    S21[ijk] = 0.5*(
                     0.5*( dXi_d*( v[ijk]*rhoInv[ijk]   //dv_dxi @i-1/2,j  
                                  -v[im1jk]*rhoInv[im1jk])
                          +dXi_d*( v[ijm1k]*rhoInv[ijm1k]   //dv_dxi @i-1/2,j-1
                                  -v[im1jm1k]*rhoInv[im1jm1k]))
                    +0.5*( dYi_d*( u[ijk]*rhoInv[ijk]   //du_deta 
                                  -u[ijm1k]*rhoInv[ijm1k])
                          +dYi_d*( u[im1jk]*rhoInv[im1jk]   //du_deta 
                                  -u[im1jm1k]*rhoInv[im1jm1k]))
                   ); //Done with S21 = 1/2*(dv_dx + du_dy)  
    /*S31 = 1/2*(dw/dx+du/dz) @ (i-1/2,j,k-1/2), assuming A-grid  */
    S31[ijk] = 0.5*(
                     0.5*( dXi_d*( w[ijk]*rhoInv[ijk]   //dw_dxi @i-1/2,j,k  
                                  -w[im1jk]*rhoInv[im1jk])
                          +dXi_d*( w[ijkm1]*rhoInv[ijkm1]   //dw_dxi @i-1/2,j,k-1
                                  -w[im1jkm1]*rhoInv[im1jkm1]))

                    +0.5*( J31_d[ijk]*dXi_d*( u[ijk]*rhoInv[ijk]   //du_dxi @i-1/2,j,k
                                             -u[im1jk]*rhoInv[im1jk])
                          +J31_d[ijk]*dXi_d*( u[ijkm1]*rhoInv[ijkm1]   //du_dxi @i-1/2,j,k-1
                                             -u[im1jkm1]*rhoInv[im1jkm1]))
                    +0.25*( J32_d[ijk]*dYi_d*( u[ijk]*rhoInv[ijk]   //du_deta @i,j-1/2,k
                                              -u[ijm1k]*rhoInv[ijm1k])
                           +J32_d[ijk]*dYi_d*( u[im1jk]*rhoInv[im1jk]   //du_deta @i-1,j-1/2,k
                                              -u[im1jm1k]*rhoInv[im1jm1k])
                           +J32_d[ijk]*dYi_d*( u[ijkm1]*rhoInv[ijkm1]   //du_deta @i,j-1/2,k-1
                                              -u[ijm1km1]*rhoInv[ijm1km1])
                           +J32_d[ijk]*dYi_d*( u[im1jkm1]*rhoInv[im1jkm1]   //du_deta @i-1,j-1/2,k-1
                                              -u[im1jm1km1]*rhoInv[im1jm1km1]))
                    +0.5*( J33_d[ijk]*dZi_d*( u[ijk]*rhoInv[ijk]   //du_dzeta @i,j,k-1/2
                                            -u[ijkm1]*rhoInv[ijkm1])
                          +J33_d[ijk]*dZi_d*( u[im1jk]*rhoInv[im1jk]   //du_deta @i-1,j,k-1/2
                                             -u[im1jkm1]*rhoInv[im1jkm1]))
                   ); //Done with S31 = 1/2*(dw_dx + du_dz)  
    /*S32 = 1/2*(dw/dy+dv/dz) @ (i,j-1/2,k-1/2), assuming A-grid  */
    S32[ijk] = 0.5*(
                     0.5*( dYi_d*( w[ijk]*rhoInv[ijk]   //dw_deta @i,j-1/2,k
                                  -w[ijm1k]*rhoInv[ijm1k])
                          +dYi_d*( w[ijkm1]*rhoInv[ijkm1]   //dw_deta  @i,j-1/2,k-1
                                  -w[ijm1km1]*rhoInv[ijm1km1]))

                    +0.25*( J31_d[ijk]*dXi_d*( v[ijk]*rhoInv[ijk]   //dv_dxi @i-1/2,j,k   
                                              -v[im1jk]*rhoInv[im1jk])
                           +J31_d[ijk]*dXi_d*( v[ijm1k]*rhoInv[ijm1k]   //dv_dxi @i-1/2,j-1,k
                                              -v[im1jm1k]*rhoInv[im1jm1k])
                           +J31_d[ijk]*dXi_d*( v[ijkm1]*rhoInv[ijkm1]   //dv_dxi @i-1/2,j,k-1  
                                              -v[im1jkm1]*rhoInv[im1jkm1])
                           +J31_d[ijk]*dXi_d*( v[ijm1km1]*rhoInv[ijm1km1]   //dv_dxi @i-1/2,j-1,k-1   
                                              -v[im1jm1km1]*rhoInv[im1jm1km1]))
                    +0.5*( J32_d[ijk]*dYi_d*( v[ijk]*rhoInv[ijk]   //dv_deta @i,j-1/2,k 
                                             -v[ijm1k]*rhoInv[ijm1k])
                          +J32_d[ijk]*dYi_d*( v[ijkm1]*rhoInv[ijkm1]   //dv_deta @i,j-1/2,k-1 
                                             -v[ijm1km1]*rhoInv[ijm1km1]))
                    +0.5*(J33_d[ijk]*dZi_d*( v[ijk]*rhoInv[ijk]   //dv_dzeta @i,j,k-1/2
                                            -v[ijkm1]*rhoInv[ijkm1])
                    +J33_d[ijk]*dZi_d*( v[ijm1k]*rhoInv[ijm1k]   //dv_dzeta @i,j-1,k-1/2
                                       -v[ijm1km1]*rhoInv[ijm1km1]))
              ); //Done with S32 = 1/2*(dw_dy + dv_dz)

    /*STH1 = (dTH/dx) @ (i-1/2), assuming A-grid  */
    STH1[ijk] = (                                   
                  (dXi_d*( theta[ijk]*rhoInv[ijk]   //dTH_dxi @i-1/2,j,k   
                          -theta[im1jk]*rhoInv[im1jk]))
                ); //Done with STH1 = (dTH/dx)

    /*STH2 = (dTH/dy) @ (i,j-1/2,k), assuming A-grid  */
    STH2[ijk] = ( 
                 (dYi_d*( theta[ijk]*rhoInv[ijk]   //dTH_deta @i,j-1/2,k 
                         -theta[ijm1k]*rhoInv[ijm1k]))
                ); //Done with STH2 = (dTH/dy)

    /*STH3 = (dTH/dz) @ (i,j,k-1/2), assuming A-grid  */
    STH3[ijk] = ( (J31_d[ijk]*dXi_d*( theta[ijk]*rhoInv[ijk]   //dTH_dxi @i-1/2,j,k   
                                     -theta[im1jk]*rhoInv[im1jk]))
                 +(J32_d[ijk]*dYi_d*( theta[ijk]*rhoInv[ijk]   //dTH_deta @i,j-1/2,k 
                                     -theta[ijm1k]*rhoInv[ijm1k]))
                 +(J33_d[ijk]*dZi_d*( theta[ijk]*rhoInv[ijk]   //dTH_dzeta @i,j,k-1/2
                                     -theta[ijkm1]*rhoInv[ijkm1]))
                ); //Done with STH3 = (dTH/dz)

   }//end if in the range of non-halo cells

} // cudaDevice_hydroCoreCalcStrainRateElements()

/*----->>>>> __device__ void  cudaDevice_hydroCoreCalcEddyDiff();  ---------------------------------------------
* This is the cuda version of calculating eddy diffusivity of momentum Km
*/
__device__ void cudaDevice_hydroCoreCalcEddyDiff(float* S11, float* S21, float* S31,
                                                 float* S32, float* S22, float* S33,
                                                 float* STH3, float* theta, float* rhoInv,
                                                 float* sgstke, float* sgstke_ls, float* Km, float* D_Jac_d){

   int i,j,k,ijk,ijkm1;
   int iStride,jStride,kStride;
   float turbDelta;
   float invPr;
   float dTh_dz;
   float BruntVaisalaSquared;
   float Km_ijk;
   float term1_Sm;
   float term2_Sm;
   float sgstke_ijk,sgstke_ijkm1;

   /*Establish necessary indices for spatial locality*/
   i = (blockIdx.x)*blockDim.x + threadIdx.x;
   j = (blockIdx.y)*blockDim.y + threadIdx.y;
   k = (blockIdx.z)*blockDim.z + threadIdx.z;
   iStride = (Ny_d+2*Nh_d)*(Nz_d+2*Nh_d);
   jStride = (Nz_d+2*Nh_d);
   kStride = 1;
   ijk = i*iStride + j*jStride + k*kStride;
   ijkm1 = i*iStride + j*jStride + (k-1)*kStride;

   if((i >= iMin_d-1)&&(i < iMax_d+1) &&
      (j >= jMin_d-1)&&(j < jMax_d+1) &&
      (k >= kMin_d)&&(k < kMax_d+1) ){

      /*Calculate the turbulent eddy viscosity for momentum*/
      /* Kappa-Momentum (Km) as the isotropic (same in all directions) eddy-diffusivity for momentum*/
      if(TKESelector_d>0){ // Lilly (prognostic SGSTKE equation)
        sgstke_ijk = sgstke[ijk]*rhoInv[ijk];
        sgstke_ijkm1 = sgstke[ijkm1]*rhoInv[ijkm1];
        Km_ijk = c_k_d*0.5*(sgstke_ls[ijk]*powf(sgstke_ijk,0.5)+sgstke_ls[ijkm1]*powf(sgstke_ijkm1,0.5));
      }else{ // Smagorinsky
        turbDelta = powf(dX_d*dY_d*dZ_d*D_Jac_d[ijk],1.0/3.0);   // "l"
        invPr = 3.0;
        dTh_dz = STH3[ijk];
        term1_Sm = 0.5*(S11[ijk]*S11[ijk]+S22[ijk]*S22[ijk]+S33[ijk]*S33[ijk]) //1/2*Tr(Sij^2)
                  +S21[ijk]*S21[ijk]+S31[ijk]*S31[ijk]+S32[ijk]*S32[ijk];      //+ the off-diagonals^2
        BruntVaisalaSquared = accel_g_d/(theta[ijk]*rhoInv[ijk])*dTh_dz;
        term2_Sm = BruntVaisalaSquared*invPr;
        Km_ijk = (c_s_d*turbDelta)*(c_s_d*turbDelta)             //c^2*l^2
                 *sqrtf(fmaxf(0.0, term1_Sm-fmaxf(0.0,term2_Sm))); //*sqrt(max(0,term1_Sm)-max(0,term2_Sm))
      }

      Km[ijk] = Km[ijk] + Km_ijk;

   }//end if in the range of non-halo cells

} // cudaDevice_hydroCoreCalcEddyDiff()

/*----->>>>> __device__ void  cudaDevice_hydroCoreCalcTaus();  ---------------------------------------------
* This is the cuda version of calculating SGS stresses or tau_ij fields for subgrid-scale mixing formulations
*/
__device__ void cudaDevice_hydroCoreCalcTaus(float* S11, float* S21, float* S31,
                                             float* S32, float* S22, float* S33,
                                             float* STH1, float* STH2, float* STH3,
                                             float* rho, float* Km, float* sgstke_ls, float* D_Jac_d){
   int i,j,k,ijk;
   int iStride,jStride,kStride;
   float turbLengthScale;
   float turbDelta;
   float invPr;

   /*Establish necessary indices for spatial locality*/
   i = (blockIdx.x)*blockDim.x + threadIdx.x;
   j = (blockIdx.y)*blockDim.y + threadIdx.y;
   k = (blockIdx.z)*blockDim.z + threadIdx.z;
   iStride = (Ny_d+2*Nh_d)*(Nz_d+2*Nh_d);
   jStride = (Nz_d+2*Nh_d);
   kStride = 1;
   ijk = i*iStride + j*jStride + k*kStride;

   if((i >= iMin_d-1)&&(i < iMax_d+1) &&
      (j >= jMin_d-1)&&(j < jMax_d+1) &&
      (k >= kMin_d)&&(k < kMax_d+1) ){

      turbDelta = powf(dX_d*dY_d*dZ_d*D_Jac_d[ijk],1.0/3.0);   // "l"
      if(TKESelector_d > 0){
        turbLengthScale = sgstke_ls[ijk];
      }else{
        turbLengthScale = turbDelta;
      }
      invPr = 1.0 + 2.0*turbLengthScale/turbDelta;
      //Factor in Km and rho to make Tau_ij=-rho*Km*Sij
      S11[ijk]  = -2.0*rho[ijk]*Km[ijk]*S11[ijk];
      S21[ijk]  = -2.0*rho[ijk]*Km[ijk]*S21[ijk];
      S31[ijk]  = -2.0*rho[ijk]*Km[ijk]*S31[ijk];
      S32[ijk]  = -2.0*rho[ijk]*Km[ijk]*S32[ijk];
      S22[ijk]  = -2.0*rho[ijk]*Km[ijk]*S22[ijk];
      S33[ijk]  = -2.0*rho[ijk]*Km[ijk]*S33[ijk];
      STH1[ijk] = -rho[ijk]*Km[ijk]*invPr*STH1[ijk];
      STH2[ijk] = -rho[ijk]*Km[ijk]*invPr*STH2[ijk];
      STH3[ijk] = -rho[ijk]*Km[ijk]*invPr*STH3[ijk];

   } //end if in the range of non-halo cells

} //cudaDevice_hydroCoreCalcTaus()

/*----->>>>> __device__ void  cudaDevice_hydroCoreCalcTaus_PrognosticTKE_DeviatoricTerm();  -------------------------
*  This is the compressible stress formulation routine with deviatoric term modeled using 
*  the prognostic TKE (i.e. TKESelector_d > 0)
*/
__device__ void cudaDevice_hydroCoreCalcTaus_PrognosticTKE_DeviatoricTerm(
                                             float* S11, float* S21, float* S31,
                                             float* S32, float* S22, float* S33,
                                             float* STH1, float* STH2, float* STH3,
                                             float* rho, float* Km, float* sgstke_ls,
                                             float* u, float* v, float* w, float* sgstke,
                                             float* J31_d, float* J32_d, float* J33_d, float* D_Jac_d){
   int i,j,k,ijk;
   int im1jk,ijm1k,ijkm1;
   int iStride,jStride,kStride;
   float turbLengthScale;
   float turbDelta;
   float invPr;
   float deviatoricTerm;

   /*Establish necessary indices for spatial locality*/
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

   if((i >= iMin_d-1)&&(i < iMax_d+1) &&
      (j >= jMin_d-1)&&(j < jMax_d+1) &&
      (k >= kMin_d)&&(k < kMax_d+1) ){

      turbDelta = powf(dX_d*dY_d*dZ_d*D_Jac_d[ijk],1.0/3.0);   // "l"
      turbLengthScale = sgstke_ls[ijk];
      invPr = 1.0 + 2.0*turbLengthScale/turbDelta;
      //Factor in Km and rho to make Tau_ij=-rho*Km*Sij
      deviatoricTerm = (2.0/3.0)*rho[ijk]*(
                                           Km[ijk]*( dXi_d*( u[ijk]/rho[ijk]   //du_dxi   
                                                            -u[im1jk]/rho[im1jk])
                                                    +dYi_d*( v[ijk]/rho[ijk]   //dv_deta 
                                                            -v[ijm1k]/rho[ijm1k])
                                                    +J31_d[ijk]*dXi_d*( w[ijk]/rho[ijk]   //dw_dxi   
                                                                       -w[im1jk]/rho[im1jk])
                                                    +J32_d[ijk]*dYi_d*( w[ijk]/rho[ijk]   //dw_deta  
                                                                       -w[ijm1k]/rho[ijm1k])
                                                    +J33_d[ijk]*dZi_d*( w[ijk]/rho[ijk]   //dw_dzeta 
                                                                       -w[ijkm1]/rho[ijkm1]) )
                                           + sgstke[ijk]/rho[ijk] );
      S11[ijk]  = -2.0*rho[ijk]*Km[ijk]*S11[ijk] + deviatoricTerm;
      S21[ijk]  = -2.0*rho[ijk]*Km[ijk]*S21[ijk];
      S31[ijk]  = -2.0*rho[ijk]*Km[ijk]*S31[ijk];
      S32[ijk]  = -2.0*rho[ijk]*Km[ijk]*S32[ijk];
      S22[ijk]  = -2.0*rho[ijk]*Km[ijk]*S22[ijk] + deviatoricTerm;
      S33[ijk]  = -2.0*rho[ijk]*Km[ijk]*S33[ijk] + deviatoricTerm;
      STH1[ijk] = -rho[ijk]*Km[ijk]*invPr*STH1[ijk];
      STH2[ijk] = -rho[ijk]*Km[ijk]*invPr*STH2[ijk];
      STH3[ijk] = -rho[ijk]*Km[ijk]*invPr*STH3[ijk];

   } //end if in the range of non-halo cells

} //cudaDevice_hydroCoreCalcTausi_PrognosticTKE_DeviatoricTerm()

/*----->>>>> __device__ void  cudaDevice_GradScalarToFaces();  --------------------------------------------------
* This cuda kernel calculates the spatial gradient of a scalar field: 1delta, gradient located at the cell face
*/
__device__ void cudaDevice_GradScalarToFaces(float* scalar, float* rhoInv, float* dSdx, float* dSdy, float* dSdz,
                                             float* J31_d, float* J32_d, float* J33_d){

  int i,j,k,ijk;
  int im1jk,ijm1k,ijkm1;
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

  if((i >= iMin_d-1)&&(i < iMax_d+1) && (j >= jMin_d-1)&&(j < jMax_d+1) && (k >= kMin_d-1)&&(k < kMax_d+1)){

    dSdx[ijk] =     (dXi_d*(scalar[ijk]*rhoInv[ijk]-scalar[im1jk]*rhoInv[im1jk]));

    dSdy[ijk] =     (dYi_d*(scalar[ijk]*rhoInv[ijk]-scalar[ijm1k]*rhoInv[ijm1k]));

    dSdz[ijk] =     (J31_d[ijk]*dXi_d*(scalar[ijk]*rhoInv[ijk]-scalar[im1jk]*rhoInv[im1jk])
                    +J32_d[ijk]*dYi_d*(scalar[ijk]*rhoInv[ijk]-scalar[ijm1k]*rhoInv[ijm1k])
                    +J33_d[ijk]*dZi_d*(scalar[ijk]*rhoInv[ijk]-scalar[ijkm1]*rhoInv[ijkm1]));

  }

} //end cudaDevice_GradScalarToFaces

/*----->>>>> __device__ void  cudaDevice_hydroCoreCalcTausScalar();  ---------------------------------------------
* This is the cuda version of calculating SGS stresses of a scalar field field for subgrid-scale mixing formulations
*/
__device__ void cudaDevice_hydroCoreCalcTausScalar(float* SM1, float* SM2, float* SM3,
                                                   float* rho, float* Km, float* sgstke_ls, float* D_Jac_d){
   int i,j,k,ijk;
   int iStride,jStride,kStride;
   float turbLengthScale;
   float turbDelta;
   float invPr;

   /*Establish necessary indices for spatial locality*/
   i = (blockIdx.x)*blockDim.x + threadIdx.x;
   j = (blockIdx.y)*blockDim.y + threadIdx.y;
   k = (blockIdx.z)*blockDim.z + threadIdx.z;
   iStride = (Ny_d+2*Nh_d)*(Nz_d+2*Nh_d);
   jStride = (Nz_d+2*Nh_d);
   kStride = 1;
   ijk = i*iStride + j*jStride + k*kStride;

   if((i >= iMin_d-1)&&(i < iMax_d+1) &&
      (j >= jMin_d-1)&&(j < jMax_d+1) &&
      (k >= kMin_d)&&(k < kMax_d+1) ){

      turbDelta = powf(dX_d*dY_d*dZ_d*D_Jac_d[ijk],1.0/3.0);   // "l"
      if(TKESelector_d > 0){
        turbLengthScale = sgstke_ls[ijk];
      }else{
        turbLengthScale = turbDelta;
      }
      invPr = 1.0 + 2.0*turbLengthScale/turbDelta;
      //Factor in Km and rho to make Tau_ij=-rho*Km*Sij
      SM1[ijk] = -rho[ijk]*Km[ijk]*invPr*SM1[ijk];
      SM2[ijk] = -rho[ijk]*Km[ijk]*invPr*SM2[ijk];
      SM3[ijk] = -rho[ijk]*Km[ijk]*invPr*SM3[ijk];

   } //end if in the range of non-halo cells

} //cudaDevice_hydroCoreCalcTausScalar()

/*----->>>>> __device__ void  cudaDevice_hydroCoreCalcTurbMixing();  ---------------------------------------------
* This is the cuda version of calculating forcing terms from subgrid-scale mixing  
*/
__device__ void cudaDevice_hydroCoreCalcTurbMixing(float* uFrhs, float* vFrhs, float* wFrhs, float* thetaFrhs,
                                                   float* T11, float* T21, float* T31,
                                                   float* T32, float* T22, float* T33,
                                                   float* TH1, float* TH2, float* TH3,
                                                   float* J31_d, float* J32_d, float* J33_d){
   int i,j,k,ijk;
   int ip1jk,ijp1k,ijkp1;
   int iStride,jStride,kStride;
   /*Establish necessary indices for spatial locality*/
   i = (blockIdx.x)*blockDim.x + threadIdx.x;
   j = (blockIdx.y)*blockDim.y + threadIdx.y;
   k = (blockIdx.z)*blockDim.z + threadIdx.z;

   if((i >= iMin_d)&&(i < iMax_d) &&
      (j >= jMin_d)&&(j < jMax_d) &&
      (k >= kMin_d)&&(k < kMax_d) ){

    iStride = (Ny_d+2*Nh_d)*(Nz_d+2*Nh_d);
    jStride = (Nz_d+2*Nh_d);
    kStride = 1;
    ijk = i*iStride + j*jStride + k*kStride;
    ip1jk = (i+1)*iStride + j*jStride + k*kStride;
    ijp1k = i*iStride + (j+1)*jStride + k*kStride;
    ijkp1 = i*iStride + j*jStride + (k+1)*kStride;
    
    /* Calculate the momentum forcing. 
    * Note that T12 = T21, T13 = T31, and T23 = T32 since the strain rate and stress tensors 
    * are symmetric tensors.
    */
    /*uFrhs = -d_dxj( -2*(Kappa_m*T1j) ), j=1,2,3 @ (i,j,k) cell-center, assuming A-grid  */
    uFrhs[ijk] = uFrhs[ijk] - (
                                dXi_d*(T11[ip1jk] - T11[ijk])
                               +dYi_d*(T21[ijp1k] - T21[ijk])
                               +J31_d[ijk]*dXi_d*(T31[ip1jk] - T31[ijk])
                               +J32_d[ijk]*dYi_d*(T31[ijp1k] - T31[ijk])
                               +J33_d[ijk]*dZi_d*(T31[ijkp1] - T31[ijk])
          ); //Done with uFrhs = -2*(d_dx(Kappa_m*T11) + d_dy(Kappa_m*T12) + d_dz(Kappa_m*T13))  
    /*vFrhs = -d_dxj( -2*(Kappa_m*T2j) ), j=1,2,3 @ (i,j,k) cell-center, assuming A-grid  */
    vFrhs[ijk] = vFrhs[ijk] - (
                                dXi_d*(T21[ip1jk] - T21[ijk])
                               +dYi_d*(T22[ijp1k] - T22[ijk])
                               +J31_d[ijk]*dXi_d*(T32[ip1jk] - T32[ijk])
                               +J32_d[ijk]*dYi_d*(T32[ijp1k] - T32[ijk])
                               +J33_d[ijk]*dZi_d*(T32[ijkp1] - T32[ijk])
          ); //Done with vFrhs = -2*(d_dx(Kappa_m*T21) + d_dy(Kappa_m*T22) + d_dz(Kappa_m*T23))  
    /*wFrhs = -d_dxj( -2*(Kappa_m*dT3j_dxj)), j=1,2,3 @ (i,j,k) cell-center, assuming A-grid  */
    wFrhs[ijk] = wFrhs[ijk] - (
                                dXi_d*(T31[ip1jk] - T31[ijk])
                               +dYi_d*(T32[ijp1k] - T32[ijk])
                               +J31_d[ijk]*dXi_d*(T33[ip1jk] - T33[ijk])
                               +J32_d[ijk]*dYi_d*(T33[ijp1k] - T33[ijk])
                               +J33_d[ijk]*dZi_d*(T33[ijkp1] - T33[ijk])
          ); //Done with wFrhs = -2*(d_dx(Kappa_m*T31) + d_dy(Kappa_m*T32) + d_dz(Kappa_m*T33))  
    thetaFrhs[ijk] = thetaFrhs[ijk] - (
                                        dXi_d*(TH1[ip1jk] - TH1[ijk])
                                       +dYi_d*(TH2[ijp1k] - TH2[ijk])
                                       +J31_d[ijk]*dXi_d*(TH3[ip1jk] - TH3[ijk])
                                       +J32_d[ijk]*dYi_d*(TH3[ijp1k] - TH3[ijk])
                                       +J33_d[ijk]*dZi_d*(TH3[ijkp1] - TH3[ijk])
          ); //Done with thetaFrhs = (d_dx(Kappa*TH1) + d_dy(TH2) + d_dz(Kappa*TH3))  

   }//end if in the range of non-halo cells
} //end cudaDevice_hydroCoreCalcTurbMixing()

/*----->>>>> __device__ void  cudaDevice_hydroCoreCalcTurbMixingScalar();  ---------------------------------------------
* This is the cuda version of calculating forcing terms from subgrid-scale mixing of a scalar field
*/
__device__ void cudaDevice_hydroCoreCalcTurbMixingScalar(float* mFrhs, float* M1, float* M2, float* M3,
                                                         float* J31_d, float* J32_d, float* J33_d){

   int i,j,k,ijk;
   int ip1jk,ijp1k,ijkp1;
   int iStride,jStride,kStride;
   /*Establish necessary indices for spatial locality*/
   i = (blockIdx.x)*blockDim.x + threadIdx.x;
   j = (blockIdx.y)*blockDim.y + threadIdx.y;
   k = (blockIdx.z)*blockDim.z + threadIdx.z;

   if((i >= iMin_d)&&(i < iMax_d) &&
      (j >= jMin_d)&&(j < jMax_d) &&
      (k >= kMin_d)&&(k < kMax_d) ){

    iStride = (Ny_d+2*Nh_d)*(Nz_d+2*Nh_d);
    jStride = (Nz_d+2*Nh_d);
    kStride = 1;
    ijk = i*iStride + j*jStride + k*kStride;
    ip1jk = (i+1)*iStride + j*jStride + k*kStride;
    ijp1k = i*iStride + (j+1)*jStride + k*kStride;
    ijkp1 = i*iStride + j*jStride + (k+1)*kStride;

    mFrhs[ijk] = mFrhs[ijk] - (
                                dXi_d*(M1[ip1jk] - M1[ijk])
                               +dYi_d*(M2[ijp1k] - M2[ijk])
                               +J31_d[ijk]*dXi_d*(M3[ip1jk] - M3[ijk])
                               +J32_d[ijk]*dYi_d*(M3[ijp1k] - M3[ijk])
                               +J33_d[ijk]*dZi_d*(M3[ijkp1] - M3[ijk])
                              ); //Done with mFrhs = (d_dx(Kappa*M1) + d_dy(Kappa*M2) + d_dz(Kappa*M3))

   }//end if in the range of non-halo cells

} //end cudaDevice_hydroCoreCalcTurbMixingScalar()
