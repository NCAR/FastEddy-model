/* FastEddy®: SRC/HYDRO_CORE/CUDA/cuda_advectionDevice.cu 
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
/*---ADVECTION*/ 
float *hydroFaceVels_d; //cell face velocities
__constant__ int advectionSelector_d;          /*advection scheme selector: 0= 1st-order upwind, 2= 3rd-order QUICK, 2= hybrid 3rd-4th order, 3= hybrid 5th-6th order */
__constant__ float b_hyb_d;                      /*hybrid advection scheme parameter: 0.0= higer-order upwind, 1.0=lower-order cetered, 0.0 < b_hyb < 1.0 = hybrid */

/*#################------------ ADVECTION submodule function definitions ------------------#############*/
/*----->>>>> int cuda_advectionDeviceSetup();       ---------------------------------------------------------
 * Used to cudaMalloc and cudaMemcpy parameters and coordinate arrays, and for the ADVECTION_CUDA submodule.
*/
extern "C" int cuda_advectionDeviceSetup(){
   int errorCode = CUDA_ADVECTION_SUCCESS;
   int Nelems;
   
   cudaMemcpyToSymbol(advectionSelector_d, &advectionSelector, sizeof(int));
   cudaMemcpyToSymbol(b_hyb_d, &b_hyb, sizeof(float));

   /*Set the full memory block number of elements for hydroCore fields*/
   Nelems = (Nxp+2*Nh)*(Nyp+2*Nh)*(Nzp+2*Nh);
   fecuda_DeviceMalloc(Nelems*3*sizeof(float), &hydroFaceVels_d); /*Cell-face Velocities*/

   return(errorCode);
} //end cuda_advectionDeviceSetup()

/*----->>>>> extern "C" int cuda_advectionDeviceCleanup();  -----------------------------------------------------------
Used to free all malloced memory by the ADVECTION_CUDA submodule.
*/
extern "C" int cuda_advectionDeviceCleanup(){
   int errorCode = CUDA_ADVECTION_SUCCESS;

   cudaFree(hydroFaceVels_d);
   
   return(errorCode);

}//end cuda_advectionDeviceCleanup()

/*----->>>>> __device__ void  cudaDevice_calcFaceVelocities()  --------------------------------------------------
* This device function calculates the cell face velocities to prepare for use in the chosen advection scheme
*/
__device__ void cudaDevice_calcFaceVelocities(float* hydroFlds_d, float* hydroFaceVels_d,
                                              float* J31_d, float* J32_d, float* J33_d, float* D_Jac_d){
   int i,j,k,ijk;
   int im1jk,ijm1k,ijkm1;
   int iStride,jStride,kStride,fldStride;
   float* rho;
   float* u;
   float* v;
   float* w;
   float* u_cf;
   float* v_cf;
   float* w_cf;

   /*Establish necessary indices for spatial locality*/
   i = (blockIdx.x)*blockDim.x + threadIdx.x;
   j = (blockIdx.y)*blockDim.y + threadIdx.y;
   k = (blockIdx.z)*blockDim.z + threadIdx.z;

   iStride = (Ny_d+2*Nh_d)*(Nz_d+2*Nh_d);
   jStride = (Nz_d+2*Nh_d);
   kStride = 1;
   fldStride = (Nx_d+2*Nh_d)*(Ny_d+2*Nh_d)*(Nz_d+2*Nh_d);
   
   rho = &hydroFlds_d[fldStride*RHO_INDX];
   u = &hydroFlds_d[fldStride*U_INDX];
   v = &hydroFlds_d[fldStride*V_INDX];
   w = &hydroFlds_d[fldStride*W_INDX];
   u_cf = &hydroFaceVels_d[fldStride*(U_INDX-1)]; /*The "-1" accounts for no RHO field in hydroFaceVels_d */
   v_cf = &hydroFaceVels_d[fldStride*(V_INDX-1)];
   w_cf = &hydroFaceVels_d[fldStride*(W_INDX-1)];

   ijk = i*iStride + j*jStride + k*kStride;
   im1jk = (i-1)*iStride + j*jStride + k*kStride;
   ijm1k = i*iStride + (j-1)*jStride + k*kStride;
   ijkm1 = i*iStride + j*jStride + (k-1)*kStride;
   if((i >= iMin_d)&&(i <= iMax_d) &&
      (j >= jMin_d)&&(j <= jMax_d) &&
      (k >= kMin_d)&&(k <= kMax_d) ){

      u_cf[ijk] = 0.5*dXi_d*
                  ( (D_Jac_d[ijk]/rho[ijk])*(u[ijk] + w[ijk]*J31_d[ijk])
                   +(D_Jac_d[im1jk]/rho[im1jk])*(u[im1jk] + w[im1jk]*J31_d[im1jk]));
      v_cf[ijk] = 0.5*dYi_d*
                  ( (D_Jac_d[ijk]/rho[ijk])*(v[ijk] + w[ijk]*J32_d[ijk])
                   +(D_Jac_d[ijm1k]/rho[ijm1k])*(v[ijm1k] + w[ijm1k]*J32_d[ijm1k]));
      w_cf[ijk] = 0.5*dZi_d*
                 ( (D_Jac_d[ijk]/rho[ijk])*(w[ijk]*J33_d[ijk])
                  +(D_Jac_d[ijkm1]/rho[ijkm1])*(w[ijkm1]*J33_d[ijkm1]));

      //Ensure ground and ceiling face vertical velocity component is set to 0
      if((k==kMin_d)||((k>=kMax_d))){
         w_cf[ijk] = 0.0;
      }//end if k==kMin_d || k>=kMax_d
#ifdef CUDA_DEBUG
//#if 1
      if((k==kMin_d)&&(fabsf(w_cf[ijk])>1e-6)){
        printf("cudaDevice_hydroCoreCalcFaceVelocities(): At (%d,%d,%d)...  u_cf= %f, v_cf= %f, w_cf= %f, rho=%f\n",
                                           i,j,k,u_cf[ijk],v_cf[ijk],w_cf[ijk], rho[ijk]);
        printf("                               : At (%d,%d,%d)... D_Jac_d = %f, u= %f, v= %f, w=%f\n",
                                           i,j,k,D_Jac_d[ijk],u[ijk],v[ijk],w[ijk]);
        printf("                               : At (%d,%d,%d)... u_cf= %f, v_cf= %f, w_cf= %f, rho=%f\n",
                                           i,j,k-1,u_cf[ijkm1],v_cf[ijkm1],w_cf[ijkm1], rho[ijkm1]);
        printf("                               : At (%d,%d,%d)... D_Jac_d = %f, u= %f, v= %f, w=%f\n",
                                           i,j,k-1,D_Jac_d[ijkm1],u[ijkm1],v[ijkm1],w[ijkm1]);
      }
#endif
   } //end if in the range of non-halo cells
} //end calcFaceVelocities()

/*----->>>>> __device__ void  cudaDevice_UpstreamDivAdvFlux();  --------------------------------------------------
* This is the cuda version of the UpstreamDivAdvFlux routine from the HYDRO_CORE module
*/
__device__ void cudaDevice_UpstreamDivAdvFlux(float* scalarField, float* scalarFadv,
                                              float* u_cf, float* v_cf, float* w_cf, float* invD_Jac_d){
  int i,j,k;
  int ijk,im1jk,ijm1k,ijkm1,ip1jk,ijp1k,ijkp1;
  int iStride,jStride,kStride;
  float DscalarDx;
  float DscalarDy;
  float DscalarDz;

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
  ip1jk = (i+1)*iStride + j*jStride + k*kStride;
  ijp1k = i*iStride + (j+1)*jStride + k*kStride;
  ijkp1 = i*iStride + j*jStride + (k+1)*kStride;
  DscalarDx = ( ( fmaxf(0.0,u_cf[ ip1jk ])*scalarField[  ijk  ]+fminf(0.0,u_cf[ ip1jk ])*scalarField[ ip1jk ])
                 -( fmaxf(0.0,u_cf[  ijk  ])*scalarField[ im1jk ]+fminf(0.0,u_cf[  ijk  ])*scalarField[  ijk  ]) );
  DscalarDy = ( ( fmaxf(0.0,v_cf[ ijp1k ])*scalarField[  ijk  ]+fminf(0.0,v_cf[ ijp1k ])*scalarField[ ijp1k ])
                 -( fmaxf(0.0,v_cf[  ijk  ])*scalarField[ ijm1k ]+fminf(0.0,v_cf[  ijk  ])*scalarField[  ijk  ]) );
  DscalarDz = ( ( fmaxf(0.0,w_cf[ ijkp1 ])*scalarField[  ijk  ]+fminf(0.0,w_cf[ ijkp1 ])*scalarField[ ijkp1 ])
               -( fmaxf(0.0,w_cf[  ijk  ])*scalarField[ ijkm1 ]+fminf(0.0,w_cf[  ijk  ])*scalarField[  ijk  ]) );
  scalarFadv[ijk] = scalarFadv[ijk] -invD_Jac_d[ijk]*(DscalarDx+DscalarDy+DscalarDz);

} //end cudaDevice_UpstreamDivAdvFlux(

/*----->>>>> __device__ void  cudaDevice_SecondDivAdvFlux();  -----------------------------------------------*/ 
__device__ void cudaDevice_SecondDivAdvFlux(float* scalarField, float* scalarFadv,
                                              float* u_cf, float* v_cf, float* w_cf, float* invD_Jac_d){

  int i,j,k;
  int ijk,im1jk,ijm1k,ijkm1,ip1jk,ijp1k,ijkp1;
  int iStride,jStride,kStride;
  float DscalarDx;
  float DscalarDy;
  float DscalarDz;
  float flxx_ipf,flxx_imf,flxy_jpf,flxy_jmf,flxz_kpf,flxz_kmf;

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
  ip1jk = (i+1)*iStride + j*jStride + k*kStride;
  ijp1k = i*iStride + (j+1)*jStride + k*kStride;
  ijkp1 = i*iStride + j*jStride + (k+1)*kStride;

  flxx_ipf = 0.5*(scalarField[ ip1jk ]+scalarField[  ijk  ]);
  flxx_imf = 0.5*(scalarField[ ijk ]+scalarField[  im1jk  ]);
  flxy_jpf = 0.5*(scalarField[ ijp1k ]+scalarField[  ijk  ]);
  flxy_jmf = 0.5*(scalarField[ ijk ]+scalarField[  ijm1k  ]);
  flxz_kpf = 0.5*(scalarField[ ijkp1 ]+scalarField[  ijk  ]);
  flxz_kmf = 0.5*(scalarField[ ijk ]+scalarField[  ijkm1  ]);

  DscalarDx = u_cf[ ip1jk ]*flxx_ipf - u_cf[  ijk  ]*flxx_imf;
  DscalarDy = v_cf[ ijp1k ]*flxy_jpf - v_cf[  ijk  ]*flxy_jmf;
  DscalarDz = w_cf[ ijkp1 ]*flxz_kpf - w_cf[  ijk  ]*flxz_kmf;
  scalarFadv[ijk] = scalarFadv[ijk] -invD_Jac_d[ijk]*(DscalarDx+DscalarDy+DscalarDz);

} //end cudaDevice_SecondDivAdvFlux()

/*----->>>>> __device__ void  cudaDevice_QUICKDivAdvFlux();  --------------------------------------------------
* This is the cuda version of the hydro_coreQUICKDivAdvFlx routine from the HYDRO_CORE module
*/ 
__device__ void cudaDevice_QUICKDivAdvFlux(float* scalarField, float* scalarFadv,
                                              float* u_cf, float* v_cf, float* w_cf, float* invD_Jac_d){

  int i,j,k;
  int ijk,im1jk,ijm1k,ijkm1,ip1jk,ijp1k,ijkp1,im2jk,ijm2k,ijkm2,ip2jk,ijp2k,ijkp2;
  int iStride,jStride,kStride;
  float DscalarDx;
  float DscalarDy;
  float DscalarDz;
  float one_eighth;

  one_eighth = 1.0/8.0;

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
  ip1jk = (i+1)*iStride + j*jStride + k*kStride;
  ijp1k = i*iStride + (j+1)*jStride + k*kStride;
  ijkp1 = i*iStride + j*jStride + (k+1)*kStride;
  im2jk = (i-2)*iStride + j*jStride + k*kStride;
  ijm2k = i*iStride + (j-2)*jStride + k*kStride;
  ijkm2 = i*iStride + j*jStride + (k-2)*kStride;
  ip2jk = (i+2)*iStride + j*jStride + k*kStride;
  ijp2k = i*iStride + (j+2)*jStride + k*kStride;
  ijkp2 = i*iStride + j*jStride + (k+2)*kStride;

  DscalarDx = one_eighth * ( ( fmaxf(0.0,u_cf[ ip1jk ])*(6.0*scalarField[  ijk  ]+3.0*scalarField[ ip1jk ]-scalarField[ im1jk ])+
                               fminf(0.0,u_cf[ ip1jk ])*(6.0*scalarField[ ip1jk ]+3.0*scalarField[  ijk  ]-scalarField[ ip2jk ]) ) -
                             ( fmaxf(0.0,u_cf[  ijk  ])*(6.0*scalarField[ im1jk ]+3.0*scalarField[  ijk  ]-scalarField[ im2jk ])+
                               fminf(0.0,u_cf[  ijk  ])*(6.0*scalarField[  ijk  ]+3.0*scalarField[ im1jk ]-scalarField[ ip1jk ]) ) );
  DscalarDy = one_eighth * ( ( fmaxf(0.0,v_cf[ ijp1k ])*(6.0*scalarField[  ijk  ]+3.0*scalarField[ ijp1k ]-scalarField[ ijm1k ])+
                               fminf(0.0,v_cf[ ijp1k ])*(6.0*scalarField[ ijp1k ]+3.0*scalarField[  ijk  ]-scalarField[ ijp2k ]) ) -
                             ( fmaxf(0.0,v_cf[  ijk  ])*(6.0*scalarField[ ijm1k ]+3.0*scalarField[  ijk  ]-scalarField[ ijm2k ])+
                               fminf(0.0,v_cf[  ijk  ])*(6.0*scalarField[  ijk  ]+3.0*scalarField[ ijm1k ]-scalarField[ ijp1k ]) ) );
  DscalarDz = one_eighth * ( ( fmaxf(0.0,w_cf[ ijkp1 ])*(6.0*scalarField[  ijk  ]+3.0*scalarField[ ijkp1 ]-scalarField[ ijkm1 ])+
                               fminf(0.0,w_cf[ ijkp1 ])*(6.0*scalarField[ ijkp1 ]+3.0*scalarField[  ijk  ]-scalarField[ ijkp2 ]) ) -
                             ( fmaxf(0.0,w_cf[  ijk  ])*(6.0*scalarField[ ijkm1 ]+3.0*scalarField[  ijk  ]-scalarField[ ijkm2 ])+
                               fminf(0.0,w_cf[  ijk  ])*(6.0*scalarField[  ijk  ]+3.0*scalarField[ ijkm1 ]-scalarField[ ijkp1 ]) ) );
  scalarFadv[ijk] = scalarFadv[ijk] -invD_Jac_d[ijk]*(DscalarDx+DscalarDy+DscalarDz);

} //end cudaDevice_QUICKDivAdvFlux(

/*----->>>>> __device__ void  cudaDevice_HYB34DivAdvFlux();  --------------------------------------------------
* This is the cuda version of the hydro_coreHYB34DivAdvFlx routine from the HYDRO_CORE module
*/ 
__device__ void cudaDevice_HYB34DivAdvFlux(float* scalarField, float* scalarFadv,
                                              float* u_cf, float* v_cf, float* w_cf, float b_hyb_p, float* invD_Jac_d){

  int i,j,k;
  int ijk,im1jk,ijm1k,ijkm1,ip1jk,ijp1k,ijkp1,im2jk,ijm2k,ijkm2,ip2jk,ijp2k,ijkp2;
  int iStride,jStride,kStride;
  float DscalarDx;
  float DscalarDy;
  float DscalarDz;
  float one_twelfth;
  float flxx_ipf,flxx_imf,flxy_jpf,flxy_jmf,flxz_kpf,flxz_kmf;

  one_twelfth = 1.0/12.0;

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
  ip1jk = (i+1)*iStride + j*jStride + k*kStride;
  ijp1k = i*iStride + (j+1)*jStride + k*kStride;
  ijkp1 = i*iStride + j*jStride + (k+1)*kStride;
  im2jk = (i-2)*iStride + j*jStride + k*kStride;
  ijm2k = i*iStride + (j-2)*jStride + k*kStride;
  ijkm2 = i*iStride + j*jStride + (k-2)*kStride;
  ip2jk = (i+2)*iStride + j*jStride + k*kStride;
  ijp2k = i*iStride + (j+2)*jStride + k*kStride;
  ijkp2 = i*iStride + j*jStride + (k+2)*kStride;

  flxx_ipf  = one_twelfth * ( 7.0*(scalarField[ ip1jk ]+scalarField[  ijk  ])-(scalarField[ ip2jk ]+scalarField[ im1jk ])+
                              (1.0-b_hyb_p)*copysign(1.0,u_cf[ ip1jk ])*((scalarField[ ip2jk ]-scalarField[ im1jk ])-3.0*(scalarField[ ip1jk ]-scalarField[  ijk  ])) );
  flxx_imf  = one_twelfth * ( 7.0*(scalarField[  ijk  ]+scalarField[ im1jk ])-(scalarField[ ip1jk ]+scalarField[ im2jk ])+
                              (1.0-b_hyb_p)*copysign(1.0,u_cf[  ijk  ])*((scalarField[ ip1jk ]-scalarField[ im2jk ])-3.0*(scalarField[  ijk  ]-scalarField[ im1jk ])) );
  flxy_jpf  = one_twelfth * ( 7.0*(scalarField[ ijp1k ]+scalarField[  ijk  ])-(scalarField[ ijp2k ]+scalarField[ ijm1k ])+
                              (1.0-b_hyb_p)*copysign(1.0,v_cf[ ijp1k ])*((scalarField[ ijp2k ]-scalarField[ ijm1k ])-3.0*(scalarField[ ijp1k ]-scalarField[  ijk  ])) );
  flxy_jmf  = one_twelfth * ( 7.0*(scalarField[  ijk  ]+scalarField[ ijm1k ])-(scalarField[ ijp1k ]+scalarField[ ijm2k ])+
                              (1.0-b_hyb_p)*copysign(1.0,v_cf[  ijk  ])*((scalarField[ ijp1k ]-scalarField[ ijm2k ])-3.0*(scalarField[  ijk  ]-scalarField[ ijm1k ])) );
  flxz_kpf  = one_twelfth * ( 7.0*(scalarField[ ijkp1 ]+scalarField[  ijk  ])-(scalarField[ ijkp2 ]+scalarField[ ijkm1 ])+
                              (1.0-b_hyb_p)*copysign(1.0,w_cf[ ijkp1 ])*((scalarField[ ijkp2 ]-scalarField[ ijkm1 ])-3.0*(scalarField[ ijkp1 ]-scalarField[  ijk  ])) );
  flxz_kmf  = one_twelfth * ( 7.0*(scalarField[  ijk  ]+scalarField[ ijkm1 ])-(scalarField[ ijkp1 ]+scalarField[ ijkm2 ])+
                              (1.0-b_hyb_p)*copysign(1.0,w_cf[  ijk  ])*((scalarField[ ijkp1 ]-scalarField[ ijkm2 ])-3.0*(scalarField[  ijk  ]-scalarField[ ijkm1 ])) );
  DscalarDx = u_cf[ ip1jk ]*flxx_ipf - u_cf[  ijk  ]*flxx_imf;
  DscalarDy = v_cf[ ijp1k ]*flxy_jpf - v_cf[  ijk  ]*flxy_jmf;
  DscalarDz = w_cf[ ijkp1 ]*flxz_kpf - w_cf[  ijk  ]*flxz_kmf;
  scalarFadv[ijk] = scalarFadv[ijk] -invD_Jac_d[ijk]*(DscalarDx+DscalarDy+DscalarDz);

} //end cudaDevice_HYB34DivAdvFlux()

/*----->>>>> __device__ void  cudaDevice_HYB56DivAdvFlux();  --------------------------------------------------
* This is the cuda version of the hydro_coreHYB56DivAdvFlx routine from the HYDRO_CORE module
*/
__device__ void cudaDevice_HYB56DivAdvFlux(float* scalarField, float* scalarFadv,
                                              float* u_cf, float* v_cf, float* w_cf, float b_hyb_p, float* invD_Jac_d){

  int i,j,k;
  int ijk,im1jk,ijm1k,ijkm1,ip1jk,ijp1k,ijkp1;
  int im2jk,ijm2k,ijkm2,ip2jk,ijp2k,ijkp2;
  int im3jk,ijm3k,ijkm3,ip3jk,ijp3k,ijkp3;
  int iStride,jStride,kStride;
  float DscalarDx;
  float DscalarDy;
  float DscalarDz;
  float one_sixtieth;
  float flxx_ipf,flxx_imf,flxy_jpf,flxy_jmf,flxz_kpf,flxz_kmf;

  one_sixtieth = 1.0/60.0;

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
  ip1jk = (i+1)*iStride + j*jStride + k*kStride;
  ijp1k = i*iStride + (j+1)*jStride + k*kStride;
  ijkp1 = i*iStride + j*jStride + (k+1)*kStride;
  im2jk = (i-2)*iStride + j*jStride + k*kStride;
  ijm2k = i*iStride + (j-2)*jStride + k*kStride;
  ijkm2 = i*iStride + j*jStride + (k-2)*kStride;
  ip2jk = (i+2)*iStride + j*jStride + k*kStride;
  ijp2k = i*iStride + (j+2)*jStride + k*kStride;
  ijkp2 = i*iStride + j*jStride + (k+2)*kStride;
  im3jk = (i-3)*iStride + j*jStride + k*kStride;
  ijm3k = i*iStride + (j-3)*jStride + k*kStride;
  ijkm3 = i*iStride + j*jStride + (k-3)*kStride;
  ip3jk = (i+3)*iStride + j*jStride + k*kStride;
  ijp3k = i*iStride + (j+3)*jStride + k*kStride;
  ijkp3 = i*iStride + j*jStride + (k+3)*kStride;

  flxx_ipf  = one_sixtieth * ( 37.0*(scalarField[ ip1jk ]+scalarField[  ijk  ])-8.0*(scalarField[ ip2jk ]+scalarField[ im1jk ])+(scalarField[ ip3jk ]+scalarField[ im2jk ])-
                              (1.0-b_hyb_p)*copysign(1.0,u_cf[ ip1jk ])*((scalarField[ ip3jk ]-scalarField[ im2jk ])-5.0*(scalarField[ ip2jk ]-scalarField[ im1jk ])+10.0*(scalarField[ ip1jk ]-scalarField[  ijk  ])) );
  flxx_imf  = one_sixtieth * ( 37.0*(scalarField[  ijk  ]+scalarField[ im1jk ])-8.0*(scalarField[ ip1jk ]+scalarField[ im2jk ])+(scalarField[ ip2jk ]+scalarField[ im3jk ])-
                              (1.0-b_hyb_p)*copysign(1.0,u_cf[  ijk  ])*((scalarField[ ip2jk ]-scalarField[ im3jk ])-5.0*(scalarField[ ip1jk ]-scalarField[ im2jk ])+10.0*(scalarField[  ijk  ]-scalarField[ im1jk ])) );
  flxy_jpf  = one_sixtieth * ( 37.0*(scalarField[ ijp1k ]+scalarField[  ijk  ])-8.0*(scalarField[ ijp2k ]+scalarField[ ijm1k ])+(scalarField[ ijp3k ]+scalarField[ ijm2k ])-
                              (1.0-b_hyb_p)*copysign(1.0,v_cf[ ijp1k ])*((scalarField[ ijp3k ]-scalarField[ ijm2k ])-5.0*(scalarField[ ijp2k ]-scalarField[ ijm1k ])+10.0*(scalarField[ ijp1k ]-scalarField[  ijk  ])) );
  flxy_jmf  = one_sixtieth * ( 37.0*(scalarField[  ijk  ]+scalarField[ ijm1k ])-8.0*(scalarField[ ijp1k ]+scalarField[ ijm2k ])+(scalarField[ ijp2k ]+scalarField[ ijm3k ])-
                              (1.0-b_hyb_p)*copysign(1.0,v_cf[  ijk  ])*((scalarField[ ijp2k ]-scalarField[ ijm3k ])-5.0*(scalarField[ ijp1k ]-scalarField[ ijm2k ])+10.0*(scalarField[  ijk  ]-scalarField[ ijm1k ])) );
  flxz_kpf  = one_sixtieth * ( 37.0*(scalarField[ ijkp1 ]+scalarField[  ijk  ])-8.0*(scalarField[ ijkp2 ]+scalarField[ ijkm1 ])+(scalarField[ ijkp3 ]+scalarField[ ijkm2 ])-
                              (1.0-b_hyb_p)*copysign(1.0,w_cf[ ijkp1 ])*((scalarField[ ijkp3 ]-scalarField[ ijkm2 ])-5.0*(scalarField[ ijkp2 ]-scalarField[ ijkm1 ])+10.0*(scalarField[ ijkp1 ]-scalarField[  ijk  ])) );
  flxz_kmf  = one_sixtieth * ( 37.0*(scalarField[  ijk  ]+scalarField[ ijkm1 ])-8.0*(scalarField[ ijkp1 ]+scalarField[ ijkm2 ])+(scalarField[ ijkp2 ]+scalarField[ ijkm3 ])-
                              (1.0-b_hyb_p)*copysign(1.0,w_cf[  ijk  ])*((scalarField[ ijkp2 ]-scalarField[ ijkm3 ])-5.0*(scalarField[ ijkp1 ]-scalarField[ ijkm2 ])+10.0*(scalarField[  ijk  ]-scalarField[ ijkm1 ])) );

  DscalarDx = u_cf[ ip1jk ]*flxx_ipf - u_cf[  ijk  ]*flxx_imf;
  DscalarDy = v_cf[ ijp1k ]*flxy_jpf - v_cf[  ijk  ]*flxy_jmf;
  DscalarDz = w_cf[ ijkp1 ]*flxz_kpf - w_cf[  ijk  ]*flxz_kmf;
  scalarFadv[ijk] = scalarFadv[ijk] -invD_Jac_d[ijk]*(DscalarDx+DscalarDy+DscalarDz);

} //end cudaDevice_HYB56DivAdvFlux(

/*----->>>>> __device__ void  cudaDevice_WENO3DivAdvFluxX();  -----------------------------------------------*/ 
__device__ void cudaDevice_WENO3DivAdvFluxX(float* scalarField, float* scalarFadv,float* u_cf, float* invD_Jac_d){

  int i,j,k;
  int ijk,im1jk,ip1jk,im2jk,ip2jk;
  int iStride,jStride,kStride;
  float DscalarDx;
  float gamma_1 = 1.0/3.0;
  float gamma_2 = 2.0/3.0;
  float tol = 1e-6;
  float fh_1,fh_2,beta_1,beta_2,w_1t,w_2t,w_1,w_2;
  float flxx_ipf_velP,flxx_ipf_velN,flxx_imf_velP,flxx_imf_velN;
  float flxx_ipf,flxx_imf;


  /*Establish necessary indices for spatial locality*/
  i = (blockIdx.x)*blockDim.x + threadIdx.x;
  j = (blockIdx.y)*blockDim.y + threadIdx.y;
  k = (blockIdx.z)*blockDim.z + threadIdx.z;

  iStride = (Ny_d+2*Nh_d)*(Nz_d+2*Nh_d);
  jStride = (Nz_d+2*Nh_d);
  kStride = 1;
  ijk = i*iStride + j*jStride + k*kStride;
  im1jk = (i-1)*iStride + j*jStride + k*kStride;
  ip1jk = (i+1)*iStride + j*jStride + k*kStride;
  im2jk = (i-2)*iStride + j*jStride + k*kStride;
  ip2jk = (i+2)*iStride + j*jStride + k*kStride;

  // NUMERICAL HIGH-ORDER FACE VALUE IN THE X-DIRECTION //
  // i+1/2 face (u > 0 case)
  fh_1 = -0.5*scalarField[ im1jk ]+1.5*scalarField[ ijk ];
  fh_2 = 0.5*scalarField[ ijk ]+0.5*scalarField[ ip1jk ];
  beta_1 = powf((scalarField[ ijk ]-scalarField[ im1jk ]),2.0);
  beta_2 = powf((scalarField[ ip1jk ]-scalarField[ ijk ]),2.0);
  w_1t = gamma_1/powf((tol+beta_1),2.0);
  w_2t = gamma_2/powf((tol+beta_2),2.0);
  w_1 = w_1t/(w_1t+w_2t);
  w_2 = w_2t/(w_1t+w_2t);
  flxx_ipf_velP = w_1*fh_1+w_2*fh_2;
  // i+1/2 face (u < 0 case)
  fh_1 = -0.5*scalarField[ ip2jk ]+1.5*scalarField[ ip1jk ];
  fh_2 = 0.5*scalarField[ ip1jk ]+0.5*scalarField[ ijk ];
  beta_1 = powf((scalarField[ ip1jk ]-scalarField[ ip2jk ]),2.0);
  beta_2 = powf((scalarField[ ijk ]-scalarField[ ip1jk ]),2.0);
  w_1t = gamma_1/powf((tol+beta_1),2.0);
  w_2t = gamma_2/powf((tol+beta_2),2.0);
  w_1 = w_1t/(w_1t+w_2t);
  w_2 = w_2t/(w_1t+w_2t);
  flxx_ipf_velN = w_1*fh_1+w_2*fh_2;
  // i+1/2 face (combined)
  flxx_ipf = 0.5*(1.0+copysign(1.0,u_cf[ ip1jk ]))*flxx_ipf_velP+0.5*(1.0-copysign(1.0,u_cf[ ip1jk ]))*flxx_ipf_velN;

  // i-1/2 face (u > 0 case)
  fh_1 = -0.5*scalarField[ im2jk ]+1.5*scalarField[ im1jk ];
  fh_2 = 0.5*scalarField[ im1jk ]+0.5*scalarField[ ijk ];
  beta_1 = powf((scalarField[ im1jk ]-scalarField[ im2jk ]),2.0);
  beta_2 = powf((scalarField[ ijk ]-scalarField[ im1jk ]),2.0);
  w_1t = gamma_1/powf((tol+beta_1),2.0);
  w_2t = gamma_2/powf((tol+beta_2),2.0);
  w_1 = w_1t/(w_1t+w_2t);
  w_2 = w_2t/(w_1t+w_2t);
  flxx_imf_velP = w_1*fh_1+w_2*fh_2;
  // i-1/2 face (u < 0 case)
  fh_1 = -0.5*scalarField[ ip1jk ]+1.5*scalarField[ ijk ];
  fh_2 = 0.5*scalarField[ ijk ]+0.5*scalarField[ im1jk ];
  beta_1 = powf((scalarField[ ijk ]-scalarField[ ip1jk ]),2.0);
  beta_2 = powf((scalarField[ im1jk ]-scalarField[ ijk ]),2.0);
  w_1t = gamma_1/powf((tol+beta_1),2.0);
  w_2t = gamma_2/powf((tol+beta_2),2.0);
  w_1 = w_1t/(w_1t+w_2t);
  w_2 = w_2t/(w_1t+w_2t);
  flxx_imf_velN = w_1*fh_1+w_2*fh_2;
  // i-1/2 face (combined)
  flxx_imf = 0.5*(1.0+copysign(1.0,u_cf[ ijk ]))*flxx_imf_velP+0.5*(1.0-copysign(1.0,u_cf[ ijk ]))*flxx_imf_velN;

  DscalarDx = u_cf[ ip1jk ]*flxx_ipf - u_cf[  ijk  ]*flxx_imf;
  scalarFadv[ijk] = scalarFadv[ijk] -invD_Jac_d[ijk]*DscalarDx;

} //end cudaDevice_WENO3DivAdvFluxX(

/*----->>>>> __device__ void  cudaDevice_WENO3DivAdvFluxY();  ------------------------------------------------*/ 
__device__ void cudaDevice_WENO3DivAdvFluxY(float* scalarField, float* scalarFadv,float* v_cf, float* invD_Jac_d){

  int i,j,k;
  int ijk,ijm1k,ijp1k,ijm2k,ijp2k;
  int iStride,jStride,kStride;
  float DscalarDy;
  float gamma_1 = 1.0/3.0;
  float gamma_2 = 2.0/3.0;
  float tol = 1e-6;
  float fh_1,fh_2,beta_1,beta_2,w_1t,w_2t,w_1,w_2;
  float flxy_jpf_velP,flxy_jpf_velN,flxy_jmf_velP,flxy_jmf_velN;
  float flxy_jpf,flxy_jmf;


  /*Establish necessary indices for spatial locality*/
  i = (blockIdx.x)*blockDim.x + threadIdx.x;
  j = (blockIdx.y)*blockDim.y + threadIdx.y;
  k = (blockIdx.z)*blockDim.z + threadIdx.z;

  iStride = (Ny_d+2*Nh_d)*(Nz_d+2*Nh_d);
  jStride = (Nz_d+2*Nh_d);
  kStride = 1;
  ijk = i*iStride + j*jStride + k*kStride;
  ijm1k = i*iStride + (j-1)*jStride + k*kStride;
  ijp1k = i*iStride + (j+1)*jStride + k*kStride;
  ijm2k = i*iStride + (j-2)*jStride + k*kStride;
  ijp2k = i*iStride + (j+2)*jStride + k*kStride;

  // NUMERICAL HIGH-ORDER FACE VALUE IN THE Y-DIRECTION //
  // j+1/2 face (v > 0 case)
  fh_1 = -0.5*scalarField[ ijm1k ]+1.5*scalarField[ ijk ];
  fh_2 = 0.5*scalarField[ ijk ]+0.5*scalarField[ ijp1k ];
  beta_1 = powf((scalarField[ ijk ]-scalarField[ ijm1k ]),2.0);
  beta_2 = powf((scalarField[ ijp1k ]-scalarField[ ijk ]),2.0);
  w_1t = gamma_1/powf((tol+beta_1),2.0);
  w_2t = gamma_2/powf((tol+beta_2),2.0);
  w_1 = w_1t/(w_1t+w_2t);
  w_2 = w_2t/(w_1t+w_2t);
  flxy_jpf_velP = w_1*fh_1+w_2*fh_2;
  // j+1/2 face (v < 0 case)
  fh_1 = -0.5*scalarField[ ijp2k ]+1.5*scalarField[ ijp1k ];
  fh_2 = 0.5*scalarField[ ijp1k ]+0.5*scalarField[ ijk ];
  beta_1 = powf((scalarField[ ijp1k ]-scalarField[ ijp2k ]),2.0);
  beta_2 = powf((scalarField[ ijk ]-scalarField[ ijp1k ]),2.0);
  w_1t = gamma_1/powf((tol+beta_1),2.0);
  w_2t = gamma_2/powf((tol+beta_2),2.0);
  w_1 = w_1t/(w_1t+w_2t);
  w_2 = w_2t/(w_1t+w_2t);
  flxy_jpf_velN = w_1*fh_1+w_2*fh_2;
  // j+1/2 face (combined)
  flxy_jpf = 0.5*(1.0+copysign(1.0,v_cf[ ijp1k ]))*flxy_jpf_velP+0.5*(1.0-copysign(1.0,v_cf[ ijp1k ]))*flxy_jpf_velN;

  // j-1/2 face (v > 0 case)
  fh_1 = -0.5*scalarField[ ijm2k ]+1.5*scalarField[ ijm1k ];
  fh_2 = 0.5*scalarField[ ijm1k ]+0.5*scalarField[ ijk ];
  beta_1 = powf((scalarField[ ijm1k ]-scalarField[ ijm2k ]),2.0);
  beta_2 = powf((scalarField[ ijk ]-scalarField[ ijm1k ]),2.0);
  w_1t = gamma_1/powf((tol+beta_1),2.0);
  w_2t = gamma_2/powf((tol+beta_2),2.0);
  w_1 = w_1t/(w_1t+w_2t);
  w_2 = w_2t/(w_1t+w_2t);
  flxy_jmf_velP = w_1*fh_1+w_2*fh_2;
  // j-1/2 face (v < 0 case)
  fh_1 = -0.5*scalarField[ ijp1k ]+1.5*scalarField[ ijk ];
  fh_2 = 0.5*scalarField[ ijk ]+0.5*scalarField[ ijm1k ];
  beta_1 = powf((scalarField[ ijk ]-scalarField[ ijp1k ]),2.0);
  beta_2 = powf((scalarField[ ijm1k ]-scalarField[ ijk ]),2.0);
  w_1t = gamma_1/powf((tol+beta_1),2.0);
  w_2t = gamma_2/powf((tol+beta_2),2.0);
  w_1 = w_1t/(w_1t+w_2t);
  w_2 = w_2t/(w_1t+w_2t);
  flxy_jmf_velN = w_1*fh_1+w_2*fh_2;
  // j-1/2 face (combined)
  flxy_jmf = 0.5*(1.0+copysign(1.0,v_cf[ ijk ]))*flxy_jmf_velP+0.5*(1.0-copysign(1.0,v_cf[ ijk ]))*flxy_jmf_velN;

  DscalarDy = v_cf[ ijp1k ]*flxy_jpf - v_cf[  ijk  ]*flxy_jmf;
  scalarFadv[ijk] = scalarFadv[ijk] -invD_Jac_d[ijk]*DscalarDy;

} //end cudaDevice_WENO3DivAdvFluxY(

/*----->>>>> __device__ void  cudaDevice_WENO3DivAdvFluxZ();  -------------------------------------------------*/ 
__device__ void cudaDevice_WENO3DivAdvFluxZ(float* scalarField, float* scalarFadv,float* w_cf, float* invD_Jac_d){

  int i,j,k;
  int ijk,ijkm1,ijkp1,ijkm2,ijkp2;
  int iStride,jStride,kStride;
  float DscalarDz;
  float gamma_1 = 1.0/3.0;
  float gamma_2 = 2.0/3.0;
  float tol = 1e-6;
  float fh_1,fh_2,beta_1,beta_2,w_1t,w_2t,w_1,w_2;
  float flxz_kpf_velP,flxz_kpf_velN,flxz_kmf_velP,flxz_kmf_velN;
  float flxz_kpf,flxz_kmf;


  /*Establish necessary indices for spatial locality*/
  i = (blockIdx.x)*blockDim.x + threadIdx.x;
  j = (blockIdx.y)*blockDim.y + threadIdx.y;
  k = (blockIdx.z)*blockDim.z + threadIdx.z;

  iStride = (Ny_d+2*Nh_d)*(Nz_d+2*Nh_d);
  jStride = (Nz_d+2*Nh_d);
  kStride = 1;
  ijk = i*iStride + j*jStride + k*kStride;
  ijkm1 = i*iStride + j*jStride + (k-1)*kStride;
  ijkp1 = i*iStride + j*jStride + (k+1)*kStride;
  ijkm2 = i*iStride + j*jStride + (k-2)*kStride;
  ijkp2 = i*iStride + j*jStride + (k+2)*kStride;

  // NUMERICAL HIGH-ORDER FACE VALUE IN THE Z-DIRECTION //
  // k+1/2 face (w > 0 case)
  fh_1 = -0.5*scalarField[ ijkm1 ]+1.5*scalarField[ ijk ];
  fh_2 = 0.5*scalarField[ ijk ]+0.5*scalarField[ ijkp1 ];
  beta_1 = powf((scalarField[ ijk ]-scalarField[ ijkm1 ]),2.0);
  beta_2 = powf((scalarField[ ijkp1 ]-scalarField[ ijk ]),2.0);
  w_1t = gamma_1/powf((tol+beta_1),2.0);
  w_2t = gamma_2/powf((tol+beta_2),2.0);
  w_1 = w_1t/(w_1t+w_2t);
  w_2 = w_2t/(w_1t+w_2t);
  flxz_kpf_velP = w_1*fh_1+w_2*fh_2;
  // k+1/2 face (w < 0 case)
  fh_1 = -0.5*scalarField[ ijkp2 ]+1.5*scalarField[ ijkp1 ];
  fh_2 = 0.5*scalarField[ ijkp1 ]+0.5*scalarField[ ijk ];
  beta_1 = powf((scalarField[ ijkp1 ]-scalarField[ ijkp2 ]),2.0);
  beta_2 = powf((scalarField[ ijk ]-scalarField[ ijkp1 ]),2.0);
  w_1t = gamma_1/powf((tol+beta_1),2.0);
  w_2t = gamma_2/powf((tol+beta_2),2.0);
  w_1 = w_1t/(w_1t+w_2t);
  w_2 = w_2t/(w_1t+w_2t);
  flxz_kpf_velN = w_1*fh_1+w_2*fh_2;
  // k+1/2 face (combined)
  flxz_kpf = 0.5*(1.0+copysign(1.0,w_cf[ ijkp1 ]))*flxz_kpf_velP+0.5*(1.0-copysign(1.0,w_cf[ ijkp1 ]))*flxz_kpf_velN;

  // k-1/2 face (w > 0 case)
  fh_1 = -0.5*scalarField[ ijkm2 ]+1.5*scalarField[ ijkm1 ];
  fh_2 = 0.5*scalarField[ ijkm1 ]+0.5*scalarField[ ijk ];
  beta_1 = powf((scalarField[ ijkm1 ]-scalarField[ ijkm2 ]),2.0);
  beta_2 = powf((scalarField[ ijk ]-scalarField[ ijkm1 ]),2.0);
  w_1t = gamma_1/powf((tol+beta_1),2.0);
  w_2t = gamma_2/powf((tol+beta_2),2.0);
  w_1 = w_1t/(w_1t+w_2t);
  w_2 = w_2t/(w_1t+w_2t);
  flxz_kmf_velP = w_1*fh_1+w_2*fh_2;
  // k-1/2 face (w < 0 case)
  fh_1 = -0.5*scalarField[ ijkp1 ]+1.5*scalarField[ ijk ];
  fh_2 = 0.5*scalarField[ ijk ]+0.5*scalarField[ ijkm1 ];
  beta_1 = powf((scalarField[ ijk ]-scalarField[ ijkp1 ]),2.0);
  beta_2 = powf((scalarField[ ijkm1 ]-scalarField[ ijk ]),2.0);
  w_1t = gamma_1/powf((tol+beta_1),2.0);
  w_2t = gamma_2/powf((tol+beta_2),2.0);
  w_1 = w_1t/(w_1t+w_2t);
  w_2 = w_2t/(w_1t+w_2t);
  flxz_kmf_velN = w_1*fh_1+w_2*fh_2;
  // k-1/2 face (combined)
  flxz_kmf = 0.5*(1.0+copysign(1.0,w_cf[ ijk ]))*flxz_kmf_velP+0.5*(1.0-copysign(1.0,w_cf[ ijk ]))*flxz_kmf_velN;

  DscalarDz = w_cf[ ijkp1 ]*flxz_kpf - w_cf[  ijk  ]*flxz_kmf;
  scalarFadv[ijk] = scalarFadv[ijk] -invD_Jac_d[ijk]*DscalarDz;

} //end cudaDevice_WENO3DivAdvFluxZ(


//#####-------------- WENO5 -----------------#######//
/*----->>>>> __device__ void  cudaDevice_WENO5DivAdvFluxX();  ------------------------------------------------*/ 
__device__ void cudaDevice_WENO5DivAdvFluxX(float* scalarField, float* scalarFadv,float* u_cf, float* invD_Jac_d){

  int i,j,k;
  int ijk,im1jk,ip1jk,im2jk,ip2jk,im3jk,ip3jk;
  int iStride,jStride,kStride;
  float DscalarDx;
  float gamma_1 = 1.0/10.0;
  float gamma_2 = 3.0/5.0;
  float gamma_3 = 3.0/10.0;
  float tol = 1e-6;
  float fh_1,fh_2,fh_3,beta_1,beta_2,beta_3,w_1t,w_2t,w_3t,w_1,w_2,w_3;
  float flxx_ipf_velP,flxx_ipf_velN,flxx_imf_velP,flxx_imf_velN;
  float flxx_ipf,flxx_imf;


  /*Establish necessary indices for spatial locality*/
  i = (blockIdx.x)*blockDim.x + threadIdx.x;
  j = (blockIdx.y)*blockDim.y + threadIdx.y;
  k = (blockIdx.z)*blockDim.z + threadIdx.z;

  iStride = (Ny_d+2*Nh_d)*(Nz_d+2*Nh_d);
  jStride = (Nz_d+2*Nh_d);
  kStride = 1;
  ijk = i*iStride + j*jStride + k*kStride;
  im1jk = (i-1)*iStride + j*jStride + k*kStride;
  ip1jk = (i+1)*iStride + j*jStride + k*kStride;
  im2jk = (i-2)*iStride + j*jStride + k*kStride;
  ip2jk = (i+2)*iStride + j*jStride + k*kStride;
  im3jk = (i-3)*iStride + j*jStride + k*kStride;
  ip3jk = (i+3)*iStride + j*jStride + k*kStride;

  // NUMERICAL HIGH-ORDER FACE VALUE IN THE X-DIRECTION //
  // i+1/2 face (u > 0 case)
  fh_1 = (1.0/3.0)*scalarField[ im2jk ]-(7.0/6.0)*scalarField[ im1jk ]+(11.0/6.0)*scalarField[ ijk ];
  fh_2 = -(1.0/6.0)*scalarField[ im1jk ]+(5.0/6.0)*scalarField[ ijk ]+(1.0/3.0)*scalarField[ ip1jk ];
  fh_3 = (1.0/3.0)*scalarField[ ijk ]+(5.0/6.0)*scalarField[ ip1jk ]-(1.0/6.0)*scalarField[ ip2jk ];
  beta_1 = (13.0/12.0)*powf((scalarField[ im2jk ]-2.0*scalarField[ im1jk ]+scalarField[ ijk ]),2.0) + (1.0/4.0)*powf((scalarField[ im2jk ]-4.0*scalarField[ im1jk ]+3.0*scalarField[ ijk ]),2.0);
  beta_2 = (13.0/12.0)*powf((scalarField[ im1jk ]-2.0*scalarField[ ijk ]+scalarField[ ip1jk ]),2.0) + (1.0/4.0)*powf((scalarField[ im1jk ]-scalarField[ ip1jk ]),2.0);
  beta_3 = (13.0/12.0)*powf((scalarField[ ijk ]-2.0*scalarField[ ip1jk ]+scalarField[ ip2jk ]),2.0) + (1.0/4.0)*powf((3.0*scalarField[ ijk ]-4.0*scalarField[ ip1jk ]+scalarField[ ip2jk ]),2.0);
  w_1t = gamma_1/powf((tol+beta_1),2.0);
  w_2t = gamma_2/powf((tol+beta_2),2.0);
  w_3t = gamma_3/powf((tol+beta_3),2.0);
  w_1 = w_1t/(w_1t+w_2t+w_3t);
  w_2 = w_2t/(w_1t+w_2t+w_3t);
  w_3 = w_3t/(w_1t+w_2t+w_3t);
  flxx_ipf_velP = w_1*fh_1+w_2*fh_2+w_3*fh_3;
  // i+1/2 face (u < 0 case)
  fh_1 = (1.0/3.0)*scalarField[ ip3jk ]-(7.0/6.0)*scalarField[ ip2jk ]+(11.0/6.0)*scalarField[ ip1jk ];
  fh_2 = -(1.0/6.0)*scalarField[ ip2jk ]+(5.0/6.0)*scalarField[ ip1jk ]+(1.0/3.0)*scalarField[ ijk ];
  fh_3 = (1.0/3.0)*scalarField[ ip1jk ]+(5.0/6.0)*scalarField[ ijk ]-(1.0/6.0)*scalarField[ im1jk ];
  beta_1 = (13.0/12.0)*powf((scalarField[ ip3jk ]-2.0*scalarField[ ip2jk ]+scalarField[ ip1jk ]),2.0) + (1.0/4.0)*powf((scalarField[ ip3jk ]-4.0*scalarField[ ip2jk ]+3.0*scalarField[ ip1jk ]),2.0);
  beta_2 = (13.0/12.0)*powf((scalarField[ ip2jk ]-2.0*scalarField[ ip1jk ]+scalarField[ ijk ]),2.0) + (1.0/4.0)*powf((scalarField[ ip2jk ]-scalarField[ ijk ]),2.0);
  beta_3 = (13.0/12.0)*powf((scalarField[ ip1jk ]-2.0*scalarField[ ijk ]+scalarField[ im1jk ]),2.0) + (1.0/4.0)*powf((3.0*scalarField[ ip1jk ]-4.0*scalarField[ ijk ]+scalarField[ im1jk ]),2.0);
  w_1t = gamma_1/powf((tol+beta_1),2.0);
  w_2t = gamma_2/powf((tol+beta_2),2.0);
  w_3t = gamma_3/powf((tol+beta_3),2.0);
  w_1 = w_1t/(w_1t+w_2t+w_3t);
  w_2 = w_2t/(w_1t+w_2t+w_3t);
  w_3 = w_3t/(w_1t+w_2t+w_3t);
  flxx_ipf_velN = w_1*fh_1+w_2*fh_2+w_3*fh_3;
  // i+1/2 face (combined)
  flxx_ipf = 0.5*(1.0+copysign(1.0,u_cf[ ip1jk ]))*flxx_ipf_velP+0.5*(1.0-copysign(1.0,u_cf[ ip1jk ]))*flxx_ipf_velN;

  // i-1/2 face (u > 0 case)
  fh_1 = (1.0/3.0)*scalarField[ im3jk ]-(7.0/6.0)*scalarField[ im2jk ]+(11.0/6.0)*scalarField[ im1jk ];
  fh_2 = -(1.0/6.0)*scalarField[ im2jk ]+(5.0/6.0)*scalarField[ im1jk ]+(1.0/3.0)*scalarField[ ijk ];
  fh_3 = (1.0/3.0)*scalarField[ im1jk ]+(5.0/6.0)*scalarField[ ijk ]-(1.0/6.0)*scalarField[ ip1jk ];
  beta_1 = (13.0/12.0)*powf((scalarField[ im3jk ]-2.0*scalarField[ im2jk ]+scalarField[ im1jk ]),2.0) + (1.0/4.0)*powf((scalarField[ im3jk ]-4.0*scalarField[ im2jk ]+3.0*scalarField[ im1jk ]),2.0);
  beta_2 = (13.0/12.0)*powf((scalarField[ im2jk ]-2.0*scalarField[ im1jk ]+scalarField[ ijk ]),2.0) + (1.0/4.0)*powf((scalarField[ im2jk ]-scalarField[ ijk ]),2.0);
  beta_3 = (13.0/12.0)*powf((scalarField[ im1jk ]-2.0*scalarField[ ijk ]+scalarField[ ip1jk ]),2.0) + (1.0/4.0)*powf((3.0*scalarField[ im1jk ]-4.0*scalarField[ ijk ]+scalarField[ ip1jk ]),2.0);
  w_1t = gamma_1/powf((tol+beta_1),2.0);
  w_2t = gamma_2/powf((tol+beta_2),2.0);
  w_3t = gamma_3/powf((tol+beta_3),2.0);
  w_1 = w_1t/(w_1t+w_2t+w_3t);
  w_2 = w_2t/(w_1t+w_2t+w_3t);
  w_3 = w_3t/(w_1t+w_2t+w_3t);
  flxx_imf_velP = w_1*fh_1+w_2*fh_2+w_3*fh_3;
  // i-1/2 face (u < 0 case)
  fh_1 = (1.0/3.0)*scalarField[ ip2jk ]-(7.0/6.0)*scalarField[ ip1jk ]+(11.0/6.0)*scalarField[ ijk ];
  fh_2 = -(1.0/6.0)*scalarField[ ip1jk ]+(5.0/6.0)*scalarField[ ijk ]+(1.0/3.0)*scalarField[ im1jk ];
  fh_3 = (1.0/3.0)*scalarField[ ijk ]+(5.0/6.0)*scalarField[ im1jk ]-(1.0/6.0)*scalarField[ im2jk ];
  beta_1 = (13.0/12.0)*powf((scalarField[ ip2jk ]-2.0*scalarField[ ip1jk ]+scalarField[ ijk ]),2.0) + (1.0/4.0)*powf((scalarField[ ip2jk ]-4.0*scalarField[ ip1jk ]+3.0*scalarField[ ijk ]),2.0);
  beta_2 = (13.0/12.0)*powf((scalarField[ ip1jk ]-2.0*scalarField[ ijk ]+scalarField[ im1jk ]),2.0) + (1.0/4.0)*powf((scalarField[ ip1jk ]-scalarField[ im1jk ]),2.0);
  beta_3 = (13.0/12.0)*powf((scalarField[ ijk ]-2.0*scalarField[ im1jk ]+scalarField[ im2jk ]),2.0) + (1.0/4.0)*powf((3.0*scalarField[ ijk ]-4.0*scalarField[ im1jk ]+scalarField[ im2jk ]),2.0);
  w_1t = gamma_1/powf((tol+beta_1),2.0);
  w_2t = gamma_2/powf((tol+beta_2),2.0);
  w_3t = gamma_3/powf((tol+beta_3),2.0);
  w_1 = w_1t/(w_1t+w_2t+w_3t);
  w_2 = w_2t/(w_1t+w_2t+w_3t);
  w_3 = w_3t/(w_1t+w_2t+w_3t);
  flxx_imf_velN = w_1*fh_1+w_2*fh_2+w_3*fh_3;
  // i-1/2 face (combined)
  flxx_imf = 0.5*(1.0+copysign(1.0,u_cf[ ijk ]))*flxx_imf_velP+0.5*(1.0-copysign(1.0,u_cf[ ijk ]))*flxx_imf_velN;

  DscalarDx = u_cf[ ip1jk ]*flxx_ipf - u_cf[  ijk  ]*flxx_imf;
  scalarFadv[ijk] = scalarFadv[ijk] -invD_Jac_d[ijk]*DscalarDx;

} //end cudaDevice_WENO5DivAdvFluxX(

/*----->>>>> __device__ void  cudaDevice_WENO5DivAdvFluxY();  ------------------------------------------------*/ 
__device__ void cudaDevice_WENO5DivAdvFluxY(float* scalarField, float* scalarFadv,float* v_cf, float* invD_Jac_d){

  int i,j,k;
  int ijk,ijm1k,ijp1k,ijm2k,ijp2k,ijm3k,ijp3k;
  int iStride,jStride,kStride;
  float DscalarDy;
  float gamma_1 = 1.0/10.0;
  float gamma_2 = 3.0/5.0;
  float gamma_3 = 3.0/10.0;
  float tol = 1e-6;
  float fh_1,fh_2,fh_3,beta_1,beta_2,beta_3,w_1t,w_2t,w_3t,w_1,w_2,w_3;
  float flxy_jpf_velP,flxy_jpf_velN,flxy_jmf_velP,flxy_jmf_velN;
  float flxy_jpf,flxy_jmf;


  /*Establish necessary indices for spatial locality*/
  i = (blockIdx.x)*blockDim.x + threadIdx.x;
  j = (blockIdx.y)*blockDim.y + threadIdx.y;
  k = (blockIdx.z)*blockDim.z + threadIdx.z;

  iStride = (Ny_d+2*Nh_d)*(Nz_d+2*Nh_d);
  jStride = (Nz_d+2*Nh_d);
  kStride = 1;
  ijk = i*iStride + j*jStride + k*kStride;
  ijm1k = i*iStride + (j-1)*jStride + k*kStride;
  ijp1k = i*iStride + (j+1)*jStride + k*kStride;
  ijm2k = i*iStride + (j-2)*jStride + k*kStride;
  ijp2k = i*iStride + (j+2)*jStride + k*kStride;
  ijm3k = i*iStride + (j-3)*jStride + k*kStride;
  ijp3k = i*iStride + (j+3)*jStride + k*kStride;

  // NUMERICAL HIGH-ORDER FACE VALUE IN THE Y-DIRECTION //
  // j+1/2 face (v > 0 case)
  fh_1 = (1.0/3.0)*scalarField[ ijm2k ]-(7.0/6.0)*scalarField[ ijm1k ]+(11.0/6.0)*scalarField[ ijk ];
  fh_2 = -(1.0/6.0)*scalarField[ ijm1k ]+(5.0/6.0)*scalarField[ ijk ]+(1.0/3.0)*scalarField[ ijp1k ];
  fh_3 = (1.0/3.0)*scalarField[ ijk ]+(5.0/6.0)*scalarField[ ijp1k ]-(1.0/6.0)*scalarField[ ijp2k ];
  beta_1 = (13.0/12.0)*powf((scalarField[ ijm2k ]-2.0*scalarField[ ijm1k ]+scalarField[ ijk ]),2.0) + (1.0/4.0)*powf((scalarField[ ijm2k ]-4.0*scalarField[ ijm1k ]+3.0*scalarField[ ijk ]),2.0);
  beta_2 = (13.0/12.0)*powf((scalarField[ ijm1k ]-2.0*scalarField[ ijk ]+scalarField[ ijp1k ]),2.0) + (1.0/4.0)*powf((scalarField[ ijm1k ]-scalarField[ ijp1k ]),2.0);
  beta_3 = (13.0/12.0)*powf((scalarField[ ijk ]-2.0*scalarField[ ijp1k ]+scalarField[ ijp2k ]),2.0) + (1.0/4.0)*powf((3.0*scalarField[ ijk ]-4.0*scalarField[ ijp1k ]+scalarField[ ijp2k ]),2.0);
  w_1t = gamma_1/powf((tol+beta_1),2.0);
  w_2t = gamma_2/powf((tol+beta_2),2.0);
  w_3t = gamma_3/powf((tol+beta_3),2.0);
  w_1 = w_1t/(w_1t+w_2t+w_3t);
  w_2 = w_2t/(w_1t+w_2t+w_3t);
  w_3 = w_3t/(w_1t+w_2t+w_3t);
  flxy_jpf_velP = w_1*fh_1+w_2*fh_2+w_3*fh_3;
  // j+1/2 face (v < 0 case)
  fh_1 = (1.0/3.0)*scalarField[ ijp3k ]-(7.0/6.0)*scalarField[ ijp2k ]+(11.0/6.0)*scalarField[ ijp1k ];
  fh_2 = -(1.0/6.0)*scalarField[ ijp2k ]+(5.0/6.0)*scalarField[ ijp1k ]+(1.0/3.0)*scalarField[ ijk ];
  fh_3 = (1.0/3.0)*scalarField[ ijp1k ]+(5.0/6.0)*scalarField[ ijk ]-(1.0/6.0)*scalarField[ ijm1k ];
  beta_1 = (13.0/12.0)*powf((scalarField[ ijp3k ]-2.0*scalarField[ ijp2k ]+scalarField[ ijp1k ]),2.0) + (1.0/4.0)*powf((scalarField[ ijp3k ]-4.0*scalarField[ ijp2k ]+3.0*scalarField[ ijp1k ]),2.0);
  beta_2 = (13.0/12.0)*powf((scalarField[ ijp2k ]-2.0*scalarField[ ijp1k ]+scalarField[ ijk ]),2.0) + (1.0/4.0)*powf((scalarField[ ijp2k ]-scalarField[ ijk ]),2.0);
  beta_3 = (13.0/12.0)*powf((scalarField[ ijp1k ]-2.0*scalarField[ ijk ]+scalarField[ ijm1k ]),2.0) + (1.0/4.0)*powf((3.0*scalarField[ ijp1k ]-4.0*scalarField[ ijk ]+scalarField[ ijm1k ]),2.0);
  w_1t = gamma_1/powf((tol+beta_1),2.0);
  w_2t = gamma_2/powf((tol+beta_2),2.0);
  w_3t = gamma_3/powf((tol+beta_3),2.0);
  w_1 = w_1t/(w_1t+w_2t+w_3t);
  w_2 = w_2t/(w_1t+w_2t+w_3t);
  w_3 = w_3t/(w_1t+w_2t+w_3t);
  flxy_jpf_velN = w_1*fh_1+w_2*fh_2+w_3*fh_3;
  // j+1/2 face (combined)
  flxy_jpf = 0.5*(1.0+copysign(1.0,v_cf[ ijp1k ]))*flxy_jpf_velP+0.5*(1.0-copysign(1.0,v_cf[ ijp1k ]))*flxy_jpf_velN;

  // j-1/2 face (v > 0 case)
  fh_1 = (1.0/3.0)*scalarField[ ijm3k ]-(7.0/6.0)*scalarField[ ijm2k ]+(11.0/6.0)*scalarField[ ijm1k ];
  fh_2 = -(1.0/6.0)*scalarField[ ijm2k ]+(5.0/6.0)*scalarField[ ijm1k ]+(1.0/3.0)*scalarField[ ijk ];
  fh_3 = (1.0/3.0)*scalarField[ ijm1k ]+(5.0/6.0)*scalarField[ ijk ]-(1.0/6.0)*scalarField[ ijp1k ];
  beta_1 = (13.0/12.0)*powf((scalarField[ ijm3k ]-2.0*scalarField[ ijm2k ]+scalarField[ ijm1k ]),2.0) + (1.0/4.0)*powf((scalarField[ ijm3k ]-4.0*scalarField[ ijm2k ]+3.0*scalarField[ ijm1k ]),2.0);
  beta_2 = (13.0/12.0)*powf((scalarField[ ijm2k ]-2.0*scalarField[ ijm1k ]+scalarField[ ijk ]),2.0) + (1.0/4.0)*powf((scalarField[ ijm2k ]-scalarField[ ijk ]),2.0);
  beta_3 = (13.0/12.0)*powf((scalarField[ ijm1k ]-2.0*scalarField[ ijk ]+scalarField[ ijp1k ]),2.0) + (1.0/4.0)*powf((3.0*scalarField[ ijm1k ]-4.0*scalarField[ ijk ]+scalarField[ ijp1k ]),2.0);
  w_1t = gamma_1/powf((tol+beta_1),2.0);
  w_2t = gamma_2/powf((tol+beta_2),2.0);
  w_3t = gamma_3/powf((tol+beta_3),2.0);
  w_1 = w_1t/(w_1t+w_2t+w_3t);
  w_2 = w_2t/(w_1t+w_2t+w_3t);
  w_3 = w_3t/(w_1t+w_2t+w_3t);
  flxy_jmf_velP = w_1*fh_1+w_2*fh_2+w_3*fh_3;
  // j-1/2 face (v < 0 case)
  fh_1 = (1.0/3.0)*scalarField[ ijp2k ]-(7.0/6.0)*scalarField[ ijp1k ]+(11.0/6.0)*scalarField[ ijk ];
  fh_2 = -(1.0/6.0)*scalarField[ ijp1k ]+(5.0/6.0)*scalarField[ ijk ]+(1.0/3.0)*scalarField[ ijm1k ];
  fh_3 = (1.0/3.0)*scalarField[ ijk ]+(5.0/6.0)*scalarField[ ijm1k ]-(1.0/6.0)*scalarField[ ijm2k ];
  beta_1 = (13.0/12.0)*powf((scalarField[ ijp2k ]-2.0*scalarField[ ijp1k ]+scalarField[ ijk ]),2.0) + (1.0/4.0)*powf((scalarField[ ijp2k ]-4.0*scalarField[ ijp1k ]+3.0*scalarField[ ijk ]),2.0);
  beta_2 = (13.0/12.0)*powf((scalarField[ ijp1k ]-2.0*scalarField[ ijk ]+scalarField[ ijm1k ]),2.0) + (1.0/4.0)*powf((scalarField[ ijp1k ]-scalarField[ ijm1k ]),2.0);
  beta_3 = (13.0/12.0)*powf((scalarField[ ijk ]-2.0*scalarField[ ijm1k ]+scalarField[ ijm2k ]),2.0) + (1.0/4.0)*powf((3.0*scalarField[ ijk ]-4.0*scalarField[ ijm1k ]+scalarField[ ijm2k ]),2.0);
  w_1t = gamma_1/powf((tol+beta_1),2.0);
  w_2t = gamma_2/powf((tol+beta_2),2.0);
  w_3t = gamma_3/powf((tol+beta_3),2.0);
  w_1 = w_1t/(w_1t+w_2t+w_3t);
  w_2 = w_2t/(w_1t+w_2t+w_3t);
  w_3 = w_3t/(w_1t+w_2t+w_3t);
  flxy_jmf_velN = w_1*fh_1+w_2*fh_2+w_3*fh_3;
  // j-1/2 face (combined)
  flxy_jmf = 0.5*(1.0+copysign(1.0,v_cf[ ijk ]))*flxy_jmf_velP+0.5*(1.0-copysign(1.0,v_cf[ ijk ]))*flxy_jmf_velN;

  DscalarDy = v_cf[ ijp1k ]*flxy_jpf - v_cf[  ijk  ]*flxy_jmf;
  scalarFadv[ijk] = scalarFadv[ijk] -invD_Jac_d[ijk]*DscalarDy;

} //end cudaDevice_WENO5DivAdvFluxY(

/*----->>>>> __device__ void  cudaDevice_WENO5DivAdvFluxZ();  -------------------------------------------------*/ 
__device__ void cudaDevice_WENO5DivAdvFluxZ(float* scalarField, float* scalarFadv,float* w_cf, float* invD_Jac_d){

  int i,j,k;
  int ijk,ijkm1,ijkp1,ijkm2,ijkp2,ijkm3,ijkp3;
  int iStride,jStride,kStride;
  float DscalarDz;
  float gamma_1 = 1.0/10.0;
  float gamma_2 = 3.0/5.0;
  float gamma_3 = 3.0/10.0;
  float tol = 1e-6;
  float fh_1,fh_2,fh_3,beta_1,beta_2,beta_3,w_1t,w_2t,w_3t,w_1,w_2,w_3;
  float flxz_kpf_velP,flxz_kpf_velN,flxz_kmf_velP,flxz_kmf_velN;
  float flxz_kpf,flxz_kmf;


  /*Establish necessary indices for spatial locality*/
  i = (blockIdx.x)*blockDim.x + threadIdx.x;
  j = (blockIdx.y)*blockDim.y + threadIdx.y;
  k = (blockIdx.z)*blockDim.z + threadIdx.z;

  iStride = (Ny_d+2*Nh_d)*(Nz_d+2*Nh_d);
  jStride = (Nz_d+2*Nh_d);
  kStride = 1;
  ijk = i*iStride + j*jStride + k*kStride;
  ijkm1 = i*iStride + j*jStride + (k-1)*kStride;
  ijkp1 = i*iStride + j*jStride + (k+1)*kStride;
  ijkm2 = i*iStride + j*jStride + (k-2)*kStride;
  ijkp2 = i*iStride + j*jStride + (k+2)*kStride;
  ijkm3 = i*iStride + j*jStride + (k-3)*kStride;
  ijkp3 = i*iStride + j*jStride + (k+3)*kStride;

  // NUMERICAL HIGH-ORDER FACE VALUE IN THE Z-DIRECTION //
  // k+1/2 face (w > 0 case)
  fh_1 = (1.0/3.0)*scalarField[ ijkm2 ]-(7.0/6.0)*scalarField[ ijkm1 ]+(11.0/6.0)*scalarField[ ijk ];
  fh_2 = -(1.0/6.0)*scalarField[ ijkm1 ]+(5.0/6.0)*scalarField[ ijk ]+(1.0/3.0)*scalarField[ ijkp1 ];
  fh_3 = (1.0/3.0)*scalarField[ ijk ]+(5.0/6.0)*scalarField[ ijkp1 ]-(1.0/6.0)*scalarField[ ijkp2 ];
  beta_1 = (13.0/12.0)*powf((scalarField[ ijkm2 ]-2.0*scalarField[ ijkm1 ]+scalarField[ ijk ]),2.0) + (1.0/4.0)*powf((scalarField[ ijkm2 ]-4.0*scalarField[ ijkm1 ]+3.0*scalarField[ ijk ]),2.0);
  beta_2 = (13.0/12.0)*powf((scalarField[ ijkm1 ]-2.0*scalarField[ ijk ]+scalarField[ ijkp1 ]),2.0) + (1.0/4.0)*powf((scalarField[ ijkm1 ]-scalarField[ ijkp1 ]),2.0);
  beta_3 = (13.0/12.0)*powf((scalarField[ ijk ]-2.0*scalarField[ ijkp1 ]+scalarField[ ijkp2 ]),2.0) + (1.0/4.0)*powf((3.0*scalarField[ ijk ]-4.0*scalarField[ ijkp1 ]+scalarField[ ijkp2 ]),2.0);
  w_1t = gamma_1/powf((tol+beta_1),2.0);
  w_2t = gamma_2/powf((tol+beta_2),2.0);
  w_3t = gamma_3/powf((tol+beta_3),2.0);
  w_1 = w_1t/(w_1t+w_2t+w_3t);
  w_2 = w_2t/(w_1t+w_2t+w_3t);
  w_3 = w_3t/(w_1t+w_2t+w_3t);
  flxz_kpf_velP = w_1*fh_1+w_2*fh_2+w_3*fh_3;
  // k+1/2 face (w < 0 case)
  fh_1 = (1.0/3.0)*scalarField[ ijkp3 ]-(7.0/6.0)*scalarField[ ijkp2 ]+(11.0/6.0)*scalarField[ ijkp1 ];
  fh_2 = -(1.0/6.0)*scalarField[ ijkp2 ]+(5.0/6.0)*scalarField[ ijkp1 ]+(1.0/3.0)*scalarField[ ijk ];
  fh_3 = (1.0/3.0)*scalarField[ ijkp1 ]+(5.0/6.0)*scalarField[ ijk ]-(1.0/6.0)*scalarField[ ijkm1 ];
  beta_1 = (13.0/12.0)*powf((scalarField[ ijkp3 ]-2.0*scalarField[ ijkp2 ]+scalarField[ ijkp1 ]),2.0) + (1.0/4.0)*powf((scalarField[ ijkp3 ]-4.0*scalarField[ ijkp2 ]+3.0*scalarField[ ijkp1 ]),2.0);
  beta_2 = (13.0/12.0)*powf((scalarField[ ijkp2 ]-2.0*scalarField[ ijkp1 ]+scalarField[ ijk ]),2.0) + (1.0/4.0)*powf((scalarField[ ijkp2 ]-scalarField[ ijk ]),2.0);
  beta_3 = (13.0/12.0)*powf((scalarField[ ijkp1 ]-2.0*scalarField[ ijk ]+scalarField[ ijkm1 ]),2.0) + (1.0/4.0)*powf((3.0*scalarField[ ijkp1 ]-4.0*scalarField[ ijk ]+scalarField[ ijkm1 ]),2.0);
  w_1t = gamma_1/powf((tol+beta_1),2.0);
  w_2t = gamma_2/powf((tol+beta_2),2.0);
  w_3t = gamma_3/powf((tol+beta_3),2.0);
  w_1 = w_1t/(w_1t+w_2t+w_3t);
  w_2 = w_2t/(w_1t+w_2t+w_3t);
  w_3 = w_3t/(w_1t+w_2t+w_3t);
  flxz_kpf_velN = w_1*fh_1+w_2*fh_2+w_3*fh_3;
  // k+1/2 face (combined)
  flxz_kpf = 0.5*(1.0+copysign(1.0,w_cf[ ijkp1 ]))*flxz_kpf_velP+0.5*(1.0-copysign(1.0,w_cf[ ijkp1 ]))*flxz_kpf_velN;

  // k-1/2 face (w > 0 case)
  fh_1 = (1.0/3.0)*scalarField[ ijkm3 ]-(7.0/6.0)*scalarField[ ijkm2 ]+(11.0/6.0)*scalarField[ ijkm1 ];
  fh_2 = -(1.0/6.0)*scalarField[ ijkm2 ]+(5.0/6.0)*scalarField[ ijkm1 ]+(1.0/3.0)*scalarField[ ijk ];
  fh_3 = (1.0/3.0)*scalarField[ ijkm1 ]+(5.0/6.0)*scalarField[ ijk ]-(1.0/6.0)*scalarField[ ijkp1 ];
  beta_1 = (13.0/12.0)*powf((scalarField[ ijkm3 ]-2.0*scalarField[ ijkm2 ]+scalarField[ ijkm1 ]),2.0) + (1.0/4.0)*powf((scalarField[ ijkm3 ]-4.0*scalarField[ ijkm2 ]+3.0*scalarField[ ijkm1 ]),2.0);
  beta_2 = (13.0/12.0)*powf((scalarField[ ijkm2 ]-2.0*scalarField[ ijkm1 ]+scalarField[ ijk ]),2.0) + (1.0/4.0)*powf((scalarField[ ijkm2 ]-scalarField[ ijk ]),2.0);
  beta_3 = (13.0/12.0)*powf((scalarField[ ijkm1 ]-2.0*scalarField[ ijk ]+scalarField[ ijkp1 ]),2.0) + (1.0/4.0)*powf((3.0*scalarField[ ijkm1 ]-4.0*scalarField[ ijk ]+scalarField[ ijkp1 ]),2.0);
  w_1t = gamma_1/powf((tol+beta_1),2.0);
  w_2t = gamma_2/powf((tol+beta_2),2.0);
  w_3t = gamma_3/powf((tol+beta_3),2.0);
  w_1 = w_1t/(w_1t+w_2t+w_3t);
  w_2 = w_2t/(w_1t+w_2t+w_3t);
  w_3 = w_3t/(w_1t+w_2t+w_3t);
  flxz_kmf_velP = w_1*fh_1+w_2*fh_2+w_3*fh_3;
  // k-1/2 face (w < 0 case)
  fh_1 = (1.0/3.0)*scalarField[ ijkp2 ]-(7.0/6.0)*scalarField[ ijkp1 ]+(11.0/6.0)*scalarField[ ijk ];
  fh_2 = -(1.0/6.0)*scalarField[ ijkp1 ]+(5.0/6.0)*scalarField[ ijk ]+(1.0/3.0)*scalarField[ ijkm1 ];
  fh_3 = (1.0/3.0)*scalarField[ ijk ]+(5.0/6.0)*scalarField[ ijkm1 ]-(1.0/6.0)*scalarField[ ijkm2 ];
  beta_1 = (13.0/12.0)*powf((scalarField[ ijkp2 ]-2.0*scalarField[ ijkp1 ]+scalarField[ ijk ]),2.0) + (1.0/4.0)*powf((scalarField[ ijkp2 ]-4.0*scalarField[ ijkp1 ]+3.0*scalarField[ ijk ]),2.0);
  beta_2 = (13.0/12.0)*powf((scalarField[ ijkp1 ]-2.0*scalarField[ ijk ]+scalarField[ ijkm1 ]),2.0) + (1.0/4.0)*powf((scalarField[ ijkp1 ]-scalarField[ ijkm1 ]),2.0);
  beta_3 = (13.0/12.0)*powf((scalarField[ ijk ]-2.0*scalarField[ ijkm1 ]+scalarField[ ijkm2 ]),2.0) + (1.0/4.0)*powf((3.0*scalarField[ ijk ]-4.0*scalarField[ ijkm1 ]+scalarField[ ijkm2 ]),2.0);
  w_1t = gamma_1/powf((tol+beta_1),2.0);
  w_2t = gamma_2/powf((tol+beta_2),2.0);
  w_3t = gamma_3/powf((tol+beta_3),2.0);
  w_1 = w_1t/(w_1t+w_2t+w_3t);
  w_2 = w_2t/(w_1t+w_2t+w_3t);
  w_3 = w_3t/(w_1t+w_2t+w_3t);
  flxz_kmf_velN = w_1*fh_1+w_2*fh_2+w_3*fh_3;
  // k-1/2 face (combined)
  flxz_kmf = 0.5*(1.0+copysign(1.0,w_cf[ ijk ]))*flxz_kmf_velP+0.5*(1.0-copysign(1.0,w_cf[ ijk ]))*flxz_kmf_velN;

  DscalarDz = w_cf[ ijkp1 ]*flxz_kpf - w_cf[  ijk  ]*flxz_kmf;
  scalarFadv[ijk] = scalarFadv[ijk] -invD_Jac_d[ijk]*DscalarDz;

} //end cudaDevice_WENO5DivAdvFluxZ(

