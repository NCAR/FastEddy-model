/*---CANOPY MODEL*/
__constant__ int canopySelector_d;         /* canopy selector: 0=off, 1=on */
__constant__ int canopySkinOpt_d;          /* canopy selector to use additional skin friction effect on drag coefficient: 0=off, 1=on */
__constant__ float canopy_cd_d;            /* non-dimensional canopy drag coefficient cd coefficient */
__constant__ float canopy_lf_d;            /* representative canopy element length scale */
float *canopy_lad_d;          /* Base Address of memory containing leaf area density (LAD) field [m^{-1}] */

/*#################------------ CANOPY submodule function definitions ------------------#############*/
/*----->>>>> int cuda_canopyDeviceSetup();       ---------------------------------------------------------
 * Used to cudaMalloc and cudaMemcpy parameters and coordinate arrays, and for the CANOPY_CUDA submodule.
*/
extern "C" int cuda_canopyDeviceSetup(){
   int errorCode = CUDA_CANOPY_SUCCESS;
   int Nelems;

   Nelems = (Nxp+2*Nh)*(Nyp+2*Nh)*(Nzp+2*Nh);

   cudaMemcpyToSymbol(canopySelector_d, &canopySelector, sizeof(int));
   cudaMemcpyToSymbol(canopySkinOpt_d, &canopySkinOpt, sizeof(int));
   cudaMemcpyToSymbol(canopy_cd_d, &canopy_cd, sizeof(float));
   cudaMemcpyToSymbol(canopy_lf_d, &canopy_lf, sizeof(float));

   Nelems = (Nxp+2*Nh)*(Nyp+2*Nh)*(Nzp+2*Nh);
   fecuda_DeviceMalloc(Nelems*sizeof(float), &canopy_lad_d);
   cudaMemcpy(canopy_lad_d, canopy_lad, Nelems*sizeof(float), cudaMemcpyHostToDevice);

   return(errorCode);
} //end cuda_canopyDeviceSetup()

/*----->>>>> extern "C" int cuda_canopyDeviceCleanup();  -----------------------------------------------------------
Used to free all malloced memory by the CANOPY submodule.
*/

extern "C" int cuda_canopyDeviceCleanup(){
   int errorCode = CUDA_CANOPY_SUCCESS;

   /* Free any CANOPY submodule arrays */
   cudaFree(canopy_lad_d);

   return(errorCode);

}//end cuda_canopyDeviceCleanup()

__global__ void cudaDevice_hydroCoreUnitTestCompleteCanopy(float* hydroFlds_d, float* hydroRhoInv_d, float* canopy_lad_d, float* hydroFldsFrhs_d){

   int fldStride;

   fldStride = (Nx_d+2*Nh_d)*(Ny_d+2*Nh_d)*(Nz_d+2*Nh_d);

   cudaDevice_canopyMomDrag(&hydroRhoInv_d[0], &hydroFlds_d[fldStride*U_INDX], &hydroFlds_d[fldStride*V_INDX],
                            &hydroFlds_d[fldStride*W_INDX], &canopy_lad_d[0],
                            &hydroFldsFrhs_d[fldStride*U_INDX], &hydroFldsFrhs_d[fldStride*V_INDX],
                            &hydroFldsFrhs_d[fldStride*W_INDX]);

} // end cudaDevice_hydroCoreUnitTestCompleteCanopy()

/*----->>>>> __device__ void  cudaDevice_canopyMomDrag();  --------------------------------------------------
*/
__device__ void cudaDevice_canopyMomDrag(float* rhoInv, float* u, float* v, float* w, float* lad, float* Frhs_u, float* Frhs_v, float* Frhs_w){

  float cd_coeff;
  float csf_coeff;
  float u_ijk,v_ijk,w_ijk,U_ijk;
  float Re_l;
  float f_canopy_u;
  float f_canopy_v;
  float f_canopy_w;
  float nu_air = 1.5e-5; // reference kinematic viscosity of air (~ 290 K)

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

    cd_coeff = canopy_cd_d;
    u_ijk = u[ijk]*rhoInv[ijk];
    v_ijk = v[ijk]*rhoInv[ijk];
    w_ijk = w[ijk]*rhoInv[ijk];
    U_ijk = powf(powf(u_ijk,2.0)+powf(v_ijk,2.0)+powf(w_ijk,2.0),0.5);
    if(canopySkinOpt_d == 1){
      Re_l = U_ijk*nu_air/canopy_lf_d;
      csf_coeff = 1.328/powf(Re_l,0.5) + 2.326/Re_l;
      cd_coeff = cd_coeff + csf_coeff; 
    }

    f_canopy_u = -cd_coeff*lad[ijk]*u[ijk]*U_ijk;
    f_canopy_v = -cd_coeff*lad[ijk]*v[ijk]*U_ijk;
    f_canopy_w = -cd_coeff*lad[ijk]*w[ijk]*U_ijk;

    Frhs_u[ijk] = Frhs_u[ijk] + f_canopy_u;
    Frhs_v[ijk] = Frhs_v[ijk] + f_canopy_v;
    Frhs_w[ijk] = Frhs_w[ijk] + f_canopy_w;
  
    /* 
    if (i==iMin_d && j==jMin_d && (k<=kMin_d+10)){
      printf("cudaDevice_canopyMomDrag(): At (%d,%d,%d) cd_coeff,lad_ijk,f_canopy_u,f_canopy_v,f_canopy_w = %f,%f,%f,%f\n",i,j,k,cd_coeff,lad[ijk],f_canopy_u,f_canopy_v,f_canopy_w);
    }
    */

  }

} //end cudaDevice_canopyMomDrag

/*----->>>>> __device__ void  cudaDevice_canopySGSTKEtransfer();  --------------------------------------------------
*/
__device__ void cudaDevice_canopySGSTKEtransfer(float* rhoInv, float* u, float* v, float* w, 
                                                float* lad, float* sgstke, float* Frhs_sgstke, int sign_term){

  float cd_coeff;
  float u_ijk,v_ijk,w_ijk,U_ijk;
  float f_canopy_sgstke;

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

    cd_coeff = canopy_cd_d;
    u_ijk = u[ijk]*rhoInv[ijk];
    v_ijk = v[ijk]*rhoInv[ijk];
    w_ijk = w[ijk]*rhoInv[ijk];
    U_ijk = powf(powf(u_ijk,2.0)+powf(v_ijk,2.0)+powf(w_ijk,2.0),0.5);

    f_canopy_sgstke = sign_term*(8.0/3.0)*cd_coeff*lad[ijk]*sgstke[ijk]*U_ijk;

    Frhs_sgstke[ijk] = Frhs_sgstke[ijk] + f_canopy_sgstke;

    /*
    if (i==iMin_d && j==jMin_d){
      printf("cudaDevice_canopySGSTKEtransfer(): At (%d,%d,%d) cd_coeff,f_canopy_sgstke = %f,%f\n",i,j,k,cd_coeff,f_canopy_sgstke);
    }
    */

  }

} //end cudaDevice_canopySGSTKEtransfer

/*----->>>>> __device__ void  cudaDevice_sgstkeLengthScaleLF();  --------------------------------------------------
*/
__device__ void cudaDevice_sgstkeLengthScaleLF(float* sgstke_ls){

  int i,j,k,ijk;
  int iStride,jStride,kStride;

  i = (blockIdx.x)*blockDim.x + threadIdx.x;
  j = (blockIdx.y)*blockDim.y + threadIdx.y;
  k = (blockIdx.z)*blockDim.z + threadIdx.z;
  iStride = (Ny_d+2*Nh_d)*(Nz_d+2*Nh_d);
  jStride = (Nz_d+2*Nh_d);
  kStride = 1;
  ijk = i*iStride + j*jStride + k*kStride;

  if((i >= iMin_d-1)&&(i < iMax_d+1) && (j >= jMin_d-1)&&(j < jMax_d+1) && (k >= kMin_d-1)&&(k < kMax_d+1)){

    sgstke_ls[ijk] = canopy_lf_d;

    /*
    if (i==iMin_d && j==jMin_d){
      printf("cudaDevice_sgstkeLengthScaleLF(): sgstke_ls(%d,%d,%d)= %f\n",i,j,k,sgstke_ls[ijk]);
    }
    */

  } // if (within the computational domain...) 

} //end cudaDevice_sgstkeLengthScaleLF

/*----->>>>> __device__ void  cudaDevice_canopySGSTKEwakeprod();  --------------------------------------------------
*/
__device__ void cudaDevice_canopySGSTKEwakeprod(float* rhoInv, float* u, float* v, float* w, 
                                                float* lad, float* Frhs_sgstke){

  float cd_coeff;
  float u_ijk,v_ijk,w_ijk,U_ijk;
  float f_canopy_sgstke;

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

    cd_coeff = canopy_cd_d;
    u_ijk = u[ijk]*rhoInv[ijk];
    v_ijk = v[ijk]*rhoInv[ijk];
    w_ijk = w[ijk]*rhoInv[ijk];
    U_ijk = powf(powf(u_ijk,2.0)+powf(v_ijk,2.0)+powf(w_ijk,2.0),0.5);

    f_canopy_sgstke = cd_coeff*lad[ijk]*powf(U_ijk,3.0)/rhoInv[ijk];

    Frhs_sgstke[ijk] = Frhs_sgstke[ijk] + f_canopy_sgstke;

    /*
    if (i==iMin_d && j==jMin_d){
      printf("cudaDevice_canopySGSTKEwakeprod(): At (%d,%d,%d) cd_coeff,f_canopy_sgstke = %f,%f\n",i,j,k,cd_coeff,f_canopy_sgstke);
    }
    */

  }

} //end cudaDevice_canopySGSTKEwakeprod
