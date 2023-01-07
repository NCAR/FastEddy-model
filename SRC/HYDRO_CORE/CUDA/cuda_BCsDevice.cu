/* FastEddy®: SRC/HYDRO_CORE/CUDA/cuda_BCsDevice.cu 
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
/*---BOUNDARY CONDITIONS*/
__constant__ int hydroBCs_d;       // hydro_core BC set selector
__constant__ float U_g_d;            /*Zonal (West-East) component of the geostrophic wind*/
__constant__ float V_g_d;            /*Meridional (South-North) component of the geostrophic wind*/
__constant__ float z_Ug_d;
__constant__ float z_Vg_d;
__constant__ float Ug_grad_d;
__constant__ float Vg_grad_d;


/*#################------------ BCS submodule function definitions ------------------#############*/
/*----->>>>> int cuda_BCsDeviceSetup();       ---------------------------------------------------------
Used to cudaMalloc and cudaMemcpy parameters and coordinate arrays, and for the BCS_CUDA submodule.
*/
extern "C" int cuda_BCsDeviceSetup(){
   int errorCode = CUDA_BCS_SUCCESS;

   cudaMemcpyToSymbol(hydroBCs_d, &hydroBCs, sizeof(int));
   cudaMemcpyToSymbol(U_g_d, &U_g, sizeof(float));
   cudaMemcpyToSymbol(V_g_d, &V_g, sizeof(float));
   cudaMemcpyToSymbol(z_Ug_d, &z_Ug, sizeof(float));
   cudaMemcpyToSymbol(z_Vg_d, &z_Vg, sizeof(float));
   cudaMemcpyToSymbol(Ug_grad_d, &Ug_grad, sizeof(float));
   cudaMemcpyToSymbol(Vg_grad_d, &Vg_grad, sizeof(float));

   return(errorCode);
} //end cuda_BCsDeviceSetup()

/*----->>>>> extern "C" int cuda_BCsDeviceCleanup();  -----------------------------------------------------------
Used to free all malloced memory by the BCS submodule.
*/

extern "C" int cuda_BCsDeviceCleanup(){
   int errorCode = CUDA_BCS_SUCCESS;

   /* Free any BCs submodule arrays */
   
   return(errorCode);

}//end cuda_moistureDeviceCleanup()

/*----->>>>> int cuda_hydroCoreDeviceSecondaryStageSetup(); ---------------------------------------------------------
*/
extern "C" int cuda_hydroCoreDeviceSecondaryStageSetup(float dt){
    int errorCode = CUDA_HYDRO_CORE_SUCCESS;
    
    /*Stubbed-out function for future configurations that need secondary stage setup*/ 

    return(errorCode);
}

__device__ void cudaDevice_HorizontalPeriodicXdirBCs(int fldIndx, float* scalarField){
  int i,j,k;
  int ijk,iTargjk;
  int iStride,jStride,kStride;

  /*Establish necessary indices for spatial locality*/
  i = (blockIdx.x)*blockDim.x + threadIdx.x;
  j = (blockIdx.y)*blockDim.y + threadIdx.y;
  k = (blockIdx.z)*blockDim.z + threadIdx.z;
  iStride = (Ny_d+2*Nh_d)*(Nz_d+2*Nh_d);
  jStride = (Nz_d+2*Nh_d);
  kStride = 1;
  ijk = i*iStride + j*jStride + k*kStride;

  if((j >= jMin_d-Nh_d)&&(j < jMax_d+Nh_d) &&
     (k >= kMin_d-Nh_d)&&(k < kMax_d+Nh_d) ){
     if((i >= iMin_d-Nh_d)&&(i<iMin_d)){
        iTargjk = (iMax_d-Nh_d+i)*iStride + j*jStride + k*kStride;
        scalarField[ijk] = scalarField[iTargjk];
     }else if((i >= iMax_d)&&(i<iMax_d+Nh_d)){
        iTargjk = (i-iMax_d+Nh_d)*iStride + j*jStride + k*kStride;
        scalarField[ijk] = scalarField[iTargjk];
     }//endif
  }//end if j>=jMin_dh
} //end cudaDevice_HorizontalPeriodicXdirBCs

__device__ void cudaDevice_HorizontalPeriodicYdirBCs(int fldIndx, float* scalarField){
  int i,j,k;
  int ijk,ijTargk;
  int iStride,jStride,kStride;

  /*Establish necessary indices for spatial locality*/
  i = (blockIdx.x)*blockDim.x + threadIdx.x;
  j = (blockIdx.y)*blockDim.y + threadIdx.y;
  k = (blockIdx.z)*blockDim.z + threadIdx.z;
  iStride = (Ny_d+2*Nh_d)*(Nz_d+2*Nh_d);
  jStride = (Nz_d+2*Nh_d);
  kStride = 1;
  ijk = i*iStride + j*jStride + k*kStride;

  if((i >= iMin_d-Nh_d)&&(i < iMax_d+Nh_d) &&
     (k >= kMin_d-Nh_d)&&(k < kMax_d+Nh_d) ){
     if((j >= jMin_d-Nh_d)&&(j < jMin_d)){
        ijTargk = i*iStride + (jMax_d-Nh_d+j)*jStride + k*kStride;
        scalarField[ijk] = scalarField[ijTargk];
     }else if((j >= jMax_d)&&(j<jMax_d+Nh_d)){
        ijTargk = i*iStride + (j-jMax_d+Nh_d)*jStride + k*kStride;
        scalarField[ijk] = scalarField[ijTargk];
     }//endif
  }//end if i>=iMin_dh

} //end cudaDevice_HorizontalPeriodicYdirBCs

__device__ void cudaDevice_VerticalAblBCs(int fldIndx, float* scalarField, float* scalarBaseStateField){

  int i,j,k;
  int ijk,ijkTarg;
  int iStride,jStride,kStride;

  /*Establish necessary indices for spatial locality*/
  i = (blockIdx.x)*blockDim.x + threadIdx.x;
  j = (blockIdx.y)*blockDim.y + threadIdx.y;
  k = (blockIdx.z)*blockDim.z + threadIdx.z;
  iStride = (Ny_d+2*Nh_d)*(Nz_d+2*Nh_d);
  jStride = (Nz_d+2*Nh_d);
  kStride = 1;
  ijk = i*iStride + j*jStride + k*kStride;

  if((i >= iMin_d-Nh_d)&&(i < iMax_d+Nh_d) &&
     (j >= jMin_d-Nh_d)&&(j < jMax_d+Nh_d) ){
     if((k >= kMin_d-Nh_d)&&(k < kMin_d)){
        if((fldIndx == U_INDX)||(fldIndx == V_INDX)){
          ijkTarg = i*iStride + j*jStride + (kMin_d)*kStride;
          scalarField[ijk] = scalarField[ijkTarg];
        }else {
          scalarField[ijk] = scalarBaseStateField[ijk];
        } //end if fldIndx
     }else if((k >= kMax_d)&&(k<kMax_d+Nh_d)){
           ijkTarg = i*iStride + j*jStride + (kMax_d-1)*kStride;
           scalarField[ijk] = scalarBaseStateField[ijk];
     }//end if(k>=kMin_d-Nh_d...) else if k>-kMax_d...
  }//end if i>=...j>= 

} //end cudaDevice_VerticalAblBCs

__device__ void cudaDevice_VerticalAblBCsMomentum(int fldIndxMom, float* scalarField, float* scalarBaseStateField, float* zPos_d){

  int i,j,k;
  int ijk,ijkTarg;
  int iStride,jStride,kStride;
  float zPos_ijk,MomBSval;

  /*Establish necessary indices for spatial locality*/
  i = (blockIdx.x)*blockDim.x + threadIdx.x;
  j = (blockIdx.y)*blockDim.y + threadIdx.y;
  k = (blockIdx.z)*blockDim.z + threadIdx.z;
  iStride = (Ny_d+2*Nh_d)*(Nz_d+2*Nh_d);
  jStride = (Nz_d+2*Nh_d);
  kStride = 1;
  ijk = i*iStride + j*jStride + k*kStride;

  if((i >= iMin_d-Nh_d)&&(i < iMax_d+Nh_d) &&
     (j >= jMin_d-Nh_d)&&(j < jMax_d+Nh_d) ){
     if((k >= kMin_d-Nh_d)&&(k < kMin_d)){
        ijkTarg = i*iStride + j*jStride + (kMin_d)*kStride;
        scalarField[ijk] = scalarField[ijkTarg];
     }else if((k >= kMax_d)&&(k<kMax_d+Nh_d)){
        ijkTarg = i*iStride + j*jStride + (kMax_d-1)*kStride;
        zPos_ijk = zPos_d[ijkTarg];
        cudaDevice_MomentumBS(fldIndxMom, zPos_ijk, &scalarBaseStateField[ijkTarg], &MomBSval);
        scalarField[ijk] = MomBSval;
     }//end if(k>=kMin_d-Nh_d...) else if k>-kMax_d...
  }//end if i>=...j>=

} //end cudaDevice_VerticalAblBCsMomentum

__device__ void cudaDevice_MomentumBS(int fldIndxMom, float zPos_ijk, float* rho_ijk, float* MomBSval){

  float z_g,vel_g,vel_g_grad;

  switch (fldIndxMom){
    case 1: // u
      z_g = z_Ug_d;
      vel_g = U_g_d;
      vel_g_grad = Ug_grad_d;
      break;
    case 2: // v
      z_g = z_Vg_d;
      vel_g = V_g_d;
      vel_g_grad = Vg_grad_d;
      break;
    default: // w
      z_g = 0.0;
      vel_g = 0.0;
      vel_g_grad = 0.0;
      break;
  }

  if (zPos_ijk < z_g){
    MomBSval[0] = vel_g*rho_ijk[0];
  } else{
    MomBSval[0] = (vel_g + vel_g_grad*(zPos_ijk-z_g))*rho_ijk[0];
  }

} // end cudaDevice_MomentumBS

__device__ void cudaDevice_VerticalAblZeroGradBCs(float* scalarField){

  int i,j,k;
  int ijk,ijkTarg;
  int iStride,jStride,kStride;

  /*Establish necessary indices for spatial locality*/
  i = (blockIdx.x)*blockDim.x + threadIdx.x;
  j = (blockIdx.y)*blockDim.y + threadIdx.y;
  k = (blockIdx.z)*blockDim.z + threadIdx.z;
  iStride = (Ny_d+2*Nh_d)*(Nz_d+2*Nh_d);
  jStride = (Nz_d+2*Nh_d);
  kStride = 1;
  ijk = i*iStride + j*jStride + k*kStride;

  if((i >= iMin_d-Nh_d)&&(i < iMax_d+Nh_d) &&
     (j >= jMin_d-Nh_d)&&(j < jMax_d+Nh_d) ){

     if((k >= kMin_d-Nh_d)&&(k < kMin_d)){
       ijkTarg = i*iStride + j*jStride + kMin_d*kStride;
       scalarField[ijk] = scalarField[ijkTarg];
     }else if((k >= kMax_d)&&(k<kMax_d+Nh_d)){
       ijkTarg = i*iStride + j*jStride + (kMax_d-1)*kStride;
       scalarField[ijk] = scalarField[ijkTarg];
     }

  } //end if i>=...j>=

} //end cudaDevice_VerticalAblZeroGradBCs
