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

/*##############------------------- BCS submodule variable definitions ---------------------#################*/
/*---BOUNDARY CONDITIONS parameters*/
__constant__ int hydroBCs_d;        // hydro_core BC set selector
__constant__ float dtBdyPlaneBCs_d; //delta in time (seconds) between update BdyPlanes sets
__constant__ int BdyUpdateSteps_d;  // number of timesteps between update of BdyPlanes sets
__constant__ int nBndyVars_d;       // number of Bndy plane variables
__constant__ int nSurfBndyVars_d;   // number of surface Bndy plane variables

__constant__ float U_g_d;            /*Zonal (West-East) component of the geostrophic wind*/
__constant__ float V_g_d;            /*Meridional (South-North) component of the geostrophic wind*/
__constant__ float z_Ug_d;
__constant__ float z_Vg_d;
__constant__ float Ug_grad_d;
__constant__ float Vg_grad_d;

/*---BOUNDARY CONDITIONS arrays*/
float *XZBdyPlanes_d;          /*Base Adress of memory block for lateral-BC XZ-planes (per rank domain)*/
float *YZBdyPlanes_d;          /*Base Adress of memory block for lateral-BC YZ-planes (per rank domain)*/
float *XYBdyPlanes_d;          /*Base Adress of memory block for surface/ceiling--BC XY-planes (per rank domain)*/
float *XZBdyPlanesNext_d;      /*Base Adress of memory block for lateral-BC XZ-planes (per rank domain)*/
float *YZBdyPlanesNext_d;      /*Base Adress of memory block for lateral-BC YZ-planes (per rank domain)*/
float *XYBdyPlanesNext_d;      /*Base Adress of memory block for surface/ceiling--BC XY-planes (per rank domain)*/
float *XZBdyPlanesBuffer_d;    /*Base Adress of memory block for lateral-BC XZ-planes (per rank domain)*/
float *YZBdyPlanesBuffer_d;    /*Base Adress of memory block for lateral-BC YZ-planes (per rank domain)*/
float *XYBdyPlanesBuffer_d;    /*Base Adress of memory block for surface/ceiling--BC XY-planes (per rank domain)*/
float *SURFBdyPlanes_d;        /*Base Adress of memory block for surfVar-BC XY-planes (per rank domain)*/
float *SURFBdyPlanesNext_d;    /*Base Adress of memory block for surfVar-BC XY-planes (per rank domain)*/
float *SURFBdyPlanesBuffer_d;  /*Base Adress of memory block for surfVar-BC XY-planes (per rank domain)*/

/*#################------------ BCS submodule function definitions ------------------#############*/
/*----->>>>> int cuda_BCsDeviceSetup();       ---------------------------------------------------------
* Used to cudaMalloc and cudaMemcpy parameters and coordinate arrays, and for the BCS_CUDA submodule.
*/
extern "C" int cuda_BCsDeviceSetup(){
   int errorCode = CUDA_BCS_SUCCESS;

   /*Initialize Constants (Symbols)*/
   cudaMemcpyToSymbol(hydroBCs_d, &hydroBCs, sizeof(int));
   if(hydroBCs==1){ //Using LAD BCs
     cudaMemcpyToSymbol(nBndyVars_d, &nBndyVars, sizeof(int));
     cudaMemcpyToSymbol(nSurfBndyVars_d, &nSurfBndyVars, sizeof(int));
     cudaMemcpyToSymbol(dtBdyPlaneBCs_d, &dtBdyPlaneBCs, sizeof(float));
   }//end if hydroBCs == 1
   cudaMemcpyToSymbol(U_g_d, &U_g, sizeof(float));
   cudaMemcpyToSymbol(V_g_d, &V_g, sizeof(float));
   cudaMemcpyToSymbol(z_Ug_d, &z_Ug, sizeof(float));
   cudaMemcpyToSymbol(z_Vg_d, &z_Vg, sizeof(float));
   cudaMemcpyToSymbol(Ug_grad_d, &Ug_grad, sizeof(float));
   cudaMemcpyToSymbol(Vg_grad_d, &Vg_grad, sizeof(float));
   /*Allocate arrays*/
   if(hydroBCs==1){ //Using LAD BCs
     if((rankYid == 0)||(rankYid == numProcsY-1)){
       fecuda_DeviceMalloc((size_t)(2*nBndyVars*(Nxp+2*Nh)*(Nzp+2*Nh)), &XZBdyPlanes_d);
       fecuda_DeviceMalloc((size_t)(2*nBndyVars*(Nxp+2*Nh)*(Nzp+2*Nh)), &XZBdyPlanesNext_d);
       fecuda_DeviceMalloc((size_t)(2*nBndyVars*(Nxp+2*Nh)*(Nzp+2*Nh)), &XZBdyPlanesBuffer_d);
     }
     if((rankXid == 0)||(rankXid == numProcsX-1)){
       fecuda_DeviceMalloc((size_t)(2*nBndyVars*(Nyp+2*Nh)*(Nzp+2*Nh)), &YZBdyPlanes_d);
       fecuda_DeviceMalloc((size_t)(2*nBndyVars*(Nyp+2*Nh)*(Nzp+2*Nh)), &YZBdyPlanesNext_d);
       fecuda_DeviceMalloc((size_t)(2*nBndyVars*(Nyp+2*Nh)*(Nzp+2*Nh)), &YZBdyPlanesBuffer_d);
     }
     fecuda_DeviceMalloc((size_t)(2*nBndyVars*(Nxp+2*Nh)*(Nyp+2*Nh)), &XYBdyPlanes_d);
     fecuda_DeviceMalloc((size_t)(2*nBndyVars*(Nxp+2*Nh)*(Nyp+2*Nh)), &XYBdyPlanesNext_d);
     fecuda_DeviceMalloc((size_t)(2*nBndyVars*(Nxp+2*Nh)*(Nyp+2*Nh)), &XYBdyPlanesBuffer_d);
     if(surflayerSelector == 3){
       fecuda_DeviceMalloc((size_t)(nSurfBndyVars*(Nxp+2*Nh)*(Nyp+2*Nh)), &SURFBdyPlanes_d);
       fecuda_DeviceMalloc((size_t)(nSurfBndyVars*(Nxp+2*Nh)*(Nyp+2*Nh)), &SURFBdyPlanesNext_d);
       fecuda_DeviceMalloc((size_t)(nSurfBndyVars*(Nxp+2*Nh)*(Nyp+2*Nh)), &SURFBdyPlanesBuffer_d);
     }
   }//end if hydroBCs == 1
  
   /*Initialize arrays*/
   if(hydroBCs==1){ //Using LAD BCs
     //Special at initialization !!! The device-next<-cpu-current is in fact truly "next" because of pointer cycling at the cpu level
     // and vice versa for device-current<-cpu-next
     if((rankYid == 0)||(rankYid == numProcsY-1)){
      cudaMemcpy(XZBdyPlanesNext_d, XZBdyPlanes, 2*nBndyVars*(Nxp+2*Nh)*(Nzp+2*Nh)*sizeof(float), cudaMemcpyHostToDevice);
     }
     if((rankXid == 0)||(rankXid == numProcsX-1)){
      cudaMemcpy(YZBdyPlanesNext_d, YZBdyPlanes, 2*nBndyVars*(Nyp+2*Nh)*(Nzp+2*Nh)*sizeof(float), cudaMemcpyHostToDevice);
     }
     cudaMemcpy(XYBdyPlanesNext_d, XYBdyPlanes, 2*nBndyVars*(Nxp+2*Nh)*(Nyp+2*Nh)*sizeof(float), cudaMemcpyHostToDevice);
     //Special at initialization !!! Next actually has the first of the two time instances of BCs...
     if((rankYid == 0)||(rankYid == numProcsY-1)){
      cudaMemcpy(XZBdyPlanes_d, XZBdyPlanesNext, 2*nBndyVars*(Nxp+2*Nh)*(Nzp+2*Nh)*sizeof(float), cudaMemcpyHostToDevice);
     }
     if((rankXid == 0)||(rankXid == numProcsX-1)){
      cudaMemcpy(YZBdyPlanes_d, YZBdyPlanesNext, 2*nBndyVars*(Nyp+2*Nh)*(Nzp+2*Nh)*sizeof(float), cudaMemcpyHostToDevice);
     }
     cudaMemcpy(XYBdyPlanes_d, XYBdyPlanesNext, 2*nBndyVars*(Nxp+2*Nh)*(Nyp+2*Nh)*sizeof(float), cudaMemcpyHostToDevice);
     if(surflayerSelector == 3){
       cudaMemcpy(SURFBdyPlanesNext_d, SURFBdyPlanes, nSurfBndyVars*(Nxp+2*Nh)*(Nyp+2*Nh)*sizeof(float), cudaMemcpyHostToDevice);
       cudaMemcpy(SURFBdyPlanes_d, SURFBdyPlanesNext, nSurfBndyVars*(Nxp+2*Nh)*(Nyp+2*Nh)*sizeof(float), cudaMemcpyHostToDevice);
     }
   }//end if hydroBCs == 1
  
   gpuErrchk( cudaPeekAtLastError() ); /*Check for errors in the cudaMemCpy calls*/
   gpuErrchk( cudaDeviceSynchronize() );

   return(errorCode);
} //end cuda_BCsDeviceSetup()

/*----->>>>> extern "C" int cuda_BCsDeviceCleanup();  -----------------------------------------------------------
* Used to free all malloced memory by the BCS submodule.
*/

extern "C" int cuda_BCsDeviceCleanup(){
   int errorCode = CUDA_BCS_SUCCESS;

   /* Free any BCs submodule arrays */
    if(hydroBCs==1){ //Using LAD BCs
     if((rankYid == 0)||(rankYid == numProcsY-1)){
       cudaFree(XZBdyPlanes_d);
       cudaFree(XZBdyPlanesNext_d);
       cudaFree(XZBdyPlanesBuffer_d);
     }
     if((rankXid == 0)||(rankXid == numProcsX-1)){
       cudaFree(YZBdyPlanes_d);
       cudaFree(YZBdyPlanesNext_d);
       cudaFree(YZBdyPlanesBuffer_d);
     }
     cudaFree(XYBdyPlanes_d);
     cudaFree(XYBdyPlanesNext_d);
     cudaFree(XYBdyPlanesBuffer_d);
     if(surflayerSelector == 3){
       cudaFree(SURFBdyPlanes_d);
       cudaFree(SURFBdyPlanesNext_d);
       cudaFree(SURFBdyPlanesBuffer_d);
     }
   }//end if hydroBCs == 1

   return(errorCode);

}//end cuda_moistureDeviceCleanup()

/*----->>>>> int cuda_hydroCoreDeviceSecondaryStageSetup(); ---------------------------------------------------------
* Secondary initializations at the device level for BCs  
*/
extern "C" int cuda_hydroCoreDeviceSecondaryStageSetup(float dt){
    int errorCode = CUDA_HYDRO_CORE_SUCCESS;
    int BdyUpdateSteps;

    /*Compute the number of timesteps between BndyPlane Updates*/ 
    BdyUpdateSteps = (int) roundf(dtBdyPlaneBCs/dt);
    cudaMemcpyToSymbol(BdyUpdateSteps_d, &BdyUpdateSteps, sizeof(int));

    printf("%d/%d cuda_hydroCoreDeviceSecondaryStageSetup(): BdyUpdateSteps = %d \n",mpi_rank_world,mpi_size_world,BdyUpdateSteps);
    fflush(stdout);
    return(errorCode);
}

/*----->>>>> int cuda_hydroCoreDeviceBdyPlanesUpdate();      -----------------------------------------------------------------
* Utility to cycle device-sided pointers and push (copy) newest BndyPlanes from Host to Device 
*/
extern "C" int cuda_hydroCoreDeviceBdyPlanesUpdate(){
    int errorCode = CUDA_HYDRO_CORE_SUCCESS;
    printf("%d/%d cuda_hydroCoreDeviceBdyPlanesUpdate(): Cycling device sided pointers and pushing new Bdy Planes into device sided next-blocks for nBdyVars = %d and nSurfBdyVars = %d.\n",
           mpi_rank_world,mpi_size_world,nBndyVars,nSurfBndyVars);
    fflush(stdout);

    if((rankYid == 0)||(rankYid == numProcsY-1)){
      XZBdyPlanesBuffer_d = XZBdyPlanes_d;
      XZBdyPlanes_d = XZBdyPlanesNext_d;
      XZBdyPlanesNext_d = XZBdyPlanesBuffer_d;
      cudaMemcpy(XZBdyPlanesNext_d, XZBdyPlanes, 2*nBndyVars*(Nxp+2*Nh)*(Nzp+2*Nh)*sizeof(float), cudaMemcpyHostToDevice);
    }
    if((rankXid == 0)||(rankXid == numProcsX-1)){
      YZBdyPlanesBuffer_d = YZBdyPlanes_d;
      YZBdyPlanes_d = YZBdyPlanesNext_d;
      YZBdyPlanesNext_d = YZBdyPlanesBuffer_d;
      cudaMemcpy(YZBdyPlanesNext_d, YZBdyPlanes, 2*nBndyVars*(Nyp+2*Nh)*(Nzp+2*Nh)*sizeof(float), cudaMemcpyHostToDevice);
    }
    XYBdyPlanesBuffer_d = XYBdyPlanes_d;
    XYBdyPlanes_d = XYBdyPlanesNext_d;
    XYBdyPlanesNext_d = XYBdyPlanesBuffer_d;
    cudaMemcpy(XYBdyPlanesNext_d, XYBdyPlanes, 2*nBndyVars*(Nxp+2*Nh)*(Nyp+2*Nh)*sizeof(float), cudaMemcpyHostToDevice);
    if(surflayerSelector == 3){
      SURFBdyPlanesBuffer_d = SURFBdyPlanes_d;
      SURFBdyPlanes_d = SURFBdyPlanesNext_d;
      SURFBdyPlanesNext_d = SURFBdyPlanesBuffer_d;
      cudaMemcpy(SURFBdyPlanesNext_d, SURFBdyPlanes, nSurfBndyVars*(Nxp+2*Nh)*(Nyp+2*Nh)*sizeof(float), cudaMemcpyHostToDevice);
    }

    return(errorCode);
} //end cuda_hydroCoreDeviceBdyPlanesUpdate()

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
  int ijk;
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
       scalarField[ijk] = scalarBaseStateField[ijk];
     }else if((k >= kMax_d)&&(k<kMax_d+Nh_d)){
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
	if(fldIndxMom==3){
          scalarField[ijk] = 0.0; //Setting below-ground halo cells to 0 for w-velocity component
	}else{
          scalarField[ijk] = scalarField[ijkTarg];
	}
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

__device__ void cudaDevice_surfaceVarBdyBCs(int fldIndx, float timeWeight, float* scalarField, float* SURFBdyPlanes_d, float* SURFBdyPlanesNext_d){
  int i,j,k;
  int ijTarg;
  int bdyBase;

  /*Establish necessary indices for spatial locality*/
  i = (blockIdx.x)*blockDim.x + threadIdx.x;
  j = (blockIdx.y)*blockDim.y + threadIdx.y;
  k = (blockIdx.z)*blockDim.z + threadIdx.z;

  if((i >= iMin_d-Nh_d)&&(i < iMax_d+Nh_d) &&
     (j >= jMin_d-Nh_d)&&(j < jMax_d+Nh_d)  &&
     (k == 0) ){
       bdyBase = fldIndx*(Nx_d+2*Nh_d)*(Ny_d+2*Nh_d);
       ijTarg = i*(Ny_d+2*Nh_d) + j;
       scalarField[ijTarg] = (1.0-timeWeight)*SURFBdyPlanes_d[bdyBase+ijTarg]+timeWeight*SURFBdyPlanesNext_d[bdyBase+ijTarg];
#ifdef CUDA_BDY_DEBUG
    if(((j==jMin_d)||(j==jMax_d-1))&&((i==iMin_d)||(i==iMax_d-1))){
      printf("cudaDevice_surfaceVarBdyBCs(): At (%d,%d,%d), iFld=%d... SURFBdyPlanes_d[%d+%d] = %f \n",
                                        i,j,k,fldIndx,bdyBase,ijTarg,SURFBdyPlanes_d[bdyBase+ijTarg]);
    }//end if j==...i==...
#endif
  }//end if i>=...j>=...
} //end cudaDevice_surfaceVarBdyBCs

__device__ void cudaDevice_ceilingBdyBCs(int fldIndx, float timeWeight, float* scalarField, float* XYBdyPlanes_d, float* XYBdyPlanesNext_d){
  int i,j,k;
  int ijk,ijTarg;
  int iStride,jStride,kStride;
  int bdyBase;

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
     if((k >= kMax_d-1)&&(k<kMax_d+Nh_d)){
       bdyBase = 2*fldIndx*(Nx_d+2*Nh_d)*(Ny_d+2*Nh_d)+(Nx_d+2*Nh_d)*(Ny_d+2*Nh_d);
       ijTarg = i*(Ny_d+2*Nh_d) + j;
       scalarField[ijk] = (1.0-timeWeight)*XYBdyPlanes_d[bdyBase+ijTarg]+timeWeight*XYBdyPlanesNext_d[bdyBase+ijTarg];
#ifdef CUDA_BDY_DEBUG
  if((k==kMax_d-1)&&((j==jMin_d)||(j==jMax_d-1))&&((i==iMin_d)||(i==iMax_d-1))){
    printf("cudaDevice_ceilingBdyBCs(): At (%d,%d,%d), iFld=%d... XYBdyPlanes_d[%d+%d] = %f \n",
                                        i,j,k,fldIndx,bdyBase,ijTarg,XYBdyPlanes_d[bdyBase+ijTarg]);
  }//end if k==kMax_d
#endif
     }//end if k>-kMax_d...
  }//end if i>=...j>=
} //end cudaDevice_ceilingBdyBCs

#define LATBDYS_HALOSONLY    // Comment this out to have lateral BdyBcs set on the first and last real domain gridcells in the BdyBCs routines below
__device__ void cudaDevice_lateralTKEBdyBCs(int fldIndx, float* scalarField, float* scalarFieldBS, int bdySelector){
  //bdySelector valid values
  // 0 = westBdy
  // 1 = eastBdy
  // 2 = southBdy
  // 3 = northBdy

  int i,j,k;
  int ijk;
  int iStride,jStride,kStride;

  /*Establish necessary indices for spatial locality*/
  i = (blockIdx.x)*blockDim.x + threadIdx.x;
  j = (blockIdx.y)*blockDim.y + threadIdx.y;
  k = (blockIdx.z)*blockDim.z + threadIdx.z;
  iStride = (Ny_d+2*Nh_d)*(Nz_d+2*Nh_d);
  jStride = (Nz_d+2*Nh_d);
  kStride = 1;
  ijk = i*iStride + j*jStride + k*kStride;

  if(bdySelector==0){ //Western Bdy
    if((j >= jMin_d-Nh_d)&&(j < jMax_d+Nh_d) &&
       (k >= kMin_d-Nh_d)&&(k < kMax_d+Nh_d) ){
#ifdef LATBDYS_HALOSONLY
       if((i >= iMin_d-Nh_d)&&(i < iMin_d)){
#else
       if((i >= iMin_d-Nh_d)&&(i <= iMin_d)){
#endif
         scalarField[ijk] = 1.0e-10;
       }//end if(i>=iMin_d-Nh_d...)
    }//end if j>=...k>=
  }//end if bdySelector = 0

  if(bdySelector==1){ //Eastern Bdy
    if((j >= jMin_d-Nh_d)&&(j < jMax_d+Nh_d) &&
       (k >= kMin_d-Nh_d)&&(k < kMax_d+Nh_d) ){
#ifdef LATBDYS_HALOSONLY
       if((i > iMax_d-1)&&(i<iMax_d+Nh_d)){
#else
       if((i >= iMax_d-1)&&(i<iMax_d+Nh_d)){
#endif
         scalarField[ijk] = 1.0e-10;
       }//end if(i>=iMin_d-Nh_d...)
    }//end if j>=...k>=
  }//end if bdySelector = 1

  if(bdySelector==2){ //Southern Bdy
    if((i >= iMin_d-Nh_d)&&(i < iMax_d+Nh_d) &&
       (k >= kMin_d-Nh_d)&&(k < kMax_d+Nh_d) ){
#ifdef LATBDYS_HALOSONLY
       if((j >= jMin_d-Nh_d)&&(j < jMin_d)){
#else
       if((j >= jMin_d-Nh_d)&&(j <= jMin_d)){
#endif
         scalarField[ijk] = 1.0e-10;
       }//end if(j>=jMin_d-Nh_d...)
    }//end if i>=...k>=
  }//end if bdySelector = 2

  if(bdySelector==3){ //Northern Bdy
    if((i >= iMin_d-Nh_d)&&(i < iMax_d+Nh_d) &&
       (k >= kMin_d-Nh_d)&&(k < kMax_d+Nh_d) ){
#ifdef LATBDYS_HALOSONLY
       if((j > jMax_d-1)&&(j<jMax_d+Nh_d)){
#else
       if((j >= jMax_d-1)&&(j<jMax_d+Nh_d)){
#endif
         scalarField[ijk] = 1.0e-10;
       }//end if(j>=jMin_d-Nh_d...)
    }//end if i>=...k>=
  }//end if bdySelector = 2

} //end cudaDevice_lateralTKEBdyBCs

__device__ void cudaDevice_westBdyBCs(int fldIndx, float timeWeight, float* scalarField, float* YZBdyPlanes_d, float* YZBdyPlanesNext_d){
  int i,j,k;
  int ijk,jkTarg;
  int iStride,jStride,kStride;
  int bdyBase;

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
#ifdef LATBDYS_HALOSONLY
     if((i >= iMin_d-Nh_d)&&(i < iMin_d)){
#else
     if((i >= iMin_d-Nh_d)&&(i <= iMin_d)){
#endif
       bdyBase = 2*fldIndx*(Ny_d+2*Nh_d)*(Nz_d+2*Nh_d);
       jkTarg = j*(Nz_d+2*Nh_d) + k;
       scalarField[ijk] = (1.0-timeWeight)*YZBdyPlanes_d[bdyBase+jkTarg]+timeWeight*YZBdyPlanesNext_d[bdyBase+jkTarg];
#ifdef CUDA_BDY_DEBUG
//#if 1
  if((i==iMin_d-1)&&((j==jMin_d)||(j==jMax_d-1))&&((k==kMin_d)||(k==kMax_d-1))){
    printf("cudaDevice_westBdyBCs(): At (%d,%d,%d), iFld=%d... YZBdyPlanes_d[%d+%d] = %f \n",
                                        i,j,k,fldIndx,bdyBase,jkTarg,YZBdyPlanes_d[bdyBase+jkTarg]);
  }//end if k==kMax_d
#endif
     }//end if(i>=iMin_d-Nh_d...)
  }//end if j>=...k>=
} //end cudaDevice_westBdyBCs

__device__ void cudaDevice_eastBdyBCs(int fldIndx, float timeWeight, float* scalarField, float* YZBdyPlanes_d, float* YZBdyPlanesNext_d){
  int i,j,k;
  int ijk,jkTarg;
  int iStride,jStride,kStride;
  int bdyBase;

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
#ifdef LATBDYS_HALOSONLY
     if((i > iMax_d-1)&&(i<iMax_d+Nh_d)){
#else
     if((i >= iMax_d-1)&&(i<iMax_d+Nh_d)){
#endif
       bdyBase = 2*fldIndx*(Ny_d+2*Nh_d)*(Nz_d+2*Nh_d)+(Ny_d+2*Nh_d)*(Nz_d+2*Nh_d);
       jkTarg = j*(Nz_d+2*Nh_d) + k;
       scalarField[ijk] = (1.0-timeWeight)*YZBdyPlanes_d[bdyBase+jkTarg]+timeWeight*YZBdyPlanesNext_d[bdyBase+jkTarg];
     }//end if i>-iMax_d...
  }//end if j>=...k>=
} //end cudaDevice_eastBdyBCs

__device__ void cudaDevice_southBdyBCs(int fldIndx, float timeWeight, float* scalarField, float* XZBdyPlanes_d, float* XZBdyPlanesNext_d){
  int i,j,k;
  int ijk,ikTarg;
  int iStride,jStride,kStride;
  int bdyBase;

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
#ifdef LATBDYS_HALOSONLY
     if((j >= jMin_d-Nh_d)&&(j < jMin_d)){
#else
     if((j >= jMin_d-Nh_d)&&(j <= jMin_d)){
#endif
       bdyBase = 2*fldIndx*(Nx_d+2*Nh_d)*(Nz_d+2*Nh_d);
       ikTarg = i*(Nz_d+2*Nh_d) + k;
       scalarField[ijk] = (1.0-timeWeight)*XZBdyPlanes_d[bdyBase+ikTarg]+timeWeight*XZBdyPlanesNext_d[bdyBase+ikTarg];
     }//end if(j>=jMin_d-Nh_d...)
  }//end if i>=...k>=
} //end cudaDevice_southBdyBCs

__device__ void cudaDevice_northBdyBCs(int fldIndx, float timeWeight, float* scalarField, float* XZBdyPlanes_d, float* XZBdyPlanesNext_d){
  int i,j,k;
  int ijk,ikTarg;
  int iStride,jStride,kStride;
  int bdyBase;

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
#ifdef LATBDYS_HALOSONLY
     if((j > jMax_d-1)&&(j<jMax_d+Nh_d)){
#else
     if((j >= jMax_d-1)&&(j<jMax_d+Nh_d)){
#endif
       bdyBase = 2*fldIndx*(Nx_d+2*Nh_d)*(Nz_d+2*Nh_d)+(Nx_d+2*Nh_d)*(Nz_d+2*Nh_d);
       ikTarg = i*(Nz_d+2*Nh_d) + k;
       scalarField[ijk] = (1.0-timeWeight)*XZBdyPlanes_d[bdyBase+ikTarg]+timeWeight*XZBdyPlanesNext_d[bdyBase+ikTarg];
#ifdef CUDA_BDY_DEBUG
//#if 1
  if((j==jMax_d-1)&&((i==iMin_d)||(i==iMax_d-1))&&((k==kMin_d)||(k==kMax_d-1))){
    printf("%d/%d cudaDevice_northBdyBCs(): At (%d,%d,%d), iFld=%d... XZBdyPlanes_d[%d+%d] = %f \n",
                                        mpi_rank_world_d,mpi_size_world_d,i,j,k,fldIndx,bdyBase,ikTarg,XZBdyPlanes_d[bdyBase+ikTarg]);
  }//end if i==iMin_d-1)&&((j==jMin_d)||(j==jMax_d-1))....
#endif
     }//end if j>-jMax_d...
  }//end if i>=...k>=
} //end cudaDevice_northBdyBCs

