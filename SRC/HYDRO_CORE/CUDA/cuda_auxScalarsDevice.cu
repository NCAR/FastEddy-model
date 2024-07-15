/*Auxiliary Scalar Fields*/
__constant__ int NhydroAuxScalars_d;       /*Number of prognostic auxiliary scalar variable fields */
__constant__ int AuxScAdvSelector_d; /*adv. scheme for auxiliary scalar fields */
__constant__ float AuxScAdvSelector_b_hyb_d; /* hybrid advection scheme parameter */
float *hydroAuxScalars_d;     /*Base Adress of memory containing prognostic auxiliary scalar variable fields */
float *hydroAuxScalarsFrhs_d; /*Base Adress of memory Auxiliary field Frhs */
float *AuxScalarsTauFlds_d;   /*Base Adress of memory Auxiliary field SGS taus */
/*Auxiliary Scalar Sources*/
__device__ __constant__ int srcAuxScTemporalType_d[MAX_AUXSC_SRC];/*Temporal characterization of source (0 = instantaneous, 1 = continuous) */
__device__ __constant__ float srcAuxScStartSeconds_d[MAX_AUXSC_SRC];     /*Source start time in seconds */
__device__ __constant__ float srcAuxScDurationSeconds_d[MAX_AUXSC_SRC];  /*Source duration in seconds */
__device__ __constant__ int srcAuxScGeometryType_d[MAX_AUXSC_SRC];   /*0 = point (single cell volume), 1 = line (line of surface cells) */
__device__ __constant__ float srcAuxScLocation_d[MAX_AUXSC_SRC*3];      /*Cartesian coordinate tuple 'center' of the source*/
__device__ __constant__ int srcAuxScMassSpecType_d[MAX_AUXSC_SRC]; /*Mass specification type 0 = strict mass in kg, 1 = mass source rate in kg/s,  */
__device__ __constant__ float srcAuxScMassSpecValue_d[MAX_AUXSC_SRC]; /*Mass specification value in kg or kg/s given by srcAuxScMassSpecType 0 or 1 */

/*#################------------ AUXSCALARS submodule function definitions ------------------#############*/
/*----->>>>> int cuda_auxScalarsDeviceSetup();       ---------------------------------------------------------
* Used to cudaMalloc and cudaMemcpy parameters and coordinate arrays, and for the AUXSCALARS_CUDA submodule.
*/
extern "C" int cuda_auxScalarsDeviceSetup(){
   int errorCode = CUDA_AUXSCALARS_SUCCESS;
   int Nelems;

   cudaMemcpyToSymbol(NhydroAuxScalars_d, &NhydroAuxScalars, sizeof(int));
   if (NhydroAuxScalars > 0){
    cudaMemcpyToSymbol(AuxScAdvSelector_d, &AuxScAdvSelector, sizeof(int));
    cudaMemcpyToSymbol(AuxScAdvSelector_b_hyb_d, &AuxScAdvSelector_b_hyb, sizeof(float));
    cudaMemcpyToSymbol(srcAuxScTemporalType_d, &srcAuxScTemporalType[0], NhydroAuxScalars*sizeof(int));
    cudaMemcpyToSymbol(srcAuxScStartSeconds_d, &srcAuxScStartSeconds[0], NhydroAuxScalars*sizeof(float));
    cudaMemcpyToSymbol(srcAuxScDurationSeconds_d, &srcAuxScDurationSeconds[0], NhydroAuxScalars*sizeof(float));
    cudaMemcpyToSymbol(srcAuxScGeometryType_d, &srcAuxScGeometryType[0], NhydroAuxScalars*sizeof(int));
    cudaMemcpyToSymbol(srcAuxScLocation_d, &srcAuxScLocation[0], NhydroAuxScalars*3*sizeof(float), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(srcAuxScMassSpecType_d, &srcAuxScMassSpecType[0], NhydroAuxScalars*sizeof(int));
    cudaMemcpyToSymbol(srcAuxScMassSpecValue_d, &srcAuxScMassSpecValue[0], NhydroAuxScalars*sizeof(float));
   }//end if NydroAuxScalars > 0

   if (NhydroAuxScalars > 0){
     Nelems = (Nxp+2*Nh)*(Nyp+2*Nh)*(Nzp+2*Nh);
     fecuda_DeviceMalloc(Nelems*NhydroAuxScalars*sizeof(float), &hydroAuxScalars_d); /*Prognostic variable fields*/
     fecuda_DeviceMalloc(Nelems*NhydroAuxScalars*sizeof(float), &hydroAuxScalarsFrhs_d); /*Prognostic variable Frhs*/
     if ((turbulenceSelector > 0) && (AuxScSGSturb > 0)){
       fecuda_DeviceMalloc(Nelems*3*sizeof(float), &AuxScalarsTauFlds_d);
     }
   } // end if NhydroAuxScalars > 0

   return(errorCode);

} //end cuda_moitureDeviceSetup()

/*----->>>>> extern "C" int cuda_auxScalarsDeviceCleanup();  -----------------------------------------------------------
Used to free all malloced memory by the AUXSCALARS submodule.
*/
extern "C" int cuda_auxScalarsDeviceCleanup(){
   int errorCode = CUDA_AUXSCALARS_SUCCESS;

   /* Free any auxscalars submodule arrays */
   if(NhydroAuxScalars > 0) { 
     cudaFree(hydroAuxScalars_d); 
     cudaFree(hydroAuxScalarsFrhs_d);
     if ((turbulenceSelector > 0) && (AuxScSGSturb > 0)){
       cudaFree(AuxScalarsTauFlds_d);
     } 
   }
   return(errorCode);

}//end cuda_moistureDeviceCleanup()

__global__ void cudaDevice_hydroCoreUnitTestCompleteAuxScalars(float simTime, float* hydroFlds,
                                                              float* hydroAuxScalars, float* hydroAuxScalarsFrhs,
                                                              float* hydroFaceVels,
                                                              float* xPos_d, float* yPos_d, float* zPos_d, float* topoPos_d,
                                                              float* J33_d, float* D_Jac_d, float* invD_Jac_d){

   int i,j,k;
   int iFld,fldStride;
   float* u_cf;
   float* v_cf;
   float* w_cf;
   float* fld;
   float* fldFrhs;

   /*Establish necessary indices for spatial locality*/
   i = (blockIdx.x)*blockDim.x + threadIdx.x;
   j = (blockIdx.y)*blockDim.y + threadIdx.y;
   k = (blockIdx.z)*blockDim.z + threadIdx.z;

   fldStride = (Nx_d+2*Nh_d)*(Ny_d+2*Nh_d)*(Nz_d+2*Nh_d);
   u_cf = &hydroFaceVels[fldStride*0];
   v_cf = &hydroFaceVels[fldStride*1];
   w_cf = &hydroFaceVels[fldStride*2];

   if((i >= iMin_d)&&(i < iMax_d) &&
      (j >= jMin_d)&&(j < jMax_d) &&
      (k >= kMin_d)&&(k < kMax_d) ){

      for(iFld=0; iFld < NhydroAuxScalars_d; iFld++){
         fld = &hydroAuxScalars[fldStride*iFld];
         fldFrhs = &hydroAuxScalarsFrhs[fldStride*iFld];
         /* Calculate scalar, cell-valued divergence of the advective flux */
         if (AuxScAdvSelector_d == 1) { // 3rd-order QUICK
           cudaDevice_QUICKDivAdvFlux(fld, fldFrhs, u_cf, v_cf, w_cf, invD_Jac_d);
         } else if (AuxScAdvSelector_d == 2) { // hybrid 3rd-4th order
           cudaDevice_HYB34DivAdvFlux(fld, fldFrhs, u_cf, v_cf, w_cf, AuxScAdvSelector_b_hyb_d, invD_Jac_d);
         } else if (AuxScAdvSelector_d == 3) { // hybrid 5th-6th order
           cudaDevice_HYB56DivAdvFlux(fld, fldFrhs, u_cf, v_cf, w_cf, AuxScAdvSelector_b_hyb_d, invD_Jac_d);
         } else if (AuxScAdvSelector_d == 4) { // 3rd-order WENO
           cudaDevice_WENO3DivAdvFluxX(fld, fldFrhs, u_cf, invD_Jac_d);
           cudaDevice_WENO3DivAdvFluxY(fld, fldFrhs, v_cf, invD_Jac_d);
           cudaDevice_WENO3DivAdvFluxZ(fld, fldFrhs, w_cf, invD_Jac_d);
         } else if (AuxScAdvSelector_d == 5) { // 5th-order WENO
           cudaDevice_WENO5DivAdvFluxX(fld, fldFrhs, u_cf, invD_Jac_d);
           cudaDevice_WENO5DivAdvFluxY(fld, fldFrhs, v_cf, invD_Jac_d);
           cudaDevice_WENO5DivAdvFluxZ(fld, fldFrhs, w_cf, invD_Jac_d);
         } else if (AuxScAdvSelector_d == 6) { // centered 2nd-order
           cudaDevice_SecondDivAdvFlux(fld, fldFrhs, u_cf, v_cf, w_cf, invD_Jac_d);
         } else { // defaults to 1st-order upwinding
           cudaDevice_UpstreamDivAdvFlux(fld, fldFrhs, u_cf, v_cf, w_cf, invD_Jac_d);
         }
#ifdef CUDA_DEBUG
	 if(i == iMin_d && j == jMin_d && k == kMin_d){
           printf("%d/%d cudaDevice_hydroCoreUnitTestCompleteAuxScalars(): NhydroAuxScalars_d = %d, iFld = %d, simTime = %f, srcAuxScStartSeconds_d[iFld] = %f, srcAuxScDurationSeconds_d[iFld] = %f.\n\t\t srcAuxScLocation_d[iFld*3+0] = %f, srcAuxScLocation_d[iFld*3+1] = %f, srcAuxScLocation_d[iFld*3+2] = %f.\n \t\t\t srcAuxScMassSpecValue_d[iFld] = %f\n",
                mpi_rank_world_d,mpi_size_world_d,NhydroAuxScalars_d,iFld, simTime,srcAuxScStartSeconds_d[iFld],srcAuxScDurationSeconds_d[iFld],
		srcAuxScLocation_d[iFld*3+0], srcAuxScLocation_d[iFld*3+1], srcAuxScLocation_d[iFld*3+2], srcAuxScMassSpecValue_d[iFld]);
	 }
#endif
         if((simTime >= srcAuxScStartSeconds_d[iFld])&&(simTime <= (srcAuxScStartSeconds_d[iFld] + srcAuxScDurationSeconds_d[iFld]) )){
           cudaDevice_calcAuxScalarSource(iFld, fldFrhs, &hydroFlds[fldStride*RHO_INDX],
                                          xPos_d, yPos_d, zPos_d, topoPos_d, J33_d, D_Jac_d);
         } //end if simTime is within a source release time window.
      }//for iFld
   }//end if in the range of non-halo cells

} // end cudaDevice_hydroCoreUnitTestCompleteAuxScalars()

__device__ void cudaDevice_calcAuxScalarSource(int iFld, float *auxScFrhs, float *rhoFld,
                                               float* xPos_d, float* yPos_d, float* zPos_d,
                                               float* topoPos_d, float* J33_d, float* D_Jac_d){  
  int i,j,k;
  int ijk;
  int ij;
  int iStride,jStride,kStride;

  /*Establish necessary indices for spatial locality*/
  i = (blockIdx.x)*blockDim.x + threadIdx.x;
  j = (blockIdx.y)*blockDim.y + threadIdx.y;
  k = (blockIdx.z)*blockDim.z + threadIdx.z;
  iStride = (Ny_d+2*Nh_d)*(Nz_d+2*Nh_d);
  jStride = (Nz_d+2*Nh_d);
  kStride = 1;
  ijk = i*iStride + j*jStride + k*kStride;
  ij = i*(Ny_d+2*Nh_d) + j*(1);

  if((fabsf(xPos_d[ijk]-srcAuxScLocation_d[iFld*3+0]) < dX_d)&&
     (fabsf(yPos_d[ijk]-srcAuxScLocation_d[iFld*3+1]) < dY_d)&&
     (fabsf(zPos_d[ijk]-topoPos_d[ij]-srcAuxScLocation_d[iFld*3+2]) < 1.0/(dZi_d*J33_d[ijk]) )){
     //Assume point source for the moment              
     auxScFrhs[ijk] = auxScFrhs[ijk] + srcAuxScMassSpecValue_d[iFld]/
                      (D_Jac_d[ijk]*(dX_d*dY_d*dZ_d)*          //cell volume
                      srcAuxScDurationSeconds_d[iFld]*(1-srcAuxScMassSpecType_d[iFld])); //this per time on/off given
                                                                                         // a kg or kg/s MassSpecType 
#ifdef CUDA_DEBUG
      printf("cudaDevice_calcAuxScalarSource(): @(i,j,k) = (%d,%d,%d) Loc_X = %12.6f, Loc_Y = %12.6f, Loc_Z = %12.6f, \n\t\tdX_d = %12.6f, dY_d = %12.6f, dZ_d = %12.6f \n",
      i,j,k,srcAuxScLocation_d[iFld*3+0],srcAuxScLocation_d[iFld*3+1],srcAuxScLocation_d[iFld*3+2], dX_d,dY_d,1.0/(dZi_d*J33_d[ijk]));
#endif
  }//end if this is a source cell   

} //cudaDevice_calcAuxScalarSource
