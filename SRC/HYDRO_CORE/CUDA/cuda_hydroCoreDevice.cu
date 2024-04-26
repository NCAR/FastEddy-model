/* FastEddy®: SRC/HYDRO_CORE/CUDA/cuda_hydroCoreDevice.cu 
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
//INCLUDED HEADERS
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <float.h>
#include <math.h>
#include <fempi.h>
#include <grid.h>
#include <mem_utils.h>
#include <hydro_core.h>
#include <fecuda_Device_cu.h>
#include <cuda_gridDevice_cu.h>
#include <cuda_hydroCoreDevice_cu.h>

//INCLUDED SOURCE FILES
#include "cuda_BaseStateDevice.cu"
#include "cuda_advectionDevice.cu" 
#include "cuda_buoyancyDevice.cu" 
#include "cuda_coriolisDevice.cu" 
#include "cuda_pressureDevice.cu" 
#include "cuda_BCsDevice.cu"
#include "cuda_rayleighDampingDevice.cu" 
#include "cuda_surfaceLayerDevice.cu"
#include "cuda_sgsTurbDevice.cu"
#include "cuda_molecularDiffDevice.cu" 
#include "cuda_sgstkeDevice.cu" 
#include "cuda_largeScaleForcingsDevice.cu" 
#include "cuda_moistureDevice.cu" 
#include "cuda_filtersDevice.cu" 

/*#################------------- CUDA_HYDRO_CORE module variable definitions ------------------#############*/
/*Parameters*/
__constant__ int Nhydro_d;       // Number of hydro_core prognostic variable fields

/* array fields */
float *hydroFlds_d;     //Base Adress of memory containing all prognostic variable fields under hydro_core
float *hydroFldsFrhs_d; //Base Adress of memory containing variable field Frhs(s) under hydro_core
float *hydroRhoInv_d;   //storage for 1.0/rho

/*HYDRO_CORE Submodule parameters*/

__constant__ int physics_oneRKonly_d; /*selector to apply physics RHS forcing only at the latest RK stage: 0= off, 1= on*/

/*Constants*/
__constant__ float R_gas_d;                 /* The ideal gas constant in J/(mol*K) */
__constant__ float R_vapor_d;               /* The ideal gas constant for water vapor in J/(mol*K) */
__constant__ float Rv_Rg_d;    /* Ratio R_vapor/R_gas */
__constant__ float cp_gas_d;                /* Specific heat of air at constant pressure */
__constant__ float cv_gas_d;                /* Specific heat of air at constant pressure */
__constant__ float accel_g_d;           /* Acceleration of gravity 9.8 m/(s^2) */
__constant__ float R_cp_d;    /* Ratio R/cp */
__constant__ float cp_R_d;    /* Ratio cp/R */
__constant__ float cp_cv_d;   /* Ratio cp/cv */
__constant__ float refPressure_d;   /* Reference pressure set constant to 1e5 Pascals or 1000 millibars */
__constant__ float kappa_d;          /*von Karman constant*/
__constant__ float L_v_d;            /* latent heat of vaporization (J/kg) */


/*#################------------ CUDA_HYDRO_CORE modune function definitions ------------------#############*/
/*----->>>>> int cuda_hydroCoreDeviceSetup();       ----------------------------------------------------------------
 * Used to cudaMalloc and cudaMemcpy parameters and coordinate arrays, and for the HYDRO_CORE_CUDA module.
*/
extern "C" int cuda_hydroCoreDeviceSetup(){
   int errorCode = CUDA_HYDRO_CORE_SUCCESS;
   int Nelems;
 
   /*Synchronize the Device*/
   gpuErrchk( cudaDeviceSynchronize() );
 
   /*Constants*/
   cudaMemcpyToSymbol(Nhydro_d, &Nhydro, sizeof(int));

   /*BCs*/
   errorCode = cuda_BCsDeviceSetup();

   /*BUOYANCY*/
   errorCode = cuda_buoyancyDeviceSetup();

   /*CORIOLIS*/
   errorCode = cuda_coriolisDeviceSetup();

   /*rayleighDamping*/
   errorCode = cuda_rayleighDampingDeviceSetup();

   cudaMemcpyToSymbol(temp_grnd_d, &temp_grnd, sizeof(float));
   cudaMemcpyToSymbol(pres_grnd_d, &pres_grnd, sizeof(float));

   cudaMemcpyToSymbol(physics_oneRKonly_d, &physics_oneRKonly, sizeof(int));

   cudaMemcpyToSymbol(R_gas_d, &R_gas, sizeof(float));
   cudaMemcpyToSymbol(R_vapor_d, &R_vapor, sizeof(float));
   cudaMemcpyToSymbol(Rv_Rg_d, &Rv_Rg, sizeof(float));
   cudaMemcpyToSymbol(cv_gas_d, &cv_gas, sizeof(float));
   cudaMemcpyToSymbol(cp_gas_d, &cp_gas, sizeof(float));
   cudaMemcpyToSymbol(accel_g_d, &accel_g, sizeof(float));
   cudaMemcpyToSymbol(R_cp_d, &R_cp, sizeof(float));
   cudaMemcpyToSymbol(cp_R_d, &cp_R, sizeof(float));
   cudaMemcpyToSymbol(cp_cv_d, &cp_cv, sizeof(float));
   cudaMemcpyToSymbol(refPressure_d, &refPressure, sizeof(float));
   cudaMemcpyToSymbol(corioConstHorz_d, &corioConstHorz, sizeof(float));
   cudaMemcpyToSymbol(corioConstVert_d, &corioConstVert, sizeof(float));
   cudaMemcpyToSymbol(corioLS_fact_d, &corioLS_fact, sizeof(float));
   cudaMemcpyToSymbol(kappa_d, &kappa, sizeof(float)); // DME
   cudaMemcpyToSymbol(L_v_d, &L_v, sizeof(float));
   gpuErrchk( cudaPeekAtLastError() ); /*Check for errors in the cudaMemCpy calls*/

   /*Set the full memory block number of elements for hydroCore fields*/
   Nelems = (Nxp+2*Nh)*(Nyp+2*Nh)*(Nzp+2*Nh); 
   /* Allocate the HYDRO_CORE arrays */
   fecuda_DeviceMalloc(Nelems*Nhydro*sizeof(float), &hydroFlds_d); /*Prognostic variable fields*/ 
   fecuda_DeviceMalloc(Nelems*Nhydro*sizeof(float), &hydroFldsFrhs_d); /*Prognostic variable field Frhs(s)*/ 
   fecuda_DeviceMalloc(Nelems*sizeof(float), &hydroRhoInv_d); 

   /*ADVECTION*/
   if(advectionSelector >= 0){
     errorCode = cuda_advectionDeviceSetup();
   }//end if advectionSelector > 0

   /*PRESSURE*/
   if(pgfSelector > 0){
     errorCode = cuda_pressureDeviceSetup();
   }//end if pgfSelector > 0

   /*BASESTATE*/
   errorCode = cuda_BaseStateDeviceSetup();

   /*SGSTURB*/
   if(turbulenceSelector > 0){
     errorCode = cuda_sgsTurbDeviceSetup();
     /* SGSTKE */
     if (TKESelector > 0) { 
       errorCode = cuda_sgstkeDeviceSetup();
     } // end if TKESelector > 0
   }//end if turbulenceSelector > 0

   if (diffusionSelector > 0) { 
     errorCode = cuda_molecularDiffDeviceSetup();
   }
   if (surflayerSelector > 0) { 
       errorCode = cuda_surfaceLayerDeviceSetup();
   }
   gpuErrchk( cudaPeekAtLastError() ); /*Check for errors in the cudaMalloc calls*/

   /* LARGE SCALE FORCINGS*/
   if (lsfSelector > 0){ 
     errorCode = cuda_lsfDeviceSetup();
   }

   /* MOISTURE*/
   if (moistureSelector >= 0){ 
     errorCode = cuda_moistureDeviceSetup();
   }

   /* EXPLICIT FILTERS */
   if (filterSelector > 0){ 
     errorCode = cuda_filtersDeviceSetup();
   }

   gpuErrchk( cudaPeekAtLastError() ); /*Check for errors in the cudaMalloc calls*/
   gpuErrchk( cudaDeviceSynchronize() );
   MPI_Barrier(MPI_COMM_WORLD);
   printf("cuda_hydroCoreDeviceSetup() complete.\n");


   /* Done */
   return(errorCode);
} //end cuda_hydroCoreDeviceSetup()

/*----->>>>> extern "C" int cuda_hydroCoreDeviceCleanup();  -----------------------------------------------------------
Used to free all malloced memory by the HYDRO_CORE module.
*/
extern "C" int cuda_hydroCoreDeviceCleanup(){
   int errorCode = HYDRO_CORE_SUCCESS;

   /* Free any HYDRO_CORE module arrays */
   cudaFree(hydroFlds_d); 
   cudaFree(hydroFldsFrhs_d); 
   cudaFree(hydroRhoInv_d); 
  
   errorCode = cuda_BCsDeviceCleanup();
   
   if(buoyancySelector >= 0) {
       errorCode = cuda_buoyancyDeviceCleanup();
   }
   if(coriolisSelector >= 0) {
       errorCode = cuda_coriolisDeviceCleanup();
   }
   if(dampingLayerSelector >= 0) {
       errorCode = cuda_rayleighDampingDeviceCleanup();
   }

   errorCode = cuda_BaseStateDeviceCleanup();

   if(advectionSelector >= 0) {
       errorCode = cuda_advectionDeviceCleanup();
   }
   if(pgfSelector > 0) { 
       errorCode = cuda_pressureDeviceCleanup();
   }
   if(turbulenceSelector > 0) {
     errorCode = cuda_sgsTurbDeviceCleanup();
     if (TKESelector > 0){
       errorCode = cuda_sgstkeDeviceCleanup();
     }
   }
   if (diffusionSelector > 0) { 
     errorCode = cuda_molecularDiffDeviceCleanup();
   } 
   if (surflayerSelector > 0) { 
     errorCode = cuda_surfaceLayerDeviceCleanup();
   }
   if (lsfSelector > 0) {
     errorCode = cuda_lsfDeviceCleanup();
   }
   if (moistureSelector > 0) {
     errorCode = cuda_moistureDeviceCleanup();
   }
   if (filterSelector > 0){
     errorCode = cuda_filtersDeviceCleanup();
   }

   return(errorCode);

}//end cuda_hydroCoreDeviceCleanup()

/*----->>>>> extern "C" int cuda_hydroCoreDeviceBuildFrhs();  --------------------------------------------------
* This routine provides the externally callable cuda-kernel call to perform a complete hydroCore build_Frhs
*/
extern "C" int cuda_hydroCoreDeviceBuildFrhs(float simTime, int simTime_it, int simTime_itRestart, float dt, int timeStage, int numRKstages){
   int errorCode = CUDA_HYDRO_CORE_SUCCESS;
   int iFld, fldStride;
#ifdef TIMERS_LEVEL2
   cudaEvent_t startE, stopE;
   float elapsedTime;
#endif
   int temp_freq;
   float temp_freq_fac;
   int mp_update;
   int simTime_diff,ldf_itNum;

#ifdef DEBUG
   printf("cuda_hydroCoreDeviceBuildFrhs: tBlock = {%d, %d, %d}\n",tBlock.x, tBlock.y, tBlock.z);
   printf("cuda_hydroCoreDeviceBuildFrhs: grid = {%d, %d, %d}\n",grid.x, grid.y, grid.z);
   fflush(stdout);
#endif 

//#define TIMERS_LEVEL1
#ifdef TIMERS_LEVEL1
   /*Launch a blocking kernel to Perform the build_Frhs "preparations" phase*/
   createAndStartEvent(&startE, &stopE);
#endif

   fldStride = (Nxp+2*Nh)*(Nyp+2*Nh)*(Nzp+2*Nh);
//#define VERBOSE_HALO
#ifdef VERBOSE_HALO
   double mpi_t1, mpi_t2, mpi_t3, mpi_t4;
   mpi_t1 = MPI_Wtime();    //Mark the walltime to measure duration of initializations.
#endif
   for(iFld=0; iFld < Nhydro; iFld++){   
     if(numProcsX>1){
       errorCode = fecuda_SendRecvWestEast(&hydroFlds_d[iFld*fldStride], &hydroFlds_d[iFld*fldStride],hydroBCs);
       errorCode = fecuda_SendRecvEastWest(&hydroFlds_d[iFld*fldStride], &hydroFlds_d[iFld*fldStride],hydroBCs);
     }//if multi-rank in X-dir
     if(numProcsY>1){
       errorCode = fecuda_SendRecvSouthNorth(&hydroFlds_d[iFld*fldStride], &hydroFlds_d[iFld*fldStride],hydroBCs);
       errorCode = fecuda_SendRecvNorthSouth(&hydroFlds_d[iFld*fldStride], &hydroFlds_d[iFld*fldStride],hydroBCs);
     }//if multi-rank in Y-dir
   } //end for iFld
   gpuErrchk( cudaDeviceSynchronize() );

   if((turbulenceSelector>0)&&(TKESelector>0)){
     for(iFld=0; iFld < TKESelector; iFld++){
       if(numProcsX>1){
         errorCode = fecuda_SendRecvWestEast(&sgstkeScalars_d[iFld*fldStride], &sgstkeScalars_d[iFld*fldStride],hydroBCs);
         errorCode = fecuda_SendRecvEastWest(&sgstkeScalars_d[iFld*fldStride], &sgstkeScalars_d[iFld*fldStride],hydroBCs);
       }//if multi-rank in X-dir
       if(numProcsY>1){
         errorCode = fecuda_SendRecvSouthNorth(&sgstkeScalars_d[iFld*fldStride], &sgstkeScalars_d[iFld*fldStride],hydroBCs);
         errorCode = fecuda_SendRecvNorthSouth(&sgstkeScalars_d[iFld*fldStride], &sgstkeScalars_d[iFld*fldStride],hydroBCs);
       }//if multi-rank in Y-dir
     }//end for iFld
     gpuErrchk( cudaDeviceSynchronize() );
   }

   if((moistureSelector>0)&&(moistureNvars>0)){
     for(iFld=0; iFld < moistureNvars; iFld++){
       if(numProcsX>1){
         errorCode = fecuda_SendRecvWestEast(&moistScalars_d[iFld*fldStride], &moistScalars_d[iFld*fldStride],hydroBCs);
         errorCode = fecuda_SendRecvEastWest(&moistScalars_d[iFld*fldStride], &moistScalars_d[iFld*fldStride],hydroBCs);
       }//if multi-rank in X-dir
       if(numProcsY>1){
         errorCode = fecuda_SendRecvSouthNorth(&moistScalars_d[iFld*fldStride], &moistScalars_d[iFld*fldStride],hydroBCs);
         errorCode = fecuda_SendRecvNorthSouth(&moistScalars_d[iFld*fldStride], &moistScalars_d[iFld*fldStride],hydroBCs);
       }//if multi-rank in Y-dir
     }//end for iFld
     gpuErrchk( cudaDeviceSynchronize() );
   }

#ifdef VERBOSE_HALO
   MPI_Barrier(MPI_COMM_WORLD);
   mpi_t2 = MPI_Wtime();    //Mark the walltime to measure duration of initializations.
   if(mpi_rank_world == 0){
     printf("Horizontal halo exchanges complete after %8.4f (s).\n", (mpi_t2-mpi_t1));
     fflush(stdout);
   } //if mpi_rank_world
#endif

   cudaDevice_hydroCoreUnitTestCommence<<<grid, tBlock>>>(simTime_it, hydroFlds_d, hydroFldsFrhs_d, 
                                                          hydroBaseStateFlds_d,  
                                                          tskin_d, qskin_d,
                                                          sgstkeScalars_d,sgstkeScalarsFrhs_d, hydroKappaM_d,
                                                          moistScalars_d, moistScalarsFrhs_d, zPos_d);
   gpuErrchk( cudaGetLastError() );
   gpuErrchk( cudaDeviceSynchronize() );
   cudaDevice_hydroCoreUnitTestCommenceRhoInvPresPert<<<grid, tBlock>>>(hydroFlds_d, hydroRhoInv_d, 
                                                          hydroBaseStateFlds_d, 
                                                          hydroPres_d, hydroBaseStatePres_d,
                                                          moistScalars_d, zPos_d);
   gpuErrchk( cudaGetLastError() );
   gpuErrchk( cudaDeviceSynchronize() );
#ifdef TIMERS_LEVEL2
   stopSynchReportDestroyEvent(&startE, &stopE, &elapsedTime);
   printf("cuda_hydroCoreUnitTestCommence()  Kernel execution time (ms): %12.8f\n", elapsedTime);
   gpuErrchk( cudaDeviceSynchronize() );

   /*Calculate cell-face Velocites, and PGF terms that involve the J** metric tensor arrays*/
   createAndStartEvent(&startE, &stopE);
#endif
  
   /*Advecting Velocities*/ 
    cudaDevice_hydroCoreCalcFaceVelocities<<<grid, tBlock>>>(simTime, simTime_it, simTime_itRestart, dt, timeStage, numRKstages,
                                                            hydroFlds_d, hydroFldsFrhs_d,
                                                            hydroFaceVels_d, hydroPres_d,
                                                            hydroNuGradXFlds_d, hydroNuGradYFlds_d,
                                                            hydroNuGradZFlds_d, hydroTauFlds_d,
                                                            cdFld_d, chFld_d, cqFld_d, fricVel_d, htFlux_d, tskin_d,
                                                            invOblen_d, z0m_d, z0t_d, qFlux_d, qskin_d, sea_mask_d,
                                                            hydroRhoInv_d, hydroKappaM_d, sgstkeScalars_d, sgstke_ls_d,
                                                            dedxi_d, moistScalars_d, moistTauFlds_d, moistScalarsFrhs_d,
                                                            J31_d, J32_d, J33_d, D_Jac_d);
   gpuErrchk( cudaGetLastError() );
#ifdef TIMERS_LEVEL2
   stopSynchReportDestroyEvent(&startE, &stopE, &elapsedTime);
   printf("cuda_hydroCoreCalcFaceVelocities()  Kernel execution time (ms): %12.8f\n", elapsedTime);
   gpuErrchk( cudaPeekAtLastError() ); //Check for errors in the cudaMemCpy calls
   gpuErrchk( cudaDeviceSynchronize() );

   /*Calculate the Frhs contributions for the advection and buoyancy terms*/
   createAndStartEvent(&startE, &stopE);
#endif
   cudaDevice_hydroCoreUnitTestComplete<<<grid, tBlock>>>(simTime, simTime_it, dt, timeStage, numRKstages, hydroFlds_d, hydroFldsFrhs_d,
                                                          hydroFaceVels_d, hydroBaseStateFlds_d, hydroTauFlds_d,
                                                          sgstkeScalars_d, sgstkeScalarsFrhs_d, moistScalars_d, moistScalarsFrhs_d, moistTauFlds_d,
                                                          J31_d, J32_d, J33_d, invD_Jac_d, zPos_d);

   if ((physics_oneRKonly==0) || (timeStage==numRKstages)) {
     if ((turbulenceSelector >0) && (TKESelector > 0)){
       cudaDevice_hydroCoreUnitTestCompleteSGSTKE<<<grid, tBlock>>>(hydroFlds_d, hydroRhoInv_d, hydroTauFlds_d,
                                                                    hydroKappaM_d, dedxi_d, sgstke_ls_d,
                                                                    sgstkeScalars_d, sgstkeScalarsFrhs_d,
                                                                    J31_d, J32_d, J33_d, D_Jac_d); //call to prognostic TKE equation
     } // end if (turbSelector >0) && (TKESelector > 0)

     if ((moistureSelector > 0)&&(moistureCond > 0)&&(moistureNvars > 1)){ // (moisture condensation forcing)
       temp_freq = roundf(fmaxf(moistureMPcallTscale,dt)/dt); // ensure minimum is time step
       mp_update = simTime_it%temp_freq;
       if (mp_update==0){
         cudaDevice_hydroCoreUnitTestCompleteMP<<<grid, tBlock>>>(hydroFlds_d, hydroFldsFrhs_d, moistScalars_d, moistScalarsFrhs_d,
                                                                    hydroRhoInv_d, hydroPres_d, fcond_d, dt, hydroBaseStateFlds_d);
       }
     }

     if (diffusionSelector == 1){  
       cudaDevice_hydroCoreUnitTestCompleteMolecularDiffusion<<<grid, tBlock>>>(hydroFlds_d, hydroFldsFrhs_d,
                                                                              hydroNuGradXFlds_d,hydroNuGradYFlds_d,hydroNuGradZFlds_d,
                                                                              J31_d, J32_d, J33_d, D_Jac_d, invD_Jac_d); // call to div of nugrad
     } // endif diffusionSelector == 1

   } // endif ((physics_oneRKonly==0) || (timeStage==numRKstages))

   gpuErrchk( cudaGetLastError() );
   gpuErrchk( cudaDeviceSynchronize() );

   simTime_diff = simTime_it - simTime_itRestart;
   ldf_itNum = (int)roundf(lsf_freq/dt);
   if ((lsfSelector==1) && (timeStage==numRKstages) && (simTime_it > simTime_itRestart) && (simTime_diff >= ldf_itNum) && (simTime_it%(int)roundf(lsf_freq/dt)==0)){ // (large-scale forcing)
     temp_freq_fac = (float)roundf(lsf_freq/dt);
     cudaDevice_hydroCoreUnitTestCompleteLSF<<<grid, tBlock>>>(temp_freq_fac, hydroBaseStateFlds_d, lsf_slabMeanPhiProfiles_d, hydroFldsFrhs_d, moistScalarsFrhs_d, zPos_d);
   }

   gpuErrchk( cudaGetLastError() );
   gpuErrchk( cudaDeviceSynchronize() );
#ifdef TIMERS_LEVEL2
   printf("cuda_hydroCoreUnitTestComplete()  Kernel execution time (ms): %12.8f\n", elapsedTime);
#endif

   if ((filterSelector > 0) && ((physics_oneRKonly==0) || (timeStage==numRKstages))){ // explicit filters
     cudaDevice_hydroCoreUnitTestCompleteFilters<<<grid, tBlock>>>(hydroFlds_d,hydroFldsFrhs_d,dt,
                                                                   moistScalars_d,moistScalarsFrhs_d,hydroPres_d,
                                                                   hydroBaseStatePres_d,timeStage);
   }

#ifdef TIMERS_LEVEL1
   stopSynchReportDestroyEvent(&startE, &stopE, &elapsedTime);

   printf("cuda_hydroCoreDeviceBuildFrhs()  Kernel execution time (ms): %12.8f\n", elapsedTime);
#endif
   gpuErrchk( cudaDeviceSynchronize() );
   
   return(errorCode);
}//end cuda_hydroCoreDeviceBuildFrhs()

/*----->>>>> __global__ void  cudaDevice_hydroCoreUnitTestCommence(); ---------------------------------------
* This is the gloabl-entry kernel routine iused by the HYDRO_CORE module
*/
__global__ void cudaDevice_hydroCoreUnitTestCommence(int simTime_it, float* hydroFlds_d, float* hydroFldsFrhs_d, 
                                                     float* hydroBaseStateFlds_d, 
                                                     float* tskin_d, float* qskin_d,
                                                     float* sgstkeScalars_d, float* sgstkeScalarsFrhs_d, float* Km_d,
                                                     float* moistScalars_d, float* moistScalarsFrhs_d, float* zPos_d){
   int iFld;
   int fldStride;
   float* fld;
   float* fldBS;
   float* fldFrhs;
   fldStride = (Nx_d+2*Nh_d)*(Ny_d+2*Nh_d)*(Nz_d+2*Nh_d);


#ifdef CUDA_DEBUG
   int i,j,k;
   /*Establish necessary indices for spatial locality*/
   i = (blockIdx.x)*blockDim.x + threadIdx.x;
   j = (blockIdx.y)*blockDim.y + threadIdx.y;
   k = (blockIdx.z)*blockDim.z + threadIdx.z;
   
   int ijk;
   int iStride,jStride,kStridea;
   iStride = (Ny_d+2*Nh_d)*(Nz_d+2*Nh_d);
   jStride = (Nz_d+2*Nh_d);
   kStride = 1;
//      if((iMin_d!=Nh_d)||(iMax_d!=130)||(jMin_d!=Nh_d)||(jMax_d!=66)||(kMin_d!=Nh_d)||(kMax_d!=34)){
        printf("cudaDevice_hydroCoreUnitTestCommence: (i,j,k)Min_d = (%d,%d,%d) and  (i,j,k)Max_d = (%d,%d,%d)\n",
                iMin_d,jMin_d,kMin_d,iMax_d,jMax_d,kMax_d);
//      }
/*      if( ((i < iMin_d+5)||(i>iMax_d-5))&&((j < jMin_d+5)||(j>jMax_d-5))&&((k < kMin_d+5)||(k>kMax_d-5))){
        printf("cudaDevice_hCUTCommence():float* hydroFlds= %p.and float* hydroRhoInv= %p\n", 
                                              hydroFlds, hydroRhoInv);
      }*/
   if( ((i==iMin_d)||(i==iMax_d-1))&&((j==jMin_d)||(j==jMax_d-1))&&((k==kMin_d)||(k==kMax_d-1)) ){
       printf("%d/%d: rankXid_d,rankYid_d = %d,%d\n",mpi_rank_world_d,mpi_size_world_d,rankXid_d,rankYid_d);
   }
#endif
   /*Set fld and fldBS for configuring BCs everywhere*/
   for(iFld=0; iFld < Nhydro_d; iFld++){
      switch(iFld){
       case 0:
         fld = &hydroFlds_d[fldStride*iFld];
         fldBS = &hydroBaseStateFlds_d[fldStride*RHO_INDX_BS];
         break;
       case 1:
         fld = &hydroFlds_d[fldStride*iFld];
         fldBS = &hydroBaseStateFlds_d[fldStride*RHO_INDX_BS];  //Dummy BS field
         break;
       case 2:
         fld = &hydroFlds_d[fldStride*iFld];
         fldBS = &hydroBaseStateFlds_d[fldStride*RHO_INDX_BS];  //Dummy BS field
         break;
       case 3:
         fld = &hydroFlds_d[fldStride*iFld];
         fldBS = &hydroBaseStateFlds_d[fldStride*RHO_INDX_BS];  //Dummy BS field
         break;
       case 4:
         fld = &hydroFlds_d[fldStride*iFld];
         fldBS = &hydroBaseStateFlds_d[fldStride*THETA_INDX_BS];
         break;
      }
      /*Apply the appropriate boundary conditions*/
      if(hydroBCs_d == 2){
        if (iFld==1 || iFld==2 || iFld==3){
          cudaDevice_VerticalAblBCsMomentum(iFld, fld, fldBS, zPos_d);
        }else{
          cudaDevice_VerticalAblBCs(iFld, fld, fldBS);
        }
        if(numProcsX_d==1){
          cudaDevice_HorizontalPeriodicXdirBCs(iFld, fld);
        }//periodic and single rank in X-dir --> implies no MPI exchanges made so perform on-device exchange
        if(numProcsY_d==1){
          cudaDevice_HorizontalPeriodicYdirBCs(iFld, fld);
        }//periodic and single rank in Y-dir --> implies no MPI exchanges made so perform on-device exchange
      } //end if hydroBCs == ...
      fldFrhs = &hydroFldsFrhs_d[fldStride*iFld];
      cudaDevice_setToZero(fldFrhs);
   }//for iFld
   // BCs and re-initializations for SGSTKE equations
   if ((turbulenceSelector_d > 0)&&(TKESelector_d == 0)){
     fldFrhs = &Km_d[fldStride*0];
     cudaDevice_setToZero(fldFrhs); // resets Km to zero for cumulative Km across SGSTKE equations
   }else if ((turbulenceSelector_d > 0)&&(TKESelector_d > 0)){
     fldFrhs = &Km_d[fldStride*0];
     cudaDevice_setToZero(fldFrhs); // resets Km to zero for cumulative Km across SGSTKE equations
     for(iFld=0; iFld < TKESelector_d; iFld++){
       fldFrhs = &sgstkeScalarsFrhs_d[fldStride*iFld];
       cudaDevice_setToZero(fldFrhs);
       fld = &sgstkeScalars_d[fldStride*iFld];
       fldBS = &sgstkeScalarsFrhs_d[fldStride*iFld]; // set rhs forcing to zero, so it can be used as zero base state
       if (hydroBCs_d == 2){
         cudaDevice_VerticalAblBCs(1, fld, fldBS); // to apply zero-gradient lower boundary BCs
        if(numProcsX_d==1){
          cudaDevice_HorizontalPeriodicXdirBCs(iFld, fld);
        }//periodic and single rank in X-dir --> implies no MPI exchanges made so perform on-device exchange
        if(numProcsY_d==1){
          cudaDevice_HorizontalPeriodicYdirBCs(iFld, fld);
        }//endif periodic and single rank in Y-dir --> implies no MPI exchanges made so perform on-device exchange
       } //end if hydroBCs == ...
     } // end for iFld=0; iFld < TKESelector_d; iFld++
   } // end else if (turbulenceSelector_d > 0) && (TKESelector_d > 0)

   // BCs and re-initializations for moisture equations
   if (moistureSelector_d > 0){
     for(iFld=0; iFld < moistureNvars_d; iFld++){
       fldFrhs = &moistScalarsFrhs_d[fldStride*iFld];
       cudaDevice_setToZero(fldFrhs);
       fld = &moistScalars_d[fldStride*iFld];
       fldBS = &moistScalars_d[fldStride*iFld]; // set rhs forcing to zero, so it can be used as zero base state
       if (hydroBCs_d == 2){
         cudaDevice_VerticalAblZeroGradBCs(fld); // to apply zero-gradient bottom/top BCs
        if(numProcsX_d==1){
          cudaDevice_HorizontalPeriodicXdirBCs(iFld, fld);
        }//endif periodic and single rank in X-dir --> implies no MPI exchanges made so perform on-device exchange
        if(numProcsY_d==1){
          cudaDevice_HorizontalPeriodicYdirBCs(iFld, fld);
        }//endif periodic and single rank in Y-dir --> implies no MPI exchanges made so perform on-device exchange
       } //end if hydroBCs == ...
     } // end for iFld=0; iFld < moistureNvars_d; iFld++
   } // end if (moitureSelector_d > 0)&&(moistureNvars_d == 0)

   //Make sure all threads in a block are synchornized, so halos are filled for core-fields.
   //subsequent function calls need value in halos to compute results from...
   __syncthreads();

} // end cudaDevice_hydroCoreUnitTestCommence()

__global__ void cudaDevice_hydroCoreUnitTestCommenceRhoInvPresPert(float* hydroFlds_d, float* hydroRhoInv_d, 
                                                     float * hydroBaseStateFlds_d, 
                                                     float* hydroPres_d, float* hydroBaseStatePres_d,
                                                     float* moistScalars_d, float* zPos_d){
   int i,j,k,ijk; 
   int iStride,jStride,kStride;
   int fldStride;

   /*Establish necessary indices for spatial locality*/
   i = (blockIdx.x)*blockDim.x + threadIdx.x;
   j = (blockIdx.y)*blockDim.y + threadIdx.y;
   k = (blockIdx.z)*blockDim.z + threadIdx.z;

   iStride = (Ny_d+2*Nh_d)*(Nz_d+2*Nh_d);
   jStride = (Nz_d+2*Nh_d);
   kStride = 1;
   fldStride = (Nx_d+2*Nh_d)*(Ny_d+2*Nh_d)*(Nz_d+2*Nh_d);
   if((i >= iMin_d-Nh_d)&&(i < iMax_d+Nh_d) &&
      (j >= jMin_d-Nh_d)&&(j < jMax_d+Nh_d) &&
      (k >= kMin_d-Nh_d)&&(k < kMax_d+Nh_d) ){
      /* Calculate rho^(-1) */
      ijk = i*iStride + j*jStride + k*kStride;
      cudaDevice_SetRhoInv(&hydroFlds_d[fldStride*RHO_INDX+ijk], &hydroRhoInv_d[ijk]);
   } //end if in the range of cells in the field (halo-inclusive)
   if ((moistureSelector_d > 0)&&(moistureNvars_d > 0)){ // moist pressure calculation
     cudaDevice_calcPerturbationPressureMoist(&hydroPres_d[0], &hydroFlds_d[fldStride*RHO_INDX], &hydroFlds_d[fldStride*THETA_INDX],
                                              &hydroBaseStateFlds_d[fldStride*THETA_INDX_BS], &moistScalars_d[0], zPos_d); // qv
   }else{ // dry pressure calculation
     cudaDevice_calcPerturbationPressure(&hydroPres_d[0], &hydroFlds_d[fldStride*THETA_INDX],
                                         &hydroBaseStateFlds_d[fldStride*THETA_INDX_BS], zPos_d);
   }
} // end cudaDevice_hydroCoreUnitTestCommenceRhoInvPresPert()

__global__ void cudaDevice_hydroCoreUnitTestComplete(float simTime, int simTime_it, float dt, int timeStage, int numRKstages,
                                                     float* hydroFlds, float* hydroFldsFrhs, 
                                                     float* hydroFaceVels, float* hydroBaseStateFlds, 
                                                     float* hydroTauFlds,
                                                     float* sgstkeScalars, float* sgstkeScalarsFrhs, 
                                                     float* moistScalars, float* moistScalarsFrhs, float* moistTauFlds,
                                                     float* J31_d, float* J32_d, float* J33_d, float* invD_Jac_d, float* zPos_d){

   int i,j,k,ijk; 
   int iFld,fldStride;
   int iStride,jStride,kStride;
   float* rho;
   float* rho_BS;
   float* u_cf;
   float* v_cf;
   float* w_cf;
   float* fld;
   float* fldFrhs;
   int TKEAdvSelector_flag;
   float TKEAdvSelector_b_hyb_flag;
   float MomBSval[3];
 
   /*Establish necessary indices for spatial locality*/
   i = (blockIdx.x)*blockDim.x + threadIdx.x;
   j = (blockIdx.y)*blockDim.y + threadIdx.y;
   k = (blockIdx.z)*blockDim.z + threadIdx.z;

   fldStride = (Nx_d+2*Nh_d)*(Ny_d+2*Nh_d)*(Nz_d+2*Nh_d);
   iStride = (Ny_d+2*Nh_d)*(Nz_d+2*Nh_d);
   jStride = (Nz_d+2*Nh_d);
   kStride = 1;
   rho = &hydroFlds[fldStride*RHO_INDX];
   rho_BS = &hydroBaseStateFlds[fldStride*RHO_INDX_BS];
   u_cf = &hydroFaceVels[fldStride*0];
   v_cf = &hydroFaceVels[fldStride*1];
   w_cf = &hydroFaceVels[fldStride*2];

   if((i >= iMin_d)&&(i < iMax_d) &&
      (j >= jMin_d)&&(j < jMax_d) &&
      (k >= kMin_d)&&(k < kMax_d) ){
      for(iFld=0; iFld < Nhydro_d; iFld++){   
         fld = &hydroFlds[fldStride*iFld];
         fldFrhs = &hydroFldsFrhs[fldStride*iFld];
         /* Calculate scalar, cell-valued divergence of the advective flux */
         if (advectionSelector_d == 1) { //  3rd-order QUICK
           cudaDevice_QUICKDivAdvFlux(fld, fldFrhs, u_cf, v_cf, w_cf, invD_Jac_d);
         } else if (advectionSelector_d == 2) { //  hybrid 3rd-4th order
           cudaDevice_HYB34DivAdvFlux(fld, fldFrhs, u_cf, v_cf, w_cf, b_hyb_d, invD_Jac_d);
         } else if (advectionSelector_d == 3) { //  hybrid 5th-6th order
           cudaDevice_HYB56DivAdvFlux(fld, fldFrhs, u_cf, v_cf, w_cf, b_hyb_d, invD_Jac_d);
         } else if (advectionSelector_d == 4) { //  3rd-order WENO
           cudaDevice_WENO3DivAdvFluxX(fld, fldFrhs, u_cf, invD_Jac_d);
           cudaDevice_WENO3DivAdvFluxY(fld, fldFrhs, v_cf, invD_Jac_d);
           cudaDevice_WENO3DivAdvFluxZ(fld, fldFrhs, w_cf, invD_Jac_d);
         } else if (advectionSelector_d == 5) { //  5th-order WENO
           cudaDevice_WENO5DivAdvFluxX(fld, fldFrhs, u_cf, invD_Jac_d);
           cudaDevice_WENO5DivAdvFluxY(fld, fldFrhs, v_cf, invD_Jac_d);
           cudaDevice_WENO5DivAdvFluxZ(fld, fldFrhs, w_cf, invD_Jac_d);
         } else if (advectionSelector_d == 6) { //  centered 2nd-order
           cudaDevice_SecondDivAdvFlux(fld, fldFrhs, u_cf, v_cf, w_cf, invD_Jac_d);
         } else { // defaults to 1st-order upwinding
           cudaDevice_UpstreamDivAdvFlux(fld, fldFrhs, u_cf, v_cf, w_cf, invD_Jac_d);
         }
         if(iFld==W_INDX){
           if(dampingLayerSelector_d > 0){          // RAYLEIGH DAMPING ON W   ******!!!!!!!!
             cudaDevice_topRayleighDampingLayerForcing(fld, fldFrhs,
                                                       &rho[0], &rho_BS[0], zPos_d);
           }  //end if dampingLayerSelector > 0 
           if(buoyancySelector_d > 0){              // BUOYANCY SOURCE?SINK OF W   ******!!!!!!!!
             ijk = i*iStride + j*jStride + k*kStride;
             if (moistureSelector_d>0){
                if(moistureNvars_d==1){ 
                  cudaDevice_calcBuoyancyMoistNvar1(&fldFrhs[ijk], &rho[ijk], &rho_BS[ijk],&moistScalars[ijk]);
                }else if(moistureNvars_d==2){ 
                  cudaDevice_calcBuoyancyMoistNvar2(&fldFrhs[ijk], &rho[ijk], &rho_BS[ijk],&moistScalars[ijk],&moistScalars[fldStride+ijk]);
                }
             }else{
               cudaDevice_calcBuoyancy(&fldFrhs[ijk], &rho[ijk], &rho_BS[ijk]);
             }
           }  //end if buoyancySelector > 0 
         }//end if iFld==W_INDX
      }//for iFld
      if ((turbulenceSelector_d>0) && (TKESelector_d>0)){ // : advection of SGSTKE fields
        for(iFld=0; iFld < TKESelector_d; iFld++){
          fld = &sgstkeScalars[fldStride*iFld];
          fldFrhs = &sgstkeScalarsFrhs[fldStride*iFld];
          TKEAdvSelector_flag = TKEAdvSelector_d;
          TKEAdvSelector_b_hyb_flag = TKEAdvSelector_b_hyb_d;
          if (TKEAdvSelector_flag == 1) { //  3rd-order QUICK
            cudaDevice_QUICKDivAdvFlux(fld, fldFrhs, u_cf, v_cf, w_cf, invD_Jac_d);
          } else if (TKEAdvSelector_flag == 2) { //  hybrid 3rd-4th order
            cudaDevice_HYB34DivAdvFlux(fld, fldFrhs, u_cf, v_cf, w_cf, TKEAdvSelector_b_hyb_flag, invD_Jac_d);
          } else if (TKEAdvSelector_flag == 3) { //  hybrid 5th-6th order
            cudaDevice_HYB56DivAdvFlux(fld, fldFrhs, u_cf, v_cf, w_cf, TKEAdvSelector_b_hyb_flag, invD_Jac_d);
          } else if (TKEAdvSelector_flag == 4) { //  3rd-order WENO
            cudaDevice_WENO3DivAdvFluxX(fld, fldFrhs, u_cf, invD_Jac_d);
            cudaDevice_WENO3DivAdvFluxY(fld, fldFrhs, v_cf, invD_Jac_d);
            cudaDevice_WENO3DivAdvFluxZ(fld, fldFrhs, w_cf, invD_Jac_d);
          } else if (TKEAdvSelector_flag == 5) { //  5th-order WENO
            cudaDevice_WENO5DivAdvFluxX(fld, fldFrhs, u_cf, invD_Jac_d);
            cudaDevice_WENO5DivAdvFluxY(fld, fldFrhs, v_cf, invD_Jac_d);
            cudaDevice_WENO5DivAdvFluxZ(fld, fldFrhs, w_cf, invD_Jac_d);
          } else if (TKEAdvSelector_flag == 6) { //  centered 2nd-order
            cudaDevice_SecondDivAdvFlux(fld, fldFrhs, u_cf, v_cf, w_cf, invD_Jac_d);
          } else { // defaults to 1st-order upwinding
            cudaDevice_UpstreamDivAdvFlux(fld, fldFrhs, u_cf, v_cf, w_cf, invD_Jac_d);
          }
        }
      }

      if ((moistureSelector_d>0) && (moistureNvars_d>0)){ // : advection of moisture fields
        for(iFld=0; iFld < moistureNvars_d; iFld++){
          fld = &moistScalars[fldStride*iFld];
          fldFrhs = &moistScalarsFrhs[fldStride*iFld];
          if (iFld==0){ // water vapor
            if (moistureAdvSelectorQv_d == 1) { //  3rd-order QUICK
              cudaDevice_QUICKDivAdvFlux(fld, fldFrhs, u_cf, v_cf, w_cf, invD_Jac_d);
            } else if (moistureAdvSelectorQv_d == 2) { //  hybrid 3rd-4th order
              cudaDevice_HYB34DivAdvFlux(fld, fldFrhs, u_cf, v_cf, w_cf, moistureAdvSelectorQv_b_d, invD_Jac_d);
            } else if (moistureAdvSelectorQv_d == 3) { //  hybrid 5th-6th order
              cudaDevice_HYB56DivAdvFlux(fld, fldFrhs, u_cf, v_cf, w_cf, moistureAdvSelectorQv_b_d, invD_Jac_d);
            } else if (moistureAdvSelectorQv_d == 4) { //  3rd-order WENO
              cudaDevice_WENO3DivAdvFluxX(fld, fldFrhs, u_cf, invD_Jac_d);
              cudaDevice_WENO3DivAdvFluxY(fld, fldFrhs, v_cf, invD_Jac_d);
              cudaDevice_WENO3DivAdvFluxZ(fld, fldFrhs, w_cf, invD_Jac_d);
            } else if (moistureAdvSelectorQv_d == 5) { //  5th-order WENO
              cudaDevice_WENO5DivAdvFluxX(fld, fldFrhs, u_cf, invD_Jac_d);
              cudaDevice_WENO5DivAdvFluxY(fld, fldFrhs, v_cf, invD_Jac_d);
              cudaDevice_WENO5DivAdvFluxZ(fld, fldFrhs, w_cf, invD_Jac_d);
            } else if (moistureAdvSelectorQv_d == 6) { //  centered 2nd-order
              cudaDevice_SecondDivAdvFlux(fld, fldFrhs, u_cf, v_cf, w_cf, invD_Jac_d);
            } else { // defaults to 1st-order upwinding
              cudaDevice_UpstreamDivAdvFlux(fld, fldFrhs, u_cf, v_cf, w_cf, invD_Jac_d);
            }
          } else { // non-qv moisture species (non-oscillatory schemes)
            if (moistureAdvSelectorQi_d == 0) { // 1st-order upstream
              cudaDevice_UpstreamDivAdvFlux(fld, fldFrhs, u_cf, v_cf, w_cf, invD_Jac_d);
            } else if (moistureAdvSelectorQi_d == 1) { // 3rd-order WENO
              cudaDevice_WENO3DivAdvFluxX(fld, fldFrhs, u_cf, invD_Jac_d);
              cudaDevice_WENO3DivAdvFluxY(fld, fldFrhs, v_cf, invD_Jac_d);
              cudaDevice_WENO3DivAdvFluxZ(fld, fldFrhs, w_cf, invD_Jac_d);
            } else if (moistureAdvSelectorQi_d == 2) { // 5th-order WENO
              cudaDevice_WENO5DivAdvFluxX(fld, fldFrhs, u_cf, invD_Jac_d);
              cudaDevice_WENO5DivAdvFluxY(fld, fldFrhs, v_cf, invD_Jac_d);
              cudaDevice_WENO5DivAdvFluxZ(fld, fldFrhs, w_cf, invD_Jac_d);
            }
          }
        }
      }

      if(coriolisSelector_d > 0){
        ijk = i*iStride + j*jStride + k*kStride;
        cudaDevice_MomentumBS(U_INDX, zPos_d[ijk], &hydroBaseStateFlds[RHO_INDX_BS*fldStride+ijk], &MomBSval[0]);
        cudaDevice_MomentumBS(V_INDX, zPos_d[ijk], &hydroBaseStateFlds[RHO_INDX_BS*fldStride+ijk], &MomBSval[1]);
        cudaDevice_MomentumBS(W_INDX, zPos_d[ijk], &hydroBaseStateFlds[RHO_INDX_BS*fldStride+ijk], &MomBSval[2]);
        cudaDevice_calcCoriolis(&hydroFldsFrhs[U_INDX*fldStride+ijk], 
                                &hydroFldsFrhs[V_INDX*fldStride+ijk],
                                &hydroFldsFrhs[W_INDX*fldStride+ijk],
                                &hydroFlds[RHO_INDX*fldStride+ijk],
                                &hydroFlds[U_INDX*fldStride+ijk], 
                                &hydroFlds[V_INDX*fldStride+ijk], 
                                &hydroFlds[W_INDX*fldStride+ijk],
                                &hydroBaseStateFlds[RHO_INDX_BS*fldStride+ijk],
                                &MomBSval[0],
                                &MomBSval[1],
                                &MomBSval[2]);
      }  //end if coriolisSelector_d > 0 
   }//end if in the range of non-halo cells
   if((turbulenceSelector_d > 0) && ((physics_oneRKonly_d==0) || (timeStage==numRKstages))){
     cudaDevice_hydroCoreCalcTurbMixing(
                                       &hydroFldsFrhs[fldStride*U_INDX], 
                                       &hydroFldsFrhs[fldStride*V_INDX], 
                                       &hydroFldsFrhs[fldStride*W_INDX],
                                       &hydroFldsFrhs[fldStride*THETA_INDX],
                                   &hydroTauFlds[fldStride*0], &hydroTauFlds[fldStride*1], &hydroTauFlds[fldStride*2],
                                   &hydroTauFlds[fldStride*3], &hydroTauFlds[fldStride*4], &hydroTauFlds[fldStride*5],
                                   &hydroTauFlds[fldStride*6], &hydroTauFlds[fldStride*7], &hydroTauFlds[fldStride*8],
                                       J31_d, J32_d, J33_d);
     if ((moistureSelector_d > 0) && (moistureSGSturb_d > 0)){
       for(iFld=0; iFld < moistureNvars_d; iFld++){ // loop over moisture equations
           cudaDevice_hydroCoreCalcTurbMixingScalar(&moistScalarsFrhs[fldStride*iFld], &moistTauFlds[fldStride*(3*iFld+0)],
                                                    &moistTauFlds[fldStride*(3*iFld+1)], &moistTauFlds[fldStride*(3*iFld+2)],
                                                    J31_d, J32_d, J33_d);
       }
     }
   }  //end if turbulenceSelector_d > 0

} // end cudaDevice_hydroCoreUnitTestComplete()

__global__ void cudaDevice_hydroCoreCalcFaceVelocities(float simTime, int simTime_it, int simTime_itRestart, 
                                                       float dt,int timeStage, int numRKstages,
                                                       float* hydroFlds_d, float* hydroFldsFrhs_d,
                                                       float* hydroFaceVels_d, float* hydroPres_d,
                                                       float* hydroNuGradXFlds_d, float* hydroNuGradYFlds_d,
                                                       float* hydroNuGradZFlds_d, float* hydroTauFlds_d,
                                                       float* cdFld_d, float* chFld_d, float* cqFld_d, float* fricVel_d,
                                                       float* htFlux_d, float* tskin_d, float* invOblen_d, 
                                                       float* z0m_d, float* z0t_d, float* qFlux_d, float* qskin_d, float* sea_mask_d,
                                                       float* hydroRhoInv_d, float* hydroKappaM_d, float* sgstkeScalars_d, float* sgstke_ls_d,
                                                       float* dedxi_d, float* moistScalars_d, float* moistTauFlds_d,
                                                       float* moistScalarsFrhs_d,
                                                       float* J31_d, float* J32_d, float* J33_d, float* D_Jac_d){
   int fldStride;
   float inv_pr; 
   int iFld; 

   fldStride = (Nx_d+2*Nh_d)*(Ny_d+2*Nh_d)*(Nz_d+2*Nh_d);

   //### ADVECTION ###//
   /* Calculate the cell-face velocity components to prepare for advection in later phases of build_Frhs*/
   cudaDevice_calcFaceVelocities(hydroFlds_d, hydroFaceVels_d, 
                                 J31_d, J32_d, J33_d, D_Jac_d);
   //### MOLECULAR DIFFUSION ###//
   if((diffusionSelector_d > 0) && ((physics_oneRKonly_d==0) || (timeStage==numRKstages))){ // calculate NuGrad fields
     for(iFld=1; iFld < Nhydro_d; iFld++){   //NOTE: core progrnotic variables excluding rho, so (u,v,w,theta) 
        if (iFld==THETA_INDX){ // theta
          inv_pr = 1.0/0.71;
        }else{ 
          inv_pr = 1.0;
        }  
        cudaDevice_diffusionDriver(&hydroFlds_d[fldStride*iFld], 
                                   &hydroNuGradXFlds_d[fldStride*(iFld-1)], 
                                   &hydroNuGradYFlds_d[fldStride*(iFld-1)], 
                                   &hydroNuGradZFlds_d[fldStride*(iFld-1)], 
                                   inv_pr,
                                   J31_d, J32_d, J33_d, D_Jac_d);
     } // end for (iFld=1; iFld < Nhydro_d; iFld++){
   } // end if diffusionSelector_d > 0

   //### PRESSURE GRADIENT FORCE ###//
   if(pgfSelector_d > 0){

     if((moistureSelector_d > 0)&&(moistureNvars_d > 0)){ // moist pressure gradient force
       cudaDevice_calcPressureGradientForceMoist(&hydroFldsFrhs_d[fldStride*U_INDX], &hydroFldsFrhs_d[fldStride*V_INDX],
                                                 &hydroFldsFrhs_d[fldStride*W_INDX], &hydroFlds_d[fldStride*RHO_INDX], &hydroPres_d[0],
                                                 &moistScalars_d[0],
                                                 J31_d, J32_d, J33_d);
     }else{ // dry pressure gradient force
       cudaDevice_calcPressureGradientForce(&hydroFldsFrhs_d[fldStride*U_INDX], &hydroFldsFrhs_d[fldStride*V_INDX],
                                            &hydroFldsFrhs_d[fldStride*W_INDX], &hydroPres_d[0],
                                            J31_d, J32_d, J33_d);
     } // end if (moistureSelector_d > 0)&&(moistureNvars_d > 0)
   } //end if pgfSelector_d > 0

   //### TURBULENCE ###//
   if((turbulenceSelector_d > 0) && ((physics_oneRKonly_d==0) || (timeStage==numRKstages))){
     cudaDevice_hydroCoreCalcStrainRateElements(&hydroFlds_d[fldStride*U_INDX],
                                                &hydroFlds_d[fldStride*V_INDX],
                                                &hydroFlds_d[fldStride*W_INDX],
                                                &hydroFlds_d[fldStride*THETA_INDX],
                                                &hydroTauFlds_d[fldStride*0], &hydroTauFlds_d[fldStride*1], &hydroTauFlds_d[fldStride*2],
                                                &hydroTauFlds_d[fldStride*3], &hydroTauFlds_d[fldStride*4], &hydroTauFlds_d[fldStride*5],
                                                &hydroTauFlds_d[fldStride*6], &hydroTauFlds_d[fldStride*7], &hydroTauFlds_d[fldStride*8],
                                                J31_d, J32_d, J33_d,
                                                &hydroRhoInv_d[0]);
     if ((moistureSelector_d > 0) && (moistureSGSturb_d > 0)){
       for(iFld=0; iFld < moistureNvars_d; iFld++){ // loop over moisture equations
         cudaDevice_GradScalarToFaces(&moistScalars_d[fldStride*iFld], &hydroRhoInv_d[0], &moistTauFlds_d[fldStride*(3*iFld+0)],
                                      &moistTauFlds_d[fldStride*(3*iFld+1)], &moistTauFlds_d[fldStride*(3*iFld+2)],
                                      J31_d, J32_d, J33_d);
       }
     }
     if(TKESelector_d > 0){ // Lilly SGSTKE based Km
       for(iFld=0; iFld < TKESelector_d; iFld++){ // loop over SGSTKE equations
         cudaDevice_sgstkeLengthScale(&hydroFlds_d[fldStride*THETA_INDX], &hydroRhoInv_d[0],
                                      &sgstkeScalars_d[fldStride*iFld], &sgstke_ls_d[fldStride*iFld],
                                      J31_d, J32_d, J33_d, D_Jac_d); // length scale calculation
         cudaDevice_hydroCoreCalcEddyDiff(&hydroTauFlds_d[fldStride*0], &hydroTauFlds_d[fldStride*1], &hydroTauFlds_d[fldStride*2],
                                          &hydroTauFlds_d[fldStride*3], &hydroTauFlds_d[fldStride*4], &hydroTauFlds_d[fldStride*5],
                                          &hydroTauFlds_d[fldStride*8], &hydroFlds_d[fldStride*THETA_INDX], &hydroRhoInv_d[0],
                                          &sgstkeScalars_d[fldStride*iFld], &sgstke_ls_d[fldStride*iFld], &hydroKappaM_d[0], D_Jac_d); // calculate Km
         cudaDevice_GradScalar(&sgstkeScalars_d[fldStride*iFld], &hydroRhoInv_d[0],
                               &dedxi_d[fldStride*(iFld*3+0)], &dedxi_d[fldStride*(iFld*3+1)], &dedxi_d[fldStride*(iFld*3+2)],
                               J31_d, J32_d, J33_d); // calculate SGSTKE spatial gradients
       }
       cudaDevice_hydroCoreCalcTaus_PrognosticTKE_DeviatoricTerm(
                                  &hydroTauFlds_d[fldStride*0], &hydroTauFlds_d[fldStride*1], &hydroTauFlds_d[fldStride*2],
                                  &hydroTauFlds_d[fldStride*3], &hydroTauFlds_d[fldStride*4], &hydroTauFlds_d[fldStride*5],
                                  &hydroTauFlds_d[fldStride*6], &hydroTauFlds_d[fldStride*7], &hydroTauFlds_d[fldStride*8],
                                  &hydroFlds_d[fldStride*RHO_INDX], &hydroKappaM_d[0], &sgstke_ls_d[0],
                                  &hydroFlds_d[fldStride*U_INDX], &hydroFlds_d[fldStride*V_INDX], &hydroFlds_d[fldStride*W_INDX],
                                  &sgstkeScalars_d[0],
                                  J31_d, J32_d, J33_d, D_Jac_d); // calculate tau_ij, tau_thj (sub-grid length scale and TKE from TKE_0)
     }else{ // Samagorinsky Km
       cudaDevice_hydroCoreCalcEddyDiff(&hydroTauFlds_d[fldStride*0], &hydroTauFlds_d[fldStride*1], &hydroTauFlds_d[fldStride*2],
                                        &hydroTauFlds_d[fldStride*3], &hydroTauFlds_d[fldStride*4], &hydroTauFlds_d[fldStride*5],
                                        &hydroTauFlds_d[fldStride*8], &hydroFlds_d[fldStride*THETA_INDX], &hydroRhoInv_d[0],
                                        &sgstkeScalars_d[0], &sgstke_ls_d[0], &hydroKappaM_d[0], D_Jac_d); // calculate Km
       cudaDevice_hydroCoreCalcTaus(&hydroTauFlds_d[fldStride*0], &hydroTauFlds_d[fldStride*1], &hydroTauFlds_d[fldStride*2],
                                    &hydroTauFlds_d[fldStride*3], &hydroTauFlds_d[fldStride*4], &hydroTauFlds_d[fldStride*5],
                                    &hydroTauFlds_d[fldStride*6], &hydroTauFlds_d[fldStride*7], &hydroTauFlds_d[fldStride*8],
                                    &hydroFlds_d[fldStride*RHO_INDX], &hydroKappaM_d[0], &sgstke_ls_d[0], D_Jac_d); // calculate tau_ij, tau_thj (sub-grid length scale is the one from TKE_0)
     }
     if((moistureSelector_d > 0) && (moistureSGSturb_d > 0)){
       for(iFld=0; iFld < moistureNvars_d; iFld++){ // loop over moisture equations
         cudaDevice_hydroCoreCalcTausScalar(&moistTauFlds_d[fldStride*(3*iFld+0)], &moistTauFlds_d[fldStride*(3*iFld+1)],
                                            &moistTauFlds_d[fldStride*(3*iFld+2)], &hydroFlds_d[fldStride*RHO_INDX],
                                            &hydroKappaM_d[0], &sgstke_ls_d[0], D_Jac_d); // calculate tau_mj
       }
     }
   } //end if turbulenceSelector_d > 0

   //### SURFACELAYER ###//
   int i,j,k,ijk,ij;
   int iStride2d,jStride2d;
   int iStride,jStride,kStride;

   /*Establish necessary indices for spatial locality*/
   i = (blockIdx.x)*blockDim.x + threadIdx.x;
   j = (blockIdx.y)*blockDim.y + threadIdx.y;
   k = (blockIdx.z)*blockDim.z + threadIdx.z;
   iStride2d = (Ny_d+2*Nh_d);
   jStride2d = 1;
   iStride = (Ny_d+2*Nh_d)*(Nz_d+2*Nh_d);
   jStride = (Nz_d+2*Nh_d);
   kStride = 1;
   if((i >= iMin_d)&&(i <= iMax_d+1) &&
      (j >= jMin_d)&&(j <= jMax_d+1) &&
      (k == kMin_d) ){
     if((surflayerSelector_d > 0) && ((physics_oneRKonly_d==0) || (timeStage==numRKstages))){ // call to surface layer parameterization
       ijk = i*iStride + j*jStride + k*kStride;
       ij = i*iStride2d + j*jStride2d; // 2-dimensional (horizontal index)

       if(moistureSelector_d > 0){ // MOIST surface layer model
         // land-surface model
         cudaDevice_SurfaceLayerLSMmoist(simTime, simTime_it, simTime_itRestart,
                                         dt, timeStage, numRKstages, ijk,
                                         &hydroFlds_d[U_INDX*fldStride+ijk], &hydroFlds_d[V_INDX*fldStride+ijk],
                                         &hydroFlds_d[RHO_INDX*fldStride+ijk], &hydroFlds_d[THETA_INDX*fldStride+ijk], &moistScalars_d[0*fldStride+ijk],
                                         &cdFld_d[ij], &chFld_d[ij], &cqFld_d[ij], &fricVel_d[ij],
                                         &htFlux_d[ij], &tskin_d[ij], &qFlux_d[ij], &qskin_d[ij],
                                         &z0m_d[ij], &z0t_d[ij], J33_d);

         cudaDevice_SurfaceLayerMOSTmoist(ijk, &hydroFlds_d[U_INDX*fldStride+ijk], &hydroFlds_d[V_INDX*fldStride+ijk],
                                          &hydroFlds_d[RHO_INDX*fldStride+ijk], &hydroFlds_d[THETA_INDX*fldStride+ijk], &moistScalars_d[0*fldStride+ijk],
                                          &hydroTauFlds_d[2*fldStride+ijk], &hydroTauFlds_d[3*fldStride+ijk], &hydroTauFlds_d[8*fldStride+ijk], &moistTauFlds_d[2*fldStride+ijk],
                                          &cdFld_d[ij], &chFld_d[ij], &cqFld_d[ij], &fricVel_d[ij],
                                          &htFlux_d[ij], &tskin_d[ij], &qFlux_d[ij], &qskin_d[ij],
                                          &invOblen_d[ij], &z0m_d[ij], &z0t_d[ij], &sea_mask_d[ij], J33_d);
       }else{ // DRY surface layer model
         // land-surface model
         cudaDevice_SurfaceLayerLSMdry(simTime, simTime_it, simTime_itRestart,
                                       dt, timeStage, numRKstages, ijk,
                                       &hydroFlds_d[U_INDX*fldStride+ijk], &hydroFlds_d[V_INDX*fldStride+ijk],
                                       &hydroFlds_d[RHO_INDX*fldStride+ijk], &hydroFlds_d[THETA_INDX*fldStride+ijk],
                                       &cdFld_d[ij], &chFld_d[ij], &fricVel_d[ij], &htFlux_d[ij], &tskin_d[ij],
                                       &z0m_d[ij], &z0t_d[ij], J33_d);
         cudaDevice_SurfaceLayerMOSTdry(ijk, &hydroFlds_d[U_INDX*fldStride+ijk], &hydroFlds_d[V_INDX*fldStride+ijk],
                                        &hydroFlds_d[RHO_INDX*fldStride+ijk], &hydroFlds_d[THETA_INDX*fldStride+ijk],
                                        &hydroTauFlds_d[2*fldStride+ijk], &hydroTauFlds_d[3*fldStride+ijk], &hydroTauFlds_d[8*fldStride+ijk],
                                        &cdFld_d[ij], &chFld_d[ij], &fricVel_d[ij],
                                        &htFlux_d[ij], &tskin_d[ij], &invOblen_d[ij], &z0m_d[ij], &z0t_d[ij], &sea_mask_d[ij], J33_d);
       } // end if moistureSelector_d
     } //end if surflayerSelector_d > 0 && k == kMin_d
   } //end if in the range of non-halo surface cells

} //end cudaDevice_hydroCoreCalcFaceVelocities
 
/*----->>>>> __device__ void  cudaDevice_SetRhoInv();  --------------------------------------------------
* This is the cuda version of the SetRhoInv routine from the HYDRO_CORE module
*/
__device__ void cudaDevice_SetRhoInv(float* rhoFld, float* hydroRhoInv){
  float epsilon = 1e-6;
   
      *hydroRhoInv = 1.0/fmaxf(*rhoFld,epsilon);     

} // end cudaDevice_SetRhoInv()

__device__ void cudaDevice_setToZero(float* fld){
  int i,j,k;
  int ijk;
  int iStride,jStride,kStride;

  i = (blockIdx.x)*blockDim.x + threadIdx.x;
  j = (blockIdx.y)*blockDim.y + threadIdx.y;
  k = (blockIdx.z)*blockDim.z + threadIdx.z;
  iStride = (Ny_d+2*Nh_d)*(Nz_d+2*Nh_d);
  jStride = (Nz_d+2*Nh_d);
  kStride = 1;
  ijk = i*iStride + j*jStride + k*kStride;
  fld[ijk] = 0.0;
} //end cudaDevice_setToZero
