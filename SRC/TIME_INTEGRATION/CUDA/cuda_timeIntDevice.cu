/* FastEddy®: SRC/TIME_INTEGRATION/CUDA/cuda_timeIntDevice.cu 
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
#include <time_integration.h>
#include <fecuda_Device_cu.h>
#include <cuda_gridDevice_cu.h>
#include <cuda_hydroCoreDevice_cu.h>
#include <cuda_timeIntDevice_cu.h>

#include "cuda_RKschemes.cu"

/*#################------------- CUDA_TIME_INTEGRATION module variable definitions ------------------#############*/
/*Parameters*/
__constant__ int timeMethod_d;   // Selector for time integration method. (default: 1= 3rd-order Runge-Kutta )
__constant__ int Nt_d;       // Number of timesteps to perform
__constant__ int NtimeTotVars_d;  // Total Number of prognostic variables to be integrated over time
__constant__ int NtBatch_d;  // Number of timesteps in a batch to perform in a CUDA kernel launch
__constant__ float dt_d;     // timestep resolution in seconds
__constant__ int simTime_itRestart_d;           //Timestep at restart (0 at start) 

/* array fields */
float *timeFlds0_d;   /* Multistage time scheme variable fields 4-D array */
float *timeFrhs0_d;   /* Multistage time scheme variable fields Frhs 4-D array */
float *timeFrhsTmp_d; /* Multistage time scheme variable fields Frhs 4-D array */

/*#################------------- CUDA_TIME_INTEGRATION module function definitions ------------------#############*/
/*----->>>>> int cuda_timeIntDeviceSetup();       ----------------------------------------------------------------
 * Used to cudaMalloc and cudaMemcpy parameters and coordinate arrays, and for the TIME_INTEGRATION_CUDA module.
*/
extern "C" int cuda_timeIntDeviceSetup(){
   int errorCode = CUDA_TIME_INTEGRATION_SUCCESS;
   int Nelems;
   int NtimeTotVars;
  
   /*Synchronize the Device*/
   gpuErrchk( cudaDeviceSynchronize() );
 
   /*Constants*/
   /* timeMethod, total timesteps, batch timesteps, timestep resolution */
   cudaMemcpyToSymbol(timeMethod_d, &timeMethod, sizeof(int));
   cudaMemcpyToSymbol(Nt_d, &Nt, sizeof(int));
   cudaMemcpyToSymbol(NtBatch_d, &NtBatch, sizeof(int));
   cudaMemcpyToSymbol(dt_d, &dt, sizeof(float));
   cudaMemcpyToSymbol(simTime_itRestart_d, &simTime_itRestart, sizeof(int));
   gpuErrchk( cudaPeekAtLastError() ); /*Check for errors in the cudaMemCpy calls*/

   /*Set the full memory block number of elements for timeInt fields*/
   Nelems = (Nxp+2*Nh)*(Nyp+2*Nh)*(Nzp+2*Nh); 
   /* Allocate the TIME_INTEGRATION arrays */
   /*TIME_INTEGRATION/CUDA internal device arrays*/
   NtimeTotVars = 5 + TKESelector*turbulenceSelector + moistureNvars*moistureSelector; 
   fecuda_DeviceMalloc(NtimeTotVars*Nelems*sizeof(float), &timeFlds0_d);
   
   gpuErrchk( cudaPeekAtLastError() ); /*Check for errors in the cudaMalloc calls*/

   //Ensure secondary time-integration dependent hydro_core parameters get initialized
   errorCode = cuda_hydroCoreDeviceSecondaryStageSetup(dt);
   //Inital Host-to-Device field copies 
   errorCode = cuda_timeIntHydroInitDevice();  //Transfer initial/restart conditions to the device
   //printf("cuda_timeIntDeviceSetup() complete.\n");

   /* Done */
   return(errorCode);
} //end cuda_timeIntDeviceSetup()

/*----->>>>> extern "C" int cuda_timeIntDeviceCleanup();  -----------------------------------------------------------
Used to free all malloced memory by the TIME_INTEGRATION module.
*/
extern "C" int cuda_timeIntDeviceCleanup(){
   int errorCode = TIME_INTEGRATION_SUCCESS;

   /* Free any TIME_INTEGRATION module arrays */
   cudaFree(timeFlds0_d); 
   gpuErrchk( cudaPeekAtLastError() ); /*Check for errors in the cudaMemCpy calls*/
   return(errorCode);

}//end cuda_timeIntDeviceCleanup()

/*----->>>>> extern "C" int cuda_timeIntDeviceCommence();  -----------------------------------------------------------
* This routine provides the externally callable cuda-kernel call to commence with timeIntegration
*/
extern "C" int cuda_timeIntDeviceCommence(int it){
   int errorCode = TIME_INTEGRATION_SUCCESS;
   int itBatch;
   int RKstage;
#ifdef TIMERS_LEVEL1
   float elapsedTime;
   cudaEvent_t startE, stopE
#endif

   /*Synchronize the Device*/
   gpuErrchk( cudaDeviceSynchronize() );

   //If this is the very first batch timestep (initial condition or restart) then 
   // make final preparations for simulation launch.
   if(it==simTime_itRestart){
      /*Copy in the master simulation time at simulation start*/
      printf("cuda_hydroCoreUnitTestCommence()  it=simTime_itRestart.\n");
      errorCode = cuda_timeIntHydroInitDevice();  //Transfer initial restart conditions to the device
   }//end if it==0
   for(itBatch=0; itBatch < NtBatch; itBatch++){     //Batch timestepping loop
     if((lsfSelector == 1) && (lsf_horMnSubTerms == 1) && (simTime_it > simTime_itRestart) && (simTime_it%(int)roundf(lsf_freq/dt)==0)){
       errorCode = cuda_lsfSlabMeans();
     }
     gpuErrchk( cudaDeviceSynchronize() );
     /*Execute the timeMethod kernel of choice on the GPU*/
     if(timeMethod == 0){    /*******  Issue the  3rd-order Runge-Kutta WS2002 **************/
       for(RKstage=0; RKstage < 3; RKstage++){
          /*Build the right hand side forcing*/
          errorCode = cuda_hydroCoreDeviceBuildFrhs(simTime,simTime_it,simTime_itRestart,dt,RKstage,numRKstages);
          /*Perform the time integration*/
#ifdef TIMERS_LEVEL1
          createAndStartEvent(&startE, &stopE);
#endif
          cudaDevice_timeIntegrationCommenceRK3_WS2002<<<grid, tBlock>>>(Nhydro, hydroFlds_d, hydroFldsFrhs_d,
                                                TKESelector*turbulenceSelector, sgstkeScalars_d, sgstkeScalarsFrhs_d,
                                                moistureNvars*moistureSelector, moistScalars_d, moistScalarsFrhs_d,
                                                timeFlds0_d, RKstage);
          gpuErrchk( cudaGetLastError() );
#ifdef TIMERS_LEVEL1
          stopSynchReportDestroyEvent(&startE, &stopE, &elapsedTime);
          printf("cuda_timeIntCommenceRK3_WS2002()  Kernel execution time (ms): %12.8f\n", elapsedTime);
#endif
       } //end for RKstage 
     } //end if(timeMethod == 0){...
     simTime = simTime + dt;   //Increment the master simulation time*/
     simTime_it = simTime_it + 1;   //Increment the master simulation time step*/
   }//end for itBatch...

   //Retrieve desired HYDRO_CORE fields from device
   errorCode = cuda_timeIntHydroSynchFromDevice();
   
   return(errorCode);
}//end cuda_timeIntDeviceCommence()

/*----->>>>> extern "C" int cuda_timeIntHydroInitDevice();  -----------------------------------------------------------
* This function handles the one-time initializations of on-device (GPU) memory by executing the appropriate sequence 
* of cudaMemcpyHostToDevice data transfers.
*/
extern "C" int cuda_timeIntHydroInitDevice(){
   int errorCode = TIME_INTEGRATION_SUCCESS;
   int Nelems;
   int Nelems2d;
   /*Set the full memory block number of elements for transfers of 2-d and 3-d fields*/
   Nelems = (Nxp+2*Nh)*(Nyp+2*Nh)*(Nzp+2*Nh);
   Nelems2d = (Nxp+2*Nh)*(Nyp+2*Nh);
   /*Copy the host hydroFlds to the device */
   cudaMemcpy(hydroFlds_d, hydroFlds, Nelems*Nhydro*sizeof(float), cudaMemcpyHostToDevice);
   if(TKESelector > 0){ /*Copy any required SGS TKE equation fields to device */ 
     cudaMemcpy(sgstkeScalars_d, sgstkeScalars, Nelems*TKESelector*sizeof(float), cudaMemcpyHostToDevice);
   }
   if(moistureSelector > 0){ /*Copy any required moisture fields to device */ 
     cudaMemcpy(moistScalars_d, moistScalars, Nelems*moistureNvars*sizeof(float), cudaMemcpyHostToDevice);
   }
   if(surflayerSelector > 0){ /*Copy any required host auxiliary sclar fields to the device */
     cudaMemcpy(tskin_d, tskin, Nelems2d*sizeof(float), cudaMemcpyHostToDevice);
     cudaMemcpy(fricVel_d, fricVel, Nelems2d*sizeof(float), cudaMemcpyHostToDevice);
     cudaMemcpy(htFlux_d, htFlux, Nelems2d*sizeof(float), cudaMemcpyHostToDevice);
     cudaMemcpy(z0m_d, z0m, Nelems2d*sizeof(float), cudaMemcpyHostToDevice);
     cudaMemcpy(z0t_d, z0t, Nelems2d*sizeof(float), cudaMemcpyHostToDevice);
     if (moistureSelector > 0){
       cudaMemcpy(qskin_d, qskin, Nelems2d*sizeof(float), cudaMemcpyHostToDevice);
       cudaMemcpy(qFlux_d, qFlux, Nelems2d*sizeof(float), cudaMemcpyHostToDevice);
     }
   }// end if surflayerSelector > 0
   gpuErrchk( cudaPeekAtLastError() ); /*Check for errors in the cudaMemCpy calls*/
   gpuErrchk( cudaDeviceSynchronize() );
   return(errorCode);
}//end cuda_timeIntHydroInitDevice()

/*----->>>>> extern "C" int cuda_timeIntHydroSynchFromDevice();  --------------------------------------------------
* This function handles the synchronization to host of on-device (GPU) fields  by executing the appropriate sequence 
* of cudaMemcpyDeviceiToHost data transfers.
*/
extern "C" int cuda_timeIntHydroSynchFromDevice(){
   int errorCode = TIME_INTEGRATION_SUCCESS;
   int Nelems;
   int Nelems2d;

   /*Set the full memory block number of elements for transfers of 2-d and 3-d fields*/
   Nelems = (Nxp+2*Nh)*(Nyp+2*Nh)*(Nzp+2*Nh);
   Nelems2d = (Nxp+2*Nh)*(Nyp+2*Nh);

   /* Send any desired GPU-computed HYDRO_CORE arrays from Device up to Host*/
   gpuErrchk( cudaMemcpy(hydroPres, hydroPres_d, Nelems*sizeof(float), cudaMemcpyDeviceToHost) );
   gpuErrchk( cudaMemcpy(hydroFlds, hydroFlds_d, Nelems*Nhydro*sizeof(float), cudaMemcpyDeviceToHost) );
   if((hydroForcingWrite==1)||(hydroForcingLog==1)){
     gpuErrchk( cudaMemcpy(hydroFldsFrhs, hydroFldsFrhs_d, Nelems*Nhydro*sizeof(float), cudaMemcpyDeviceToHost) );
   } //endif we need to send up the Frhs
   if (TKESelector > 0){ 
     gpuErrchk( cudaMemcpy(sgstkeScalars, sgstkeScalars_d, Nelems*TKESelector*sizeof(float), cudaMemcpyDeviceToHost) );
     if ((hydroForcingWrite==1)||(hydroForcingLog==1)){
       gpuErrchk( cudaMemcpy(sgstkeScalarsFrhs, sgstkeScalarsFrhs_d, Nelems*TKESelector*sizeof(float), cudaMemcpyDeviceToHost) );
     }
   }
   if (moistureSelector > 0){ 
     gpuErrchk( cudaMemcpy(moistScalars, moistScalars_d, Nelems*moistureNvars*sizeof(float), cudaMemcpyDeviceToHost) );
     if ((hydroForcingWrite==1)||(hydroForcingLog==1)){
       gpuErrchk( cudaMemcpy(moistScalarsFrhs, moistScalarsFrhs_d, Nelems*moistureNvars*sizeof(float), cudaMemcpyDeviceToHost) );
     }
   }
   if(surflayerSelector > 0){
     gpuErrchk( cudaMemcpy(fricVel, fricVel_d, Nelems2d*sizeof(float), cudaMemcpyDeviceToHost) );
     gpuErrchk( cudaMemcpy(htFlux, htFlux_d, Nelems2d*sizeof(float), cudaMemcpyDeviceToHost) );
     gpuErrchk( cudaMemcpy(tskin, tskin_d, Nelems2d*sizeof(float), cudaMemcpyDeviceToHost) );
     gpuErrchk( cudaMemcpy(invOblen, invOblen_d, Nelems2d*sizeof(float), cudaMemcpyDeviceToHost) );
     gpuErrchk( cudaMemcpy(z0m, z0m_d, Nelems2d*sizeof(float), cudaMemcpyDeviceToHost) );
     gpuErrchk( cudaMemcpy(z0t, z0t_d, Nelems2d*sizeof(float), cudaMemcpyDeviceToHost) );
     if (moistureSelector > 0){
       gpuErrchk( cudaMemcpy(qFlux, qFlux_d, Nelems2d*sizeof(float), cudaMemcpyDeviceToHost) );
       gpuErrchk( cudaMemcpy(qskin, qskin_d, Nelems2d*sizeof(float), cudaMemcpyDeviceToHost) );
     }
   }//endif surflayerSelector > 0
   if(hydroSubGridWrite==1){
     if(turbulenceSelector > 0){
       // The 6 Tau_i-j and 3 Tau_TH,j fields
       gpuErrchk( cudaMemcpy(hydroTauFlds, hydroTauFlds_d, Nelems*9*sizeof(float), cudaMemcpyDeviceToHost) );
     }//endif 
     if(moistureSGSturb==1){
       // The moistureNvars*3 tau moisture fields (3 spatial components per moist species)
       gpuErrchk( cudaMemcpy(moistTauFlds, moistTauFlds_d, Nelems*moistureNvars*3*sizeof(float), cudaMemcpyDeviceToHost) );
     }
   } //endif hydroSubGridWrite==1 
   gpuErrchk( cudaPeekAtLastError() ); /*Check for errors in the cudaMemCpy calls*/
//#ifdef DEBUG
#if 1
   MPI_Barrier(MPI_COMM_WORLD);
   printf("Rank %d/%d: Batch complete results sent via cudaMemcpyDeviceToHost.\n",mpi_rank_world, mpi_size_world);
   fflush(stdout);
   MPI_Barrier(MPI_COMM_WORLD);
#endif

   return(errorCode);
}//end cuda_timeIntHydrosynchFromDevice()
