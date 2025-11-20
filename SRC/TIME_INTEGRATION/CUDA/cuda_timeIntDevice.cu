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
   size_t Nelems;
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
   Nelems = (size_t)((Nxp+2*Nh)*(Nyp+2*Nh)*(Nzp+2*Nh)); 
   /* Allocate the TIME_INTEGRATION arrays */
   /*TIME_INTEGRATION/CUDA internal device arrays*/
   NtimeTotVars = 5 + TKESelector*turbulenceSelector + moistureNvars*moistureSelector + NhydroAuxScalars; 
   fecuda_DeviceMalloc(NtimeTotVars*Nelems, &timeFlds0_d);
   
   gpuErrchk( cudaPeekAtLastError() ); /*Check for errors in the cudaMalloc calls*/

   //Ensure secondary time-integration dependent hydro_core parameters get initialized
   errorCode = cuda_hydroCoreDeviceSecondaryStageSetup(dt);
   //Inital Host-to-Device field copies 
   errorCode = cuda_hydroCoreInitFieldsDevice();  //Transfer initial/restart conditions to the device
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

   /*Update LAD BCs if appropriate*/
   if(hydroBCs==1){ //Using LAD BCs
     if((it%((int)roundf(dtBdyPlaneBCs/dt))==0)&&(it > simTime_itRestart)){
       printf("cuda_timeIntDeviceCommence:Updating device-side BdyPlanes at it=%d...\n",it);
       fflush(stdout);
       errorCode = cuda_hydroCoreDeviceBdyPlanesUpdate();
       if((cellpertSelector==1)&&(cellpert_tvcp==1)){ // update CP parameters with dynamic LBCs
         errorCode = cuda_hydroCoreTVCP();
       } // end if((cellpertSelector==1)&&(cellpert_tvcp==1))
     }
   }//end if hydroBCs==1

   for(itBatch=0; itBatch < NtBatch; itBatch++){     //Batch timestepping loop
     if((lsfSelector == 1) && (lsf_horMnSubTerms == 1) && (simTime_it > simTime_itRestart) && (simTime_it%(int)roundf(lsf_freq/dt)==0)){
       errorCode = cuda_lsfSlabMeans();
     }
     gpuErrchk( cudaDeviceSynchronize() );
     /*Execute the timeMethod kernel of choice on the GPU*/
     if(timeMethod == 0){    /*******  Issue the  3rd-order Runge-Kutta WS2002 **************/
       if(cellpertSelector==1 && simTime_it%cellpert_nts==0){ /***** Issue cell perturbation method here *****/
         errorCode = cuda_hydroCoreDeviceBuildCPmethod(simTime_it); // call to buildCPmethod
       }
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
						NhydroAuxScalars, hydroAuxScalars_d, hydroAuxScalarsFrhs_d,
                                                timeFlds0_d, RKstage);
          gpuErrchk( cudaGetLastError() );
#ifdef TIMERS_LEVEL1
          stopSynchReportDestroyEvent(&startE, &stopE, &elapsedTime);
          printf("cuda_timeIntCommenceRK3_WS2002()  Kernel execution time (ms): %12.8f\n", elapsedTime);
#endif
       } //end for RKstage 
     } //end if(timeMethod == 0){...
     simTime_it = simTime_it + 1;   //Increment the master simulation time step
     simTime = simTime_it * dt;   /*Increment the master simulation time*/
   }//end for itBatch...

   //Retrieve desired HYDRO_CORE fields from device
   errorCode = cuda_hydroCoreSynchFieldsFromDevice();
   
   return(errorCode);
}//end cuda_timeIntDeviceCommence()

