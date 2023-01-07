/* FastEddy®: SRC/TIME_INTEGRATION/CUDA/cuda_timeIntDevice_cu.h 
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
#ifndef _TIME_INTEGRATION_CUDADEV_CU_H
#define _TIME_INTEGRATION_CUDADEV_CU_H

/*timeInt_ return codes */
#define CUDA_TIME_INTEGRATION_SUCCESS               0

/*##############------------- TIME_INTEGRATION module variable declarations ---------------------#################*/
/* Parameters */
extern __constant__ int timeMethod_d;   // Selector for time integration method. (default: 1= 3rd-order Runge-Kutta )
extern __constant__ int Nt_d;           // Number of timesteps to perform
extern __constant__ int NtimeTotVars_d;  // Total Number of prognostic variables to be integrated over time
extern __constant__ int NtBatch_d;  // Number of timesteps in a batch to perform in a CUDA kernel launch
extern __constant__ float dt_d;     // timestep resolution in seconds
extern __constant__ int simTime_itRestart_d;           //Timestep at restart (0 at start) 


/* array fields */
extern float *timeFlds0_d;   /* Multistage time scheme variable fields 4-D array */
extern float *timeFrhs0_d;   /* Multistage time scheme variable fields Frhs 4-D array */
extern float *timeFrhsTmp_d; /* Multistage time scheme variable fields Frhs 4-D array */

/*---RKSCHEMES */
#include <cuda_RKschemes_cu.h>

/*##############-------------- TIME_INTEGRATION_CUDADEV module function declarations ------------------############*/

/*----->>>>> int cuda_timeIntDeviceSetup();      ---------------------------------------------------------------------
* Used to cudaMalloc and cudaMemcpy parameters and coordinate arrays for the TIME_INTEGRATION_CUDADEV module.
*/
extern "C" int cuda_timeIntDeviceSetup();

/*----->>>>> int cuda_timeIntDeviceCleanup();    -------------------------------------------------------------------
* Used to free all malloced memory by the TIME_INTEGRATION_CUDADEV module.
*/
extern "C" int cuda_timeIntDeviceCleanup();

/*----->>>>> extern "C" int cuda_timeIntDeviceCommence();  -----------------------------------------------------------
* This routine provides the externally callable cuda-kernel call to commence with timeIntegration
*/
extern "C" int cuda_timeIntDeviceCommence(int it);

/*----->>>>> extern "C" int cuda_timeIntHydroInitDevice();  -----------------------------------------------------------
* This function handles the one-time initializations of on-device (GPU) memory by executing the appropriate sequence 
* of cudaMemcpyHostToDevice data transfers.
*/
extern "C" int cuda_timeIntHydroInitDevice();

/*----->>>>> extern "C" int cuda_timeIntHydroSynchFromDevice();  -----------------------------------------------------------
* This function handles the synchronization to host of on-device (GPU) fields  by executing the appropriate sequence 
* of cudaMemcpyDeviceiToHost data transfers.
*/
extern "C" int cuda_timeIntHydroSynchFromDevice();

#endif // _TIME_INTEGRATION_CUDADEV_CU_H
