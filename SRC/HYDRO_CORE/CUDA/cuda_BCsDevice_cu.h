/* FastEddy®: SRC/HYDRO_CORE/CUDA/cuda_BCsDevice_cu.h 
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
#ifndef _BCS_CUDADEV_CU_H
#define _BCS_CUDADEV_CU_H

/*BCs_ return codes */
#define CUDA_BCS_SUCCESS    0

/*##############------------------- BCS submodule variable declarations ---------------------#################*/
/*---BOUNDARY CONDITIONS parameters*/
extern __constant__ int hydroBCs_d;        // hydro_core BC set selector
extern __constant__ float dtBdyPlaneBCs_d; //delta in time (seconds) between update BdyPlanes sets
extern __constant__ int BdyUpdateSteps_d;  // number of timesteps between update of BdyPlanes sets
extern __constant__ int nBndyVars_d;       // number of Bndy plane variables
extern __constant__ int nSurfBndyVars_d;   // number of surface Bndy plane variables

extern __constant__ float U_g_d;            /*Zonal (West-East) component of the geostrophic wind*/
extern __constant__ float V_g_d;            /*Meridional (South-North) component of the geostrophic wind*/
extern __constant__ float z_Ug_d;
extern __constant__ float z_Vg_d;
extern __constant__ float Ug_grad_d;
extern __constant__ float Vg_grad_d;

/*---BOUNDARY CONDITIONS arrays*/
extern float *XZBdyPlanes_d;          /*Base Adress of memory block for lateral-BC XZ-planes (per rank domain)*/
extern float *YZBdyPlanes_d;          /*Base Adress of memory block for lateral-BC YZ-planes (per rank domain)*/
extern float *XYBdyPlanes_d;          /*Base Adress of memory block for surface/ceiling--BC XY-planes (per rank domain)*/
extern float *XZBdyPlanesNext_d;      /*Base Adress of memory block for lateral-BC XZ-planes (per rank domain)*/
extern float *YZBdyPlanesNext_d;      /*Base Adress of memory block for lateral-BC YZ-planes (per rank domain)*/
extern float *XYBdyPlanesNext_d;      /*Base Adress of memory block for surface/ceiling--BC XY-planes (per rank domain)*/
extern float *XZBdyPlanesBuffer_d;    /*Base Adress of memory block for lateral-BC XZ-planes (per rank domain)*/
extern float *YZBdyPlanesBuffer_d;    /*Base Adress of memory block for lateral-BC YZ-planes (per rank domain)*/
extern float *XYBdyPlanesBuffer_d;    /*Base Adress of memory block for surface/ceiling--BC XY-planes (per rank domain)*/
extern float *SURFBdyPlanes_d;        /*Base Adress of memory block for surfVar-BC XY-planes (per rank domain)*/
extern float *SURFBdyPlanesNext_d;    /*Base Adress of memory block for surfVar-BC XY-planes (per rank domain)*/
extern float *SURFBdyPlanesBuffer_d;  /*Base Adress of memory block for surfVar-BC XY-planes (per rank domain)*/

/*##############-------------- BCS_CUDADEV submodule function declarations ------------------############*/
/*----->>>>> int cuda_BCsDeviceSetup(); ---------------------------------------------------------
* Used to cudaMalloc and cudaMemcpy parameters and coordinate arrays, and for the BCS_CUDA submodule.
*/
extern "C" int cuda_BCsDeviceSetup();

/*----->>>>> extern "C" int cuda_BCsDeviceCleanup(); -----------------------------------------------------------
* Used to free all malloced memory by the BCS submodule.
*/
extern "C" int cuda_BCsDeviceCleanup();

/*----->>>>> int cuda_hydroCoreDeviceSecondaryStageSetup(float dt); -----------------------------------------------------------------
* Secondary initializations at the device level for BCs  
*/
extern "C" int cuda_hydroCoreDeviceSecondaryStageSetup(float dt);

/*----->>>>> int cuda_hydroCoreDeviceBdyPlanesUpdate();      -----------------------------------------------------------------
* Utility to cycle device-sided pointers and push (copy) newest BndyPlanes from Host to Device
*/
extern "C" int cuda_hydroCoreDeviceBdyPlanesUpdate();

__device__ void cudaDevice_HorizontalPeriodicXdirBCs(int fldIndx, float* scalarField);

__device__ void cudaDevice_HorizontalPeriodicYdirBCs(int fldIndx, float* scalarField);

__device__ void cudaDevice_VerticalAblBCs(int fldIndx, float* scalarField, float* scalarBaseStateField);

__device__ void cudaDevice_VerticalAblBCsMomentum(int fldIndxMom, float* scalarField, float* scalarBaseStateField, float* zPos_d);

__device__ void cudaDevice_MomentumBS(int fldIndxMom, float zPos_ijk, float* rho_ijk, float* MomBSval);

__device__ void cudaDevice_VerticalAblZeroGradBCs(float* scalarField);

__device__ void cudaDevice_surfaceVarBdyBCs(int fldIndx, float timeWeight, float* scalarField, float* SURFBdyPlanes_d, float* SURFBdyPlanesNext_d);

__device__ void cudaDevice_ceilingBdyBCs(int fldIndx, float timeWeight, float* scalarField, float* XYBdyPlanes_d, float* XYBdyPlanesNext_d);

__device__ void cudaDevice_lateralTKEBdyBCs(int fldIndx, float* scalarField, float* scalarFieldBS, int bdySelector);

__device__ void cudaDevice_westBdyBCs(int fldIndx, float timeWeight, float* scalarField, float* YZBdyPlanes_d, float* YZBdyPlanesNext_d);

__device__ void cudaDevice_eastBdyBCs(int fldIndx, float timeWeight, float* scalarField, float* YZBdyPlanes_d, float* YZBdyPlanesNext_d);

__device__ void cudaDevice_southBdyBCs(int fldIndx, float timeWeight, float* scalarField, float* XZBdyPlanes_d, float* XZBdyPlanesNext_d);

__device__ void cudaDevice_northBdyBCs(int fldIndx, float timeWeight, float* scalarField, float* XZBdyPlanes_d, float* XZBdyPlanesNext_d);

#endif // _BCS_CUDADEV_CU_H
