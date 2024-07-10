#ifndef _AUXSCALARS_CUDADEV_CU_H
#define _AUXSCALARS_CUDADEV_CU_H

/*auxscalars_ return codes */
#define CUDA_AUXSCALARS_SUCCESS    0

/*##############------------------- AUXILIARY SCALARS submodule variable declarations ---------------------#################*/
/*Auxiliary Scalar Fields*/
extern __constant__ int NhydroAuxScalars_d;       /*Number of prognostic auxiliary scalar variable fields */
extern __constant__ int AuxScAdvSelector_d; /*adv. scheme for auxiliary scalar fields */
extern __constant__ float AuxScAdvSelector_b_hyb_d; /* hybrid advection scheme parameter */
extern float *hydroAuxScalars_d;     /*Base Adress of memory containing prognostic auxiliary scalar variable fields */
extern float *hydroAuxScalarsFrhs_d; /*Base Adress of memory Auxiliary field Frhs */
extern float *AuxScalarsTauFlds_d;   /*Base Adress of memory Auxiliary field SGS taus */
/*Auxiliary Scalar Sources*/
extern __constant__ int srcAuxScTemporalType_d[MAX_AUXSC_SRC];/*Temporal characterization of source (0 = instantaneous, 1 = continuous) */
extern __constant__ float srcAuxScStartSeconds_d[MAX_AUXSC_SRC];     /*Source start time in seconds */
extern __constant__ float srcAuxScDurationSeconds_d[MAX_AUXSC_SRC];  /*Source duration in seconds */
extern __constant__ int srcAuxScGeometryType_d[MAX_AUXSC_SRC];   /*0 = point (single cell volume), 1 = line (line of surface cells) */
extern __constant__ float srcAuxScLocation_d[MAX_AUXSC_SRC*3];      /*Cartesian coordinate tuple 'center' of the source*/
extern __constant__ int srcAuxScMassSpecType_d[MAX_AUXSC_SRC]; /*Mass specification type 0 = strict mass in kg, 1 = mass source rate in kg/s,  */
extern __constant__ float srcAuxScMassSpecValue_d[MAX_AUXSC_SRC]; /*Mass specification value in kg or kg/s given by srcAuxScMassSpecType 0 or 1 */

/*#################------------ AUXILIARY SCALARS submodule function definitions ------------------#############*/
/*----->>>>> int cuda_auxScalarsDeviceSetup();       ---------------------------------------------------------
* Used to cudaMalloc and cudaMemcpy parameters and coordinate arrays, and for the AUXSCALARS_CUDA submodule.
*/
extern "C" int cuda_auxScalarsDeviceSetup();

/*----->>>>> extern "C" int cuda_auxScalarsDeviceCleanup();  -----------------------------------------------------------
Used to free all malloced memory by the AUXSCALARS submodule.
*/
extern "C" int cuda_auxScalarsDeviceCleanup();

/*----->>>>> __global__ void cudaDevice_hydroCoreUnitTestCompleteAuxScalars();  ----------------------------------------------
* This device kernel for computing rhs forcing on auxiliary scalar fields.
*/
__global__ void cudaDevice_hydroCoreUnitTestCompleteAuxScalars(float simTime, float* hydroFlds,
                                                     float* hydroAuxScalars, float* hydroAuxScalarsFrhs,
                                                     float* hydroFaceVels,
                                                     float* xPos_d, float* yPos_d, float* zPos_d, float* topoPos_d,
                                                     float* J33_d, float* D_Jac_d, float* invD_Jac_d);

/*----->>>>> __device__ void  cudaDevice_calcAuxScalarSource();  ----------------------------------------------
* This applies a source term for an auxiliary scalar field.
*/
__device__ void cudaDevice_calcAuxScalarSource(int iFld, float *auxScFrhs, float *rhoFld, 
                                               float* xPos_d, float* yPos_d, float* zPos_d,
                                               float* topoPos_d, float* J33_d, float* D_Jac_d);


#endif // _AUXSCALARS_CUDADEV_CU_H
