
#ifndef _URBAN_CUDADEV_CU_H
#define _URBAN_CUDADEV_CU_H

/*urban_ return codes */
#define CUDA_URBAN_SUCCESS               0

/*##############------------------- URBAN submodule variable declarations ---------------------#################*/
/* Parameters */
extern __constant__ int urbanSelector_d;          /* urban selector: 0=off, 1=on */
extern __constant__ float cd_build_d;             /* c_d coefficient (m-1) used by the drag-based building formulation: -c_d|u_i|u_i */
extern __constant__ float ct_build_d;             /* c_t coefficient (s-1) used by the drag-based building formulation: -c_t(rho*theta-rho_b*theta_b) & -c_t(rho-rho_b) */
extern __constant__ float delta_aware_bdg_d;      /* scale-aware correction for building forcing and limiters */
/* array fields */
extern float* building_mask_d;                    /* Base Address of memory containing building mask field: 0 (atmosphere) or 1 (building) */
extern __constant__ int urban_heatRedis_d;        /* selector to activate surface heat redistribution */
extern float *urban_heat_redis_d;                 /* Base Address of memory containing 2d map of heat redistribution coefficient in urban areas */

/*##############-------------- URBAN_CUDADEV submodule function declarations ------------------############*/

/*----->>>>> int cuda_urbanDeviceSetup();      -----------------------------------------------------------------
* Used to cudaMalloc and cudaMemcpy parameters and coordinate arrays for the URBAN_CUDADEV submodule.
*/
extern "C" int cuda_urbanDeviceSetup();

/*----->>>>> int cuda_urbanDeviceCleanup();    ---------------------------------------------------------------
* Used to free all malloced memory by the URBAN_CUDADEV submodule.
*/
extern "C" int cuda_urbanDeviceCleanup();

__global__ void cudaDevice_URBANinter(float* hydroTauFlds, float* moistTauFlds, float* fricVel,float* htFlux,float* qFlux,float* invOblen,float* bdg_mask);
__global__ void cudaDevice_URBANinterRedis(float* hydroTauFlds, float* moistTauFlds, float* fricVel,float* htFlux,float* qFlux,float* invOblen,float* bdg_mask, float* urban_redis);
__global__ void cudaDevice_URBANfinal(float* hydroFlds_d, float* hydroFldsFrhs_d, float* hydroBaseStateFlds_d, float* building_mask_d);
__global__ void cudaDevice_URBANfinalAuxSc(float* hydroAuxScalars_d, float* hydroAuxScalarsFrhs_d, float* building_mask_d);
__global__ void cudaDevice_URBANfinalMoist(float* hydroFldsFrhsMoist_d, float* building_mask_d);

/*----->>>>> __device__ void  cudaDevice_UrbanDragMethod();  --------------------------------------------------
 *  */ // This cuda kerne lsets up the cells and their id in the urban drag-based approach
__device__ void cudaDevice_UrbanDragMethod(float* rho, float* u, float* v, float* w, float* th, float* th_base, float* rho_base, float* Frhs_u, float* Frhs_v, float* Frhs_w, float* Frhs_th, float* Frhs_rho, float* bdg_mask);
__device__ void cudaDevice_UrbanDragMethodMoist(float* Frhs_qv, float* bdg_mask);
__device__ void cudaDevice_UrbanDragMethodAuxScalar(float* AuxScalar, float* Frhs_AuxScalar, float* bdg_mask);

#endif // _URBAN_CUDADEV_CU_H
