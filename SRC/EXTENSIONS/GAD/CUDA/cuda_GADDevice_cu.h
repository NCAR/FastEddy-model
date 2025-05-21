#ifndef _GAD_CUDADEV_CU_H
#define _GAD_CUDADEV_CU_H

/*GAD return codes */
#define CUDA_GAD_SUCCESS    0

/*##############------------------- GAD submodule variable declarations ---------------------#################*/
extern __constant__ int GADSelector_d;     /* Generalized Actuator Disk Selector: 0=off, 1=on */
extern __constant__ int GADoutputForces_d;    /* Flag to include GAD forces in the output: 0=off, 1=on */
extern __constant__ int GADofflineForces_d;   /* Flag to compute GAD forces in an offline mode: 0=off, 1=on */
extern __constant__ int GADaxialInduction_d;   /* Flag to compute axial induction factor: 0==off (uses prescribed GADaxialIndVal), 1==on */
extern __constant__ float GADaxialIndVal_d;    /* Prescribed constant axial induction factor when GADaxialInduction==1 */
extern __constant__ int GADrefSwitch_d;   /* Switch to use reference windspeed: 0=off, 1=on */
extern __constant__ float GADrefU_d;    /* Prescribed constant reference hub-height windspeed*/
extern __constant__ int GADForcingSwitch_d;    /* Switch to use the GADrefU-based or local windspeed in computing GAD forces: 0=local, 1=ref */
extern __constant__ int GADNumTurbines_d;     /* Number of GAD Turbines */
extern __constant__ int GADNumTurbineTypes_d;  /* Number of GAD Turbine Types */
extern __constant__ int turbinePolyOrderMax_d; /* Maximum Polynomial order across all turbine types */
extern __constant__ int turbinePolyClCdrNormSegments_d; /* Number of segments in the normalized radius for the lift and drag coefficient polynomial */
extern __constant__ int alphaBounds_d;         /* Number of elements in the min/max angle of attack array for the lift/drag curves */

extern __constant__ int numgridCells_away_d; /*Halo-region of cells considered in rotor disk distance-wise smoothing function*/

extern int* GAD_turbineType_d;    /* Integer class-label for turbine type*/
extern float* GAD_Xcoords_d; /* turbine x-location [m] from SW domain corner */
extern float* GAD_Ycoords_d; /* turbine y-location [m] from SW domain corner */
extern float* GAD_rotorTheta_d; /* turbine yaw angle [deg. North] */
extern float* GAD_hubHeights_d; /* turbine hub height [m AGL] */
extern float* GAD_rotorD_d; /* turbine rotor diameter [m] */
extern float* GAD_nacelleD_d; /* nacelle diameter [m] */
extern float* turbinePolyTwist_d; /* turbine-type-specific twist polynomial coefficients*/
extern float* turbinePolyChord_d; /* turbine-type-specific chord polynomial coefficients*/
extern float* turbinePolyPitch_d; /* turbine-type-specific pitch polynomial coefficients*/
extern float* turbinePolyOmega_d; /* turbine-type-specific omega polynomial coefficients*/
extern float* rnorm_vect_d;       /* turbine-type-specific normalized radious segment limits*/
extern float* alpha_minmax_vect_d;/* turbine-type-specific maximum and minimum angle of attack for the lift/drag curves*/
extern float* turbinePolyCl_d;    /* turbine-type-specific lift coefficient polynomial coefficients*/
extern float* turbinePolyCd_d;    /* turbine-type-specific drag coefficient polynomial coefficients*/

extern float* GAD_turbineVolMask_d; /* turbine Volume mask (0 if turbine free cell in domain, else turbine ID of cell in turbine yaw-swept volume*/
extern float* GAD_forceX_d;         /* turbine forces in the x-direction */
extern float* GAD_forceY_d;         /* turbine forces in the y-direction */
extern float* GAD_forceZ_d;         /* turbine forces in the z-direction */

/*##############-------------- GAD_CUDADEV submodule function declarations ------------------############*/

/*----->>>>> int cuda_GADDeviceSetup();       ---------------------------------------------------------
* Used to cudaMalloc and cudaMemcpy parameters and coordinate arrays, and for the GAD_CUDA submodule.
*/
extern "C" int cuda_GADDeviceSetup();

/*----->>>>> extern "C" int cuda_GADDeviceCleanup();  -----------------------------------------------------------
* Used to free all malloced memory by the GAD submodule.
*/
extern "C" int cuda_GADDeviceCleanup();

/*----->>>>> __global__ void  cudaDevice_GADComputeFrhs();  --------------------------------------------------
* This function is the global entry kernel for computing GAD forcing from turbines
*/
__global__ void cudaDevice_GADComputeFrhs(float* xPos_d, float* yPos_d, float* zPos_d, float* topoPos_d, 
                                            float* hydroFlds_d, float* hydroFldsFrhs_d,
                                            int* GAD_turbineType_d, float* GAD_turbineVolMask_d,
                                            float* GAD_Xcoords_d, float* GAD_Ycoords_d, float* GAD_rotorTheta_d,
                                            float* GAD_hubHeights_d, float* GAD_rotorD_d, float* GAD_nacelleD_d,
                                            float* turbinePolyTwist_d, float* turbinePolyChord_d,
                                            float* turbinePolyPitch_d, float* turbinePolyOmega_d,
                                            float* rnorm_vect_d, float* alpha_minmax_vect_d,
                                            float* turbinePolyCl_d, float* turbinePolyCd_d,
                                            float* GAD_forceX_d, float* GAD_forceY_d, float* GAD_forceZ_d);

/*----->>>>> __device__ void  cudaDevice_cellInRotorOrig();  --------------------------------------------------
* This functions calculates a radial vector and setes a flag to detrmine if a cell is in a rotor disk area
*/
__device__ void cudaDevice_cellInRotorOrig(float* cell_inRotor, float* cell_rVector,
                                       int iturb, float turbX, float turbY, float turbTheta, float turbHubHgt, float turbD,
                                       float xLoc, float yLoc, float zLoc, float dx, float dy);

/*----->>>>> __device__ void  cudaDevice_cellInRotor();  --------------------------------------------------
 * This functions calculates a radial vector and setes a flag to detrmine if a cell is in a rotor disk area
 */
__device__ void cudaDevice_cellInRotor(float* cell_inRotor, float* cell_rVector,
                                       int iturb, float turbX, float turbY,
                                       float turbTheta, float turbHubHgt, float tiltAngle,
                                       float rotorD, float nacelleD,
                                       float xLoc, float yLoc, float zLoc, float dx, float dy);

/*----->>>>> __device__ void cudaDevice_GADtwistChord();  --------------------------------------------------
*/
__device__ void cudaDevice_GADtwistChord(float* turbinePolyTwist_d, float* turbinePolyChord_d,
                                         float rotorD, float turbineRadius, float* twist_angle, float* chord_length);

/*----->>>>> __device__ void cudaDevice_GADbetaOmega();  --------------------------------------------------
*/
__device__ void cudaDevice_GADbetaOmega(float u, float v, float rho, float* turbinePolyPitch_d, float* turbinePolyOmega_d,
                                        float rotorD, float turbineRadius, float twist_angle, float* beta_angle, float* omega_rot);

/*----->>>>> __device__ void cudaDevice_GADforcesCompute();  --------------------------------------------------
*/
__device__ void cudaDevice_GADforcesCompute(float u, float v, float rho, float rotorD, float nacelleD,
                                            float turbineRadius, float beta_angle, float omega_rot, float chord_length,
                                            float *rnorm_vect, float *alpha_minmax_vect, float *turbinePolyCl, float *turbinePolyCd,
                                            float *GADforce_n, float *GADforce_t);

/*----->>>>> __device__ void cudaDevice_GADforcesApply();  --------------------------------------------------
*/
__device__ void cudaDevice_GADforcesApply(float rho, float turb_Xcoord, float turb_Ycoord, float hubHeight, float rotorTheta, float rotorD, 
                                          float xLoc, float yLoc, float zLoc,
                                          float GADforce_n, float GADforce_t, float* GADforce_x, float* GADforce_y, float* GADforce_z,
                                          float* GAD_fX, float* GAD_fY, float* GAD_fZ, float turbineRadius, float nacelleD);

/*----->>>>> __device__ void compute_ClCd_incoeff();  --------------------------------------------------
*/
__device__ void compute_ClCd_incoeff(float* rnorm_vect, float* turbinePolyCl, float* turbinePolyCd, float alpha, float r_norm, float* C_l, float* C_d);

/*----->>>>> __device__ void distribute_GADforces();  --------------------------------------------------
*/
__device__ void distribute_GADforces(float xLoc, float yLoc, float x_turb, float y_turb, float theta_turb, float rotorD, float* F_dist_fact);

#endif // _GAD_CUDADEV_CU_H
