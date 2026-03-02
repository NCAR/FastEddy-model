/* FastEddy®: SRC/EXTENSIONS/GAD/CUDA/cuda_GADDevice_cu.h
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
#ifndef _GAD_CUDADEV_CU_H
#define _GAD_CUDADEV_CU_H

/*GAD return codes */
#define CUDA_GAD_SUCCESS    0

/*##############------------------- GAD submodule variable declarations ---------------------#################*/
extern __constant__ int GADSelector_d;     /* Generalized Actuator Disk Selector: 0=off, 1=on */
extern __constant__ int GADoutputForces_d;    /* Flag to include GAD forces in the output: 0=off, 1=on */
extern __constant__ int GADofflineForces_d;   /* Flag to compute GAD forces in an offline mode: 0=off, 1=on */
extern __constant__ int GADaxialInduction_d;   /* Flag to compute axial induction factor: 0==off (uses prescribed GADaxialIndVal), 1==on */
extern __constant__ float GADaxialIndVal_d;    /* Prescribed constant axial induction factor when GADaxialInduction==0 */
extern __constant__ int GADrefSwitch_d;   /* Switch to use reference windspeed: 0=off, 1=on */
extern __constant__ float GADrefU_d;    /* Prescribed constant reference hub-height windspeed*/
extern __constant__ int GADForcingSwitch_d;    /* Switch to use the GADrefU-based or local windspeed in computing GAD forces: 0=local, 1=ref */
extern __constant__ int GADNumTurbines_d;     /* Number of GAD Turbines */
extern __constant__ int GADNumTurbineTypes_d;  /* Number of GAD Turbine Types */
extern __constant__ int turbinePolyOrderMax_d; /* Maximum Polynomial order across all turbine types */
extern __constant__ int turbinePolyClCdrNormSegments_d; /* Number of segments in the normalized radius for the lift and drag coefficient polynomial */
extern __constant__ int alphaBounds_d;         /* Number of elements in the min/max angle of attack array for the lift/drag curves */

extern __constant__ int GADsamplingAvgLength_d;   /*length of sampling average windows (averaging over fastest timescales)*/
extern __constant__ float GADsamplingAvgWeight_d;   /*weight of instances in taking sampling average*/
extern __constant__ int GADrefSeriesLength_d;   /*number of sample average windows to incorporate into full Reference average*/
extern __constant__ float GADrefSeriesWeight_d; /*precalculated averaging weight for Reference average*/

extern __constant__ int numgridCells_away_d; /*Halo-region of cells considered in rotor disk distance-wise smoothing function*/

extern int* GAD_turbineType_d;     /* Integer class-label for turbine type*/
extern int* GAD_turbineRank_d;     /* Integer mpi-rank of nacelle center cell for each turbine reference velMag and velDir grid cell*/
extern int* GAD_turbineRefi_d;     /* Integer i-index of nacelle center cell for each turbine reference velMag and velDir grid cell*/
extern int* GAD_turbineRefj_d;     /* Integer j-index of nacelle center cell for each turbine reference velMag and velDir grid cell*/
extern int* GAD_turbineRefk_d;     /* Integer k-index of nacelle center cell for each turbine reference velMag and velDir grid cell*/
extern int* GAD_turbineYawing_d;   /* Integer indicating in a turbine is currently yawing ==1*/
extern float* GAD_Xcoords_d;       /* turbine x-location [m] from SW domain corner */
extern float* GAD_Ycoords_d;       /* turbine y-location [m] from SW domain corner */
extern float* GAD_turbineRefMag_d; /* Reference "ambient" velocity magnitude for yaw control and beta/omega [m/s]*/
extern float* GAD_turbineRefDir_d; /* *Reference "ambient" velocity direction (horizontal, met. standard orientation) for yaw control and beta/omega [degrees]*/
extern float* GAD_turbineUseries_d;/* uSeries of sample averages spanning the rolling-average reference period */
extern float* GAD_turbineVseries_d;/* vSeries of sample averages spanning the rolling-average reference period */
extern float* u_sampAvg_d;         /* u sample averages for each turbine */
extern float* v_sampAvg_d;         /* v sample averages for each turbine */
extern float* GAD_yawError_d;      /* yaw error between the incoming wind and the turbine orientation */
extern float* GAD_anFactor_d;     /* turbine axial induction factor at hub heigth*/
extern float* GAD_rotorTheta_d;    /* turbine yaw angle [deg. North] */
extern float* GAD_hubHeights_d;    /* turbine hub height [m AGL] */
extern float* GAD_rotorD_d;        /* turbine rotor diameter [m] */
extern float* GAD_nacelleD_d;      /* nacelle diameter [m] */
extern float* turbinePolyTwist_d;  /* turbine-type-specific twist polynomial coefficients*/
extern float* turbinePolyChord_d;  /* turbine-type-specific chord polynomial coefficients*/
extern float* turbinePolyPitch_d;  /* turbine-type-specific pitch polynomial coefficients*/
extern float* turbinePolyOmega_d;  /* turbine-type-specific omega polynomial coefficients*/
extern float* rnorm_vect_d;        /* turbine-type-specific normalized radious segment limits*/
extern float* alpha_minmax_vect_d; /* turbine-type-specific maximum and minimum angle of attack for the lift/drag curves*/
extern float* turbinePolyCl_d;     /* turbine-type-specific lift coefficient polynomial coefficients*/
extern float* turbinePolyCd_d;     /* turbine-type-specific drag coefficient polynomial coefficients*/

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

/*----->>>>> __global__ void  cudaDevice_GADinter();  --------------------------------------------------
* This function is the global entry kernel for computing reference values for GAD yawing and other turbine characteristics
*/
__global__ void cudaDevice_GADinter(float* xPos_d, float* yPos_d, float* zPos_d, float* topoPos_d,
		                    int simTime_it, int timeStage, int numRKstages, float dt,
		                    float* hydroFlds_d, int* GAD_turbineType_d, float* GAD_turbineVolMask_d,
                                    float* GAD_Xcoords_d, float* GAD_Ycoords_d, float* GAD_rotorTheta_d,
                                    float* GAD_hubHeights_d, float* GAD_rotorD_d, float* GAD_nacelleD_d,
                                    float* turbinePolyTwist_d, float* turbinePolyChord_d,
                                    float* turbinePolyPitch_d, float* turbinePolyOmega_d,
                                    float* rnorm_vect_d, float* alpha_minmax_vect_d,
                                    float* turbinePolyCl_d, float* turbinePolyCd_d,
                                    int* GAD_turbineRank_d, int* GAD_turbineRefi_d, int* GAD_turbineRefj_d, int* GAD_turbineRefk_d,
                                    float* u_sampAvg_d, float* v_sampAvg_d,
                                    float* GAD_turbineUseries_d, float* GAD_turbineVseries_d,
                                    float* GAD_turbineRefMag_d, float* GAD_turbineRefDir_d,
				    int* GAD_turbineYawing_d, float* GAD_yawError_d, float* GAD_anFactor_d);

/*----->>>>> __global__ void  cudaDevice_GADfinal();  --------------------------------------------------
* This function is the global entry kernel for computing GAD forcing from turbines
*/
__global__ void cudaDevice_GADfinal(float* xPos_d, float* yPos_d, float* zPos_d, float* topoPos_d,
                                    float* hydroFlds_d, float* hydroFldsFrhs_d, int simTime_it, float dt,
                                    int* GAD_turbineType_d, float* GAD_turbineVolMask_d,
                                    float* GAD_Xcoords_d, float* GAD_Ycoords_d, float* GAD_rotorTheta_d,
                                    float* GAD_hubHeights_d, float* GAD_rotorD_d, float* GAD_nacelleD_d,
                                    float* turbinePolyTwist_d, float* turbinePolyChord_d,
                                    float* turbinePolyPitch_d, float* turbinePolyOmega_d,
                                    float* rnorm_vect_d, float* alpha_minmax_vect_d,
                                    float* turbinePolyCl_d, float* turbinePolyCd_d,
				    float* GAD_turbineRefMag_d, float* GAD_anFactor_d,
                                    float* GAD_forceX_d, float* GAD_forceY_d, float* GAD_forceZ_d);

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
__device__ void cudaDevice_GADbetaOmega(float turbineRefMag, float anFactor, float* turbinePolyPitch_d, float* turbinePolyOmega_d,
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

/*----->>>>> __device__ void update_sampleRefVel();  --------------------------------------------------
*/
__device__ void update_sampleRefVel(float u, float v, float rho, float* u_sampAvg, float* v_sampAvg);

/*----->>>>> __device__ void update_turbineRefMagDir();  --------------------------------------------------
*/
__device__ void update_turbineRefMagDir(int sampleIndex, float u_sampAvg, float v_sampAvg,
                                        float* uSeries, float* vSeries, float* turbineRefMag, float* turbineRefDir);

/*----->>>>> __device__ void update_yawError();  --------------------------------------------------
*/
__device__ void update_yawError(float* turbineRefDir, float* rotorTheta, float* yawError, int* turbineYawing, float dt);

/*----->>>>> __device__ void update_rotorTheta();  --------------------------------------------------
*/
__device__ void update_rotorTheta(float* turbineRefDir, float* rotorTheta, float* yawError, int* turbineYawing, float dt);
/*----->>>>> __device__ void Angle_TurbWind();  --------------------------------------------------
*/
__device__ void Angle_TurbWind(float turbineRefDir, float rotorTheta, float* diff_angle);
/*----->>>>> __device__ void compute_normalInduction();  --------------------------------------------------
*/
__device__ void compute_normalInduction(float turbineRefMag, float rotorD, float nacelleD,
                                        float turbineRadius, float beta_angle, float omega_rot, float chord_length,
                                        float *rnorm_vect, float *alpha_minmax_vect, float *turbinePolyCl, float *turbinePolyCd,
					float *turbineRefAn);
#endif // _GAD_CUDADEV_CU_H
