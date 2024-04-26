/* FastEddy®: SRC/HYDRO_CORE/CUDA/cuda_surfaceLayerDevice_cu.h 
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
#ifndef _SURFLAYER_CUDADEV_CU_H
#define _SURFLAYER_CUDADEV_CU_H

/*SURFLAYER_ return codes */
#define CUDA_SURFLAYER_SUCCESS               0

/*##############------------------- SURFLAYER Submodule variable declarations ---------------------#################*/

/*SURFLAYER Submodule parameters*/
/*---SURFACE LAYER*/
extern __constant__ int surflayerSelector_d;  /*Monin-Obukhov surface layer selector: 0= off, 1= on */
extern __constant__ float surflayer_z0_d;     /* roughness length (momentum) */
extern __constant__ float surflayer_z0t_d;    /* roughness length (temoerature) */
extern __constant__ float surflayer_wth_d;    /* kinematic sensible heat flux at the surface */
extern __constant__ float surflayer_tr_d;     /* surface cooling rate K h-1 */
extern __constant__ float surflayer_wq_d;     /* kinematic latent heat flux at the surface */
extern __constant__ float surflayer_qr_d;     /* surface water vapor rate (g/kg) h-1 */
extern __constant__ int surflayer_qskin_input_d;/* selector to use file input (restart) value for qskin under surflayerSelector == 2 */
extern __constant__ float temp_grnd_d;        /* initial surface temperature */
extern __constant__ float pres_grnd_d;        /* initial surface pressure */
extern __constant__ int surflayer_stab_d;    /* exchange coeffcient stability correction selector: 0= on, 1= off */
extern float* cdFld_d;                        /*Base address for momentum exchange coefficient*/ 
extern float* chFld_d;                        /*Base address for sensible heat exchange coefficient*/ 
extern float* cqFld_d;                        /*Base address for latent heat exchange coefficient (2d-array)*/
extern float* fricVel_d;                      /*Base address for friction velocity*/ 
extern float* htFlux_d;                       /*Base address for sensible heat flux*/ 
extern float* qFlux_d;                        /*Base address for latent heat flux*/
extern float* tskin_d;                        /*Base address for skin temperature*/
extern float* qskin_d;                        /*Base address for skin moisture*/
extern float* invOblen_d;                     /*Base address for Monin-Obukhov length*/
extern float* z0m_d;                          /*Base address for roughness length (momentum)*/
extern float* z0t_d;                          /*Base address for roughness length (temperature)*/
extern __constant__ int surflayer_idealsine_d;   /*selector for idealized sinusoidal surface heat flux or skin temperature forcing*/
extern __constant__ float surflayer_ideal_ts_d;  /*start time in seconds for the idealized sinusoidal surface forcing*/
extern __constant__ float surflayer_ideal_te_d;  /*end time in seconds for the idealized sinusoidal surface forcing*/
extern __constant__ float surflayer_ideal_amp_d; /*maximum amplitude of the idealized sinusoidal surface forcing*/
extern __constant__ float surflayer_ideal_qts_d;  /*start time in seconds for the idealized sinusoidal surface forcing of latent heat flux*/
extern __constant__ float surflayer_ideal_qte_d;  /*end time in seconds for the idealized sinusoidal surface forcing of latent heat flux*/
extern __constant__ float surflayer_ideal_qamp_d; /*maximum amplitude of the idealized sinusoidal surface forcing of latent heat flux*/
/*Offshore roughness parameters*/
extern __constant__ int surflayer_offshore_d;         /* offshore selector: 0=off, 1=on */
extern __constant__ int surflayer_offshore_opt_d;     /* offshore roughness parameterization: ==0 (Charnock), ==1 (Charnock with variable alpha), ==2 (Taylor & Yelland), ==3 (Donelan), ==4 (Drennan), ==5 (Porchetta) */
extern __constant__ int surflayer_offshore_dyn_d;     /* selector to use parameterized ocean parameters: 0=off, 1=on (default) */
extern __constant__ float surflayer_offshore_hs_d;    /* significant wave height */
extern __constant__ float surflayer_offshore_lp_d;    /* peak wavelength */
extern __constant__ float surflayer_offshore_cp_d;    /* wave phase speed */
extern __constant__ float surflayer_offshore_theta_d; /* wave/wind angle */
extern __constant__ int surflayer_offshore_visc_d;    /* viscous term on z0m: 0=off, 1=on (default) */

/*##############-------------- SURFLAYER_CUDADEV Submodule function declarations ------------------############*/

/*----->>>>> int cuda_surfaceLayerDeviceSetup();       -------------------------------------------------------------
 *  * Used to cudaMalloc and cudaMemcpy parameters and coordinate arrays, and for the SURFLAYER HC-Submodule.
 *  */
extern "C" int cuda_surfaceLayerDeviceSetup();
/*----->>>>> extern "C" int cuda_surfaceLayerDeviceCleanup();  -------------------------------------------------------
 * Used to free all malloced memory by the SURFLAYER HC-Submodule.
 * */
extern "C" int cuda_surfaceLayerDeviceCleanup();

/*----->>>>> __device__ void  cudaDevice_SurfaceLayer();  --------------------------------------------------
 * This is the cuda version of the hydro_coreSurfLayer routine from the SURFLAYER module
*/
__device__ void cudaDevice_SurfaceLayerLSMdry(float simTime, int simTime_it, int simTime_itRestart,
                                              float dt, int timeStage, int maxSteps, int ijk,
                                              float* u, float* v, float* rho, float* theta,
                                              float* cd_iter, float* ch_iter, float* cq_iter, float* fricVel, float* htFlux, float* tskin,
                                              float* z0m, float* z0t, float* J33_d);

__device__ void cudaDevice_SurfaceLayerLSMmoist(float simTime, int simTime_it, int simTime_itRestart,
                                                float dt, int timeStage, int maxSteps, int ijk,
                                                float* u, float* v, float* rho, float* theta, float* qv,
                                                float* cd_iter, float* ch_iter, float* cq_iter, float* fricVel,
                                                float* htFlux, float* tskin, float* qFlux, float* qskin,
                                                float* z0m, float* z0t, float* J33_d);

__device__ void cudaDevice_SurfaceLayerMOSTdry(int ijk, float* u, float* v, float* rho, float* theta,
                                               float* tau31, float* tau32, float* tauTH3,
                                               float* cd_iter, float* ch_iter, float* fricVel,
                                               float* htFlux, float* tskin, float* invOblen, float* z0m,
                                               float* z0t, float* sea_mask, float* J33_d);

__device__ void cudaDevice_SurfaceLayerMOSTmoist(int ijk, float* u, float* v, float* rho, float* theta, float* qv,
                                                 float* tau31, float* tau32, float* tauTH3, float* tauQ3,
                                                 float* cd_iter, float* ch_iter, float* cq_iter, float* fricVel,
                                                 float* htFlux, float* tskin, float* qFlux, float* qskin,
                                                 float* invOblen, float* z0m, float* z0t, float* sea_mask, float* J33_d);

__device__ void cudaDevice_offshoreRoughness(float* z0m, float* z0t, float* fricVel, float u_1, float v_1, float* sea_mask);

#endif // _SURFLAYER_CUDADEV_CU_H
