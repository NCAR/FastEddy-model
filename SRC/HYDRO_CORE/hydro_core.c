/* FastEddy®: SRC/HYDRO_CORE/hydro_core.c 
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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <float.h>
#include <fempi.h>
#include <parameters.h>
#include <mem_utils.h>
#include <io.h>
#include <grid.h>
#include <hydro_core.h>


/*##################------------------- HYDRO_CORE module variable definitions ---------------------#################*/
int Nhydro = 5;              /*Number of prognostic variable fields under hydro_core */
int hydroBCs;          /*selector for hydro BC set. 0 = triply periodic, 1=baseStateBox, 2= periodicHorizBSVertical */

int hydroForcingWrite;     /*switching for dumping forcing fields of prognostic variables. 0-off (default), 1= on*/
int hydroForcingLog;     /*switch for logging Frhs summary metrics. 0-off (default), 1= on*/
int hydroSubGridWrite;   /*switch for SGS fields 0-off (default), 1= on*/
float *hydroFlds;        /*Base Adress of memory containing all prognostic variable fields under hydro_core */
float *hydroFldsFrhs;    /*Base Adress of memory containing all prognostic variable fields Frhs under hydro_core */
float *hydroFaceVels;      /*Base Adress of memory containing 3 cell-face velocity components under hydro_core */
float *hydroRhoInv;        /*Base Adress of memory to store (1.0/rho) */
float *hydroBaseStateFlds; /*Base Adress of memory containing field base-states for rho and theta */
float *hydroPres;          /*Base Adress of memory containing the diagnostic perturbation pressure field in Pascals*/
float *hydroBaseStatePres; /*Base Adress of memory containing the diagnostic base-state pressure field in Pascals*/ 
/*Constants*/
float R_gas;          /* The ideal gas constant in J/(kg*K) */
float R_vapor;        /* The ideal gas constant for water vapor in J/(kg*K) */
float Rv_Rg;          /* Ratio R_vapor/R_gas */
float cv_gas;         /* Specific heat of air at constant volume ****and temperature 300 K in J/(kg*K) */
float cp_gas;         /* Specific heat of air at constant pressure ****and temperature 300 K in J/(kg*K) */
float accel_g;        /* Acceleration of gravity 9.8 m/(s^2) */ 
float R_cp;           /* Ratio R/cp*/
float cp_R;           /* Ratio cp/R*/
float cp_cv;          /* Ratio cp/cv*/
float refPressure;    /* Reference pressure set constant to 1e5 Pascals or 1000 millibars) */
float kappa;          /* von Karman constant */
float L_v;            /* latent heat of vaporization (J/kg) */

/*HYDRO_CORE Submodule parameters*/
/*----Pressure Gradient Force*/ 
int pgfSelector;          /*Pressure Gradient Force (pgf) selector: 0=off, 1=on*/

/*----Buoyancy Force*/ 
int buoyancySelector;     /*buoyancy Force selector: 0=off, 1=on*/

/*----Coriolis*/ 
int coriolisSelector;   /* Coriolis selector, (0 = none, 1 = horizontal terms only, 2 = horizontal and vertical terms*/
float coriolisLatitude; /*Charactersitc latitude in degrees from equator of the LES domain*/
float corioConstHorz;   /*Latitude dependent horizontal Coriolis term constant */
float corioConstVert;   /*Latitude dependent Vertical Coriolis term constant */
float corioLS_fact;     /*large-scale factor on Coriolis term*/

/*----Turbulence*/ 
int turbulenceSelector;         /*turbulence scheme selector: 0= none, 1= Lilly/Smagorinsky */
int TKESelector;                /* Prognostic TKE selector: 0= none, 1= Prognostic */
int TKEAdvSelector;             /* SGSTKE advection scheme selector */
float TKEAdvSelector_b_hyb;     /*hybrid advection scheme parameter */
float c_s;                      /* Smagorinsky turbulence model constant used for turbulenceSelector = 1 with TKESelector = 0 */
float c_k;                      /* Lilly turbulence model constant used for turbulenceSelector = 1 with TKESelector > 0 */
float *sgstkeScalars;     /* Base Adress of memory containing all prognostic "sgstke" variable fields */ 
float *sgstkeScalarsFrhs; /* Base Adress of memory containing all prognostic "sgstke" RHS forcing fields */ 

/*----Advection*/ 
int advectionSelector;    /*advection scheme selector: 0= 1st-order upwind, 1= 3rd-order QUICK, 
                                              2= hybrid 3rd-4th order, 3= hybrid 5th-6th order */
float b_hyb;      /*hybrid advection scheme parameter: 0.0= lower-order upwind,
                                          1.0=higher-order cetered, 0.0 < b_hyb < 1.0 = hybrid */

/*----Diffusion*/ 
int diffusionSelector;    /*diffusion Term-type selector: 0= none, 1= constant, 2= scalar turbulent-diffusivity*/
float nu_0;               /* constant diffusivity used when diffusionSelector = 1 */
float* hydroDiffNuFld;    /*Base adress for diffusivity array used with all prognostic fields*/
float* hydroTauFlds;      /*Base address for scratch/work Tau tensor array*/
float* hydroDiffTauXFlds; /*Base adress for diffusion TauX arrays for all prognostic fields*/
float* hydroDiffTauYFlds; /*Base adress for diffusion TauY arrays for all prognostic fields*/
float* hydroDiffTauZFlds; /*Base adress for diffusion TauZ arrays for all prognostic fields*/

/*---Monin-Obukhov surface layer---*/ 
int surflayerSelector;    /*Monin-Obukhov surface layer selector: 0= off, 1= on */
float surflayer_z0;       /* roughness length (momentum) */
float surflayer_z0t;      /* roughness length (temperature) */
float surflayer_wth;      /* kinematic sensible heat flux at the surface */
float surflayer_tr;       /* surface temperature rate in K h-1 */
float surflayer_wq;       /* kinematic latent heat flux at the surface */
float surflayer_qr;       /* surface water vapor rate (kg/kg) h-1 */
int surflayer_qskin_input;/* selector to use file input (restart) value for qskin under surflayerSelector == 2 */
int surflayer_stab;       /* exchange coeffcient stability correction selector: 0= on, 1= off */
float* cdFld;             /*Base adress for momentum exchange coefficient (2d-array)*/
float* chFld;             /*Base adress for sensible heat exchange coefficient (2d-array)*/
float* cqFld;             /*Base address for latent heat exchange coefficient (2d-array)*/
float* fricVel;           /*Surface friction velocity "u*" 2-d array (x by y) (ms^-1)*/
float* htFlux;            /*Surface heat flux "(w'th')" 2-d array (x by y) (Kms^-1)*/
float* tskin;             /*Surface skin temperature 2-d array (x by y) (K)*/
float* qFlux;             /*Base address for latent heat flux*/
float* qskin;             /*Base address for skin moisture*/
float* invOblen;          /*Surface Monin-Obukhov length "()" 2-d array (x by y) (m)*/
float* z0m;               /*roughness length for momentum "()" 2-d array (x by y) (m)*/
float* z0t;               /*roughness length for temperature "()" 2-d array (x by y) (m)*/
int surflayer_idealsine;   /*selector for idealized sinusoidal surface heat flux or skin temperature forcing*/
float surflayer_ideal_ts;  /*start time in seconds for the idealized sinusoidal surface forcing*/
float surflayer_ideal_te;  /*end time in seconds for the idealized sinusoidal surface forcing*/
float surflayer_ideal_amp; /*maximum amplitude of the idealized sinusoidal surface forcing*/
float surflayer_ideal_qts;  /*start time in seconds for the idealized sinusoidal surface forcing of latent heat flux*/
float surflayer_ideal_qte;  /*end time in seconds for the idealized sinusoidal surface forcing of latent heat flux*/
float surflayer_ideal_qamp; /*maximum amplitude of the idealized sinusoidal surface forcing of latent heat flux*/

/*Offshore roughness parameters*/
int surflayer_offshore;         /* offshore selector: 0=off, 1=on */
int surflayer_offshore_opt;     /* offshore roughness parameterization: ==0 (Charnock), ==1 (Charnock with variable alpha), ==2 (Taylor & Yelland), ==3 (Donelan), ==4 (Drennan), ==5 (Porchetta) */
int surflayer_offshore_dyn;     /* selector to use parameterized ocean parameters: 0=off, 1=on (default) */
float surflayer_offshore_hs;    /* significant wave height */
float surflayer_offshore_lp;    /* peak wavelength */
float surflayer_offshore_cp;    /* wave phase speed */
float surflayer_offshore_theta; /* wave/wind angle */
int surflayer_offshore_visc;    /* viscous term on z0m: 0=off, 1=on (default) */
float* sea_mask;                /* Base Address of memory containing sea mask 0,1 field */

/*Large-scale forcings parameters*/ 
int lsfSelector;         /* large-scale forcings selector: 0=off, 1=on */
float lsf_w_surf;        /* lsf to w at the surface */
float lsf_w_lev1;        /* lsf to w at the first specified level */
float lsf_w_lev2;        /* lsf to w at the second specified level */
float lsf_w_zlev1;       /* lsf to w height 1 */
float lsf_w_zlev2;       /* lsf to w height 2 */
float lsf_th_surf;       /* lsf to theta at the surface */
float lsf_th_lev1;       /* lsf to theta at the first specified level */
float lsf_th_lev2;       /* lsf to theta at the second specified level */
float lsf_th_zlev1;      /* lsf to theta height 1 */
float lsf_th_zlev2;      /* lsf to theta height 2 */
float lsf_qv_surf;       /* lsf to qv at the surface */
float lsf_qv_lev1;       /* lsf to qv at the first specified level */
float lsf_qv_lev2;       /* lsf to qv at the second specified level */
float lsf_qv_zlev1;      /* lsf to qv height 1 */
float lsf_qv_zlev2;      /* lsf to qv height 2 */
int lsf_horMnSubTerms;   /* Switch 0=off, 1=on */
int lsf_numPhiVars;      /* number of variables in the slabMeanPhiProfiles set (e.g. rho,u,v,theta,qv=5) */
float lsf_freq;          /* large-scale forcing frequency (seconds) */

/*Moisture parameters*/ 
int moistureSelector;        /* moisture selector: 0=off, 1=on */
int moistureNvars;           /* number of moisture species */
int moistureAdvSelectorQv;     /* water vapor advection scheme selector */
float moistureAdvSelectorQv_b; /*hybrid advection scheme parameter */
int moistureSGSturb;         /* selector to apply sub-grid scale diffusion to moisture fields */
int moistureCond;            /* selector to apply condensation to mositure fields */
float *moistScalars;         /*Base address for moisture field arrays*/
float *moistScalarsFrhs;     /*Base address for moisture forcing field arrays*/
float *moistTauFlds;         /*Base address for SGS moisture field arrays*/
int moistureAdvSelectorQi; /* moisture advection scheme selector for non-qv fields (non-oscillatory schemes) */
float moistureCondTscale;  /* relaxation time in seconds */
int moistureCondBasePres;  /* selector to use base pressure for microphysics */
float moistureMPcallTscale;/* time scale for microphysics to be called */

/*Filters parameters*/
int filterSelector;               /* explicit filter selector: 0=off, 1=on */
int filter_6thdiff_vert;          /* vertical 6th-order filter on w selector: 0=off, 1=on */
float filter_6thdiff_vert_coeff;  /* vertical 6th-order filter factor: 0.0=off, 1.0=full */
int filter_6thdiff_hori;          /* horizontal 6th-order filter on rho,theta,qv selector: 0=off, 1=on */
float filter_6thdiff_hori_coeff;  /* horizontal 6th-order filter factor: 0.0=off, 1.0=full */
int filter_divdamp;               /* divergence damping selector: 0=off, 1=on */

/*--- Rayleigh Damping Layer ---*/
int dampingLayerSelector;       // Rayleigh Damping Layer selector
float dampingLayerDepth;       // Rayleigh Damping Layer Depth

/*---BASE_STATE*/
int stabilityScheme;  /*Base-State stability setup scheme, (0 = none, 1 = profile, 2 = linear in theta)*/
float temp_grnd;
float pres_grnd;
float rho_grnd;
float theta_grnd;
float zStableBottom;
float stableGradient;
float zStableBottom2;
float stableGradient2;
float zStableBottom3;
float stableGradient3;
float U_g;            /*Zonal (West-East) component of the geostrophic wind*/
float V_g;            /*Meridional (South-North)  component of the geostrophic wind*/
float z_Ug,z_Vg;
float Ug_grad,Vg_grad;
int thetaPerturbationSwitch; /* Initial theta perturbations switch: 0=off, 1=on*/
float thetaHeight; /* Initial theta perturbations maximum height*/
float thetaAmplitude; /* Initial theta perturbation (maximum amplitude in K)*/

int physics_oneRKonly; /* selector to apply physics RHS forcing only at the latest RK stage */
 
/*###################------------------- HYDRO_CORE module function definitions ---------------------#################*/

/*----->>>>> int hydro_coreGetParams();   ----------------------------------------------------------------------
Obtain the complete set of parameters for the HYDRO_CORE module
*/
int hydro_coreGetParams(){
   int errorCode = HYDRO_CORE_SUCCESS;

   /*query for each HYDRO_CORE parameter */
   hydroBCs = 0; //Default to triply-periodic
   errorCode = queryIntegerParameter("hydroBCs", &hydroBCs, 2, 2, PARAM_MANDATORY);
   hydroForcingWrite = 0; //Default to off
   errorCode = queryIntegerParameter("hydroForcingWrite", &hydroForcingWrite, 0, 1, PARAM_MANDATORY);
   hydroForcingLog = 0; //Default to off
   errorCode = queryIntegerParameter("hydroForcingLog", &hydroForcingLog, 0, 1, PARAM_MANDATORY);
   hydroSubGridWrite = 0; //Default to off
   errorCode = queryIntegerParameter("hydroSubGridWrite", &hydroSubGridWrite, 0, 1, PARAM_MANDATORY);
   pgfSelector = 1; //Default to off
   errorCode = queryIntegerParameter("pgfSelector", &pgfSelector, 0, 1, PARAM_OPTIONAL);
   buoyancySelector = 1; //Default to off
   errorCode = queryIntegerParameter("buoyancySelector", &buoyancySelector, 0, 1, PARAM_OPTIONAL);
   coriolisSelector = 0; //Default to off
   errorCode = queryIntegerParameter("coriolisSelector", &coriolisSelector, 0, 2, PARAM_MANDATORY);
   coriolisLatitude = 54.0; //Default to latitude 54.0 N 
   errorCode = queryFloatParameter("coriolisLatitude", &coriolisLatitude, -90.0, 90.0, PARAM_MANDATORY);
   turbulenceSelector = 0; //Default to off
   errorCode = queryIntegerParameter("turbulenceSelector", &turbulenceSelector, 0, 1, PARAM_MANDATORY);
   TKESelector = 0; //Default to none
   errorCode = queryIntegerParameter("TKESelector", &TKESelector, 0, 1, PARAM_MANDATORY);
   TKEAdvSelector = 0; //Default to 0 for monotonic 1st-order upstream
   errorCode = queryIntegerParameter("TKEAdvSelector", &TKEAdvSelector, 0, 6, PARAM_MANDATORY);
   TKEAdvSelector_b_hyb = 0.0; //Default to 0.0
   errorCode = queryFloatParameter("TKEAdvSelector_b_hyb", &TKEAdvSelector_b_hyb, 0.0, 1.0, PARAM_MANDATORY);
   c_s = 0.18; //Default to 0.18
   errorCode = queryFloatParameter("c_s", &c_s, 1e-6, 1e6, PARAM_MANDATORY);
   c_k = 0.10; //Default to 0.1
   errorCode = queryFloatParameter("c_k", &c_k, 1e-6, 1e6, PARAM_MANDATORY);
   advectionSelector = 0; //Default to 0
   errorCode = queryIntegerParameter("advectionSelector", &advectionSelector, 0, 6, PARAM_MANDATORY);
   b_hyb = 0.8; //Default to 0.8
   errorCode = queryFloatParameter("b_hyb", &b_hyb, 0.0, 1.0, PARAM_MANDATORY);
   diffusionSelector = 0; //Default to off
   errorCode = queryIntegerParameter("diffusionSelector", &diffusionSelector, 0, 1, PARAM_MANDATORY);
   nu_0 = 1.0; //Default to 1.0 m/s^2
   errorCode = queryFloatParameter("nu_0", &nu_0, 0, FLT_MAX, PARAM_MANDATORY);
   surflayerSelector = 0; // Default to off
   errorCode = queryIntegerParameter("surflayerSelector", &surflayerSelector, 0, 2, PARAM_MANDATORY);
   surflayer_z0 = 0.1; // Default to 0.1 m 
   errorCode = queryFloatParameter("surflayer_z0", &surflayer_z0, 1e-6, 1e+0, PARAM_MANDATORY);
   surflayer_z0t = 0.1; // Default to 0.1 m 
   errorCode = queryFloatParameter("surflayer_z0t", &surflayer_z0t, 1e-6, 1e+1, PARAM_MANDATORY);
   surflayer_tr = 0.0; // Default to 0.0 K h-1 
   errorCode = queryFloatParameter("surflayer_tr", &surflayer_tr, -1e+1, 1e+1, PARAM_MANDATORY);
   surflayer_wth = 0.0; // Default to 0.0 K m s-1 
   errorCode = queryFloatParameter("surflayer_wth", &surflayer_wth, -5e+0, 5e+0, PARAM_MANDATORY);
   surflayer_idealsine = 0; //Default to off 
   errorCode = queryIntegerParameter("surflayer_idealsine", &surflayer_idealsine, 0, 1, PARAM_MANDATORY);
   surflayer_ideal_ts = 0.0; // Default to 0.0 s
   surflayer_ideal_te = 0.0; // Default to 0.0 s
   surflayer_ideal_amp = 0.1; // Default to 0.1
   if (surflayer_idealsine > 0){
     errorCode = queryFloatParameter("surflayer_ideal_ts", &surflayer_ideal_ts, 0, 1e+5, PARAM_MANDATORY);
     errorCode = queryFloatParameter("surflayer_ideal_te", &surflayer_ideal_te, 0, 1e+5, PARAM_MANDATORY);
     errorCode = queryFloatParameter("surflayer_ideal_amp", &surflayer_ideal_amp, 0, 1e+3, PARAM_MANDATORY);
   }
   surflayer_wq = 0.0; // Default to 0.0 kg/kg m s-1 
   surflayer_qr = 0.0; // Default to 0.0 kg/kg h-1
   surflayer_qskin_input = 0; // Default to off
   surflayer_ideal_qts = 0.0; // Default to 0.0 s
   surflayer_ideal_qte = 0.0; // Default to 0.0 s
   surflayer_ideal_qamp = 0.1; // Default to 0.1
   //
   surflayer_stab = 0; // Default to on 
   errorCode = queryIntegerParameter("surflayer_stab", &surflayer_stab, 0, 1, PARAM_OPTIONAL);
   surflayer_offshore = 0; // Default to off
   surflayer_offshore_opt = 0;
   surflayer_offshore_dyn = 1;
   surflayer_offshore_hs = 0.0;
   surflayer_offshore_lp = 0.1;
   surflayer_offshore_cp = 0.1;
   surflayer_offshore_theta = 0.0;
   surflayer_offshore_visc = 1;
   errorCode = queryIntegerParameter("surflayer_offshore", &surflayer_offshore, 0, 1, PARAM_MANDATORY);
   errorCode = queryIntegerParameter("surflayer_offshore_visc", &surflayer_offshore_visc, 0, 1, PARAM_OPTIONAL);
   if (surflayer_offshore > 0){
     errorCode = queryIntegerParameter("surflayer_offshore_opt", &surflayer_offshore_opt, 0, 5, PARAM_MANDATORY);
     errorCode = queryIntegerParameter("surflayer_offshore_dyn", &surflayer_offshore_dyn, 0, 1, PARAM_OPTIONAL);
     if (surflayer_offshore_dyn == 0){
       if (surflayer_offshore_opt == 2){
         errorCode = queryFloatParameter("surflayer_offshore_hs", &surflayer_offshore_hs, 0, 1e+2, PARAM_MANDATORY);
         errorCode = queryFloatParameter("surflayer_offshore_lp", &surflayer_offshore_lp, 0.1, 1e+3, PARAM_MANDATORY);
       } else if (surflayer_offshore_opt == 3){
         errorCode = queryFloatParameter("surflayer_offshore_hs", &surflayer_offshore_hs, 0, 1e+2, PARAM_MANDATORY);
         errorCode = queryFloatParameter("surflayer_offshore_cp", &surflayer_offshore_cp, 0.1, 1e+2, PARAM_MANDATORY);
       } else if (surflayer_offshore_opt == 4){
         errorCode = queryFloatParameter("surflayer_offshore_hs", &surflayer_offshore_hs, 0, 1e+2, PARAM_MANDATORY);
         errorCode = queryFloatParameter("surflayer_offshore_cp", &surflayer_offshore_cp, 0.1, 1e+2, PARAM_MANDATORY);
       } else if (surflayer_offshore_opt == 5){
         errorCode = queryFloatParameter("surflayer_offshore_hs", &surflayer_offshore_hs, 0, 1e+2, PARAM_MANDATORY);
         errorCode = queryFloatParameter("surflayer_offshore_cp", &surflayer_offshore_cp, 0.1, 1e+2, PARAM_MANDATORY);
         errorCode = queryFloatParameter("surflayer_offshore_theta", &surflayer_offshore_theta, 0.0, 180.0, PARAM_MANDATORY);
       }
     } else if (surflayer_offshore_dyn==1){
       if (surflayer_offshore_opt == 5){
         errorCode = queryFloatParameter("surflayer_offshore_theta", &surflayer_offshore_theta, 0.0, 180.0, PARAM_MANDATORY);
       }
     }
   }
   //
   lsfSelector = 0; // Default to off 
   errorCode = queryIntegerParameter("lsfSelector", &lsfSelector, 0, 1, PARAM_MANDATORY);
   lsf_w_surf = 0.0; // Default to 0.0
   lsf_w_lev1 = 0.0; // Default to 0.0
   lsf_w_lev2 = 0.0; // Default to 0.0
   lsf_w_zlev1 = 100.0; // Default to 100.0
   lsf_w_zlev2 = 200.0; // Default to 200.0
   lsf_th_surf = 0.0; // Default to 0.0
   lsf_th_lev1 = 0.0; // Default to 0.0
   lsf_th_lev2 = 0.0; // Default to 0.0
   lsf_th_zlev1 = 100.0; // Default to 100.0
   lsf_th_zlev2 = 200.0; // Default to 200.0
   lsf_qv_surf = 0.0; // Default to 0.0
   lsf_qv_lev1 = 0.0; // Default to 0.0
   lsf_qv_lev2 = 0.0; // Default to 0.0
   lsf_qv_zlev1 = 100.0; // Default to 100.0
   lsf_qv_zlev2 = 200.0; // Default to 200.0
   lsf_horMnSubTerms = 0; //Default to 0=off
   lsf_freq = 1.0; // Default to 1 sec
   if (lsfSelector > 0){
     errorCode = queryFloatParameter("lsf_w_surf", &lsf_w_surf, -1e+4, 1e+4, PARAM_MANDATORY);
     errorCode = queryFloatParameter("lsf_w_lev1", &lsf_w_lev1, -1e+4, 1e+4, PARAM_MANDATORY);
     errorCode = queryFloatParameter("lsf_w_lev2", &lsf_w_lev2, -1e+4, 1e+4, PARAM_MANDATORY);
     errorCode = queryFloatParameter("lsf_w_zlev1", &lsf_w_zlev1, 0.0, 1e+4, PARAM_MANDATORY);
     errorCode = queryFloatParameter("lsf_w_zlev2", &lsf_w_zlev2, 0.0, 1e+4, PARAM_MANDATORY);
     errorCode = queryFloatParameter("lsf_th_surf", &lsf_th_surf, -1e+4, 1e+4, PARAM_MANDATORY);
     errorCode = queryFloatParameter("lsf_th_lev1", &lsf_th_lev1, -1e+4, 1e+4, PARAM_MANDATORY);
     errorCode = queryFloatParameter("lsf_th_lev2", &lsf_th_lev2, -1e+4, 1e+4, PARAM_MANDATORY);
     errorCode = queryFloatParameter("lsf_th_zlev1", &lsf_th_zlev1, 0.0, 1e+4, PARAM_MANDATORY);
     errorCode = queryFloatParameter("lsf_th_zlev2", &lsf_th_zlev2, 0.0, 1e+4, PARAM_MANDATORY);
     errorCode = queryFloatParameter("lsf_qv_surf", &lsf_qv_surf, -1e+4, 1e+4, PARAM_MANDATORY);
     errorCode = queryFloatParameter("lsf_qv_lev1", &lsf_qv_lev1, -1e+4, 1e+4, PARAM_MANDATORY);
     errorCode = queryFloatParameter("lsf_qv_lev2", &lsf_qv_lev2, -1e+4, 1e+4, PARAM_MANDATORY);
     errorCode = queryFloatParameter("lsf_qv_zlev1", &lsf_qv_zlev1, 0.0, 1e+4, PARAM_MANDATORY);
     errorCode = queryFloatParameter("lsf_qv_zlev2", &lsf_qv_zlev2, 0.0, 1e+4, PARAM_MANDATORY);
     errorCode = queryIntegerParameter("lsf_horMnSubTerms", &lsf_horMnSubTerms, 0, 1, PARAM_MANDATORY);
     errorCode = queryFloatParameter("lsf_freq", &lsf_freq, 1e-3, 1e+3, PARAM_MANDATORY);
     /*Initializing lsf_numPhiVars here when lsf_horMnSubTerms = 1
      * with strictly one fixed option implemented at this time of {rho,u,v,theta,qv} = 5 */
     if(lsf_horMnSubTerms==1){
       lsf_numPhiVars = 5;
       if(mpi_size_world>1){
         printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
         printf("!!!!!! WARNING: lsf_horMnSubTerms==1 is performed on a per-GPU basis !!!!!!\n");
         printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
         fflush(stdout);
       }
     }
   }
   moistureSelector = 0; // Default to off 
   errorCode = queryIntegerParameter("moistureSelector", &moistureSelector, 0, 1, PARAM_MANDATORY);
   moistureNvars = 0; // Default to 0
   moistureAdvSelectorQv = 0; // Default to 0
   moistureAdvSelectorQv_b = 0.0; // Default to 0.0
   moistureAdvSelectorQi = 0; // Default to 0
   moistureSGSturb = 0; // Default to 0
   moistureCond = 1; // Default to 1
   moistureCondTscale = 1.0; // Default to 1.0 s
   moistureCondBasePres = 0; // Default to off
   moistureMPcallTscale = 1.0; // Default to 1.0 s
   if (moistureSelector > 0){
     errorCode = queryIntegerParameter("moistureNvars", &moistureNvars, 0, 2, PARAM_MANDATORY);
     errorCode = queryIntegerParameter("moistureAdvSelectorQv", &moistureAdvSelectorQv, 0, 6, PARAM_MANDATORY);
     errorCode = queryFloatParameter("moistureAdvSelectorQv_b", &moistureAdvSelectorQv_b, 0.0, 1.0, PARAM_MANDATORY);
     errorCode = queryIntegerParameter("moistureAdvSelectorQi", &moistureAdvSelectorQi, 0, 2, PARAM_MANDATORY);
     errorCode = queryIntegerParameter("moistureSGSturb", &moistureSGSturb, 0, 1, PARAM_MANDATORY);
     errorCode = queryIntegerParameter("moistureCond", &moistureCond, 1, 4, PARAM_MANDATORY);
     errorCode = queryFloatParameter("moistureCondTscale", &moistureCondTscale, 1e-4, 1000.0, PARAM_MANDATORY);
     errorCode = queryIntegerParameter("moistureCondBasePres", &moistureCondBasePres, 0, 1, PARAM_MANDATORY);
     errorCode = queryFloatParameter("moistureMPcallTscale", &moistureMPcallTscale, 1e-4, 1000.0, PARAM_MANDATORY);
     errorCode = queryFloatParameter("surflayer_wq", &surflayer_wq, -5e+0, 5e+0, PARAM_MANDATORY);
     errorCode = queryFloatParameter("surflayer_qr", &surflayer_qr, -1e+1, 1e+1, PARAM_MANDATORY);
     errorCode = queryIntegerParameter("surflayer_qskin_input", &surflayer_qskin_input, 0, 1, PARAM_OPTIONAL);
     if (surflayer_idealsine > 0){
       errorCode = queryFloatParameter("surflayer_ideal_qts", &surflayer_ideal_qts, 0, 1e+5, PARAM_MANDATORY);
       errorCode = queryFloatParameter("surflayer_ideal_qte", &surflayer_ideal_qte, 0, 1e+5, PARAM_MANDATORY);
       errorCode = queryFloatParameter("surflayer_ideal_qamp", &surflayer_ideal_qamp, 0, 1e+3, PARAM_MANDATORY);
     }
   }
   filterSelector = 0; // Default to off
   filter_6thdiff_vert = 0; // Default to off
   filter_6thdiff_vert_coeff = 0.03; // Default to 0.03
   filter_6thdiff_hori = 0; // Default to off
   filter_6thdiff_hori_coeff = 0.03; // Default to 0.03
   filter_divdamp = 0; // Default to off
   errorCode = queryIntegerParameter("filterSelector", &filterSelector, 0, 1, PARAM_MANDATORY);
   if (filterSelector == 1){
     errorCode = queryIntegerParameter("filter_6thdiff_vert", &filter_6thdiff_vert, 0, 1, PARAM_OPTIONAL);
     errorCode = queryIntegerParameter("filter_6thdiff_hori", &filter_6thdiff_hori, 0, 1, PARAM_OPTIONAL);
     errorCode = queryIntegerParameter("filter_divdamp", &filter_divdamp, 0, 1, PARAM_OPTIONAL);
     if (filter_6thdiff_vert == 1){
       errorCode = queryFloatParameter("filter_6thdiff_vert_coeff", &filter_6thdiff_vert_coeff, 0.0, 1.0, PARAM_MANDATORY);
     }
     if (filter_6thdiff_hori == 1){
       errorCode = queryFloatParameter("filter_6thdiff_hori_coeff", &filter_6thdiff_hori_coeff, 0.0, 1.0, PARAM_MANDATORY);
     }
   }
   dampingLayerSelector = 0; // Default to off 
   errorCode = queryIntegerParameter("dampingLayerSelector", &dampingLayerSelector, 0, 1, PARAM_MANDATORY);
   dampingLayerDepth = 100.0; //Default to 100.0 (meters)  
   errorCode = queryFloatParameter("dampingLayerDepth", &dampingLayerDepth, 0.0, FLT_MAX, PARAM_MANDATORY);
   stabilityScheme = 0; //Default to constant rho & theta
   errorCode = queryIntegerParameter("stabilityScheme", &stabilityScheme, 0, 4, PARAM_MANDATORY);
   temp_grnd = 300.0; //Default to 300.0-(Kelvin) = 80.33-(Fahrenheit) = 26.85-(Celsius) 
   errorCode = queryFloatParameter("temp_grnd", &temp_grnd, FLT_MIN, FLT_MAX, PARAM_MANDATORY);
   pres_grnd = 1.0e5; //Default to refPressure 100,000-(pascals) = 1000-(millibars)
   errorCode = queryFloatParameter("pres_grnd", &pres_grnd, FLT_MIN, FLT_MAX, PARAM_MANDATORY);
   zStableBottom = 1000.0; //Default to 1000 meters for the bottom of the first upper stable layer
   errorCode = queryFloatParameter("zStableBottom", &zStableBottom, 0, FLT_MAX, PARAM_MANDATORY);
   stableGradient = 0.1; //Default to 0.1 K/meter for the vertical gradient of the first upper stable layer
   errorCode = queryFloatParameter("stableGradient", &stableGradient, FLT_MIN, FLT_MAX, PARAM_MANDATORY);
   zStableBottom2 = 1100.0; //Default to 1100 meters for the bottom of the second upper stable layer
   errorCode = queryFloatParameter("zStableBottom2", &zStableBottom2, 0, FLT_MAX, PARAM_MANDATORY);
   stableGradient2 = 0.03; //Default to 0.03 K/meter for the vertical gradient of the second upper stable layer
   errorCode = queryFloatParameter("stableGradient2", &stableGradient2, FLT_MIN, FLT_MAX, PARAM_MANDATORY);
   zStableBottom3 = 1500.0; //Default to 1500 meters for the bottom of the third upper stable layer
   errorCode = queryFloatParameter("zStableBottom3", &zStableBottom3, 0, FLT_MAX, PARAM_MANDATORY);
   stableGradient3 = 0.03; //Default to 0.03 K/meter for the vertical gradient of third upper stable layer
   errorCode = queryFloatParameter("stableGradient3", &stableGradient3, FLT_MIN, FLT_MAX, PARAM_MANDATORY);
   U_g = 0.0; //Default to 0.0 meters/second for the zonal component of the geostrophic wind
   errorCode = queryFloatParameter("U_g", &U_g, -FLT_MAX, FLT_MAX, PARAM_MANDATORY);
   V_g = 0.0; //Default to 0.0 meters/second for the meridional component of the geostrophic wind
   errorCode = queryFloatParameter("V_g", &V_g, -FLT_MAX, FLT_MAX, PARAM_MANDATORY);
   z_Ug = 10000.0; //Default to 10000.0 m
   errorCode = queryFloatParameter("z_Ug", &z_Ug, 0.0, FLT_MAX, PARAM_MANDATORY);
   z_Vg = 10000.0; //Default to 10000.0 m
   errorCode = queryFloatParameter("z_Vg", &z_Vg, 0.0, FLT_MAX, PARAM_MANDATORY);
   Ug_grad = 0.0; //Default to 0.0 (m/s)/m
   errorCode = queryFloatParameter("Ug_grad", &Ug_grad, -1e2, 1e2, PARAM_MANDATORY);
   Vg_grad = 0.0; //Default to 0.0 (m/s)/m
   errorCode = queryFloatParameter("Vg_grad", &Vg_grad, -1e2, 1e2, PARAM_MANDATORY);
   thetaPerturbationSwitch = 0; //Default to initial theta perturbations off
   errorCode = queryIntegerParameter("thetaPerturbationSwitch", &thetaPerturbationSwitch, 0, 1, PARAM_MANDATORY);
   thetaHeight = 0.0; //Default to 0.0 meters for initial theta perturbation maximum height
   errorCode = queryFloatParameter("thetaHeight", &thetaHeight, 0.0, FLT_MAX, PARAM_MANDATORY);
   thetaAmplitude = 0.0; //Default to +- K for initial theta perturbation maximum height
   errorCode = queryFloatParameter("thetaAmplitude", &thetaAmplitude, 0.0, 2.0, PARAM_MANDATORY);

   physics_oneRKonly = 1; //Default 1 (physics only at the last stage of RK scheme)
   errorCode = queryIntegerParameter("physics_oneRKonly", &physics_oneRKonly, 0, 1, PARAM_OPTIONAL);

   return(errorCode);
} //end hydro_coreGetParams()

/*----->>>>> int hydro_coreInit();   ----------------------------------------------------------------------
* Used to broadcast and print parameters, allocate memory, and initialize configuration settings for HYDRO_CORE.
*/
int hydro_coreInit(){
   int errorCode = HYDRO_CORE_SUCCESS;
   int i,j,k,ijk,ij;
   int iFld; //simple integer index for the ith Fld in the hydro_core memory block, hydroFlds.
   int iFld2; //simple integer index
   char fldName[MAX_HC_FLDNAME_LENGTH];
   char frhsName[MAX_HC_FLDNAME_LENGTH+2];
   char TauScName[MAX_HC_FLDNAME_LENGTH];
   char sgstkeScName[MAX_HC_FLDNAME_LENGTH];
   char moistName[MAX_HC_FLDNAME_LENGTH];
   char moistName_base[MAX_HC_FLDNAME_LENGTH];
   char moistName_tmp[MAX_HC_FLDNAME_LENGTH];
   float pi;
   int fldStride;
   float z1oz0,z1,z1ozt0;

   MPI_Barrier(MPI_COMM_WORLD); 
   printf("Entering hydro_coreInit:------\n"); 
   fflush(stdout);
   /*Print the module parameters we are using.*/
   if(mpi_rank_world == 0){
      printComment("HYDRO_CORE parameters---");
      printComment("----------: HYDRO_CORE Submodule Selectors ---");
      printComment("----------: Boundary Conditions Set ---");
      printParameter("hydroBCs", "Selector for hydro BC set. 2= periodicHorizVerticalAbl");
      printParameter("hydroForcingWrite", "Switch for dumping hydroFldsFrhs for prognositic fields. 0 = off, 1=on");
      printParameter("hydroSubGridWrite", "Switch for dumping Tauij fields. 0 = off, 1=on");
      printParameter("hydroForcingLog", "Switch for logging Frhs summary metrics. 0 = off, 1=on");
      printComment("----------: PRESSURE GRADIENT FORCE ---");
      printParameter("pgfSelector", "Pressure Gradient Force (pgf) selector: 0=off, 1=on");
      printComment("----------: BUOYANCY ---");
      printParameter("buoyancySelector", "Buoyancy force  selector: 0=off, 1=on");
      printComment("----------: CORIOLIS ---");
      printParameter("coriolisSelector", "Corilis force selector: 0= none, 1= horiz. terms, 2= horiz. & vert. terms");
      printParameter("coriolisLatitude", "Charactersitc latitude in degrees from equator of the LES domain");
      printComment("----------: TURBULENCE ---");
      printParameter("turbulenceSelector", "turbulence scheme selector: 0= none, 1= Lilly/Smagorinsky ");
      printParameter("TKESelector", "Prognostic TKE selector: 0= none, 1= Prognostic");
      printParameter("TKEAdvSelector", "advection scheme for SGSTKE equation");
      printParameter("TKEAdvSelector_b_hyb","hybrid advection scheme parameter");
      printParameter("c_s", "Smagorinsky model constant used for turbulenceSelector = 1 and TKESelector = 0");
      printParameter("c_k", "Lilly model constant used for turbulenceSelector = 1 and TKESelector > 0");
      printComment("----------: ADVECTION ---");
      printParameter("advectionSelector", "advection scheme selector: 0= 1st-order upwind, 1= 3rd-order QUICK, 2= hybrid 3rd-4th order, 3= hybrid 5th-6th order");
      printParameter("b_hyb", "hybrid advection scheme parameter: 0.0= lower-order upwind, 1.0=higher-order cetered, 0.0 < b_hyb < 1.0 = hybrid");
      printComment("----------: DIFFUSION ---");
      printParameter("diffusionSelector", "diffusivity selector: 0= none, 1= const.");
      printParameter("nu_0", "constant diffusivity used when diffusionSelector = 1");
      printComment("----------: SURFACE LAYER ---"); 
      printParameter("surflayerSelector", "surfacelayer selector: 0= off, 1,2= on");
      printParameter("surflayer_z0", "roughness length (momentum) when surflayerSelector > 0");
      printParameter("surflayer_z0t", "roughness length (temperature) when surflayerSelector > 0");
      printParameter("surflayer_wth", "kinematic sensible heat flux at the surface when surflayerSelector = 1");
      printParameter("surflayer_wq", "kinematic latent heat flux at the surface when surflayerSelector = 1");
      printParameter("surflayer_tr", "temperature rate at the surface when surflayerSelector = 2 (>0 for warming; <0 for cooling)");
      printParameter("surflayer_qr", "moisture rate at the surface when surflayerSelector = 2 (>0 for warming; <0 for cooling)");
      printParameter("surflayer_qskin_input", "selector to use file input (restart) value for qskin under surflayerSelector == 2");
      printParameter("surflayer_idealsine", "selector for idealized sinusoidal surface heat flux or skin temperature forcing: 0= off, 1= on");
      printParameter("surflayer_ideal_ts", "start time in seconds for the idealized sinusoidal surface forcing");
      printParameter("surflayer_ideal_te", "end time in seconds for the idealized sinusoidal surface forcing");
      printParameter("surflayer_ideal_amp", "maximum amplitude of the idealized sinusoidal surface forcing");
      printParameter("surflayer_ideal_qts", "start time in seconds for the idealized sinusoidal surface forcing (qv)");
      printParameter("surflayer_ideal_qte", "end time in seconds for the idealized sinusoidal surface forcing (qv)");
      printParameter("surflayer_ideal_qamp", "maximum amplitude of the idealized sinusoidal surface forcing (qv)");
      printParameter("surflayer_stab", "exchange coeffcient stability correction selector: 0= on, 1= off");
      printComment("----------: OFFSHORE ROUGHNESS ---");
      printParameter("surflayer_offshore", "offshore selector: 0=off, 1=on");
      printParameter("surflayer_offshore_opt", "offshore roughness parameterization: ==0 (Charnock), ==1 (Charnock with variable alpha), ==2 (Taylor & Yelland), ==3 (Donelan), ==4 (Drennan), ==5 (Porchetta)");
      printComment("----------: LARGE-SCALE FORCINGS MODEL ---");
      printParameter("lsfSelector", "large-scale forcings selector: 0= off, 1= on");
      if (lsfSelector > 0){
        printParameter("lsf_w_surf", "large-scale forcing to w at the first specified level");
        printParameter("lsf_w_lev1", "large-scale forcing w at height 1");
        printParameter("lsf_w_lev2", "large-scale forcing w at height 2");
        printParameter("lsf_w_zlev1", "large-scale forcing w height 1");
        printParameter("lsf_w_zlev2", "large-scale forcing w height 2");
        printParameter("lsf_th_surf", "large-scale forcing to theta at the first specified level");
        printParameter("lsf_th_lev1", "large-scale forcing theta at height 1");
        printParameter("lsf_th_lev2", "large-scale forcing theta at height 2");
        printParameter("lsf_th_zlev1", "large-scale forcing theta height 1");
        printParameter("lsf_th_zlev2", "large-scale forcing theta height 2");
        printParameter("lsf_qv_surf", "large-scale forcing to qv at the first specified level");
        printParameter("lsf_qv_lev1", "large-scale forcing qv at height 1");
        printParameter("lsf_qv_lev2", "large-scale forcing qv at height 2");
        printParameter("lsf_qv_zlev1", "large-scale forcing qv height 1");
        printParameter("lsf_qv_zlev2", "large-scale forcing qv height 2");
        printParameter("lsf_horMnSubTerms", "large-scale subsidence terms Switch: 0= off, 1= on");
        printParameter("lsf_freq", "large-scale forcing frequency (seconds)");
      }
      printComment("----------: MOISTURE ---");
      printParameter("moistureSelector", "moisture selector: 0= off, 1= on");
      if (moistureSelector > 0){
        printParameter("moistureNvars", "number of moisture species");
        printParameter("moistureAdvSelectorQv", "water vapor advection scheme selector");
        printParameter("moistureAdvSelectorQv_b", "hybrid advection scheme parameter for water vapor");
        printParameter("moistureAdvSelectorQi", "moisture advection scheme selector for non-qv fields (non-oscillatory schemes)");
        printParameter("moistureSGSturb", "selector to apply sub-grid scale diffusion to moisture fields");
        printParameter("moistureCond", "selector to apply condensation to moisture fields");
        printParameter("moistureCondTscale", "relaxation time in seconds");
        printParameter("moistureCondBasePres", "selector to use base pressure for microphysics");
        printParameter("moistureMPcallTscale", "time scale for microphysics to be called (in seconds)");
      }
      printComment("----------: EXPLICIT FILTERS ---");
      printParameter("filterSelector", "explicit filter selector: 0=off, 1=on");
      printParameter("filter_6thdiff_vert", "vertical 6th-order filter on w selector: 0=off, 1=on");
      printParameter("filter_6thdiff_vert_coeff", "vertical 6th-order filter factor: 0.0=off, 1.0=full");
      printParameter("filter_6thdiff_hori", "horizontal 6th-order filter on rho,theta,qv selector: 0=off, 1=on");
      printParameter("filter_6thdiff_hori_coeff", "horizontal 6th-order filter factor: 0.0=off, 1.0=full");
      printParameter("filter_divdamp", "divergence damping selector: 0=off, 1=on");
      printComment("----------: RAYLEIGH DAMPING LAYER ---"); 
      printParameter("dampingLayerSelector", "Rayleigh damping layer selector: 0= off, 1= on.");
      printParameter("dampingLayerDepth", "Rayleigh damping layer depth in meters");
      printComment("----------: BASE-STATE ---");
      printParameter("stabilityScheme", "Scheme used to set hydrostatic, stability-dependent Base-State EOS fields");
      printParameter("temp_grnd", "Air Temperature (K) at the ground used to set hydrostatic Base-State EOS fields");
      printParameter("pres_grnd", "Pressure (Pa) at the ground used to set hydrostatic Base-State EOS fields");
      printParameter("zStableBottom", "Height (m) of the first stable upper-layer when stabilityScheme = 1 or 2");
      printParameter("stableGradient", 
                     "Vertical gradient (K/m) of the first stable upper-layer when stabilityScheme = 1 or 2");
      printParameter("zStableBottom2", "Height (m) of the second stable upper-layer when stabilityScheme = 2");
      printParameter("zStableBottom3", "Height (m) of the third stable upper-layer when stabilityScheme = 2");
      printParameter("stableGradient2",
                     "Vertical gradient (K/m) of the second stable upper-layer when stabilityScheme = 2");
      printParameter("stableGradient3",
                     "Vertical gradient (K/m) of the third stable upper-layer when stabilityScheme = 2");
      printParameter("U_g", "Zonal (West-East) component of the geostrophic wind (m/s).");
      printParameter("V_g", "Meridional (South-North) component of the geostrophic wind (m/s).");
      printParameter("z_Ug", "Height (m) above ground for linear geostrophic wind gradient (zonal component).");
      printParameter("z_Vg", "Height (m) above ground for linear geostrophic wind gradient (meridional component).");
      printParameter("Ug_grad", "U_g gradient above z_Ug (ms-1/m).");
      printParameter("Vg_grad", "V_g gradient above z_Vg (ms-1/m).");
      printParameter("thetaPerturbationSwitch", "Switch to include initial theta perturbations: 0=off, 1=on");
      printParameter("thetaHeight", "Height below which to include initial theta perturbations: (meters)");
      printParameter("thetaAmplitude", "Maximum amplitude for theta perturbations: thetaAmplitude*[-1,+1] K");
      printParameter("physics_oneRKonly", "selector to apply physics RHS forcing only at the latest RK stage: 0= off, 1= on");
   } //end if(mpi_rank_world == 0)

   /*Broadcast the parameters across mpi_ranks*/
   MPI_Bcast(&hydroBCs, 1, MPI_INT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&hydroForcingWrite, 1, MPI_INT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&hydroSubGridWrite, 1, MPI_INT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&hydroForcingLog, 1, MPI_INT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&pgfSelector, 1, MPI_INT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&buoyancySelector, 1, MPI_INT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&coriolisSelector, 1, MPI_INT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&coriolisLatitude, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&turbulenceSelector, 1, MPI_INT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&TKESelector, 1, MPI_INT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&TKEAdvSelector, 1, MPI_INT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&TKEAdvSelector_b_hyb, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&c_s, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&c_k, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&advectionSelector, 1, MPI_INT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&b_hyb, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&diffusionSelector, 1, MPI_INT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&nu_0, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&surflayerSelector, 1, MPI_INT, 0, MPI_COMM_WORLD); 
   MPI_Bcast(&surflayer_z0, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&surflayer_z0t, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&surflayer_tr, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&surflayer_wth, 1, MPI_FLOAT, 0, MPI_COMM_WORLD); 
   MPI_Bcast(&surflayer_qr, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&surflayer_wq, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&surflayer_qskin_input, 1, MPI_INT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&surflayer_idealsine, 1, MPI_INT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&surflayer_ideal_ts, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&surflayer_ideal_te, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&surflayer_ideal_amp, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&surflayer_ideal_qts, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&surflayer_ideal_qte, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&surflayer_ideal_qamp, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&surflayer_stab, 1, MPI_INT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&surflayer_offshore, 1, MPI_INT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&surflayer_offshore_opt, 1, MPI_INT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&surflayer_offshore_dyn, 1, MPI_INT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&surflayer_offshore_hs, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&surflayer_offshore_lp, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&surflayer_offshore_cp, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&surflayer_offshore_theta, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&surflayer_offshore_visc, 1, MPI_INT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&lsfSelector, 1, MPI_INT, 0, MPI_COMM_WORLD);
   if (lsfSelector > 0){
     MPI_Bcast(&lsf_w_surf, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
     MPI_Bcast(&lsf_w_lev1, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
     MPI_Bcast(&lsf_w_lev2, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
     MPI_Bcast(&lsf_w_zlev1, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
     MPI_Bcast(&lsf_w_zlev2, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
     MPI_Bcast(&lsf_th_surf, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
     MPI_Bcast(&lsf_th_zlev1, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
     MPI_Bcast(&lsf_th_zlev2, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
     MPI_Bcast(&lsf_th_lev1, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
     MPI_Bcast(&lsf_th_lev2, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
     MPI_Bcast(&lsf_qv_surf, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
     MPI_Bcast(&lsf_qv_zlev1, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
     MPI_Bcast(&lsf_qv_zlev2, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
     MPI_Bcast(&lsf_qv_lev1, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
     MPI_Bcast(&lsf_qv_lev2, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
     MPI_Bcast(&lsf_horMnSubTerms, 1, MPI_INT, 0, MPI_COMM_WORLD);
     MPI_Bcast(&lsf_freq, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
     if (lsf_horMnSubTerms==1){
       MPI_Bcast(&lsf_numPhiVars, 1, MPI_INT, 0, MPI_COMM_WORLD);
     }
   }
   MPI_Bcast(&moistureSelector, 1, MPI_INT, 0, MPI_COMM_WORLD);
   if (moistureSelector > 0){
     MPI_Bcast(&moistureNvars, 1, MPI_INT, 0, MPI_COMM_WORLD);
     MPI_Bcast(&moistureAdvSelectorQv, 1, MPI_INT, 0, MPI_COMM_WORLD);
     MPI_Bcast(&moistureAdvSelectorQv_b, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
     MPI_Bcast(&moistureAdvSelectorQi, 1, MPI_INT, 0, MPI_COMM_WORLD);
     MPI_Bcast(&moistureSGSturb, 1, MPI_INT, 0, MPI_COMM_WORLD);
     MPI_Bcast(&moistureCond, 1, MPI_INT, 0, MPI_COMM_WORLD);
     MPI_Bcast(&moistureCondTscale, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
     MPI_Bcast(&moistureCondBasePres, 1, MPI_INT, 0, MPI_COMM_WORLD);
     MPI_Bcast(&moistureMPcallTscale, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
   }
   MPI_Bcast(&filterSelector, 1, MPI_INT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&filter_6thdiff_vert, 1, MPI_INT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&filter_6thdiff_vert_coeff, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&filter_6thdiff_hori, 1, MPI_INT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&filter_6thdiff_hori_coeff, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&filter_divdamp, 1, MPI_INT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&dampingLayerSelector, 1, MPI_INT, 0, MPI_COMM_WORLD); 
   MPI_Bcast(&dampingLayerDepth, 1, MPI_FLOAT, 0, MPI_COMM_WORLD); 
   MPI_Bcast(&stabilityScheme, 1, MPI_INT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&temp_grnd, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&pres_grnd, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&zStableBottom, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&stableGradient, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&zStableBottom2, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&stableGradient2, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&zStableBottom3, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&stableGradient3, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&U_g, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&V_g, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&z_Ug, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&z_Vg, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&Ug_grad, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&Vg_grad, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&thetaPerturbationSwitch, 1, MPI_INT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&thetaHeight, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&thetaAmplitude, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&physics_oneRKonly, 1, MPI_INT, 0, MPI_COMM_WORLD);

   printf("hydro_coreInit: allocating/registering arrays/fields with IO\n"); 
   fflush(stdout);
   /* Allocate the HYDRO_CORE arrays */
   /* Field  Arrays */
   hydroFlds = memAllocateFloat4DField(Nhydro, Nxp, Nyp, Nzp, Nh, "hydroFlds");
   fldStride = (Nxp+2*Nh)*(Nyp+2*Nh)*(Nzp+2*Nh);

   /*Register these fields with the IO module*/
   for(iFld = 0; iFld < Nhydro; iFld ++){
     errorCode = hydro_coreGetFieldName( &fldName[0], iFld);
     errorCode = ioRegisterVar(&fldName[0], "float", 4, dims4d, &hydroFlds[iFld*fldStride]);
     printf("hydro_coreInit:hydroFlds[%d] = %s stored at %p, has been registered with IO.\n",
            iFld,&fldName[0],&hydroFlds[iFld*fldStride]);
     fflush(stdout);
   } //end for iFld...
   
   /* Diagnostic Perturbation Pressure field */
   hydroPres = memAllocateFloat3DField(Nxp, Nyp, Nzp, Nh, "hydroPres");
   errorCode = sprintf(&fldName[0],"pressure");
   errorCode = ioRegisterVar(&fldName[0], "float", 4, dims4d, hydroPres);
   printf("hydro_coreInit:Field = %s stored at %p, has been registered with IO.\n",
          &fldName[0],hydroPres);
   
   /* Frhs */
   hydroFldsFrhs = memAllocateFloat4DField(Nhydro, Nxp, Nyp, Nzp, Nh, "hydroFldsFrhs");
   for(iFld = 0; iFld < Nhydro; iFld ++){
     if(hydroForcingWrite == 1){
       errorCode = hydro_coreGetFieldName( &fldName[0], iFld);
       sprintf(&frhsName[0],"F_%s",&fldName[0]);
       errorCode = ioRegisterVar(&frhsName[0], "float", 4, dims4d, &hydroFldsFrhs[iFld*fldStride]);
       printf("hydro_coreInit:hydroFldsFrhs[%d] = %s stored at %p, has been registered with IO.\n",
              iFld,&frhsName[0],&hydroFldsFrhs[iFld*fldStride]);
     }
   } //end for iFld...

   /* Prognostic SGSTKE equation and associated Frhs */ 
   if(TKESelector > 0){
     sgstkeScalars = memAllocateFloat4DField(TKESelector, Nxp, Nyp, Nzp, Nh, "sgstkeScalars");
     sgstkeScalarsFrhs = memAllocateFloat4DField(TKESelector, Nxp, Nyp, Nzp, Nh, "sgstkeScalarsFrhs");
     for(iFld = 0; iFld < TKESelector; iFld ++){
        sprintf(&sgstkeScName[0],"TKE_%d",iFld);
        errorCode = ioRegisterVar(&sgstkeScName[0], "float", 4, dims4d, &sgstkeScalars[iFld*fldStride]);
        printf("hydro_coreInit:sgstkeScalars[%d] = %s stored at %p, has been registered with IO.\n",
               iFld,&sgstkeScName[0],&sgstkeScalars[iFld*fldStride]);
        fflush(stdout);
     } //end for iFld...
     if(hydroForcingWrite == 1){ // add rhs forcing of SGSTKE equation
       for(iFld = 0; iFld < TKESelector; iFld ++){
         sprintf(&sgstkeScName[0],"F_TKE%d",iFld);
         errorCode = ioRegisterVar(&sgstkeScName[0], "float", 4, dims4d, &sgstkeScalarsFrhs[iFld*fldStride]);
         printf("hydro_coreInit:sgstkeScalarsFrhs[%d] = %s stored at %p, has been registered with IO.\n",
                iFld,&sgstkeScName[0],&sgstkeScalarsFrhs[iFld*fldStride]);
       }
     }
   } //end if TKESelector > 0

   printf("hydro_coreInit: allocating internal arrays\n"); 
   fflush(stdout);
   /* Face Velocities */
   hydroFaceVels = memAllocateFloat4DField(3, Nxp, Nyp, Nzp, Nh, "hydroFaceVels");
   /* The inverse of rho (1.0/rho) */
   hydroRhoInv = memAllocateFloat3DField(Nxp, Nyp, Nzp, Nh, "hydroRhoInv");
   /* Base states for rho and theta */
   hydroBaseStateFlds = memAllocateFloat4DField(2, Nxp, Nyp, Nzp, Nh, "hydroBaseStateFlds");
   /* Diagnostic Base-state  Pressure field */
   hydroBaseStatePres = memAllocateFloat3DField(Nxp, Nyp, Nzp, Nh, "hydroBaseStatePres");
#ifdef BASE_STATE_IO_DUMP
//#if 1
   for(iFld = 0; iFld < 2; iFld ++){
     errorCode = hydro_coreGetFieldName( &fldName[0], iFld);
     errorCode = sprintf(&fldName[0],"BS_%d",iFld);
     errorCode = ioRegisterVar(&fldName[0], "float", 4, dims4d, &hydroBaseStateFlds[iFld*fldStride]);
     printf("hydro_coreInit:hydroBaseStateFlds[%d] = %s stored at %p, has been registered with IO.\n",
            iFld,&fldName[0],&hydroBaseStateFlds[iFld*fldStride]);
     fflush(stdout);
   } //end for iFld...
   errorCode = sprintf(&fldName[0],"BS_pressure");
   errorCode = ioRegisterVar(&fldName[0], "float", 4, dims4d, hydroBaseStatePres);
   printf("hydro_coreInit:Field = %s stored at %p, has been registered with IO.\n",
          &fldName[0],hydroBaseStatePres);
#endif

   if(turbulenceSelector > 0){
     hydroDiffNuFld = memAllocateFloat3DField(Nxp, Nyp, Nzp, Nh, "hydroDiffNuFld");
     hydroTauFlds = memAllocateFloat4DField(9, Nxp, Nyp, Nzp, Nh, "hydroTauFlds");
   } //end if turbulencSelector > 0 && diffusionSelector == 0
   if(diffusionSelector > 0){
     /* Diffusion-- cell face "gradient of a field" tensor array */
     hydroDiffNuFld = memAllocateFloat3DField(Nxp, Nyp, Nzp, Nh, "hydroDiffNuFld");
     hydroDiffTauXFlds = memAllocateFloat4DField(Nhydro, Nxp, Nyp, Nzp, Nh, "hydroDiffTauXFlds");
     hydroDiffTauYFlds = memAllocateFloat4DField(Nhydro, Nxp, Nyp, Nzp, Nh, "hydroDiffTauYFlds");
     hydroDiffTauZFlds = memAllocateFloat4DField(Nhydro, Nxp, Nyp, Nzp, Nh, "hydroDiffTauZFlds");
     if(diffusionSelector == 1){
       /*Set a constant diffusivity */
       for(i=iMin-Nh; i < iMax+Nh; i++){
          for(j=jMin-Nh; j < jMax+Nh; j++){
             for(k=kMin-Nh; k < kMax+Nh; k++){
                ijk = i*(Nyp+2*Nh)*(Nzp+2*Nh)+j*(Nzp+2*Nh)+k;
                hydroDiffNuFld[ijk] = nu_0;
             } //end for(k...
          } // end for(j...
       } // end for(i...   
     }//end if diffusionSelector == 1}
   }//end if diffusionSelector == 0
   if((hydroSubGridWrite == 1) && (turbulenceSelector > 0)){
     for(iFld = 0; iFld < 6; iFld ++){
        switch (iFld){
         case 0:
           sprintf(&TauScName[0],"Tau%d%d",1,1);
           break;
         case 1:
           sprintf(&TauScName[0],"Tau%d%d",2,1);
           break;
         case 2:
           sprintf(&TauScName[0],"Tau%d%d",3,1);
           break;
         case 3:
           sprintf(&TauScName[0],"Tau%d%d",3,2);
           break;
         case 4:
           sprintf(&TauScName[0],"Tau%d%d",2,2);
           break;
         case 5:
           sprintf(&TauScName[0],"Tau%d%d",3,3);
           break;
         default:    //invalid iFld value
           printf("hydro_coreInit:hydroTauFlds[iFld=%d], invalid value for iFld.\n",iFld);
           errorCode = -1;
           break;
        }//end switch(iFld)
        errorCode = ioRegisterVar(&TauScName[0], "float", 4, dims4d, &hydroTauFlds[iFld*fldStride]);
        printf("hydro_coreInit:hydroTauFlds[%d] = %s stored at %p, has been registered with IO.\n",
               iFld,&TauScName[0],&hydroTauFlds[iFld*fldStride]);
        fflush(stdout);
     } //end for iFld...
     sprintf(&TauScName[0],"TauTH%d",1);
     errorCode = ioRegisterVar(&TauScName[0], "float", 4, dims4d, &hydroTauFlds[6*fldStride]);
     printf("hydro_coreInit:hydroTauFlds[6] = %s stored at %p, has been registered with IO.\n",
             &TauScName[0],&hydroTauFlds[6*fldStride]);
     sprintf(&TauScName[0],"TauTH%d",2);
     errorCode = ioRegisterVar(&TauScName[0], "float", 4, dims4d, &hydroTauFlds[7*fldStride]);
     printf("hydro_coreInit:hydroTauFlds[7] = %s stored at %p, has been registered with IO.\n",
             &TauScName[0],&hydroTauFlds[7*fldStride]);
     sprintf(&TauScName[0],"TauTH%d",3);
     errorCode = ioRegisterVar(&TauScName[0], "float", 4, dims4d, &hydroTauFlds[8*fldStride]);
     printf("hydro_coreInit:hydroTauFlds[8] = %s stored at %p, has been registered with IO.\n",
             &TauScName[0],&hydroTauFlds[8*fldStride]);
     fflush(stdout);
   }//end if hydroSubGridWrite
   kappa = 0.4;               /* von Karman constant */
   if(surflayerSelector > 0){ 
     cdFld = memAllocateFloat2DField(Nxp, Nyp, Nh, "cdFld");
     chFld = memAllocateFloat2DField(Nxp, Nyp, Nh, "chFld");
     cqFld = memAllocateFloat2DField(Nxp, Nyp, Nh, "cqFld");
     
     fricVel = memAllocateFloat2DField(Nxp, Nyp, Nh, "fricVel");
     htFlux = memAllocateFloat2DField(Nxp, Nyp, Nh, "htFlux");
     tskin = memAllocateFloat2DField(Nxp, Nyp, Nh, "tskin");
     invOblen = memAllocateFloat2DField(Nxp, Nyp, Nh, "invOblen");
     if (moistureSelector > 0){
       qFlux = memAllocateFloat2DField(Nxp, Nyp, Nh, "qFlux");
       qskin = memAllocateFloat2DField(Nxp, Nyp, Nh, "qskin");
     }
     z0m = memAllocateFloat2DField(Nxp, Nyp, Nh, "z0m");
     z0t = memAllocateFloat2DField(Nxp, Nyp, Nh, "z0t");

     errorCode = sprintf(&fldName[0],"tskin");
     errorCode = ioRegisterVar(&fldName[0], "float", 3, dims2dTD, tskin);
     printf("hydro_coreInit:Field = %s stored at %p, has been registered with IO.\n",
             &fldName[0],tskin);
     fflush(stdout);
     errorCode = sprintf(&fldName[0],"fricVel");
     errorCode = ioRegisterVar(&fldName[0], "float", 3, dims2dTD, fricVel);
     printf("hydro_coreInit:Field = %s stored at %p, has been registered with IO.\n",
             &fldName[0],fricVel);
     fflush(stdout);
     errorCode = sprintf(&fldName[0],"htFlux");
     errorCode = ioRegisterVar(&fldName[0], "float", 3, dims2dTD, htFlux);
     printf("hydro_coreInit:Field = %s stored at %p, has been registered with IO.\n",
             &fldName[0],htFlux);
     fflush(stdout);
     errorCode = sprintf(&fldName[0],"invOblen");
     errorCode = ioRegisterVar(&fldName[0], "float", 3, dims2dTD, invOblen);
     printf("hydro_coreInit:Field = %s stored at %p, has been registered with IO.\n",
             &fldName[0],invOblen);
     fflush(stdout);
     if (moistureSelector > 0){
       errorCode = sprintf(&fldName[0],"qskin");
       errorCode = ioRegisterVar(&fldName[0], "float", 3, dims2dTD, qskin);
       printf("hydro_coreInit:Field = %s stored at %p, has been registered with IO.\n",
               &fldName[0],qskin);
       fflush(stdout);
       errorCode = sprintf(&fldName[0],"qFlux");
       errorCode = ioRegisterVar(&fldName[0], "float", 3, dims2dTD, qFlux);
       printf("hydro_coreInit:Field = %s stored at %p, has been registered with IO.\n",
               &fldName[0],qFlux);
       fflush(stdout);
     }
     errorCode = sprintf(&fldName[0],"z0m");
     errorCode = ioRegisterVar(&fldName[0], "float", 3, dims2dTD, z0m);
     printf("hydro_coreInit:Field = %s stored at %p, has been registered with IO.\n",
             &fldName[0],z0m);
     fflush(stdout);
     errorCode = sprintf(&fldName[0],"z0t");
     errorCode = ioRegisterVar(&fldName[0], "float", 3, dims2dTD, z0t);
     printf("hydro_coreInit:Field = %s stored at %p, has been registered with IO.\n",
             &fldName[0],z0t);
     fflush(stdout);

     MPI_Barrier(MPI_COMM_WORLD);   
     
     /* Provide intial approximation for the momentum and heat exchange coefficient at all surface locations*/
     k = kMin;
     for(i=iMin-Nh; i < iMax+Nh; i++){
       for(j=jMin-Nh; j < jMax+Nh; j++){
         ij = i*(Nyp+2*Nh)+j;
         ijk = i*(Nyp+2*Nh)*(Nzp+2*Nh)+j*(Nzp+2*Nh)+k;
         z0m[ij] = surflayer_z0;
         z0t[ij] = surflayer_z0t;
         z1 = 0.5/(dZi*J33[ijk]);
         z1oz0 = (z1+z0m[ij])/z0m[ij];
         z1ozt0 = (z1+z0t[ij])/z0t[ij];
         cdFld[ij] = pow(kappa,2.0)/pow(log(logf(z1oz0)),2.0);
         chFld[ij] = pow(kappa,2.0)/pow(log(logf(z1ozt0)),2.0);
         cqFld[ij] = chFld[ij];
         if (surflayerSelector == 1){
           htFlux[ij] = surflayer_wth;
           if (moistureSelector > 0){
             qFlux[ij] = surflayer_wq;
           }
         }
         tskin[ij] = temp_grnd/pow((refPressure/pres_grnd),R_cp); // initialize skin temperature to match reference ground potential temperature and pressure
         if (moistureSelector > 0){
           qskin[ij] = 0.0;
         }
       }
     }
     if(surflayerSelector == 3){
       if(hydroBCs != 5){
         printf("\n\n\nERROR: hydro_coreInit: surflayerSelector = 3, but hydroBCs ~= 5... No surfVarBndy planes available for surflayerSelector = 3. \n\n\n\n");
         fflush(stdout);
       }
     }
   } // end of surflayerSelector > 0

   if(surflayer_offshore>0){
     sea_mask = memAllocateFloat2DField(Nxp, Nyp, Nh, "sea_mask");
     errorCode = sprintf(&fldName[0],"SeaMask");
     errorCode = ioRegisterVar(&fldName[0], "float", 3, dims2dTD, sea_mask);
     printf("surflayer_offshore:Field = %s stored at %p, has been registered with IO.\n",
             &fldName[0],sea_mask);
     fflush(stdout);
   }

   if(moistureSelector > 0){ 
     moistScalars = memAllocateFloat4DField(moistureNvars, Nxp, Nyp, Nzp, Nh, "moistScalars");
     moistScalarsFrhs = memAllocateFloat4DField(moistureNvars, Nxp, Nyp, Nzp, Nh, "moistScalarsFrhs");
     for(iFld = 0; iFld < moistureNvars; iFld ++){
        if (iFld==0){
          sprintf(&moistName[0],"qv");
        } else if (iFld==1){
          sprintf(&moistName[0],"ql");
        }
        errorCode = ioRegisterVar(&moistName[0], "float", 4, dims4d, &moistScalars[iFld*fldStride]);
        printf("hydro_coreInit:moistScalars[%d] = %s stored at %p, has been registered with IO.\n",
               iFld,&moistName[0],&moistScalars[iFld*fldStride]);
        fflush(stdout);
     } //end for iFld...
     if(hydroForcingWrite == 1){ // add rhs forcing of SGSTKE equation
       for(iFld = 0; iFld < moistureNvars; iFld ++){
         if (iFld==0){
           sprintf(&moistName[0],"F_qv");
         } else if (iFld==1){
           sprintf(&moistName[0],"F_ql");
         }
         errorCode = ioRegisterVar(&moistName[0], "float", 4, dims4d, &moistScalarsFrhs[iFld*fldStride]);
         printf("hydro_coreInit:moistScalarsFrhs[%d] = %s stored at %p, has been registered with IO.\n",
                iFld,&moistName[0],&moistScalarsFrhs[iFld*fldStride]);
       }
     }

     // writing SGS moisture fields to output netcdf file
     if ((hydroSubGridWrite == 1) && (moistureSGSturb > 0)){
       moistTauFlds = memAllocateFloat4DField(moistureNvars*3, Nxp, Nyp, Nzp, Nh, "moistTauFlds");
       for(iFld = 0; iFld < moistureNvars; iFld ++){
          if (iFld==0){
            sprintf(&moistName_base[0],"TauQv");
          } else if (iFld==1){
            sprintf(&moistName_base[0],"TauQl");
          }
          for(iFld2 = 0; iFld2 < 3; iFld2 ++){ // three spatial directions
             switch (iFld2){
              case 0:
                sprintf(&moistName_tmp[0],"%s",&moistName_base[0]);
                sprintf(&moistName[0],"%d",1);
                strcat(&moistName_tmp[0],&moistName[0]);
                sprintf(&moistName[0],"%s",&moistName_tmp[0]);
                break;
              case 1:
                sprintf(&moistName_tmp[0],"%s",&moistName_base[0]);
                sprintf(&moistName[0],"%d",2);
                strcat(&moistName_tmp[0],&moistName[0]);
                sprintf(&moistName[0],"%s",&moistName_tmp[0]);
                break;
              case 2:
                sprintf(&moistName_tmp[0],"%s",&moistName_base[0]);
                sprintf(&moistName[0],"%d",3);
                strcat(&moistName_tmp[0],&moistName[0]);
                sprintf(&moistName[0],"%s",&moistName_tmp[0]);
                break;
              default:    //invalid iFld value
                printf("hydro_coreInit:moistTauFlds[iFld=%d], invalid value for iFld.\n",iFld*3+iFld2);
                errorCode = -1;
                break;
             }//end switch(iFld)
             errorCode = ioRegisterVar(&moistName[0], "float", 4, dims4d, &moistTauFlds[(iFld*3+iFld2)*fldStride]);
             printf("hydro_coreInit:moistTauFlds[%d] = %s stored at %p, has been registered with IO.\n",
                    iFld*3+iFld2,&moistName[0],&moistTauFlds[(iFld*3+iFld2)*fldStride]);
             fflush(stdout);
          }
       }
     }

   } // end of moistureSelector > 0

   /* Set Constant values */
   accel_g = 9.81;           /* Acceleration of gravity 9.8 m/s^2 */
   R_gas = 287.04;          /* The ideal gas constant in J/(kg*K) */
   R_vapor = 461.60;        /* The ideal gas constant for water vapor in J/(kg*K) */
   cv_gas = 718.0;          /* Specific heat of air at constant volume ****and temperature 300 K in J/(kg*K) */
   cp_gas = R_gas+cv_gas;   /* Specific heat of air at constant pressure ****and temperature 300 K in J/(kg*K) */
   L_v = 2.5e6;             /* latent heat of vaporization (J/kg) */

   R_cp = R_gas/cp_gas;       /* Ratio R/cp*/
   cp_R = cp_gas/R_gas;       /* Ratio cp/R*/
   cp_cv = cp_gas/cv_gas;     /* Ratio cp/cv*/
   refPressure = 1.0e5;       /* Reference pressure set constant to 1e5 Pascals or 1000 millibars) */
   Rv_Rg = R_vapor/R_gas;     /* Ratio R_vapor/R_gas*/

   /* Coriolis-term constants */
   pi = acos(-1);   
   if(coriolisSelector > 0){
     corioConstHorz = 1.45842e-4*sin(pi/180.0*coriolisLatitude); //1.45842e-4 = 2*Earth-Omega
     if(coriolisSelector > 1){  
       corioConstVert = 1.45842e-4*cos(pi/180.0*coriolisLatitude);
     }else{
       corioConstVert = 0.0;
     } //end if vert
     corioLS_fact = 1.0;
   }else{
     corioConstHorz = 0.0;
     corioConstVert = 0.0;
   } //end if coriolisSelector > 0... else

   /* If this is a periodic domain according to hydroBCs set up the rank neighbor topoloogy to be cyclic */
   if(hydroBCs == 0){
     errorCode = fempi_SetupPeriodicDomainDecompositionRankTopology(1, 1); //Periodic in x and y
   }else if (hydroBCs == 2){
     errorCode = fempi_SetupPeriodicDomainDecompositionRankTopology(1, 1); //Periodic in x and y
   }// end if hydroBCS == 0 or 2 setup periodic horizontal neighbor rank topology
  
   return(errorCode);
} //end hydro_coreInit()

/*----->>>>> int hydro_corePrepareFromInitialConditions();   -------------------------------------------------
* Used to undertake the sequence of steps to build the Frhs of all hydro_core prognostic variable fields.
*/
int hydro_corePrepareFromInitialConditions(){
  int errorCode = HYDRO_CORE_SUCCESS;
  
  if(surflayerSelector > 0){  
    ///Perform halo exchange for the 2-d fields associated with hydro_core(surface layer)
    errorCode=fempi_XdirHaloExchange2dXY(Nxp, Nyp, Nh, z0m);
    errorCode=fempi_YdirHaloExchange2dXY(Nxp, Nyp, Nh, z0m);

    errorCode=fempi_XdirHaloExchange2dXY(Nxp, Nyp, Nh, z0t);
    errorCode=fempi_YdirHaloExchange2dXY(Nxp, Nyp, Nh, z0t);

    errorCode=fempi_XdirHaloExchange2dXY(Nxp, Nyp, Nh, htFlux);
    errorCode=fempi_YdirHaloExchange2dXY(Nxp, Nyp, Nh, htFlux);


    errorCode=fempi_XdirHaloExchange2dXY(Nxp, Nyp, Nh, fricVel);
    errorCode=fempi_YdirHaloExchange2dXY(Nxp, Nyp, Nh, fricVel);

    errorCode=fempi_XdirHaloExchange2dXY(Nxp, Nyp, Nh, tskin);
    errorCode=fempi_YdirHaloExchange2dXY(Nxp, Nyp, Nh, tskin);

    if(moistureSelector > 0){  
      errorCode=fempi_XdirHaloExchange2dXY(Nxp, Nyp, Nh, qFlux);
      errorCode=fempi_YdirHaloExchange2dXY(Nxp, Nyp, Nh, qFlux);
    
      errorCode=fempi_XdirHaloExchange2dXY(Nxp, Nyp, Nh, qskin);
      errorCode=fempi_YdirHaloExchange2dXY(Nxp, Nyp, Nh, qskin);
    } //end if moistureSelector >0

    if(surflayer_offshore > 0){
     errorCode=fempi_XdirHaloExchange2dXY(Nxp, Nyp, Nh, sea_mask);
     errorCode=fempi_YdirHaloExchange2dXY(Nxp, Nyp, Nh, sea_mask);
    }

  } //end if surflayerSelector >0 

  return(errorCode);
} //end hydro_corePrepareFromInitialConditions()

/*----->>>>> int hydro_coreGetFieldName();   ----------------------------------------------------------------------
* Used to fill a caller-allocated character array with the i^(th) field name in the hydoFlds memory block .
*/
int hydro_coreGetFieldName(char * fldName, int iFld){
   int errorCode = HYDRO_CORE_SUCCESS;

   /*fill a caller-allocated char array (buffer) with the requested fieldName*/ 
   switch(iFld){
      case 0:    //Total air density
          errorCode = sprintf(fldName,"rho");
          break;
      case 1:    //Zonal (x-direction) velocity component 
          errorCode = sprintf(fldName,"u");
          break;
      case 2:    //Meridional (y-direction) velocity component 
          errorCode = sprintf(fldName,"v");
          break;
      case 3:    //Vertical (z-direction) velocity component 
          errorCode = sprintf(fldName,"w");
          break;
      case 4:    //perturbation potential temperature 
          errorCode = sprintf(fldName,"theta");
          break;
      case 5:    //arbitrary field
          errorCode = sprintf(fldName,"phi");
          break;
      default:    //invalid iFld value
          errorCode = -1;
          break;
   }// end switch iFld

   return(errorCode);
} //end hydro_coreGetFieldName()

/*----->>>>> int hydro_coreSetBaseState();   -----------------------------------------------------------
 * Used to set the Base-State fields for all prognostic variables and pressure.
*/
int hydro_coreSetBaseState(){
   int errorCode = HYDRO_CORE_SUCCESS;
   int i,j,k,ijk;
   int iFld;
   int fldStride;
   float* rhoBase;
   float* uIni;
   float* vIni;
   float* wIni;
   float* thetaBase;
   float constant_1;
   float BS_Temp;
   float *fldBase;
   float *fldBaseBS;

   fldStride = (Nxp+2*Nh)*(Nyp+2*Nh)*(Nzp+2*Nh);
   constant_1 = R_gas/(pow(refPressure,R_cp)); 
   /* Stability Regimes and Equation of State variables (rho, theta, pressure)*/
   /* ----From temp_grnd and pres_grnd, establish rho_grnd and theta_grnd*/
   rho_grnd = pres_grnd/(R_gas*temp_grnd);
   theta_grnd = temp_grnd*pow(pres_grnd/refPressure,-R_cp);
   rhoBase = &hydroBaseStateFlds[RHO_INDX_BS*fldStride];
   thetaBase = &hydroBaseStateFlds[THETA_INDX_BS*fldStride];
   /* ----Based on stabilityScheme setup Base-State rho,theta, and pressure profiles */ 
   if(stabilityScheme == 0){   /* None, constant density, theta (potential temperature), and pressure fields */
     for(i=iMin-Nh; i < iMax+Nh; i++){       // Cover the halos in X 
       for(j=jMin-Nh; j < jMax+Nh; j++){     // Cover the halos in Y 
         for(k=kMin-Nh; k < kMax+Nh; k++){   // Cover the halos in Z 
           ijk = i*(Nyp+2*Nh)*(Nzp+2*Nh)+j*(Nzp+2*Nh)+k;
           rhoBase[ijk] = rho_grnd;
           thetaBase[ijk] = rho_grnd*theta_grnd;
           hydroBaseStatePres[ijk] = pow(thetaBase[ijk]*constant_1,cp_cv);
         } //end for(k...
       } // end for(j...
     } // end for(i...
     printf("stabilityScheme == 0: Base State setup complete.\n");
   }else if(stabilityScheme == 1){  /* stable linear potential temperature profile above some height zStableBottom, 
                                       neutral below zStableBottom*/
     for(i=iMin-Nh; i < iMax+Nh; i++){       // Cover the halos in X 
       for(j=jMin-Nh; j < jMax+Nh; j++){     // Cover the halos in Y 
         for(k=kMin-Nh; k < kMax+Nh; k++){   // Cover the halos in Z 
           ijk = i*(Nyp+2*Nh)*(Nzp+2*Nh)+j*(Nzp+2*Nh)+k;
           if(zPos[ijk] <= zStableBottom){ //This point is within the neutral lower-layer
             thetaBase[ijk] = theta_grnd;
             hydroBaseStatePres[ijk] = refPressure*pow( (-accel_g/cp_gas)*( zPos[ijk]/theta_grnd )
                                                        +pow(pres_grnd/refPressure,R_cp)  //base of the first pow (...)
                                                        ,cp_R);  //exponent of the first pow(...)
           }else{ //This point is within the stable upper-layer
             //Set theta
             thetaBase[ijk] = theta_grnd + stableGradient*(zPos[ijk]-zStableBottom);
             //set base state  pressure
             hydroBaseStatePres[ijk] = refPressure*pow( (-accel_g/cp_gas)*( zStableBottom/theta_grnd 
                                                                     +(1.0/stableGradient)*log(1.0+stableGradient*(zPos[ijk]-zStableBottom)/theta_grnd))
                                                        +pow(pres_grnd/refPressure,R_cp)  //base of the first pow (...)
                                                        ,cp_R);  //exponent of the first pow(...)
           } //end zPos[ijk >= zStableBottom
           //back out base state air temperature
           BS_Temp = thetaBase[ijk]*pow( hydroBaseStatePres[ijk]/refPressure,R_cp);
           //back out base state density
           rhoBase[ijk] = hydroBaseStatePres[ijk]/(BS_Temp*R_gas);
           //Given this density set the flux form of the potential temperature prognostic field (rho*theta)
           thetaBase[ijk] = thetaBase[ijk]*rhoBase[ijk];           
           //Finally recast the base state pressure in a "discretisation-consistent" manner
           hydroBaseStatePres[ijk] = pow(thetaBase[ijk]*constant_1, cp_cv); //This minimizes round off under the pressure formulation in calcPerturbationPRessure()
         } //end for(k...
       } // end for(j...
     } // end for(i...
     printf("stabilityScheme == 1: Base State setup complete.\n");
   }else if(stabilityScheme == 2){ 
     for(i=iMin-Nh; i < iMax+Nh; i++){       // Cover the halos in X 
       for(j=jMin-Nh; j < jMax+Nh; j++){     // Cover the halos in Y 
         for(k=kMin-Nh; k < kMax+Nh; k++){   // Cover the halos in Z 
           ijk = i*(Nyp+2*Nh)*(Nzp+2*Nh)+j*(Nzp+2*Nh)+k;
           if(zPos[ijk] <= zStableBottom){ //This point is within the neutral lower-layer
             thetaBase[ijk] = theta_grnd;
             hydroBaseStatePres[ijk] = refPressure*pow( (-accel_g/cp_gas)*( zPos[ijk]/theta_grnd )
                                                        +pow(pres_grnd/refPressure,R_cp)  //base of the first pow (...)
                                                        ,cp_R);  //exponent of the first pow(...)
           }else if((zStableBottom < zPos[ijk])&&((zPos[ijk] <= zStableBottom2))){ //This point is within the first stable upper-layer
             //Set theta
             thetaBase[ijk] = theta_grnd + stableGradient*(zPos[ijk]-zStableBottom);
             //set base state  pressure
             hydroBaseStatePres[ijk] = refPressure*pow( (-accel_g/cp_gas)*( zStableBottom/theta_grnd 
                                                                     +(1.0/stableGradient)*log(1.0+stableGradient*(zPos[ijk]-zStableBottom)/theta_grnd))
                                                        +pow(pres_grnd/refPressure,R_cp)  //base of the first pow (...)
                                                        ,cp_R);  //exponent of the first pow(...)
           }else if((zStableBottom2 < zPos[ijk])&&((zPos[ijk] <= zStableBottom3))){ //This point is within the third stable upper-layer
             //Set theta
             thetaBase[ijk] = theta_grnd + stableGradient*(zStableBottom2-zStableBottom) + stableGradient2*(zPos[ijk]-zStableBottom2);
             //set base state  pressure
             hydroBaseStatePres[ijk] = refPressure*pow( (-accel_g/cp_gas)*( zStableBottom/theta_grnd 
                                                                     +(1.0/stableGradient)*log(1.0+stableGradient*(zStableBottom2-zStableBottom)/theta_grnd)
                                                                     +(1.0/stableGradient2)*log(1.0+stableGradient2*(zPos[ijk]-zStableBottom2)/(theta_grnd+stableGradient*(zStableBottom2-zStableBottom))))
                                                        +pow(pres_grnd/refPressure,R_cp)  //base of the first pow (...)
                                                        ,cp_R);  //exponent of the first pow(...)
           }else{ //This point is within the second stable upper-layer
             //Set theta
             thetaBase[ijk] = theta_grnd + stableGradient*(zStableBottom2-zStableBottom) + stableGradient2*(zStableBottom3-zStableBottom2)+ stableGradient3*(zPos[ijk]-zStableBottom3);
             //set base state  pressure
             hydroBaseStatePres[ijk] = refPressure*pow( (-accel_g/cp_gas)*( zStableBottom/theta_grnd 
                                                                     +(1.0/stableGradient)*log(1.0+stableGradient*(zStableBottom2-zStableBottom)/theta_grnd)
                                                                     +(1.0/stableGradient2)*log(1.0+stableGradient2*(zStableBottom3-zStableBottom2)/(theta_grnd+stableGradient*(zStableBottom2-zStableBottom)))
                                                                     +(1.0/stableGradient3)*log(1.0+stableGradient3*(zPos[ijk]-zStableBottom3)/(theta_grnd+stableGradient*(zStableBottom2-zStableBottom)+stableGradient2*(zStableBottom3-zStableBottom2))))
                                                        +pow(pres_grnd/refPressure,R_cp)  //base of the first pow (...)
                                                        ,cp_R);  //exponent of the first pow(...)
           } //end if zPos[ijk  < zStableBottom...
           //back out base state air temperature
           BS_Temp = thetaBase[ijk]*pow( hydroBaseStatePres[ijk]/refPressure,R_cp);
           //back out base state density
           rhoBase[ijk] = hydroBaseStatePres[ijk]/(BS_Temp*R_gas);
           //Given this density set the flux form of the potential temperature prognostic field (rho*theta)
           thetaBase[ijk] = thetaBase[ijk]*rhoBase[ijk];           
           //Finally recast the base state pressure in a "discretisation-consistent" manner
           hydroBaseStatePres[ijk] = pow(thetaBase[ijk]*constant_1, cp_cv); //This minimizes round off under the pressure formulation in calcPerturbationPRessure()
         } //end for(k...
       } // end for(j...
     } // end for(i...
     printf("stabilityScheme == 2: Base State setup complete.\n");
   }else if(stabilityScheme == 3){ 
     printf("stabilityScheme == 3: Using initial Conditions as BaseState if hydroBCs != 4!! \n");
   }else if(stabilityScheme == 4){ /*Experimental setup for constant rho and constant theta profiles.
                                     Use only for total domain vertical extent < 10m. */
      rho_grnd = 1.1;
      for(i=iMin-Nh; i < iMax+Nh; i++){       // Cover the halos in X 
       for(j=jMin-Nh; j < jMax+Nh; j++){     // Cover the halos in Y 
         for(k=kMin-Nh; k < kMax+Nh; k++){   // Cover the halos in Z 
           ijk = i*(Nyp+2*Nh)*(Nzp+2*Nh)+j*(Nzp+2*Nh)+k;
           if(zPos[ijk] <= 0.5){
             rhoBase[ijk] = rho_grnd;
           }else{
             rhoBase[ijk] = rho_grnd;
           }
           thetaBase[ijk] = rho_grnd*theta_grnd;
           hydroBaseStatePres[ijk] = pow(thetaBase[ijk]*constant_1,cp_cv);
         } //end for(k...
       } // end for(j...
     } // end for(i...
   } //end if-else... stabilityScheme...

   if(inFile == NULL){
      /* Prescribe a geostrophic wind profile as initial conditions on U & V with W = 0 */
      uIni = &hydroFlds[U_INDX*fldStride];
      vIni = &hydroFlds[V_INDX*fldStride];
      wIni = &hydroFlds[W_INDX*fldStride];
      for(i=iMin-Nh; i < iMax+Nh; i++){       // Cover the halos in X 
        for(j=jMin-Nh; j < jMax+Nh; j++){     // Cover the halos in Y 
          for(k=kMin; k < kMax+Nh; k++){   // Cover the halos in Z 
            ijk = i*(Nyp+2*Nh)*(Nzp+2*Nh)+j*(Nzp+2*Nh)+k;
            if (zPos[ijk] < z_Ug){ 
              uIni[ijk] = U_g*rhoBase[ijk];
            } else{
              uIni[ijk] = (U_g+Ug_grad*(zPos[ijk]-z_Ug))*rhoBase[ijk];
            }
            if (zPos[ijk] < z_Vg){
              vIni[ijk] = V_g*rhoBase[ijk];
            } else{
              vIni[ijk] = (V_g+Vg_grad*(zPos[ijk]-z_Vg))*rhoBase[ijk];
            }
            wIni[ijk] = 0.0;
          } //end for(k...
        } // end for(j...
     } // end for(i...
     /*Set initial conditions  on rho and Theta to match base state */
     for(iFld=0; iFld < 2; iFld++){
       switch (iFld){
         case 0:
           fldBase = &hydroFlds[RHO_INDX*fldStride];
           fldBaseBS = &hydroBaseStateFlds[RHO_INDX_BS*fldStride];
           break;
         case 1:
           fldBase = &hydroFlds[THETA_INDX*fldStride];
           fldBaseBS = &hydroBaseStateFlds[THETA_INDX_BS*fldStride];
           break;
       }
       for(i=iMin-Nh; i < iMax+Nh; i++){       // Cover the halos in X 
         for(j=jMin-Nh; j < jMax+Nh; j++){     // Cover the halos in Y 
           for(k=kMin-Nh; k < kMax+Nh; k++){   // Cover the halos in Z 
             ijk = i*(Nyp+2*Nh)*(Nzp+2*Nh)+j*(Nzp+2*Nh)+k;
             fldBase[ijk] = fldBaseBS[ijk];
           } //end for(k...
         } // end for(j...
       } // end for(i...
     }//end if-else iFld==0   
// Introduce theta perturbation to accelerate spinup?
     if(thetaPerturbationSwitch == 1){
       rhoBase = &hydroFlds[RHO_INDX*fldStride];
       fldBase = &hydroFlds[THETA_INDX*fldStride];
       for(i=iMin; i < iMax; i++){
         for(j=jMin; j < jMax; j++){
           for(k=kMin; k < kMax; k++){
             ijk = i*(Nyp+2*Nh)*(Nzp+2*Nh)+j*(Nzp+2*Nh)+k;
             if(zPos[ijk]<=thetaHeight){
               fldBase[ijk] = (fldBase[ijk]/rhoBase[ijk]+2.0*thetaAmplitude*(((float)rand()/(float)(RAND_MAX))-0.5))*rhoBase[ijk]; 
             }
           } //end for(k...
         } // end for(j...
       } // end for(i...
     } //endif thetaPerturbationSwitch==1

   }else{ //Initial conditions were provided...
     if(stabilityScheme==3){
      for(iFld=0; iFld < 2; iFld++){
         switch (iFld){
           case 0:
             fldBase = &hydroFlds[RHO_INDX*fldStride];
             fldBaseBS = &hydroBaseStateFlds[RHO_INDX_BS*fldStride];
             break;
           case 1:
             fldBase = &hydroFlds[THETA_INDX*fldStride];
             fldBaseBS = &hydroBaseStateFlds[THETA_INDX_BS*fldStride];
             break;
         }
         for(i=iMin-Nh; i < iMax+Nh; i++){       // Cover the halos in X 
           for(j=jMin-Nh; j < jMax+Nh; j++){     // Cover the halos in Y 
             for(k=kMin-Nh; k < kMax+Nh; k++){   // Cover the halos in Z 
               ijk = i*(Nyp+2*Nh)*(Nzp+2*Nh)+j*(Nzp+2*Nh)+k;
               fldBaseBS[ijk] = fldBase[ijk];
             } //end for(k...
           } // end for(j...
         } // end for(i...
       }//end if-else iFld==0    
     }//end if stabilityScheme==3
   }//If no initial conditions were specified
   return(errorCode);
}// end coreSetBaseState

/*----->>>>> int hydro_coreStateLogDump();  -------------------------------------------------------
* Utility function to produce field state summaries for a desired set of hydro_core fields. 
*/
int hydro_coreStateLogDump(){
   int errorCode = HYDRO_CORE_SUCCESS;
   int fldStride;
   int iFld;
   float* fldBase;
   char fldName[MAX_HC_FLDNAME_LENGTH];
   char frhsName[MAX_HC_FLDNAME_LENGTH+2];

   /* setup fldStride */
   fldStride = (Nxp+2*Nh)*(Nyp+2*Nh)*(Nzp+2*Nh);
   MPI_Barrier(MPI_COMM_WORLD);
   fflush(stdout);
   MPI_Barrier(MPI_COMM_WORLD);
   if(mpi_rank_world == 0){
     printf("Field \t|\t max\t     --(i,j,k)\t\t |\t  min\t     --(i,j,k)\t\t |\t mean\n");
     printf("---------------------------------------------------------------------------------------------------------\n");
     printf("********-----\n Model Fields -\n ********-----\n");
     fflush(stdout);
   } //end if mpi_rank_world==0
   MPI_Barrier(MPI_COMM_WORLD);
   /*Log each prognostic variable field's summry metrics*/
   for(iFld=0; iFld < Nhydro; iFld++){        //Run through (hydroFlds)
     MPI_Barrier(MPI_COMM_WORLD);
     if((mpi_rank_world == 0)&&(iFld>0)){
       printf("-----\n");
       fflush(stdout);
     } //end if mpi_rank_world==0
     fldBase = &hydroFlds[iFld*fldStride];
     errorCode = hydro_coreGetFieldName( &fldName[0], iFld);
     printf("%s\t", &fldName[0]);
     if(iFld == RHO_INDX){
       errorCode = hydro_coreFldStateLogEntry(fldBase, 0); 
     }else{
       errorCode = hydro_coreFldStateLogEntry(fldBase, 1); 
     } //end if-else iFld = RHO_INDX
     MPI_Barrier(MPI_COMM_WORLD);
   } //end for iFld
   if((mpi_rank_world == 0)&&(iFld>0)){
     printf("-----\n");
     fflush(stdout);
   } //end if mpi_rank_world==0
   fldBase = hydroPres;
   errorCode = sprintf(&fldName[0],"press");
   printf("%s\t", &fldName[0]);
   errorCode = hydro_coreFldStateLogEntry(fldBase, 0); 
   // SGSTKE equation 
   for(iFld=0; iFld < TKESelector; iFld++){    //Run through (sgstkeScalars)
     MPI_Barrier(MPI_COMM_WORLD);
     if((mpi_rank_world == 0)&&(iFld>=0)){
       printf("-----\n");
       fflush(stdout);
     } //end if mpi_rank_world==0
     fldBase = &sgstkeScalars[iFld*fldStride];
     errorCode = sprintf(&fldName[0],"TKE_%d",iFld);
     printf("%s\t", &fldName[0]);
     errorCode = hydro_coreFldStateLogEntry(fldBase, 1); 
   } //end for iFld
   // moisture 
   if (moistureSelector > 0){
     for(iFld=0; iFld < moistureNvars; iFld++){    //Run through (sgstkeScalars)
       MPI_Barrier(MPI_COMM_WORLD);
       if((mpi_rank_world == 0)&&(iFld>=0)){
         printf("-----\n");
         fflush(stdout);
       } //end if mpi_rank_world==0
       fldBase = &moistScalars[iFld*fldStride];
       if (iFld==0){
         errorCode = sprintf(&fldName[0],"qv");
       } else if(iFld==1){
         errorCode = sprintf(&fldName[0],"ql");
       } else if(iFld==2){
         errorCode = sprintf(&fldName[0],"qr");
       }
       printf("%s\t", &fldName[0]);
       errorCode = hydro_coreFldStateLogEntry(fldBase, 1);
     } //end for iFld
   }
    
   /* Log the Frhs fields */
   if(hydroForcingLog == 1){
     MPI_Barrier(MPI_COMM_WORLD);
     if(mpi_rank_world == 0){
       printf("********-----\n Frhs Fields --\n ********-----\n");
       fflush(stdout);
     } //end if mpi_rank_world==0
     for(iFld=0; iFld < Nhydro; iFld++){        //Run through (hydroFldsFrhs) 
       MPI_Barrier(MPI_COMM_WORLD);
       if((mpi_rank_world == 0)&&(iFld>0)){
         printf("-----\n");
         fflush(stdout);
       } //end if mpi_rank_world==0
       fldBase = &hydroFldsFrhs[iFld*fldStride];
       errorCode = hydro_coreGetFieldName( &fldName[0], iFld);
       sprintf(&frhsName[0],"F_%s",&fldName[0]);
       printf("%s\t", &frhsName[0]);
       errorCode = hydro_coreFldStateLogEntry(fldBase, 0); 
       MPI_Barrier(MPI_COMM_WORLD);
     }// for iFld
     for(iFld=0; iFld < TKESelector; iFld++){  //Run through (sgstkeScalarsFrhs) 
       MPI_Barrier(MPI_COMM_WORLD);
       if((mpi_rank_world == 0)&&(iFld>=0)){
         printf("-----\n");
         fflush(stdout);
       } //end if mpi_rank_world==0
       fldBase = &sgstkeScalarsFrhs[iFld*fldStride];
       errorCode = sprintf(&fldName[0],"F_TKE%d",iFld);
       printf("%s\t", &fldName[0]);
       errorCode = hydro_coreFldStateLogEntry(fldBase, 1); 
       MPI_Barrier(MPI_COMM_WORLD);
     } //end for iFld
     // moisture 
     if (moistureSelector > 0){
       for(iFld=0; iFld < moistureNvars; iFld++){    //Run through (sgstkeScalars)
         MPI_Barrier(MPI_COMM_WORLD);
         if((mpi_rank_world == 0)&&(iFld>=0)){
           printf("-----\n");
           fflush(stdout);
         } //end if mpi_rank_world==0
         fldBase = &moistScalarsFrhs[iFld*fldStride];
         if (iFld==0){
           errorCode = sprintf(&fldName[0],"F_qv");
         } else if(iFld==1){
           errorCode = sprintf(&fldName[0],"F_ql");
         } else if(iFld==2){
           errorCode = sprintf(&fldName[0],"F_qr");
         }
         printf("%s\t", &fldName[0]);
         errorCode = hydro_coreFldStateLogEntry(fldBase, 1);
       } //end for iFld
     }
   }//end if (hydroForcingLog == 1)

   MPI_Barrier(MPI_COMM_WORLD); 
   return(errorCode);
}//end hydro_coreStateLogDump()

/*----->>>>> int hydro_coreFldStateLogEntry(float * Fld);  ----------------------------------------------------------------------
 * Utility function to log an arbitrary hydro_core field state summary. 
 * e.g. [ max(loc) min(loc) mean, variance, isNan(loc), is Inf(loc) ]
 */
int hydro_coreFldStateLogEntry(float * Fld, int fluxConservativeFlag){
   int errorCode = HYDRO_CORE_SUCCESS;
   int i,j,k,ijk,iRank;
   float* rho;
   int fldStride;
   int fldCount = 0;
   int nanCount = 0;
   int infCount = 0;
   int maxLoc[3] = {0,0,0};
   int minLoc[3] = {0,0,0};
   float maxValue = -1.0e16;
   float minValue =  1.0e16;
   float meanValue;
   float rhoInv;
   int iStride,jStride,kStride;

   iStride = (Nyp+2*Nh)*(Nzp+2*Nh);
   jStride = (Nzp+2*Nh);
   kStride = 1;


   fldStride = (Nxp+2*Nh)*(Nyp+2*Nh)*(Nzp+2*Nh);
   rho = &hydroFlds[RHO_INDX*fldStride];
   meanValue = 0.0;
   /*Loop over the non-Halo, domain extents*/
   for(i=iMin; i < iMax; i++){
    for(j=jMin; j < jMax; j++){
      for(k=kMin; k < kMax; k++){
        ijk = i*iStride + j*jStride + k*kStride;
        if(fluxConservativeFlag == 1){ // fluxConservativeFlag == 1 (true)
          rhoInv = 1.0/rho[ijk];
        }else{ 
          rhoInv = 1.0;
        }// end if-else fluxConservativeFlag == 1 

        /* Perform the state-summary calculations -----------------------*/
        /*************************** Check this element for nan/inf state *****************************/
        if(!isfinite(Fld[ijk]*rhoInv)){ /********************  HOUSTON WE HAVE A PROBLEM!!! ******************/
          if(isnan(Fld[ijk]*rhoInv)){      /* Is it a nan? */
             nanCount = nanCount + 1;
          }else if(isinf(Fld[ijk]*rhoInv)){
             infCount = infCount + 1;
          }//end if(isnan)...else if(isinf)...[TODO ?: else if(isnormal) and/or if(isunordered)] 
        }else{     /********************  Fld[ijk] = GOOD !!! ******************/
          /*###### State Summary Metrics ########*/
          if(maxValue < Fld[ijk]*rhoInv){ // new maxValue
             maxValue =  Fld[ijk]*rhoInv;
             maxLoc[0] = i;
             maxLoc[1] = j;
             maxLoc[2] = k;
          } //endif new maxValue
          if(minValue > Fld[ijk]*rhoInv){ // new minValue
             minValue =  Fld[ijk]*rhoInv;
             minLoc[0] = i;
             minLoc[1] = j;
             minLoc[2] = k;
          } //endif new minValue
          /*Incorporate the current element into summary-statistics metrics*/
          meanValue = meanValue + Fld[ijk]*rhoInv;
        } //end if-else (!isfinite(Fld[ijk]) 
        fldCount = fldCount + 1;
      }//end for k=kMax...
    }//end for j=jMin...
  }//end for i=iMin... 
  meanValue = meanValue/((float)fldCount);

  /*TODO: Find the Variance ( and maybe the median while we are at it? )
  float varianceValue;
  for(i=iMin; i < iMax; i++){
    for(j=jMin; j < jMax; j++){
      for(k=kMin; k < kMin; k++){
          varianceValue = varianceValue + (Fld[ijk]/rhoValue - meanValue)*(Fld[ijk]/rhoValue - meanValue);
      }//end for k=kMax...
    }//end for j=jMin...
  }//end for i=iMin...
  varianceValue = varianceValue/((float)fldCount-1);
  */

  /* TODO: Run an mpi collective operation for each of the metrics*/

 
  /* Write the state-summary information for this fiels to the output log*/ 
  
  //All ranks write a line:--if(mpi_rank_world == 0){
  for(iRank=0; iRank < mpi_size_world; iRank++){ 
    MPI_Barrier(MPI_COMM_WORLD); 
    if(mpi_rank_world == iRank){
      if((nanCount==0)&&(infCount==0)){
       printf("Rank %d/%d: %16.8f \t (%d,%d,%d)\t |  %16.8f \t (%d,%d,%d)\t | %16.8f \n",
                   mpi_rank_world, mpi_size_world,
                   maxValue,maxLoc[0],maxLoc[1],maxLoc[2], 
                   minValue,minLoc[0],minLoc[1],minLoc[2],
                   meanValue);
       fflush(stdout);
      }else{
       printf("Rank %d/%d: ****CORRUPTED*** --- (#NaN, #Inf)/ [#cells] = (%d, %d)/[%d]\n",
                                 mpi_rank_world, mpi_size_world,nanCount,infCount,fldCount);
       fflush(stdout);
      }//if nan or inf were found 
    } //end if(mpi_rank_world == iRank) 
    MPI_Barrier(MPI_COMM_WORLD); 
  }//end for iRank
  return(errorCode);
}//end hydro_coreFldStateLogEntry()
/*----->>>>> int hydro_coreCleanup();  ----------------------------------------------------------------------
* Used to free all malloced memory by the HYDRO_CORE module.
*/
int hydro_coreCleanup(){
   int errorCode = HYDRO_CORE_SUCCESS;

   /* Free any HYDRO_CORE module arrays */
   memReleaseFloat(hydroFlds);
   memReleaseFloat(hydroFldsFrhs);
   memReleaseFloat(hydroFaceVels);
   memReleaseFloat(hydroRhoInv);
   memReleaseFloat(hydroBaseStateFlds);
   memReleaseFloat(hydroPres);
   memReleaseFloat(hydroBaseStatePres);
   if(diffusionSelector > 0){
     memReleaseFloat(hydroDiffTauXFlds);
     memReleaseFloat(hydroDiffTauYFlds);
     memReleaseFloat(hydroDiffTauZFlds);
   }
   if(turbulenceSelector > 0){
     memReleaseFloat(hydroDiffNuFld);
     memReleaseFloat(hydroTauFlds);
     if (TKESelector > 0){ 
       memReleaseFloat(sgstkeScalars);
       memReleaseFloat(sgstkeScalarsFrhs);
     }
   } 
   if (surflayerSelector > 0) { 
     memReleaseFloat(cdFld);
     memReleaseFloat(chFld);
     memReleaseFloat(fricVel);
     memReleaseFloat(htFlux);
     memReleaseFloat(tskin);
     memReleaseFloat(invOblen);
     if (moistureSelector > 0){
       memReleaseFloat(qFlux);
       memReleaseFloat(qskin);
     }
     memReleaseFloat(z0m);
     memReleaseFloat(z0t);
     if (surflayer_offshore > 0){
       memReleaseFloat(sea_mask);
     }
   }//end if surface selector > 0
   if(moistureSelector > 0){
     memReleaseFloat(moistScalars);
     memReleaseFloat(moistScalarsFrhs);
   }
   return(errorCode);

}//end hydro_coreCleanup()
