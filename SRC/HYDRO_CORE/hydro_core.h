/* FastEddy®: SRC/HYDRO_CORE/hydro_core.h 
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
#ifndef _HYDRO_CORE_H
#define _HYDRO_CORE_H

/*hydro_core_ return codes */
#define HYDRO_CORE_SUCCESS    0

#define RHO_INDX              0
#define U_INDX                1
#define V_INDX                2
#define W_INDX                3
#define THETA_INDX            4

#define RHO_INDX_BS           0
#define THETA_INDX_BS         1

#define MAX_HC_FLDNAME_LENGTH 256

/*#################------------------- HYDRO_CORE module variable declarations ---------------------#################*/
/* Parameters */
extern int Nhydro;          /*Number of prognostic variable fields under hydro_core */
extern int hydroBCs;          /*selector for hydro BC set. 2= periodicHorizVerticalAbl */

extern int hydroForcingWrite;   /*switch for dumping forcing fields of prognostic variables. 0-off (default), 1= on*/
extern int hydroForcingLog;     /*switch for logging Frhs summary metrics. 0-off (default), 1= on*/
extern int hydroSubGridWrite;   /*switch for SGS fields 0-off (default), 1= on*/
extern float *hydroFlds;        /*Base Adress of memory containing all prognostic variable fields under hydro_core */
extern float *hydroFldsFrhs;    /*Base Adress of memory containing all prognostic variable Frhs  under hydro_core */
extern float *hydroFaceVels;    /*cell face velocities*/
extern float *hydroRhoInv;      /*Inverse of the air density (1.0/rho)*/
extern float *hydroBaseStateFlds; /*Base Adress of memory containing field base-states for rho and theta */
extern float *hydroPres;          /*Base Adress of memory containing the diagnostic perturbation pressure field */
extern float *hydroBaseStatePres; /*Base Adress of memory containing the diagnostic base-state pressure field */ 

/*Constants*/ 
extern float R_gas;             /* The ideal gas constant in J/(mol*K) */
extern float R_vapor;           /* The ideal gas constant for water vapor in J/(mol*K) */
extern float Rv_Rg;             /* Ratio R_vapor/R_gas*/
extern float cp_gas;            /* Specific heat of air at constant pressure */ 
extern float cv_gas;            /* Specific heat of air at constant pressure */
extern float accel_g;           /* Acceleration of gravity 9.8 m/(s^2) */
extern float R_cp;              /* Ratio R/cp*/
extern float cp_R;              /* Ratio cp/R*/
extern float cp_cv;             /* Ratio cp/cv*/
extern float refPressure;       /* Reference pressure set constant to 1e5 Pascals or 1000 millibars) */
extern float kappa;             /* von Karman constant */
extern float L_v;               /* latent heat of vaporization (J/kg) */


/*HYDRO_CORE Submodule parameters*/ 

/*---PRESSURE_GRADIENT_FORCE*/
extern int pgfSelector;          /*Pressure Gradient Force (pgf) selector: 0=off, 1=on*/
/*---BUOYANCY*/
extern int buoyancySelector;     /*buoyancy Force selector: 0=off, 1=on*/
/*---CORIOLIS*/
extern int coriolisSelector;     /*coriolis Force selector: 0= none, 1= Horiz.-only, 2=Horz. & Vert.*/
extern float coriolisLatitude;   /*Charactersitc latitude in degrees from equator of the LES domain*/
extern float corioConstHorz;     /*Latitude dependent horizontal Coriolis term constant */
extern float corioConstVert;     /*Latitude dependent Vertical Coriolis term constant */
extern float corioLS_fact;       /*large-scale factor on Coriolis term*/
/*---TURBULENCE*/
extern int turbulenceSelector;    /*turbulence scheme selector: 0= none, 1= Lilly/Smagorinsky */
extern int TKESelector;           /* Prognostic TKE selector: 0= none, 1= Prognostic */
extern int TKEAdvSelector;        /* SGSTKE advection scheme selector */
extern float TKEAdvSelector_b_hyb;     /*hybrid advection scheme parameter */
extern float c_s;     /* Smagorinsky turbulence model constant used for turbulenceSelector = 1 with TKESelector = 0 */
extern float c_k;    /* Lilly turbulence model constant used for turbulenceSelector = 1 with TKESelector > 0 */
extern float *sgstkeScalars;  /* Base Adress of memory containing all prognostic "sgstke" variable fields */ 
extern float *sgstkeScalarsFrhs; /* Base Adress of memory containing all prognostic "sgstke" RHS forcing fields */ 
/*---DIFFUSION*/
extern int diffusionSelector;  /*diffusion Term-type selector: 0= none, 1= constant, 2= scalar turbulent-diffusivity*/
extern float nu_0;            /* constant diffusivity used when diffusionSelector = 1 */
extern float *hydroTauFlds;      /*Base address for scratch/work Tau tensor array*/
extern float *hydroDiffTauXFlds; /*Base address for diffusion TauX arrays for all prognostic fields*/
extern float *hydroDiffTauYFlds; /*Base address for diffusion TauY arrays for all prognostic fields*/
extern float *hydroDiffTauZFlds; /*Base address for diffusion TauZ arrays for all prognostic fields*/
/*---ADVECTION*/
extern int advectionSelector;    /*advection scheme selector: 0= 1st-order upwind, 2= 3rd-order QUICK */
extern float b_hyb; /*hybrid advection scheme parameter: 0.0= lower-order upwind, 
                             1.0=higher-order cetered, 0.0 < b_hyb < 1.0 = hybrid */
/*---SURFACE LAYER*/
extern int surflayerSelector;    /*Monin-Obukhov surface layer selector: 0= off, 1= on */
extern float surflayer_z0;       /* roughness length (momentum) */
extern float surflayer_z0t;      /* roughness length (temperature) */
extern float surflayer_wth;      /* kinematic sensible heat flux at the surface */
extern float surflayer_tr;       /* surface temperature rate K h-1 */
extern float surflayer_wq;       /* kinematic latent heat flux at the surface */
extern float surflayer_qr;       /* surface water vapor rate (g/kg) h-1 */
extern int surflayer_qskin_input;/* selector for file input (restart) value for qskin under surflayerSelector == 2 */
extern int surflayer_stab;       /* exchange coeffcient stability correction selector: 0= on, 1= off */
extern float *fricVel;           /*Surface friction velocity "u*" 2-d array (x by y) (ms^-1)*/
extern float *htFlux;            /*Surface heat flux "(w'th')" 2-d array (x by y) (Kms^-1)*/
extern float *tskin;             /*Surface skin temperature 2-d array (x by y) (K)*/
extern float *invOblen;          /*Surface Monin-Obukhov length "()" 2-d array (x by y) (m)*/
extern float* z0m;               /*roughness length for momentum "()" 2-d array (x by y) (m)*/
extern float* z0t;               /*roughness length for temperature "()" 2-d array (x by y) (m)*/
extern float *qFlux;             /*Base address for latent heat flux*/
extern float *qskin;             /*Base address for skin moisture*/
extern int surflayer_idealsine;   /*selector for idealized sinusoidal surface heat flux or skin temperature forcing*/
extern float surflayer_ideal_ts;  /*start time in seconds for the idealized sinusoidal surface forcing*/
extern float surflayer_ideal_te;  /*end time in seconds for the idealized sinusoidal surface forcing*/
extern float surflayer_ideal_amp; /*maximum amplitude of the idealized sinusoidal surface forcing*/
extern float surflayer_ideal_qts;  /*start time (seconds) for idealized sinusoidal surf. forcing of latent heat flux*/
extern float surflayer_ideal_qte;  /*end time (seconds) for idealized sinusoidal surf. forcing of latent heat flux*/
extern float surflayer_ideal_qamp; /*maximum amplitude of idealized sinusoidal surface forcing of latent heat flux*/
/*OFFSHORE ROUGNESS PARAMETERS*/
extern int surflayer_offshore;       /* offshore selector: 0=off, 1=on */
extern int surflayer_offshore_opt;   /* offshore roughness parameterization: ==0 (Charnock), ==1 (Charnock with variable alpha), ==2 (Taylor & Yelland), ==3 (Donelan), ==4 (Drennan), ==5 (Porchetta) */
extern int surflayer_offshore_dyn;     /* selector to use parameterized ocean parameters: 0=off, 1=on (default) */
extern float surflayer_offshore_hs;    /* significant wave height */
extern float surflayer_offshore_lp;    /* peak wavelength */
extern float surflayer_offshore_cp;    /* wave phase speed */
extern float surflayer_offshore_theta; /* wave/wind angle */
extern int surflayer_offshore_visc;    /* viscous term on z0m: 0=off, 1=on (default) */
extern float* sea_mask;              /* Base Address of memory containing sea mask 0,1 field */
/*---LARGE SCALE FORCING*/ 
extern int lsfSelector;         /* large-scale forcings selector: 0=off, 1=on */
extern float lsf_w_surf;        /* lsf to w at the surface */
extern float lsf_w_lev1;        /* lsf to w at the first specified level */
extern float lsf_w_lev2;        /* lsf to w at the second specified level */
extern float lsf_w_zlev1;       /* lsf to w height 1 */
extern float lsf_w_zlev2;       /* lsf to w height 2 */
extern float lsf_th_surf;       /* lsf to theta at the surface */
extern float lsf_th_lev1;       /* lsf to theta at the first specified level */
extern float lsf_th_lev2;       /* lsf to theta at the second specified level */
extern float lsf_th_zlev1;      /* lsf to theta height 1 */
extern float lsf_th_zlev2;      /* lsf to theta height 2 */
extern float lsf_qv_surf;       /* lsf to qv at the surface */
extern float lsf_qv_lev1;       /* lsf to qv at the first specified level */
extern float lsf_qv_lev2;       /* lsf to qv at the second specified level */
extern float lsf_qv_zlev1;      /* lsf to qv height 1 */
extern float lsf_qv_zlev2;      /* lsf to qv height 2 */
extern int lsf_horMnSubTerms;   /* Switch 0=off, 1=on */
extern int lsf_numPhiVars;      /* number of variables in the slabMeanPhiProfiles set (e.g. rho,u,v,theta,qv=5) */
extern float lsf_freq;          /* large-scale forcing frequency (seconds) */
/*---MOISTURE*/ 
extern int moistureSelector;        /* moisture selector: 0=off, 1=on */
extern int moistureNvars;           /* number of moisture species */
extern int moistureAdvSelectorQv;     /* water vapor advection scheme selector */
extern float moistureAdvSelectorQv_b; /*hybrid advection scheme parameter */
extern int moistureSGSturb;         /* selector to apply sub-grid scale diffusion to moisture fields */
extern int moistureCond;            /* selector to apply condensation to mositure fields */
extern float *moistScalars;         /*Base address for moisture field arrays*/
extern float *moistScalarsFrhs;     /*Base address for moisture forcing field arrays*/
extern float *moistTauFlds;         /*Base address for moisture SGS field arrays*/
extern float *moistTauFlds;         /*Base address for SGS moisture field arrays*/
extern int moistureAdvSelectorQi; /* moisture advection scheme selector for non-qv fields (non-oscillatory schemes) */
extern float moistureCondTscale;  /* relaxation time in seconds */
extern int moistureCondBasePres;  /* selector to use base pressure for microphysics */
extern float moistureMPcallTscale;  /* time scale for microphysics to be called */
/*---FILTERS*/
extern int filterSelector;               /* explicit filter selector: 0=off, 1=on */
extern int filter_6thdiff_vert;          /* vertical 6th-order filter on w selector: 0=off, 1=on */
extern float filter_6thdiff_vert_coeff;  /* vertical 6th-order filter w factor: 0.0=off, 1.0=full */
extern int filter_6thdiff_hori;          /* horizontal 6th-order filter on rho,theta,qv selector: 0=off, 1=on */
extern float filter_6thdiff_hori_coeff;  /* horizontal 6th-order filter factor: 0.0=off, 1.0=full */
extern int filter_divdamp;               /* divergence damping selector: 0=off, 1=on */
/*---RAYLEIGH DAMPING LAYER*/
extern int dampingLayerSelector;       // Rayleigh Damping Layer selector
extern float dampingLayerDepth;       // Rayleigh Damping Layer Depth
 
/*---BASE_STATE*/
extern int stabilityScheme;  /*Base-State stability setup scheme, (0 = none, 1 = profile, 2 = linear in theta)*/
extern float temp_grnd;
extern float pres_grnd;
extern float rho_grnd;
extern float theta_grnd;
extern float zStableBottom;
extern float stableiGradient;
extern float zStableBottom2;
extern float stableiGradient2;
extern float zStableBottom3;
extern float stableiGradient3;
extern float U_g;            /*Zonal (West-East) component of the geostrophic wind*/
extern float V_g;            /*Meridional (South-North)  component of the geostrophic wind*/
extern float z_Ug,z_Vg;
extern float Ug_grad,Vg_grad;
extern int thetaPerturbationSwitch; /* Initial theta perturbations switch: 0=off, 1=on*/
extern float thetaPerturbationHeight; /* Initial theta perturbations maximum height*/
extern float thetaPerturbationAmplitude; /* Initial theta perturbations maximum amplitude*/

extern int physics_oneRKonly; /* selector to apply physics RHS forcing only at the latest RK stage */

/*###################------------- HYDRO_CORE module function declarations ---------------------#################*/

/*----->>>>> int hydro_coreGetParams();    ----------------------------------------------------------------------
* Used to populate the set of parameters for the HYDRO_CORE module
*/
int hydro_coreGetParams();

/*----->>>>> int hydro_coreInit();       ----------------------------------------------------------------------
* Used to broadcast and print parameters, allocate memory, and initialize configuration settings for HYDRO_CORE.
*/
int hydro_coreInit();

/*----->>>>> int hydro_corePrepareFromInitialConditions();   -------------------------------------------------
 * * Used to undertake the sequence of steps to build the Frhs of all hydro_core prognostic variable fields.
 * */
int hydro_corePrepareFromInitialConditions();

/*----->>>>> int hydro_coreGetFieldName();   ----------------------------------------------------------------------
* Used to fill a caller-allocated character array with the i^(th) field name in the hydoFlds memory block .
*/
int hydro_coreGetFieldName(char * fldName, int iFld);

/*----->>>>> int hydro_coreSetBaseState();   -----------------------------------------------------------
 *  * Used to set the Base-State fields for all prognostic variables and pressure.
 *  */
int hydro_coreSetBaseState();

/*----->>>>> int hydro_coreFldStateLogDump(float * Fld);  --------------------------------------------------------
* Utility function to carry out a log-dump summary of the hydro_core state
*/
int hydro_coreStateLogDump();

/*----->>>>> int hydro_coreFldStateLogEntry(float * Fld);  --------------------------------------------------------
* Utility function to log an arbitrary hydro_core field state summary. 
* e.g. [ max(loc) min(loc) mean, variance, isNan(loc), is Inf(loc) ]
*/
int hydro_coreFldStateLogEntry(float* Fld, int fluxConservativeFlag);

/*----->>>>> int hydro_coreFldStateLogEntry(float * Fld);  -----------------------------------------------
* Utility function to log an arbitrary hydro_core field state summary. 
* e.g. [ max(loc) min(loc) mean, variance, isNan(loc), is Inf(loc) ]
*/
int hydro_coreFldStateLogEntry(float * Fld, int fluxConservativeFlag);

/*----->>>>> int hydro_coreCleanup();     ----------------------------------------------------------------
* Used to free all malloced memory by the HYDRO_CORE module.
*/
int hydro_coreCleanup();


#endif // _HYDRO_CORE_H
