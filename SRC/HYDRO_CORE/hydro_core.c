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

/*Model-Extensions includes*/
#ifdef URBAN_EXT
  #include <urban.c>
#endif
#ifdef GAD_EXT
  #include <GAD.c>
#endif

/*##################------------------- HYDRO_CORE module variable definitions ---------------------#################*/
int Nhydro = 5;              /*Number of prognostic variable fields under hydro_core */
int hydroBCs;          /*selector for hydro BC set. 1= LAD, Dirichlet lateral, ceiling and surface boundary conditions,
			                            2= periodicHorizBSVertical */

int hydroForcingWrite;     /*switching for writing output of forcing fields of prognostic variables. 0-off (default), 1= on*/
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
/*HYDRO_CORE Limited Area Domain (LAD) Dirichlet boundary condition parameters and array pointers*/
char *hydroBndysFileBase;   /*Base file name LAD BC set (hydroBCs = 1)*/
char *hydroBndysFile;       /*File name for LAD BC set (hydroBCs = 1)*/
int hydroBndysFileStart;    /*start counter value for LAD BC set files (hydroBCs = 1)*/
int hydroBndysFileEnd;      /*end counter value for LAD BC set files (hydroBCs = 1)*/
int hydroBndysFileCounter;  /*counter value for LAD BC set files (hydroBCs = 1)*/
int nBndyVars;              /*Number of variable fields expected in Bdy-Planes input files*/
int nSurfBndyVars;          /*Number of surface variable fields expected in Bdy-Planes input files*/

float dtBdyPlaneBCs;         /*delta in time (seconds) between BdyPlane sets */
float *XZBdyPlanesGlobal;    /*Base Adress of memory block for lateral-BC XZ-planes (Global domain)*/
float *YZBdyPlanesGlobal;    /*Base Adress of memory block for lateral-BC YZ-planes (Global domain)*/
float *XYBdyPlanesGlobal;    /*Base Adress of memory block for surface/ceiling-BC XY-planes (Global domain)*/
float *XZBdyPlanes;          /*Base Adress of memory block for lateral-BC XZ-planes (per rank domain)*/
float *YZBdyPlanes;          /*Base Adress of memory block for lateral-BC YZ-planes (per rank domain)*/
float *XYBdyPlanes;          /*Base Adress of memory block for surface/ceiling-BC XY-planes (per rank domain)*/
float *XZBdyPlanesPrev;      /*Base Adress of memory block for lateral-BC XZ-planes (per rank domain)*/
float *YZBdyPlanesPrev;      /*Base Adress of memory block for lateral-BC YZ-planes (per rank domain)*/
float *XYBdyPlanesPrev;      /*Base Adress of memory block for surface/ceiling-BC XY-planes (per rank domain)*/
float *XZBdyPlanesNext;      /*Base Adress of memory block for lateral-BC XZ-planes (per rank domain)*/
float *YZBdyPlanesNext;      /*Base Adress of memory block for lateral-BC YZ-planes (per rank domain)*/
float *XYBdyPlanesNext;      /*Base Adress of memory block for surface/ceiling-BC XY-planes (per rank domain)*/
float *SURFBdyPlanesGlobal;  /*Base Adress of memory block for surfaceVariable-BC XY-planes (Global domain)*/
float *SURFBdyPlanes;        /*Base Adress of memory block for surfaceVariable-BC XY-planes (per rank domain)*/
float *SURFBdyPlanesPrev;    /*Base Adress of memory block for surfaceVariable-BC XY-planes (per rank domain)*/
float *SURFBdyPlanesNext;    /*Base Adress of memory block for surfaceVariable-BC XY-planes (per rank domain)*/

/*----Pressure Gradient Force*/ 
int pgfSelector;          /*Pressure Gradient Force (pgf) selector: 0=off, 1=on*/

/*----Buoyancy Force*/ 
int buoyancySelector;     /*buoyancy Force selector: 0=off, 1=on*/

/*----Coriolis*/ 
int coriolisSelector;   /* Coriolis selector, (0 = none, 1 = horizontal terms only, 2 = horizontal and vertical terms*/
float coriolisLatitude; /*Characteristic latitude in degrees from equator of the LES domain*/
float corioConstHorz;   /*Latitude dependent horizontal Coriolis term constant */
float corioConstVert;   /*Latitude dependent Vertical Coriolis term constant */
int coriolis_LAD = 0;       /*Coriolis force selector for LAD BC cases (hydroBCs==1): 0=off, 1=on*/
float corioLS_fact;     /*large-scale factor on Coriolis term*/

/*----Turbulence*/ 
int turbulenceSelector;         /*turbulence scheme selector: 0= none, 1= Lilly/Smagorinsky */
int TKESelector;                /* Prognostic TKE selector: 0= none, 1= Prognostic, 2= requires canopySelector=1 */
int TKEAdvSelector;             /* SGSTKE advection scheme selector */
float TKEAdvSelector_b_hyb;     /*hybrid advection scheme parameter */
float c_s;                      /* Smagorinsky turbulence model constant used for turbulenceSelector = 1 with TKESelector = 0 */
float c_k;                      /* Lilly turbulence model constant used for turbulenceSelector = 1 with TKESelector > 0 */
float *sgstkeScalars;     /* Base Adress of memory containing all prognostic "sgstke" variable fields */ 
float *sgstkeScalarsFrhs; /* Base Adress of memory containing all prognostic "sgstke" RHS forcing fields */ 

/*----Advection*/ 
int advectionSelector;    /*advection scheme selector: 0=1st-order upwind, 1=3rd-order QUICK, 2=hybrid 3rd-4th order,
			    3=hybrid 5th-6th order, 4=3rd-order WENO, 5=5th-order WENO, 6=2nd-order centered */
int ceilingAdvectionBC;   /*selector to allow advection through the domain ceiling 1=on, 0=off (w-ceiling = 0)*/
float b_hyb;      /*hybrid advection scheme parameter: 0.0= lower-order upwind,
                                          1.0=higher-order centered, 0.0 < b_hyb < 1.0 = hybrid */

/*----Diffusion*/ 
int diffusionSelector;    /*diffusion Term-type selector: 0= none, 1= constant, 2= scalar turbulent-diffusivity*/
float nu_0;               /* constant diffusivity used when diffusionSelector = 1 */
float* hydroDiffNuFld;    /*Base adress for diffusivity array used with all prognostic fields*/
float* hydroTauFlds;      /*Base address for scratch/work Tau tensor array*/
float* hydroDiffTauXFlds; /*Base adress for diffusion TauX arrays for all prognostic fields*/
float* hydroDiffTauYFlds; /*Base adress for diffusion TauY arrays for all prognostic fields*/
float* hydroDiffTauZFlds; /*Base adress for diffusion TauZ arrays for all prognostic fields*/

/*---Monin-Obukhov surface layer---*/ 
int surflayerSelector;    /*Monin-Obukhov surface layer selector: 0=off, 1=surface kinematic heat flux (surflayer_wth), 2=skin temperature rate (surflayer_tr) */
float surflayer_z0;       /* roughness length (momentum) */
float surflayer_z0t;      /* roughness length (temperature) */
float surflayer_wth;      /* kinematic sensible heat flux at the surface */
float surflayer_tr;       /* surface temperature rate in K h-1 when surflayerSelector == 2*/
float surflayer_wq;       /* kinematic latent heat flux at the surface */
float surflayer_qr;       /* surface water vapor rate (kg/kg) h-1 */
int surflayer_qskin_input;/* selector to use file input (restart) value for qskin under surflayerSelector == 2 */
int surflayer_stab;       /* exchange coeffcient stability correction selector: 0= on, 1= off */
int surflayer_z0tdyn;     /* dynamic z0t calculation following Zilitinkevich (1995) approach: 0= off, 1= constant Zilitinkevich coeff, 2= variable Zilitinkevich coeff */
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

/*Canopy module parameters*/
int canopySelector;         /* canopy selector: 0=off, 1=on */
int canopySkinOpt;          /* canopy selector to use additional skin friction effect on drag coefficient: 0=off, 1=on */
float canopy_cd;            /* non-dimensional canopy drag coefficient cd coefficient */
float canopy_lf;            /* representative canopy element length scale */
float *canopy_lad;          /* Base Address of memory containing leaf area density (LAD) field [m^{-1}] */

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
int moistureCond;            /* selector to apply condensation to moisture fields */
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

/*---Cell perturbation (CP) method---*/
int cellpertSelector;     /*CP method selector: 0= off, 1= on */
int cellpert_sw2b;        /* switch to do: 0= all four lateral boundaries, 1= only south & west boundaries, 2= only south boundary */
float cellpert_amp;       /* maximum amplitude for the potential temperature perturbations */
int cellpert_nts;         /* number of time steps after which perturbations are seeded */
int cellpert_gppc;        /* number of grid points conforming the cell */
int cellpert_ndbc;        /* number of cells normal to domain lateral boundaries */
int cellpert_kbottom;     /* z-grid point where the perturbations start */
int cellpert_ktop;        /* z-grid point where the perturbations end */
int cellpert_ktop_prev[4];/* z-grid point where the perturbations end array previous time step */
int cellpert_tvcp;        /* time-varying CP method selector: 0= off, 1= on (when hydroBCs == 1) */
float cellpert_eckert;    /* Eckert number for the potential temperature perturbations (when hydroBCs == 1) */
float cellpert_tsfact;    /* factor on the refreshing perturbation time scale (when hydroBCs == 1) */

/*--- Rayleigh Damping Layer ---*/
int dampingLayerSelector;       // Rayleigh Damping Layer selector
float dampingLayerDepth;       // Rayleigh Damping Layer Depth

/*---AUX_SCALARS*/
/*Auxiliary Scalar Fields*/
int NhydroAuxScalars;    /*Number of prognostic auxiliary scalar variable fields */
int AuxScAdvSelector;    /*Advection Scheme to use for auxiliary scalar variable fields */
float AuxScAdvSelector_b_hyb; /* hybrid advection scheme parameter */
float *hydroAuxScalars;  /*Base Adress of memory containing all prognostic auxiliary scalar variable fields */
float *hydroAuxScalarsFrhs; /*Base Adress of memory containing all prognostic auxiliary scalar variable fields */
int AuxScSGSturb;           /* selector to apply sub-grid scale diffusion to auxiliary scalar fields */

/*Auxiliary Scalar Sources*/
char *srcAuxScFile;        /* The path+filename to an Auxilliary Scalar Sources specification file*/
int *srcAuxScTemporalType;     /*Temporal characterization of source (0 = instantaneous, 1 = continuous) */
float *srcAuxScStartSeconds;     /*Source start time in seconds */
float *srcAuxScDurationSeconds;  /*Source duration in seconds */
int *srcAuxScGeometryType;         /*0 = point (single cell volume), 1 = line (line of surface cells) */
float *srcAuxScLocation;      /*Cartesian coordinate tuple 'center' of the source*/
float *srcAuxScGeometryBounds;      /*Cartesian coordinate tuple 'center' of the source*/
int *srcAuxScMassSpecType; /*Mass specification type 0 = strict mass in kg, 1 = mass source rate in kg/s,  */
float *srcAuxScMassSpecValue; /*Mass specification value in kg or kg/s given by srcAuxScMassSpecType 0 or 1 */

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
   hydroBCs = 2; //Default to periodicHorizVerticalAbl
   errorCode = queryIntegerParameter("hydroBCs", &hydroBCs, 1, 2, PARAM_MANDATORY);
   if(hydroBCs==1){
     errorCode = queryStringParameter("hydroBndysFileBase", &hydroBndysFileBase, PARAM_MANDATORY);  
     hydroBndysFileStart = 0;
     errorCode = queryIntegerParameter("hydroBndysFileStart", &hydroBndysFileStart, 0, 500000, PARAM_MANDATORY);
     hydroBndysFileEnd = 0;
     errorCode = queryIntegerParameter("hydroBndysFileEnd", &hydroBndysFileEnd, 0, 500000, PARAM_MANDATORY);
     dtBdyPlaneBCs = 0.0;
     errorCode = queryFloatParameter("dtBdyPlaneBCs", &dtBdyPlaneBCs, 0.0, 6e5, PARAM_MANDATORY);
   }
   hydroForcingWrite = 0; //Default to off
   errorCode = queryIntegerParameter("hydroForcingWrite", &hydroForcingWrite, 0, 1, PARAM_OPTIONAL);
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
   errorCode = queryIntegerParameter("TKESelector", &TKESelector, 0, 2, PARAM_OPTIONAL);
   TKEAdvSelector = 0; //Default to 0 for monotonic 1st-order upstream
   errorCode = queryIntegerParameter("TKEAdvSelector", &TKEAdvSelector, 0, 6, PARAM_OPTIONAL);
   TKEAdvSelector_b_hyb = 0.0; //Default to 0.0
   errorCode = queryFloatParameter("TKEAdvSelector_b_hyb", &TKEAdvSelector_b_hyb, 0.0, 1.0, PARAM_OPTIONAL);
   if (turbulenceSelector == 1){
      if (TKESelector == 0){
         c_s = 0.18; //Default to 0.18
         errorCode = queryFloatParameter("c_s", &c_s, 1e-6, 1e6, PARAM_OPTIONAL);
      }else if (TKESelector > 0){
         c_k = 0.10; //Default to 0.1
         errorCode = queryFloatParameter("c_k", &c_k, 1e-6, 1e6, PARAM_OPTIONAL);
      }
   }
   advectionSelector = 3; //Default to 3
   errorCode = queryIntegerParameter("advectionSelector", &advectionSelector, 0, 6, PARAM_OPTIONAL);
   ceilingAdvectionBC = 0;
   errorCode = queryIntegerParameter("ceilingAdvectionBC", &ceilingAdvectionBC, 0, 1, PARAM_OPTIONAL);
   if ((advectionSelector == 2) || (advectionSelector == 3)){
     b_hyb = 0.8; //Default to 0.8
     errorCode = queryFloatParameter("b_hyb", &b_hyb, 0.0, 1.0, PARAM_OPTIONAL);
   }
   diffusionSelector = 0; //Default to off
   errorCode = queryIntegerParameter("diffusionSelector", &diffusionSelector, 0, 1, PARAM_OPTIONAL);
   if (diffusionSelector==1){
     nu_0 = 1.0; //Default to 1.0 m/s^2
     errorCode = queryFloatParameter("nu_0", &nu_0, 0, FLT_MAX, PARAM_OPTIONAL);
   }
   surflayerSelector = 0; // Default to off
   errorCode = queryIntegerParameter("surflayerSelector", &surflayerSelector, 0, 3, PARAM_MANDATORY);
   surflayer_z0 = 0.1; // Default to 0.1 m 
   errorCode = queryFloatParameter("surflayer_z0", &surflayer_z0, 1e-12, 1e+0, PARAM_MANDATORY);
   surflayer_z0t = 0.1; // Default to 0.1 m 
   errorCode = queryFloatParameter("surflayer_z0t", &surflayer_z0t, 1e-6, 1e+1, PARAM_MANDATORY);
   if (surflayerSelector == 2){
     surflayer_tr = 0.0; // Default to 0.0 K h-1 
     errorCode = queryFloatParameter("surflayer_tr", &surflayer_tr, -1e+1, 1e+1, PARAM_MANDATORY);
   }
   if (surflayerSelector == 1){
     surflayer_wth = 0.0; // Default to 0.0 K m s-1 
     errorCode = queryFloatParameter("surflayer_wth", &surflayer_wth, -5e+0, 5e+0, PARAM_MANDATORY);
   }
   surflayer_idealsine = 0; //Default to off 
   errorCode = queryIntegerParameter("surflayer_idealsine", &surflayer_idealsine, 0, 1, PARAM_OPTIONAL);
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
   surflayer_z0tdyn = 1; // Default to option 1
   errorCode = queryIntegerParameter("surflayer_z0tdyn", &surflayer_z0tdyn, 0, 2, PARAM_OPTIONAL);
   surflayer_offshore = 1; // Default to on
   surflayer_offshore_opt = 4; // Default to 4
   surflayer_offshore_dyn = 1;
   surflayer_offshore_hs = 0.0;
   surflayer_offshore_lp = 0.1;
   surflayer_offshore_cp = 0.1;
   surflayer_offshore_theta = 0.0;
   surflayer_offshore_visc = 1;
   errorCode = queryIntegerParameter("surflayer_offshore", &surflayer_offshore, 0, 1, PARAM_OPTIONAL);
   errorCode = queryIntegerParameter("surflayer_offshore_visc", &surflayer_offshore_visc, 0, 1, PARAM_OPTIONAL);
   if (surflayer_offshore > 0){
     errorCode = queryIntegerParameter("surflayer_offshore_opt", &surflayer_offshore_opt, 0, 5, PARAM_OPTIONAL);
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
   canopySelector = 0; // Default to off
   errorCode = queryIntegerParameter("canopySelector", &canopySelector, 0, 1, PARAM_OPTIONAL);
   canopySkinOpt = 0; // Default to off
   canopy_cd = 0.15; // Default to 0.15
   canopy_lf = 0.1; // Default to 0.1
   if (canopySelector > 0){
     errorCode = queryIntegerParameter("canopySkinOpt", &canopySkinOpt, 0, 1, PARAM_MANDATORY);
     errorCode = queryFloatParameter("canopy_cd", &canopy_cd, 0.0, 1e+2, PARAM_MANDATORY);
     errorCode = queryFloatParameter("canopy_lf", &canopy_lf, 0.0, 1e+2, PARAM_MANDATORY);
   }
   //
   cellpertSelector = 0; // Default to off
   errorCode = queryIntegerParameter("cellpertSelector", &cellpertSelector, 0, 1, PARAM_OPTIONAL);
   cellpert_nts = 500; // Default to 500 time steps
   errorCode = queryIntegerParameter("cellpert_nts", &cellpert_nts, 0, 1e+6, PARAM_OPTIONAL);
   if (cellpertSelector > 0){
     errorCode = queryIntegerParameter("cellpertSelector", &cellpertSelector, 0, 1, PARAM_OPTIONAL);
     cellpert_sw2b = 0; // Default to 0
     errorCode = queryIntegerParameter("cellpert_sw2b", &cellpert_sw2b, 0, 3, PARAM_OPTIONAL);
     cellpert_amp = 0.5; // Default to 0.5 K
     errorCode = queryFloatParameter("cellpert_amp", &cellpert_amp, 0.0, 20.0, PARAM_OPTIONAL);
     cellpert_gppc = 8; // Default to 8 grid points per cell
     errorCode = queryIntegerParameter("cellpert_gppc", &cellpert_gppc, 0, 50, PARAM_OPTIONAL);
     cellpert_ndbc = 3; // Default to 3 cells
     errorCode = queryIntegerParameter("cellpert_ndbc", &cellpert_ndbc, 0, 10, PARAM_OPTIONAL);
     cellpert_kbottom = 1; // Default to 1st grid point above surface
     errorCode = queryIntegerParameter("cellpert_kbottom", &cellpert_kbottom, 1, 10, PARAM_OPTIONAL);
     cellpert_ktop = 20; // Default to 20th grid point above surface
     errorCode = queryIntegerParameter("cellpert_ktop", &cellpert_ktop, 0, 200, PARAM_OPTIONAL);
     if (cellpert_ktop > Nz){
       cellpert_ktop = Nz-10;
     }
     cellpert_tvcp = 0; // Default to off 
     errorCode = queryIntegerParameter("cellpert_tvcp", &cellpert_tvcp, 0, 1, PARAM_OPTIONAL);
     cellpert_eckert = 0.2; // Default to Ec = 0.2
     errorCode = queryFloatParameter("cellpert_eckert", &cellpert_eckert, 0.0, 10.0, PARAM_OPTIONAL);
     cellpert_tsfact = 1.0; // Default to cellpert_tsfact = 1.0
     errorCode = queryFloatParameter("cellpert_tsfact", &cellpert_tsfact, 0.0, 10.0, PARAM_OPTIONAL);
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
     if (surflayerSelector == 2){
       errorCode = queryFloatParameter("surflayer_qr", &surflayer_qr, -1e+1, 1e+1, PARAM_MANDATORY);
       errorCode = queryIntegerParameter("surflayer_qskin_input", &surflayer_qskin_input, 0, 1, PARAM_OPTIONAL);
     }
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
   errorCode = queryIntegerParameter("filterSelector", &filterSelector, 0, 1, PARAM_OPTIONAL);
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
#ifdef GAD_EXT
   /*New EXTENSIONS sub-module style call to get parameters for the GAD sub-module*/
   errorCode = GADGetParams();
#endif
   dampingLayerSelector = 0; // Default to off 
   errorCode = queryIntegerParameter("dampingLayerSelector", &dampingLayerSelector, 0, 1, PARAM_OPTIONAL);
   if(dampingLayerSelector == 1){
     dampingLayerDepth = 100.0; //Default to 100.0 (meters)  
     errorCode = queryFloatParameter("dampingLayerDepth", &dampingLayerDepth, 0.0, FLT_MAX, PARAM_OPTIONAL);
   }
   /*Auxiliary scalar parameters*/
   NhydroAuxScalars = 0; // Default to zero auxiliary scalars
   errorCode = queryIntegerParameter("NhydroAuxScalars", &NhydroAuxScalars, 0, MAX_AUXSC_SRC, PARAM_OPTIONAL);
   if(NhydroAuxScalars > 0){
     AuxScAdvSelector = 0; //Default to 0 for monotonic 1st-order upstream
     errorCode = queryIntegerParameter("AuxScAdvSelector", &AuxScAdvSelector, 0, 6, PARAM_MANDATORY);
     AuxScAdvSelector_b_hyb = 0.0; //Default to 0.0
     errorCode = queryFloatParameter("AuxScAdvSelector_b_hyb", &AuxScAdvSelector_b_hyb, 0.0, 1.0, PARAM_MANDATORY);
     errorCode = queryFileParameter("srcAuxScFile", &srcAuxScFile, PARAM_OPTIONAL);
     AuxScSGSturb = 0; // Default to off
     errorCode = queryIntegerParameter("AuxScSGSturb", &AuxScSGSturb, 0, 1, PARAM_MANDATORY);
     /*Allocate space for source specification arrays*/
     srcAuxScTemporalType = malloc(NhydroAuxScalars*sizeof(int));
     srcAuxScStartSeconds = malloc(NhydroAuxScalars*sizeof(float));
     srcAuxScDurationSeconds = malloc(NhydroAuxScalars*sizeof(float));
     srcAuxScGeometryType = malloc(NhydroAuxScalars*sizeof(int));
     srcAuxScMassSpecType = malloc(NhydroAuxScalars*sizeof(int));
     srcAuxScMassSpecValue = malloc(NhydroAuxScalars*sizeof(float));
     srcAuxScLocation = malloc(NhydroAuxScalars*3*sizeof(float));
     if((srcAuxScFile == NULL) && (NhydroAuxScalars == 1)){  //Can read a single AuxScalar's source characteristics as parameters (original implementation)...
       srcAuxScTemporalType[0] = 0; /*Default to instantaneous */
       errorCode = queryIntegerParameter("srcAuxScTemporalType", &srcAuxScTemporalType[0], 0, 1, PARAM_MANDATORY);
       srcAuxScStartSeconds[0] = 0; /*Default to from simulation start*/
       errorCode = queryFloatParameter("srcAuxScStartSeconds", &srcAuxScStartSeconds[0], 0, FLT_MAX, PARAM_MANDATORY);
       srcAuxScDurationSeconds[0] = 30; /* Default to 30 seconds */
       errorCode = queryFloatParameter("srcAuxScDurationSeconds", &srcAuxScDurationSeconds[0], 0, FLT_MAX, PARAM_MANDATORY);
       srcAuxScGeometryType[0] = 0;  /*Default to 0 = point (single cell volume) */
       errorCode = queryIntegerParameter("srcAuxScGeometryType", &srcAuxScGeometryType[0], 0, 0, PARAM_MANDATORY);
       srcAuxScLocation[0] = 0.0; /* Default to domain origin in x-direction */
       errorCode = queryFloatParameter("srcAuxScLocation_X", &srcAuxScLocation[0], -FLT_MAX, FLT_MAX, PARAM_MANDATORY);
       srcAuxScLocation[1] = 0.0; /* Default to domain origin in y-direction */
       errorCode = queryFloatParameter("srcAuxScLocation_Y", &srcAuxScLocation[1], -FLT_MAX, FLT_MAX, PARAM_MANDATORY);
       srcAuxScLocation[2] = 0.0; /* Default to domain origin in z-direction */
       errorCode = queryFloatParameter("srcAuxScLocation_Z", &srcAuxScLocation[2], -FLT_MAX, FLT_MAX, PARAM_MANDATORY);
       if(srcAuxScGeometryType[0] > 0){
         printf("!!!!!ERROR-- srcAuxScGeometryType = %d > 0 is not yet supported. Source not included --ERROR!!!!!!\n",
                 srcAuxScGeometryType[0]);
         fflush(stdout);
          /*TODO Malloc and initialize the bounds arrays appropriately*/
          //*srcAuxScGeometryBounds;      /*Cartesian coordinate tuple 'center' of the source*/
       }
       srcAuxScMassSpecType[0] = 0; /*Default to a strict total mass*/
       errorCode = queryIntegerParameter("srcAuxScMassSpecType", &srcAuxScMassSpecType[0], 0, 0, PARAM_MANDATORY);
       srcAuxScMassSpecValue[0] = 1.0; /*Default to 1.0 kg or 1 kg/s given by srcAuxScMassSpecType = 0 or 1 */
       errorCode = queryFloatParameter("srcAuxScMassSpecValue", &srcAuxScMassSpecValue[0],
                                        0.0, FLT_MAX, PARAM_MANDATORY);
     } //endif srcAuxScFile == NULL...
   }// endif NhydroAuxScalars > 0
   stabilityScheme = 2; //Default to 2
   errorCode = queryIntegerParameter("stabilityScheme", &stabilityScheme, 2, 2, PARAM_MANDATORY);
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

#ifdef URBAN_EXT
   /*New EXTENSIONS sub-module style call to get parameters for the URBAN sub-module*/
   errorCode = URBANGetParams();
#endif

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
   char AuxScName[MAX_HC_FLDNAME_LENGTH];
   char TauScName[MAX_HC_FLDNAME_LENGTH];
   char sgstkeScName[MAX_HC_FLDNAME_LENGTH];
   char moistName[MAX_HC_FLDNAME_LENGTH];
   char moistName_base[MAX_HC_FLDNAME_LENGTH];
   char moistName_tmp[MAX_HC_FLDNAME_LENGTH];
   float pi;
   int fldStride;
   float z1oz0,z1,z1ozt0;
   int strLength;

   MPI_Barrier(MPI_COMM_WORLD); 
   printf("Entering hydro_coreInit:------\n"); 
   fflush(stdout);
   /*Print the module parameters we are using.*/
   if(mpi_rank_world == 0){
      printComment("HYDRO_CORE parameters---");
      printComment("----------: HYDRO_CORE Submodule Selectors ---");
      printComment("----------: Boundary Conditions Set ---");
      printParameter("hydroBCs", "Selector for hydro BC set. 1= LAD, Dirichlet lateral, ceiling and surface, 2 = periodic-horizontal & ABL-vertical");
      if(hydroBCs==1){ // Using LAD BCs
        printParameter("hydroBndysFileBase", "Basename of limited area domain boundary condition files.");
        printParameter("hydroBndysFileStart", "start counter value for BdyPlane sets");
        printParameter("hydroBndysFileEnd", "end counter value for BdyPlane sets");
        printParameter("dtBdyPlaneBCs", "delta in time (seconds) between BdyPlane sets (default = 0.0)");
      }
      printParameter("hydroForcingWrite", "Switch for writing output of hydroFldsFrhs for prognostic fields. 0 = off, 1=on");
      printParameter("hydroSubGridWrite", "Switch for writing output of Tauij fields. 0 = off, 1=on");
      printParameter("hydroForcingLog", "Switch for logging Frhs summary metrics. 0 = off, 1=on");
      printComment("----------: PRESSURE GRADIENT FORCE ---");
      printParameter("pgfSelector", "Pressure Gradient Force (pgf) selector: 0=off, 1=on");
      printComment("----------: BUOYANCY ---");
      printParameter("buoyancySelector", "Buoyancy force  selector: 0=off, 1=on");
      printComment("----------: CORIOLIS ---");
      printParameter("coriolisSelector", "Corilis force selector: 0= none, 1= horiz. terms, 2= horiz. & vert. terms");
      printParameter("coriolisLatitude", "Characteristic latitude in degrees from equator of the LES domain");
      printComment("----------: TURBULENCE ---");
      printParameter("turbulenceSelector", "turbulence scheme selector: 0= none, 1= Lilly/Smagorinsky ");
      printParameter("TKESelector", "Prognostic TKE selector: 0= none, 1= Prognostic, 2= requires canopySelector=1");
      printParameter("TKEAdvSelector", "advection scheme for SGSTKE equation");
      printParameter("TKEAdvSelector_b_hyb","hybrid advection scheme parameter");
      printParameter("c_s", "Smagorinsky model constant used for turbulenceSelector = 1 and TKESelector = 0");
      printParameter("c_k", "Lilly model constant used for turbulenceSelector = 1 and TKESelector > 0");
      printComment("----------: ADVECTION ---");
      printParameter("advectionSelector", "advection scheme selector: 0=1st-order upwind, 1=3rd-order QUICK, 2=hybrid 3rd-4th order, 3=hybrid 5th-6th order, 4=3rd-order WENO, 5=5th-order WENO, 6=2nd-order centered");
      printParameter("ceilingAdvectionBC", "selector to allow advection through the domain ceiling 1=on, 0=off (w-ceiling = 0)");
      printParameter("b_hyb", "hybrid advection scheme parameter: 0.0= lower-order upwind, 1.0=higher-order centered, 0.0 < b_hyb < 1.0 = hybrid");
      printComment("----------: DIFFUSION ---");
      printParameter("diffusionSelector", "diffusivity selector: 0= none, 1= const.");
      printParameter("nu_0", "constant diffusivity used when diffusionSelector = 1");
      printComment("----------: SURFACE LAYER ---"); 
      printParameter("surflayerSelector", "surfacelayer selector: 0=off, 1=surface kinematic heat flux (surflayer_wth), 2=skin temperature rate (surflayer_tr)");
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
      printParameter("surflayer_z0tdyn", "dynamic z0t calculation following Zilitinkevich (1995) approach: 0= off, 1= constant Zilitinkevich coeff, 2= variable Zilitinkevich coeff");
      printComment("----------: OFFSHORE ROUGHNESS ---");
      printParameter("surflayer_offshore", "offshore selector: 0=off, 1=on");
      printParameter("surflayer_offshore_opt", "offshore roughness parameterization: ==0 (Charnock), ==1 (Charnock with variable alpha), ==2 (Taylor & Yelland), ==3 (Donelan), ==4 (Drennan), ==5 (Porchetta)");
      printComment("----------: CANOPY MODEL ---");
      printParameter("canopySelector", "canopy selector: 0= off, 1= on");
      if (canopySelector > 0){
        printParameter("canopySkinOpt", "canopy selector to use additional skin friction effect on drag coefficient: 0=off, 1=on");
        printParameter("canopy_cd", "non-dimensional canopy drag coefficient when canopySelector > 0");
        printParameter("canopy_lf", "representative canopy element length scale when canopySelector > 0");
      }
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

      printComment("----------: CELL PERTURBATION METHOD ---");
      printParameter("cellpertSelector", "CP method selector: 0= off, 1= on");
      if (cellpertSelector > 0){
        printParameter("cellpert_sw2b", "switch to do: 0=all four lateral boundaries, 1= only south & west boundaries, 2= only south boundary");
        printParameter("cellpert_amp", "maximum amplitude for the potential temperature perturbations");
        printParameter("cellpert_nts", "number of time steps after which perturbations are seeded");
        printParameter("cellpert_gppc", "number of grid points conforming the cell");
        printParameter("cellpert_ndbc", "number of cells normal to domain lateral boundaries");
        printParameter("cellpert_kbottom", "z-grid point where the perturbations start");
        printParameter("cellpert_ktop", "z-grid point where the perturbations end");
        printParameter("cellpert_tvcp", "time-varying CP method selector: 0= off, 1= on");
        printParameter("cellpert_eckert", "Eckert number for the potential temperature perturbations (when hydroBCs == 1)");
        printParameter("cellpert_tsfact", "factor on the refreshing perturbation time scale (when hydroBCs == 1)");
      }
      printComment("----------: RAYLEIGH DAMPING LAYER ---"); 
      printParameter("dampingLayerSelector", "Rayleigh damping layer selector: 0= off, 1= on.");
      printParameter("dampingLayerDepth", "Rayleigh damping layer depth in meters");

      printComment("----------: AUXILIARY SCALARS ---");
      printParameter("NhydroAuxScalars", "Number of prognostic auxiliary scalar fields");
      if(NhydroAuxScalars > 0){
        printParameter("AuxScAdvSelector", "Advection Scheme to use for auxiliary scalar fields");
        printParameter("AuxScAdvSelector_b_hyb", "hybrid advection scheme parameter");
        printParameter("AuxScSGSturb", "selector to apply sub-grid scale diffusion to auxiliary scalar fields");
        printComment("----------: AUXILIARY SCALAR SOURCES ---");
        printParameter("srcAuxScFile",
                       "The path+filename to an Auxilliary Scalar Sources specification file");
        if((srcAuxScFile == NULL) && (NhydroAuxScalars == 1)){
          printParameter("srcAuxScTemporalType",
                         "Temporal characterization of source (0 = instantaneous, 1 = continuous)");
          printParameter("srcAuxScStartSeconds",
                         "Source start time in seconds from start of simulation (i.e. time = 0.0)");
          printParameter("srcAuxScDurationSeconds", "Source duration in seconds from srcAuxScStartSeconds");
          printParameter("srcAuxScGeometryType", "0 = point (single cell volume), 1 = line (line of surface cells)");
          printParameter("srcAuxScLocation_X", "Source geometry centroid position in x (west-east)");
          printParameter("srcAuxScLocation_Y", "Source geometry centroid position in y (south-north)");
          printParameter("srcAuxScLocation_Z", "Source geometry centroid position in z (vertical above the surface)");
          printParameter("srcAuxScMassSpecType",
                         "Source mass specification type 0 = mass in kg, 1 = mass source rate in kg/s");
          printParameter("srcAuxScMassSpecValue",
                         "Source mass specification in kg or kg/s given by srcAuxScMassSpecType 0 or 1");
        }//end if (srcAuxScFile == NULL) ....
      }// endif NhydroAuxScalars > 0

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
#ifdef URBAN_EXT
   /*New EXTENSIONS sub-module style call to print parameters for the URBAN sub-module*/
   printComment("----------: URBAN MODEL ---");
   errorCode = URBANPrintParams();
#endif
#ifdef GAD_EXT
   /*New EXTENSIONS sub-module style call to print parameters for the GAD sub-module*/
   errorCode = GADPrintParams();
#endif
   /*Broadcast the parameters across mpi_ranks*/
   MPI_Bcast(&hydroBCs, 1, MPI_INT, 0, MPI_COMM_WORLD);
   if(hydroBCs==1){  // Using LAD BCs
     /*Determine strLength of hydroBndysFile*/
     if(mpi_rank_world == 0){
       if(hydroBndysFileBase != NULL){
         strLength = strlen(hydroBndysFileBase)+1;
       }else{
         strLength = 0;
       }
     } //end if(mpi_rank_world == 0)
     MPI_Bcast(&strLength, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);
     if(strLength > 0){
       if(mpi_rank_world != 0){
         hydroBndysFileBase = (char *) malloc(strLength*sizeof(char));
       } //if a non-root mpi_rank
       MPI_Bcast(hydroBndysFileBase, strLength, MPI_CHARACTER, 0, MPI_COMM_WORLD);
     }
     //Allocate for a full hydroBndysFile string (including up to 16 counter-digit characters)
     hydroBndysFile = (char *) malloc(strLength+16*sizeof(char));
     //Set coriolis_LAD switch to "on"
     coriolis_LAD = 1;
     //Broadcast remaining LAD BC set parameters     
     MPI_Bcast(&hydroBndysFileStart, 1, MPI_INT, 0, MPI_COMM_WORLD);
     MPI_Bcast(&hydroBndysFileEnd, 1, MPI_INT, 0, MPI_COMM_WORLD);
     MPI_Bcast(&dtBdyPlaneBCs, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
   } //end if hydroBCs == 1
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
   MPI_Bcast(&ceilingAdvectionBC, 1, MPI_INT, 0, MPI_COMM_WORLD);
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
   MPI_Bcast(&surflayer_z0tdyn, 1, MPI_INT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&surflayer_offshore, 1, MPI_INT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&surflayer_offshore_opt, 1, MPI_INT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&surflayer_offshore_dyn, 1, MPI_INT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&surflayer_offshore_hs, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&surflayer_offshore_lp, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&surflayer_offshore_cp, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&surflayer_offshore_theta, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&surflayer_offshore_visc, 1, MPI_INT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&canopySelector, 1, MPI_INT, 0, MPI_COMM_WORLD);
   if (canopySelector > 0){
     MPI_Bcast(&canopySkinOpt, 1, MPI_INT, 0, MPI_COMM_WORLD);
     MPI_Bcast(&canopy_cd, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
     MPI_Bcast(&canopy_lf, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
   }
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
   MPI_Bcast(&cellpertSelector, 1, MPI_INT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&cellpert_nts, 1, MPI_INT, 0, MPI_COMM_WORLD);
   if (cellpertSelector > 0){
     MPI_Bcast(&cellpert_sw2b, 1, MPI_INT, 0, MPI_COMM_WORLD);
     MPI_Bcast(&cellpert_amp, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
     MPI_Bcast(&cellpert_gppc, 1, MPI_INT, 0, MPI_COMM_WORLD);
     MPI_Bcast(&cellpert_ndbc, 1, MPI_INT, 0, MPI_COMM_WORLD);
     MPI_Bcast(&cellpert_kbottom, 1, MPI_INT, 0, MPI_COMM_WORLD);
     MPI_Bcast(&cellpert_ktop, 1, MPI_INT, 0, MPI_COMM_WORLD);
     //Initialize the ktop_prev vector on all ranks
     cellpert_ktop_prev[0] = cellpert_ktop; // use params value as previous cellpert_ktop value
     cellpert_ktop_prev[1] = cellpert_ktop; // use params value as previous cellpert_ktop value
     cellpert_ktop_prev[2] = cellpert_ktop; // use params value as previous cellpert_ktop value
     cellpert_ktop_prev[3] = cellpert_ktop; // use params value as previous cellpert_ktop value
     MPI_Bcast(&cellpert_tvcp, 1, MPI_INT, 0, MPI_COMM_WORLD);
     MPI_Bcast(&cellpert_eckert, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
     MPI_Bcast(&cellpert_tsfact, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
   }
   MPI_Bcast(&dampingLayerSelector, 1, MPI_INT, 0, MPI_COMM_WORLD); 
   MPI_Bcast(&dampingLayerDepth, 1, MPI_FLOAT, 0, MPI_COMM_WORLD); 
   MPI_Bcast(&NhydroAuxScalars, 1, MPI_INT, 0, MPI_COMM_WORLD);
   if(NhydroAuxScalars > 0){
     MPI_Bcast(&AuxScAdvSelector, 1, MPI_INT, 0, MPI_COMM_WORLD);
     MPI_Bcast(&AuxScAdvSelector_b_hyb, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
     MPI_Bcast(&AuxScSGSturb, 1, MPI_INT, 0, MPI_COMM_WORLD);
     /*Determine strLength of srcAuxScFile*/
     strLength = 0;
     if(mpi_rank_world == 0){
        if(srcAuxScFile != NULL){
           strLength = strlen(srcAuxScFile)+1;
        }else{
         strLength = 0;
       }
     } //end if(mpi_rank_world == 0)
     MPI_Bcast(&strLength, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);
     if(strLength > 0){
       if(mpi_rank_world != 0){
          srcAuxScFile = (char *) malloc(strLength*sizeof(char));
       } //if a non-root mpi_rank
       MPI_Bcast(srcAuxScFile, strLength, MPI_CHARACTER, 0, MPI_COMM_WORLD);
     }//endif strLength of srcAuxScFile > 0 
     if(mpi_rank_world != 0){
       srcAuxScTemporalType = malloc(NhydroAuxScalars*sizeof(int));
       srcAuxScStartSeconds = malloc(NhydroAuxScalars*sizeof(float));
       srcAuxScDurationSeconds = malloc(NhydroAuxScalars*sizeof(float));
       srcAuxScGeometryType = malloc(NhydroAuxScalars*sizeof(int));
       srcAuxScMassSpecType = malloc(NhydroAuxScalars*sizeof(int));
       srcAuxScMassSpecValue = malloc(NhydroAuxScalars*sizeof(float));
       srcAuxScLocation = malloc(NhydroAuxScalars*3*sizeof(float));
     } //if a non-root mpi_rank
     if((srcAuxScFile == NULL) && (NhydroAuxScalars == 1)){
       MPI_Bcast(&srcAuxScTemporalType[0], NhydroAuxScalars, MPI_INT, 0, MPI_COMM_WORLD);
       MPI_Bcast(&srcAuxScStartSeconds[0], NhydroAuxScalars, MPI_FLOAT, 0, MPI_COMM_WORLD);
       MPI_Bcast(&srcAuxScDurationSeconds[0], NhydroAuxScalars, MPI_FLOAT, 0, MPI_COMM_WORLD);
       MPI_Bcast(&srcAuxScGeometryType[0], NhydroAuxScalars, MPI_INT, 0, MPI_COMM_WORLD);
       MPI_Bcast(&srcAuxScLocation[0], NhydroAuxScalars*3, MPI_FLOAT, 0, MPI_COMM_WORLD);
       MPI_Bcast(&srcAuxScMassSpecType[0], NhydroAuxScalars, MPI_INT, 0, MPI_COMM_WORLD);
       MPI_Bcast(&srcAuxScMassSpecValue[0], NhydroAuxScalars, MPI_FLOAT, 0, MPI_COMM_WORLD);
     }else if((srcAuxScFile == NULL) && (NhydroAuxScalars > 1)){
       printf("srcAuxFile is NULL, but NhydroAuxScalars = %d is > 1. INVALID!!!i Use a srcAuxScFile to specifiy > 1 AuxScalar sources...\n", NhydroAuxScalars);
       fflush(stdout);
       errorCode = 1;
       exit(errorCode);
     }else{
       errorCode = srcAuxScConstructor();
     }//endif (srcAuxScFile == NULL)...
   }// endif NhydroAuxScalars > 0 
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
     // Add NetCDF attributes based on field type
     errorCode = hydro_coreAddFieldAttributes(&fldName[0], 0); // 0 = not forcing
     printf("hydro_coreInit:hydroFlds[%d] = %s stored at %p, has been registered with IO.\n",
            iFld,&fldName[0],&hydroFlds[iFld*fldStride]);
     fflush(stdout);
   } //end for iFld...
   
   /* Diagnostic Perturbation Pressure field */
   hydroPres = memAllocateFloat3DField(Nxp, Nyp, Nzp, Nh, "hydroPres");
   errorCode = sprintf(&fldName[0],"pressure");
   errorCode = ioRegisterVar(&fldName[0], "float", 4, dims4d, hydroPres);
   // Add attributes for pressure field
   errorCode = hydro_coreAddFieldAttributes(&fldName[0], 0); // 0 = not forcing
   printf("hydro_coreInit:Field = %s stored at %p, has been registered with IO.\n",
          &fldName[0],hydroPres);
   
   /* Frhs */
   hydroFldsFrhs = memAllocateFloat4DField(Nhydro, Nxp, Nyp, Nzp, Nh, "hydroFldsFrhs");
   for(iFld = 0; iFld < Nhydro; iFld ++){
     if(hydroForcingWrite == 1){
       errorCode = hydro_coreGetFieldName( &fldName[0], iFld);
       sprintf(&frhsName[0],"F_%s",&fldName[0]);
       errorCode = ioRegisterVar(&frhsName[0], "float", 4, dims4d, &hydroFldsFrhs[iFld*fldStride]);
       // Add attributes for forcing fields
       errorCode = hydro_coreAddFieldAttributes(&frhsName[0], 1); // 1 = is forcing
       printf("hydro_coreInit:hydroFldsFrhs[%d] = %s stored at %p, has been registered with IO.\n",
              iFld,&frhsName[0],&hydroFldsFrhs[iFld*fldStride]);
     }
   } //end for iFld...

   /* Auxiliary Scalars and associated Frhs */
   if(NhydroAuxScalars > 0){
     hydroAuxScalars = memAllocateFloat4DField(NhydroAuxScalars, Nxp, Nyp, Nzp, Nh, "hydroAuxScalars");
     hydroAuxScalarsFrhs = memAllocateFloat4DField(NhydroAuxScalars, Nxp, Nyp, Nzp, Nh, "hydroAuxScalarsFrhs");
     for(iFld = 0; iFld < NhydroAuxScalars; iFld ++){
        sprintf(&AuxScName[0],"AuxScalar_%d",iFld);
        errorCode = ioRegisterVar(&AuxScName[0], "float", 4, dims4d, &hydroAuxScalars[iFld*fldStride]);
	// Add attributes for auxiliary scalar
        errorCode = hydro_coreAddFieldAttributes(&AuxScName[0], 0);
        printf("hydro_coreInit:hydroAuxScalars[%d] = %s stored at %p, has been registered with IO.\n",
               iFld,&AuxScName[0],&hydroAuxScalars[iFld*fldStride]);
        fflush(stdout);
     } //end for iFld...
   } //end if NhydroAuxScalars
       
   /* Prognostic SGSTKE equation and associated Frhs */ 
   if(TKESelector > 0){
     sgstkeScalars = memAllocateFloat4DField(TKESelector, Nxp, Nyp, Nzp, Nh, "sgstkeScalars");
     sgstkeScalarsFrhs = memAllocateFloat4DField(TKESelector, Nxp, Nyp, Nzp, Nh, "sgstkeScalarsFrhs");
     for(iFld = 0; iFld < TKESelector; iFld ++){
        sprintf(&sgstkeScName[0],"TKE_%d",iFld);
        errorCode = ioRegisterVar(&sgstkeScName[0], "float", 4, dims4d, &sgstkeScalars[iFld*fldStride]);
        printf("hydro_coreInit:sgstkeScalars[%d] = %s stored at %p, has been registered with IO.\n",
               iFld,&sgstkeScName[0],&sgstkeScalars[iFld*fldStride]);
	// Add attributes for TKE scalar
	errorCode = hydro_coreAddFieldAttributes(&sgstkeScName[0], 0);
        fflush(stdout);
     } //end for iFld...
     if(hydroForcingWrite == 1){ // add rhs forcing of SGSTKE equation
       for(iFld = 0; iFld < TKESelector; iFld ++){
         sprintf(&sgstkeScName[0],"F_TKE%d",iFld);
         errorCode = ioRegisterVar(&sgstkeScName[0], "float", 4, dims4d, &sgstkeScalarsFrhs[iFld*fldStride]);
         // Add attributes for TKE forcing field
         errorCode = hydro_coreAddFieldAttributes(&sgstkeScName[0], 1);
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
     // Add NetCDF attributes after registration
     errorCode = hydro_coreAddFieldAttributes(&fldName[0], 0); // 0 = not a forcing field
     printf("hydro_coreInit:hydroBaseStateFlds[%d] = %s stored at %p, has been registered with IO.\n",
            iFld,&fldName[0],&hydroBaseStateFlds[iFld*fldStride]);
     fflush(stdout);
   } //end for iFld...
   errorCode = sprintf(&fldName[0],"BS_pressure");
   errorCode = ioRegisterVar(&fldName[0], "float", 4, dims4d, hydroBaseStatePres);
   // Add NetCDF attributes for pressure field
   errorCode = hydro_coreAddFieldAttributes(&fldName[0], 0); // 0 = not a forcing field
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
	// Add NetCDF attributes for the registered variable
	errorCode = hydro_coreAddFieldAttributes(&TauScName[0], 0);
        printf("hydro_coreInit:hydroTauFlds[%d] = %s stored at %p, has been registered with IO.\n",
               iFld,&TauScName[0],&hydroTauFlds[iFld*fldStride]);
        fflush(stdout);
     } //end for iFld...
     sprintf(&TauScName[0],"TauTH%d",1);
     errorCode = ioRegisterVar(&TauScName[0], "float", 4, dims4d, &hydroTauFlds[6*fldStride]);
     // Add NetCDF attributes for TauTH1
     errorCode = hydro_coreAddFieldAttributes(&TauScName[0], 0);
     printf("hydro_coreInit:hydroTauFlds[6] = %s stored at %p, has been registered with IO.\n",
             &TauScName[0],&hydroTauFlds[6*fldStride]);
     sprintf(&TauScName[0],"TauTH%d",2);
     errorCode = ioRegisterVar(&TauScName[0], "float", 4, dims4d, &hydroTauFlds[7*fldStride]);
     // Add NetCDF attributes for TauTH2
     errorCode = hydro_coreAddFieldAttributes(&TauScName[0], 0);
     printf("hydro_coreInit:hydroTauFlds[7] = %s stored at %p, has been registered with IO.\n",
             &TauScName[0],&hydroTauFlds[7*fldStride]);
     sprintf(&TauScName[0],"TauTH%d",3);
     errorCode = ioRegisterVar(&TauScName[0], "float", 4, dims4d, &hydroTauFlds[8*fldStride]);
     // Add NetCDF attributes for TauTH3
     errorCode = hydro_coreAddFieldAttributes(&TauScName[0], 0);
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
     // Add NetCDF attributes for the registered variable
     errorCode = hydro_coreAddFieldAttributes(&fldName[0], 0);
     printf("hydro_coreInit:Field = %s stored at %p, has been registered with IO.\n",
             &fldName[0],tskin);
     fflush(stdout);
     errorCode = sprintf(&fldName[0],"fricVel");
     errorCode = ioRegisterVar(&fldName[0], "float", 3, dims2dTD, fricVel);
     // Add NetCDF attributes for the registered variable
     errorCode = hydro_coreAddFieldAttributes(&fldName[0], 0);
     printf("hydro_coreInit:Field = %s stored at %p, has been registered with IO.\n",
             &fldName[0],fricVel);
     fflush(stdout);
     errorCode = sprintf(&fldName[0],"htFlux");
     errorCode = ioRegisterVar(&fldName[0], "float", 3, dims2dTD, htFlux);
     // Add NetCDF attributes for the registered variable
     errorCode = hydro_coreAddFieldAttributes(&fldName[0], 0);
     printf("hydro_coreInit:Field = %s stored at %p, has been registered with IO.\n",
             &fldName[0],htFlux);
     fflush(stdout);
     errorCode = sprintf(&fldName[0],"invOblen");
     errorCode = ioRegisterVar(&fldName[0], "float", 3, dims2dTD, invOblen);
     // Add NetCDF attributes for the registered variable
     errorCode = hydro_coreAddFieldAttributes(&fldName[0], 0);
     printf("hydro_coreInit:Field = %s stored at %p, has been registered with IO.\n",
             &fldName[0],invOblen);
     fflush(stdout);
     if (moistureSelector > 0){
       errorCode = sprintf(&fldName[0],"qskin");
       errorCode = ioRegisterVar(&fldName[0], "float", 3, dims2dTD, qskin);
       // Add NetCDF attributes for the registered variable
       errorCode = hydro_coreAddFieldAttributes(&fldName[0], 0);
       printf("hydro_coreInit:Field = %s stored at %p, has been registered with IO.\n",
               &fldName[0],qskin);
       fflush(stdout);
       errorCode = sprintf(&fldName[0],"qFlux");
       errorCode = ioRegisterVar(&fldName[0], "float", 3, dims2dTD, qFlux);
       // Add NetCDF attributes for the registered variable
       errorCode = hydro_coreAddFieldAttributes(&fldName[0], 0);
       printf("hydro_coreInit:Field = %s stored at %p, has been registered with IO.\n",
               &fldName[0],qFlux);
       fflush(stdout);
     }
     errorCode = sprintf(&fldName[0],"z0m");
     errorCode = ioRegisterVar(&fldName[0], "float", 3, dims2dTD, z0m);
     // Add NetCDF attributes for the registered variable
     errorCode = hydro_coreAddFieldAttributes(&fldName[0], 0);
     printf("hydro_coreInit:Field = %s stored at %p, has been registered with IO.\n",
             &fldName[0],z0m);
     fflush(stdout);
     errorCode = sprintf(&fldName[0],"z0t");
     errorCode = ioRegisterVar(&fldName[0], "float", 3, dims2dTD, z0t);
     // Add NetCDF attributes for the registered variable
     errorCode = hydro_coreAddFieldAttributes(&fldName[0], 0);
     printf("hydro_coreInit:Field = %s stored at %p, has been registered with IO.\n",
             &fldName[0],z0t);
     fflush(stdout);
#ifdef URBAN_EXT
     /*New sub-module style Init() call for URBAN initialization.*/
     errorCode=URBANInit();
#endif
#ifdef GAD_EXT
     /*New sub-module style Init() call for GAD initialization.*/
     errorCode=GADInit();
#endif
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
       if(hydroBCs!=1){
         printf("\n\n\nERROR: hydro_coreInit: surflayerSelector = 3, but hydroBCs ~= 1... No surfVarBndy planes available for surflayerSelector = 3. \n\n\n\n");
         fflush(stdout);
       }
     }
   } // end of surflayerSelector > 0

   if(surflayer_offshore>0){
     sea_mask = memAllocateFloat2DField(Nxp, Nyp, Nh, "sea_mask");
     errorCode = sprintf(&fldName[0],"SeaMask");
     errorCode = ioRegisterVar(&fldName[0], "float", 3, dims2dTD, sea_mask);
     // Add NetCDF attributes for the registered variable
     errorCode = hydro_coreAddFieldAttributes(&fldName[0], 0);
     printf("surflayer_offshore:Field = %s stored at %p, has been registered with IO.\n",
             &fldName[0],sea_mask);
     fflush(stdout);
   }

   if(cellpertSelector>0){ //Cell Perturbation parameters (time-varying when cellpert_tvcp == 1 and hydroBCs==1)
     errorCode = sprintf(&fldName[0],"cellpert_amp");
     errorCode = ioRegisterVar(&fldName[0], "float", 1, dims1dTD, &cellpert_amp);
     // Add NetCDF attributes for the registered variable
     errorCode = ioAddStandardAttrs(&fldName[0], "K", "Cell perturbation amplitude", NULL);
     printf("cellpert:Variable = %s stored at %p, has been registered with IO.\n",
            &fldName[0],&cellpert_amp);
     fflush(stdout);
     errorCode = sprintf(&fldName[0],"cellpert_nts");
     errorCode = ioRegisterVar(&fldName[0], "int", 1, dims1dTD, &cellpert_nts);
     // Add NetCDF attributes for the registered variable
     errorCode = ioAddStandardAttrs(&fldName[0], "-", "Cell perturbation refresh rate in time steps", NULL);
     printf("cellpert:Variable = %s stored at %p, has been registered with IO.\n",
            &fldName[0],&cellpert_nts);
     fflush(stdout);
     errorCode = sprintf(&fldName[0],"cellpert_ktop");
     errorCode = ioRegisterVar(&fldName[0], "int", 1, dims1dTD, &cellpert_ktop);
     // Add NetCDF attributes for the registered variable
     errorCode = ioAddStandardAttrs(&fldName[0], "-", "Cell perturbation uppermost vertical grid level", NULL);
     printf("cellpert:Variable = %s stored at %p, has been registered with IO.\n",
            &fldName[0],&cellpert_ktop);
     fflush(stdout);
   } // end of cellpertSelector > 0
     
   if(canopySelector>0){
     canopy_lad = memAllocateFloat3DField(Nxp, Nyp, Nzp, Nh, "canopy_lad");
     errorCode = sprintf(&fldName[0],"CanopyLAD");
     errorCode = ioRegisterVar(&fldName[0], "float", 4, dims4d, canopy_lad);
     // Add NetCDF attributes for the registered variable
     errorCode = hydro_coreAddFieldAttributes(&fldName[0], 0);
     printf("canopy:Field = %s stored at %p, has been registered with IO.\n",
            &fldName[0],canopy_lad);
     fflush(stdout);
   } // end of canopySelector > 0

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
	// Add NetCDF attributes for the registered variable
	errorCode = hydro_coreAddFieldAttributes(&moistName[0], 0);
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
	 // Add NetCDF attributes for the registered variable
	 errorCode = hydro_coreAddFieldAttributes(&moistName[0], 1);
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
	     // Add NetCDF attributes for the moisture SGS field
             char longName[256];
             char *direction[] = {"x", "y", "z"};
             if (iFld == 0) { // TauQv (water vapor)
                 sprintf(longName, "Subgrid-%s water vapor flux in %s direction", direction[iFld2], direction[iFld2]);
                 errorCode = ioAddStandardAttrs(&moistName[0], "kg kg-1 m s-1", longName, NULL);
             } else if (iFld == 1) { // TauQl (liquid water)
                 sprintf(longName, "Subgrid-%s liquid water flux in %s direction", direction[iFld2], direction[iFld2]);
                 errorCode = ioAddStandardAttrs(&moistName[0], "kg kg-1 m s-1", longName, NULL);
             } else {
                 // Generic moisture SGS field for other moisture species
                 sprintf(longName, "Subgrid-%s moisture flux in %s direction", direction[iFld2], direction[iFld2]);
                 errorCode = ioAddStandardAttrs(&moistName[0], "kg kg-1 m s-1", longName, NULL);
             }
             printf("hydro_coreInit:moistTauFlds[%d] = %s stored at %p, has been registered with IO.\n",
                    iFld*3+iFld2,&moistName[0],&moistTauFlds[(iFld*3+iFld2)*fldStride]);
             fflush(stdout);
          }
       }
     }

   } // end of moistureSelector > 0

   /* Check related parameters and Allocate for LAD BCs (hydroBCs == 1)*/
   if(hydroBCs==1){  // Using LAD BCs
     if(ceilingAdvectionBC==0){
       printf("!!!! WARNING !!!!!-- The parameter ceilingAdvectionBC = 0 enforces rigid lid (w = 0 at ceiling), but currently hydroBCs = 1 implying Dirchlet boundary values are provided.\n");
       printf("!!!! WARNING !!!!!-- When using hydroBCs = 1, ceilingAdvectionBC = 1 is recommended.");
       fflush(stdout);
     }
     if( moistureSelector > 0){
       nBndyVars = Nhydro+moistureNvars;
       nSurfBndyVars = 2;   //Only allows tskin and qskin
     }else{
       nBndyVars = Nhydro;
       nSurfBndyVars = 1;   //Only allows tskin
     } //end if moisture is on else not   //NOTE: Doesn't handle any AuxScalars or TKE-related Prog. variables.
     XZBdyPlanesGlobal = (float *) malloc( 2*(nBndyVars)*Nx*Nz*sizeof(float) );
     YZBdyPlanesGlobal = (float *) malloc( 2*(nBndyVars)*Ny*Nz*sizeof(float) );
     XYBdyPlanesGlobal = (float *) malloc( 2*(nBndyVars)*Nx*Ny*sizeof(float) );
     XZBdyPlanes = (float *) malloc( 2*(nBndyVars)*(Nxp+2*Nh)*(Nzp+2*Nh)*sizeof(float) );
     YZBdyPlanes = (float *) malloc( 2*(nBndyVars)*(Nyp+2*Nh)*(Nzp+2*Nh)*sizeof(float) );
     XYBdyPlanes = (float *) malloc( 2*(nBndyVars)*(Nxp+2*Nh)*(Nyp+2*Nh)*sizeof(float) );
     XZBdyPlanesNext = (float *) malloc( 2*(nBndyVars)*(Nxp+2*Nh)*(Nzp+2*Nh)*sizeof(float) );
     YZBdyPlanesNext = (float *) malloc( 2*(nBndyVars)*(Nyp+2*Nh)*(Nzp+2*Nh)*sizeof(float) );
     XYBdyPlanesNext = (float *) malloc( 2*(nBndyVars)*(Nxp+2*Nh)*(Nyp+2*Nh)*sizeof(float) );
     if(surflayerSelector == 3){  //These BdyPlanes memory blocks are 1-sided (i.e. low-side, surface only)
       SURFBdyPlanesGlobal = (float *) malloc( (nSurfBndyVars)*Nx*Ny*sizeof(float) );
       SURFBdyPlanes = (float *) malloc( (nSurfBndyVars)*(Nxp+2*Nh)*(Nyp+2*Nh)*sizeof(float) );
       SURFBdyPlanesNext = (float *) malloc( (nSurfBndyVars)*(Nxp+2*Nh)*(Nyp+2*Nh)*sizeof(float) );
     }
   } //end if hydroBCs == 1

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
   }else{
     corioConstHorz = 0.0;
     corioConstVert = 0.0;
   } //end if coriolisSelector > 0... else
   if(coriolis_LAD > 0){
     corioLS_fact = 0.0;
     if(hydroBCs != 1){
       printf("*************************************************************************************\n");
       printf("**** WARNING: coriolis_LAD = 1 SHOULD ONLY BE USED TOGETHER WITH hydroBCs == 1!!! ***\n");
       printf("*************************************************************************************\n");
       fflush(stdout);
     }
   }else{
       corioLS_fact = 1.0;
   }//end if coriolis_LAD > 0, else ...
   
   /* If this is a periodic domain according to hydroBCs set up the rank neighbor topoloogy to be cyclic */
   if(hydroBCs == 0){
     errorCode = fempi_SetupPeriodicDomainDecompositionRankTopology(1, 1); //Periodic in x and y
   }else if (hydroBCs == 2){
     errorCode = fempi_SetupPeriodicDomainDecompositionRankTopology(1, 1); //Periodic in x and y
   }// end if hydroBCS == 0 or 2 setup periodic horizontal neighbor rank topology
  
   return(errorCode);
} //end hydro_coreInit()

/*----->>>>> int hydro_coreSecondaryPrepariations();   -------------------------------------------------
* Secondary preparations (initializations) in the HYDRO_CORE module following secondary
* GRID module preparations  i.e. definition of the domain coordinate system and Jacobians
* and TIME_INTEGRATION module initialization
*/
int hydro_coreSecondaryPreparations(float dt){
  int errorCode;

  /*Now that the grid module is completely defined, setup the base state*/
  errorCode = hydro_coreSetBaseState();

#ifdef GAD_EXT
  /*If GAD is included, define the mask arrays from the turbine characteristics array inputs*/
  if(GADSelector > 0){
    /*Create the swept-volume mask for the turbine array read in through this constructor*/
    errorCode = GADInitTurbineRefChars(dt);
    errorCode = GADCreateTurbineVolMask();
    errorCode = GADCreateTurbineRotorMask();
  }//end if GADSelector > 0
#endif

  return(errorCode);
} //end hydro_coreSecondaryPrepariations()

/*----->>>>> int hydro_corePrepareFromInitialConditions();   -------------------------------------------------
* Used to undertake the sequence of steps to build the Frhs of all hydro_core prognostic variable fields.
*/
int hydro_corePrepareFromInitialConditions(int simTime_itRestart, float dt){
  int errorCode = HYDRO_CORE_SUCCESS;
 
  if(hydroBCs==1){ //Using LAD BCs
    printf("mpi_rank_world--%d/%d: Starting hydro_coreSetupBndyPlanesAllRanks() under restart\n",
           mpi_rank_world,mpi_size_world);
    fflush(stdout);
    hydroBndysFileCounter=hydroBndysFileStart+(simTime_itRestart/((int)roundf(dtBdyPlaneBCs/dt)));
    printf("mpi_rank_world--%d/%d: Re-starting with hydroBndysFileCounter = %d\n",
           mpi_rank_world,mpi_size_world,hydroBndysFileCounter);
    fflush(stdout);
    sprintf(hydroBndysFile,"%s.%d",hydroBndysFileBase,hydroBndysFileCounter);
    errorCode = hydro_coreSetupBndyPlanesAllRanks();
    printf("mpi_rank_world--%d/%d: hydro_coreSetupBndyPlanesAllRanks() under restart completed!\n",
            mpi_rank_world,mpi_size_world);
    fflush(stdout);

    errorCode = hydro_coreReadNextBndyPlanesFile();
  }//end if hydroBCs==1
 
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
     printf("stabilityScheme == 3: ERROR: No scheme implemented for stabilityScheme == 3, use instead 1, 2, or 4!! \n");
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

/*----->>>>> int hydro_coreSetupBndyPlanesAllRanks();   ---------------------------------------------------
* Utility to read/scatter (across ranks as appropriate) the next set of BdyPlanes in the series
*/
int hydro_coreSetupBndyPlanesAllRanks(){
   int errorCode = HYDRO_CORE_SUCCESS;
   int ncid;

   int fieldIndex;
   char fieldName[256];

   /* Boundary P Conditions */
   if(hydroBndysFile != NULL){ /*If a Bndys file exists read it, else log an error.*/
     if(mpi_rank_world == 0){
       /*Open the File*/
       printf("Attempting to open hydroBndysFile = %s\n",hydroBndysFile);
       errorCode = ioOpenNetCDFinFile(hydroBndysFile, &ncid);
       printf("Opened hydroBndysFile = %s with ncid = %d\n",hydroBndysFile,ncid);
       fflush(stdout);
       sprintf(fieldName,"rho");
       fieldIndex = 0;
       errorCode = hydro_coreReadFieldBndyPlanes(ncid, fieldName, fieldIndex);
       sprintf(fieldName,"u");
       fieldIndex = 1;
       errorCode = hydro_coreReadFieldBndyPlanes(ncid, fieldName, fieldIndex);
       sprintf(fieldName,"v");
       fieldIndex = 2;
       errorCode = hydro_coreReadFieldBndyPlanes(ncid, fieldName,fieldIndex);
       sprintf(fieldName,"w");
       fieldIndex = 3;
       errorCode = hydro_coreReadFieldBndyPlanes(ncid, fieldName, fieldIndex);
       sprintf(fieldName,"theta");
       fieldIndex = 4;
       errorCode = hydro_coreReadFieldBndyPlanes(ncid, fieldName, fieldIndex);
       if(moistureSelector > 0){
         if(moistureNvars > 0){
           sprintf(fieldName,"qv");
           fieldIndex = 5;
           errorCode = hydro_coreReadFieldBndyPlanes(ncid, fieldName, fieldIndex);
         }
         if(moistureNvars > 1){
           sprintf(fieldName,"ql");
           fieldIndex = 6;
           errorCode = hydro_coreReadFieldBndyPlanes(ncid, fieldName, fieldIndex);
         }
         if(moistureNvars > 2){
           sprintf(fieldName,"qr");
           fieldIndex = 7;
           errorCode = hydro_coreReadFieldBndyPlanes(ncid, fieldName, fieldIndex);
         }
       }//end if moistureSelector > 0
       if(surflayerSelector == 3){     //Get any expected surface variable planes (skin-fields, e.g. tskin,qskin)
         sprintf(fieldName,"tskin");
         fieldIndex = nBndyVars+0;  //first of nSurfBndyVars = 1 or 2
         errorCode = hydro_coreReadFieldBndyPlanes(ncid, fieldName, fieldIndex);
         if(moistureSelector > 0){
           sprintf(fieldName,"qskin");
           fieldIndex = nBndyVars+1;  //second of nSurfBndyVars = 2
           errorCode = hydro_coreReadFieldBndyPlanes(ncid, fieldName, fieldIndex);
         }
       }
       printf("Done Reading Bndy-variables from hydroBndysFile = %s\n",hydroBndysFile);
       fflush(stdout);
       /* close the file */
       errorCode = ioCloseNetCDFfile(ncid);
     } //end if(mpi_rank_world == 0)
   } //end if a hydroBndysFile was defined
   printf("%d/%d: Waiting on root-rank reading Bndy-variables from hydroBndysFile = %s\n",mpi_rank_world,mpi_size_world,hydroBndysFile);
   fflush(stdout);
   MPI_Barrier(MPI_COMM_WORLD);

   //Fields have been read, scatter them across ranks
   errorCode = hydro_coreScatterFieldBndyPlanes(nBndyVars);

   /*Cycle the BdyPlane pointers*/
   XZBdyPlanesPrev = XZBdyPlanes;   //Prev are a temp pointer to the last used memory blocks
   YZBdyPlanesPrev = YZBdyPlanes;
   XYBdyPlanesPrev = XYBdyPlanes;
   SURFBdyPlanesPrev = SURFBdyPlanes;
   XZBdyPlanes = XZBdyPlanesNext;   //Move the last used to point at the Bdy planes that were just read in
   YZBdyPlanes = YZBdyPlanesNext;
   XYBdyPlanes = XYBdyPlanesNext;
   SURFBdyPlanes = SURFBdyPlanesNext;
   XZBdyPlanesNext = XZBdyPlanesPrev; //Next pointers inow use the last used memory blocks to read next time 
   YZBdyPlanesNext = YZBdyPlanesPrev;
   XYBdyPlanesNext = XYBdyPlanesPrev;
   SURFBdyPlanesNext = SURFBdyPlanesPrev;
   
   return(errorCode);
}//end hydro_coreSetupBndyPlanesAllRanks()

/*----->>>>> int hydro_coreReadNextBndyPlanesFile();   ----------------------------------------------------
* Utility to increment the BdyPlanes files counter and invoke hydro_coreSetupBndyPlanesAllRanks()
* to read/scatter the next set of BdyPlanes in the series
*/
int hydro_coreReadNextBndyPlanesFile(){
   int errorCode = HYDRO_CORE_SUCCESS;

   hydroBndysFileCounter=hydroBndysFileCounter + 1;  //Increment the counter
   sprintf(hydroBndysFile,"%s.%d",hydroBndysFileBase,hydroBndysFileCounter);
   printf("mpi_rank_world--%d/%d: hydro_coreReadNextBndyPlanesFile() on %s.\n",
          mpi_rank_world,mpi_size_world,hydroBndysFile);
   fflush(stdout);
   errorCode = hydro_coreSetupBndyPlanesAllRanks();
   printf("mpi_rank_world--%d/%d: hydro_coreReadNextBndyPlanesFile() on %s complete.\n",
          mpi_rank_world,mpi_size_world,hydroBndysFile);
   fflush(stdout);
   return(errorCode);
}// end hydro_coreReadNextBndyPlanesFile()

/*----->>>>> int hydro_coreReadFieldBndyPlanes();   ---------------------------------------------------
*
*/
int hydro_coreReadFieldBndyPlanes(int ncid, char* field, int fieldNum){
   int errorCode = HYDRO_CORE_SUCCESS;
   int ncbndyfldid;

   int dimids[16];
   size_t count[16];
   size_t countYZ[16];
   size_t countXZ[16];
   size_t countXY[16];
   size_t *countPtr;
   size_t start[16];

   int i,j,k;
   int fldBaseYZ;
   int fldBaseXZ;
   int fldBaseXY;

   char varName[64];

   printf("%d/%d: Working on the field --> %s\n",mpi_rank_world,mpi_size_world,field);
   fflush(stdout);
   /* Boundary Plane Conditions */
   if(hydroBCs==1){    //If limited-area domain with 5-BC-planes for rho,u,v,w,theta+TKE(s)+moist(s)
     if(hydroBndysFile != NULL){ /*If a Bndys file exists read it, else log an error.*/
       if(mpi_rank_world == 0){
         /* Inquire for the dimension-ids*/
         if ((errorCode = nc_inq_dimid(ncid, "time", &dimids[0]))){
           ERR(errorCode);
         }
         if ((errorCode = nc_inq_dimid(ncid, "zIndex", &dimids[1]))){
           ERR(errorCode);
         }
         if ((errorCode = nc_inq_dimid(ncid, "yIndex", &dimids[2]))){
           ERR(errorCode);
         }
         if ((errorCode = nc_inq_dimid(ncid, "xIndex", &dimids[3]))){
           ERR(errorCode);
         }
#ifdef DEBUG
         printf("Established BndyFile data dimension ids of xIndex,yIndex,zIndex,time = %d, %d, %d, %d\n",
                dimids[3],dimids[2],dimids[1],dimids[0]);
#endif
         if ((errorCode = nc_inq_dimlen(ncid, dimids[0], &count[dimids[0]]))){
           ERR(errorCode);
         }
         if ((errorCode = nc_inq_dimlen(ncid, dimids[1], &count[dimids[1]]))){
           ERR(errorCode);
         }
         if ((errorCode = nc_inq_dimlen(ncid, dimids[2], &count[dimids[2]]))){
           ERR(errorCode);
         }
         if ((errorCode = nc_inq_dimlen(ncid, dimids[3], &count[dimids[3]]))){
           ERR(errorCode);
         }
#ifdef DEBUG
         printf("Established BndyFile data dimensions of xIndex,yIndex,zIndex,time = %lu, %lu, %lu, %lu\n",
                 count[dimids[3]],count[dimids[2]],count[dimids[1]],count[dimids[0]]);
#endif
         start[dimids[0]] = 0;   //Take the first record
         start[dimids[1]] = 0;
         start[dimids[2]] = 0;
         start[dimids[3]] = 0;

         if(fieldNum < nBndyVars){   //Non-surface variable Bdy planes (i.e. rho,u,v,w,theta,TKE_0,TKE_1,qv,ql)
         //YZ planes
         countYZ[0] = 1;
         countYZ[1] = count[dimids[1]];
         countYZ[2] = count[dimids[2]];
         countPtr=countYZ;
         fldBaseYZ = 2*Ny*Nz*fieldNum;
         sprintf(varName,"%s_YZL",field);
         if ( (errorCode = nc_inq_varid(ncid, varName, &ncbndyfldid)) ){
             ERR(errorCode);
             printf("Error hydro_coreReadFieldBndyPlane(): Variable field = %s was not found in this file,!\n",varName);
             fflush(stdout);
         } //if nc_inq_varid
#ifdef DEBUG
         if ((errorCode = nc_inq_varndims(ncid, ncbndyfldid, &nDimsBndy))){
              ERR(errorCode);
         }
         printf("Variable field = %s has nDims = %d\n",varName,nDimsBndy);
         if ((errorCode = nc_inq_vardimid(ncid, ncbndyfldid, tmpDimids))){
            ERR(errorCode);
         }
#endif
         if ((errorCode = nc_get_vara_float(ncid, ncbndyfldid, start, countPtr, &YZBdyPlanesGlobal[fldBaseYZ+0] ))){
                ERR(errorCode);
         }
         sprintf(varName,"%s_YZH",field);
         if ( (errorCode = nc_inq_varid(ncid, varName, &ncbndyfldid)) ){
             ERR(errorCode);
             printf("Error hydro_coreSetupBndyPlanesAllRanks(): Variable field = %s was not found in this file,!\n",
                    varName);
             fflush(stdout);
         }
         if ((errorCode = nc_get_vara_float(ncid, ncbndyfldid, start, countPtr, &YZBdyPlanesGlobal[fldBaseYZ+Ny*Nz] ))){
                ERR(errorCode);
         }
         //If fieldNum == 1,2,3, or 4 muliply by rho for flux conservative form of the field quantity
         if((fieldNum > 0) && (fieldNum < nBndyVars)){
           for(j=0; j < Ny; j++){
             for(k=0; k < Nz; k++){
               YZBdyPlanesGlobal[fldBaseYZ+j*Nz+k] =  YZBdyPlanesGlobal[fldBaseYZ+j*Nz+k] //low-sided Bndy
                                                     *YZBdyPlanesGlobal[0+j*Nz+k];  //This is the factor of rho
               YZBdyPlanesGlobal[fldBaseYZ+Ny*Nz+j*Nz+k] =  YZBdyPlanesGlobal[fldBaseYZ+Ny*Nz+j*Nz+k] //high-sided Bndy
                                                     *YZBdyPlanesGlobal[0+Ny*Nz+j*Nz+k];  //This is the factor of rho
             } //end for k
           } // end for j
         } //end if fieldNum = 1,2,3 or 4

         //XZ planes
         countXZ[0] = 1;
         countXZ[1] = count[dimids[1]];
         countXZ[2] = count[dimids[3]];
         countPtr=countXZ;
         fldBaseXZ = 2*Nx*Nz*fieldNum;
         sprintf(varName,"%s_XZL",field);
         if ( (errorCode = nc_inq_varid(ncid, varName, &ncbndyfldid)) ){
             ERR(errorCode);
             printf("Error hydro_coreSetupBndyPlanesAllRanks(): Variable field = %s was not found in this file,!\n",
                    varName);
             fflush(stdout);
         }
         if ((errorCode = nc_get_vara_float(ncid, ncbndyfldid, start, countPtr, &XZBdyPlanesGlobal[fldBaseXZ+0] ))){
                ERR(errorCode);
         }
         sprintf(varName,"%s_XZH",field);
         if ( (errorCode = nc_inq_varid(ncid, varName, &ncbndyfldid)) ){
             ERR(errorCode);
             printf("Error hydro_coreSetupBndyPlanesAllRanks(): Variable field = %s was not found in this file,!\n",
                    varName);
             fflush(stdout);
         }
         if ((errorCode = nc_get_vara_float(ncid, ncbndyfldid, start, countPtr, &XZBdyPlanesGlobal[fldBaseXZ+Nx*Nz] ))){
                ERR(errorCode);
         }
         //If fieldNum == 1,2,3, or 4 muliply by rho for flux conservative form of the field quantity
         if((fieldNum > 0) && (fieldNum < nBndyVars)){
           for(i=0; i < Nx; i++){
             for(k=0; k < Nz; k++){
               XZBdyPlanesGlobal[fldBaseXZ+i*Nz+k] =  XZBdyPlanesGlobal[fldBaseXZ+i*Nz+k] //low-sided Bndy
                                                     *XZBdyPlanesGlobal[0+i*Nz+k];  //This is the factor of rho
               XZBdyPlanesGlobal[fldBaseXZ+Nx*Nz+i*Nz+k] =  XZBdyPlanesGlobal[fldBaseXZ+Nx*Nz+i*Nz+k] //high-sided Bndy
                                                     *XZBdyPlanesGlobal[0+Nx*Nz+i*Nz+k];  //This is the factor of rho
             } //end for k
           } // end for j
         } //end if fieldNum = 1,2,3 or 4


         //XY planes
         countXY[0] = 1;
         countXY[1] = count[dimids[2]];
         countXY[2] = count[dimids[3]];
         countPtr=countXY;
         fldBaseXY = 2*Nx*Ny*fieldNum;
         sprintf(varName,"%s_XYL",field);
         if ( (errorCode = nc_inq_varid(ncid, varName, &ncbndyfldid)) ){
             ERR(errorCode);
             printf("Error hydro_coreSetupBndyPlanesAllRanks(): Variable field = %s was not found in this file,!\n",
                    varName);
             fflush(stdout);
         }
         if ((errorCode = nc_get_vara_float(ncid, ncbndyfldid, start, countPtr, &XYBdyPlanesGlobal[fldBaseXY+0] ))){
                ERR(errorCode);
         }
         sprintf(varName,"%s_XYH",field);
         if ( (errorCode = nc_inq_varid(ncid, varName, &ncbndyfldid)) ){
             ERR(errorCode);
             printf("Error hydro_coreSetupBndyPlanesAllRanks(): Variable field = %s was not found in this file,!\n",
                    varName);
             fflush(stdout);
         }
         if ((errorCode = nc_get_vara_float(ncid, ncbndyfldid, start, countPtr, &XYBdyPlanesGlobal[fldBaseXY+Nx*Ny] ))){
                ERR(errorCode);
         }
         //If fieldNum == 1,2,3, or 4 muliply by rho for flux conservative form of the field quantity
         if((fieldNum > 0) && (fieldNum < nBndyVars)){   //moist variables assumed to be rho_q(v,l) for now
           for(i=0; i < Nx; i++){
             for(j=0; j < Ny; j++){
               XYBdyPlanesGlobal[fldBaseXY+i*Ny+j] =  XYBdyPlanesGlobal[fldBaseXY+i*Ny+j] //low-sided Bndy
                                                     *XYBdyPlanesGlobal[0+i*Ny+j];  //This is the factor of rho
               XYBdyPlanesGlobal[fldBaseXY+Nx*Ny+i*Ny+j] =  XYBdyPlanesGlobal[fldBaseXY+Nx*Ny+i*Ny+j] //high-sided Bndy
                                                     *XYBdyPlanesGlobal[0+Nx*Ny+i*Ny+j];  //This is the factor of rho
             } //end for k
           } // end for j
         } //end if fieldNum = 1,2,3 or 4

//#ifdef BDYPLANE_DEBUG    
#if 1
         printf("Bdy_plane[0] = %f, Bdy_plane[Ny*Nz-1] = %f\n",YZBdyPlanesGlobal[fldBaseYZ+0],YZBdyPlanesGlobal[fldBaseYZ+Ny*Nz-1]);
         printf("Bdy_plane[Ny*Nz] = %f, Bdy_plane[2*Ny*Nz-1] = %f\n",YZBdyPlanesGlobal[fldBaseYZ+Ny*Nz],YZBdyPlanesGlobal[fldBaseYZ+2*Ny*Nz-1]);

         printf("Bdy_plane[0] = %f, Bdy_plane[Nx*Nz-1] = %f\n",XZBdyPlanesGlobal[fldBaseXZ+0],XZBdyPlanesGlobal[fldBaseXZ+Nx*Nz-1]);
         printf("Bdy_plane[Nx*Nz] = %f, Bdy_plane[2*Nx*Nz-1] = %f\n",XZBdyPlanesGlobal[fldBaseXZ+Nx*Nz],XZBdyPlanesGlobal[fldBaseXZ+2*Nx*Nz-1]);

         printf("Bdy_plane[0] = %f, Bdy_plane[Nx*Ny-1] = %f\n",XYBdyPlanesGlobal[fldBaseXY+0],XYBdyPlanesGlobal[fldBaseXY+Nx*Ny-1]);
         printf("Bdy_plane[Nx*Ny] = %f, Bdy_plane[2*Nx*Ny-1] = %f\n",XYBdyPlanesGlobal[fldBaseXY+Nx*Ny],XYBdyPlanesGlobal[fldBaseXY+2*Nx*Ny-1]);
         fflush(stdout);
#endif

         } else if(fieldNum >= nBndyVars){   //Surface variable Bdy planes (i.e. tskin, qskin) 
           //SURF planes (XY-shaped)
           countXY[0] = 1;
           countXY[1] = count[dimids[2]];
           countXY[2] = count[dimids[3]];
           countPtr=countXY;
           fldBaseXY = Nx*Ny*(fieldNum-nBndyVars);
           sprintf(varName,"%s",field);
           if ( (errorCode = nc_inq_varid(ncid, varName, &ncbndyfldid)) ){
               ERR(errorCode);
               printf("Error hydro_coreSetupBndyPlanesAllRanks(): Variable field = %s was not found in this file,!\n",
                    varName);
               fflush(stdout);
           }
           if ((errorCode = nc_get_vara_float(ncid, ncbndyfldid, start, countPtr, &SURFBdyPlanesGlobal[fldBaseXY] ))){
                ERR(errorCode);
           }
         } //if(fieldNum >= nBndyVars)  //For surface variable Bdy Planes

       } //end if(mpi_rank_world == 0)
     } //end if a hydroBndysFile was defined
   }//end if hydroBCs == 1
   printf("%d/%d hydro_coreReadFieldBndyPlanes: Done, returning!\n",mpi_rank_world,mpi_size_world);
   fflush(stdout);
   return(errorCode);
}//end hydro_coreReadFieldBndyPlanes()

/*----->>>>> int hydro_coreScatterFieldBndyPlanes();   ---------------------------------------------------
* 
*/
int hydro_coreScatterFieldBndyPlanes(int Nfields){
   int errorCode = HYDRO_CORE_SUCCESS;
   int i,j,k;
   int ij,ik,jk;
   int iGlobal,jGlobal,kGlobal;
   int ijGlobal,ikGlobal,jkGlobal;
   int iFld; //simple integer index for the ith Fld in the hydro_core memory block, hydroFlds.
   int globalfldBaseYZ;
   int globalfldBaseXZ;
   int globalfldBaseXY;
   int fldBaseYZ;
   int fldBaseXZ;
   int fldBaseXY;
   int globalYZstride;
   int globalXZstride;
   int globalXYstride;
   int YZstride;
   int XZstride;
   int XYstride;

   globalYZstride = Ny*Nz;
   globalXZstride = Nx*Nz;
   globalXYstride = Nx*Ny;
   YZstride = (Nyp+2*Nh)*(Nzp+2*Nh);
   XZstride = (Nxp+2*Nh)*(Nzp+2*Nh);
   XYstride = (Nxp+2*Nh)*(Nyp+2*Nh);

   /* Broadcast the global domain BDY planes to all ranks for simplicity
    * If necessary one could selectively communicate strictly with necessary mpi_ranks
    * However, BDY-planes are for now assumed to be modest in terms of memory requirements.
   */
   MPI_Bcast(XZBdyPlanesGlobal, 2*(nBndyVars)*Nx*Nz, MPI_FLOAT, 0, MPI_COMM_WORLD);
   MPI_Bcast(YZBdyPlanesGlobal, 2*(nBndyVars)*Ny*Nz, MPI_FLOAT, 0, MPI_COMM_WORLD);
   MPI_Bcast(XYBdyPlanesGlobal, 2*(nBndyVars)*Nx*Ny, MPI_FLOAT, 0, MPI_COMM_WORLD);
   if(surflayerSelector==3){
     MPI_Bcast(SURFBdyPlanesGlobal, (nSurfBndyVars)*Nx*Ny, MPI_FLOAT, 0, MPI_COMM_WORLD);
   }

   printf("hydro_coreScatterFieldBndyPlanes(): broadcasts complete \n");
   fflush(stdout);

   for(iFld = 0; iFld < nBndyVars; iFld ++){  //{rho, u, v, w, theta, z}
     if(hydroBCs==1){    //If limited-area domain with 5-BC-planes for rho,u,v,w,theta+TKE(s)+moist(s)
       if(hydroBndysFile != NULL){ /*If a Bndys file exists read it, else log an error.*/
         /*Create mpi_rank local bdy plane copies as needed for mpi_rank subdomains*/
         if(rankXid == 0){ //Western BDY plane
           globalfldBaseYZ = 2*iFld*globalYZstride; // shape = 2*2*(Nfields)*Ny*Nz 
           fldBaseYZ = 2*iFld*YZstride;
           for(j=jMin-Nh; j < jMax+Nh; j++){
             for(k=kMin-Nh; k < kMax+Nh; k++){
              jk = (j)*(Nzp+2*Nh)+k;
              kGlobal = k-Nh;
              if(kGlobal < 0){
                kGlobal=0;
              }
              if(kGlobal > Nz-1){
                kGlobal=Nz-1;
              }
              jGlobal = rankYid*(Nyp)+j-Nh;
              if(jGlobal < 0){
                jGlobal=0;
              }
              if(jGlobal > Ny-1){
                jGlobal=Ny-1;
              }
              jkGlobal = kGlobal*Ny+jGlobal;
              YZBdyPlanesNext[fldBaseYZ + jk] = YZBdyPlanesGlobal[globalfldBaseYZ + jkGlobal];
#ifdef DEBUG_YZPLANES
              //if((j==jMin)&&(k==kMin)){
              if((j==jMin)||(j==jMax-1)){
                printf("%d/%d hydro_coreScatterFieldBndyPlanes(): iFld=%d, at (%d,%d), YZBdyPlanesGlobal[%d + %d] = %f and at (%d,%d), YZBdyPlanes[%d+%d] = %f \n",
                        mpi_rank_world,mpi_size_world,iFld,jGlobal,kGlobal,globalfldBaseYZ,jkGlobal,YZBdyPlanesGlobal[globalfldBaseYZ + jkGlobal],
                        j,k,fldBaseYZ,jk,YZBdyPlanesNext[fldBaseYZ+jk]);
              }//end if j==jMin || j==jMax-1
#endif
             } //end for k
           } //end for j
         } //end if rankXid == 0  //Western BDY plane
         if(rankXid == numProcsX-1){ //Eastern BDY plane
           globalfldBaseYZ = (2*iFld+1)*globalYZstride;
           fldBaseYZ = (2*iFld+1)*YZstride;
           for(j=jMin-Nh; j < jMax+Nh; j++){
             for(k=kMin-Nh; k < kMax+Nh; k++){
              jk = (j)*(Nzp+2*Nh)+k;
              kGlobal = k-Nh;
              if(kGlobal < 0){
                kGlobal=0;
              }
              if(kGlobal > Nz-1){
                kGlobal=Nz-1;
              }
              jGlobal = rankYid*(Nyp)+j-Nh;
              if(jGlobal < 0){
                jGlobal=0;
              }
              if(jGlobal > Ny-1){
                jGlobal=Ny-1;
              }
              jkGlobal = kGlobal*Ny+jGlobal;
              YZBdyPlanesNext[fldBaseYZ + jk] = YZBdyPlanesGlobal[globalfldBaseYZ + jkGlobal];
#ifdef DEBUG_YZPLANES
              //if((j==jMin)&&(k==kMin)){
              if((j==jMin)||(j==jMax-1)){
                printf("%d/%d hydro_coreScatterFieldBndyPlanes(): iFld=%d, at (%d,%d), YZBdyPlanesGlobal[%d + %d] = %f and at (%d,%d), YZBdyPlanes[%d+%d] = %f \n",
                        mpi_rank_world,mpi_size_world,iFld,jGlobal,kGlobal,globalfldBaseYZ,jkGlobal,YZBdyPlanesGlobal[globalfldBaseYZ + jkGlobal],
                        j,k,fldBaseYZ,jk,YZBdyPlanesNext[fldBaseYZ+jk]);
              }//end if j==jMin || j==jMax-1 
#endif
             } //end for k
           } //end for j
         } //end if rankXid == numProcsX-1  //Eastern BDY plane
         if(rankYid == 0){ //Southern BDY plane
           globalfldBaseXZ = 2*iFld*globalXZstride;
           fldBaseXZ = 2*iFld*XZstride;
           for(i=iMin-Nh; i < iMax+Nh; i++){
             for(k=kMin-Nh; k < kMax+Nh; k++){
              ik = (i)*(Nzp+2*Nh)+k;
              kGlobal = k-Nh;
              if(kGlobal < 0){
                kGlobal=0;
              }
              if(kGlobal > Nz-1){
                kGlobal=Nz-1;
              }
              iGlobal = rankXid*Nxp+i-Nh;
              if(iGlobal < 0){
                iGlobal=0;
              }
              if(iGlobal > Nx-1){
                iGlobal=Nx-1;
              }
              ikGlobal = kGlobal*Nx+iGlobal;
              XZBdyPlanesNext[fldBaseXZ + ik] = XZBdyPlanesGlobal[globalfldBaseXZ + ikGlobal];
             } //end for k
           } //end for i
         }  //end if rankYid == 0 //Southern BDY plane
         if(rankYid == numProcsY-1){ //Northern BDY plane
           globalfldBaseXZ = (2*iFld+1)*globalXZstride;
           fldBaseXZ = (2*iFld+1)*XZstride;
           for(i=iMin-Nh; i < iMax+Nh; i++){
             for(k=kMin-Nh; k < kMax+Nh; k++){
              ik = (i)*(Nzp+2*Nh)+k;
              kGlobal = k-Nh;
              if(kGlobal < 0){
                kGlobal=0;
              }
              if(kGlobal > Nz-1){
                kGlobal=Nz-1;
              }
              iGlobal = rankXid*Nxp+i-Nh;
              if(iGlobal < 0){
                iGlobal=0;
              }
              if(iGlobal > Nx-1){
                iGlobal=Nx-1;
              }
              ikGlobal = kGlobal*Nx+iGlobal;
              XZBdyPlanesNext[fldBaseXZ + ik] = XZBdyPlanesGlobal[globalfldBaseXZ + ikGlobal];
#ifdef DEBUG_XZPLANES
              if((i==iMin)||(i==iMax-1)){
                printf("%d/%d: hydro_coreScatterFieldBndyPlanes(): iFld=%d, at (%d,%d), XZBdyPlanesGlobal[%d + %d] = %f and at (%d,%d), XZBdyPlanes[%d+%d] = %f \n",
                       mpi_rank_world,mpi_size_world,iFld,iGlobal,kGlobal,globalfldBaseXZ,ikGlobal,XZBdyPlanesGlobal[globalfldBaseXZ + ikGlobal],
                       i,k,fldBaseXZ,ik,XZBdyPlanesNext[fldBaseXZ+ik]);
              }//end if i==iMin || i==iMax-1 
#endif
             } //end for k
           } //end for i
         } //end if rankYid == numProcsY-1 //Northern BDY plane
         //Finally set the ceiling boundary conditions
         globalfldBaseXY = (2*iFld+1)*globalXYstride;
         fldBaseXY = (2*iFld+1)*XYstride;
         for(i=iMin-Nh; i < iMax+Nh; i++){
           for(j=jMin-Nh; j < jMax+Nh; j++){
              ij = (i)*(Nyp+2*Nh)+j;
              iGlobal = rankXid*Nxp+i-Nh;
              if(iGlobal < 0){
                iGlobal=0;
              }
              if(iGlobal > Nx-1){
                iGlobal=Nx-1;
              }
              jGlobal = rankYid*Nyp+j-Nh;
              if(jGlobal < 0){
                jGlobal=0;
              }
              if(jGlobal > Ny-1){
                jGlobal=Ny-1;
              }
              ijGlobal = jGlobal*Nx+iGlobal;    //Note the implied transpose
              XYBdyPlanesNext[fldBaseXY + ij] = XYBdyPlanesGlobal[globalfldBaseXY + ijGlobal];
           } //end for i
         } //end for j
       } //end if a hydroBndysFile was defined
     }//end if hydroBCs == 1
   }//end for iFld 
   if(surflayerSelector ==3){
     for(iFld = 0; iFld < nSurfBndyVars; iFld ++){  //{tskin,qskin}
        //Finally set the ceiling boundary conditions
        globalfldBaseXY = (iFld)*globalXYstride;   //SurfBdys-planes are only low-sided so seperated strictly per field rather than 2*per field
        fldBaseXY = (iFld)*XYstride;
        for(i=iMin-Nh; i < iMax+Nh; i++){
           for(j=jMin-Nh; j < jMax+Nh; j++){
              ij = (i)*(Nyp+2*Nh)+j;
              iGlobal = rankXid*Nxp+i-Nh;
              if(iGlobal < 0){
                iGlobal=0;
              }
              if(iGlobal > Nx-1){
                iGlobal=Nx-1;
              }
              jGlobal = rankYid*Nyp+j-Nh;
              if(jGlobal < 0){
                jGlobal=0;
              }
              if(jGlobal > Ny-1){
                jGlobal=Ny-1;
              }
              ijGlobal = jGlobal*Nx+iGlobal;    //Note the implied transpose
              SURFBdyPlanesNext[fldBaseXY + ij] = SURFBdyPlanesGlobal[globalfldBaseXY + ijGlobal];
           } //end for i
        } //end for j
     }//end for iFld 
   } //end if surflayerSelector
   printf("hydro_coreScatterFieldBndyPlanes(): Scatters to local bndy-array complete \n");
   fflush(stdout);
   return(errorCode);
}//end hydro_coreScatterFieldBndyPlanes()

/*----->>>>> int hydro_coreTVCP();  -----------------------------------------------------------
* Updates model parameters used by the CELLPERT submodule from dynamic lateral BNDY conditions.
*/
int hydro_coreTVCP(float dt){
    int errorCode = HYDRO_CORE_SUCCESS;

    int bdy_id; // 0==North; 1==East; 2==South; 3==West;
    float cellpert_amp_array[4];
    int cellpert_ktop_array[4];
    int cellpert_nts_array[4];
    float cellpert_amp_aver;
    int cellpert_ktop_aver;
    int cellpert_nts_aver;

    /*--- North boundary ---*/
    bdy_id = 0;
    hydro_coreTVCP_LBCparams(bdy_id, XZBdyPlanesGlobal, cellpert_amp_array, cellpert_ktop_array, cellpert_ktop_prev, cellpert_nts_array, dt);
    /*--- East boundary ---*/
    bdy_id = 1;
    hydro_coreTVCP_LBCparams(bdy_id, YZBdyPlanesGlobal, cellpert_amp_array, cellpert_ktop_array, cellpert_ktop_prev, cellpert_nts_array, dt);
    /*--- South boundary ---*/
    bdy_id = 2;
    hydro_coreTVCP_LBCparams(bdy_id, XZBdyPlanesGlobal, cellpert_amp_array, cellpert_ktop_array, cellpert_ktop_prev, cellpert_nts_array, dt);
    /*--- West boundary ---*/
    bdy_id = 3;
    hydro_coreTVCP_LBCparams(bdy_id, YZBdyPlanesGlobal, cellpert_amp_array, cellpert_ktop_array, cellpert_ktop_prev, cellpert_nts_array, dt);

    cellpert_amp_aver = 0.25*(cellpert_amp_array[0]+cellpert_amp_array[1]+cellpert_amp_array[2]+cellpert_amp_array[3]);
    cellpert_ktop_aver = 0.25*(cellpert_ktop_array[0]+cellpert_ktop_array[1]+cellpert_ktop_array[2]+cellpert_ktop_array[3]);
    cellpert_nts_aver = 0.25*(cellpert_nts_array[0]+cellpert_nts_array[1]+cellpert_nts_array[2]+cellpert_nts_array[3]);
    cellpert_amp = cellpert_amp_aver;
    cellpert_ktop = cellpert_ktop_aver;
    cellpert_nts = cellpert_nts_aver;

    cellpert_ktop_prev[0] = cellpert_ktop;
    cellpert_ktop_prev[1] = cellpert_ktop;
    cellpert_ktop_prev[2] = cellpert_ktop;
    cellpert_ktop_prev[3] = cellpert_ktop;

//#define TVCP_DEBUG
#ifdef TVCP_DEBUG
     printf("cellpert_amp_array[%d] = %f \n",0,cellpert_amp_array[0]);
     printf("cellpert_amp_array[%d] = %f \n",1,cellpert_amp_array[1]);
     printf("cellpert_amp_array[%d] = %f \n",2,cellpert_amp_array[2]);
     printf("cellpert_amp_array[%d] = %f \n",3,cellpert_amp_array[3]);
     printf("cellpert_ktop_array[%d] = %d \n",0,cellpert_ktop_array[0]);
     printf("cellpert_ktop_array[%d] = %d \n",1,cellpert_ktop_array[1]);
     printf("cellpert_ktop_array[%d] = %d \n",2,cellpert_ktop_array[2]);
     printf("cellpert_ktop_array[%d] = %d \n",3,cellpert_ktop_array[3]);
     printf("cellpert_nts_array[%d] = %d \n",0,cellpert_nts_array[0]);
     printf("cellpert_nts_array[%d] = %d \n",1,cellpert_nts_array[1]);
     printf("cellpert_nts_array[%d] = %d \n",2,cellpert_nts_array[2]);
     printf("cellpert_nts_array[%d] = %d \n",3,cellpert_nts_array[3]);
#endif
if(mpi_rank_world == 0){
     printf("---------------------- \n");
     printf("Updating CP parameters \n");
     printf("---------------------- \n");
     printf("cellpert_amp = %f K \n",cellpert_amp);
     printf("cellpert_ktop = %d grid points \n",cellpert_ktop);
     printf("cellpert_nts = %d time steps \n",cellpert_nts);
     fflush(stdout);
}

    return(errorCode);
}//end hydro_coreTVCP()

/*----->>>>> int hydro_coreTVCP_LBCparams();  -----------------------------------------------------------
* Computes model parameters used by the CELLPERT submodule from dynamic lateral BNDY conditions.
*/
int hydro_coreTVCP_LBCparams(int bdy_id, float* var_LBCplane, float* cellpert_amp_array,
                             int* cellpert_ktop_array, int* cellpert_ktop_prev, int* cellpert_nts_array, float dt){
    int errorCode = HYDRO_CORE_SUCCESS;
    int bndyfldStride_LZ;
    int Nl, k, l, ind_lk;
    int bdy_lh;
    int ku_s, kv_s, krho_s, kth_s;
    float u_tmp, v_tmp, rho_tmp, th_tmp, ws_tmp;
    float uz_tmp[Nz];
    float thz_tmp[Nz];
    float ws_tmp_2, ws_diff, ws_abl;
    float th_min, th_diff, th_diff_2;
    float ntsf_tmp;
    int ku_max;
    int kth_zi;
    int kzi_tmp, kzi_tmp_old;
    int k_low = 3;
    float dzidt_tend = 0.05;
    int dzidt_max;
    float dzi_tend_sign;

    if(bdy_id==0 || bdy_id==2){ // N or S boundary planes
      Nl = Nx;
    }else{ // E or W boundary planes
      Nl = Ny;
    }
    bndyfldStride_LZ = Nl*Nz;

    bdy_lh = 0;  //Initialize for safety
    if(bdy_id == 0){ // N
      bdy_lh = 1;
    } else if(bdy_id == 1){ // E
      bdy_lh = 1;
    } else if(bdy_id == 2){ // S
      bdy_lh = 0;
    } else if(bdy_id == 3){ // W
      bdy_lh = 0;
    }


    //printf("bdy_id = %d Nl = %d \n",bdy_id, Nl);
    //    //fflush(stdout);

    ku_s = (U_INDX*2+bdy_lh)*bndyfldStride_LZ;
    kv_s = (V_INDX*2+bdy_lh)*bndyfldStride_LZ;
    krho_s = (RHO_INDX*2+bdy_lh)*bndyfldStride_LZ;
    kth_s = (THETA_INDX*2+bdy_lh)*bndyfldStride_LZ;

    for(k=0; k < Nz; k++){
       uz_tmp[k] = 0.0;
       thz_tmp[k] = 0.0;
       for(l=0; l < Nl; l++){
         ind_lk = l*Nz+k;
         u_tmp = var_LBCplane[ku_s+ind_lk];
         v_tmp = var_LBCplane[kv_s+ind_lk];
         rho_tmp = var_LBCplane[krho_s+ind_lk];
         th_tmp = var_LBCplane[kth_s+ind_lk];

         u_tmp = u_tmp/rho_tmp;
         v_tmp = v_tmp/rho_tmp;
         th_tmp = th_tmp/rho_tmp;
         ws_tmp = sqrt(pow(u_tmp,2.0)+pow(v_tmp,2.0));

         uz_tmp[k] = uz_tmp[k] + ws_tmp;
         thz_tmp[k] = thz_tmp[k] + th_tmp;
       }
       uz_tmp[k] = uz_tmp[k]/(float)(Nl);
       thz_tmp[k] = thz_tmp[k]/(float)(Nl);
     }

     ws_tmp = 0.0;
     th_min = 500.0;
     for(k=0; k < Nz; k++){
       // wind speed maximum //
       ws_tmp_2 = uz_tmp[k];
       ws_diff = ws_tmp_2 - ws_tmp;
       if(ws_diff > 0.0){
         ku_max = k;
       }
       ws_tmp = ws_tmp_2;
       // min theta //
       th_tmp = thz_tmp[k];
       th_diff = th_tmp - th_min;
       if (th_diff < 0.0){
         th_min = th_tmp;
       }
     }

     // closest level to min theta + 1.5 K //
     th_diff_2 = 500.0;
     for(k=0; k < Nz; k++){
       th_diff = fabs(thz_tmp[k] - (th_min + 1.5));
       if(th_diff < th_diff_2){
         th_diff_2 = th_diff;
         kth_zi = k;
       }
     }
     kzi_tmp = floor(0.5*(ku_max+kth_zi));

     // limit ABL rate of change //
     kzi_tmp_old = cellpert_ktop_prev[bdy_id];
     dzidt_max = dzidt_tend*kzi_tmp_old;
     dzi_tend_sign = copysignf(1.0,kzi_tmp - kzi_tmp_old);

     if(dzi_tend_sign >= 0.0){
       if(kzi_tmp_old+dzidt_max < kzi_tmp){
         kzi_tmp = kzi_tmp_old+dzidt_max;
       }
     } else{
       if(kzi_tmp_old+dzidt_max > kzi_tmp){
         kzi_tmp = kzi_tmp_old-dzidt_max;
       }
     }

     // ABL wind speed //
     ws_abl = 0.0;
     for(k=k_low; k < kzi_tmp; k++){
       ws_abl = ws_abl + uz_tmp[k];
     }
     ws_abl = ws_abl/(float)(kzi_tmp-k_low);

     //printf("k_low = %d kzi_tmp = %d, ws_abl = %f\n",k_low,kzi_tmp,ws_abl);
     //     //fflush(stdout);

     cellpert_ktop_array[bdy_id] = kzi_tmp;
     cellpert_amp_array[bdy_id] = pow(uz_tmp[kzi_tmp],2.0)/(cellpert_eckert*cp_gas);
     ntsf_tmp = (float)(cellpert_gppc*cellpert_ndbc*cellpert_tsfact)*d_xi/ws_abl;
     cellpert_nts_array[bdy_id] = floor(ntsf_tmp/dt);

     return(errorCode);
}//end hydro_coreTVCP_LBCparams()


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
   //Auxiliary Scalars
   for(iFld=0; iFld < NhydroAuxScalars; iFld++){    //Run through (hydroAuxScalars)    
     MPI_Barrier(MPI_COMM_WORLD);
     if((mpi_rank_world == 0)&&(iFld>0)){
       printf("-----\n");
       fflush(stdout);
     } //end if mpi_rank_world==0
     fldBase = &hydroAuxScalars[iFld*fldStride];
     errorCode = sprintf(&fldName[0],"AuxSc_%d",iFld);
     printf("%s\t", &fldName[0]);
     errorCode = hydro_coreFldStateLogEntry(fldBase, 1);
   } //end for iFld
 
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
         MPI_Barrier(MPI_COMM_WORLD);
       } //end for iFld
     }
     //Auxiliary Scalars
     for(iFld=0; iFld < NhydroAuxScalars; iFld++){  //Run through (hydroAuxScalarsFrhs)
       MPI_Barrier(MPI_COMM_WORLD);
       if((mpi_rank_world == 0)&&(iFld>0)){
         printf("-----\n");
         fflush(stdout);
       } //end if mpi_rank_world==0
       fldBase = &hydroAuxScalarsFrhs[iFld*fldStride];
       errorCode = sprintf(&fldName[0],"F_AS_%d",iFld);
       printf("%s\t", &fldName[0]);
       errorCode = hydro_coreFldStateLogEntry(fldBase, 1);
       MPI_Barrier(MPI_COMM_WORLD);
     } //end for iFld
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
/*----->>>>> int srcAuxScConstructor();   ----------------------------------------------------------------------
* This function constructs the srcAuxScalar instance by reading a srcAuxScFile (netCDF) input configuration file,
* allocating CPU-level memory for srcAuxScalar arrays, and initializing these arrays with values specified in 
* the inputs file.
*/
int srcAuxScConstructor(){
  int errorCode = HYDRO_CORE_SUCCESS;
  int ncid;
  int ncfldid;
  int dimids[64];
  size_t count[64];
  size_t start[64];
  char fldName[64];
  int dim0;
  int dim1;

  //Root-rank should read the netcdf turbineSpecsFile
  if(mpi_rank_world == 0){
#define DEBUG_SASCONSTRUCTOR
#ifdef DEBUG_SASCONSTRUCTOR
    printf("Attempting to open srcAuxScFile = %s\n",srcAuxScFile);
    fflush(stdout);
#endif

    //Open the netcdf file
    errorCode = ioOpenNetCDFinFile(srcAuxScFile, &ncid);
    if(errorCode > 0){
      printf("Failed to open srcAuxScFile = %s EXITING NOW!!!!\n",srcAuxScFile);
      fflush(stdout);
      exit(0);
    }
#ifdef DEBUG_SASCONSTRUCTOR
    printf("Opened srcAuxScFile = %s with ncid = %d\n",srcAuxScFile,ncid);
    fflush(stdout);
#endif

    //Inquire for the dimID of the fundamental parameter, NhydroAuxScalars
    if((errorCode = nc_inq_dimid(ncid, "NhydroAuxScalars", &dimids[0]))){
      ERR(errorCode);
    }
    //Inquire for the value of NhydroAuxScalars
    if((errorCode = nc_inq_dimlen(ncid, dimids[0], &count[dimids[0]]))){
      ERR(errorCode);
    }
    //Inquire for the dimID of the fundamental parameter, locDim (should be equal to 3 always!)
    if((errorCode = nc_inq_dimid(ncid, "locDim", &dimids[1]))){
      ERR(errorCode);
    }
    //Inquire for the value of NhydroAuxScalars
    if((errorCode = nc_inq_dimlen(ncid, dimids[1], &count[dimids[1]]))){
      ERR(errorCode);
    }
    dim0 = (int)count[dimids[0]];
    dim1 = (int)count[dimids[1]];
  }//end if mpi_rank == 0 
  MPI_Bcast(&dim0, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);
  MPI_Bcast(&dim1, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);
  /*Assert that the value of the dimension in the file matches the value from the FastEddy parameters file*/
  if(dim0 != NhydroAuxScalars){
    printf("%d/%d: ERROR!! srcAuxScFile dimension (NhydroAuxScalars) %d does not match NhydroAuxScalars=%d from the parameter file. EXITING NOW!!!!\n",mpi_rank_world,mpi_size_world,dim0,NhydroAuxScalars);
    fflush(stdout);
    exit(0);
  }//end if dimid[0] != NhydroAuxScalars 

  if(mpi_rank_world == 0){
    //Read in vector of srcAuxScTemporalTypes (integers)
    sprintf(fldName,"srcAuxScTemporalType");
    start[0] = 0;
    if ( (errorCode = nc_inq_varid(ncid, fldName, &ncfldid)) ){
       ERR(errorCode);
       printf("Error srcAuxScConstructor(): field = %s was not found in this file,!\n",fldName);
       fflush(stdout);
    } //if nc_inq_varid
    if ((errorCode = nc_get_vara_int(ncid, ncfldid, &start[0], &count[dimids[0]], &srcAuxScTemporalType[0] )) ){
       ERR(errorCode);
    }
    //Read in vector of srcAuxScStartSeconds (floats)
    sprintf(fldName,"srcAuxScStartSeconds");
    start[0] = 0;
    if ( (errorCode = nc_inq_varid(ncid, fldName, &ncfldid)) ){
       ERR(errorCode);
       printf("Error srcAuxScConstructor(): field = %s was not found in this file,!\n",fldName);
       fflush(stdout);
    } //if nc_inq_varid
    if ((errorCode = nc_get_vara_float(ncid, ncfldid, &start[0], &count[dimids[0]], &srcAuxScStartSeconds[0] )) ){
       ERR(errorCode);
    }
    //Read in vector of srcAuxScDurationSeconds (floats)
    sprintf(fldName,"srcAuxScDurationSeconds");
    start[0] = 0;
    if ( (errorCode = nc_inq_varid(ncid, fldName, &ncfldid)) ){
       ERR(errorCode);
       printf("Error srcAuxScConstructor(): field = %s was not found in this file,!\n", fldName);
       fflush(stdout);
    } //if nc_inq_varid
    if ((errorCode = nc_get_vara_float(ncid, ncfldid, &start[0], &count[dimids[0]], &srcAuxScDurationSeconds[0] )) ){
       ERR(errorCode);
    }
    //Read in vector of srcAuxScGeometryTypes (integers)
    sprintf(fldName,"srcAuxScGeometryType");
    start[0] = 0;
    if ( (errorCode = nc_inq_varid(ncid, fldName, &ncfldid)) ){
       ERR(errorCode);
       printf("Error srcAuxScConstructor(): field = %s was not found in this file,!\n",fldName);
       fflush(stdout);
    } //if nc_inq_varid
    if ((errorCode = nc_get_vara_int(ncid, ncfldid, &start[0], &count[dimids[0]], &srcAuxScGeometryType[0] )) ){
       ERR(errorCode);
    }
    //Read in vector of srcAuxScLocations (floats)
    sprintf(fldName,"srcAuxScLocation");
    start[0] = 0;
    start[1] = 0;
    if ( (errorCode = nc_inq_varid(ncid, fldName, &ncfldid)) ){
       ERR(errorCode);
       printf("Error srcAuxScConstructor(): field = %s was not found in this file,!\n", fldName);
       fflush(stdout);
    } //if nc_inq_varid
    if ((errorCode = nc_get_vara_float(ncid, ncfldid, &start[0], &count[dimids[0]], &srcAuxScLocation[0] )) ){
       ERR(errorCode);
    }
    //Read in vector of srcAuxScMassSpecTypes (integers)
    sprintf(fldName,"srcAuxScMassSpecType");
    start[0] = 0;
    if ( (errorCode = nc_inq_varid(ncid, fldName, &ncfldid)) ){
       ERR(errorCode);
       printf("Error srcAuxScConstructor(): field = %s was not found in this file,!\n",fldName);
       fflush(stdout);
    } //if nc_inq_varid
    if ((errorCode = nc_get_vara_int(ncid, ncfldid, &start[0], &count[dimids[0]], &srcAuxScMassSpecType[0] )) ){
       ERR(errorCode);
    }
    //Read in vector of srcAuxScMassSpecValues (floats)
    sprintf(fldName,"srcAuxScMassSpecValue");
    start[0] = 0;
    if ( (errorCode = nc_inq_varid(ncid, fldName, &ncfldid)) ){
       ERR(errorCode);
       printf("Error srcAuxScConstructor(): field = %s was not found in this file,!\n", fldName);
       fflush(stdout);
    } //if nc_inq_varid
    if ((errorCode = nc_get_vara_float(ncid, ncfldid, &start[0], &count[dimids[0]], &srcAuxScMassSpecValue[0] )) ){
       ERR(errorCode);
    }
  }//end if mpi_rank == 0
  //Broadcast values to all ranks
  MPI_Bcast(&srcAuxScTemporalType[0], NhydroAuxScalars, MPI_INTEGER, 0, MPI_COMM_WORLD);
  MPI_Bcast(&srcAuxScGeometryType[0], NhydroAuxScalars, MPI_INTEGER, 0, MPI_COMM_WORLD);
  MPI_Bcast(&srcAuxScMassSpecType[0], NhydroAuxScalars, MPI_INTEGER, 0, MPI_COMM_WORLD);
  MPI_Bcast(&srcAuxScStartSeconds[0], NhydroAuxScalars, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&srcAuxScDurationSeconds[0], NhydroAuxScalars, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&srcAuxScLocation[0], NhydroAuxScalars*3, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&srcAuxScMassSpecValue[0], NhydroAuxScalars, MPI_FLOAT, 0, MPI_COMM_WORLD);
#ifdef DEBUG_SASCONSTRUCTOR
  int iRank;
  int iFld;
  int i;
  float *fldPtr;
  int *intfldPtr;
  for(iFld = 0; iFld < 3; iFld++){
    switch (iFld){
      case 0:
        sprintf(fldName,"srcAuxScTemporalType");
        intfldPtr = &srcAuxScTemporalType[0];
        break;
      case 1:
        sprintf(fldName,"srcAuxScGeometryType");
        intfldPtr = &srcAuxScGeometryType[0];
        break;
      case 2:
        sprintf(fldName,"srcAuxScMassSpecType");
        intfldPtr = &srcAuxScMassSpecType[0];
        break;
      default:
        sprintf(fldName,"");
        intfldPtr = NULL;
        break;
    }
    for(iRank = 0; iRank < mpi_size_world; iRank++){
      MPI_Barrier(MPI_COMM_WORLD);
      if(iRank == mpi_rank_world){
        printf("%d/%d %s:--------- \n[",mpi_rank_world,mpi_size_world,fldName);
        for(i=0; i < NhydroAuxScalars; i++){
          printf(" %d ",intfldPtr[i]);
        }
        printf("]\n");
        fflush(stdout);
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }//end iRank
  }//end for iFld
  printf("\n");
  fflush(stdout);
  for(iFld = 0; iFld < 3; iFld++){
    switch (iFld){
      case 0:
        sprintf(fldName,"srcAuxScStartSeconds");
        fldPtr = &srcAuxScStartSeconds[0];
        break;
      case 1:
        sprintf(fldName,"srcAuxScDurationSeconds");
        fldPtr = &srcAuxScDurationSeconds[0];
        break;
      case 2:
        sprintf(fldName,"srcAuxScMassSpecValue");
        fldPtr = &srcAuxScMassSpecValue[0];
        break;
      default:
        sprintf(fldName,"");
        fldPtr = NULL;
        break;
    }
    for(iRank = 0; iRank < mpi_size_world; iRank++){
      MPI_Barrier(MPI_COMM_WORLD);
      if(iRank == mpi_rank_world){
        printf("%d/%d %s:--------- \n[",mpi_rank_world,mpi_size_world,fldName);
        for(i=0; i < NhydroAuxScalars; i++){
          printf(" %f ",fldPtr[i]);
        }
        printf("]\n");
        fflush(stdout);
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }//end iRank
  }//end for iFld
  sprintf(fldName,"srcAuxScLocation");
  fldPtr = &srcAuxScLocation[0];
  for(iRank = 0; iRank < mpi_size_world; iRank++){
      MPI_Barrier(MPI_COMM_WORLD);
      if(iRank == mpi_rank_world){
        printf("%d/%d %s:--------- \n[",mpi_rank_world,mpi_size_world,fldName);
        for(i=0; i < NhydroAuxScalars; i++){
          printf(" (%f, %f, %f) ",fldPtr[i*3+0],fldPtr[i*3+1],fldPtr[i*3+2]);
        }
        printf("]\n");
        fflush(stdout);
      }
      MPI_Barrier(MPI_COMM_WORLD);
  }//end iRank
#endif
  return(errorCode);
}//end srcAuxScConstructor()

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
   if(canopySelector > 0){
     memReleaseFloat(canopy_lad);
   }
   if(moistureSelector > 0){
     memReleaseFloat(moistScalars);
     memReleaseFloat(moistScalarsFrhs);
   }
   if(NhydroAuxScalars > 0){
     free(srcAuxScTemporalType);
     free(srcAuxScStartSeconds);
     free(srcAuxScDurationSeconds);
     free(srcAuxScGeometryType);
     free(srcAuxScMassSpecType);
     free(srcAuxScMassSpecValue);
     free(srcAuxScLocation);
     memReleaseFloat(hydroAuxScalars);
     memReleaseFloat(hydroAuxScalarsFrhs);
   } //end if NhydroAuxScalars

#ifdef GAD_EXT
   if(GADSelector > 0){
     errorCode = GADCleanup();
   }
#endif

   if(hydroBCs==1){
     free(hydroBndysFile);
     free(XZBdyPlanesGlobal);
     free(YZBdyPlanesGlobal);
     free(XYBdyPlanesGlobal);
     free(XZBdyPlanes);
     free(YZBdyPlanes);
     free(XYBdyPlanes);
     free(XZBdyPlanesNext);
     free(YZBdyPlanesNext);
     free(XYBdyPlanesNext);
     if(surflayerSelector == 3){
       free(SURFBdyPlanesGlobal);
       free(SURFBdyPlanes);
       free(SURFBdyPlanesNext);
     }
   } //end if hydroBCs==1

#ifdef URBAN_EXT
   if(urbanSelector > 0){
     errorCode = URBANCleanup();
   }
#endif

   return(errorCode);
}//end hydro_coreCleanup()

/*----->>>>> helper functions to create forcing strings --------------------------------------------------*/

// Increment the exponent of "s" if present, else append " s-1"
static char* make_forcing_units(const char *units) {
    if(units == NULL) return NULL;

    const char *s_ptr = strstr(units, "s-");
    if(s_ptr) {
        // Found "s-" pattern, try to increment number after it
        const char *exp_ptr = s_ptr + 2;
        int exp = atoi(exp_ptr);   // atoi will return 0 if not a number
        if(exp > 0) {
            exp++; // increment existing exponent

            // copy prefix (up to "s-")
            size_t prefix_len = s_ptr - units + 2;
            char prefix[prefix_len + 1];
            strncpy(prefix, units, prefix_len);
            prefix[prefix_len] = '\0';


            // format new string
            size_t buf_len = strlen(units) + 10; // extra space
            char *result = (char*)malloc(buf_len);
            if(!result) return NULL;

            snprintf(result, buf_len, "%s%d", prefix, exp);
            return result;
        }
    }

    // If no "s-" pattern or exponent, just append " s-1"
    size_t len = strlen(units);
    const char *suffix = " s-1";
    char *result = (char*)malloc(len + strlen(suffix) + 1);
    if(!result) return NULL;

    strcpy(result, units);
    strcat(result, suffix);
    return result;
}//end make_forcing_units()

// Allocates new string with " forcing" appended to long_name
static char* make_forcing_long_name(const char *long_name) {
    if(long_name == NULL) return NULL;

    size_t len = strlen(long_name);
    const char *suffix = " forcing";
    char *result = (char*)malloc(len + strlen(suffix) + 1);
    if(!result) return NULL;

    strcpy(result, long_name);
    strcat(result, suffix);
    return result;
}//end make_forcing_long_name()

/*----->>>>> int hydro_coreAddFieldAttributes();  -----------------------------------------------
* Utility function to add NetCDF attributes to hydro core fields based on field name
* Parameters:
*   fieldName - name of the field to add attributes to
*   isForcing - flag indicating if this is a forcing field (affects units)
* Returns: error code (0 = success, non-zero = error)
*/

int hydro_coreAddFieldAttributes(char *fieldName, int isForcing) {
    int errorCode = 0;
    char *baseFieldName = fieldName;

    // If this is a forcing field, skip the "F_" prefix to get the base field name
    if(isForcing && strncmp(fieldName, "F_", 2) == 0) {
        baseFieldName = fieldName + 2; // Skip "F_" prefix  
    }

    // Define field metadata structure                                                                                                                              
    typedef struct {
        char *pattern;
        char *units;
        char *long_name;
        char *standard_name;
    } field_metadata_t;

    // Field metadata lookup table
    field_metadata_t field_metadata[] = {
        {"BS_pressure", "Pa",            "Base state pressure",                                           "air_pressure"},
        {"TauTH1",      "K m s-1",       "Subgrid-x turbulent flux of potential temperature",             NULL},
        {"TauTH2",      "K m s-1",       "Subgrid-y turbulent flux of potential temperature",             NULL},
        {"TauTH3",      "K m s-1",       "Subgrid-z turbulent flux of potential temperature",             NULL},	
        {"Tau11",       "m2 s-2",        "Subgrid-xx stress tensor component",                            NULL},
        {"Tau21",       "m2 s-2",        "Subgrid-yx stress tensor component",                            NULL},
        {"Tau31",       "m2 s-2",        "Subgrid-zx stress tensor component",                            NULL},
        {"Tau32",       "m2 s-2",        "Subgrid-zy stress tensor component",                            NULL},
        {"Tau22",       "m2 s-2",        "Subgrid-yy stress tensor component",                            NULL},
        {"Tau33",       "m2 s-2",        "Subgrid-zz stress tensor component",                            NULL},
        {"rho",         "kg m-3",        "Air density",                                                   "air_density"},
        {"u",           "m s-1",         "Zonal wind velocity",                                           "eastward_wind"},
        {"v",           "m s-1",         "Meridional wind velocity",                                      "northward_wind"},
        {"w",           "m s-1",         "Vertical wind velocity",                                        "upward_air_velocity"},
        {"theta",       "K",             "Potential temperature",                                         "air_potential_temperature"},
        {"pressure",    "Pa",            "Perturbation pressure",                                         NULL},
        {"TKE_0",       "m2 s-2",        "Subgrid turbulent kinetic energy of air at grid-filter scale",  NULL},
        {"TKE_1",       "m2 s-2",        "Subgrid turbulent kinetic energy of air at canopy leaf scale",  NULL},
        {"AuxScalar",   "-",             "Auxiliary scalar",                                              NULL},
        {"moisture",    "kg kg-1",       "Water vapor mixing ratio",                                      "humidity_mixing_ratio"},
        {"qv",          "g kg-1",        "Water vapor mixing ratio",                                      "humidity_mixing_ratio"},
        {"ql",          "g kg-1",        "Cloud liquid water mixing ratio",                               "cloud_liquid_water_mixing_ratio"},
        {"fricVel",     "m s-1",         "Surface friction velocity",                                     "surface_friction_velocity"},
        {"htFlux",      "K m s-1",       "Surface sensible heat flux",                                    "surface_upward_sensible_heat_flux"},
        {"qFlux",       "kg kg-1 m s-1", "Surface latent heat flux",                                      "surface_upward_latent_heat_flux"},
        {"tskin",       "K",             "Surface skin temperature",                                      "surface_temperature"},
        {"qskin",       "kg kg-1",       "Surface skin water vapor mixing ratio",                         NULL},
        {"z0m",         "m",             "Roughness length for momentum",                                 "surface_roughness_length_for_momentum_in_air"},
        {"z0t",         "m",             "Roughness length for heat",                                     "surface_roughness_length_for_heat_in_air"},
        {"invOblen",    "m-1",           "Inverse Obukhov length",                                        "atmosphere_boundary_layer_thickness"},
        {"CanopyLAD",   "m-1",           "Leaf area density",                                             "leaf_area_density"},
        {"SeaMask",     "-",             "Sea mask",                                                      "sea_area_fraction"},
        {NULL, NULL, NULL, NULL} // End marker                                                                                                           
    };

    // Search for matching field pattern
    for(int i = 0; field_metadata[i].pattern != NULL; i++) {
	if (strncmp(baseFieldName, field_metadata[i].pattern, strlen(field_metadata[i].pattern)) == 0) {
            if(isForcing) {
                char *forcing_units = make_forcing_units(field_metadata[i].units);
                char *forcing_long_name = make_forcing_long_name(field_metadata[i].long_name);

 
                errorCode = ioAddStandardAttrs(fieldName,
                                               forcing_units,
                                               forcing_long_name,
                                               NULL); // No standard name for forcing fields
		
                free(forcing_units);
                free(forcing_long_name);
            } else {
                errorCode = ioAddStandardAttrs(fieldName,
                                               field_metadata[i].units,
                                               field_metadata[i].long_name,
                                               field_metadata[i].standard_name);
            }
            return errorCode;
        }
    }

    // Handle special case for BS_ fields with numeric identifiers
    if(strncmp(baseFieldName, "BS_", 3) == 0) {
        char *endptr;
        int fieldIndex = strtol(baseFieldName + 3, &endptr, 10);
        if(*endptr == '\0') { // Successfully parsed a number                                                                                                        
            if(fieldIndex == RHO_INDX_BS) { // 0 = rho base state                                                                                                    
                errorCode = ioAddStandardAttrs(fieldName, "kg m-3", "Base state air density", "air_density");
            }
            else if(fieldIndex == THETA_INDX_BS) { // 1 = theta base state                                                                                           
                errorCode = ioAddStandardAttrs(fieldName, "K", "Base state potential temperature", "air_potential_temperature");
            }
            else {
                errorCode = ioAddStandardAttrs(fieldName, "-", "Base state field", NULL);
            }
            return errorCode;
        }
    }

    // For unrecognized fields, add generic attributes
    printf("Warning: Unrecognized field '%s' in hydro_coreAddFieldAttributes, adding generic attributes\n", fieldName);
    if(isForcing) {
        errorCode = ioAddStandardAttrs(fieldName, "s-1", "Generic field forcing", NULL);
    } else {
        errorCode = ioAddStandardAttrs(fieldName, "-", "Generic field", NULL);
    }

    return errorCode;
}//end hydro_coreAddFieldAttributes()    

