/* FastEddy®: SRC/EXTENSIONS/GAD/GAD.h
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

#ifndef _GAD_H
#define _GAD_H

/*GAD return codes */
#define GAD_SUCCESS    0
#define GAD_FAIL       10

/*---GAD parameters*/
extern int GADSelector;         /* Generalized Actuator Disk Selector: 0=off, 1=on */
extern char *turbineSpecsFile;  /* The path+filename to a turbine specifications file*/
extern int GADoutputForces;     /* Flag to include GAD forces in the output: 0=off, 1=on */
extern int GADofflineForces;    /* Flag to compute GAD forces in an offline mode: 0=off, 1=on */
extern int GADaxialInduction;   /* Flag to compute axial induction factor: 0==off (uses prescribed GADaxialIndVal), 1==on */
extern float GADaxialIndVal;    /* Prescribed constant axial induction factor when GADaxialInduction==0 */
extern int GADrefSwitch;        /* Switch to use reference windspeed: 0=off, 1=on */
extern float GADrefU;           /* Prescribed constant reference hub-height windspeed*/
extern float GADrefSampWindow;  /* Sample duration (in seconds) over which to average per-timestep values (filtering out highest frequencies)*/
extern int GADsamplingAvgLength;/* number of timestep in the prescribed sample window */
extern float GADsamplingAvgWeight;/* sample window averaging weight*/
extern int GADrefSeriesLength;  /* Number of sampling windows over which to average again for reference velocity magnitude and direction */
extern float GADrefSeriesWeight;  /* ref Series averaging weight */
extern int GADForcingSwitch;    /* Switch to use the GADrefU-based or local windspeed in computing GAD forces: 0=local, 1=ref */
extern int GADNumTurbines;      /* Number of GAD Turbines */
extern int GADNumTurbineTypes;  /* Number of GAD Turbine Types */
extern int turbinePolyOrderMax; /* Maximum Polynomial order across all turbine types */
extern int turbinePolyClCdrNormSegments; /* Number of segments in the normalized radius for the lift and drag coefficient polynomial */
extern int alphaBounds;         /* Number of elements in the min/max angle of attack array for the lift/drag curves */

extern int numgridCells_away; /*Halo-region of cells considered in rotor disk distance-wise smoothing function*/

/*---GAD turbine characteristics arrays */
extern int* GAD_turbineType;    /* Integer class-label for turbine type*/ 
extern int* GAD_turbineRank;    /* Integer mpi-rank of nacelle center cell for each turbine reference velMag and velDir grid cell*/ 
extern int* GAD_turbineRefi;    /* Integer i-index of nacelle center cell for each turbine reference velMag and velDir grid cell*/ 
extern int* GAD_turbineRefj;    /* Integer j-index of nacelle center cell for each turbine reference velMag and velDir grid cell*/   
extern int* GAD_turbineRefk;    /* Integer k-index of nacelle center cell for each turbine reference velMag and velDir grid cell*/    
extern int* GAD_turbineYawing;  /* Integer indicating in a turbine is currently yawing ==1*/
extern float* GAD_Xcoords;      /* SW-corner (0,0)-relative x-coordinate of turbines [m]*/ 
extern float* GAD_Ycoords;      /* SW-corner (0,0)-relative y-coordinate of turbines [m]*/
extern float* GAD_turbineRefMag;/* Reference "ambient" velocity magnitude for yaw control and beta/omega [m/s]*/
extern float* GAD_turbineRefDir;/* *Reference "ambient" velocity direction (horizontal, met. standard orientation) for yaw control and beta/omega [degrees]*/
extern float* GAD_yawError;     /* yaw error between the incoming wind and the turbine orientation */
extern float* GAD_anFactor;     /* turbine axial induction factor at hub heigth*/
extern float* GAD_rotorTheta;   /* rotor-normal horizontal angle from North [degrees]*/
extern float* GAD_hubHeights;   /* Above-ground-level hub-heights of turbines [m]*/
extern float* GAD_rotorD;       /* turbine-specific rotor diameters  [m]*/
extern float* GAD_nacelleD;     /* turbine-specific nacelle diameters [m]*/
extern float* turbinePolyTwist; /* turbine-type-specific twist polynomial coefficients*/
extern float* turbinePolyChord; /* turbine-type-specific chord polynomial coefficients*/
extern float* turbinePolyPitch; /* turbine-type-specific pitch polynomial coefficients*/
extern float* turbinePolyOmega; /* turbine-type-specific omega polynomial coefficients*/
extern float* rnorm_vect;       /* turbine-type-specific normalized radious segment limits*/
extern float* alpha_minmax_vect;/* turbine-type-specific maximum and minimum angle of attack for the lift/drag curves*/
extern float* turbinePolyCl;    /* turbine-type-specific lift coefficient polynomial coefficients*/
extern float* turbinePolyCd;    /* turbine-type-specific drag coefficient polynomial coefficients*/


extern float* GAD_turbineVolMask; /* turbine Volume mask (0 if turbine free cell in domain, else turbine ID of cell in turbine yaw-swept volume*/
extern float* GAD_turbineRotorMask; /* turbine Rotor-disk  mask (0 if turbine free cell in domain, else 1.0 in turbine yaw-centric disk*/
extern float* GAD_forceX;         /* turbine forces in the x-direction */
extern float* GAD_forceY;         /* turbine forces in the y-direction */
extern float* GAD_forceZ;         /* turbine forces in the z-direction */

/*----->>>>> int GADGetParams();   ----------------------------------------------------------------------
 * Obtain parameters for the GAD sub-module
*/
int GADGetParams();

/*----->>>>> int GADPrintParams();   ----------------------------------------------------------------------
* Print parameters for the GAD sub-module
*/
int GADPrintParams();

/*----->>>>> int GADInit();   ----------------------------------------------------------------------
 * Used to broadcast and print parameters, allocate memory, and initialize configuration settings 
 * for the GAD sub-module.
 */
int GADInit();

/*----->>>>> int GADConstructor();   ----------------------------------------------------------------------
* This function constructs the GAD sub-module instance by reading a GAD (netCDF) input configuration file,
* allocating CPU-level memory for GAD arrays, and initializing these arrays with values specified in 
* the inputs file.
*/
int GADConstructor();

/*----->>>>> int GADInitTurbineRefChars();   ----------------------------------------------------------------------
* This function iinitializes turbine reference location characteristic values (location mpi_rank and i,j,k indices).
*/
int GADInitTurbineRefChars(float dt);

/*----->>>>> int GADCreateTurbineVolMask();   ----------------------------------------------------------------------
* This function creates the swept-volume mask (of turbine IDs as floats) for the turbine array
*/
int GADCreateTurbineVolMask();

/*----->>>>> int GADCreateTurbineRotorMask();   ----------------------------------------------------------------------
* This function creates the yaw-specific rotor-disk mask for turbines in the simulation
*/
int GADCreateTurbineRotorMask();

/*----->>>>> int GADUpdateTurbineRotorMask()   ----------------------------------------------------------------------
* This function updates (from GAD_rotorTheta) the yaw-specific rotor-disk mask for turbines in the simulation
*/
int GADUpdateTurbineRotorMask();

/*----->>>>> int GADDestructor();   ----------------------------------------------------------------------
* This function frees allocated memory of turbine characteristics arrays in the GAD module
*/
int GADDestructor();

/*----->>>>> int GADCleanup();  ----------------------------------------------------------------------
* Used to free all malloced memory by the GAD module.
*/
int GADCleanup();

#endif // _GAD_H
