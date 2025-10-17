/* FastEddy®: SRC/EXTENSIONS/GAD/GAD.c 
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
/*##################------------------- GAD sub-module variable definitions ---------------------#################*/

/*---GAD parameters */
int GADSelector;         /* Generalized Actuator Disk Selector: 0=off, 1=on */
char *turbineSpecsFile;  /* The path+filename to a turbine specifications file*/
int GADoutputForces;     /* Flag to include GAD forces in the output: 0=off, 1=on */
int GADofflineForces;    /* Flag to compute GAD forces in an offline mode: 0=off, 1=on */
int GADaxialInduction;   /* Flag to compute axial induction factor: 0==off (uses prescribed GADaxialIndVal), 1==on */
float GADaxialIndVal;    /* Prescribed constant axial induction factor when GADaxialInduction==0 */
int GADrefSwitch;        /* Switch to use reference windspeed: 0=off, 1=on */
float GADrefU;           /* Prescribed constant reference hub-height windspeed*/
float GADrefSampleWindow;/* Sample duration (in seconds) over which to average per-timestep values (filtering out highest frequencies) */
int GADsamplingAvgLength;/* number of timestep in the prescribed sample window */
float GADsamplingAvgWeight;/* sample window averaging weight*/
int GADrefSeriesLength;  /* Number of sampling windows over which to average again for reference velocity magnitude and direction */
float GADrefSeriesWeight;  /* ref Series averaging weight */
int GADForcingSwitch;    /* Switch to use the GADrefU-based or local windspeed in computing GAD forces: 0=local, 1=ref */
int GADNumTurbines = 0;      /* Number of GAD Turbines */
int GADNumTurbineTypes;  /* Number of GAD Turbine Types */
int turbinePolyOrderMax; /* Maximum Polynomial order across all turbine types */
int turbinePolyClCdrNormSegments; /* Number of segments in the normalized radius for the lift and drag coefficient polynomial */
int alphaBounds;         /* Number of elements in the min/max angle of attack array for the lift/drag curves */

int numgridCells_away; /*Halo-region of cells considered in rotor disk distance-wise smoothing function*/
/*---GAD turbine characteristics arrays */
int* GAD_turbineType;    /* Integer class-label for turbine type*/
int* GAD_turbineRank;    /* Integer mpi-rank of nacelle center cell for each turbine reference velMag and velDir grid cell*/
int* GAD_turbineRefi;    /* Integer i-index of nacelle center cell for each turbine reference velMag and velDir grid cell*/
int* GAD_turbineRefj;    /* Integer j-index of nacelle center cell for each turbine reference velMag and velDir grid cell*/
int* GAD_turbineRefk;    /* Integer k-index of nacelle center cell for each turbine reference velMag and velDir grid cell*/
int* GAD_turbineYawing;  /* Integer indicating in a turbine is currently yawing ==1*/
float* GAD_Xcoords;      /* SW-corner (0,0)-relative x-coordinate of turbines [m]*/ 
float* GAD_Ycoords;      /* SW-corner (0,0)-relative y-coordinate of turbines [m]*/
float* GAD_turbineRefMag;/* Reference "ambient" velocity magnitude for yaw control and beta/omega [m/s]*/
float* GAD_turbineRefDir;/* *Reference "ambient" velocity direction (horizontal, met. standard orientation) for yaw control and beta/omega [degrees]*/
float* GAD_yawError;     /* yaw error between the incoming wind and the turbine orientation */
float* GAD_anFactor;     /* turbine axial induction factor at hub heigth*/
float* GAD_rotorTheta;   /* rotor-normal horizontal angle from North [degrees]*/
float* GAD_hubHeights;   /* Above-ground-level hub-heights of turbines [m]*/
float* GAD_rotorD;       /* turbine-specific rotor diameters [m]*/
float* GAD_nacelleD;     /* turbine-specific nacelle diameters [m]*/
float* turbinePolyTwist; /* turbine-type-specific twist polynomial coefficients*/
float* turbinePolyChord; /* turbine-type-specific chord polynomial coefficients*/
float* turbinePolyPitch; /* turbine-type-specific pitch polynomial coefficients*/
float* turbinePolyOmega; /* turbine-type-specific omega polynomial coefficients*/
float* rnorm_vect;       /* turbine-type-specific normalized radious segment limits*/
float* alpha_minmax_vect;/* turbine-type-specific maximum and minimum angle of attack for the lift/drag curves*/
float* turbinePolyCl;    /* turbine-type-specific lift coefficient polynomial coefficients*/
float* turbinePolyCd;    /* turbine-type-specific drag coefficient polynomial coefficients*/


float* GAD_turbineVolMask; /* turbine Volume mask (0 if turbine free cell in domain, else turbine ID of cell in turbine yaw-swept volume*/
float* GAD_turbineRotorMask; /* turbine Rotor-disk  mask (0 if turbine free cell in domain, else 1.0 in turbine yaw-centric disk*/
float* GAD_forceX;         /* turbine forces in the x-direction */
float* GAD_forceY;         /* turbine forces in the y-direction */
float* GAD_forceZ;         /* turbine forces in the z-direction */

/*----->>>>> int GADGetParams();   ----------------------------------------------------------------------
 * Obtain parameters for the GAD sub-module
*/
int GADGetParams(){
   int errorCode = GAD_SUCCESS;

   GADSelector = 0; // Default to 0
   errorCode = queryIntegerParameter("GADSelector", &GADSelector, 0, 1, PARAM_OPTIONAL);
   if(GADSelector > 0){
     errorCode = queryFileParameter("turbineSpecsFile", &turbineSpecsFile, PARAM_OPTIONAL);
     GADoutputForces = 0; // default off
     errorCode = queryIntegerParameter("GADoutputForces", &GADoutputForces, 0, 1, PARAM_OPTIONAL);
     GADofflineForces = 0; // default off
     errorCode = queryIntegerParameter("GADofflineForces", &GADofflineForces, 0, 1, PARAM_OPTIONAL);
     GADaxialInduction = 1; // default off
     errorCode = queryIntegerParameter("GADaxialInduction", &GADaxialInduction, 0, 1, PARAM_OPTIONAL);
     GADaxialIndVal = 0.02; // default to 2%
     errorCode = queryFloatParameter("GADaxialIndVal", &GADaxialIndVal, 0.0, 1.0, PARAM_OPTIONAL);
     GADrefSwitch = 0; // default off
     errorCode = queryIntegerParameter("GADrefSwitch", &GADrefSwitch, 0, 1, PARAM_OPTIONAL);
     GADrefSampleWindow = 30.0; // default to 30.0 seconds, limit in range 1.0-60.0 seconds
     errorCode = queryFloatParameter("GADrefSampleWindow", &GADrefSampleWindow, 1.0, 300.0, PARAM_OPTIONAL);
     GADrefSeriesLength = 10; // default to a series length of 10 sample-window averaged values
     errorCode = queryIntegerParameter("GADrefSeriesLength", &GADrefSeriesLength, 1, 360, PARAM_OPTIONAL);
     GADForcingSwitch = 0; // default off
     errorCode = queryIntegerParameter("GADForcingSwitch", &GADForcingSwitch, 0, 1, PARAM_OPTIONAL);
     if (GADrefSwitch == 1){
       GADrefU = 0.0; // default to 0.0 m/s
       errorCode = queryFloatParameter("GADrefU", &GADrefU, 0.0, 50.0, PARAM_MANDATORY);
     }
   }//End if GADSelector > 0
   
   return(errorCode);
} //end GADGetParams()

/*----->>>>> int GADPrintParams();   ----------------------------------------------------------------------
* Print parameters for the GAD sub-module
*/
int GADPrintParams(){
   int errorCode = GAD_SUCCESS;
   if(mpi_rank_world == 0){
     printComment("----------: GAD ---");
     printParameter("GADSelector", "Generalized Actuator Disk Selector: 0=off, 1=on ");
     if(GADSelector > 0){
       printParameter("turbineSpecsFile", "netCDF file with turbine specifications ");
       printParameter("GADoutputForces", "Flag to include GAD forces in the output: 0=off, 1=on");
       printParameter("GADofflineForces", "Flag to compute GAD forces in an offline mode: 0=off, 1=on");
       printParameter("GADaxialInduction", "Flag to compute axial induction factor: 0==off (uses prescribed GADaxialIndVal), 1==on");
       printParameter("GADaxialIndVal", "Prescribed constant axial induction factor when GADaxialInduction==0");
       printParameter("GADrefSwitch", "Switch to use reference windspeed: 0=off, 1=on");
       printParameter("GADrefU", "Prescribed constant reference hub-height windspeed");
       printParameter("GADrefSampleWindow", "Sample duration (in seconds) over which to average per-timestep values (filtering out highest frequencies)");
       printParameter("GADrefSeriesLength", "Number of sampling windows over which to average again for reference velocity magnitude and direction");
       printParameter("GADForcingSwitch", "Switch to use the GADrefU-based or local windspeed in computing GAD forces: 0=local, 1=ref");
     }
   } //end if(mpi_rank_world == 0)
   return(errorCode);
} //end GADPrintParams()

/*----->>>>> int GADInit();   ----------------------------------------------------------------------
 * Used to broadcast and print parameters, allocate memory, and initialize configuration settings 
 * for the GAD sub-module.
*/
int GADInit(){
   int errorCode = GAD_SUCCESS;
   int strLength;

   /*Broadcast GAD parameters to all mpi_ranks*/
   MPI_Bcast(&GADSelector, 1, MPI_INT, 0, MPI_COMM_WORLD);
   if(GADSelector > 0){
      strLength = 0;
      if(mpi_rank_world == 0){
        if(turbineSpecsFile != NULL){
          strLength = strlen(turbineSpecsFile)+1;
        }else{
          strLength = 0;
        }
      } //end if(mpi_rank_world == 0)
      MPI_Bcast(&strLength, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);
      if(strLength > 0){
        if(mpi_rank_world != 0){
           turbineSpecsFile = (char *) malloc(strLength*sizeof(char));
        } //if a non-root mpi_rank
        MPI_Bcast(turbineSpecsFile, strLength, MPI_CHARACTER, 0, MPI_COMM_WORLD);
      }

      /*Call the GAD module instance turbine array configuration constructor*/
      errorCode = GADConstructor();

      MPI_Bcast(&GADoutputForces, 1, MPI_INT, 0, MPI_COMM_WORLD);
      MPI_Bcast(&GADofflineForces, 1, MPI_INT, 0, MPI_COMM_WORLD);
      MPI_Bcast(&GADaxialInduction, 1, MPI_INT, 0, MPI_COMM_WORLD);
      MPI_Bcast(&GADaxialIndVal, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
      MPI_Bcast(&GADrefSwitch, 1, MPI_INT, 0, MPI_COMM_WORLD);
      MPI_Bcast(&GADrefU, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
      MPI_Bcast(&GADForcingSwitch, 1, MPI_INT, 0, MPI_COMM_WORLD);
   } //end if GADSelector > 0

   
   /*Could set this to be a runtime parameter at some point, setting constant for now.*/
   numgridCells_away = 3;
 
   return(errorCode);
} //end GADInit()

/*----->>>>> int GADConstructor();   ----------------------------------------------------------------------
* This function constructs the GAD sub-module instance by reading a GAD (netCDF) input configuration file,
* allocating CPU-level memory for GAD arrays, and initializing these arrays with values specified in 
* the inputs file.
*/
int GADConstructor(){
  int errorCode = GAD_SUCCESS;
  int ncid;
  int ncfldid;
  int iFld;
  int dimids[64];
  size_t count[64];
  size_t start[64];
  size_t polycount[64];
  size_t polystart[64];
  char fldName[64];
  float **fldPtr;
  int **intfldPtr;

  //Root-rank should read the netcdf turbineSpecsFile
  if(mpi_rank_world == 0){
//#define DEBUG_GADCONSTRUCTOR
#ifdef DEBUG_GADCONSTRUCTOR
    printf("Attempting to open turbineSpecsFile = %s\n",turbineSpecsFile);
    fflush(stdout);
#endif

    //Open the netcdf GAD inputs file
    errorCode = ioOpenNetCDFinFile(turbineSpecsFile, &ncid);
    if(errorCode > 0){
      printf("Failed to open turbineSpecsFile = %s EXITING NOW!!!!\n",turbineSpecsFile);
      fflush(stdout);
      exit(0);
    }
#ifdef DEBUG_GADCONSTRUCTOR
    printf("Opened turbineSpecsFile = %s with ncid = %d\n",turbineSpecsFile,ncid);
    fflush(stdout);
#endif

    //Inquire for the dimID of the turbine configuration fundamental parameter, GADNumTurbines
    if((errorCode = nc_inq_dimid(ncid, "GADNumTurbines", &dimids[0]))){
      ERR(errorCode);
    }
    //Inquire for the value of GADNumTurbines 
    if((errorCode = nc_inq_dimlen(ncid, dimids[0], &count[dimids[0]]))){
      ERR(errorCode);
    }
    //Assign the GADNumTurbines to the value of the netCDF file dimension  
    GADNumTurbines = (int)count[dimids[0]];
    
    //Inquire for the dimID of the turbine configuration fundamental parameter, GADNumTurbineTypes
    if((errorCode = nc_inq_dimid(ncid, "GADNumTurbineTypes", &dimids[1]))){
      ERR(errorCode);
    }
    //Inquire for the value of GADNumTurbineTypes 
    if((errorCode = nc_inq_dimlen(ncid, dimids[1], &count[dimids[1]]))){
      ERR(errorCode);
    }
    //Assign the GADNumTurbineTypes to the value of the netCDF file dimension  
    GADNumTurbineTypes = (int)count[dimids[1]];
   
    
    //Inquire for the dimID of the turbine configuration fundamental parameter, turbinePolyOrderMax 
    if((errorCode = nc_inq_dimid(ncid, "turbinePolyOrderMax", &dimids[2]))){
      ERR(errorCode);
    }
    //Inquire for the value of turbinePolyOrderMax 
    if((errorCode = nc_inq_dimlen(ncid, dimids[2], &count[dimids[2]]))){
      ERR(errorCode);
    }
    //Assign the turbinePolyOrderMax to the value of the netCDF file dimension  
    turbinePolyOrderMax = (int)count[dimids[2]];

    //Inquire for the dimID of the turbine configuration fundamental parameter, turbinePolyClCdrNormSegments
    if((errorCode = nc_inq_dimid(ncid, "turbinePolyClCdrNormSegments", &dimids[3]))){
      ERR(errorCode);
    }
    //Inquire for the value of turbinePolyClCdrNormSegments
    if((errorCode = nc_inq_dimlen(ncid, dimids[3], &count[dimids[3]]))){
      ERR(errorCode);
    }
    //Assign the turbinePolyClCdrNormSegments to the value of the netCDF file dimension
    turbinePolyClCdrNormSegments = (int)count[dimids[3]];

    //Inquire for the dimID of the turbine configuration fundamental parameter, alphaBounds
    if((errorCode = nc_inq_dimid(ncid, "alphaBounds", &dimids[4]))){
      ERR(errorCode);
    }
    //Inquire for the value of alphaBounds
    if((errorCode = nc_inq_dimlen(ncid, dimids[4], &count[dimids[4]]))){
      ERR(errorCode);
    }
    //Assign the alphaBounds to the value of the netCDF file dimension
    alphaBounds = (int)count[dimids[4]];

  } //end if(mpi_rank_world == 0)
  MPI_Bcast(&GADNumTurbines, 1, MPI_INTEGER, 0, MPI_COMM_WORLD); 
  MPI_Bcast(&GADNumTurbineTypes, 1, MPI_INTEGER, 0, MPI_COMM_WORLD); 
  MPI_Bcast(&turbinePolyOrderMax, 1, MPI_INTEGER, 0, MPI_COMM_WORLD); 
  MPI_Bcast(&turbinePolyClCdrNormSegments, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);
  MPI_Bcast(&alphaBounds, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);
  
#ifdef DEBUG_GADCONSTRUCTOR
  printf("%d/%d GADNumTurbines = %d\n",mpi_rank_world,mpi_size_world, GADNumTurbines);
  printf("%d/%d GADNumTurbineTypes = %d\n",mpi_rank_world,mpi_size_world, GADNumTurbineTypes);
  printf("%d/%d turbinePolyOrderMax = %d\n",mpi_rank_world,mpi_size_world, turbinePolyOrderMax);
  printf("%d/%d turbinePolyClCdrNormSegments = %d\n",mpi_rank_world,mpi_size_world, turbinePolyClCdrNormSegments);
  printf("%d/%d alphaBounds = %d\n",mpi_rank_world,mpi_size_world, alphaBounds);
  fflush(stdout);
#endif
    

  /*Allocate for the GAD_turbineType array*/
  sprintf(fldName,"GAD_turbineType");
  intfldPtr = &GAD_turbineType;
  *intfldPtr = (int*) malloc(GADNumTurbines*sizeof(int));
  /* Read in netCDF file values for each turbine-Type */
  if(mpi_rank_world == 0){
    start[0] = 0;
    if ( (errorCode = nc_inq_varid(ncid, fldName, &ncfldid)) ){
       ERR(errorCode);
       printf("Error GADConstructor(): field = %s was not found in this file,!\n",fldName);
       fflush(stdout);
    } //if nc_inq_varid
    if ((errorCode = nc_get_vara_int(ncid, ncfldid, &start[0], &count[dimids[0]], *intfldPtr )) ){
       ERR(errorCode);
    }
  } //end if mpi_rank_world == 0
  MPI_Bcast(*intfldPtr, GADNumTurbines, MPI_INTEGER, 0, MPI_COMM_WORLD);

  /*Allocate, read in field values (root-rank only), and broadcast
   * each of the turbine-general characteristics arrays */
  for(iFld = 0; iFld < 3; iFld++){
     switch (iFld){
         case 0:
           sprintf(fldName,"GAD_Xcoords");
           fldPtr = &GAD_Xcoords;
           break;
         case 1:
           sprintf(fldName,"GAD_Ycoords");
           fldPtr = &GAD_Ycoords;
           break;
         case 2:
           sprintf(fldName,"GAD_rotorTheta");
           fldPtr = &GAD_rotorTheta;
           break;
         default:    //invalid iFld value
           sprintf(fldName,"");
           fldPtr = NULL;
           errorCode = GAD_FAIL;
           break;
     }//end switch(iFld)
     if(iFld<3){
       /*Allocate for the field*/
       *fldPtr = (float*) malloc(GADNumTurbines*sizeof(float));
       /* Read in netCDF file values for each of the turbine characteristics arrays */
       if(mpi_rank_world == 0){
         start[0] = 0;
         if ( (errorCode = nc_inq_varid(ncid, fldName, &ncfldid)) ){
            ERR(errorCode);
             printf("Error GADConstructor(): field = %s was not found in this file,!\n",fldName);
             fflush(stdout);
         } //if nc_inq_varid
         if ((errorCode = nc_get_vara_float(ncid, ncfldid, &start[0], &count[dimids[0]], *fldPtr )) ){
                ERR(errorCode);
         }
       } //end if mpi_rank_world == 0
       MPI_Bcast(*fldPtr, GADNumTurbines, MPI_FLOAT, 0, MPI_COMM_WORLD);
       if(iFld == 2){
         errorCode = ioRegisterVar(&fldName[0], "float", 2, dims1dTD_GAD, *fldPtr);
         errorCode = ioAddStandardAttrs("GAD_rotorTheta", "degrees", "rotor angle from west increasing counter-clockwise", NULL);
         printf("%d/%d: GADConstructor()-- %s stored at %p, has been registered with IO.\n",
                mpi_rank_world, mpi_size_world, &fldName[0], *fldPtr);
         fflush(stdout);
       } //end if iFld ==2 ... GAD_rotorTheta
#ifdef DEBUG_GADCONSTRUCTOR
       int iRank,i,j,ipoly;
       for(iRank = 0; iRank < mpi_size_world; iRank++){
         MPI_Barrier(MPI_COMM_WORLD);
         if(iRank == mpi_rank_world){
           printf("%d/%d %s:--------- \n",mpi_rank_world,mpi_size_world, fldName);
           for(i = 0; i < GADNumTurbines; i++){
               printf("\t%d \t= \t%f\n",i,(*fldPtr)[i]);
           }// end for i...
           printf("\n");
           fflush(stdout);
         } //end if iRank == mpi_rank_world 
         MPI_Barrier(MPI_COMM_WORLD);
       }//end for iRank...
#endif
    }//end if iFld < 3
  }// end for iFld...
 
  /*Allocate, read in field values (root-rank only), and broadcast
   * each of the turbine-type-specific characteristics arrays */
  for(iFld = 0; iFld < 3; iFld++){
     switch (iFld){
         case 0:
           sprintf(fldName,"GAD_hubHeights");
           fldPtr = &GAD_hubHeights;
           break;
         case 1:
           sprintf(fldName,"GAD_rotorD");
           fldPtr = &GAD_rotorD;
           break;
         case 2:
           sprintf(fldName,"GAD_nacelleD");
           fldPtr = &GAD_nacelleD;
           break;
         default:    //invalid iFld value
           sprintf(fldName,"");
           fldPtr = NULL;
           errorCode = GAD_FAIL;
           break;
     }//end switch(iFld)
     if(iFld<3){
       /*Allocate for the field*/
       *fldPtr = (float*) malloc(GADNumTurbineTypes*sizeof(float));
       /* Read in netCDF file values for each of the turbine characteristics arrays */
       if(mpi_rank_world == 0){
         start[1] = 0;
         if ( (errorCode = nc_inq_varid(ncid, fldName, &ncfldid)) ){
            ERR(errorCode);
             printf("Error GADConstructor(): field = %s was not found in this file,!\n",fldName);
             fflush(stdout);
         } //if nc_inq_varid
         if ((errorCode = nc_get_vara_float(ncid, ncfldid, &start[1], &count[dimids[1]], *fldPtr )) ){
                ERR(errorCode);
         }
       } //end if mpi_rank_world == 0
       MPI_Bcast(*fldPtr, GADNumTurbineTypes, MPI_FLOAT, 0, MPI_COMM_WORLD);
#ifdef DEBUG_GADCONSTRUCTOR
       for(iRank = 0; iRank < mpi_size_world; iRank++){
         MPI_Barrier(MPI_COMM_WORLD);
         if(iRank == mpi_rank_world){
           printf("%d/%d %s:--------- \n",mpi_rank_world,mpi_size_world, fldName);
           for(i = 0; i < GADNumTurbineTypes; i++){
             printf("\t%d \t= \t%f\n",i,(*fldPtr)[i]);
           }// end for i...
           printf("\n");
           fflush(stdout);
         } //end if iRank == mpi_rank_world 
         MPI_Barrier(MPI_COMM_WORLD);
       }//end for iRank...
#endif
    }//end if iFld < 3
  }// end for iFld...
  
  /*Allocate, read in field values (root-rank only), and broadcast
   * each of the turbine-type-specific polynomial coefficients arrays */
  for(iFld = 0; iFld < 8; iFld++){
     switch (iFld){
         case 0:
           sprintf(fldName,"turbinePolyTwist");
           fldPtr = &turbinePolyTwist;
           break;
         case 1:
           sprintf(fldName,"turbinePolyChord");
           fldPtr = &turbinePolyChord;
           break;
         case 2:
           sprintf(fldName,"turbinePolyPitch");
           fldPtr = &turbinePolyPitch;
           break;
         case 3:
           sprintf(fldName,"turbinePolyOmega");
           fldPtr = &turbinePolyOmega;
           break;
         case 4:
           sprintf(fldName,"rnorm_vect");
           fldPtr = &rnorm_vect;
           break;
         case 5:
           sprintf(fldName,"alpha_minmax_vect");
           fldPtr = &alpha_minmax_vect;
           break;
         case 6:
           sprintf(fldName,"turbinePolyCl");
           fldPtr = &turbinePolyCl;
           break;
         case 7:
           sprintf(fldName,"turbinePolyCd");
           fldPtr = &turbinePolyCd;
           break;
         default:    //invalid iFld value
           sprintf(fldName,"");
           fldPtr = NULL;
           errorCode = GAD_FAIL;
           break;
     }//end switch(iFld)
     if(iFld<4){
       /*Allocate for the field*/
       *fldPtr = (float*) malloc(GADNumTurbineTypes*turbinePolyOrderMax*sizeof(float));
       /* Read in netCDF file values for each of the turbine characteristics arrays */
       if(mpi_rank_world == 0){
         polystart[0] = 0;
         polystart[1] = 0;
         polycount[0] = count[dimids[1]];
         polycount[1] = count[dimids[2]];
         if ( (errorCode = nc_inq_varid(ncid, fldName, &ncfldid)) ){
            ERR(errorCode);
             printf("Error GADConstructor(): field = %s was not found in this file,!\n",fldName);
             fflush(stdout);
         } //if nc_inq_varid
         if ((errorCode = nc_get_vara_float(ncid, ncfldid, &polystart[0], &polycount[0], *fldPtr )) ){ //Note fortuitous reuse of count[1-2] being the right dims.
                ERR(errorCode);
         }  
       } //end if mpi_rank_world == 0
       MPI_Bcast(*fldPtr, GADNumTurbineTypes*turbinePolyOrderMax, MPI_FLOAT, 0, MPI_COMM_WORLD);
#ifdef DEBUG_GADCONSTRUCTOR
       for(iRank = 0; iRank < mpi_size_world; iRank++){
         MPI_Barrier(MPI_COMM_WORLD);
         if(iRank == mpi_rank_world){
           printf("%d/%d %s:--------- \n",mpi_rank_world,mpi_size_world, fldName);
           for(i = 0; i < GADNumTurbineTypes; i++){
               printf("\t%d \t=\t",i);
             for(ipoly = 0; ipoly < turbinePolyOrderMax; ipoly++){
               printf("%f\t",(*fldPtr)[i*turbinePolyOrderMax+ipoly]);
             }// end for i...
             printf("\n");
           }// end for i...
           printf("\n");
           fflush(stdout);
         } //end if iRank == mpi_rank_world 
         MPI_Barrier(MPI_COMM_WORLD);
       }//end for iRank...
#endif
     }else if(iFld==4){ // rnorm_vect
       /*Allocate for the field*/
       *fldPtr = (float*) malloc(GADNumTurbineTypes*(turbinePolyClCdrNormSegments+1)*sizeof(float));
       /* Read in netCDF file values for each of the turbine characteristics arrays */
       if(mpi_rank_world == 0){
         polystart[0] = 0;
         polystart[1] = 0;
         polycount[0] = count[dimids[1]];
         polycount[1] = count[dimids[3]]+1;
         if ( (errorCode = nc_inq_varid(ncid, fldName, &ncfldid)) ){
            ERR(errorCode);
             printf("Error GADConstructor(): field = %s was not found in this file,!\n",fldName);
             fflush(stdout);
         } //if nc_inq_varid
         if ((errorCode = nc_get_vara_float(ncid, ncfldid, &polystart[0], &polycount[0], *fldPtr )) ){ //Note fortuitous reuse of count[1-2] being the right dims.
                ERR(errorCode);
         }
       } //end if mpi_rank_world == 0
       MPI_Bcast(*fldPtr, GADNumTurbineTypes*(turbinePolyClCdrNormSegments+1), MPI_FLOAT, 0, MPI_COMM_WORLD);
#ifdef DEBUG_GADCONSTRUCTOR
       for(iRank = 0; iRank < mpi_size_world; iRank++){
         MPI_Barrier(MPI_COMM_WORLD);
         if(iRank == mpi_rank_world){
           printf("%d/%d %s:--------- \n",mpi_rank_world,mpi_size_world, fldName);
           for(i = 0; i < GADNumTurbineTypes; i++){
               printf("\t%d \t=\t",i);
             for(ipoly = 0; ipoly < turbinePolyClCdrNormSegments+1; ipoly++){
               printf("%f\t",(*fldPtr)[i*(turbinePolyClCdrNormSegments+1)+ipoly]);
             }// end for i...
             printf("\n");
           }// end for i...
           printf("\n");
           fflush(stdout);
         } //end if iRank == mpi_rank_world
         MPI_Barrier(MPI_COMM_WORLD);
       }//end for iRank...
#endif
     }else if(iFld==5){ // alpha_minmax_vect
       /*Allocate for the field*/
       *fldPtr = (float*) malloc(GADNumTurbineTypes*alphaBounds*sizeof(float));
       /* Read in netCDF file values for each of the turbine characteristics arrays */
       if(mpi_rank_world == 0){
         polystart[0] = 0;
         polystart[1] = 0;
         polycount[0] = count[dimids[1]];
         polycount[1] = count[dimids[4]];
         if ( (errorCode = nc_inq_varid(ncid, fldName, &ncfldid)) ){
            ERR(errorCode);
             printf("Error GADConstructor(): field = %s was not found in this file,!\n",fldName);
             fflush(stdout);
         } //if nc_inq_varid
         if ((errorCode = nc_get_vara_float(ncid, ncfldid, &polystart[0], &polycount[0], *fldPtr )) ){ //Note fortuitous reuse of count[1-2] being the right dims.
                ERR(errorCode);
         }
       } //end if mpi_rank_world == 0
       MPI_Bcast(*fldPtr, GADNumTurbineTypes*alphaBounds, MPI_FLOAT, 0, MPI_COMM_WORLD);
#ifdef DEBUG_GADCONSTRUCTOR
       for(iRank = 0; iRank < mpi_size_world; iRank++){
         MPI_Barrier(MPI_COMM_WORLD);
         if(iRank == mpi_rank_world){
           printf("%d/%d %s:--------- \n",mpi_rank_world,mpi_size_world, fldName);
           for(i = 0; i < GADNumTurbineTypes; i++){
               printf("\t%d \t=\t",i);
             for(ipoly = 0; ipoly < alphaBounds; ipoly++){
               printf("%f\t",(*fldPtr)[i*alphaBounds+ipoly]);
             }// end for i...
             printf("\n");
           }// end for i...
           printf("\n");
           fflush(stdout);
         } //end if iRank == mpi_rank_world
         MPI_Barrier(MPI_COMM_WORLD);
       }//end for iRank...
#endif
     }else if(iFld>5 && iFld<8){ // turbinePolyCl,turbinePolyCd
       /*Allocate for the field*/
       *fldPtr = (float*) malloc(GADNumTurbineTypes*turbinePolyClCdrNormSegments*turbinePolyOrderMax*sizeof(float));
       /* Read in netCDF file values for each of the turbine characteristics arrays */
       if(mpi_rank_world == 0){
         polystart[0] = 0;
         polystart[1] = 0;
         polystart[2] = 0;
         polycount[0] = count[dimids[1]];
         polycount[1] = count[dimids[3]];
         polycount[2] = count[dimids[2]];
         if ( (errorCode = nc_inq_varid(ncid, fldName, &ncfldid)) ){
            ERR(errorCode);
             printf("Error GADConstructor(): field = %s was not found in this file,!\n",fldName);
             fflush(stdout);
         } //if nc_inq_varid
         if ((errorCode = nc_get_vara_float(ncid, ncfldid, &polystart[0], &polycount[0], *fldPtr )) ){ //Note fortuitous reuse of count[1-2] being the right dims.
                ERR(errorCode);
         }
       } //end if mpi_rank_world == 0
       MPI_Bcast(*fldPtr, GADNumTurbineTypes*turbinePolyClCdrNormSegments*turbinePolyOrderMax, MPI_FLOAT, 0, MPI_COMM_WORLD);
#ifdef DEBUG_GADCONSTRUCTOR
       for(iRank = 0; iRank < mpi_size_world; iRank++){
         MPI_Barrier(MPI_COMM_WORLD);
         if(iRank == mpi_rank_world){
           printf("%d/%d %s:--------- \n",mpi_rank_world,mpi_size_world, fldName);
           for(i = 0; i < GADNumTurbineTypes; i++){
               printf("\t i %d \t=\t",i);
             for(j = 0; j < turbinePolyClCdrNormSegments; j++){
                printf("\n\t segment %d \t=\t",j);
                for(ipoly = 0; ipoly < turbinePolyOrderMax; ipoly++){
                  printf("%g\t",(*fldPtr)[i*(turbinePolyOrderMax*turbinePolyClCdrNormSegments)+j*turbinePolyOrderMax+ipoly]);
               }// end for ipoly...
             }// end for j...
             printf("\n");
           }// end for i...
           printf("\n");
           fflush(stdout);
         } //end if iRank == mpi_rank_world
         MPI_Barrier(MPI_COMM_WORLD);
       }//end for iRank...
#endif
     }//end if iFld read in data
  }// end for iFld...


  //Done reading the turbineSpecs netCDF
  if(mpi_rank_world == 0){
    /* Close the file. */
    if ((errorCode = nc_close(ncid))){
     ERR(errorCode);
    }
  } //end if mpi_rank_world == 0

  /*Allocate for other turbine-specific internal characteristics arrays*/
  GAD_turbineRank = (int*) malloc(GADNumTurbines*sizeof(int));
  GAD_turbineRefi = (int*) malloc(GADNumTurbines*sizeof(int));
  GAD_turbineRefj = (int*) malloc(GADNumTurbines*sizeof(int));
  GAD_turbineRefk = (int*) malloc(GADNumTurbines*sizeof(int));
  GAD_turbineYawing = (int*) malloc(GADNumTurbines*sizeof(int));
  sprintf(&fldName[0],"GAD_turbineYawing");
  errorCode = ioRegisterVar(&fldName[0], "int", 2, dims1dTD_GAD, &GAD_turbineYawing[0]);
  errorCode = ioAddStandardAttrs("GAD_turbineYawing", "-", "flag indicating turbine is in the process of yawing", NULL);
  GAD_turbineRefMag = (float*) malloc(GADNumTurbines*sizeof(float));
  sprintf(&fldName[0],"GAD_turbineRefMag");
  errorCode = ioRegisterVar(&fldName[0], "float", 2, dims1dTD_GAD, &GAD_turbineRefMag[0]);
  errorCode = ioAddStandardAttrs("GAD_turbineRefMag", "m s-1", "turbine reference wind speed", NULL);
  GAD_turbineRefDir = (float*) malloc(GADNumTurbines*sizeof(float));
  sprintf(&fldName[0],"GAD_turbineRefDir");
  errorCode = ioRegisterVar(&fldName[0], "float", 2, dims1dTD_GAD, &GAD_turbineRefDir[0]);
  errorCode = ioAddStandardAttrs("GAD_turbineRefDir", "degrees", "turbine reference wind direction", NULL);
  GAD_yawError = (float*) malloc(GADNumTurbines*sizeof(float));
  sprintf(&fldName[0],"GAD_yawError");
  errorCode = ioRegisterVar(&fldName[0], "float", 2, dims1dTD_GAD, &GAD_yawError[0]);
  errorCode = ioAddStandardAttrs("GAD_yawError", "degrees2 s", "turbine-wind misalignment error for yaw-controller", NULL);
  GAD_anFactor = (float*) malloc(GADNumTurbines*sizeof(float));
  sprintf(&fldName[0],"GAD_anFactor");
  errorCode = ioRegisterVar(&fldName[0], "float", 2, dims1dTD_GAD, &GAD_anFactor[0]);
  errorCode = ioAddStandardAttrs("GAD_anFactor", "-", "rotor-normal upstream wind speed induction factor", NULL);

  return(errorCode);
} //end GADConstructor()

/*----->>>>> int GADInitTurbineRefChars();   ----------------------------------------------------------------------
* This function iinitializes turbine reference location characteristic values (location mpi_rank and i,j,k indices). 
*/
int GADInitTurbineRefChars(float dt){
  int errorCode = GAD_SUCCESS;
  int iturb,i,j,k;
  int ijk;
  int ij;
  float rVec;
  float rVec0;
  float deltaz, deltaz0;
  
  /*Initialize requisite parameters for the RefMag and RefDir calculations*/
  GADsamplingAvgLength = (int) floor(GADrefSampleWindow/dt);    //Determine the number of model timesteps in a sample window (high frequencies filter)
  MPI_Bcast(&GADsamplingAvgLength, 1, MPI_INT, 0, MPI_COMM_WORLD);
  GADsamplingAvgWeight = 1.0/((float) GADsamplingAvgLength); //Precompute the averaging weight across instances in a sample window.
  MPI_Bcast(&GADsamplingAvgWeight, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
  GADrefSeriesWeight = 1.0/((float) GADrefSeriesLength);  //Precompute the averaging weight across the full reference averaging period series of sample average values  
  MPI_Bcast(&GADrefSeriesLength, 1, MPI_INT, 0, MPI_COMM_WORLD); //Broadcast the read-in parameter for series length to all ranks
  MPI_Bcast(&GADrefSeriesWeight, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
//#define DEBUG_TURBCHAR
#ifdef DEBUG_TURBCHAR
  printf("%d/%d: GADsamplingAvgLength=%d, GADsamplingAvgWeight=%f, GADrefSeriesLength=%d, GADrefSeriesWeight=%f\n",
         mpi_rank_world,mpi_size_world,GADsamplingAvgLength,GADsamplingAvgWeight,GADrefSeriesLength,GADrefSeriesWeight);
  fflush(stdout);
#endif

  for(iturb = 0; iturb < GADNumTurbines; iturb++){
    rVec0 = 99999.9999;
    deltaz0 = 99999.9999;
      GAD_turbineRank[iturb] = -999;  //Initialize to special "absent" value of -999
      GAD_turbineRefi[iturb] = -999;  //Initialize to special "absent" value of -999
      GAD_turbineRefj[iturb] = -999;  //Initialize to special "absent" value of -999
      GAD_turbineRefk[iturb] = -999;  //Initialize to special "absent" value of -999
    if(inFile == NULL){
      GAD_turbineRefMag[iturb] = 0.0; //Standard initialization to zero
      GAD_turbineRefDir[iturb] = 0.0; //Standard initialization to zero
      GAD_turbineYawing[iturb] = 0;   //Initialize to all turbines not rotating
      GAD_yawError[iturb] = 0.0;      //Initialize to zero yaw error
      GAD_anFactor[iturb] = 0.0;      //Initialize to zero axial induction factor
    }//end if inFile == NULL
    for(i=iMin-Nh; i < iMax+Nh; i++){
      for(j=jMin-Nh; j < jMax+Nh; j++){
        for(k=kMin-Nh; k < kMax+Nh; k++){
           ijk = i*(Nyp+2*Nh)*(Nzp+2*Nh)+j*(Nzp+2*Nh)+k;
           ij = i*(Nyp+2*Nh)+j;
           rVec = sqrt( pow((GAD_Xcoords[iturb]-xPos[ijk]),2.0)
                       +pow((GAD_Ycoords[iturb]-yPos[ijk]),2.0));
           if(rVec <= sqrt(pow(dX,2.0)+pow(dY,2.0))){ //Should be a candiate gridcell for (nacelle center) reference location
	     if(rVec <= rVec0){
	       if(rVec < rVec0){
	         rVec0 = rVec;
	         GAD_turbineRank[iturb] = mpi_rank_world; 
                 GAD_turbineRefi[iturb] = i;
	         GAD_turbineRefj[iturb] = j;
	       }
	       deltaz = sqrt(pow((GAD_hubHeights[GAD_turbineType[iturb]]-(zPos[ijk]-topoPos[ij])),2.0));
#ifdef DEBUG_TURBCHAR
               printf("%d/%d: deltaz = %f, 0.5/(J33[ijk]*dZi)) = %f\n",
                      mpi_rank_world,mpi_size_world, deltaz, 0.5/(J33[ijk]*dZi));
#endif
               if(deltaz <= 0.5/(J33[ijk]*dZi)){         // 1/(J33[ijk]*dZi)) = dz of cell
		 deltaz0 = deltaz;      
		 GAD_turbineRefk[iturb] = k;      
               }//end if vertical delta < dz ...
             }//end if rVec < rVec0...
           }//end if rVec...
        } //end for(k...
      } // end for(j...
    } // end for(i...
    //dummy update here making use of deltaz0 avoiding compiler warning for an usused variable
    deltaz = deltaz0;
#ifdef DEBUG_TURBCHAR
    if(GAD_turbineRank[iturb] == mpi_rank_world){
      printf("%d/%d: Turbine %d has determined nacelle center cell @ %d,%d,%d with rVec0 = %f and deltaz0 = %f\n",
             mpi_rank_world,mpi_size_world,iturb,GAD_turbineRefi[iturb],GAD_turbineRefj[iturb],GAD_turbineRefk[iturb],rVec0, deltaz0);
      fflush(stdout);
    }
#endif
  }// end for iturb...

  //Having determined the mpi_rank of each turbine, zero out the 
  //GAD_rotorTheta and other elements for any turbines which do NOT belong to a given mpi_rank_world
  //This allows subsequent MPI_Reduce in IO using MPI_SUM op to update GAD_rotoTheta with time  
  for(iturb = 0; iturb < GADNumTurbines; iturb++){
     if(GAD_turbineRank[iturb] != mpi_rank_world){
       GAD_rotorTheta[iturb] = 0.0;
       GAD_turbineRefMag[iturb] = 0.0;
       GAD_turbineRefDir[iturb] = 0.0;
       GAD_turbineYawing[iturb] = 0;
       GAD_yawError[iturb] = 0.0;
       GAD_anFactor[iturb] = 0.0;
     }
  }// end for iturb...

  return(errorCode);
}//end int GADInitTurbineRefChars() 

/*----->>>>> int GADCreateTurbineVolMask();   ----------------------------------------------------------------------
* This function creates the swept-volume mask (of turbine IDs as floats) for the turbine array
*/
int GADCreateTurbineVolMask(){
  int errorCode = GAD_SUCCESS;
  int iturb,i,j,k;
  int ijk;
  int ij;
  float turbineID;
  float rVec;
  char fldName[64];

  GAD_turbineVolMask = memAllocateFloat3DField(Nxp, Nyp, Nzp, Nh, "GAD_turbineVolMask");
  sprintf(&fldName[0],"GAD_turbineVolMask");
  errorCode = ioRegisterVar(&fldName[0], "float", 4, dims4d, &GAD_turbineVolMask[0]);
  errorCode = ioAddStandardAttrs("GAD_turbineVolMask", "-", "Turbine Volume Mask", NULL);
  if(mpi_rank_world == 0){
    printf("GADCreateTurbineVolMask: %s stored at %p, has been registered with IO.\n",
    &fldName[0],&GAD_turbineVolMask[0]);
    fflush(stdout);
  }
  turbineID = 0.0;
  for(iturb = 0; iturb < GADNumTurbines; iturb++){
    turbineID = (float) (iturb+1.0);
    for(i=iMin-Nh; i < iMax+Nh; i++){
      for(j=jMin-Nh; j < jMax+Nh; j++){
        for(k=kMin-Nh; k < kMax+Nh; k++){
           ijk = i*(Nyp+2*Nh)*(Nzp+2*Nh)+j*(Nzp+2*Nh)+k;
           ij = i*(Nyp+2*Nh)+j;
           rVec = sqrt( pow((GAD_Xcoords[iturb]-xPos[ijk]),2.0)
                       +pow((GAD_Ycoords[iturb]-yPos[ijk]),2.0)
                       +pow((GAD_hubHeights[GAD_turbineType[iturb]]-(zPos[ijk]-topoPos[ij])),2.0)); 
           if(rVec <= (0.5*GAD_rotorD[GAD_turbineType[iturb]])){
             GAD_turbineVolMask[ijk] = turbineID;
           }//end if rVec... 
        } //end for(k...
      } // end for(j...
    } // end for(i... 
  }// end for iturb...
  if (GADoutputForces == 1){
    GAD_forceX = memAllocateFloat3DField(Nxp, Nyp, Nzp, Nh, "GAD_forceX");
    sprintf(&fldName[0],"GAD_forceX");
    errorCode = ioRegisterVar(&fldName[0], "float", 4, dims4d, &GAD_forceX[0]);
    errorCode = ioAddStandardAttrs("GAD_forceX", "N m-3", "Turbine Forces in X-Direction", NULL);
    
    GAD_forceY = memAllocateFloat3DField(Nxp, Nyp, Nzp, Nh, "GAD_forceY");
    sprintf(&fldName[0],"GAD_forceY");
    errorCode = ioRegisterVar(&fldName[0], "float", 4, dims4d, &GAD_forceY[0]);
    errorCode = ioAddStandardAttrs("GAD_forceY", "N m-3", "Turbine Forces in Y-Direction", NULL);
    
    GAD_forceZ = memAllocateFloat3DField(Nxp, Nyp, Nzp, Nh, "GAD_forceZ");
    sprintf(&fldName[0],"GAD_forceZ");
    errorCode = ioRegisterVar(&fldName[0], "float", 4, dims4d, &GAD_forceZ[0]);
    errorCode = ioAddStandardAttrs("GAD_forceZ", "N m-3", "Turbine Forces in Z-Direction", NULL);
  }

  return(errorCode);
}//end GADCreateTurbineVolMask()

/*----->>>>> int GADCreateTurbineRotorMask()   ----------------------------------------------------------------------
* This function creates the yaw-specific rotor-disk mask for turbines in the simulation
*/
int GADCreateTurbineRotorMask(){
  int errorCode = GAD_SUCCESS;
  char fldName[64];

  GAD_turbineRotorMask = memAllocateFloat3DField(Nxp, Nyp, Nzp, Nh, "GAD_turbineRotorMask");
  sprintf(&fldName[0],"GAD_turbineRotorMask");
  errorCode = ioRegisterVar(&fldName[0], "float", 4, dims4d, &GAD_turbineRotorMask[0]);
  errorCode = ioAddStandardAttrs("GAD_turbineRotorMask", "-", "Turbine Rotor-Disk Mask", NULL);
  if(mpi_rank_world == 0){
    printf("GADCreateTurbineRotorMask: %s stored at %p, has been registered with IO.\n",
    &fldName[0],&GAD_turbineRotorMask[0]);
    fflush(stdout);
  }

  errorCode = GADUpdateTurbineRotorMask();

  return(errorCode);
}//end GADCreateTurbineRotorMask()

/*----->>>>> int GADUpdateTurbineRotorMask()   ----------------------------------------------------------------------
* This function updates (from GAD_rotorTheta) the yaw-specific rotor-disk mask for turbines in the simulation
*/
int GADUpdateTurbineRotorMask(){
  int errorCode = GAD_SUCCESS;
  int iturb,i,j,k;
  int ijk;
  int ij;
  float mask_value;
  float pi = 3.1415926535;
#define CELLINROTOR
#ifdef CELLINROTOR
  float x_hat[3];
  float dr[3];
  float tiltAngle = 0.0;
#else
  float x1,x2,x3,y1,y2,y3;
  float parallelDist;
#endif
  float perpdx_rot;
  float perpDist;
  float rVec;
  void  *memsetReturnVal;
  
  /*Reset the mask to zero everywhere*/
  memsetReturnVal = memset(GAD_turbineRotorMask,0,(Nxp+2*Nh)*(Nyp+2*Nh)*(Nzp+2*Nh)*sizeof(float));
  if(memsetReturnVal == NULL){
       fprintf(stderr, "Rank %d/%d GADUpdateTurbineRotorMask():WARNING memsetReturnVal == NULL!\n",
               mpi_rank_world,mpi_size_world);
  } 
  for(iturb = 0; iturb < GADNumTurbines; iturb++){
    for(i=iMin-Nh; i < iMax+Nh; i++){
      for(j=jMin-Nh; j < jMax+Nh; j++){
        for(k=kMin-Nh; k < kMax+Nh; k++){
           ijk = i*(Nyp+2*Nh)*(Nzp+2*Nh)+j*(Nzp+2*Nh)+k;
           ij = i*(Nyp+2*Nh)+j;
           mask_value = 0.0;
#ifdef CELLINROTOR
	   //Unit horizontal vector normal to the rotor-disk plane
           x_hat[0] = cosf(tiltAngle*pi/180.0)*cosf(GAD_rotorTheta[iturb]*pi/180.0);
           x_hat[1] = cosf(tiltAngle*pi/180.0)*sinf(GAD_rotorTheta[iturb]*pi/180.0);
           x_hat[2] = -sinf(tiltAngle*pi/180.0);

           //Vector from nacelle center to current grid point
           dr[0] = xPos[ijk]-GAD_Xcoords[iturb];
           dr[1] = yPos[ijk]-GAD_Ycoords[iturb];
           dr[2] = (zPos[ijk]-topoPos[ij])-GAD_hubHeights[GAD_turbineType[iturb]];
    
           //Perpendicular distance from nacelle-center to current grid point (normal to the rotor-disk plane)
           perpDist = dr[0]*x_hat[0] + dr[1]*x_hat[1] + dr[2]*x_hat[2];

           //Proper radial distance of blade segment (accounts for tilted rotor?) 
	   rVec = sqrtf( powf(dr[0]-perpDist*x_hat[0],2.0)
                        +powf(dr[1]-perpDist*x_hat[1],2.0)
                        +powf(dr[2]-perpDist*x_hat[2],2.0) );
#else
           /* Define the rotor plane */
           x1 = GAD_Xcoords[iturb] - 0.5*GAD_rotorD[GAD_turbineType[iturb]]*cos(0.5*pi + GAD_rotorTheta[iturb]*pi/180.0);
           y1 = GAD_Ycoords[iturb] - 0.5*GAD_rotorD[GAD_turbineType[iturb]]*sin(0.5*pi + GAD_rotorTheta[iturb]*pi/180.0);
           x2 = GAD_Xcoords[iturb] + 0.5*GAD_rotorD[GAD_turbineType[iturb]]*cos(0.5*pi + GAD_rotorTheta[iturb]*pi/180.0);
           y2 = GAD_Ycoords[iturb] + 0.5*GAD_rotorD[GAD_turbineType[iturb]]*sin(0.5*pi + GAD_rotorTheta[iturb]*pi/180.0);
           x3 = fabs(GAD_Xcoords[iturb]-xPos[ijk])*cos(0.5*pi + GAD_rotorTheta[iturb]*pi/180.0);
           y3 = fabs(GAD_Ycoords[iturb]-yPos[ijk])*sin(0.5*pi + GAD_rotorTheta[iturb]*pi/180.0);

           /*Find the perpendicular distance from this i,j,k cell center to the rotor-plane*/
           perpDist = fabs( (x2-x1)*(y1-yPos[ijk]) - (x1-xPos[ijk])*(y2-y1) )/sqrt(pow((x2-x1),2.0) + pow((y2-y1),2.0));
           parallelDist = sqrt(pow(x3,2.0)+pow(y3,2.0)); 
           /*Recalculate the radial vector of the yaw-projected rotor disk...*/
           rVec = sqrt(pow(parallelDist,2.0) + pow((GAD_hubHeights[GAD_turbineType[iturb]]-(zPos[ijk]-topoPos[ij])),2.0));
#endif
           /*Define ithe perpendicular "dx" in the rotated "x-y" plane  */
           perpdx_rot =  fabs(dX*cosf(0.5*pi + GAD_rotorTheta[iturb]*pi/180.0))
                       + fabs(dY*sinf(0.5*pi + GAD_rotorTheta[iturb]*pi/180.0));

           if(   (fabs(perpDist) < ((float) numgridCells_away)*perpdx_rot)
              && rVec <= (0.5*GAD_rotorD[GAD_turbineType[iturb]])
	      && rVec >  (0.5*GAD_nacelleD[GAD_turbineType[iturb]]) ){
//#define DEBUG_GAD_UPDATEROTOR
#ifdef DEBUG_GAD_UPDATEROTOR
             //float m,b;
             if((mpi_rank_world == 6) && (k==20) ){
              // printf("GADCreateTurbineRotorMask: Turbine %d @ %d,%d,%d: perpDist = %f , perpdx_rot = %f, parallelDist = %f, zDist = %f, m = %f, b = %f.\n",
              // iturb,i,j,k,perpDist, perpdx_rot, parallelDist, sqrt(pow((GAD_hubHeights[GAD_turbineType[iturb]]-(zPos[ijk]-topoPos[ij])),2.0)), m, b ) ;
               printf("GADCreateTurbineRotorMask: Turbine %d @ %d,%d,%d: rVec = %f , perpDist = %f , perpdx_rot = %f, zDist = %f.\n",
               iturb,i,j,k, rVec, perpDist, perpdx_rot, sqrtf(powf((GAD_hubHeights[GAD_turbineType[iturb]]-(zPos[ijk]-topoPos[ij])),2.0))) ;
               fflush(stdout);
             }
#endif
              mask_value = 1.0;
              GAD_turbineRotorMask[ijk] = mask_value;
           }
        } //end for(k...
      } // end for(j...
    } // end for(i... 
  }// end for iturb...
  
  return(errorCode);
}//end GADUpdateTurbineRotorMask()

/*----->>>>> int GADDestructor();   ----------------------------------------------------------------------
 * This function frees allocated memory of turbine characteristics arrays in the GAD module
*/
int GADDestructor(){
  int errorCode = GAD_SUCCESS;

  free(GAD_turbineType);
  free(GAD_turbineRank);
  free(GAD_turbineRefi);
  free(GAD_turbineRefj);
  free(GAD_turbineRefk);
  free(GAD_turbineYawing);
  free(GAD_Xcoords);
  free(GAD_Ycoords);
  free(GAD_turbineRefMag);
  free(GAD_turbineRefDir);
  free(GAD_yawError);
  free(GAD_anFactor);
  free(GAD_rotorTheta);
  free(GAD_hubHeights);
  free(GAD_rotorD);
  free(GAD_nacelleD);
  free(turbinePolyTwist);
  free(turbinePolyChord);
  free(turbinePolyPitch);
  free(turbinePolyOmega);

  return(errorCode);
}//end GADDestructor()

/*----->>>>> int GADCleanup();  ----------------------------------------------------------------------
* Used to free all malloced memory by the GAD module.
*/
int GADCleanup(){
   int errorCode = GAD_SUCCESS;

   if(GADSelector > 0){
     /* Free any GAD module arrays */
     errorCode = GADDestructor();
    
     free(GAD_turbineVolMask);
     free(GAD_turbineRotorMask);
   } //end if GADSelector > 0

   return(errorCode);
}//end GADCleanup()
