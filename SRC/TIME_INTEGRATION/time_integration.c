/* FastEddy®: SRC/TIME_INTEGRATION/time_integration.c 
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
#include <float.h>
#include <math.h>
#include <fempi.h>
#include <parameters.h>
#include <mem_utils.h>
#include <io.h>
#include <grid.h>
#include <time_integration.h>
#include <hydro_core.h> 

/*######################------------------- TIME_INTEGRATION module variable declarations ---------------------#################*/
/* Parameters */
int timeMethod;  // Selector for time integration method. [0=RK1, 1=RK3-WS2002 (default)]
int Nt;          // Number of timesteps to perform
float dt;        // timestep resolution in seconds
int NtBatch;     // Number of timesteps in a batch to perform in a CUDA kernel launch

/*static scalars*/
float simTime; /*Master simulation time*/
int simTime_it; /*Master simulation time step*/ 
int simTime_itRestart; /*Master simulation 'Restart' time step*/
int numRKstages; /* number of stages in the time scheme */

/* array fields */


/*######################------------------- TIME_INTEGRATION module function definitions ---------------------#################*/

/*----->>>>> int timeGetParams();       ----------------------------------------------------------------------
* Obtain the complete set of parameters for the TIME_INTEGRATION module
*/
int timeGetParams(){
   int errorCode = TIME_INTEGRATION_SUCCESS;

   /*query for each TIME_INTEGRATION parameter */
   timeMethod = 0;
   errorCode = queryIntegerParameter("timeMethod", &timeMethod, 0, 0, PARAM_OPTIONAL);
   Nt = 1000;
   errorCode = queryIntegerParameter("Nt", &Nt, 1, INT_MAX, PARAM_MANDATORY);
   dt = 1.0;
   errorCode = queryFloatParameter("dt", &dt, FLT_MIN, FLT_MAX, PARAM_MANDATORY);
   NtBatch = 1;
   errorCode = queryIntegerParameter("NtBatch", &NtBatch, 1, Nt, PARAM_MANDATORY);
   return(errorCode);
} //end timeGetParams()

/*----->>>>> int timeInit();       ----------------------------------------------------------------------
Used to broadcast and print parameters, allocate memory, and initialize configuration settings for  the TIME_INTEGRATION module.
*/
int timeInit(){
   int errorCode = TIME_INTEGRATION_SUCCESS;
   char *strelem;
   int strLength;
   char varName[MAX_HC_FLDNAME_LENGTH];

   if(mpi_rank_world == 0){
      printComment("TIME_INTEGRATION parameters---");
      printParameter("timeMethod", "Selector for time integration method. [0=RK3-WS2002 (default)]");
      printParameter("Nt", "Number of timesteps to perform.");
      printParameter("dt", "timestep resolution in seconds.");
      printParameter("NtBatch", "Number of timesteps to compute in batch launch, must have NtBatch <= Nt.");
   } //end if(mpi_rank_world == 0)

   /*Broadcast the parameters across mpi_ranks*/
   MPI_Bcast(&timeMethod, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);
   MPI_Bcast(&Nt, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);
   MPI_Bcast(&dt, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&NtBatch, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);
   
#ifdef DEBUG
   MPI_Barrier(MPI_COMM_WORLD);
   printf("mpi_rank_world--%d/%d Allocating arrays in timeInit()!\n",mpi_rank_world,mpi_size_world);
   fflush(stdout);
#endif
   
   /*Initialize the master simulation time to 0.0*/
   if(inFile == NULL){
     simTime = 0.0;
     simTime_it = 0; 
     simTime_itRestart = simTime_it; 
   }else{
     //Parse the inFile for the Restart iteration
     strelem = strchr(inFile, '.');    //Find the index of the timestep suffix seperator character
     strLength = (int)(strelem-inFile)+1; //set the offset to parse the integer timestep of the inFile
     sscanf(&inFile[strLength],"%d",&simTime_itRestart); 
     simTime_it = simTime_itRestart;
     simTime = dt*simTime_it;
   } //endif inFile==NULL.. else...
   printf("mpi_rank_world--%d/%d: in timeInit(), simTime_it = %d and simTime = %16.6f !\n",
          mpi_rank_world,mpi_size_world,simTime_it,simTime);
   fflush(stdout);

   /*Register a time variable holding "simTime" or the master simulation time*/
   errorCode = sprintf(&varName[0],"time");
   errorCode = ioRegisterVar(&varName[0], "float", 1, dims1dTD, &simTime);
   
   /* Add NetCDF attributes for the time variable */
   errorCode = timeAddTimeAttributes();
   
   printf(":Variable = %s stored at %p, has been registered with IO.\n",
          &varName[0],&simTime);
   fflush(stdout);

   // assign numRKstages and bcast
   if (timeMethod==0) { // 3rd-order Runge-Kutta
     numRKstages = 2;
   }

   /* Done */
   return(errorCode);
} //end timeInit()

/*----->>>>> int timeAddTimeAttributes();       ----------------------------------------------------------------------
* Add NetCDF attributes to time-related variables registered by the TIME_INTEGRATION module.
*/
int timeAddTimeAttributes(){
   int errorCode = TIME_INTEGRATION_SUCCESS;

   /* Add standard CF convention attributes for the time variable */
   errorCode = ioAddStandardAttrs("time", "s", "Simulation time", "time");
   if(errorCode != TIME_INTEGRATION_SUCCESS){
      printf("Error adding standard attributes to time variable: %d\n", errorCode);
      return errorCode;
   }

   errorCode = ioAddVarAttr("time", "axis", "text", "T");
   if(errorCode != TIME_INTEGRATION_SUCCESS){
      printf("Error adding axis attribute to time variable: %d\n", errorCode);
      return errorCode;
   }

   return errorCode;
} //end timeAddTimeAttributes()

/*----->>>>> int timeIntBdyPlaneUpdates();       ----------------------------------------------------------------------
 * Used to broadcast and print parameters, allocate memory, and initialize configuration settings 
 * for the TIME_INTEGRATION module.
 */
int timeIntBdyPlaneUpdates(){
    int errorCode = TIME_INTEGRATION_SUCCESS;

    //Time integration wrapped call to hydro_core BdyPlane BCs update
    errorCode = hydro_coreReadNextBndyPlanesFile();

    return(errorCode);
} //end timeIntBdyPlaneUpdates(()
  
/*----->>>>> int timeCleanup();       ----------------------------------------------------------------------
Used to free all malloced memory by the TIME_INTEGRATION module.
*/
int timeCleanup(){
   int errorCode = TIME_INTEGRATION_SUCCESS;

   /* Free any TIME_INTEGRATION module arrays */
   //currently none

   return(errorCode);

}//end timeCleanup()
