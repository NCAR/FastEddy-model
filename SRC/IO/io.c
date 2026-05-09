/* FastEddy®: SRC/IO/io.c 
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
#include <fempi.h>
#include <parameters.h>
#include <ioVarsList.h>
#include <io.h>

#define MAXLEN 256
int dimids[MAXDIMS];
size_t count[MAXDIMS];
size_t start[MAXDIMS];
size_t count2d[MAXDIMS];
size_t start2d[MAXDIMS];
size_t count2dTD[MAXDIMS];
size_t start2dTD[MAXDIMS];

  
// Include the netCDF-centric source code
#include <io_netcdf.c>

// Include the unformatted N-to-N binary-centric source code
#include <io_binary.c>

/*######################------------------- IO module variable definitions ---------------------#################*/
/* Parameters */
int ioOutputMode;  /*0: N-to-1 gather and write to a netCDF file, 1: N-to-N writes of FastEddy binary files*/
char *outPath;     /* Directory Path where output files are to be written */
char *outFileBase; /* Base name of the output file series as in (outFileBase).element-in-series */
char *inPath;      /* Directory Path where input files are to be read from */
char *inFile;      /* Name of the input file */
int frqOutput;     /*frequency (in timesteps) at which to produce output; should be an even multiple of NtBatch*/

/*static Variables*/
char *outSubString; /*subString portion of outFile holding element-in-series as in path/base.substring */
char *outFileName;      /*full name instance of outFileName =  path/base.substring */
char *inFileName;      /*full name instance of inFileName =  path/infile */
int registeredVars;   /* Total number of variables registered with the primary ioVarsList */
int registered3dVars; /* Number of 3-dimensional variables registered with the primary ioVarsList */
int registered2dVars; /* Number of 2-dimensional variables registered with the primary ioVarsList */

/*IO-Buffers*/
float *ioBuffField;
float *ioBuffFieldTransposed;
float *ioBuffFieldRho;
float *ioBuffFieldTransposed2D;
int *ioBuffFieldInt;

int nz_varid;
int ny_varid;
int nx_varid;

/*IO-profiles*/
int towerIOSelector;
char *towerSpecsFile;
char *towerPath;     /* Directory Path where tower files are to be written */
int nProfs;
ioProfiles_t towerProfiles;

/*######################------------------- IO module function definitions ---------------------#################*/

/*----->>>>> int ioGetParams();       ----------------------------------------------------------------------

Obtain the complete set of parameters for the IO module

*/
int ioGetParams(){
   int errorCode = IO_SUCCESS;

   /*query for each IO parameter */
   ioOutputMode=0; //default  = 0
   errorCode = queryIntegerParameter("ioOutputMode", &ioOutputMode, 0, 1, PARAM_OPTIONAL);
   errorCode = queryPathParameter("inPath", &inPath, PARAM_OPTIONAL);
   errorCode = queryStringParameter("inFile", &inFile, PARAM_OPTIONAL);
   errorCode = queryPathParameter("outPath", &outPath, PARAM_MANDATORY);
   errorCode = queryStringParameter("outFileBase", &outFileBase, PARAM_MANDATORY);
   errorCode = queryIntegerParameter("frqOutput", &frqOutput, 0, INT_MAX, PARAM_MANDATORY);
   towerIOSelector = 0; // Default to 0
   errorCode = queryIntegerParameter("towerIOSelector", &towerIOSelector, 0, 1, PARAM_OPTIONAL);
   if(towerIOSelector > 0){
     errorCode = queryFileParameter("towerSpecsFile", &towerSpecsFile, PARAM_MANDATORY);
     errorCode = queryPathParameter("towerPath", &towerPath, PARAM_MANDATORY);
   }
   return(errorCode);
} //end ioGetParams()

/*----->>>>> int ioInit();       ----------------------------------------------------------------------
Used to broadcast and print parameters, allocate memory, and initialize configuration settings for  the IO module.
*/
int ioInit(){
   int errorCode = IO_SUCCESS;
   int strLength;
 
   if(mpi_rank_world == 0){
      printComment("IO parameters---");
      printParameter("ioOutputMode", "0: N-to-1 gather and write to a netCDF file, 1:N-to-N writes of FastEddy binary files");
      printParameter("inPath", "Path where initial/restart file is read in from");
      printParameter("inFile", "name of the input file for coordinate system and initial or restart conditions");
      printParameter("outPath", "Path where output files are to be written");
      printParameter("outFileBase", "Base name of the output file series as in (outFileBase).element-in-series");
      printParameter("frqOutput", "frequency (in timesteps) at which to produce output; should be an even multiple of NtBatch");
      printParameter("towerIOSelector", "Virtual Tower IO-Selector: 0=off, 1=on ");
      if(towerIOSelector > 0){
        printParameter("towerSpecsFile", "netCDF file with virtual tower IO specifications ");
        printParameter("towerPath", "Path where tower files are to be written");
      }
   } //end if(mpi_rank_world == 0) 

   /*Broadcast the parameters across mpi_ranks*/
   /*ioOutputMode*/
   MPI_Bcast(&ioOutputMode, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);
   /*inPath string*/
   strLength = 0;
   if(mpi_rank_world == 0){
      if(inPath != NULL){
         strLength = strlen(inPath)+1;
      }else{
         strLength = 0;
      }
   } //end if(mpi_rank_world == 0)
   MPI_Bcast(&strLength, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);
   if(mpi_rank_world != 0){
      inPath = (char *) malloc(strLength*sizeof(char));
   } //if a non-root mpi_rank
   MPI_Bcast(inPath, strLength, MPI_CHARACTER, 0, MPI_COMM_WORLD);
     
   /*inFile string*/
   strLength = 0;
   if(mpi_rank_world == 0){
      if(inFile != NULL){
         strLength = strlen(inFile)+1;
      }else{
         strLength = 0;
      }
   } //end if(mpi_rank_world == 0)
   MPI_Bcast(&strLength, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);
   if(mpi_rank_world != 0){
      inFile = (char *) malloc(strLength*sizeof(char));
   } //if a non-root mpi_rank
   if(strLength > 0){
      MPI_Bcast(inFile, strLength, MPI_CHARACTER, 0, MPI_COMM_WORLD);
   }else{
      if(mpi_rank_world != 0){
         inFile = NULL;
      } //if a non-root mpi_rank
   }
   printf("mpi_rank_world--%d/%d inFile = %s !!\n",mpi_rank_world,mpi_size_world,inFile);
   fflush(stdout);
  
   /*outPath string*/
   strLength = 0;
   if(mpi_rank_world == 0){
      if(outPath != NULL){
         strLength = strlen(outPath)+1;
      }else{
         strLength = 0;
      }
   } //end if(mpi_rank_world == 0)
   MPI_Bcast(&strLength, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);
   if(mpi_rank_world != 0){
      outPath = (char *) malloc(strLength*sizeof(char));
   } //if a non-root mpi_rank
   MPI_Bcast(outPath, strLength, MPI_CHARACTER, 0, MPI_COMM_WORLD);
    
   /*outFileBase string*/
   strLength = 0;
   if(mpi_rank_world == 0){
      if(outFileBase != NULL){
         strLength = strlen(outFileBase)+1;
      }else{
         strLength = 0;
      }
   } //end if(mpi_rank_world == 0)
   MPI_Bcast(&strLength, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);
   if(mpi_rank_world != 0){
      outFileBase = (char *) malloc(strLength*sizeof(char));
   } //if a non-root mpi_rank
   MPI_Bcast(outFileBase, strLength, MPI_CHARACTER, 0, MPI_COMM_WORLD);
   /*frqOutput*/
   MPI_Bcast(&frqOutput, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);
   /*towerIOSelector*/
   MPI_Bcast(&towerIOSelector, 1, MPI_INT, 0, MPI_COMM_WORLD);
   if(towerIOSelector > 0){
      strLength = 0;
      if(mpi_rank_world == 0){
        if(towerSpecsFile != NULL){
          strLength = strlen(towerSpecsFile)+1;
        }else{
          strLength = 0;
        }
      } //end if(mpi_rank_world == 0)
      MPI_Bcast(&strLength, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);
      if(strLength > 0){
        if(mpi_rank_world != 0){
           towerSpecsFile = (char *) malloc(strLength*sizeof(char));
        } //if a non-root mpi_rank
        MPI_Bcast(towerSpecsFile, strLength, MPI_CHARACTER, 0, MPI_COMM_WORLD);
      }
      /*towerPath string*/
      strLength = 0;
      if(mpi_rank_world == 0){
         if(towerPath != NULL){
            strLength = strlen(towerPath)+1;
         }else{
            strLength = 0;
         }
      } //end if(mpi_rank_world == 0)
      MPI_Bcast(&strLength, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);
      if(mpi_rank_world != 0){
         towerPath = (char *) malloc(strLength*sizeof(char));
      } //if a non-root mpi_rank
      MPI_Bcast(towerPath, strLength, MPI_CHARACTER, 0, MPI_COMM_WORLD);
   }
   /*end-- Broadcast the parameters... */

   /*Allocate IO private arrays*/
   inFileName = (char *) malloc(3*MAX_LEN*sizeof(char)); /* 3 for each part of path/base.subString */
   outSubString = (char *) malloc(MAX_LEN*sizeof(char));
   outFileName = (char *) malloc(3*MAX_LEN*sizeof(char)); /* 3 for each part of path/base.subString */

   return(errorCode);
} //end ioInit()

/*----->>>>> int ioAllocateBuffers();   -----------------------------------
* Allocate memory to io-buffers for reading/writing IO-registered fields
*/
int ioAllocateBuffers(int globalNx, int globalNy, int globalNz){
   int errorCode = IO_SUCCESS;
   int numElems;
   int numElems2D;

   numElems = globalNx*globalNy*globalNz;
   numElems2D = globalNx*globalNy;
   if(mpi_rank_world==0){
     ioBuffField = (float *) malloc(numElems*sizeof(float));
     ioBuffFieldTransposed = (float *) malloc(numElems*sizeof(float));
     ioBuffFieldRho = (float *) malloc(numElems*sizeof(float));
     ioBuffFieldTransposed2D = (float *) malloc(numElems2D*sizeof(float));
     ioBuffFieldInt = (int *) malloc(numElems*sizeof(int));
   } //endif mpi_Rank_world==0
   return(errorCode);
} //end ioAllocateBuffers()

/*----->>>>> int ioProfilePreparations();   ------------------------------------------------------------
 * Profile Preparations routine for the IO module. Includes counting registered variables, reading 
 * profiles, allocating associated arrays,
 * and...
 */
int ioProfilePreparations(){
  int errorCode = IO_SUCCESS;
  int ncid; 
  int ncfldid;
  int dimids[64];
  size_t count[64];
  size_t start[64];
  char fldName[64];

  int iprofile;

  //Count the total number of variables registered with IO
  registeredVars = countVarsInList(&registered3dVars,&registered2dVars);
  printf("mpi_rank_world--%d/%d ioProfilePreparations(): There are %d total registered variables in the primary ioVarsList.\n",
	 mpi_rank_world,mpi_size_world,registeredVars);
  printf("mpi_rank_world--%d/%d ioProfilePreparations(): There are %d registered 3-dimensional variables in the primary ioVarsList.\n",
	 mpi_rank_world,mpi_size_world,registered3dVars);
  printf("mpi_rank_world--%d/%d ioProfilePreparations(): There are %d registered 2-dimensional variables in the primary ioVarsList.\n",
	 mpi_rank_world,mpi_size_world,registered2dVars);
  fflush(stdout);
#if 0
  if(mpi_rank_world == 0){ 
     errorCode = printList(); 
  }
  fflush(stdout); 
#endif

  /* ------------------------  Read in the towerSpecsFile -----------------------------*/
  //Root-rank should read the netcdf towerSpecsFile
  if(mpi_rank_world == 0){  
    //Open the netcdf towerSpecsFile 
    errorCode = ioOpenNetCDFinFile(towerSpecsFile, &ncid);
    if(errorCode > 0){
      printf("Failed to open towerSpecsFile = %s EXITING NOW!!!!\n",towerSpecsFile);
      fflush(stdout);
      exit(0);
    }
    //Inquire for the dimID of the towers input specification fundamental parameter, nProfs
    if((errorCode = nc_inq_dimid(ncid, "nProfs", &dimids[0]))){
      ERR(errorCode);
    }
    //Inquire for the value of nProfs 
    if((errorCode = nc_inq_dimlen(ncid, dimids[0], &count[dimids[0]]))){
      ERR(errorCode);
    }
    //Assign the nProfs to the value of the netCDF file dimension  
    nProfs = (int)count[dimids[0]];
  }//end if mpi_rank_world == 0
  MPI_Bcast(&nProfs, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);
  if(nProfs > 0){ 
    towerProfiles.profIDs = malloc(nProfs*sizeof(int));
    towerProfiles.mpi_ranks = malloc(nProfs*sizeof(int));
    towerProfiles.coordsSN = malloc(nProfs*sizeof(float));
    towerProfiles.coordsWE = malloc(nProfs*sizeof(float));
    towerProfiles.coordsLat = malloc(nProfs*sizeof(double));
    towerProfiles.coordsLon = malloc(nProfs*sizeof(double));
  }
  if(mpi_rank_world == 0){
    //Setup the profIDs (these are common profile indices, shared across all ranks)
    for(iprofile = 0; iprofile < nProfs; iprofile++){
       towerProfiles.profIDs[iprofile] = iprofile;
    }
    start[0] = 0;
    sprintf(fldName,"coordType");
    if ( (errorCode = nc_inq_varid(ncid, fldName, &ncfldid)) ){
       ERR(errorCode);
    } //if nc_inq_varid
    if ((errorCode = nc_get_var_int(ncid, ncfldid, &towerProfiles.coordType )) ){
       ERR(errorCode);
    }
    if(towerProfiles.coordType == 0){
      sprintf(fldName,"coordsLat");
      if ( (errorCode = nc_inq_varid(ncid, fldName, &ncfldid)) ){
         ERR(errorCode);
      } //if nc_inq_varid
      if ((errorCode = nc_get_vara_double(ncid, ncfldid, &start[0], &count[dimids[0]], towerProfiles.coordsLat )) ){
         ERR(errorCode);
      }
      sprintf(fldName,"coordsLon");
      if ( (errorCode = nc_inq_varid(ncid, fldName, &ncfldid)) ){
         ERR(errorCode);
      } //if nc_inq_varid
      if ((errorCode = nc_get_vara_double(ncid, ncfldid, &start[0], &count[dimids[0]], towerProfiles.coordsLon )) ){
         ERR(errorCode);
      }
    }else{
      sprintf(fldName,"coordsSN");
      if ( (errorCode = nc_inq_varid(ncid, fldName, &ncfldid)) ){
         ERR(errorCode);
      } //if nc_inq_varid
      if ((errorCode = nc_get_vara_float(ncid, ncfldid, &start[0], &count[dimids[0]], towerProfiles.coordsSN )) ){
         ERR(errorCode);
      }
      sprintf(fldName,"coordsWE");
      if ( (errorCode = nc_inq_varid(ncid, fldName, &ncfldid)) ){
         ERR(errorCode);
      } //if nc_inq_varid
      if ((errorCode = nc_get_vara_float(ncid, ncfldid, &start[0], &count[dimids[0]], towerProfiles.coordsWE )) ){
         ERR(errorCode);
      }
    }//end if coordType == 0, else
  } //end if mpi_rank_world == 0
  MPI_Bcast(towerProfiles.profIDs, nProfs, MPI_INTEGER, 0, MPI_COMM_WORLD);
  MPI_Bcast(&towerProfiles.coordType, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);

  if(towerProfiles.coordType == 0){
    MPI_Bcast(towerProfiles.coordsLat, nProfs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(towerProfiles.coordsLon, nProfs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    for(iprofile = 0; iprofile< nProfs; iprofile++){
      printf("mpi_rank_world--%d/%d: profIDs[%d] = %d-- coordType = %d, coords[%d](lat,lon) = (%f,%f)\n",
             mpi_rank_world,mpi_size_world,
  	     iprofile,towerProfiles.profIDs[iprofile],
  	     towerProfiles.coordType,
             iprofile,towerProfiles.coordsLat[iprofile],towerProfiles.coordsLon[iprofile]);
    }
  }else{
    MPI_Bcast(towerProfiles.coordsSN, nProfs, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(towerProfiles.coordsWE, nProfs, MPI_FLOAT, 0, MPI_COMM_WORLD);
    for(iprofile = 0; iprofile< nProfs; iprofile++){
      printf("mpi_rank_world--%d/%d: profIDs[%d] = %d-- coordType = %d, coords[%d](y,x) = (%f,%f)\n",
             mpi_rank_world,mpi_size_world,
  	     iprofile,towerProfiles.profIDs[iprofile],
  	     towerProfiles.coordType,
             iprofile,towerProfiles.coordsSN[iprofile],towerProfiles.coordsWE[iprofile]);
    }
  }//end if coordType == 0, else 
  fflush(stdout);
  return(errorCode);
} //end ioProfilePreparations()

/*----->>>>> int ioCleanupProfiles();       ----------------------------------------------------------------------
Used to free all malloced memory associated with Priofiles output from the IO module.
*/
int ioCleanupProfiles(){
   int errorCode = IO_SUCCESS;
   
   if(towerIOSelector > 0 && nProfs > 0){
     //SOA
     free(towerProfiles.profIDs);
     free(towerProfiles.mpi_ranks);
     free(towerProfiles.coordsSN);
     free(towerProfiles.coordsWE);
     free(towerProfiles.coordsLat);
     free(towerProfiles.coordsLon);
     free(towerSpecsFile);
     free(towerPath);
   }
   return(errorCode);

}//end ioCleanupProfiles()

/*----->>>>> int ioCleanup();       ----------------------------------------------------------------------
Used to free all malloced memory by the IO module.
*/
int ioCleanup(){
   int errorCode = IO_SUCCESS;

   errorCode = ioCleanupProfiles();
   
   /*free the io-buffers*/
   if(mpi_rank_world == 0){
     free(ioBuffField);
     free(ioBuffFieldTransposed);
     free(ioBuffFieldRho);
     free(ioBuffFieldTransposed2D);
     free(ioBuffFieldInt);
   } //end if mpi_rank_world == 0
   /*free the registry list*/
   destroyList();

   /* Free any IO module arrays */
   free(outFileName);
   free(outSubString);
   return(errorCode);

}//end ioCleanup()

/*----->>>>> int ioRegisterVar();    ---------------------------------------------------------------------
* Used by other modules to register a variable in the IO module list of variables to read/write as input/output.
*/
int ioRegisterVar(char *name, char *type, int nDims, int *dimids, void *varMemAddress){
    int errorCode = IO_SUCCESS;
    int tmperrorCode = 0;
/* The supplied values of dimids here should
* always assume to define the dimensions in this order (time), X, Y, Z.
* because the dims will be defined in that order for the netCDF/binary files.
* For now time is omitted,so the dimids for a 3-D field in our 
* X,Y,Z space should be  0,1,2 respectively*/

    tmperrorCode = addVarToList(name, type, nDims, dimids, varMemAddress);
    if(tmperrorCode!=0){   //Handle any error from the variable addition
      printf("ERROR = %d returned by addVarToList()...",tmperrorCode);
    }
    return(errorCode);
} //end ioRegisterVar()

/*----->>>>> int ioAddVarAttr(); -------------------------------------------------------------------
* Add a single attribute to an already registered variable
*/
int ioAddVarAttr(char *varName, char *attrName, char *attrType, char *attrValue){
    int errorCode = IO_SUCCESS;
    int tmperrorCode = 0;

    /* Check if variable exists first */
    if(getNamedVarFromList(varName) == NULL){
        printf("ERROR: Variable %s not found in registry...", varName);
        return IO_ERROR_VAR_NOT_FOUND;
    }

    /* Add the attribute */
    tmperrorCode = addAttrToVar(varName, attrName, attrType, attrValue);
    if(tmperrorCode != 0){
        printf("ERROR = %d returned by addAttrToVar() for attribute %s...", tmperrorCode, attrName);
        return IO_ERROR_ATTR_ADD;
    }

    return errorCode;
}

/*----->>>>> int ioAddStandardAttrs(); -------------------------------------------------------------
* Add standard CF convention attributes to a variable (units, long_name, standard_name)
*/
int ioAddStandardAttrs(char *varName, char *units, char *longName, char *standardName){
    int errorCode = IO_SUCCESS;
    int tmperrorCode = 0;

    /* Check if variable exists first */
    if(getNamedVarFromList(varName) == NULL){
        printf("ERROR: Variable %s not found in registry...", varName);
        return IO_ERROR_VAR_NOT_FOUND;
    }

    /* Add the standard attribute using the ioVarsList function */
    tmperrorCode = addStandardAttrsToVar(varName, units, longName, standardName);
    if(tmperrorCode != 0){
        printf("ERROR = %d returned by addStandardAttrsToVar()...", tmperrorCode);
        return IO_ERROR_ATTR_ADD;
    }

    return errorCode;
}
