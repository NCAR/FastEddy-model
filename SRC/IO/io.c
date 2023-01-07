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

/*######################------------------- IO module variable definitions ---------------------#################*/
/* Parameters */
int ioOutputMode;  /*0: N-to-1 gather and write to a netcdf file, 1:N-to-N writes of FastEddy binary files*/
char *outPath;     /* Directory Path where output files are to be written */
char *outFileBase; /* Base name of the output file series as in (outFileBase).element-in-series */
char *inPath;      /* Directory Path where input files are to be read from */
char *inFile;      /* Name of the input file */
int frqOutput;     /*frequency (in timesteps) at which to produce output*/

/*static Variables*/
char *outSubString; /*subString portion of outFile holding element-in-series as in path/base.substring */
char *outFileName;      /*full name instance of outFileName =  path/base.substring */
char *inFileName;      /*full name instance of inFileName =  path/infile */

/*IO-Buffers*/
float *ioBuffField;
float *ioBuffFieldTransposed;
float *ioBuffFieldRho;
float *ioBuffFieldTransposed2D;

int nz_varid;
int ny_varid;
int nx_varid;
/*######################------------------- IO module function definitions ---------------------#################*/

/*----->>>>> int ioGetParams();       ----------------------------------------------------------------------

Obtain the complete set of parameters for the IO module

*/
int ioGetParams(){
   int errorCode = IO_SUCCESS;

   /*query for each IO parameter */
   ioOutputMode=0;
   errorCode = queryIntegerParameter("ioOutputMode", &ioOutputMode, 0, 1, PARAM_OPTIONAL);
   errorCode = queryPathParameter("inPath", &inPath, PARAM_OPTIONAL);
   errorCode = queryStringParameter("inFile", &inFile, PARAM_OPTIONAL);
   errorCode = queryPathParameter("outPath", &outPath, PARAM_MANDATORY);
   errorCode = queryStringParameter("outFileBase", &outFileBase, PARAM_MANDATORY);
   errorCode = queryIntegerParameter("frqOutput", &frqOutput, 0, INT_MAX, PARAM_MANDATORY);
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
      printParameter("ioOutputMode", "0: N-to-1 gather and write to a netcdf file, 1:N-to-N writes of FastEddy binary files");
      printParameter("inPath", "Path where initial/restart file is read in from");
      printParameter("inFile", "name of the input file for coordinate system and initial or restart conditions");
      printParameter("outPath", "Path where output files are to be written");
      printParameter("outFileBase", "Base name of the output file series as in (outFileBase).element-in-series");
      printParameter("frqOutput", "frequency (in timesteps) at which to produce output");
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
   } //endif mpi_Rank_world==0

   return(errorCode);
} //end ioAllocateBuffers()

// Include the netCDF-centric source code
#include <io_netcdf.c>

// Include the unformatted N-to-N binary-centric source code
#include <io_binary.c>

/*----->>>>> int ioCleanup();       ----------------------------------------------------------------------
Used to free all malloced memory by the IO module.
*/
int ioCleanup(){
   int errorCode = IO_SUCCESS;

   /*free the io-buffers*/
   free(ioBuffField);
   free(ioBuffFieldTransposed);
   free(ioBuffFieldRho);
   free(ioBuffFieldTransposed2D);

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

