/* FastEddy®: SRC/FECUDA/fecuda.c 
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
#include <fecuda.h>
#include <fecuda_Device.h>
#include <fecuda_Utils.h>


/*###################------------------- FECUDA module variable definitions ---------------------##############*/
int tBx;
int tBy;
int tBz;

/*###################------------------- FECUDA module function definitions ---------------------##############*/

/*----->>>>> int fecuda_GetParams();       ----------------------------------------------------------------------
Obtain the complete set of parameters for the FECUDA module
*/
int fecuda_GetParams(){
   int errorCode = FECUDA_SUCCESS;

   /*query for each FECUDA parameter */
   errorCode = queryIntegerParameter("tBx", &tBx, 1, INT_MAX, PARAM_MANDATORY);
   errorCode = queryIntegerParameter("tBy", &tBy, 1, INT_MAX, PARAM_MANDATORY);
   errorCode = queryIntegerParameter("tBz", &tBz, 1, INT_MAX, PARAM_MANDATORY);

   return(errorCode);
} //end fecuda_GetParams()

/*#########------------------- FECUDA module CUDA-C wrapper function declarations ------------#################*/
/* Each of these functions has a corrseponding "Device" function called by the wrapper. 
 * This allows mixed compilation/linking of C and CUDA. */

/*----->>>>> int fecuda_Init();       ----------------------------------------------------------------------
*Used to broadcast and print parameters, allocate memory, and initialize configuration settings for  the FECUDA module.
*/
int fecuda_Init(){
   int errorCode = FECUDA_SUCCESS;

   /* Setup the tBlock */
   if(mpi_rank_world == 0){
      printComment("FECUDA parameters---");
      printParameter("tBx", "Number of threads in x-dimension.");
      printParameter("tBy", "Number of threads in y-dimension.");
      printParameter("tBz", "Number of threads in z-dimension.");
   } //end if(mpi_rank_world == 0)

   /*Broadcast the parameters across mpi_ranks*/
   MPI_Bcast(&tBx, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);
   MPI_Bcast(&tBy, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);
   MPI_Bcast(&tBz, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);

   /*Setup the threads per block, dim3 vector*/
   errorCode = fecuda_DeviceSetup(tBx, tBy, tBz);
      
   return(errorCode);
} //end fecuda_Init()

/*----->>>>> int fecuda_Cleanup(); -------------------------------------------------------------------
*Used to free all malloced memory by the FECUDA module.
*/
int fecuda_Cleanup(){
   int errorCode = FECUDA_SUCCESS;

   /* Free any FECUDA module device-arrays */
   errorCode = fecuda_DeviceCleanup();

   return(errorCode);

}//end fecuda_Cleanup()

/*----->>>>> int fecuda_SetBlocksPerGrid();    -----------------------------------------------------------
* Used to set the grid and tBlock (threadBlock) configuration parameters for the device..
*/
int fecuda_SetBlocksPerGrid(int Nx, int Ny, int Nz){
   int errorCode = FECUDA_SUCCESS;

   errorCode = fecuda_DeviceSetBlocksPerGrid(Nx, Ny, Nz);
   return(errorCode);
}

/*----->>>>> int fecuda_AllocateHaloBuffers(); -------------------------------------------------------------
 * * Used to allocate device memory buffers for coalesced halo exchanges in the FECUDA module.
 * */
int fecuda_AllocateHaloBuffers(int Nxp, int Nyp, int Nzp, int Nh){
   int errorCode = FECUDA_SUCCESS;

   errorCode = fecuda_UtilsAllocateHaloBuffers(Nxp, Nyp, Nzp, Nh);
   return(errorCode);
}

/*----->>>>> int fecuda_DeallocateHaloBuffers(); ---------------------------------------------------------
 * * Used to free device memory buffers for coalesced halo exchanges in the FECUDA module.
 * */
int fecuda_DeallocateHaloBuffers(){
   int errorCode = FECUDA_SUCCESS;

   errorCode = fecuda_UtilsDeallocateHaloBuffers();
   return(errorCode);
}
