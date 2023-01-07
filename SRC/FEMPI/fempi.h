/* FastEddy®: SRC/FEMPI/fempi.h 
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
#ifndef _FEMPI_H
#define _FEMPI_H
/*fempi__ return codes */
#define FEMPI_SUCCESS               0
#define FEMPI_ERROR_SIZE            1
/*fempi__ includes*/
#include <mpi.h>
/*####################------------------ FEMPI module variable declarations ---------------------#################*/
/* Parameters */
extern int mpi_size_world;       //total number of ranks in our MPI_COMM_WORLD
extern int mpi_rank_world;       //This process' rank in MPI_COMM_WORLD
extern int mpi_nbrXlo;           //The process rank of this process' low-side X-direction neighbor
extern int mpi_nbrXhi;           //The process rank of this process' high-side X-direction neighbor
extern int mpi_nbrYlo;           //The process rank of this process' low-side X-direction neighbor
extern int mpi_nbrYhi;           //The process rank of this process' hi-side X-direction neighbor
extern int mpi_XloBndyRank;      //Flag to indicate if this rank owns a global "Xlo" boundary
extern int mpi_XhiBndyRank;      //Flag to indicate if this rank owns a global "Xhi" boundary
extern int mpi_YloBndyRank;      //Flag to indicate if this rank owns a global "Ylo" boundary
extern int mpi_YhiBndyRank;      //Flag to indicate if this rank owns a global "Yhi" boundary
extern int numProcsX, numProcsY; //Number of cores to be used for horizontal domain decomposition in X and Y directions 
extern int Nxp, Nyp, Nzp;         //This process' subdomain extents in the X and Y and Z directions
extern int rankXid;               //x-direction rankID in the 2-D horizontal domain decomposition 
extern int rankYid;               //y-direction rankID in the 2-D horizontal domain decomposition 
/*static scalars*/
 
/* array fields */
extern float *fempi_DataBuffer; //Buffer used in collective scatter/gather functions
extern float *lorcv_buffer; //Buffer used in halo exchange routines 
extern float *losnd_buffer; //Buffer used in halo exchange routines 
extern float *hircv_buffer; //Buffer used in halo exchange routines 
extern float *hisnd_buffer; //Buffer used in halo exchange routines 

/*####################------------------ FEMPI module function declarations ---------------------#################*/

/*----->>>>> int fempi_LaunchMPI();       ----------------------------------------------------------------------
* Used to launch the base MPI environment. 
*/
int fempi_LaunchMPI(int argc, char **argv);
   
/*----->>>>> int fempi_GetParams();       ----------------------------------------------------------------------
* Used to populate the set of parameters for the FEMPI module
*/
int fempi_GetParams();

/*----->>>>> int fempi_Init();       ----------------------------------------------------------------------
* Used to broadcast and print parameters, allocate memory, and initialize configuration settings for FEMPI.
*/
int fempi_Init();

/*----->>>>> int fempi_Cleanup();       ----------------------------------------------------------------------
* Used to free all malloced memory by the FEMPI module.
*/
int fempi_Cleanup();

/*----->>>>> int fempi_AllocateBuffers();   -----------------------------------
* Allocate memory to fempi_-buffers for collective (scatter/gather) operations
*/
int fempi_AllocateBuffers(int perRankNx, int perRankNy, int perRankNz, int perRankNh);

/*----->>>>> int fempi_XdirHaloExchange2dXY();   -------------------------
* Routine to exchange x-direction halo cells between 
* all pairs of mpi_rank_world x-neighbors for a 2-d x-y field 
*/
int fempi_XdirHaloExchange2dXY(int perRankNx, int perRankNy, int perRankNh,float *field);

/*----->>>>> int fempi_YdirHaloExchange2dXY();   -------------------------
* Routine to exchange y-direction halo cells between 
* all pairs of mpi_rank_world y-neighbors for a 2-d x-y field 
*/
int fempi_YdirHaloExchange2dXY(int perRankNx, int perRankNy, int perRankNh, float *field);

#if __cplusplus
extern "C"{
#endif
/*----->>>>> int fempi_XdirHaloExchange();   -------------------------
* Routine to exchange x-direction halo cells between 
* all pairs of mpi_rank_world x-neighbors 
*/
extern int fempi_XdirHaloExchange(int perRankNx, int perRankNy, int perRankNz, int perRankNh,float *field);

/*----->>>>> int fempi_YdirHaloExchange();   -------------------------
* Routine to exchange y-direction halo cells between 
* all pairs of mpi_rank_world y-neighbors 
*/
extern int fempi_YdirHaloExchange(int perRankNx, int perRankNy, int perRankNz, int perRankNh, float *field);

#if __cplusplus
}
#endif
/*----->>>>> int fempi_SetupPeriodicDomainDecompositionTopology();   -------------------------
* Callable routine to set rank topology neighbor ids for cyclic horizontal 'global domain' boundaries
*/
int fempi_SetupPeriodicDomainDecompositionRankTopology(int xPeriodicSwitch, int yPeriodicSwitch);

/*----->>>>> int fempi_ScatterVariable();     --------------------------------------------------------------------
* Scatters a root-rank variable field defined on a collective domain across rank-based subdomains in a 2-D 
* horizontal decomposition where procID_X = , and procID_Y =...
*/
int fempi_ScatterVariable(int srcNx,int srcNy,int srcNz,
                             int destNx,int destNy,int destNz, int destNh,
                             float* srcFld, float* destFld);

/*----->>>>> int fempi_GatherVariable();     --------------------------------------------------------------------
* Gathers a root-rank variable field defined on a collective domain from sub-domain partitions across ranks 
* in a 2-D horizontal decomposition where procID_X = , and procID_Y =...
*/
int fempi_GatherVariable(int srcNx,int srcNy,int srcNz, int srcNh,
                            int destNx,int destNy,int destNz,
                            float* srcFld,float* destFld);

/*----->>>>> int fempi_FinalizeMPI();       ----------------------------------------------------------------------
* Used to finalize the base MPI environment. 
*/
int fempi_FinalizeMPI();

#endif // _FEMPI_H
