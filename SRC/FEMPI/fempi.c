/* FastEddy®: SRC/FEMPI/fempi.c 
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

/*####################------------------- FEMPI module variable definitions ---------------------#################*/
int mpi_size_world;            //total number of ranks in our MPI_COMM_WORLD
int mpi_rank_world;            //This process' rank in MPI_COMM_WORLD
int mpi_nbrXlo;                //The process rank of this process' low-side X-direction neighbor
int mpi_nbrXhi;                //The process rank of this process' high-side X-direction neighbor
int mpi_nbrYlo;                //The process rank of this process' low-side X-direction neighbor
int mpi_nbrYhi;                //The process rank of this process' hi-side X-direction neighbor
int mpi_XloBndyRank;      //Flag to indicate if this rank owns a global "Xlo" boundary
int mpi_XhiBndyRank;      //Flag to indicate if this rank owns a global "Xhi" boundary
int mpi_YloBndyRank;      //Flag to indicate if this rank owns a global "Ylo" boundary
int mpi_YhiBndyRank;      //Flag to indicate if this rank owns a global "Yhi" boundary
int numProcsX, numProcsY;        //Number of cores to be used for horizontal domain decomposition in X and Y directions 
int Nxp, Nyp, Nzp;         //This process' subdomain extents in the X and Y and Z directions
int rankXid;               //x-direction rankID in the 2-D horizontal domain decomposition 
int rankYid;               //y-direction rankID in the 2-D horizontal domain decomposition 
float* fempi_DataBuffer; //Buffer used in collective scatter/gather functions
float *lorcv_buffer; //Buffer used in halo exchange routines 
float *losnd_buffer; //Buffer used in halo exchange routines 
float *hircv_buffer; //Buffer used in halo exchange routines 
float *hisnd_buffer; //Buffer used in halo exchange routines 

/*####################------------------- FEMPI module function definitions ---------------------#################*/

/*----->>>>> int fempi_LaunchMPI();       ----------------------------------------------------------------------
* Used to launch the base MPI environment. 
*/
int fempi_LaunchMPI(int argc, char **argv){
   int errorCode = FEMPI_SUCCESS;
   /* Launch the mpi environment */
   errorCode = MPI_Init(&argc, &argv);

   /* Get the number of processes */
   MPI_Comm_size(MPI_COMM_WORLD, &mpi_size_world);

   /* Get the rank of each process */
   MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank_world);

#ifdef DEBUG
   int iRank;
   for(iRank=0; iRank < mpi_size_world; iRank++){
     if(mpi_rank_world == iRank){
        printf("fempi_LaunchMPI--- Rank %d/%d: active.\n",mpi_rank_world, mpi_size_world);
        fflush(stdout);
     }
   }
#endif

   /* A synchronization point */
   MPI_Barrier(MPI_COMM_WORLD);
   return(errorCode);
} //fempi_LaunchMPI()

/*----->>>>> int fempi_GetParams();   --------------------------------------------------------------------
* Obtain the complete set of parameters for the FEMPI module
*/
int fempi_GetParams(){
   int errorCode = FEMPI_SUCCESS;

   /*query for each FEMPI parameter */
   errorCode = queryIntegerParameter("numProcsX", &numProcsX, 1, INT_MAX, PARAM_MANDATORY);
   errorCode = queryIntegerParameter("numProcsY", &numProcsY, 1, INT_MAX, PARAM_MANDATORY);

   /*Perform a check that the product of numProcsX*numProcsY = mpi_size_world else return an appropriate error code*/
   if(mpi_size_world != (numProcsX*numProcsY)){
     printf("ERROR!: mpi_size_world = %d not equal to numProcsX*numProcsY = %d\n",mpi_size_world,numProcsX*numProcsY);
     printf("Exiting now!\n");
     fflush(stdout);
     exit(FEMPI_ERROR_SIZE); 
   }
   return(errorCode);
} //end fempi_GetParams()

/*----->>>>> int fempi_Init();     --------------------------------------------------------------------
* Used to broadcast and print parameters, allocate memory, and initialize settings for the FEMPI module.
*/
int fempi_Init(){
   int errorCode = FEMPI_SUCCESS;
   int nbrXloRankX,nbrXhiRankX;
   int nbrYloRankY,nbrYhiRankY;
   if(mpi_rank_world == 0){
      printComment("FEMPI parameters---");
      printParameter("numProcsX", "Number of cores to be used for horizontal domain decomposition in X");
      printParameter("numProcsY", "Number of cores to be used for horizontal domain decomposition in Y");
   }
   /*Broadcast the parameters across mpi_ranks*/ 
   MPI_Bcast(&numProcsX, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);
   MPI_Bcast(&numProcsY, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);

   /* Determine x and y -direction rankIDs under 2-d horizontal domain decomposition */
   rankXid = mpi_rank_world%numProcsX;
   rankYid = ((int) (mpi_rank_world/numProcsX)); 
   printf("Rank %d/%d:  [rankXid, rankYid] = [%d,%d]\n",
          mpi_rank_world, mpi_size_world, rankXid, rankYid);
   //Setup neighbor rank topology identifiers
   nbrXloRankX = rankXid-1;
   nbrXhiRankX = rankXid+1; 
   nbrYloRankY = rankYid-1;
   nbrYhiRankY = rankYid+1; 
   if(nbrXloRankX < 0){
     mpi_nbrXlo = -1;
     mpi_XloBndyRank = 1;
   }else{
     mpi_nbrXlo = (rankYid)*numProcsX+nbrXloRankX;
     mpi_XloBndyRank = 0;
   } //end if(nbrXloRankX < 0)...else...
   if(nbrXhiRankX > numProcsX-1){
     mpi_nbrXhi = -1; 
     mpi_XhiBndyRank = 1;
   }else{
     mpi_nbrXhi = (rankYid)*numProcsX+nbrXhiRankX; 
     mpi_XhiBndyRank = 0;
   } //end if(nbrXhiRankX > numProcsX-1)...else...
   if(nbrYloRankY < 0){
     mpi_nbrYlo = -1; 
     mpi_YloBndyRank = 1;
   }else{
     mpi_nbrYlo = (rankYid-1)*numProcsX+(rankXid);
     mpi_YloBndyRank = 0;
   } //end if(nbrYloRankY < 0)...else...
   if(nbrYhiRankY > numProcsY-1){
     mpi_nbrYhi = -1; 
     mpi_YhiBndyRank = 1;
   }else{
     mpi_nbrYhi = (rankYid+1)*numProcsX+(rankXid); 
     mpi_YhiBndyRank = 0;
   } //end if(nbrYhiRankY > numProcsY-1)...else...

   /* Done */
   return(errorCode);
} //end fempi_Init()

/*----->>>>> int fempi_AllocateBuffers();   -----------------------------------
* Allocate memory to fempi_-buffers for collective (scatter/gather) operations
*/
int fempi_AllocateBuffers(int perRankNx, int perRankNy, int perRankNz, int perRankNh){
   int errorCode = FEMPI_SUCCESS;
   void* memsetReturnVal;
   void* m_field; 
   int snd_rcv_bufsize;

   /*Allocate a data buffer on the root-rank for use in scatter/gather routines*/ 
   if(mpi_rank_world == 0){
     if(posix_memalign(&m_field, 128,
                      (mpi_size_world)
                      *(perRankNx+2*perRankNh)
                      *(perRankNy+2*perRankNh)
                      *(perRankNz+2*perRankNh)*sizeof(float))) {
        fprintf(stderr, "Rank %d/%d fempi_AllocateBuffersi(): Memory Allocation of (%s) failed!\n",
             mpi_rank_world,mpi_size_world,"fempi_DataBuffer");
        exit(1);
     } //endif (allocation failed)

     /*initialize the allocated space to zero everywhere*/
     memsetReturnVal = memset(m_field,0,
                              (mpi_size_world)
                              *(perRankNx+2*perRankNh)
                              *(perRankNy+2*perRankNh)
                              *(perRankNz+2*perRankNh)*sizeof(float));

     fempi_DataBuffer = (float *) m_field;
   } //endif mpi_rank_world == 0 

   snd_rcv_bufsize =((int) fmax((perRankNx+2*perRankNh),(perRankNy+2*perRankNh)))
                     *(perRankNz+2*perRankNh)*perRankNh;
   /*Allocate data buffers on all ranks for use in lateral halo exchange routines*/ 
   if(posix_memalign(&m_field, 128,snd_rcv_bufsize*sizeof(float))) {
        fprintf(stderr, "Rank %d/%d fempi_AllocateBuffers(): Memory Allocation of (%s) failed!\n",
                mpi_rank_world,mpi_size_world,"lorcv_buffer");
        exit(1);
   } //endif (allocation failed)
   memsetReturnVal = memset(m_field,0,snd_rcv_bufsize*sizeof(float));
   lorcv_buffer = (float *) m_field;
   if(posix_memalign(&m_field, 128,snd_rcv_bufsize*sizeof(float))) {
        fprintf(stderr, "Rank %d/%d fempi_AllocateBuffers(): Memory Allocation of (%s) failed!\n",
                mpi_rank_world,mpi_size_world,"losnd_buffer");
        exit(1);
   } //endif (allocation failed)
   memsetReturnVal = memset(m_field,0,snd_rcv_bufsize*sizeof(float));
   losnd_buffer = (float *) m_field;
   if(posix_memalign(&m_field, 128,snd_rcv_bufsize*sizeof(float))) {
        fprintf(stderr, "Rank %d/%d fempi_AllocateBuffers(): Memory Allocation of (%s) failed!\n",
                mpi_rank_world,mpi_size_world,"hircv_buffer");
        exit(1);
   } //endif (allocation failed)
   memsetReturnVal = memset(m_field,0,snd_rcv_bufsize*sizeof(float));
   hircv_buffer = (float *) m_field;
   if(posix_memalign(&m_field, 128,snd_rcv_bufsize*sizeof(float))) {
        fprintf(stderr, "Rank %d/%d fempi_AllocateBuffers(): Memory Allocation of (%s) failed!\n",
                mpi_rank_world,mpi_size_world,"hisnd_buffer");
        exit(1);
   } //endif (allocation failed)
   memsetReturnVal = memset(m_field,0,snd_rcv_bufsize*sizeof(float));
   hisnd_buffer = (float *) m_field;
   if(memsetReturnVal == NULL){
     fprintf(stderr, "Rank %d/%d fempi_AllocateBuffers():WARNING memsetReturnVal == NULL!\n",
                mpi_rank_world,mpi_size_world);    
   } 
   return(errorCode);
} //end fempi_AllocateBuffers()

/*----->>>>> int fempi_SetupPeriodicDomainDecompositionTopology();   -------------------------
* Routine to set rank topology neighbor ids for cyclic horizontal 'global domain' boundaries
*/
int fempi_SetupPeriodicDomainDecompositionRankTopology(int xPeriodicSwitch, int yPeriodicSwitch){
   int errorCode = FEMPI_SUCCESS;

   //Setup the x-direction cyclic neighbors
   if(xPeriodicSwitch == 1){
     if(mpi_nbrXlo < 0){   
      mpi_nbrXlo = mpi_rank_world+(numProcsX-1);
     } //end if(mpi_nbrXlo < 0)
     if(mpi_nbrXhi < 0){
       mpi_nbrXhi = mpi_rank_world-(numProcsX-1);
     } //end if(mpi_nbrXhi < 0)
   } //end if(xPeriodicSwitch == 1)
   //Setup the y-direction cyclic neighbors
   if(yPeriodicSwitch == 1){
     if(mpi_nbrYlo < 0){   
      mpi_nbrYlo = mpi_rank_world+(numProcsY-1)*numProcsX;
     } //end if(mpi_nbrYlo < 0)
     if(mpi_nbrYhi < 0){   
      mpi_nbrYhi = mpi_rank_world-(numProcsY-1)*numProcsX;
     } //end if(mpi_nbrYhi < 0)
   } //end if(yPeriodicSwitch == 1)
   return(errorCode);
} //end fempi_SetupPeriodicDomainDecompositionRankTopology()

/*----->>>>> int fempi_XdirHaloExchange2dXY();   -------------------------
* Routine to exchange x-direction halo cells between 
* all pairs of mpi_rank_world x-neighbors for a 2-d x-y field 
*/
int fempi_XdirHaloExchange2dXY(int perRankNx, int perRankNy, int perRankNh,float *field){
   int errorCode = FEMPI_SUCCESS;
   MPI_Status status;
   MPI_Request lorcv_req, losnd_req;
   MPI_Request hircv_req, hisnd_req;
   int lorcv_tag;
   int losnd_tag;
   int hircv_tag;
   int hisnd_tag;
   int nBuffElems;
   int ih;
   int ihTarg;
   int ihBuff;
 
   /*Set the number of buffer elements to be sent or recieved*/
   nBuffElems = perRankNh*(perRankNy+2*perRankNh);

   /*If this process has an Xlo neighbor */
   if(mpi_nbrXlo >= 0){
      /* Setup send and recieve tags */
      losnd_tag = (mpi_rank_world+numProcsX)*(mpi_nbrXlo+2*numProcsX*numProcsY)+100;
      lorcv_tag = (mpi_rank_world+2*numProcsX*numProcsY)*(mpi_nbrXlo+numProcsX)+200;
      /*Copy the Xlo y-z surface into the send lobuffer*/
      for(ih=0; ih < perRankNh; ih++){
        ihTarg = (perRankNh+ih)*(perRankNy+2*perRankNh);
        ihBuff = (ih)*(perRankNy+2*perRankNh);
        memcpy(&losnd_buffer[ihBuff], &field[ihTarg], (perRankNy+2*perRankNh)*sizeof(float));
      } //end for ih
      /*Recieve a y-z buffer from Xlo!*/
      errorCode = MPI_Irecv(lorcv_buffer, nBuffElems, MPI_FLOAT, mpi_nbrXlo, lorcv_tag, MPI_COMM_WORLD, &lorcv_req);
      /*Send send buffer to the Xlo neighbor!*/
      errorCode = MPI_Isend(losnd_buffer, nBuffElems, MPI_FLOAT, mpi_nbrXlo, losnd_tag, MPI_COMM_WORLD, &losnd_req);
   }//end if mpi_nbrXlo >= 0

   /*If this process has an Xhi neighbor */
   if(mpi_nbrXhi >= 0){
      /* Setup send and recieve tags */
      hisnd_tag = (mpi_rank_world+numProcsX)*(mpi_nbrXhi+2*numProcsX*numProcsY)+200;
      hircv_tag = (mpi_rank_world+2*numProcsX*numProcsY)*(mpi_nbrXhi+numProcsX)+100;
      /*Copy the Xhi y-z surface into the send hibuffer*/
      for(ih=0; ih < perRankNh; ih++){
          ihTarg = (perRankNx+ih)*(perRankNy+2*perRankNh);
          ihBuff = (ih)*(perRankNy+2*perRankNh);
          memcpy(&hisnd_buffer[ihBuff], &field[ihTarg], (perRankNy+2*perRankNh)*sizeof(float));
      } //end for ih

      /*Recieve a rcv_buffer from Xhi!*/
      errorCode = MPI_Irecv(hircv_buffer, nBuffElems, MPI_FLOAT, mpi_nbrXhi, hircv_tag, MPI_COMM_WORLD, &hircv_req);
      /* Send the snd_buffer to the Xhi neighbor! */
      errorCode = MPI_Isend(hisnd_buffer, nBuffElems, MPI_FLOAT, mpi_nbrXhi, hisnd_tag, MPI_COMM_WORLD, &hisnd_req);
   } //end if(mpi_nbrXhi >= 0)

   if(mpi_nbrXlo >= 0){
      errorCode = MPI_Wait(&lorcv_req, &status);
      /*Copy the contents of the recv buffer into the Xlo y-z surface halos of *field*/
      for(ih=0; ih < perRankNh; ih++){
          ihTarg = (ih)*(perRankNy+2*perRankNh);
          ihBuff = (ih)*(perRankNy+2*perRankNh);
          memcpy(&field[ihTarg], &lorcv_buffer[ihBuff], (perRankNy+2*perRankNh)*sizeof(float));
      } //end for ih
   }//end if mpi_nbrXlo >= 0
   if(mpi_nbrXhi >= 0){
      errorCode = MPI_Wait(&hircv_req, &status);
      /*Copy the contents of the recv hibuffer into the Xhi y-z surface halos of *field*/
      for(ih=0; ih < perRankNh ; ih++){
          ihTarg = (perRankNx+perRankNh+ih)*(perRankNy+2*perRankNh);
          ihBuff = (ih)*(perRankNy+2*perRankNh);
          memcpy(&field[ihTarg], &hircv_buffer[ihBuff], (perRankNy+2*perRankNh)*sizeof(float));
      } //end for ih
   } //end if(mpi_nbrXhi >= 0)
   
   MPI_Barrier(MPI_COMM_WORLD);
   return(errorCode);
} //end fempi_XdirHaloExchange2dXY()

/*----->>>>> int fempi_YdirHaloExchange2dXY();   -------------------------
* Routine to exchange y-direction halo cells between 
* all pairs of mpi_rank_world y-neighbors for a 2-d x-y field 
*/
int fempi_YdirHaloExchange2dXY(int perRankNx, int perRankNy, int perRankNh, float *field){
   int errorCode = FEMPI_SUCCESS;
   MPI_Status status;
   MPI_Request lorcv_req, losnd_req;
   MPI_Request hircv_req, hisnd_req;
   int lorcv_tag;
   int losnd_tag;
   int hircv_tag;
   int hisnd_tag;
   int nBuffElems;
   int ih,i;
   int ihiTarg;
   int ihiBuff;

   /*Set the number of buffer elements to be sent or recieved*/
   nBuffElems = perRankNh*(perRankNx+2*perRankNh);
   
   /*If this process has an Ylo neighbor */
   if(mpi_nbrYlo >= 0){
      /* Setup send and recieve tags */
      losnd_tag = (mpi_rank_world+numProcsX)*(mpi_nbrYlo+2*numProcsX*numProcsY)+100;
      lorcv_tag = (mpi_rank_world+2*numProcsX*numProcsY)*(mpi_nbrYlo+numProcsX)+200;
      /*Copy the Xlo y-z surface into the send lobuffer*/
      for(ih=0; ih < perRankNh; ih++){
        for(i=0; i < perRankNx+2*perRankNh; i++){
          ihiTarg = (i)*(perRankNy+2*perRankNh)+(perRankNh+ih);
          ihiBuff = (ih)*(perRankNx+2*perRankNh)+i;
          losnd_buffer[ihiBuff] = field[ihiTarg];
        } //end for j
      } //end for ih

      /*Recieve a rcv_buffer from Ylo!*/
      errorCode = MPI_Irecv(lorcv_buffer, nBuffElems, MPI_FLOAT, mpi_nbrYlo, lorcv_tag, MPI_COMM_WORLD, &lorcv_req);
      /*Send send buffer to the Ylo neighbor!*/
      errorCode = MPI_Isend(losnd_buffer, nBuffElems, MPI_FLOAT, mpi_nbrYlo, losnd_tag, MPI_COMM_WORLD, &losnd_req);
   }//end if mpi_nbrYlo >= 0

   /*If this process has an Yhi neighbor */
   if(mpi_nbrYhi >= 0){
      /* Setup send and recieve tags */
      hisnd_tag = (mpi_rank_world+numProcsX)*(mpi_nbrYhi+2*numProcsX*numProcsY)+200;
      hircv_tag = (mpi_rank_world+2*numProcsX*numProcsY)*(mpi_nbrYhi+numProcsX)+100;
      /*Copy the Yhi x-z surface into the send hibuffer*/
      for(ih=0; ih < perRankNh; ih++){
        for(i=0; i < perRankNx+2*perRankNh; i++){
          ihiTarg = (i)*(perRankNy+2*perRankNh)+(perRankNy+ih);
          ihiBuff = (ih)*(perRankNx+2*perRankNh)+i;
          hisnd_buffer[ihiBuff] = field[ihiTarg];
        } //end for j
      } //end for ih

      /*Recieve a rcv_buffer from Yhi!*/
      errorCode = MPI_Irecv(hircv_buffer, nBuffElems, MPI_FLOAT, mpi_nbrYhi, hircv_tag, MPI_COMM_WORLD, &hircv_req);
      /* Send the snd_buffer to the Yhi neighbor! */
      errorCode = MPI_Isend(hisnd_buffer, nBuffElems, MPI_FLOAT, mpi_nbrYhi, hisnd_tag, MPI_COMM_WORLD, &hisnd_req);
   } //end if(mpi_nbrXhi >= 0)

   if(mpi_nbrYlo >= 0){
      errorCode = MPI_Wait(&lorcv_req, &status);
      /*Copy the contents of the recv buffer into the Ylo x-z surface halos of *field*/
      for(ih=0; ih < perRankNh; ih++){
        for(i=0; i < perRankNx+2*perRankNh; i++){
          ihiTarg = (i)*(perRankNy+2*perRankNh)+ih;
          ihiBuff = (ih)*(perRankNx+2*perRankNh)+i;
          field[ihiTarg] = lorcv_buffer[ihiBuff];
        } //end for i
      } //end for ih
   }//end if mpi_nbrYlo >= 0

   if(mpi_nbrYhi >= 0){
      errorCode = MPI_Wait(&hircv_req, &status);
      /*Copy the contents of the recv hibuffer into the Yhi x-z surface halos of *field*/
      for(ih=0; ih < perRankNh ; ih++){
        for(i=0; i < perRankNx+2*perRankNh; i++){
          ihiTarg = (i)*(perRankNy+2*perRankNh)+(perRankNy+perRankNh+ih);
          ihiBuff = (ih)*(perRankNx+2*perRankNh)+i;
          field[ihiTarg] = hircv_buffer[ihiBuff];
        } //end for i
      } //end for ih
   } //end if(mpi_nbrYhi >= 0)


   MPI_Barrier(MPI_COMM_WORLD);
   return(errorCode);
} //end fempi_YdirHaloExchange2dXY()

/*----->>>>> int fempi_XdirHaloExchange();   -------------------------
* Routine to exchange x-direction halo cells between 
* all pairs of mpi_rank_world x-neighbors 
*/
int fempi_XdirHaloExchange(int perRankNx, int perRankNy, int perRankNz, int perRankNh,float *field){
   int errorCode = FEMPI_SUCCESS;
   MPI_Status status;  
   MPI_Request lorcv_req, losnd_req;
   MPI_Request hircv_req, hisnd_req;
   int lorcv_tag;
   int losnd_tag;
   int hircv_tag;
   int hisnd_tag;
   int nBuffElems;
   int ih,j;
   int ihjTarg;
   int ihjBuff;

   /*Set the number of buffer elements to be sent or recieved*/
   nBuffElems = perRankNh*(perRankNy+2*perRankNh)*(perRankNz+2*perRankNh);
 
   /*If this process has an Xlo neighbor */
   if(mpi_nbrXlo >= 0){
      /* Setup send and recieve tags */
      losnd_tag = (mpi_rank_world+numProcsX)*(mpi_nbrXlo+2*numProcsX*numProcsY)+100;
      lorcv_tag = (mpi_rank_world+2*numProcsX*numProcsY)*(mpi_nbrXlo+numProcsX)+200;
      /*Copy the Xlo y-z surface into the send lobuffer*/
      for(ih=0; ih < perRankNh; ih++){
        for(j=0; j < perRankNy+2*perRankNh; j++){
          ihjTarg = (perRankNh+ih)*(perRankNy+2*perRankNh)*(perRankNz+2*perRankNh)+j*(perRankNz+2*perRankNh);
          ihjBuff = (ih)*(perRankNy+2*perRankNh)*(perRankNz+2*perRankNh)+j*(perRankNz+2*perRankNh); 
          memcpy(&losnd_buffer[ihjBuff], &field[ihjTarg], (perRankNz+2*perRankNh)*sizeof(float));
        } //end for j
      } //end for ih
      
      /*Recieve a y-z buffer from Xlo!*/ 
      errorCode = MPI_Irecv(lorcv_buffer, nBuffElems, MPI_FLOAT, mpi_nbrXlo, lorcv_tag, MPI_COMM_WORLD, &lorcv_req); 
      /*Send send buffer to the Xlo neighbor!*/
      errorCode = MPI_Isend(losnd_buffer, nBuffElems, MPI_FLOAT, mpi_nbrXlo, losnd_tag, MPI_COMM_WORLD, &losnd_req); 
   }//end if mpi_nbrXlo >= 0

   /*If this process has an Xhi neighbor */
   if(mpi_nbrXhi >= 0){
      /* Setup send and recieve tags */
      hisnd_tag = (mpi_rank_world+numProcsX)*(mpi_nbrXhi+2*numProcsX*numProcsY)+200;
      hircv_tag = (mpi_rank_world+2*numProcsX*numProcsY)*(mpi_nbrXhi+numProcsX)+100;
      /*Copy the Xhi y-z surface into the send hibuffer*/
      for(ih=0; ih < perRankNh; ih++){
        for(j=0; j < perRankNy+2*perRankNh; j++){
          ihjTarg = (perRankNx+ih)*(perRankNy+2*perRankNh)*(perRankNz+2*perRankNh)
                    +j*(perRankNz+2*perRankNh);
          ihjBuff = (ih)*(perRankNy+2*perRankNh)*(perRankNz+2*perRankNh)+j*(perRankNz+2*perRankNh); 
          memcpy(&hisnd_buffer[ihjBuff], &field[ihjTarg], (perRankNz+2*perRankNh)*sizeof(float));
        } //end for j
      } //end for ih

      /*Recieve a rcv_buffer from Xhi!*/ 
      errorCode = MPI_Irecv(hircv_buffer, nBuffElems, MPI_FLOAT, mpi_nbrXhi, hircv_tag, MPI_COMM_WORLD, &hircv_req); 
      /* Send the snd_buffer to the Xhi neighbor! */  
      errorCode = MPI_Isend(hisnd_buffer, nBuffElems, MPI_FLOAT, mpi_nbrXhi, hisnd_tag, MPI_COMM_WORLD, &hisnd_req); 
   } //end if(mpi_nbrXhi >= 0)

   if(mpi_nbrXlo >= 0){
      errorCode = MPI_Wait(&lorcv_req, &status);
      /*Copy the contents of the recv buffer into the Xlo y-z surface halos of *field*/
      for(ih=0; ih < perRankNh; ih++){
        for(j=0; j < perRankNy+2*perRankNh; j++){
          ihjTarg = (ih)*(perRankNy+2*perRankNh)*(perRankNz+2*perRankNh)+j*(perRankNz+2*perRankNh);
          ihjBuff = (ih)*(perRankNy+2*perRankNh)*(perRankNz+2*perRankNh)+j*(perRankNz+2*perRankNh); 
          memcpy(&field[ihjTarg], &lorcv_buffer[ihjBuff], (perRankNz+2*perRankNh)*sizeof(float));
        } //end for j
      } //end for ih
   }//end if mpi_nbrXlo >= 0
   if(mpi_nbrXhi >= 0){
      errorCode = MPI_Wait(&hircv_req, &status);
      /*Copy the contents of the recv hibuffer into the Xhi y-z surface halos of *field*/
      for(ih=0; ih < perRankNh ; ih++){
        for(j=0; j < perRankNy+2*perRankNh; j++){
          ihjTarg = (perRankNx+perRankNh+ih)*(perRankNy+2*perRankNh)*(perRankNz+2*perRankNh)+j*(perRankNz+2*perRankNh);
          ihjBuff = (ih)*(perRankNy+2*perRankNh)*(perRankNz+2*perRankNh)+j*(perRankNz+2*perRankNh); 
          memcpy(&field[ihjTarg], &hircv_buffer[ihjBuff], (perRankNz+2*perRankNh)*sizeof(float));
        } //end for j
      } //end for ih
   } //end if(mpi_nbrXhi >= 0)

   MPI_Barrier(MPI_COMM_WORLD); 
   return(errorCode);
} //end fempi_XdirHaloExchange()

/*----->>>>> int fempi_YdirHaloExchange();   -------------------------
* Routine to exchange y-direction halo cells between 
* all pairs of mpi_rank_world y-neighbors 
*/
int fempi_YdirHaloExchange(int perRankNx, int perRankNy, int perRankNz, int perRankNh, float *field){
   int errorCode = FEMPI_SUCCESS;
   MPI_Status status;
   MPI_Request lorcv_req, losnd_req;
   MPI_Request hircv_req, hisnd_req;
   int lorcv_tag;
   int losnd_tag;
   int hircv_tag;
   int hisnd_tag;
   int nBuffElems;
   int ih,i;
   int ihiTarg;
   int ihiBuff;

   /*Set the number of buffer elements to be sent or recieved*/
   nBuffElems = perRankNh*(perRankNx+2*perRankNh)*(perRankNz+2*perRankNh);

   /*If this process has an Ylo neighbor */
   if(mpi_nbrYlo >= 0){
      /* Setup send and recieve tags */
      losnd_tag = (mpi_rank_world+numProcsX)*(mpi_nbrYlo+2*numProcsX*numProcsY)+100;
      lorcv_tag = (mpi_rank_world+2*numProcsX*numProcsY)*(mpi_nbrYlo+numProcsX)+200;
      /*Copy the Xlo y-z surface into the send lobuffer*/
      for(ih=0; ih < perRankNh; ih++){
        for(i=0; i < perRankNx+2*perRankNh; i++){
          ihiTarg = (i)*(perRankNy+2*perRankNh)*(perRankNz+2*perRankNh)
                    +(perRankNh+ih)*(perRankNz+2*perRankNh);
          ihiBuff = (ih)*(perRankNx+2*perRankNh)*(perRankNz+2*perRankNh)+i*(perRankNz+2*perRankNh);
          memcpy(&losnd_buffer[ihiBuff], &field[ihiTarg], (perRankNz+2*perRankNh)*sizeof(float));
        } //end for j
      } //end for ih

      /*Recieve a rcv_buffer from Ylo!*/
      errorCode = MPI_Irecv(lorcv_buffer, nBuffElems, MPI_FLOAT, mpi_nbrYlo, lorcv_tag, MPI_COMM_WORLD, &lorcv_req);
      /*Send send buffer to the Ylo neighbor!*/
      errorCode = MPI_Isend(losnd_buffer, nBuffElems, MPI_FLOAT, mpi_nbrYlo, losnd_tag, MPI_COMM_WORLD, &losnd_req);
   }//end if mpi_nbrYlo >= 0

   /*If this process has an Yhi neighbor */
   if(mpi_nbrYhi >= 0){
      /* Setup send and recieve tags */
      hisnd_tag = (mpi_rank_world+numProcsX)*(mpi_nbrYhi+2*numProcsX*numProcsY)+200;
      hircv_tag = (mpi_rank_world+2*numProcsX*numProcsY)*(mpi_nbrYhi+numProcsX)+100;
      /*Copy the Yhi x-z surface into the send hibuffer*/
      for(ih=0; ih < perRankNh; ih++){
        for(i=0; i < perRankNx+2*perRankNh; i++){
          ihiTarg = (i)*(perRankNy+2*perRankNh)*(perRankNz+2*perRankNh)
                    +(perRankNy+ih)*(perRankNz+2*perRankNh);
          ihiBuff = (ih)*(perRankNx+2*perRankNh)*(perRankNz+2*perRankNh)+i*(perRankNz+2*perRankNh);
          memcpy(&hisnd_buffer[ihiBuff], &field[ihiTarg], (perRankNz+2*perRankNh)*sizeof(float));
        } //end for j
      } //end for ih

      /*Recieve a rcv_buffer from Yhi!*/
      errorCode = MPI_Irecv(hircv_buffer, nBuffElems, MPI_FLOAT, mpi_nbrYhi, hircv_tag, MPI_COMM_WORLD, &hircv_req);
      /* Send the snd_buffer to the Yhi neighbor! */
      errorCode = MPI_Isend(hisnd_buffer, nBuffElems, MPI_FLOAT, mpi_nbrYhi, hisnd_tag, MPI_COMM_WORLD, &hisnd_req);
   } //end if(mpi_nbrXhi >= 0)

   if(mpi_nbrYlo >= 0){
      errorCode = MPI_Wait(&lorcv_req, &status);
      /*Copy the contents of the recv buffer into the Ylo x-z surface halos of *field*/
      for(ih=0; ih < perRankNh; ih++){
        for(i=0; i < perRankNx+2*perRankNh; i++){
          ihiTarg = (i)*(perRankNy+2*perRankNh)*(perRankNz+2*perRankNh)+
                    (ih)*(perRankNz+2*perRankNh);
          ihiBuff = (ih)*(perRankNx+2*perRankNh)*(perRankNz+2*perRankNh)+i*(perRankNz+2*perRankNh);
          memcpy(&field[ihiTarg], &lorcv_buffer[ihiBuff], (perRankNz+2*perRankNh)*sizeof(float));
        } //end for i
      } //end for ih
   }//end if mpi_nbrYlo >= 0

   if(mpi_nbrYhi >= 0){
      errorCode = MPI_Wait(&hircv_req, &status);
      /*Copy the contents of the recv hibuffer into the Yhi x-z surface halos of *field*/
      for(ih=0; ih < perRankNh ; ih++){
        for(i=0; i < perRankNx+2*perRankNh; i++){
          ihiTarg = (i)*(perRankNy+2*perRankNh)*(perRankNz+2*perRankNh)+
                    (perRankNy+perRankNh+ih)*(perRankNz+2*perRankNh);
          ihiBuff = (ih)*(perRankNx+2*perRankNh)*(perRankNz+2*perRankNh)+i*(perRankNz+2*perRankNh);
          memcpy(&field[ihiTarg], &hircv_buffer[ihiBuff], (perRankNz+2*perRankNh)*sizeof(float));
        } //end for i
      } //end for ih
   } //end if(mpi_nbrYhi >= 0)
   
   MPI_Barrier(MPI_COMM_WORLD); 
   return(errorCode);
} //end fempi_YdirHaloExchange()

/*----->>>>> int fempi_ScatterVariable();     --------------------------------------------------------------------
* Scatters a root-rank variable field defined on a collective domain across rank-based subdomains in a 2-D 
* horizontal decomposition where procID_X = , and procID_Y =...
*/
int fempi_ScatterVariable(int srcNx,int srcNy,int srcNz, 
                             int destNx,int destNy,int destNz, int destNh,
                             float* srcFld, float* destFld){
   int errorCode = FEMPI_SUCCESS;
   int i,j,k,ijkSrc,ijkDest,iRank;
   int iMin,iMax,jMin,jMax,kMin;
   int rankStart;
   int destNhz;

   //Determine if this is 3-d or 2-d field and set/omit z-direction halos accordingly
   if(destNz > 1){
    destNhz = destNh;
   }else{
    destNhz = 0;
   }
   //Set the looping limits for looping over the Scatter DataBuffer
   iMin = destNh;
   iMax = destNx+destNh;
   jMin = destNh;
   jMax = destNy+destNh;
   kMin = destNh;
 
   /*Note we will move xy-columns (holding values in z) so k=0 is the reference index in z, for the column*/ 
   k=0;
   /*Prepare a buffer data array to scatter the global data*/
   if(mpi_rank_world == 0){
     if(srcNz > 1){ //This must be a 3-D field
       for(iRank=0; iRank < numProcsX*numProcsY; iRank++){
         for(i=iMin; i < iMax; i++){
           for(j=jMin; j < jMax; j++){
              rankStart = iRank*((destNx+2*destNh)*(destNy+2*destNh)*(destNz+2*destNhz));
              ijkSrc = (iRank%numProcsX)*destNx*srcNy*srcNz
                        +(i-destNh)*srcNy*srcNz
                        +((int) (iRank/numProcsX))*destNy*(srcNz)
                        +(j-destNh)*srcNz+k;
              ijkDest = i*(destNy+2*destNh)*(destNz+2*destNhz)+j*(destNz+2*destNhz)+(k+kMin);
              memcpy(&fempi_DataBuffer[rankStart+ijkDest],&srcFld[ijkSrc],destNz*sizeof(float));
           } // end for(j...
         } // end for(i...
       } // for(iRank...
     }else{ //This must be a 2-D field since srcNz <=1
       /*Map from buffer array to global array */
       j=jMin;
       for(iRank=0; iRank < numProcsX*numProcsY; iRank++){
          for(i=iMin; i < iMax; i++){
                 rankStart = iRank*((destNx+2*destNh)*(destNy+2*destNh));
                 ijkSrc = (iRank%numProcsX)*destNx*srcNy
                         +(i-destNh)*srcNy
                         +((int) (iRank/numProcsX))*destNy
                         +(j-destNh);
                 ijkDest = i*(destNy+2*destNh)+j;
                 memcpy(&fempi_DataBuffer[rankStart+ijkDest],&srcFld[ijkSrc],destNy*sizeof(float));
          } // end for(i...
       } // for(iRank...
     } //if/else srcNz >1 
   } //end if mpi_rank_world == 0   
   /* Scatter out the 3-D array of floats into rank-specific sub-domain, "local" arrays*/
   MPI_Scatter(fempi_DataBuffer,(destNx+2*destNh)*(destNy+2*destNh)*(destNz+2*destNhz),MPI_FLOAT,
               destFld, (destNx+2*destNh)*(destNy+2*destNh)*(destNz+2*destNhz),MPI_FLOAT, 0, MPI_COMM_WORLD);

   return(errorCode);
} //end fempi_ScatterVariable()

/*----->>>>> int fempi_GatherVariable();     --------------------------------------------------------------------
* Gathers a root-rank variable field defined on a collective domain from sub-domain partitions across ranks 
* in a 2-D horizontal decomposition where procID_X = , and procID_Y =...
*/
int fempi_GatherVariable(int srcNx,int srcNy,int srcNz, int srcNh,
                            int destNx,int destNy,int destNz,
                            float* srcFld,float* destFld){
   int errorCode = FEMPI_SUCCESS;
   int i,j,k,ijkSrc,ijkDest,iRank;
   int iMin,iMax,jMin,jMax,kMin;
   int rankStart;
   int srcNhz;

   if(srcNz > 1){
    srcNhz = srcNh;
   }else{
    srcNhz = 0;
   } 
   //Set the looping limits for looping over the Gather DataBuffer
   iMin = srcNh;
   iMax = srcNx+srcNh;
   jMin = srcNh;
   jMax = srcNy+srcNh;
   kMin = srcNh;

   //Gather the srcFld arrays back into fempi_DataBuffer and collect into fempi_DataBuffer
   MPI_Gather(srcFld,(srcNx+2*srcNh)*(srcNy+2*srcNh)*(srcNz+2*srcNhz),MPI_FLOAT,
              fempi_DataBuffer, (srcNx+2*srcNh)*(srcNy+2*srcNh)*(srcNz+2*srcNhz),MPI_FLOAT, 0, MPI_COMM_WORLD);
   /*Note we will move xy-columns (holding values in z) so k=0 is the reference index in z, for the column*/ 
   k=0;
   if(mpi_rank_world == 0){
     if(srcNz > 1){ //This must be a 3-D field
       /*Map from buffer array to global array */
       for(iRank=0; iRank < numProcsX*numProcsY; iRank++){
          for(i=iMin; i < iMax; i++){
             for(j=jMin; j < jMax; j++){
                 rankStart = iRank*((srcNx+2*srcNh)*(srcNy+2*srcNh)*(srcNz+2*srcNh));
                 ijkDest = (iRank%numProcsX)*srcNx*destNy*destNz
                         +(i-srcNh)*destNy*destNz
                         +((int) (iRank/numProcsX))*srcNy*(destNz)
                         +(j-srcNh)*destNz+k;
                 ijkSrc = i*(srcNy+2*srcNh)*(srcNz+2*srcNhz)+j*(srcNz+2*srcNhz)+(k+kMin);
                 memcpy(&destFld[ijkDest],&fempi_DataBuffer[rankStart+ijkSrc],srcNz*sizeof(float));
             } // end for(j...
          } // end for(i...
       } // for(iRank...
     }else{ //This must be a 2-D field since srcNz <=1
       /*Map from buffer array to global array */
       j=jMin;
       for(iRank=0; iRank < numProcsX*numProcsY; iRank++){
          for(i=iMin; i < iMax; i++){
                 rankStart = iRank*((srcNx+2*srcNh)*(srcNy+2*srcNh));
                 ijkDest = (iRank%numProcsX)*srcNx*destNy
                         +(i-srcNh)*destNy
                         +((int) (iRank/numProcsX))*srcNy
                         +(j-srcNh);
                 ijkSrc = i*(srcNy+2*srcNh)+j;
                 memcpy(&destFld[ijkDest],&fempi_DataBuffer[rankStart+ijkSrc],srcNy*sizeof(float));
          } // end for(i...
       } // for(iRank...
     } //if/else srcNz >1 
   } //end if mpi_rank_world == 0   

   /*If possible we would like to avoid a barrier here, but for now play it safe...*/
   MPI_Barrier(MPI_COMM_WORLD); 
   return(errorCode);
} //end fempi_GatherVariable()

/*----->>>>> int fempi_Cleanup();     ------------------------------------------------------------------
* Used to free all malloced memory by the FEMPI module.
*/
int fempi_Cleanup(){
   int errorCode = FEMPI_SUCCESS;

   /* Free any FEMPI module arrays */
   free(lorcv_buffer);
   free(losnd_buffer);
   free(hircv_buffer);
   free(hisnd_buffer);
   /* Free any root-rank-only FEMPI module arrays */
   if(mpi_rank_world == 0){
     free(fempi_DataBuffer);
   }//endif mpi_rank_world == 0 
   return(errorCode);
}//end fempi_Cleanup()

/*----->>>>> int fempi_FinalizeMPI();       ----------------------------------------------------------------------
* Used to finalize the base MPI environment. 
*/
int fempi_FinalizeMPI(){
   int errorCode = FEMPI_SUCCESS;
   
   /* Ensure synchronization at this point*/
   MPI_Barrier(MPI_COMM_WORLD); 
   /*Finalize the mpi environemnt*/
   MPI_Finalize();
 
   return(errorCode);
} //end fempi_FinalizeMPI()
