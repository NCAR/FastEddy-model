/* FastEddy®: SRC/FECUDA/fecuda_Utils.cu 
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
#include <fempi.h>
#include <grid.h>
#include <fecuda.h>
#include <fecuda_Device_cu.h>

/*##############--------- FECUDA Utility (fecuda_Utils.cu) variable declarations -------------#################*/
float *haloSendBuff_d;  //Send-Buffer for coalesced halo exchanges
float *haloRecvBuff_d;  //Recieve-Buffer for coalesced halo exchanges

/*##########---------- FECUDA Utility (fecuda_Utils.cu) function declarations -----------#################*/
void createAndStartEvent(cudaEvent_t *startE, cudaEvent_t *stopE){
  cudaEventCreate(startE);
  cudaEventCreate(stopE);
  cudaEventRecord(*startE,0);
}

void stopSynchReportDestroyEvent(cudaEvent_t *startE, cudaEvent_t *stopE, float *elapsedTime){
  cudaEventRecord(*stopE,0);
  cudaEventSynchronize(*stopE);
  cudaEventElapsedTime(elapsedTime, *startE, *stopE);
  cudaEventDestroy(*startE);
  cudaEventDestroy(*stopE);
}
__global__ void fecuda_UtilsPackLoSideHaloBuff(float* sendFld_d,float* haloSendBuff_d,int Nxp,int Nyp,int Nzp,int Nh){
   int iStride,jStride,kStride;
   int i,j,k;
   int ijk,ijkBuff;
   int iMin,iMax;
   int jMin;
   int kMin,kMax;
   iMin = Nh;
   iMax = Nxp+Nh;
   jMin = Nh;
   kMin = Nh;
   kMax = Nzp+Nh;
   iStride = (Nyp+2*Nh)*(Nzp+2*Nh);
   jStride = (Nzp+2*Nh);
   kStride = 1;
   /*Establish necessary indices for spatial locality*/
   i = (blockIdx.x)*blockDim.x + threadIdx.x;
   j = (blockIdx.y)*blockDim.y + threadIdx.y;
   k = (blockIdx.z)*blockDim.z + threadIdx.z;

   if((i >= iMin-Nh)&&(i < iMax+Nh) &&     //for i=[0...Nxp+2*Nh)
      (j >= jMin)&&(j < jMin+Nh) &&        //for j=[Nh...2*Nh)
      (k >= kMin-Nh)&&(k < kMax+Nh) ){     //for k=[0...Nzp+2*Nh)
      ijk = i*iStride + j*jStride + k*kStride;
      ijkBuff=(i)*(Nh*jStride) + (j-Nh)*jStride + k*kStride;
      haloSendBuff_d[ijkBuff]=sendFld_d[ijk];
   }
}//end fecuda_UtilsPackLoHaloBuff
__global__ void fecuda_UtilsUnpackLoSideHaloBuff(float* recvFld_d,float* haloRecvBuff_d,int Nxp,int Nyp,int Nzp,int Nh){
   int iStride,jStride,kStride;
   int i,j,k;
   int ijk,ijkBuff;
   int iMin,iMax;
   int jMin;
   int kMin,kMax;
   iMin = Nh;
   iMax = Nxp+Nh;
   jMin = Nh;
   kMin = Nh;
   kMax = Nzp+Nh;
   iStride = (Nyp+2*Nh)*(Nzp+2*Nh);
   jStride = (Nzp+2*Nh);
   kStride = 1;
   /*Establish necessary indices for spatial locality*/
   i = (blockIdx.x)*blockDim.x + threadIdx.x;
   j = (blockIdx.y)*blockDim.y + threadIdx.y;
   k = (blockIdx.z)*blockDim.z + threadIdx.z;

   if((i >= iMin-Nh)&&(i < iMax+Nh) &&     //for i=[0...Nxp+2*Nh)
      (j >= jMin-Nh)&&(j < jMin) &&        //for j=[0...Nh)
      (k >= kMin-Nh)&&(k < kMax+Nh) ){     //for k=[0...Nzp+2*Nh)
      ijk = i*iStride + j*jStride + k*kStride;
      ijkBuff=(i)*(Nh*jStride) + (j)*jStride + k*kStride;
      recvFld_d[ijk]=haloRecvBuff_d[ijkBuff];
   }
}//end fecuda_UtilsUnpackLoHaloBuff

__global__ void fecuda_UtilsPackHiSideHaloBuff(float* sendFld_d,float* haloSendBuff_d,int Nxp,int Nyp,int Nzp,int Nh){
   int iStride,jStride,kStride;
   int i,j,k;
   int ijk,ijkBuff;
   int iMin,iMax;
   int jMax;
   int kMin,kMax;
   iMin = Nh;
   iMax = Nxp+Nh;
   jMax = Nyp+Nh;
   kMin = Nh;
   kMax = Nzp+Nh;
   iStride = (Nyp+2*Nh)*(Nzp+2*Nh);
   jStride = (Nzp+2*Nh);
   kStride = 1;
   /*Establish necessary indices for spatial locality*/
   i = (blockIdx.x)*blockDim.x + threadIdx.x;
   j = (blockIdx.y)*blockDim.y + threadIdx.y;
   k = (blockIdx.z)*blockDim.z + threadIdx.z;

   if((i >= iMin-Nh)&&(i < iMax+Nh) &&       //for i=[0...Nxp+2*Nh)
      (j >= jMax-Nh)&&(j < jMax) &&          //for j=[Nyp+Nh-Nh...Nyp+Nh)
      (k >= kMin-Nh)&&(k < kMax+Nh) ){       //for k=[0...Nzp+2*Nh)
      ijk = i*iStride + j*jStride + k*kStride;
      ijkBuff=(i)*(Nh*jStride) + (j-(jMax-Nh))*jStride + k*kStride;
      haloSendBuff_d[ijkBuff]=sendFld_d[ijk];
   }
}//end fecuda_UtilsPackHiHaloBuff
__global__ void fecuda_UtilsUnpackHiSideHaloBuff(float* recvFld_d,float* haloRecvBuff_d,int Nxp,int Nyp,int Nzp,int Nh){
   int iStride,jStride,kStride;
   int i,j,k;
   int ijk,ijkBuff;
   int iMin,iMax;
   int jMax;
   int kMin,kMax;
   iMin = Nh;
   iMax = Nxp+Nh;
   jMax = Nyp+Nh;
   kMin = Nh;
   kMax = Nzp+Nh;
   iStride = (Nyp+2*Nh)*(Nzp+2*Nh);
   jStride = (Nzp+2*Nh);
   kStride = 1;
   /*Establish necessary indices for spatial locality*/
   i = (blockIdx.x)*blockDim.x + threadIdx.x;
   j = (blockIdx.y)*blockDim.y + threadIdx.y;
   k = (blockIdx.z)*blockDim.z + threadIdx.z;
   
   if((i >= iMin-Nh)&&(i < iMax+Nh) &&    //for i=[0...Nxp+2*Nh)
      (j >= jMax)&&(j < jMax+Nh) &&       //for j=[Nyp+Nh...Nyp+2*Nh)
      (k >= kMin-Nh)&&(k < kMax+Nh) ){    //for k=[0...Nzp+2*Nh)
      ijk = i*iStride + j*jStride + k*kStride;
      ijkBuff=(i)*(Nh*jStride) + (j-(jMax))*jStride + k*kStride;
      recvFld_d[ijk]=haloRecvBuff_d[ijkBuff];
   }  
}//end fecuda_UtilsUnpackHiHaloBuff

/*----->>>>> int fecuda_UtilsAllocateHaloBuffers(); -------------------------------------------------------------
Used to allocate device memory buffers for coalesced halo exchanges in the FECUDA module.
*/
extern "C" int fecuda_UtilsAllocateHaloBuffers(int Nxp, int Nyp, int Nzp, int Nh){ //These extents are per-rank
   int errorCode = FECUDA_SUCCESS;
   int Nelems;
   Nelems=(Nxp+2*Nh)*(Nh)*(Nzp+2*Nh);   //Needs to hold the x-direction extent worth of Nh deep in y-dir columns of Nzp+2*Nh

   fecuda_DeviceMalloc(Nelems*sizeof(float), &haloSendBuff_d);
   fecuda_DeviceMalloc(Nelems*sizeof(float), &haloRecvBuff_d);

   return(errorCode);
}//end fecuda_UtilsAllocateHaloBuffers()

/*----->>>>> int fecuda_UtilsDeallocateHaloBuffers(); -------------------------------------------------------------
Used to free device memory buffers for coalesced halo exchanges in the FECUDA module.
*/
extern "C" int fecuda_UtilsDeallocateHaloBuffers(){ 
   int errorCode = FECUDA_SUCCESS;

   cudaFree(haloSendBuff_d);
   cudaFree(haloRecvBuff_d);

   return(errorCode);
}//end fecuda_UtilsDeallocateHaloBuffers()

/*----->>>>> void fecuda_DeviceMalloc();    -----------------------------------------------------------
* Used to allocate device memory float blocks and set the  host memory addresses of device memory pointers.
*/
extern "C" void fecuda_DeviceMalloc(int Nelems, float** memBlock_d) {
    cudaMalloc((void**)memBlock_d,sizeof(float)*Nelems);
    gpuErrchk( cudaPeekAtLastError() );
    cudaMemset(*memBlock_d,'\0',sizeof(float)*Nelems);    
    gpuErrchk( cudaPeekAtLastError() );
#ifdef DEBUG
    printf("New device memory allocation, device pointer is stored at host address %p as %p\n",memBlock_d, *memBlock_d);
#endif
}
extern "C" void fecuda_DeviceMallocInt(int Nelems, int** memBlock_d) {
    cudaMalloc((void**)memBlock_d,sizeof(int)*Nelems);
    gpuErrchk( cudaPeekAtLastError() );
    cudaMemset(*memBlock_d,'\0',sizeof(int)*Nelems);
    gpuErrchk( cudaPeekAtLastError() );
#ifdef DEBUG
    printf("New device memory allocation, device pointer is stored at host address %p as %p\n",memBlock_d, *memBlock_d);
#endif
}

/*----->>>>> int fecuda_SendRecvWestEast(); -------------------------------------------------------------------
Used to perform western/eastern device domain halo exchange for an arbitrary field.
*/
extern "C" int fecuda_SendRecvWestEast(float* sendFld_d, float* recvFld_d, int hydroBCs){
   int errorCode = FECUDA_SUCCESS;
#if HALO_SNDRCV_FORM 
   /*West-send to East-rev*/
   MPI_Status status;
   MPI_Sendrecv(&sendFld_d[Nh*(Nyp+2*Nh)*(Nzp+2*Nh)],Nh*(Nyp+2*Nh)*(Nzp+2*Nh),MPI_FLOAT,mpi_nbrXlo,123,
                &recvFld_d[(Nxp+Nh)*(Nyp+2*Nh)*(Nzp+2*Nh)],Nh*(Nyp+2*Nh)*(Nzp+2*Nh),MPI_FLOAT,mpi_nbrXhi,123,
                MPI_COMM_WORLD, &status);
#else
   MPI_Status send_status;
   MPI_Status recv_status;
   MPI_Request send_request;
   MPI_Request recv_request;
   if(hydroBCs==2){ //if periodic, everyone exchange
     errorCode=MPI_Isend(&sendFld_d[Nh*(Nyp+2*Nh)*(Nzp+2*Nh)],Nh*(Nyp+2*Nh)*(Nzp+2*Nh),MPI_FLOAT,mpi_nbrXlo,123,MPI_COMM_WORLD,&send_request);
     errorCode=MPI_Irecv(&recvFld_d[(Nxp+Nh)*(Nyp+2*Nh)*(Nzp+2*Nh)],Nh*(Nyp+2*Nh)*(Nzp+2*Nh),MPI_FLOAT,mpi_nbrXhi,123,MPI_COMM_WORLD,&recv_request);
     errorCode=MPI_Wait(&send_request,&send_status);
     errorCode=MPI_Wait(&recv_request,&recv_status);
   }else{
     if(mpi_XloBndyRank != 1){
       errorCode=MPI_Isend(&sendFld_d[Nh*(Nyp+2*Nh)*(Nzp+2*Nh)],Nh*(Nyp+2*Nh)*(Nzp+2*Nh),MPI_FLOAT,mpi_nbrXlo,123,MPI_COMM_WORLD,&send_request);
     }
     if(mpi_XhiBndyRank != 1){
       errorCode=MPI_Irecv(&recvFld_d[(Nxp+Nh)*(Nyp+2*Nh)*(Nzp+2*Nh)],Nh*(Nyp+2*Nh)*(Nzp+2*Nh),MPI_FLOAT,mpi_nbrXhi,123,MPI_COMM_WORLD,&recv_request);
     }
     if(mpi_XloBndyRank != 1){
       errorCode=MPI_Wait(&send_request,&send_status);
     }
     if(mpi_XhiBndyRank != 1){
       errorCode=MPI_Wait(&recv_request,&recv_status);
     }
   } //end if not periodic!
#endif
   return(errorCode);
}//end fecuda_SendRecvWestEast()

/*----->>>>> int fecuda_SendRecvEastWest(); -------------------------------------------------------------------
Used to perform eastern/western device domain halo exchange for an arbitrary field.
*/
extern "C" int fecuda_SendRecvEastWest(float* sendFld_d, float* recvFld_d, int hydroBCs){
   int errorCode = FECUDA_SUCCESS;

#if HALO_SNDRCV_FORM 
   MPI_Status status;
   /*East-send to West-recv*/
   MPI_Sendrecv(&sendFld_d[(Nxp)*(Nyp+2*Nh)*(Nzp+2*Nh)],Nh*(Nyp+2*Nh)*(Nzp+2*Nh),MPI_FLOAT,mpi_nbrXhi,456,
                &recvFld_d[0],Nh*(Nyp+2*Nh)*(Nzp+2*Nh),MPI_FLOAT,mpi_nbrXlo,456,
                MPI_COMM_WORLD, &status);
#else
   MPI_Status send_status;
   MPI_Status recv_status;
   MPI_Request send_request;
   MPI_Request recv_request;
   if(hydroBCs==2){ //if periodic, everyone exchange
     errorCode=MPI_Isend(&sendFld_d[(Nxp)*(Nyp+2*Nh)*(Nzp+2*Nh)],Nh*(Nyp+2*Nh)*(Nzp+2*Nh),MPI_FLOAT,mpi_nbrXhi,456,MPI_COMM_WORLD,&send_request);
     errorCode=MPI_Irecv(&recvFld_d[0],Nh*(Nyp+2*Nh)*(Nzp+2*Nh),MPI_FLOAT,mpi_nbrXlo,456,MPI_COMM_WORLD,&recv_request);
     errorCode=MPI_Wait(&send_request,&send_status);
     errorCode=MPI_Wait(&recv_request,&recv_status);
   }else{
     if(mpi_XhiBndyRank != 1){
       errorCode=MPI_Isend(&sendFld_d[(Nxp)*(Nyp+2*Nh)*(Nzp+2*Nh)],Nh*(Nyp+2*Nh)*(Nzp+2*Nh),MPI_FLOAT,mpi_nbrXhi,456,MPI_COMM_WORLD,&send_request);
     }
     if(mpi_XloBndyRank != 1){
       errorCode=MPI_Irecv(&recvFld_d[0],Nh*(Nyp+2*Nh)*(Nzp+2*Nh),MPI_FLOAT,mpi_nbrXlo,456,MPI_COMM_WORLD,&recv_request);
     }
     if(mpi_XhiBndyRank != 1){
       errorCode=MPI_Wait(&send_request,&send_status);
     }
     if(mpi_XloBndyRank != 1){
       errorCode=MPI_Wait(&recv_request,&recv_status);
     }
   } //end if not periodic!
#endif
   return(errorCode);

}//end fecuda_SendRecvEastWest()

/*----->>>>> int fecuda_SendRecvSouthNorth(); -------------------------------------------------------------------
Used to perform southern/northern device domain halo exchange for an arbitrary field.
*/
extern "C" int fecuda_SendRecvSouthNorth(float* sendFld_d, float* recvFld_d, int hydroBCs){
   int errorCode = FECUDA_SUCCESS;
   int i;
//#define NONOPTIMAL
#if HALO_SNDRCV_FORM 
   MPI_Status status;
   /*South-send to North-recv*/
   for(i=Nh; i<Nxp+Nh; i++){
     MPI_Sendrecv(&sendFld_d[i*(Nyp+2*Nh)*(Nzp+2*Nh)+Nh*(Nzp+2*Nh)],Nh*(Nzp+2*Nh),MPI_FLOAT,mpi_nbrYlo,789,
                  &recvFld_d[i*(Nyp+2*Nh)*(Nzp+2*Nh)+(Nyp+Nh)*(Nzp+2*Nh)],Nh*(Nzp+2*Nh),MPI_FLOAT,mpi_nbrYhi,789,
                  MPI_COMM_WORLD, &status);
   }//end for i
#else
   MPI_Status send_status;
   MPI_Status recv_status;
   MPI_Request send_request;
   MPI_Request recv_request;
   if(hydroBCs==2){ //if periodic, everyone exchange
#ifdef NONOPTIMAL
     for(i=Nh; i<Nxp+Nh; i++){
       errorCode=MPI_Isend(&sendFld_d[i*(Nyp+2*Nh)*(Nzp+2*Nh)+Nh*(Nzp+2*Nh)],
                           Nh*(Nzp+2*Nh),MPI_FLOAT,mpi_nbrYlo,789,MPI_COMM_WORLD,&send_request);
       errorCode=MPI_Irecv(&recvFld_d[i*(Nyp+2*Nh)*(Nzp+2*Nh)+(Nyp+Nh)*(Nzp+2*Nh)],
                           Nh*(Nzp+2*Nh),MPI_FLOAT,mpi_nbrYhi,789,MPI_COMM_WORLD,&recv_request);
       errorCode=MPI_Wait(&send_request,&send_status);
       errorCode=MPI_Wait(&recv_request,&recv_status);
     }//end for i
#else
     //Pack the buffer
     fecuda_UtilsPackLoSideHaloBuff<<<grid, tBlock>>>(sendFld_d,haloSendBuff_d,Nxp,Nyp,Nzp,Nh);
     gpuErrchk( cudaGetLastError() );
     gpuErrchk( cudaDeviceSynchronize() );
     //Send/Recv the buffered halos 
     errorCode=MPI_Isend(&haloSendBuff_d[0],
                         (Nxp+2*Nh)*Nh*(Nzp+2*Nh),MPI_FLOAT,mpi_nbrYlo,789,MPI_COMM_WORLD,&send_request);
     errorCode=MPI_Irecv(&haloRecvBuff_d[0],
                         (Nxp+2*Nh)*Nh*(Nzp+2*Nh),MPI_FLOAT,mpi_nbrYhi,789,MPI_COMM_WORLD,&recv_request);
     errorCode=MPI_Wait(&send_request,&send_status);
     errorCode=MPI_Wait(&recv_request,&recv_status);
     //Unpack the buffer
     fecuda_UtilsUnpackHiSideHaloBuff<<<grid, tBlock>>>(recvFld_d,haloRecvBuff_d,Nxp,Nyp,Nzp,Nh);
     gpuErrchk( cudaGetLastError() );
     gpuErrchk( cudaDeviceSynchronize() );
#endif
   }else{
     for(i=Nh; i<Nxp+Nh; i++){
       if(mpi_YloBndyRank != 1){
         errorCode=MPI_Isend(&sendFld_d[i*(Nyp+2*Nh)*(Nzp+2*Nh)+Nh*(Nzp+2*Nh)],
                             Nh*(Nzp+2*Nh),MPI_FLOAT,mpi_nbrYlo,789+i*mpi_rank_world,MPI_COMM_WORLD,&send_request);
       } //if this is not a lower-boundary rank (implying no low neighbor)
       if(mpi_YhiBndyRank != 1){ 
         errorCode=MPI_Irecv(&recvFld_d[i*(Nyp+2*Nh)*(Nzp+2*Nh)+(Nyp+Nh)*(Nzp+2*Nh)],
                             Nh*(Nzp+2*Nh),MPI_FLOAT,mpi_nbrYhi,789+i*mpi_nbrYhi,MPI_COMM_WORLD,&recv_request);
       } //if this is not a hi-boundary rank (implying no high neighbor)
       if(mpi_YloBndyRank != 1){
         errorCode=MPI_Wait(&send_request,&send_status);
       } //if this is not a lower-boundary ranka (implying no low neighbor)
       if(mpi_YhiBndyRank != 1){ 
         errorCode=MPI_Wait(&recv_request,&recv_status);
       } //if this is not a hi-boundary rank (implying no high neighbor)
     }//end for i
   } //end if not periodic!
#endif

   return(errorCode);
}//end fecuda_SendRecvSouthNorth()

/*----->>>>> int fecuda_SendRecvNorthSouth(); -------------------------------------------------------------------
Used to perform northern/southern device domain halo exchange for an arbitrary field.
*/
extern "C" int fecuda_SendRecvNorthSouth(float* sendFld_d, float* recvFld_d, int hydroBCs){
   int errorCode = FECUDA_SUCCESS;
   int i;
#if HALO_SNDRCV_FORM 
   MPI_Status status;
   /*South-send to North-recv*/
   for(i=Nh; i<Nxp+Nh; i++){
     MPI_Sendrecv(&sendFld_d[i*(Nyp+2*Nh)*(Nzp+2*Nh)+(Nyp)*(Nzp+2*Nh)],Nh*(Nzp+2*Nh),MPI_FLOAT,mpi_nbrYhi,987,
                  &recvFld_d[i*(Nyp+2*Nh)*(Nzp+2*Nh)],Nh*(Nzp+2*Nh),MPI_FLOAT,mpi_nbrYlo,987,
                  MPI_COMM_WORLD, &status);
   }//end for i
#else
   MPI_Status send_status;
   MPI_Status recv_status;
   MPI_Request send_request;
   MPI_Request recv_request;
   if(hydroBCs==2){ //if periodic, everyone exchange
#ifdef NONOPTIMAL
     for(i=Nh; i<Nxp+Nh; i++){
       errorCode=MPI_Isend(&sendFld_d[i*(Nyp+2*Nh)*(Nzp+2*Nh)+(Nyp)*(Nzp+2*Nh)],
                           Nh*(Nzp+2*Nh),MPI_FLOAT,mpi_nbrYhi,987,MPI_COMM_WORLD,&send_request);
       errorCode=MPI_Irecv(&recvFld_d[i*(Nyp+2*Nh)*(Nzp+2*Nh)],
                           Nh*(Nzp+2*Nh),MPI_FLOAT,mpi_nbrYlo,987,MPI_COMM_WORLD,&recv_request);
       errorCode=MPI_Wait(&send_request,&send_status);
       errorCode=MPI_Wait(&recv_request,&recv_status);
     }//end for i
#else
     //Pack the buffer
     fecuda_UtilsPackHiSideHaloBuff<<<grid, tBlock>>>(sendFld_d,haloSendBuff_d,Nxp,Nyp,Nzp,Nh);
     gpuErrchk( cudaGetLastError() );
     gpuErrchk( cudaDeviceSynchronize() );
     //Send/Recv the buffered halos 
     errorCode=MPI_Isend(&haloSendBuff_d[0],
                         (Nxp+2*Nh)*Nh*(Nzp+2*Nh),MPI_FLOAT,mpi_nbrYhi,987,MPI_COMM_WORLD,&send_request);
     errorCode=MPI_Irecv(&haloRecvBuff_d[0],
                         (Nxp+2*Nh)*Nh*(Nzp+2*Nh),MPI_FLOAT,mpi_nbrYlo,987,MPI_COMM_WORLD,&recv_request);
     errorCode=MPI_Wait(&send_request,&send_status);
     errorCode=MPI_Wait(&recv_request,&recv_status);
     //Unpack the buffer
     fecuda_UtilsUnpackLoSideHaloBuff<<<grid, tBlock>>>(recvFld_d,haloRecvBuff_d,Nxp,Nyp,Nzp,Nh);
     gpuErrchk( cudaGetLastError() );
     gpuErrchk( cudaDeviceSynchronize() );
#endif
   }else{
     for(i=Nh; i<Nxp+Nh; i++){
       if(mpi_YhiBndyRank != 1){
         errorCode=MPI_Isend(&sendFld_d[i*(Nyp+2*Nh)*(Nzp+2*Nh)+(Nyp)*(Nzp+2*Nh)],
                             Nh*(Nzp+2*Nh),MPI_FLOAT,mpi_nbrYhi,987+i*mpi_rank_world,MPI_COMM_WORLD,&send_request);
       } //if this is not a lower-boundary rank (implying no low neighbor)
       if(mpi_YloBndyRank != 1){
         errorCode=MPI_Irecv(&recvFld_d[i*(Nyp+2*Nh)*(Nzp+2*Nh)],
                             Nh*(Nzp+2*Nh),MPI_FLOAT,mpi_nbrYlo,987+i*mpi_nbrYlo,MPI_COMM_WORLD,&recv_request);
       } //if this is not a hi-boundary rank (implying no high neighbor)
       if(mpi_YhiBndyRank != 1){
         errorCode=MPI_Wait(&send_request,&send_status);
       } //if this is not a hi-boundary rank (implying no high neighbor)
       if(mpi_YloBndyRank != 1){
         errorCode=MPI_Wait(&recv_request,&recv_status);
       } //if this is not a lower-boundary rank (implying no low neighbor)
     }//end for i
   } //end if not periodic!   
#endif
   return(errorCode);
}//end fecuda_SendRecvNorthSouth()

