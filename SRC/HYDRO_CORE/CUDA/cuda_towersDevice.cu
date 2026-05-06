/* FastEddy®: SRC/HYDRO_CORE/CUDA/cuda_towersDevice.cu 
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


/*---TOWERS*/ 
__constant__ int rank_nTowers_d;   /*Number of towers within an mpi_rank_world subdomain*/
__constant__ int towerInstanceSize_d;    /*size of a timestep instance of all tower data*/
__constant__ int towerSurfInstanceSize_d;    /*size of a timestep instance of all tower surface data*/
float* towersData_d; /* Data Structure to store virtual tower data on device*/
float* towersSurfData_d; /* Data Structure to store virtual tower surface data on device*/
int* tower_iInds_d; /*rank-centric i-indices */
int* tower_jInds_d; /*rank-centric j-indices */

/*#################------------ TOWERS submodule function definitions ------------------#############*/
/*----->>>>> int cuda_towersDeviceSetup();       ---------------------------------------------------------
* Used to cudaMemcpy parameters and allocate arrays for the TOWERS_CUDA submodule.
*/
extern "C" int cuda_towersDeviceSetup(int NtBatch, int rank_nTowers, int towerInstanceSize, int towerSurfInstanceSize){   
   int errorCode = CUDA_TOWERS_SUCCESS;
   size_t NelemsTowers;
   size_t NelemsTowersSurf;

   cudaMemcpyToSymbol(rank_nTowers_d, &rank_nTowers, sizeof(int));
   cudaMemcpyToSymbol(towerInstanceSize_d, &towerInstanceSize, sizeof(int));
   cudaMemcpyToSymbol(towerSurfInstanceSize_d, &towerSurfInstanceSize, sizeof(int));

   if(rank_nTowers > 0){
     NelemsTowers = (size_t)(NtBatch*rank_nTowers*towerInstanceSize);
     fecuda_DeviceMalloc(NelemsTowers, &towersData_d);
     NelemsTowersSurf = (size_t)(NtBatch*rank_nTowers*towerSurfInstanceSize);
     fecuda_DeviceMalloc(NelemsTowersSurf, &towersSurfData_d);
     fecuda_DeviceMallocInt(rank_nTowers, &tower_iInds_d);
     cudaMemcpy(tower_iInds_d, tower_iInds, rank_nTowers*sizeof(int), cudaMemcpyHostToDevice);
     fecuda_DeviceMallocInt(rank_nTowers, &tower_jInds_d);
     cudaMemcpy(tower_jInds_d, tower_jInds, rank_nTowers*sizeof(int), cudaMemcpyHostToDevice);
   }
   printf("cuda_towersDeviceSetup: NelemsTowers = %d, NelemsTowersSurf = %d\n",NelemsTowers,NelemsTowersSurf);
   fflush(stdout);
   return(errorCode);
} //end cuda_towersDeviceSetup()

/*----->>>>> extern "C" int cuda_towersDeviceCleanup();  -----------------------------------------------------------
Used to free all malloced memory by the TOWERS submodule.
*/

extern "C" int cuda_towersDeviceCleanup(){
   int errorCode = CUDA_TOWERS_SUCCESS;

   /* Free any TOWERS submodule arrays */
   if(rank_nTowers > 0){
     cudaFree(towersData_d);
     cudaFree(towersSurfData_d);
     cudaFree(tower_iInds_d);
     cudaFree(tower_jInds_d);

   }
   return(errorCode);

}//end cuda_towersDeviceCleanup()

/*----->>>>> __global__ void  cudaDevice_towerAppendBuffers();  ------------------------------------------
* This is the gloabl-entry kernel routine for updating tower device-sided buffers
*/
__global__ void cudaDevice_towerAppendBuffers(int itBatch, int batchSize,
		                              float *towersData_d, float *towersSurfData_d, int *tower_iInds_d, int *tower_jInds_d,
                                              int Nhydro, float *hydroFlds_d,
                                              int Nsgstke, float *sgstkeScalars_d,
                                              int Nmoist, float *moistScalars_d,
                                              int NauxSc, float *hydroAuxScalars_d,
                                              int Ntaus, float *hydroTauFlds_d,
                                              int NtausMoist, float *moistTauFlds_d,
                                              float *z0m_d, float *z0t_d, float *tskin_d, float *qskin_d,
                                              float *fricVel_d, float *invOblen_d, float *htFlux_d, float *qFlux_d){
   
   int towerCount;
   int i,j,k;
   int ijk;
   int ij;
   int iStride,jStride,kStride;
   int iFld,fldStride;
   int towerBaseAddress;
   int towerSurfBaseAddress;
   int towerFld_size;
   int towerFld_cnt;
   int towIndx;
   /*Establish necessary indices for spatial locality*/
   i = (blockIdx.x)*blockDim.x + threadIdx.x;
   j = (blockIdx.y)*blockDim.y + threadIdx.y;
   k = (blockIdx.z)*blockDim.z + threadIdx.z;
   iStride = (Ny_d+2*Nh_d)*(Nz_d+2*Nh_d);
   jStride = (Nz_d+2*Nh_d);
   kStride = 1;
   fldStride = (Nx_d+2*Nh_d)*(Ny_d+2*Nh_d)*(Nz_d+2*Nh_d);
   towerFld_size = Nz_d;

   for(towerCount = 0; towerCount < rank_nTowers_d; towerCount++){
      if((i == tower_iInds_d[towerCount])&&
         (j == tower_jInds_d[towerCount]) && 
         (k >= kMin_d)&&(k < kMax_d) ){
         
         towerBaseAddress = towerCount*(batchSize*towerInstanceSize_d);
	 towerFld_cnt = 0;
	 ijk = i*iStride + j*jStride + k*kStride;
	 ij = i*(Ny_d+2*Nh_d) + j;
         for(iFld=0; iFld < Nhydro; iFld++){
            towIndx = towerBaseAddress + itBatch*towerInstanceSize_d + towerFld_cnt*towerFld_size + k-Nh_d;
            towersData_d[towIndx] = hydroFlds_d[iFld*fldStride+ijk];
	    towerFld_cnt += 1;
	 }
         for(iFld=0; iFld < Nsgstke; iFld++){
            towIndx = towerBaseAddress + itBatch*towerInstanceSize_d + towerFld_cnt*towerFld_size + k-Nh_d;
            towersData_d[towIndx] = sgstkeScalars_d[iFld*fldStride+ijk];
	    towerFld_cnt += 1;
	 }
         for(iFld=0; iFld < Nmoist; iFld++){
            towIndx = towerBaseAddress + itBatch*towerInstanceSize_d + towerFld_cnt*towerFld_size + k-Nh_d;
            towersData_d[towIndx] = moistScalars_d[iFld*fldStride+ijk];
	    towerFld_cnt += 1;
	 }
         for(iFld=0; iFld < NauxSc; iFld++){
            towIndx = towerBaseAddress + itBatch*towerInstanceSize_d + towerFld_cnt*towerFld_size + k-Nh_d;
            towersData_d[towIndx] = hydroAuxScalars_d[iFld*fldStride+ijk];
	    towerFld_cnt += 1;
	 }
         for(iFld=0; iFld < Ntaus; iFld++){
            towIndx = towerBaseAddress + itBatch*towerInstanceSize_d + towerFld_cnt*towerFld_size + k-Nh_d;
            towersData_d[towIndx] = hydroTauFlds_d[iFld*fldStride+ijk];
	    towerFld_cnt += 1;
	 }
         for(iFld=0; iFld < NtausMoist; iFld++){
            towIndx = towerBaseAddress + itBatch*towerInstanceSize_d + towerFld_cnt*towerFld_size + k-Nh_d;
            towersData_d[towIndx] = moistTauFlds_d[iFld*fldStride+ijk];
	    towerFld_cnt += 1;
	 }
	 if(k == kMin_d){
           towerSurfBaseAddress = towerCount*(batchSize*towerSurfInstanceSize_d);
           towIndx = towerSurfBaseAddress + itBatch*towerSurfInstanceSize_d;
	   towersSurfData_d[towIndx] = z0m_d[ij];
	   towIndx += 1; //only a single surface value so increment by 1
	   towersSurfData_d[towIndx] = z0t_d[ij];
	   towIndx += 1; //only a single surface value so increment by 1
	   towersSurfData_d[towIndx] = tskin_d[ij];
	   towIndx += 1; //only a single surface value so increment by 1
	   towersSurfData_d[towIndx] = fricVel_d[ij];
	   towIndx += 1; //only a single surface value so increment by 1
	   towersSurfData_d[towIndx] = invOblen_d[ij];
	   towIndx += 1; //only a single surface value so increment by 1
	   towersSurfData_d[towIndx] = htFlux_d[ij];
	   towIndx += 1; //only a single surface value so increment by 1
	   if(Nmoist > 0){
	     towersSurfData_d[towIndx] = qskin_d[ij];
	     towIndx += 1; //only a single surface value so increment by 1
	     towersSurfData_d[towIndx] = qFlux_d[ij];
	     towIndx += 1; //only a single surface value so increment by 1
	   }//end if Nmoist > 0
	 }//end if k == kMin_d
      }//end if i= tower_iInds_d[towerCount] && ...   
   }//end for towerCount
}//end cudaDevice_towerAppendBuffers
