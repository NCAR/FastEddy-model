/* FastEddy®: SRC/HYDRO_CORE/CUDA/cuda_cellpertDevice.cu
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
/*---CELL PERTURBATION METHOD*/
__constant__ int cellpertSelector_d;    /*CP method selector: 0= off, 1= on */
__constant__ int cellpert_sw2b_d;       /* switch to do: 0= all four lateral boundaries, 1= only south & west boundaries, 2= only south boundary */
__constant__ float cellpert_amp_d;      /* maximum amplitude for the potential temperature perturbations */
__constant__ int cellpert_nts_d;        /* number of time steps after which perturbations are seeded */
__constant__ int cellpert_gppc_d;       /* number of grid points conforming the cell */
__constant__ int cellpert_ndbc_d;       /* number of cells normal to domain lateral boundaries */
__constant__ int cellpert_kbottom_d;    /* z-grid point where the perturbations start */
__constant__ int cellpert_ktop_d;       /* z-grid point where the perturbations end */
__constant__ float cellpert_eckert_d;   /* Eckert number for the potential temperature perturbations (hydroBCs == 5) */
__constant__ float cellpert_tsfact_d;   /* factor on the refreshing perturbation time scale (hydroBCs == 5) */
float* randcp_d;            /*Base address for pseudo-random numbers used for cell perturbations (1d-array)*/

/*#################------------ CELLPERT submodule function definitions ------------------#############*/
/*----->>>>> int cuda_moistureDeviceSetup();       ---------------------------------------------------------
 * Used to cudaMalloc and cudaMemcpy parameters and coordinate arrays, and for the CELLPERT_CUDA submodule.
*/
extern "C" int cuda_cellpertDeviceSetup(){
   int errorCode = CUDA_CELLPERT_SUCCESS;
   int Nelems1d_xy, Nelems1d;

   /*Constants*/
   cudaMemcpyToSymbol(cellpertSelector_d, &cellpertSelector, sizeof(int));
   cudaMemcpyToSymbol(cellpert_sw2b_d, &cellpert_sw2b, sizeof(int));
   cudaMemcpyToSymbol(cellpert_amp_d, &cellpert_amp, sizeof(float));
   cudaMemcpyToSymbol(cellpert_nts_d, &cellpert_nts, sizeof(int));
   cudaMemcpyToSymbol(cellpert_gppc_d, &cellpert_gppc, sizeof(int));
   cudaMemcpyToSymbol(cellpert_ndbc_d, &cellpert_ndbc, sizeof(int));
   cudaMemcpyToSymbol(cellpert_kbottom_d, &cellpert_kbottom, sizeof(int));
   cudaMemcpyToSymbol(cellpert_ktop_d, &cellpert_ktop, sizeof(int));

   Nelems1d_xy = (Nx/cellpert_gppc+min(Nx%cellpert_gppc,1))*(2*cellpert_ndbc+min(Ny%cellpert_gppc,1)) + (Ny/cellpert_gppc-2*cellpert_ndbc)*(2*cellpert_ndbc+min(Nx%cellpert_gppc,1));
   Nelems1d = Nelems1d_xy*(cellpert_ktop-cellpert_kbottom+1);
   fecuda_DeviceMalloc(Nelems1d*sizeof(float), &randcp_d);

   return(errorCode);
} //end cuda_cellpertDeviceSetup()

/*----->>>>> extern "C" int cuda_cellpertDeviceCleanup();  -----------------------------------------------------------
Used to free all malloced memory by the CELLPERT submodule.
*/

extern "C" int cuda_cellpertDeviceCleanup(){
   int errorCode = CUDA_CELLPERT_SUCCESS;

   /* Free any cellpert submodule arrays */
   cudaFree(randcp_d);

   return(errorCode);

}//end cuda_cellpertDeviceCleanup()

/*----->>>>> extern "C" int cuda_hydroCoreDeviceBuildCPmethod();  --------------------------------------------------
* This routine provides the externally callable cuda-kernel call to apply cell perturbation method
*/
extern "C" int cuda_hydroCoreDeviceBuildCPmethod(int simTime_it){
   int errorCode = CUDA_CELLPERT_SUCCESS;
#ifdef TIMERS_LEVEL2
   cudaEvent_t startE, stopE;
   float elapsedTime;
#endif
   curandGenerator_t gen;
   int n_xy,n_tot;

#ifdef DEBUG
   printf("cuda_hydroCoreDeviceBuildCPmethod: tBlock = {%d, %d, %d}\n",tBlock.x, tBlock.y, tBlock.z);
   printf("cuda_hydroCoreDeviceBuildCPmethod: grid = {%d, %d, %d}\n",grid.x, grid.y, grid.z);
   fflush(stdout);
#endif

//#define TIMERS_LEVEL1
#ifdef TIMERS_LEVEL1
   /*Launch a blocking kernel to Perform the CP method */
   createAndStartEvent(&startE, &stopE);
#endif

// uniform distribution of pseudo-random numbers on randcp_d (1d-array)
   n_xy = (Nx/cellpert_gppc+min(Nx%cellpert_gppc,1))*(2*cellpert_ndbc+min(Ny%cellpert_gppc,1)) + (Ny/cellpert_gppc-2*cellpert_ndbc)*(2*cellpert_ndbc+min(Nx%cellpert_gppc,1));
   n_tot = n_xy*(cellpert_ktop-cellpert_kbottom+1);

   curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT);
   curandSetPseudoRandomGeneratorSeed(gen,simTime_it);
   curandGenerateUniform(gen,randcp_d,n_tot);

#ifdef URBAN_EXT
   if(urbanSelector > 0){
     cudaDevice_hydroCoreCompleteCellPerturbationMasked<<<grid, tBlock>>>(hydroFlds_d,randcp_d,mpi_rank_world,numProcsX,numProcsY,building_mask_d);
   }else{
     cudaDevice_hydroCoreCompleteCellPerturbation<<<grid, tBlock>>>(hydroFlds_d,randcp_d,mpi_rank_world,numProcsX,numProcsY);
   }
#else
     cudaDevice_hydroCoreCompleteCellPerturbation<<<grid, tBlock>>>(hydroFlds_d,randcp_d,mpi_rank_world,numProcsX,numProcsY);
#endif

//#define TIMERS_LEVEL2
#ifdef TIMERS_LEVEL2
   //gpuErrchk( cudaPeekAtLastError() );
   stopSynchReportDestroyEvent(&startE, &stopE, &elapsedTime);
   printf("cuda_hydroCoreDeviceBuildCPmethod()  Kernel execution time (ms): %12.8f\n", elapsedTime);
   gpuErrchk( cudaPeekAtLastError() ); //Check for errors in the cudaMemCpy calls
   gpuErrchk( cudaDeviceSynchronize() );
#endif

   gpuErrchk( cudaGetLastError() );
   gpuErrchk( cudaDeviceSynchronize() );

   return(errorCode);
}//end cuda_hydroCoreDeviceBuildCPmethod()

/*----->>>>> extern "C" int cuda_hydroCoreTVCP();  -----------------------------------------------------------
* Updates device-sided parameters used by the CELLPERT submodule from dynamic lateral BNDY conditions
*/
extern "C" int cuda_hydroCoreTVCP(){
    int errorCode = CUDA_CELLPERT_SUCCESS;

    cudaMemcpyToSymbol(cellpert_amp_d, &cellpert_amp, sizeof(float));
    cudaMemcpyToSymbol(cellpert_ktop_d, &cellpert_ktop, sizeof(int));
    cudaMemcpyToSymbol(cellpert_nts_d, &cellpert_nts, sizeof(int));

    return(errorCode);
} //end cuda_hydroCoreTVCP()

__global__ void cudaDevice_hydroCoreCompleteCellPerturbation(float* hydroFlds, float* randcp_d, int my_mpi, int numpx, int numpy){

   int i,j,k,ijk;
   int fldStride;
   int iStride,jStride,kStride;

   /*Establish necessary indices for spatial locality*/
   i = (blockIdx.x)*blockDim.x + threadIdx.x;
   j = (blockIdx.y)*blockDim.y + threadIdx.y;
   k = (blockIdx.z)*blockDim.z + threadIdx.z;

   fldStride = (Nx_d+2*Nh_d)*(Ny_d+2*Nh_d)*(Nz_d+2*Nh_d);
   iStride = (Ny_d+2*Nh_d)*(Nz_d+2*Nh_d);
   jStride = (Nz_d+2*Nh_d);
   kStride = 1;

   if((i >= iMin_d)&&(i < iMax_d) &&
      (j >= jMin_d)&&(j < jMax_d) &&
      (k >= kMin_d)&&(k < kMax_d) ){
      if((k >= (cellpert_kbottom_d+Nh_d-1))&&(k <= (cellpert_ktop_d+Nh_d-1))){ // call to cell perturbation device kernel
        ijk = i*iStride + j*jStride + k*kStride;
        cudaDevice_CellPerturbation(i,j,k,Nx_d,Ny_d,Nz_d,Nh_d,my_mpi,numpx,numpy,&hydroFlds[RHO_INDX*fldStride+ijk],&hydroFlds[THETA_INDX*fldStride+ijk],randcp_d);
      }
   }//end if in the range of non-halo cells

} // end cudaDevice_hydroCoreCompleteCellPerturbation()

__global__ void cudaDevice_hydroCoreCompleteCellPerturbationMasked(float* hydroFlds, float* randcp_d, int my_mpi, int numpx, int numpy, float* bdg_mask){

   int i,j,k,ijk;
   int fldStride;
   int iStride,jStride,kStride;

   /*Establish necessary indices for spatial locality*/
   i = (blockIdx.x)*blockDim.x + threadIdx.x;
   j = (blockIdx.y)*blockDim.y + threadIdx.y;
   k = (blockIdx.z)*blockDim.z + threadIdx.z;

   fldStride = (Nx_d+2*Nh_d)*(Ny_d+2*Nh_d)*(Nz_d+2*Nh_d);
   iStride = (Ny_d+2*Nh_d)*(Nz_d+2*Nh_d);
   jStride = (Nz_d+2*Nh_d);
   kStride = 1;

   if((i >= iMin_d)&&(i < iMax_d) &&
      (j >= jMin_d)&&(j < jMax_d) &&
      (k >= kMin_d)&&(k < kMax_d) ){
      if((k >= (cellpert_kbottom_d+Nh_d-1))&&(k <= (cellpert_ktop_d+Nh_d-1))){ // call to cell perturbation device kernel
        ijk = i*iStride + j*jStride + k*kStride;
        cudaDevice_CellPerturbationMasked(i,j,k,Nx_d,Ny_d,Nz_d,Nh_d,my_mpi,numpx,numpy,&hydroFlds[RHO_INDX*fldStride+ijk],&hydroFlds[THETA_INDX*fldStride+ijk],randcp_d,&bdg_mask[ijk]);
      }
   }//end if in the range of non-halo cells

} // end cudaDevice_hydroCoreCompleteCellPerturbationMasked()

/*----->>>>> __device__ void  cudaDevice_CellPerturbation();  --------------------------------------------------
*/
__device__ void cudaDevice_CellPerturbation(int i_ind, int j_ind, int k_ind, int Nx, int Ny, int Nz, int Nh, int my_mpi, int numpx, int numpy, float* rho, float* theta, float* rand_1darray){
  float th_ijk,rho_ijk;
  int ncx,ncy,ncz,cell_id,nc_tot,nc_xy,ngpb,ncx_tt,ncy_tt,nc_x2p,nc_y2p;
  int ii,jj,kk;
  int gppc,ndbc,ncx_p,ncy_p;
  int p_1,p_2,p_3,p_4,p_5,p_6,p_7;
  int xoffset,yoffset,Nx_tot,Ny_tot;
  // Nx, Ny domain dimensions passed in are single GPU dimensions

  xoffset = (my_mpi%numpx)*Nx;
  yoffset = (my_mpi%numpy)*Ny;
  Nx_tot = numpx*Nx; // total domain extent in the x-direction
  Ny_tot = numpy*Ny; // total domain extent in the y-direction

  th_ijk = *theta/ *rho;
  rho_ijk = *rho;
  ii = i_ind-Nh+xoffset; // remove halo index, starts from 0 + makes it global
  jj = j_ind-Nh+yoffset; // remove halo index, starts from 0 + makes it global
  kk = k_ind - (cellpert_kbottom_d+Nh-1); // start from 0 at the first perturbation level

  gppc = cellpert_gppc_d;
  ndbc = cellpert_ndbc_d;
  ncx_p = min(Nx_tot%gppc,1);
  ncx = (Nx_tot/gppc)+ ncx_p;
  ncy_p = min(Ny_tot%gppc,1);
  ncy = (Ny_tot/gppc);
  ncz = cellpert_ktop_d-cellpert_kbottom_d+1;

// only domain boundary ring
  nc_xy = ncx*(2*ndbc+ncy_p) + (ncy-2*ndbc)*(2*ndbc+ncx_p);
  nc_tot = nc_xy*ncz; // total number of cells in the domain (random number array size)
  ngpb = gppc*ndbc;
  nc_x2p = ncx - ndbc;
  nc_y2p = ncy - ndbc;

  if ( cellpert_sw2b_d == 1){ // 2-boundary version (south and west) for idealized cases
    if (ii >= ngpb && jj >= ngpb){ // DO NOTHING (k limits checked on function call)
    }else{
      ncx_tt = ii/gppc;
      ncy_tt = jj/gppc;
      p_1 = nc_xy*kk + min(ncy_tt,ndbc)*ncx + ncx_tt*(1-min(ncy_tt/ndbc,1));
      p_2 = min(max(ncy_tt-ndbc,0),nc_y2p-ndbc-1)*(2*ndbc+ncx_p);
      p_3 = min(ncx_tt,ndbc-1) + max(min(ncx_tt-nc_x2p+1+ncx_p,ndbc+ncx_p),0);
      p_3 = p_3*min(max(ncy_tt-ndbc+1,0),1);
      p_4 = (ncy_tt/nc_y2p)*max(ncy_tt-nc_y2p,0)*(ncx);
      p_5 = (ncy_tt/nc_y2p)*(2*ndbc+ncx_p);
      p_6 = (ncy_tt/nc_y2p)*min(ncx_tt/ndbc,1)*(ncx_tt-ndbc+1);
      p_7 = -(ncy_tt/nc_y2p)*(ncx_tt/(nc_x2p-ncx_p))*(ncx_tt-nc_x2p+1+ncx_p);
      cell_id = max(min(p_1+p_2+p_3+p_4+p_5+p_6+p_7,nc_tot-1),0);
      *theta = rho_ijk*(th_ijk+cellpert_amp_d*(rand_1darray[cell_id]-0.5));
    }
  }else if( cellpert_sw2b_d == 2){ // 1-boundary version (south) for idealized cases
    if (jj >= ngpb){ // DO NOTHING (k limits checked on function call)
    }else{
      ncx_tt = ii/gppc;
      ncy_tt = jj/gppc;
      p_1 = nc_xy*kk + min(ncy_tt,ndbc)*ncx + ncx_tt*(1-min(ncy_tt/ndbc,1));
      cell_id = max(min(p_1,nc_tot-1),0);
      *theta = rho_ijk*(th_ijk+cellpert_amp_d*(rand_1darray[cell_id]-0.5));
    }
  }else if( cellpert_sw2b_d == 3){ // 1-boundary version (west) for idealized cases
    if (ii >= ngpb){ // DO NOTHING (k limits checked on function call)
    }else{
      ncx_tt = ii/gppc;
      ncy_tt = jj/gppc;
      p_1 = nc_xy*kk + min(ncy_tt,ndbc)*ncx + ncx_tt*(1-min(ncy_tt/ndbc,1));
      p_2 = min(max(ncy_tt-ndbc,0),nc_y2p-ndbc-1)*(2*ndbc+ncx_p);
      p_3 = min(ncx_tt,ndbc-1) + max(min(ncx_tt-nc_x2p+1+ncx_p,ndbc+ncx_p),0);
      p_3 = p_3*min(max(ncy_tt-ndbc+1,0),1);
      p_4 = (ncy_tt/nc_y2p)*max(ncy_tt-nc_y2p,0)*(ncx);
      p_5 = (ncy_tt/nc_y2p)*(2*ndbc+ncx_p);
      p_6 = (ncy_tt/nc_y2p)*min(ncx_tt/ndbc,1)*(ncx_tt-ndbc+1);
      p_7 = -(ncy_tt/nc_y2p)*(ncx_tt/(nc_x2p-ncx_p))*(ncx_tt-nc_x2p+1+ncx_p);
      cell_id = max(min(p_1+p_2+p_3+p_4+p_5+p_6+p_7,nc_tot-1),0);
      *theta = rho_ijk*(th_ijk+cellpert_amp_d*(rand_1darray[cell_id]-0.5));
    }
  }else{ // 4-boundaries version (all lateral boundaries)
    if (ii >= ngpb && ii < Nx_tot-ngpb && jj >= ngpb && jj < Ny_tot-ngpb){ // DO NOTHING (k limits checked on function call)
    }else{
      ncx_tt = ii/gppc;
      ncy_tt = jj/gppc;
      p_1 = nc_xy*kk + min(ncy_tt,ndbc)*ncx + ncx_tt*(1-min(ncy_tt/ndbc,1));
      p_2 = min(max(ncy_tt-ndbc,0),nc_y2p-ndbc-1)*(2*ndbc+ncx_p);
      p_3 = min(ncx_tt,ndbc-1) + max(min(ncx_tt-nc_x2p+1+ncx_p,ndbc+ncx_p),0);
      p_3 = p_3*min(max(ncy_tt-ndbc+1,0),1);
      p_4 = (ncy_tt/nc_y2p)*max(ncy_tt-nc_y2p,0)*(ncx);
      p_5 = (ncy_tt/nc_y2p)*(2*ndbc+ncx_p);
      p_6 = (ncy_tt/nc_y2p)*min(ncx_tt/ndbc,1)*(ncx_tt-ndbc+1);
      p_7 = -(ncy_tt/nc_y2p)*(ncx_tt/(nc_x2p-ncx_p))*(ncx_tt-nc_x2p+1+ncx_p);
      cell_id = max(min(p_1+p_2+p_3+p_4+p_5+p_6+p_7,nc_tot-1),0);
      *theta = rho_ijk*(th_ijk+cellpert_amp_d*(rand_1darray[cell_id]-0.5));
    }
  }

} //end cudaDevice_CellPerturbation
__device__ void cudaDevice_CellPerturbationMasked(int i_ind, int j_ind, int k_ind, int Nx, int Ny, int Nz, int Nh, int my_mpi, int numpx, int numpy, float* rho, float* theta, float* rand_1darray,float* bdg_mask){
  float th_ijk,rho_ijk;
  int ncx,ncy,ncz,cell_id,nc_tot,nc_xy,ngpb,ncx_tt,ncy_tt,nc_x2p,nc_y2p;
  int ii,jj,kk;
  int gppc,ndbc,ncx_p,ncy_p;
  int p_1,p_2,p_3,p_4,p_5,p_6,p_7;
  int xoffset,yoffset,Nx_tot,Ny_tot;
  // Nx, Ny domain dimensions passed in are single GPU dimensions

  xoffset = (my_mpi%numpx)*Nx;
  yoffset = (my_mpi%numpy)*Ny;
  Nx_tot = numpx*Nx; // total domain extent in the x-direction
  Ny_tot = numpy*Ny; // total domain extent in the y-direction

  th_ijk = *theta/ *rho;
  rho_ijk = *rho;
  ii = i_ind-Nh+xoffset; // remove halo index, starts from 0 + makes it global
  jj = j_ind-Nh+yoffset; // remove halo index, starts from 0 + makes it global
  kk = k_ind - (cellpert_kbottom_d+Nh-1); // start from 0 at the first perturbation level

  gppc = cellpert_gppc_d;
  ndbc = cellpert_ndbc_d;
  ncx_p = min(Nx_tot%gppc,1);
  ncx = (Nx_tot/gppc)+ ncx_p;
  ncy_p = min(Ny_tot%gppc,1);
  ncy = (Ny_tot/gppc);
  ncz = cellpert_ktop_d-cellpert_kbottom_d+1;

// only domain boundary ring
  nc_xy = ncx*(2*ndbc+ncy_p) + (ncy-2*ndbc)*(2*ndbc+ncx_p);
  nc_tot = nc_xy*ncz; // total number of cells in the domain (random number array size)
  ngpb = gppc*ndbc;
  nc_x2p = ncx - ndbc;
  nc_y2p = ncy - ndbc;

  if ( cellpert_sw2b_d == 1){ // 2-boundary version (south and west) for idealized cases
    if (ii >= ngpb && jj >= ngpb){ // DO NOTHING (k limits checked on function call)
    }else{
      ncx_tt = ii/gppc;
      ncy_tt = jj/gppc;
      p_1 = nc_xy*kk + min(ncy_tt,ndbc)*ncx + ncx_tt*(1-min(ncy_tt/ndbc,1));
      p_2 = min(max(ncy_tt-ndbc,0),nc_y2p-ndbc-1)*(2*ndbc+ncx_p);
      p_3 = min(ncx_tt,ndbc-1) + max(min(ncx_tt-nc_x2p+1+ncx_p,ndbc+ncx_p),0);
      p_3 = p_3*min(max(ncy_tt-ndbc+1,0),1);
      p_4 = (ncy_tt/nc_y2p)*max(ncy_tt-nc_y2p,0)*(ncx);
      p_5 = (ncy_tt/nc_y2p)*(2*ndbc+ncx_p);
      p_6 = (ncy_tt/nc_y2p)*min(ncx_tt/ndbc,1)*(ncx_tt-ndbc+1);
      p_7 = -(ncy_tt/nc_y2p)*(ncx_tt/(nc_x2p-ncx_p))*(ncx_tt-nc_x2p+1+ncx_p);
      cell_id = max(min(p_1+p_2+p_3+p_4+p_5+p_6+p_7,nc_tot-1),0);
      *theta = rho_ijk*(th_ijk+(1.0-*bdg_mask)*cellpert_amp_d*(rand_1darray[cell_id]-0.5));
    }
  }else if( cellpert_sw2b_d == 2){ // 1-boundary version (south) for idealized cases
    if (jj >= ngpb){ // DO NOTHING (k limits checked on function call)
    }else{
      ncx_tt = ii/gppc;
      ncy_tt = jj/gppc;
      p_1 = nc_xy*kk + min(ncy_tt,ndbc)*ncx + ncx_tt*(1-min(ncy_tt/ndbc,1));
      cell_id = max(min(p_1,nc_tot-1),0);
      *theta = rho_ijk*(th_ijk+(1.0-*bdg_mask)*cellpert_amp_d*(rand_1darray[cell_id]-0.5));
    }
  }else if( cellpert_sw2b_d == 3){ // 1-boundary version (west) for idealized cases
    if (ii >= ngpb){ // DO NOTHING (k limits checked on function call)
    }else{
      ncx_tt = ii/gppc;
      ncy_tt = jj/gppc;
      p_1 = nc_xy*kk + min(ncy_tt,ndbc)*ncx + ncx_tt*(1-min(ncy_tt/ndbc,1));
      p_2 = min(max(ncy_tt-ndbc,0),nc_y2p-ndbc-1)*(2*ndbc+ncx_p);
      p_3 = min(ncx_tt,ndbc-1) + max(min(ncx_tt-nc_x2p+1+ncx_p,ndbc+ncx_p),0);
      p_3 = p_3*min(max(ncy_tt-ndbc+1,0),1);
      p_4 = (ncy_tt/nc_y2p)*max(ncy_tt-nc_y2p,0)*(ncx);
      p_5 = (ncy_tt/nc_y2p)*(2*ndbc+ncx_p);
      p_6 = (ncy_tt/nc_y2p)*min(ncx_tt/ndbc,1)*(ncx_tt-ndbc+1);
      p_7 = -(ncy_tt/nc_y2p)*(ncx_tt/(nc_x2p-ncx_p))*(ncx_tt-nc_x2p+1+ncx_p);
      cell_id = max(min(p_1+p_2+p_3+p_4+p_5+p_6+p_7,nc_tot-1),0);
      *theta = rho_ijk*(th_ijk+(1.0-*bdg_mask)*cellpert_amp_d*(rand_1darray[cell_id]-0.5));
    }
  }else{ // 4-boundaries version (all lateral boundaries)
    if (ii >= ngpb && ii < Nx_tot-ngpb && jj >= ngpb && jj < Ny_tot-ngpb){ // DO NOTHING (k limits checked on function call)
    }else{
      ncx_tt = ii/gppc;
      ncy_tt = jj/gppc;
      p_1 = nc_xy*kk + min(ncy_tt,ndbc)*ncx + ncx_tt*(1-min(ncy_tt/ndbc,1));
      p_2 = min(max(ncy_tt-ndbc,0),nc_y2p-ndbc-1)*(2*ndbc+ncx_p);
      p_3 = min(ncx_tt,ndbc-1) + max(min(ncx_tt-nc_x2p+1+ncx_p,ndbc+ncx_p),0);
      p_3 = p_3*min(max(ncy_tt-ndbc+1,0),1);
      p_4 = (ncy_tt/nc_y2p)*max(ncy_tt-nc_y2p,0)*(ncx);
      p_5 = (ncy_tt/nc_y2p)*(2*ndbc+ncx_p);
      p_6 = (ncy_tt/nc_y2p)*min(ncx_tt/ndbc,1)*(ncx_tt-ndbc+1);
      p_7 = -(ncy_tt/nc_y2p)*(ncx_tt/(nc_x2p-ncx_p))*(ncx_tt-nc_x2p+1+ncx_p);
      cell_id = max(min(p_1+p_2+p_3+p_4+p_5+p_6+p_7,nc_tot-1),0);
      *theta = rho_ijk*(th_ijk+(1.0-*bdg_mask)*cellpert_amp_d*(rand_1darray[cell_id]-0.5));
    }
  }

} //end cudaDevice_CellPerturbationMasked
