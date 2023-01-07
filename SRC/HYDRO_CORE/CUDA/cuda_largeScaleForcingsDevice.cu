/* FastEddy®: SRC/HYDRO_CORE/CUDA/cuda_largeScaleForcingsDevice.cu 
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
/*---LARGE SCALE FORCINGS*/ 
__constant__ int lsfSelector_d;         /* large-scale forcings selector: 0=off, 1=on */
__constant__ float lsf_w_surf_d;        /* lsf to w at the surface */
__constant__ float lsf_w_lev1_d;        /* lsf to w at the first specified level */
__constant__ float lsf_w_lev2_d;        /* lsf to w at the second specified level */
__constant__ float lsf_w_zlev1_d;       /* lsf to w height 1 */
__constant__ float lsf_w_zlev2_d;       /* lsf to w height 2 */
__constant__ float lsf_th_surf_d;       /* lsf to theta at the surface */
__constant__ float lsf_th_lev1_d;       /* lsf to theta at the first specified level */
__constant__ float lsf_th_lev2_d;       /* lsf to theta at the second specified level */
__constant__ float lsf_th_zlev1_d;      /* lsf to theta height 1 */
__constant__ float lsf_th_zlev2_d;      /* lsf to theta height 2 */
__constant__ float lsf_qv_surf_d;       /* lsf to qv at the surface */
__constant__ float lsf_qv_lev1_d;       /* lsf to qv at the first specified level */
__constant__ float lsf_qv_lev2_d;       /* lsf to qv at the second specified level */
__constant__ float lsf_qv_zlev1_d;      /* lsf to qv height 1 */
__constant__ float lsf_qv_zlev2_d;      /* lsf to qv height 2 */

__constant__ int lsf_horMnSubTerms_d;   /* Switch 0=off, 1=on */
__constant__ int lsf_numPhiVars_d;      /* number of variables in the slabMeanPhiProfiles set (e.g. rho,u,v,theta,qv=5) */
float* lsf_slabMeanPhiProfiles_d;       /*Base address of -w*(d_phi/dz) lsf term horz mean profiles of phi variables*/
float* lsf_meanPhiBlock_d;              /*Base address of work arrray for block reduction Mean */

/*#################------------ LARGE SCALE FORCINGS submodule function definitions ------------------#############*/
/*----->>>>> int cuda_lsfDeviceSetup();       ---------------------------------------------------------
 * Used to cudaMalloc and cudaMemcpy parameters and coordinate arrays, and for the LSF_CUDA submodule.
*/
extern "C" int cuda_lsfDeviceSetup(){
   int errorCode = CUDA_LSF_SUCCESS;
   int Nelems;

   cudaMemcpyToSymbol(lsfSelector_d, &lsfSelector, sizeof(int));
   cudaMemcpyToSymbol(lsf_w_surf_d, &lsf_w_surf, sizeof(float));
   cudaMemcpyToSymbol(lsf_w_lev1_d, &lsf_w_lev1, sizeof(float));
   cudaMemcpyToSymbol(lsf_w_lev2_d, &lsf_w_lev2, sizeof(float));
   cudaMemcpyToSymbol(lsf_w_zlev1_d, &lsf_w_zlev1, sizeof(float));
   cudaMemcpyToSymbol(lsf_w_zlev2_d, &lsf_w_zlev2, sizeof(float));
   cudaMemcpyToSymbol(lsf_th_surf_d, &lsf_th_surf, sizeof(float));
   cudaMemcpyToSymbol(lsf_th_lev1_d, &lsf_th_lev1, sizeof(float));
   cudaMemcpyToSymbol(lsf_th_lev2_d, &lsf_th_lev2, sizeof(float));
   cudaMemcpyToSymbol(lsf_th_zlev1_d, &lsf_th_zlev1, sizeof(float));
   cudaMemcpyToSymbol(lsf_th_zlev2_d, &lsf_th_zlev2, sizeof(float));
   cudaMemcpyToSymbol(lsf_qv_surf_d, &lsf_qv_surf, sizeof(float));
   cudaMemcpyToSymbol(lsf_qv_lev1_d, &lsf_qv_lev1, sizeof(float));
   cudaMemcpyToSymbol(lsf_qv_lev2_d, &lsf_qv_lev2, sizeof(float));
   cudaMemcpyToSymbol(lsf_qv_zlev1_d, &lsf_qv_zlev1, sizeof(float));
   cudaMemcpyToSymbol(lsf_qv_zlev2_d, &lsf_qv_zlev2, sizeof(float));
   cudaMemcpyToSymbol(lsf_horMnSubTerms_d, &lsf_horMnSubTerms, sizeof(int));

   if(lsf_horMnSubTerms==1){
     if(mpi_rank_world==0){
       printf("lsf_horMnSubTerms = %d, setting up for %d phi-fields.",
              lsf_horMnSubTerms,lsf_numPhiVars);
       fflush(stdout);
     }
     cudaMemcpyToSymbol(lsf_numPhiVars_d, &lsf_numPhiVars, sizeof(float));
     Nelems = (Nzp+2*Nh);
     fecuda_DeviceMalloc(Nelems*lsf_numPhiVars*sizeof(float), &lsf_slabMeanPhiProfiles_d);
     fecuda_DeviceMalloc(grid_red.x*grid_red.y*grid_red.z*sizeof(float), &lsf_meanPhiBlock_d);
   }

   return(errorCode);
} //end cuda_lsfDeviceSetup()

/*----->>>>> extern "C" int cuda_lsfDeviceCleanup();  -----------------------------------------------------------
Used to free all malloced memory by the LSF submodule.
*/
extern "C" int cuda_lsfDeviceCleanup(){
   int errorCode = CUDA_LSF_SUCCESS;

   //Free device memory as necesary
   if(lsf_horMnSubTerms==1){
     cudaFree(lsf_slabMeanPhiProfiles_d);
     cudaFree(lsf_meanPhiBlock_d);
   }

   return(errorCode);

}//end cuda_lsfDeviceCleanup()

/*----->>>>> extern "C" int cuda_lsfSlabMeans();  -----------------------------------------------------------
*  Obtain the slab means of rho, u, v, theta, and qv
*/
extern "C" int cuda_lsfSlabMeans(){
   int errorCode = CUDA_LSF_SUCCESS;
   int iFld;
   int iFldProf;
   int fldProfStride;
   int fldStride;

   fldStride = (Nxp+2*Nh)*(Nyp+2*Nh)*(Nzp+2*Nh);
   fldProfStride = (Nzp+2*Nh);
   //Blank out the temporary storage and atmonic result
   cudaMemset(&lsf_slabMeanPhiProfiles_d[0],0.0,lsf_numPhiVars*fldProfStride*sizeof(float));

   for(iFldProf=0; iFldProf<lsf_numPhiVars-1; iFldProf++){
      if(iFldProf==0){ // rho
        iFld = RHO_INDX;
      } else if(iFldProf==1){ // u
        iFld = U_INDX;
      } else if(iFldProf==2){ //v
        iFld = V_INDX;
      } else if(iFldProf==3){ //theta
        iFld = THETA_INDX;
      }
      cuda_singleRankHorizSlabMeans<<<grid_red, tBlock_red>>>(&hydroFlds_d[RHO_INDX*fldStride],
                                                              &hydroFlds_d[iFld*fldStride],lsf_meanPhiBlock_d,
                                                              &lsf_slabMeanPhiProfiles_d[iFldProf*fldProfStride]);
   }//end for iFldProf
   //Blank out the temporary storage and atmonic result
   iFldProf=lsf_numPhiVars-1; //qv
   iFld=0;
   cuda_singleRankHorizSlabMeans<<<grid_red, tBlock_red>>>(&hydroFlds_d[RHO_INDX*fldStride],
                                                           &moistScalars_d[iFld*fldStride],lsf_meanPhiBlock_d,
                                                           &lsf_slabMeanPhiProfiles_d[iFldProf*fldProfStride]);
#ifdef DEBUG
   if(mpi_rank_world==0){
     printf("lsf_horMnSubTerms for phi-fields complete.\n");
     fflush(stdout);
   }
#endif

   return(errorCode);

} // cuda_lsfSlabMeans()

/*----->>>>> __global__ void  cudaDevice_hydroCoreUnitTestComplete();  ----------------------------------------------
* Global Kernel for calculating/accumulating large-scale forcing Frhs terms   
*/
__global__ void cudaDevice_hydroCoreUnitTestCompleteLSF(float temp_freq_fac, float* hydroFlds_d, float* lsf_slabMeanPhiProfiles_d, float* hydroFldsFrhs_d, float* moistScalarsFrhs_d, float* zPos_d){ 

   int fldStride;

   fldStride = (Nx_d+2*Nh_d)*(Ny_d+2*Nh_d)*(Nz_d+2*Nh_d);

   cudaDevice_lsfRHS(temp_freq_fac, &hydroFlds_d[fldStride*RHO_INDX],
                     &lsf_slabMeanPhiProfiles_d[0], &hydroFldsFrhs_d[0], &moistScalarsFrhs_d[0], zPos_d);

} // end cudaDevice_hydroCoreUnitTestCompleteLSF()

__device__ void cudaDevice_lsfRHS(float temp_freq_fac, float* rho,
                                  float* lsf_slabMeanPhiProfiles_d,float* Frhs_HC, float* Frhs_qv, float* zPos_d){

  int i,j,k,ijk,ijkm1,ijkp1;
  int iStride,jStride,kStride;
  int fldStride;
  int fldProfStride;
  float z_ijk;
  float slope_w,slope_th,slope_qv,f_lsf_w,f_lsf_th,f_lsf_qv;

  i = (blockIdx.x)*blockDim.x + threadIdx.x;
  j = (blockIdx.y)*blockDim.y + threadIdx.y;
  k = (blockIdx.z)*blockDim.z + threadIdx.z;
  iStride = (Ny_d+2*Nh_d)*(Nz_d+2*Nh_d);
  jStride = (Nz_d+2*Nh_d);
  kStride = 1;
  fldStride = (Nx_d+2*Nh_d)*(Ny_d+2*Nh_d)*(Nz_d+2*Nh_d);
  fldProfStride = (Nz_d+2*Nh_d);
  ijk = i*iStride + j*jStride + k*kStride;
  ijkm1 = i*iStride + j*jStride + (k-1)*kStride;
  ijkp1 = i*iStride + j*jStride + (k+1)*kStride;

  if((i >= iMin_d)&&(i < iMax_d) && (j >= jMin_d)&&(j < jMax_d) && (k >= kMin_d)&&(k < kMax_d)){

    z_ijk = zPos_d[ijk];

    // theta-forcing (large-scale advection)
    if (z_ijk <= lsf_th_zlev1_d){ // layer 1
      slope_th = (lsf_th_lev1_d - lsf_th_surf_d)/lsf_th_zlev1_d;
      f_lsf_th = (lsf_th_surf_d + slope_th*z_ijk)*rho[ijk];
    } else if((z_ijk>lsf_th_zlev1_d) && (z_ijk<=lsf_th_zlev2_d)){ // layer 2
      slope_th = (lsf_th_lev2_d - lsf_th_lev1_d)/(lsf_th_zlev2_d - lsf_th_zlev1_d);
      f_lsf_th = (lsf_th_lev1_d + slope_th*(z_ijk-lsf_th_zlev1_d))*rho[ijk];
    } else{ // layer 3
      f_lsf_th = lsf_th_lev2_d*rho[ijk];
    }

    if(moistureSelector_d > 0){
      // water vapor qv-forcing (large-scale advection)
      if (z_ijk <= lsf_qv_zlev1_d){ // layer 1
        slope_qv = (lsf_qv_lev1_d - lsf_qv_surf_d)/lsf_qv_zlev1_d;
        f_lsf_qv = (lsf_qv_surf_d + slope_qv*z_ijk)*rho[ijk];
      } else if((z_ijk>lsf_qv_zlev1_d) && (z_ijk<=lsf_qv_zlev2_d)){ // layer 2
        slope_qv = (lsf_qv_lev2_d - lsf_qv_lev1_d)/(lsf_qv_zlev2_d - lsf_qv_zlev1_d);
        f_lsf_qv = (lsf_qv_lev1_d + slope_qv*(z_ijk-lsf_qv_zlev1_d))*rho[ijk];
      } else{ // layer 3
        f_lsf_qv = lsf_qv_lev2_d*rho[ijk];
      }
    }// end if moistureSelector_d > 0

    // input vaules are per hour
    f_lsf_th = f_lsf_th*temp_freq_fac/3600.0;

    Frhs_HC[THETA_INDX*fldStride+ijk] = Frhs_HC[THETA_INDX*fldStride+ijk] + f_lsf_th; // large-scale advection of temperature

    if(moistureSelector_d > 0){
      f_lsf_qv = f_lsf_qv*temp_freq_fac/3600.0;
      Frhs_qv[ijk] = Frhs_qv[ijk] + f_lsf_qv; // large-scale advection of water vapor
    }// end if moistureSelector_d > 0

    /* subsidence terms */
    if(lsf_horMnSubTerms_d == 1){

      // w-forcing (subsidence)
      if (z_ijk <= lsf_w_zlev1_d){ // layer 1
        slope_w = (lsf_w_lev1_d - lsf_w_surf_d)/lsf_w_zlev1_d;
        f_lsf_w = (lsf_w_surf_d + slope_w*z_ijk)*rho[ijk];
      } else if((z_ijk>lsf_w_zlev1_d) && (z_ijk<=lsf_w_zlev2_d)){ // layer 2
        slope_w = (lsf_w_lev2_d - lsf_w_lev1_d)/(lsf_w_zlev2_d - lsf_w_zlev1_d);
        f_lsf_w = (lsf_w_lev1_d + slope_w*(z_ijk-lsf_w_zlev1_d))*rho[ijk];
      } else{ // layer 3
        f_lsf_w = lsf_w_lev2_d*rho[ijk];
      }
      f_lsf_w = f_lsf_w*temp_freq_fac/3600.0;
      /*u*/
      Frhs_HC[U_INDX*fldStride+ijk] = Frhs_HC[U_INDX*fldStride+ijk]
                                      -f_lsf_w*rho[ijk]*(lsf_slabMeanPhiProfiles_d[1*fldProfStride+k+1]/lsf_slabMeanPhiProfiles_d[0*fldProfStride+k+1]
                                               -lsf_slabMeanPhiProfiles_d[1*fldProfStride+k-1]/lsf_slabMeanPhiProfiles_d[0*fldProfStride+k-1])
                                              /(zPos_d[ijkp1]-zPos_d[ijkm1]);

      /*v*/
      Frhs_HC[V_INDX*fldStride+ijk] = Frhs_HC[V_INDX*fldStride+ijk]
                                      -f_lsf_w*rho[ijk]*(lsf_slabMeanPhiProfiles_d[2*fldProfStride+k+1]/lsf_slabMeanPhiProfiles_d[0*fldProfStride+k+1]
                                               -lsf_slabMeanPhiProfiles_d[2*fldProfStride+k-1]/lsf_slabMeanPhiProfiles_d[0*fldProfStride+k-1])
                                              /(zPos_d[ijkp1]-zPos_d[ijkm1]);
      /*theta*/
      Frhs_HC[THETA_INDX*fldStride+ijk] = Frhs_HC[THETA_INDX*fldStride+ijk]
                                      -f_lsf_w*rho[ijk]*(lsf_slabMeanPhiProfiles_d[3*fldProfStride+k+1]/lsf_slabMeanPhiProfiles_d[0*fldProfStride+k+1]
                                               -lsf_slabMeanPhiProfiles_d[3*fldProfStride+k-1]/lsf_slabMeanPhiProfiles_d[0*fldProfStride+k-1])
                                              /(zPos_d[ijkp1]-zPos_d[ijkm1]);
      /*qv*/
      Frhs_qv[ijk] = Frhs_qv[ijk]-f_lsf_w*rho[ijk]*(lsf_slabMeanPhiProfiles_d[4*fldProfStride+k+1]/lsf_slabMeanPhiProfiles_d[0*fldProfStride+k+1]
                                          -lsf_slabMeanPhiProfiles_d[4*fldProfStride+k-1]/lsf_slabMeanPhiProfiles_d[0*fldProfStride+k-1])
                                         /(zPos_d[ijkp1]-zPos_d[ijkm1]);
    }

  }

} //end cudaDevice_lsfRHS
