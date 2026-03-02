/* FastEddy®: SRC/EXTENSIONS/URBAN/CUDA/cuda_urbanDevice.cu
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
/*---URBAN*/
/*Parameters*/
__constant__ int urbanSelector_d;          /* urban selector: 0=off, 1=on */
__constant__ float cd_build_d;             /* c_d coefficient (m-1) used by the drag-based building formulation: -c_d|u|u_i */
__constant__ float ct_build_d;             /* c_t coefficient (s-1) used by the drag-based building formulation: -c_t(rho*theta-rho_b*theta_b) & -c_t(rho-rho_b) */
__constant__ float delta_aware_bdg_d;      /* scale-aware correction for building forcing and limiters */
float* building_mask_d;                    /* building mask field: 0 (atmosphere) or 1 (building) */
__constant__ int urban_heatRedis_d;        /* selector to activate surface heat redistribution */
float *urban_heat_redis_d;                 /* Base Address of memory containing 2d map of heat redistribution coefficient in urban areas */

/*#################------------ URBAN submodule function definitions ------------------#############*/
/*----->>>>> int cuda_urbanDeviceSetup();       ---------------------------------------------------------
 * Used to cudaMalloc and cudaMemcpy parameters and coordinate arrays, and for the URBAN_CUDA submodule.
*/
extern "C" int cuda_urbanDeviceSetup(){
   int errorCode = CUDA_URBAN_SUCCESS;
   size_t Nelems;

   cudaMemcpyToSymbol(urbanSelector_d, &urbanSelector, sizeof(int));
   cudaMemcpyToSymbol(cd_build_d, &cd_build, sizeof(float));
   cudaMemcpyToSymbol(ct_build_d, &ct_build, sizeof(float));

   Nelems = (size_t)((Nxp+2*Nh)*(Nyp+2*Nh)*(Nzp+2*Nh));
   fecuda_DeviceMalloc(Nelems, &building_mask_d);
   cudaMemcpy(building_mask_d, building_mask, Nelems*sizeof(float), cudaMemcpyHostToDevice);

   cudaMemcpyToSymbol(delta_aware_bdg_d, &delta_aware_bdg, sizeof(float));

   if(urban_heatRedis > 0){
     Nelems = (Nxp+2*Nh)*(Nyp+2*Nh);
     fecuda_DeviceMalloc(Nelems, &urban_heat_redis_d);
     cudaMemcpy(urban_heat_redis_d, urban_heat_redis, Nelems*sizeof(float), cudaMemcpyHostToDevice);
   }

   return(errorCode);
} //end cuda_urbanDeviceSetup()

/*----->>>>> extern "C" int cuda_urbanDeviceCleanup();  -----------------------------------------------------------
Used to free all malloced memory by the URBAN submodule.
*/
extern "C" int cuda_urbanDeviceCleanup(){
   int errorCode = CUDA_URBAN_SUCCESS;

   /* Free any URBAN submodule arrays */
   cudaFree(building_mask_d);
   if(urban_heatRedis > 0){
     cudaFree(urban_heat_redis_d);
   }

   return(errorCode);

}//end cuda_urbanDeviceCleanup()

__global__ void cudaDevice_URBANinter(float* z0m, float* z0t, float* hydroTauFlds, float* moistTauFlds, 
		                      float* fricVel, float* htFlux, float* qFlux, float* invOblen, 
				      float* bdg_mask, float* sea_mask, float* urban_redis){

   int i,j,k,ijk,ij;
   int fldStride;
   int iStride,jStride,kStride;
   int iStride2d,jStride2d;

   /*Establish necessary indices for spatial locality*/
   i = (blockIdx.x)*blockDim.x + threadIdx.x;
   j = (blockIdx.y)*blockDim.y + threadIdx.y;
   k = (blockIdx.z)*blockDim.z + threadIdx.z;

   fldStride = (Nx_d+2*Nh_d)*(Ny_d+2*Nh_d)*(Nz_d+2*Nh_d);
   iStride = (Ny_d+2*Nh_d)*(Nz_d+2*Nh_d);
   jStride = (Nz_d+2*Nh_d);
   kStride = 1;
   iStride2d = (Ny_d+2*Nh_d);
   jStride2d = 1;

   if((i >= iMin_d)&&(i < iMax_d) &&
      (j >= jMin_d)&&(j < jMax_d) &&
      (k == kMin_d)){
      ijk = i*iStride + j*jStride + k*kStride;
      ij = i*iStride2d + j*jStride2d; // 2-dimensional (horizontal index)

      //Standard dynamic z0t or redistribution adjustment z0t as appropriate
      if ( (surflayer_z0tdyn_d>0) && ((surflayer_offshore_d==0) || ((surflayer_offshore_d==1) && (sea_mask[ij]<1e-4))) ){ // dynamic z0t calculation
        if (urban_heatRedis_d > 0){ //Redistribution-aware dynamic z0t (Only compute in places where redistributed heat flux will NOT occur)
           if (urban_redis[ij] <= (1.0+1e-5)){
              cudaDevice_z0tdyn(&z0m[ij], &z0t[ij], &fricVel[ij]);
           }
        }else{ //standard dynamic z0t
              cudaDevice_z0tdyn(&z0m[ij], &z0t[ij], &fricVel[ij]);
        } //if-else urban_heatRedis_d > 0
      }

      //Intermediate-timestep stage calculations
      if (bdg_mask[ijk] > 0.0){
        hydroTauFlds[2*fldStride+ijk] = 0.0;
        hydroTauFlds[3*fldStride+ijk] = 0.0;
        hydroTauFlds[8*fldStride+ijk] = 0.0;
        fricVel[ij] = 0.0;
        htFlux[ij] = 0.0;
        invOblen[ij] = 0.0;
        if (moistureSelector_d > 0){
          moistTauFlds[2*fldStride+ijk] = 0.0;
	  qFlux[ij] = 0.0;
        }
      }

      if (urban_heatRedis_d > 0){ //Redistribution-aware dynamic z0t (Only compute in places where redistributed heat flux will NOT occur)
         if (urban_redis[ij] > (1.0+1e-5)){ // urban heat redistribution
            htFlux[ij] = urban_redis[ij]*htFlux[ij];
    	    if (moistureSelector_d > 0){
               qFlux[ij]  = urban_redis[ij]*qFlux[ij];
	    }
         }
      } //if urban_heatRedis_d > 0
   }//end if in the range of non-halo cells

} // end cudaDevice_URBANinter()

__global__ void cudaDevice_URBANfinal(float* hydroFlds_d, float* hydroFldsFrhs_d, float* hydroBaseStateFlds_d, 
		                      float* hydroAuxScalars_d, float* hydroAuxScalarsFrhs_d,
				      float* hydroFldsFrhsMoist_d,
				      float* building_mask_d){

   int i,j,k,ijk;
   int iFld;
   int iFldMoist;
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
      ijk = i*iStride + j*jStride + k*kStride;

      cudaDevice_UrbanDragMethod(&hydroFlds_d[fldStride*RHO_INDX+ijk],&hydroFlds_d[fldStride*U_INDX+ijk],&hydroFlds_d[fldStride*V_INDX+ijk],&hydroFlds_d[fldStride*W_INDX+ijk],
                                 &hydroFlds_d[fldStride*THETA_INDX+ijk],&hydroBaseStateFlds_d[fldStride*THETA_INDX+ijk],&hydroBaseStateFlds_d[fldStride*RHO_INDX+ijk],
                                 &hydroFldsFrhs_d[fldStride*U_INDX+ijk],&hydroFldsFrhs_d[fldStride*V_INDX+ijk],&hydroFldsFrhs_d[fldStride*W_INDX+ijk],
                                 &hydroFldsFrhs_d[fldStride*THETA_INDX+ijk],&hydroFldsFrhs_d[fldStride*RHO_INDX+ijk],&building_mask_d[ijk]);
      if(NhydroAuxScalars_d > 0){
        for(iFld=0; iFld < NhydroAuxScalars_d; iFld++){
          cudaDevice_UrbanDragMethodAuxScalar(&hydroAuxScalars_d[fldStride*iFld+ijk], &hydroAuxScalarsFrhs_d[fldStride*iFld+ijk], &building_mask_d[ijk]);
        }
      }// end if NhydroAuxScalars_d > 0

      if(moistureSelector_d > 0){
        for(iFldMoist=0; iFldMoist < moistureNvars_d; iFldMoist++){
	   cudaDevice_UrbanDragMethodMoist(&hydroFldsFrhsMoist_d[fldStride*iFldMoist+ijk],&building_mask_d[ijk]);
        }
      }// end if moistureSelector_d > 0
   }//end if in the range of non-halo cells

} // end cudaDevice_URBANfinal()

/*----->>>>> __device__ void  cudaDevice_UrbanDragMethod();  --------------------------------------------------
*/
__device__ void cudaDevice_UrbanDragMethod(float* rho, float* u, float* v, float* w, float* th, float* th_base, float* rho_base, float* Frhs_u, float* Frhs_v, float* Frhs_w, float* Frhs_th, float* Frhs_rho, float* bdg_mask){
  float u_ijk,v_ijk,w_ijk;
  float fBuild_u,fBuild_v,fBuild_w,fBuild_th;
  float fBuild_rho;
  float fBuild_ulim_fact = 1.25;
  float fBuild_ulim_u; // limiter for u-velocity drag forcing from buildings
  float fBuild_ulim_v; // limiter for v-velocity drag forcing from buildings
  float fBuild_ulim_w; // limiter for w-velcoity drag forcing from buildings
  float fBuild_ulim_th; // limiter for temperature damping forcing from buildings
  float fBuild_ulim_rho; // limiter for density damping forcing from buildings

  u_ijk = *u/ *rho;
  v_ijk = *v/ *rho;
  w_ijk = *w/ *rho;

  fBuild_u  = cd_build_d*delta_aware_bdg_d*fabsf(u_ijk)*(*u)*(*bdg_mask);
  fBuild_v  = cd_build_d*delta_aware_bdg_d*fabsf(v_ijk)*(*v)*(*bdg_mask);
  fBuild_w  = cd_build_d*delta_aware_bdg_d*fabsf(w_ijk)*(*w)*(*bdg_mask);
  fBuild_th = ct_build_d*delta_aware_bdg_d*((*th)-(*th_base))*(*bdg_mask);
  fBuild_rho = ct_build_d*delta_aware_bdg_d*((*rho)-(*rho_base))*(*bdg_mask);

  fBuild_ulim_u = fabsf((*Frhs_u)*fBuild_ulim_fact);
  fBuild_ulim_v = fabsf((*Frhs_v)*fBuild_ulim_fact);
  fBuild_ulim_w = fabsf((*Frhs_w)*fBuild_ulim_fact);
  fBuild_ulim_th = fabsf((*Frhs_th)*fBuild_ulim_fact);
  fBuild_ulim_rho = fabsf((*Frhs_rho)*fBuild_ulim_fact);

  *Frhs_u  = *Frhs_u - copysign(1.0,fBuild_u)*fminf(fabsf(fBuild_u),fBuild_ulim_u);
  *Frhs_v  = *Frhs_v - copysign(1.0,fBuild_v)*fminf(fabsf(fBuild_v),fBuild_ulim_v);
  *Frhs_w  = *Frhs_w - copysign(1.0,fBuild_w)*fminf(fabsf(fBuild_w),fBuild_ulim_w);
  if(urbanSelector_d==1){
    *Frhs_th = *Frhs_th*(1.0-(*bdg_mask)) ;
    *Frhs_rho = *Frhs_rho*(1.0-(*bdg_mask));
  }else if(urbanSelector_d==2){
    *Frhs_th = *Frhs_th - copysign(1.0,fBuild_th)*fminf(fabsf(fBuild_th),fBuild_ulim_th);
    *Frhs_rho = *Frhs_rho - copysign(1.0,fBuild_rho)*fminf(fabsf(fBuild_rho),fBuild_ulim_rho);
  }//end if 

} //end cudaDevice_UrbanDragMethod

__device__ void cudaDevice_UrbanDragMethodMoist(float * Frhs_qMoistFld, float* bdg_mask){

  *Frhs_qMoistFld = *Frhs_qMoistFld*(1.0-(*bdg_mask));

} //end cudaDevice_UrbanDragMethodMoist

__device__ void cudaDevice_UrbanDragMethodAuxScalar(float* AuxScalar, float* Frhs_AuxScalar, float* bdg_mask){
  float fBuild_AuxSc;
  float fBuild_ulim_fact = 1.25;
  float fBuild_ulim_AuxSc; // limiter for damping forcing inside building mask


  fBuild_AuxSc  = cd_build_d*delta_aware_bdg_d*fabsf(*AuxScalar)*(*bdg_mask);
  fBuild_ulim_AuxSc = fabsf((*Frhs_AuxScalar)*fBuild_ulim_fact);

  if(urbanSelector_d==1){
    *Frhs_AuxScalar = *Frhs_AuxScalar*(1.0-(*bdg_mask));
  }else if(urbanSelector_d==2){
    *Frhs_AuxScalar = *Frhs_AuxScalar - copysign(1.0,fBuild_AuxSc)*fminf(fabsf(fBuild_AuxSc),fBuild_ulim_AuxSc);
  }//end if 

} //end cudaDevice_UrbanDragMethodAuxScalar
