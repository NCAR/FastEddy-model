/* FastEddy®: SRC/HYDRO_CORE/CUDA/cuda_rayleighDampingDevice.cu 
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
/*---RAYLEIGH DAMPING LAYER*/
__constant__ int dampingLayerSelector_d;       // Rayleigh Damping Layer selector
__constant__ float dampingLayerDepth_d;       // Rayleigh Damping Layer Depth

/*#################------------ RAYLEIGHDAMPING submodule function definitions ------------------#############*/
/*----->>>>> int cuda_rayleighDampingDeviceSetup();       ---------------------------------------------------------
 * Used to cudaMalloc and cudaMemcpy parameters and coordinate arrays, and for the RAYLEIGHDAMPING_CUDA submodule.
*/
extern "C" int cuda_rayleighDampingDeviceSetup(){
   int errorCode = CUDA_RAYLEIGHDAMPING_SUCCESS;

   cudaMemcpyToSymbol(dampingLayerSelector_d, &dampingLayerSelector, sizeof(int));
   cudaMemcpyToSymbol(dampingLayerDepth_d, &dampingLayerDepth, sizeof(float));
   
   return(errorCode);
} //end cuda_rayleighDampingDeviceSetup()

/*----->>>>> extern "C" int cuda_rayleighDampingDeviceCleanup();  --------------------------------------------------
Used to free all malloced memory by the RAYLEIGHDAMPING submodule.
*/

extern "C" int cuda_rayleighDampingDeviceCleanup(){
   int errorCode = CUDA_RAYLEIGHDAMPING_SUCCESS;

   /* Free any RAYLEIGHDAMPING submodule arrays */

   return(errorCode);

}//end cuda_rayleighDampingDeviceCleanup()

/*----->>>>> __device__ void cudaDevice_topRayleighDampingLayerForcing();  ------------------------------------------
* Rayleigh damping layer forcing term 
*/
__device__ void cudaDevice_topRayleighDampingLayerForcing(float* scalarField, float* scalarFrhs,
                                                          float* rho, float* rhoBS, float* zPos_d){

  int i,j,k;
  int ijk,ijkTop;
  int iStride,jStride,kStride;
  float pi_o_2;
  float wBSval;

  pi_o_2 = 0.5*acos(-1.0);
  /*Establish necessary indices for spatial locality*/
  i = (blockIdx.x)*blockDim.x + threadIdx.x;
  j = (blockIdx.y)*blockDim.y + threadIdx.y;
  k = (blockIdx.z)*blockDim.z + threadIdx.z;
  iStride = (Ny_d+2*Nh_d)*(Nz_d+2*Nh_d);
  jStride = (Nz_d+2*Nh_d);
  kStride = 1;
  ijk = i*iStride + j*jStride + k*kStride;
  ijkTop = i*iStride + j*jStride + (kMax_d-1)*kStride;
  if((i >= iMin_d-Nh_d)&&(i < iMax_d+Nh_d) &&
     (j >= jMin_d-Nh_d)&&(j < jMax_d+Nh_d) ){
     if(zPos_d[ijk] >= (zPos_d[ijkTop]-dampingLayerDepth_d)){
        cudaDevice_MomentumBS(W_INDX,zPos_d[ijk],&rhoBS[RHO_INDX_BS+ijk],&wBSval);
        scalarFrhs[ijk] = scalarFrhs[ijk]
                         -0.2*rho[ijk]*( pow(sinf(pi_o_2
                                    *(1.0-(zPos_d[ijkTop]-zPos_d[ijk])/dampingLayerDepth_d)) ,2) )
                             *(scalarField[ijk]/rho[ijk]-wBSval/rhoBS[RHO_INDX_BS+ijk]);
     }//endif zPos >= (ztop-z_d)
  }//end if k>=kMin_dh

} // end cudaDevice_topRayleighDampingLayerForcing

