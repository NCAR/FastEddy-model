/* FastEddy®: SRC/TIME_INTEGRATION/CUDA/cuda_RKSchemes.cu 
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

/*----->>>>> __global__ void  cudaDevice_timeIntegrationCommenceRK3_WS2002();  ------------------------------------
* This is the gloabl-entry kernel routine used by the TIME_INTEGRATION module for the
* Runge-Kutta 3 Wicker and Skamarock (2002) MWR paper formulation.
*/
__global__ void cudaDevice_timeIntegrationCommenceRK3_WS2002(int Nphi, float* phi_Flds, float* phi_Frhs,
                                                             int Nsgstke, float* sgstkeSc_Flds, float* sgstkeSc_Frhs,
                                                             int Nmoist, float* moistSc_Flds, float* moistSc_Frhs,
                                                             float* timeFlds0, int RKstage){
   int i,j,k;
   int ijk;
   int iStride,jStride,kStride;
   int iFld,fldStride;
   float* currFld;
   float* currFrhs;
   float* currFld0;

   /*Establish necessary indices for spatial locality*/
   i = (blockIdx.x)*blockDim.x + threadIdx.x;
   j = (blockIdx.y)*blockDim.y + threadIdx.y;
   k = (blockIdx.z)*blockDim.z + threadIdx.z;
   iStride = (Ny_d+2*Nh_d)*(Nz_d+2*Nh_d);
   jStride = (Nz_d+2*Nh_d);
   kStride = 1;
   fldStride = (Nx_d+2*Nh_d)*(Ny_d+2*Nh_d)*(Nz_d+2*Nh_d);
   /*if this thread is in the range of non-halo cells*/
   if((i >= iMin_d)&&(i < iMax_d) &&
      (j >= jMin_d)&&(j < jMax_d) &&
      (k >= kMin_d)&&(k < kMax_d) ){

      /* Begin the loop over core phi variables */

      for(iFld=0; iFld < Nphi; iFld++){
         currFld = &phi_Flds[iFld*fldStride];
         currFrhs = &phi_Frhs[iFld*fldStride];
         currFld0 = &timeFlds0[iFld*fldStride];
         ijk =     i  *iStride +   j  *jStride +   k  *kStride;
         switch(RKstage){    
          case 0:
            cudaDevice_RungeKutta3WS02Stage1(&currFld[ijk], &currFrhs[ijk], &currFld0[ijk]);
            break;
          case 1:
            cudaDevice_RungeKutta3WS02Stage2(&currFld[ijk], &currFrhs[ijk], &currFld0[ijk]);
            break;
          case 2:
            cudaDevice_RungeKutta3WS02Stage3(&currFld[ijk], &currFrhs[ijk], &currFld0[ijk]);
            break;
         }//end switch RKstage
      } //end for iFld

      for(iFld=0; iFld < Nsgstke; iFld++){ // time integration of SGSTKE equations
         currFld = &sgstkeSc_Flds[iFld*fldStride];
         currFrhs = &sgstkeSc_Frhs[iFld*fldStride];
         currFld0 = &timeFlds0[(Nphi+iFld)*fldStride];
         ijk =     i  *iStride +   j  *jStride +   k  *kStride;
         switch(RKstage){    
          case 0:
            cudaDevice_RungeKutta3WS02Stage1(&currFld[ijk], &currFrhs[ijk], &currFld0[ijk]);
            break;
          case 1:
            cudaDevice_RungeKutta3WS02Stage2(&currFld[ijk], &currFrhs[ijk], &currFld0[ijk]);
            break;
          case 2:
            cudaDevice_RungeKutta3WS02Stage3(&currFld[ijk], &currFrhs[ijk], &currFld0[ijk]);
            break;
         }//end switch RKstage
         cudaDevice_PositiveDef(&sgstkeSc_Flds[iFld*fldStride],1e-10); // enforce positive definite SGSTKE
      } // end iFld < Nsgstke

      for(iFld=0; iFld < Nmoist; iFld++){ // time integration of moisture equations
         currFld = &moistSc_Flds[iFld*fldStride];
         currFrhs = &moistSc_Frhs[iFld*fldStride];
         currFld0 = &timeFlds0[(Nphi+Nsgstke+iFld)*fldStride];
         ijk =     i  *iStride +   j  *jStride +   k  *kStride;
         switch(RKstage){    
          case 0:
            cudaDevice_RungeKutta3WS02Stage1(&currFld[ijk], &currFrhs[ijk], &currFld0[ijk]);
            break;
          case 1:
            cudaDevice_RungeKutta3WS02Stage2(&currFld[ijk], &currFrhs[ijk], &currFld0[ijk]);
            break;
          case 2:
            cudaDevice_RungeKutta3WS02Stage3(&currFld[ijk], &currFrhs[ijk], &currFld0[ijk]);
            break;
         }//end switch RKstage
         cudaDevice_PositiveDef(&moistSc_Flds[iFld*fldStride],0.0); // enforce positive definite moisture fields
      } // end iFld < Nsgstke

    /* End the timestepping loop */

   }//end if in the range of non-halo cells

} // end cudaDevice_timeIntegrationCommenceRK3_WS2002()

/*----->>>>> __device__ void  cudaDevice_RungeKutta3WS02Stage1();  --------------------------------------------------
* This is the device function to perform stage 1 of 3 from the  Runge-Kutta-3 WS02 time_integration scheme 
*/
__device__ void cudaDevice_RungeKutta3WS02Stage1(float* currFld, float* currFrhs, float* currFld0){

  *currFld0 = *currFld;  // Save the initial stage values ifor reuse in later stages
  *currFld = *currFld + (1.0/3.0)*dt_d*(*currFrhs); // Set the next value from which to calculate build_Frhs

} // end cudaDevice_RungeKutta3WS02Stage1()

/*----->>>>> __device__ void  cudaDevice_RungeKutta3WS02Stage2();  --------------------------------------------------
* This is the device function to perform stage 2 of 3 from the Runge-Kutta-3 WS02 time_integration scheme 
*/
__device__ void cudaDevice_RungeKutta3WS02Stage2(float* currFld, float* currFrhs, float* currFld0){

  *currFld = *currFld0 + 0.5*dt_d*(*currFrhs); // Set the phi** calculate build_Frhs(phi**)

} // end cudaDevice_RungeKutta3WS02Stage2()

/*----->>>>> __device__ void  cudaDevice_RungeKutta3WS02Stage3();  --------------------------------------------------
* This is the device function to perform stage 3 of 3 from the Runge-Kutta-3 WS02 time_integration scheme 
*/
__device__ void cudaDevice_RungeKutta3WS02Stage3(float* currFld, float* currFrhs, float* currFld0){

  *currFld = *currFld0 + dt_d*(*currFrhs); // Set the final RK3_WS2002 predicted value

} // end cudaDevice_RungeKutta3WS02Stage3()

/*----->>>>> __device__ void  cudaDevice_PositiveDef();  --------------------------------------------------
*/ // Def
__device__ void cudaDevice_PositiveDef(float* Fld, float min_threshold){

  int i,j,k,ijk;
  int iStride,jStride,kStride;

  i = (blockIdx.x)*blockDim.x + threadIdx.x;
  j = (blockIdx.y)*blockDim.y + threadIdx.y;
  k = (blockIdx.z)*blockDim.z + threadIdx.z;
  iStride = (Ny_d+2*Nh_d)*(Nz_d+2*Nh_d);
  jStride = (Nz_d+2*Nh_d);
  kStride = 1;
  ijk = i*iStride + j*jStride + k*kStride;

  if((i >= iMin_d-Nh_d)&&(i < iMax_d+Nh_d) && (j >= jMin_d-Nh_d)&&(j < jMax_d+Nh_d) && (k >= kMin_d-Nh_d)&&(k < kMax_d+Nh_d)){
    Fld[ijk] = fmaxf(Fld[ijk],min_threshold);
  }

} //end cudaDevice_PositiveDef()
