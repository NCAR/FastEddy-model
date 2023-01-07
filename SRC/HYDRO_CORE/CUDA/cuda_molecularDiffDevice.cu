/* FastEddy®: SRC/HYDRO_CORE/CUDA/cuda_molecularDiffDevice.cu 
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
/*---MOLECULAR DIFFUSION*/ 
/*Parameters*/
__constant__ int diffusionSelector_d;      /* molecular diffusion selector: 0=off, 1=on */
__constant__ float nu_0_d;                 /* constant molecular diffusivity used when diffusionSelector_d == 1 */
float* hydroNuGradXFlds_d;                 /* Base address for diffusion for nu*grad_x */
float* hydroNuGradYFlds_d;                 /* Base address for diffusion for nu*grad_y */
float* hydroNuGradZFlds_d;                 /* Base address for diffusion for nu*grad_z */

/*#################------------ MOLECULAR DIFFUSION submodule function definitions ------------------#############*/
/*----->>>>> int cuda_molecularDiffDeviceSetup();       ---------------------------------------------------------
 * Used to cudaMalloc and cudaMemcpy parameters and coordinate arrays, and for the MOLECULAR DIFFUSION submodule.
*/
extern "C" int cuda_molecularDiffDeviceSetup(){
   int errorCode = CUDA_MOLDIFF_SUCCESS;
   int Nelems;

   cudaMemcpyToSymbol(diffusionSelector_d, &diffusionSelector, sizeof(int));
   cudaMemcpyToSymbol(nu_0_d, &nu_0, sizeof(float));

   Nelems = (Nxp+2*Nh)*(Nyp+2*Nh)*(Nzp+2*Nh);
   fecuda_DeviceMalloc(Nelems*(Nhydro-1)*sizeof(float), &hydroNuGradXFlds_d); // all Nhydro except density 
   fecuda_DeviceMalloc(Nelems*(Nhydro-1)*sizeof(float), &hydroNuGradYFlds_d);  
   fecuda_DeviceMalloc(Nelems*(Nhydro-1)*sizeof(float), &hydroNuGradZFlds_d); 

   return(errorCode);
} //end cuda_molecularDiffDeviceSetup()


/*----->>>>> extern "C" int cuda_molecularDiffDeviceCleanup();  ----------------------------------------------------
Used to free all malloced memory by the MOLECULAR DIFFUSION submodule.
*/
extern "C" int cuda_molecularDiffDeviceCleanup(){
   int errorCode = CUDA_MOLDIFF_SUCCESS;

   /* Free any MOLECULAR DIFFUSION submodule arrays */
   cudaFree(hydroNuGradXFlds_d);
   cudaFree(hydroNuGradYFlds_d);
   cudaFree(hydroNuGradZFlds_d);

   return(errorCode);

}//end cuda_molecularDiffDeviceCleanup()

/*----->>>>> __global__ void  cudaDevice_hydroCoreUnitTestCompleteMolecularDiffusion();  -------------------------
* Global Kernel for calculating/accumulating molecular diffusion Frhs terms   
*/
__global__ void cudaDevice_hydroCoreUnitTestCompleteMolecularDiffusion(float* hydroFlds, float* hydroFldsFrhs,
                                                                       float* hydroNuGradXFlds_d, float* hydroNuGradYFlds_d, float* hydroNuGradZFlds_d,
                                                                       float* J31_d, float* J32_d, float* J33_d,
                                                                       float* D_Jac_d, float* invD_Jac_d){ // calculate divergence of NuGrad
   int i,j,k;
   int iFld,fldStride;

   /*Establish necessary indices for spatial locality*/
   i = (blockIdx.x)*blockDim.x + threadIdx.x;
   j = (blockIdx.y)*blockDim.y + threadIdx.y;
   k = (blockIdx.z)*blockDim.z + threadIdx.z;

   fldStride = (Nx_d+2*Nh_d)*(Ny_d+2*Nh_d)*(Nz_d+2*Nh_d);

   if((i >= iMin_d)&&(i < iMax_d) &&
      (j >= jMin_d)&&(j < jMax_d) &&
      (k >= kMin_d)&&(k < kMax_d) ){
      for(iFld=1; iFld < Nhydro_d; iFld++){ //Note: starts at iFld=1, excluding rho... 
         cudaDevice_calcDivNuGrad(&hydroFldsFrhs[fldStride*iFld],&hydroFlds[fldStride*RHO_INDX],
                                  &hydroNuGradXFlds_d[fldStride*(iFld-1)],
                                  &hydroNuGradYFlds_d[fldStride*(iFld-1)],
                                  &hydroNuGradZFlds_d[fldStride*(iFld-1)],iFld,
                                  J31_d, J32_d, J33_d, D_Jac_d, invD_Jac_d);
      }//for iFld
   }//end if in the range of non-halo cells

} // end cudaDevice_hydroCoreUnitTestCompleteMolecularDiffusion()

/*----->>>>> __device__ void cudaDevice_diffusionDriver();  --------------------------------------------------
* This function drives the element-wise calls to cudaDevice_calcConstNuGrad() for molecular diffusion 
* of an arbitrary field. 
*/
__device__ void cudaDevice_diffusionDriver(float* fld, float* NuGradX, float* NuGradY,float* NuGradZ, float inv_pr,
                                           float* J31_d, float* J32_d, float* J33_d,
                                           float* D_Jac_d){
  
   int i,j,k,ijk;
   int im1jk,ijm1k,ijkm1;
   int ip1jk,ijp1k,ijkp1;
   int iStride,jStride,kStride;
   int im1jm1k,im1jkm1,ijm1km1,im1jp1k,im1jkp1,ijm1kp1,ip1jkm1,ip1jm1k,ijp1km1;
 
   /*Establish necessary indices for spatial locality*/
   i = (blockIdx.x)*blockDim.x + threadIdx.x;
   j = (blockIdx.y)*blockDim.y + threadIdx.y;
   k = (blockIdx.z)*blockDim.z + threadIdx.z;

   iStride = (Ny_d+2*Nh_d)*(Nz_d+2*Nh_d);
   jStride = (Nz_d+2*Nh_d);
   kStride = 1;

   if((i >= iMin_d)&&(i <= iMax_d) &&
      (j >= jMin_d)&&(j <= jMax_d) &&
      (k >= kMin_d)&&(k <= kMax_d) ){

     ijk = i*iStride + j*jStride + k*kStride;
     im1jk = (i-1)*iStride + j*jStride + k*kStride;
     ijm1k = i*iStride + (j-1)*jStride + k*kStride;
     ijkm1 = i*iStride + j*jStride + (k-1)*kStride;
     ip1jk = (i+1)*iStride + j*jStride + k*kStride;
     ijp1k = i*iStride + (j+1)*jStride + k*kStride;
     ijkp1 = i*iStride + j*jStride + (k+1)*kStride;
     im1jm1k = (i-1)*iStride + (j-1)*jStride + k*kStride;
     im1jkm1 = (i-1)*iStride + j*jStride + (k-1)*kStride;
     ijm1km1 = i*iStride + (j-1)*jStride + (k-1)*kStride;
     im1jp1k = (i-1)*iStride + (j+1)*jStride + k*kStride;
     im1jkp1 = (i-1)*iStride + j*jStride + (k+1)*kStride;
     ijm1kp1 = i*iStride + (j-1)*jStride + (k+1)*kStride;
     ip1jkm1 = (i+1)*iStride + j*jStride + (k-1)*kStride;
     ip1jm1k = (i+1)*iStride + (j-1)*jStride + k*kStride;
     ijp1km1 = i*iStride + (j+1)*jStride + (k-1)*kStride;
     cudaDevice_calcConstNuGrad( &NuGradX[ijk], &NuGradY[ijk], &NuGradZ[ijk],
                                 &fld[ijk], &fld[im1jk], &fld[ijm1k], &fld[ijkm1], &fld[ip1jk], &fld[ijp1k], &fld[ijkp1],
                                 &fld[im1jm1k], &fld[im1jkm1], &fld[ijm1km1], &fld[im1jp1k], &fld[im1jkp1],&fld[ijm1kp1], 
                                 &fld[ip1jkm1], &fld[ip1jm1k], &fld[ijp1km1],inv_pr,
                                 J31_d, J32_d, J33_d, D_Jac_d);
  }//end if in the range of non-halo cells
}//end diffusionDriver()

/*----->>>>> __device__ void cudaDevice_calcConstNuGrad();  --------------------------------------------------
* This is the cuda form of formulating the gradient of a field with constant molecular diffusivity
*/
__device__ void cudaDevice_calcConstNuGrad(float* NuGradX, float* NuGradY, float* NuGradZ,
                                           float* sFld_ijk, float* sFld_im1jk, float* sFld_ijm1k, float* sFld_ijkm1,
                                           float* sFld_ip1jk, float* sFld_ijp1k, float* sFld_ijkp1,
                                           float* sFld_im1jm1k, float* sFld_im1jkm1, float* sFld_ijm1km1,
                                           float* sFld_im1jp1k, float* sFld_im1jkp1, float* sFld_ijm1kp1,
                                           float* sFld_ip1jkm1, float* sFld_ip1jm1k, float* sFld_ijp1km1, float inv_pr,
                                           float* J31_d, float* J32_d, float* J33_d,
                                           float* D_Jac_d){
   int i,j,k,ijk;
   int im1jk,ijm1k,ijkm1;
   int im1jkm1,ijm1km1;
   int iStride,jStride,kStride;
   float Txx,Tyy,Tzz;
   float Txz,Tyz;
   i = (blockIdx.x)*blockDim.x + threadIdx.x;
   j = (blockIdx.y)*blockDim.y + threadIdx.y;
   k = (blockIdx.z)*blockDim.z + threadIdx.z;
   iStride = (Ny_d+2*Nh_d)*(Nz_d+2*Nh_d);
   jStride = (Nz_d+2*Nh_d);
   kStride = 1;
   ijk = i*iStride + j*jStride + k*kStride;
   im1jk = (i-1)*iStride + j*jStride + k*kStride;
   ijm1k = i*iStride + (j-1)*jStride + k*kStride;
   ijkm1 = i*iStride + j*jStride + (k-1)*kStride;
   im1jkm1 = (i-1)*iStride + j*jStride + (k-1)*kStride;
   ijm1km1 = i*iStride + (j-1)*jStride + (k-1)*kStride;

   Txx = nu_0_d*inv_pr*(
                        dXi_d*(*sFld_ijk-(*sFld_im1jk))             //dphi_dXi @ i-1/2
                       );

   Tyy = nu_0_d*inv_pr*(
                        dYi_d*(*sFld_ijk-(*sFld_ijm1k))             //dphi_dEta @ j-1/2
                       );

   Txz = nu_0_d*inv_pr*( 0.5*(J31_d[ijk]+J31_d[ijkm1])
                        *0.5*(
                              +(J31_d[ijk]+J31_d[ijkm1])    //Txz part-->dphi_dXi
                               *0.25*dXi_d*( (*sFld_ip1jk)+(*sFld_ip1jkm1)   //dphi_dXi @k-1/2
                                            -(*sFld_im1jk)-(*sFld_im1jkm1))
                              +(J32_d[ijk]+J32_d[ijkm1])    //Txz part-->dphi_dEta
                               *0.25*dYi_d*( (*sFld_ijp1k)+(*sFld_ijp1km1)   //dphi_dEta @k-1/2
                                            -(*sFld_ijm1k)-(*sFld_ijm1km1))
                              +(J33_d[ijk]+J33_d[ijkm1])   //Txz part-->dphi_dZeta
                               *dZi_d*(*sFld_ijk-(*sFld_ijkm1))           //dphi_dZeta @ k-1/2
                        ));
   Tyz = nu_0_d*inv_pr*( 0.5*(J32_d[ijk]+J32_d[ijkm1])
                        *0.5*(
                              +(J31_d[ijk]+J31_d[ijkm1])    //Tyz part-->dphi_dXi
                               *0.25*dXi_d*( (*sFld_ip1jk)+(*sFld_ip1jkm1)   //dphi_dXi @k-1/2
                                            -(*sFld_im1jk)-(*sFld_im1jkm1))
                              +(J32_d[ijk]+J32_d[ijkm1])    //Tyz part-->dphi_dEta
                               *0.25*dYi_d*( (*sFld_ijp1k)+(*sFld_ijp1km1)   //dphi_dEta @k-1/2
                                            -(*sFld_ijm1k)-(*sFld_ijm1km1))
                              +(J33_d[ijk]+J33_d[ijkm1])   //Tyz part-->dphi_dZeta
                               *dZi_d*(*sFld_ijk-(*sFld_ijkm1))           //dphi_dZeta @ k-1/2
                        ));
   Tzz = nu_0_d*inv_pr*( 0.5*(J33_d[ijk]+J33_d[ijkm1])
                        *0.5*(
                              +(J31_d[ijk]+J31_d[ijkm1])    //Tzz part-->dphi_dXi
                               *0.25*dXi_d*( (*sFld_ip1jk)+(*sFld_ip1jkm1)   //dphi_dXi @k-1/2
                                            -(*sFld_im1jk)-(*sFld_im1jkm1))
                              +(J32_d[ijk]+J32_d[ijkm1])    //Tzz part-->dphi_dEta
                               *0.25*dYi_d*( (*sFld_ijp1k)+(*sFld_ijp1km1)   //dphi_dEta @k-1/2
                                            -(*sFld_ijm1k)-(*sFld_ijm1km1))
                              +(J33_d[ijk]+J33_d[ijkm1])   //Tzz part-->dphi_dZeta
                               *dZi_d*(*sFld_ijk-(*sFld_ijkm1))           //dphi_dZeta @ k-1/2
                        ));
   Txx = Txx*D_Jac_d[ijk];
   Tyy = Tyy*D_Jac_d[ijk];
   Txz = Txz*0.25*(D_Jac_d[ijk]+D_Jac_d[im1jk]+D_Jac_d[ijkm1]+D_Jac_d[im1jkm1]);
   Tyz = Tyz*0.25*(D_Jac_d[ijk]+D_Jac_d[ijm1k]+D_Jac_d[ijkm1]+D_Jac_d[ijm1km1]);
   Tzz = Tzz*D_Jac_d[ijk];
   *NuGradX = Txx;
   *NuGradY = Tyy;
   *NuGradZ = Txz+Tyz+Tzz;

} // end cudaDevice_calcConstNuGrad()

/*----->>>>> __device__ void cudaDevice_calcDivNuGrad();  --------------------------------------------------
* This is the cuda version of taking the divergence of nu_0 times the gradient of a field
*/
__device__ void cudaDevice_calcDivNuGrad(float* scalarFrhs, float* rho, float* NuGradX, float* NuGradY, float* NuGradZ, int iFld,
                                         float* J31_d, float* J32_d, float* J33_d,
                                         float* D_Jac_d, float* invD_Jac_d){

   int i,j,k,ijk;
   int im1jk,ijm1k,ijkm1;
   int ip1jk,ijp1k,ijkp1;
   int iStride,jStride,kStride;
   float Frhs_tmp; 
    i = (blockIdx.x)*blockDim.x + threadIdx.x;
    j = (blockIdx.y)*blockDim.y + threadIdx.y;
    k = (blockIdx.z)*blockDim.z + threadIdx.z;
    iStride = (Ny_d+2*Nh_d)*(Nz_d+2*Nh_d);
    jStride = (Nz_d+2*Nh_d);
    kStride = 1;
    ijk = i*iStride + j*jStride + k*kStride;
    im1jk = (i-1)*iStride + j*jStride + k*kStride;
    ijm1k = i*iStride + (j-1)*jStride + k*kStride;
    ijkm1 = i*iStride + j*jStride + (k-1)*kStride;
    ip1jk = (i+1)*iStride + j*jStride + k*kStride;
    ijp1k = i*iStride + (j+1)*jStride + k*kStride;
    ijkp1 = i*iStride + j*jStride + (k+1)*kStride;
    Frhs_tmp = rho[ijk]*invD_Jac_d[ijk]*D_Jac_d[ijk]
                                        *( (NuGradX[ip1jk]-(NuGradX[im1jk]))*dXi_d*0.5
                                          +(NuGradY[ijp1k]-(NuGradY[ijm1k]))*dYi_d*0.5
                                          +(NuGradZ[ip1jk]-(NuGradZ[im1jk]))*dXi_d*0.5*J31_d[ijk]
                                          +(NuGradZ[ijp1k]-(NuGradZ[ijm1k]))*dYi_d*0.5*J32_d[ijk]
                                          +(NuGradZ[ijkp1]-(NuGradZ[ijkm1]))*dZi_d*0.5*J33_d[ijk]
                                         );
    scalarFrhs[ijk] = scalarFrhs[ijk] + Frhs_tmp;
} // end cudaDevice_calcDivNuGrad()
