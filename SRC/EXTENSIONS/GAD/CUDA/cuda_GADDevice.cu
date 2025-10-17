/* FastEddy®: SRC/EXTENSIONS/GAD/CUDA/cuda_GADDevice.cu
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
/*--- GAD*/ 
__constant__ int GADSelector_d;     /* Generalized Actuator Disk Selector: 0=off, 1=on */
__constant__ int GADoutputForces_d;   /* Flag to include GAD forces in the output: 0=off, 1=on */
__constant__ int GADofflineForces_d;  /* Flag to compute GAD forces in an offline mode: 0=off, 1=on */
__constant__ int GADaxialInduction_d;   /* Flag to compute axial induction factor: 0==off (uses prescribed GADaxialIndVal), 1==on */
__constant__ float GADaxialIndVal_d;    /* Prescribed constant axial induction factor when GADaxialInduction==0 */
__constant__ int GADrefSwitch_d;   /* Switch to use reference windspeed: 0=off, 1=on */
__constant__ float GADrefU_d;    /* Prescribed constant reference hub-height windspeed*/
__constant__ int GADForcingSwitch_d;    /* Switch to use the GADrefU-based or local windspeed in computing GAD forces: 0=local, 1=ref */
__constant__ int GADNumTurbines_d;     /* Number of GAD Turbines */
__constant__ int GADNumTurbineTypes_d;  /* Number of GAD Turbine Types */
__constant__ int turbinePolyOrderMax_d; /* Maximum Polynomial order across all turbine types */
__constant__ int turbinePolyClCdrNormSegments_d; /* Number of segments in the normalized radius for the lift and drag coefficient polynomial */
__constant__ int alphaBounds_d;         /* Number of elements in the min/max angle of attack array for the lift/drag curves */

__constant__ int GADsamplingAvgLength_d;   /*length of sampling average windows (averaging over fastest timescales)*/
__constant__ float GADsamplingAvgWeight_d;   /*weight of instances in taking sampling average*/
__constant__ int GADrefSeriesLength_d;   /*number of sample average windows to incorporate into full Reference average*/
__constant__ float GADrefSeriesWeight_d; /*precalculated averaging weight for Reference average*/

__constant__ int numgridCells_away_d; /*Halo-region of cells considered in rotor disk distance-wise smoothing function*/

int* GAD_turbineType_d;     /* Integer class-label for turbine type*/
int* GAD_turbineRank_d;     /* Integer mpi-rank of nacelle center cell for each turbine reference velMag and velDir grid cell*/
int* GAD_turbineRefi_d;     /* Integer i-index of nacelle center cell for each turbine reference velMag and velDir grid cell*/
int* GAD_turbineRefj_d;     /* Integer j-index of nacelle center cell for each turbine reference velMag and velDir grid cell*/
int* GAD_turbineRefk_d;     /* Integer k-index of nacelle center cell for each turbine reference velMag and velDir grid cell*/
int* GAD_turbineYawing_d;   /* Integer indicating in a turbine is currently yawing ==1*/
float* GAD_Xcoords_d;       /* turbine x-location [m] from SW domain corner */
float* GAD_Ycoords_d;       /* turbine y-location [m] from SW domain corner */
float* GAD_turbineRefMag_d; /* Reference "ambient" velocity magnitude for yaw control and beta/omega [m/s]*/
float* GAD_turbineRefDir_d; /* *Reference "ambient" velocity direction (horizontal, met. standard orientation) for yaw control and beta/omega [degrees]*/
float* GAD_turbineUseries_d;/* uSeries of sample averages spanning the rolling-average reference period */
float* GAD_turbineVseries_d;/* vSeries of sample averages spanning the rolling-average reference period */
float* u_sampAvg_d;         /* u sample averages for each turbine */
float* v_sampAvg_d;         /* v sample averages for each turbine */
float* GAD_yawError_d;      /* yaw error between the incoming wind and the turbine orientation */
float* GAD_anFactor_d;     /* turbine axial induction factor at hub heigth*/
float* GAD_rotorTheta_d;    /* turbine yaw angle [deg. North] */
float* GAD_hubHeights_d;    /* turbine hub height [m AGL] */
float* GAD_rotorD_d;        /* turbine rotor diameter [m] */
float* GAD_nacelleD_d;      /* nacelle diameter [m] */
float* turbinePolyTwist_d;  /* turbine-type-specific twist polynomial coefficients*/
float* turbinePolyChord_d;  /* turbine-type-specific chord polynomial coefficients*/
float* turbinePolyPitch_d;  /* turbine-type-specific pitch polynomial coefficients*/
float* turbinePolyOmega_d;  /* turbine-type-specific omega polynomial coefficients*/
float* rnorm_vect_d;        /* turbine-type-specific normalized radious segment limits*/
float* alpha_minmax_vect_d; /* turbine-type-specific maximum and minimum angle of attack for the lift/drag curves*/
float* turbinePolyCl_d;     /* turbine-type-specific lift coefficient polynomial coefficients*/
float* turbinePolyCd_d;     /* turbine-type-specific drag coefficient polynomial coefficients*/

float* GAD_turbineVolMask_d; /* turbine Volume mask (0 if turbine free cell in domain, else turbine ID of cell in turbine yaw-swept volume*/
float* GAD_forceX_d;         /* turbine forces in the x-direction */
float* GAD_forceY_d;         /* turbine forces in the y-direction */
float* GAD_forceZ_d;         /* turbine forces in the z-direction */


/*#################------------ GAD submodule function definitions ------------------#############*/
/*----->>>>> int cuda_GADDeviceSetup();       ---------------------------------------------------------
 * Used to cudaMalloc and cudaMemcpy parameters and arrays, and for the GAD_CUDA submodule.
*/
extern "C" int cuda_GADDeviceSetup(){
   int errorCode = CUDA_GAD_SUCCESS;
   float* tmp_vector;
   float pi=acosf(-1.0);
   int i,iturb;

   cudaMemcpyToSymbol(GADSelector_d, &GADSelector, sizeof(int));
   if(GADSelector > 0){
    /*Host-to-Device memcopy constant values */
    cudaMemcpyToSymbol(GADNumTurbines_d, &GADNumTurbines, sizeof(int));
    cudaMemcpyToSymbol(GADoutputForces_d, &GADoutputForces, sizeof(int));
    cudaMemcpyToSymbol(GADofflineForces_d, &GADofflineForces, sizeof(int));
    cudaMemcpyToSymbol(GADaxialInduction_d, &GADaxialInduction, sizeof(int));
    cudaMemcpyToSymbol(GADaxialIndVal_d, &GADaxialIndVal, sizeof(float));
    cudaMemcpyToSymbol(GADrefSwitch_d, &GADrefSwitch, sizeof(int));
    cudaMemcpyToSymbol(GADrefU_d, &GADrefU, sizeof(float));
    cudaMemcpyToSymbol(GADForcingSwitch_d, &GADForcingSwitch_d, sizeof(int)); 
    cudaMemcpyToSymbol(GADNumTurbineTypes_d, &GADNumTurbineTypes, sizeof(int));
    cudaMemcpyToSymbol(turbinePolyOrderMax_d, &turbinePolyOrderMax, sizeof(int));
    cudaMemcpyToSymbol(turbinePolyClCdrNormSegments_d, &turbinePolyClCdrNormSegments, sizeof(int));
    cudaMemcpyToSymbol(alphaBounds_d, &alphaBounds, sizeof(int));
    cudaMemcpyToSymbol(GADsamplingAvgLength_d, &GADsamplingAvgLength, sizeof(int));
    cudaMemcpyToSymbol(GADsamplingAvgWeight_d, &GADsamplingAvgWeight, sizeof(float));
    cudaMemcpyToSymbol(GADrefSeriesLength_d, &GADrefSeriesLength, sizeof(int));
    cudaMemcpyToSymbol(GADrefSeriesWeight_d, &GADrefSeriesWeight, sizeof(float));
    cudaMemcpyToSymbol(numgridCells_away_d, &numgridCells_away, sizeof(int));

    /*Device memory allocations and Host-to-Device memcopy for turbine arrays */
    fecuda_DeviceMallocInt(GADNumTurbines*sizeof(int), &GAD_turbineType_d);
    fecuda_DeviceMallocInt(GADNumTurbines*sizeof(int), &GAD_turbineRank_d);
    fecuda_DeviceMallocInt(GADNumTurbines*sizeof(int), &GAD_turbineRefi_d);
    fecuda_DeviceMallocInt(GADNumTurbines*sizeof(int), &GAD_turbineRefj_d);
    fecuda_DeviceMallocInt(GADNumTurbines*sizeof(int), &GAD_turbineRefk_d);
    fecuda_DeviceMallocInt(GADNumTurbines*sizeof(int), &GAD_turbineYawing_d);
    cudaMemcpy(GAD_turbineType_d, GAD_turbineType, GADNumTurbines*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(GAD_turbineRank_d, GAD_turbineRank, GADNumTurbines*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(GAD_turbineRefi_d, GAD_turbineRefi, GADNumTurbines*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(GAD_turbineRefj_d, GAD_turbineRefj, GADNumTurbines*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(GAD_turbineRefk_d, GAD_turbineRefk, GADNumTurbines*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(GAD_turbineYawing_d, GAD_turbineYawing, GADNumTurbines*sizeof(int), cudaMemcpyHostToDevice);

    fecuda_DeviceMalloc(GADNumTurbines*sizeof(float), &GAD_turbineRefMag_d);
    fecuda_DeviceMalloc(GADNumTurbines*sizeof(float), &GAD_turbineRefDir_d);
    fecuda_DeviceMalloc(GADNumTurbines*sizeof(float), &GAD_Xcoords_d);
    fecuda_DeviceMalloc(GADNumTurbines*sizeof(float), &GAD_Ycoords_d);
    fecuda_DeviceMalloc(GADNumTurbines*sizeof(float), &GAD_rotorTheta_d);
    fecuda_DeviceMalloc(GADNumTurbines*sizeof(float), &GAD_yawError_d);
    fecuda_DeviceMalloc(GADNumTurbines*sizeof(float), &GAD_anFactor_d);
    cudaMemcpy(GAD_turbineRefMag_d, GAD_turbineRefMag, GADNumTurbines*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(GAD_turbineRefDir_d, GAD_turbineRefDir, GADNumTurbines*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(GAD_Xcoords_d, GAD_Xcoords, GADNumTurbines*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(GAD_Ycoords_d, GAD_Ycoords, GADNumTurbines*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(GAD_rotorTheta_d, GAD_rotorTheta, GADNumTurbines*sizeof(float), cudaMemcpyHostToDevice);
#ifdef DEBUG_GAD
    for (iturb=0; iturb<GADNumTurbines; iturb++){
       printf("%d/%d: iturb--%d: rotorTheta=%f\n",
              mpi_rank_world,mpi_size_world,iturb,GAD_rotorTheta[iturb]);
    }
#endif
    cudaMemcpy(GAD_yawError_d, GAD_yawError, GADNumTurbines*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(GAD_anFactor_d, GAD_anFactor, GADNumTurbines*sizeof(float), cudaMemcpyHostToDevice);
    
    fecuda_DeviceMalloc(GADNumTurbines*GADrefSeriesLength*sizeof(float), &GAD_turbineUseries_d);
    fecuda_DeviceMalloc(GADNumTurbines*GADrefSeriesLength*sizeof(float), &GAD_turbineVseries_d);
    fecuda_DeviceMalloc(GADNumTurbines*sizeof(float), &u_sampAvg_d);
    fecuda_DeviceMalloc(GADNumTurbines*sizeof(float), &v_sampAvg_d);

    //Initialize u_sampAvg & GAD_turbineUseries as constant (per-turbine) then send down to the device 
    tmp_vector = (float *) malloc(GADrefSeriesLength*sizeof(float));
    for (iturb=0; iturb<GADNumTurbines; iturb++){
       tmp_vector[0] = -GAD_turbineRefMag[iturb]*sinf(GAD_turbineRefDir[iturb]*pi/180.0);
#ifdef DEBUG_GAD
       printf("%d/%d: iturb--%d \tu_refSeries_initial=%f\n",
              mpi_rank_world,mpi_size_world,iturb,tmp_vector[0]);
#endif
       for (i=1; i<GADrefSeriesLength; i++){
           tmp_vector[i] = tmp_vector[0];
       }

       cudaMemcpy(&u_sampAvg_d[iturb], &tmp_vector[0], sizeof(float), cudaMemcpyHostToDevice);
       cudaMemcpy(&GAD_turbineUseries_d[iturb*GADrefSeriesLength], tmp_vector, GADrefSeriesLength*sizeof(float), cudaMemcpyHostToDevice);
    }
    //Initialize v_sampAvg & GAD_turbineVseries as constant (per-turbine) then send down to the device 
    for (iturb=0; iturb<GADNumTurbines; iturb++){
       tmp_vector[0] = -GAD_turbineRefMag[iturb]*cosf(GAD_turbineRefDir[iturb]*pi/180.0);
#ifdef DEBUG_GAD
       printf("%d/%d: iturb--%d \tv_refSeries_initial=%f\n",
              mpi_rank_world,mpi_size_world,iturb,tmp_vector[0]);
#endif
       for (i=1; i<GADrefSeriesLength; i++){
           tmp_vector[i] = tmp_vector[0];
       }
       cudaMemcpy(&v_sampAvg_d[iturb], &tmp_vector[0], sizeof(float), cudaMemcpyHostToDevice);
       cudaMemcpy(&GAD_turbineVseries_d[iturb*GADrefSeriesLength], tmp_vector, GADrefSeriesLength*sizeof(float), cudaMemcpyHostToDevice);
    }
    free(tmp_vector);

    fecuda_DeviceMalloc(GADNumTurbineTypes*sizeof(float), &GAD_hubHeights_d);
    fecuda_DeviceMalloc(GADNumTurbineTypes*sizeof(float), &GAD_rotorD_d);
    fecuda_DeviceMalloc(GADNumTurbineTypes*sizeof(float), &GAD_nacelleD_d);
    cudaMemcpy(GAD_hubHeights_d, GAD_hubHeights, GADNumTurbineTypes*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(GAD_rotorD_d, GAD_rotorD, GADNumTurbineTypes*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(GAD_nacelleD_d, GAD_nacelleD, GADNumTurbineTypes*sizeof(float), cudaMemcpyHostToDevice);
   
     
    fecuda_DeviceMalloc(GADNumTurbineTypes*turbinePolyOrderMax*sizeof(float), &turbinePolyTwist_d);
    fecuda_DeviceMalloc(GADNumTurbineTypes*turbinePolyOrderMax*sizeof(float), &turbinePolyChord_d);
    fecuda_DeviceMalloc(GADNumTurbineTypes*turbinePolyOrderMax*sizeof(float), &turbinePolyPitch_d);
    fecuda_DeviceMalloc(GADNumTurbineTypes*turbinePolyOrderMax*sizeof(float), &turbinePolyOmega_d);
    cudaMemcpy(turbinePolyTwist_d, turbinePolyTwist, GADNumTurbineTypes*turbinePolyOrderMax*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(turbinePolyChord_d, turbinePolyChord, GADNumTurbineTypes*turbinePolyOrderMax*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(turbinePolyPitch_d, turbinePolyPitch, GADNumTurbineTypes*turbinePolyOrderMax*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(turbinePolyOmega_d, turbinePolyOmega, GADNumTurbineTypes*turbinePolyOrderMax*sizeof(float), cudaMemcpyHostToDevice);

    fecuda_DeviceMalloc(GADNumTurbineTypes*(turbinePolyClCdrNormSegments+1)*sizeof(float), &rnorm_vect_d);
    fecuda_DeviceMalloc(GADNumTurbineTypes*alphaBounds*sizeof(float), &alpha_minmax_vect_d);
    fecuda_DeviceMalloc(GADNumTurbineTypes*turbinePolyClCdrNormSegments*turbinePolyOrderMax*sizeof(float), &turbinePolyCl_d);
    fecuda_DeviceMalloc(GADNumTurbineTypes*turbinePolyClCdrNormSegments*turbinePolyOrderMax*sizeof(float), &turbinePolyCd_d);

    cudaMemcpy(rnorm_vect_d, rnorm_vect, GADNumTurbineTypes*(turbinePolyClCdrNormSegments+1)*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(alpha_minmax_vect_d, alpha_minmax_vect, GADNumTurbineTypes*alphaBounds*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(turbinePolyCd_d, turbinePolyCd, GADNumTurbineTypes*turbinePolyClCdrNormSegments*turbinePolyOrderMax*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(turbinePolyCl_d, turbinePolyCl, GADNumTurbineTypes*turbinePolyClCdrNormSegments*turbinePolyOrderMax*sizeof(float), cudaMemcpyHostToDevice);

    fecuda_DeviceMalloc((Nxp+2*Nh)*(Nyp+2*Nh)*(Nzp+2*Nh)*sizeof(float), &GAD_turbineVolMask_d);
    cudaMemcpy(GAD_turbineVolMask_d, GAD_turbineVolMask, (Nxp+2*Nh)*(Nyp+2*Nh)*(Nzp+2*Nh)*sizeof(float), cudaMemcpyHostToDevice);

    if (GADoutputForces == 1){
      fecuda_DeviceMalloc((Nxp+2*Nh)*(Nyp+2*Nh)*(Nzp+2*Nh)*sizeof(float), &GAD_forceX_d);
      fecuda_DeviceMalloc((Nxp+2*Nh)*(Nyp+2*Nh)*(Nzp+2*Nh)*sizeof(float), &GAD_forceY_d);
      fecuda_DeviceMalloc((Nxp+2*Nh)*(Nyp+2*Nh)*(Nzp+2*Nh)*sizeof(float), &GAD_forceZ_d);
      cudaMemcpy(GAD_forceX_d, GAD_forceX, (Nxp+2*Nh)*(Nyp+2*Nh)*(Nzp+2*Nh)*sizeof(float), cudaMemcpyHostToDevice);
      cudaMemcpy(GAD_forceY_d, GAD_forceY, (Nxp+2*Nh)*(Nyp+2*Nh)*(Nzp+2*Nh)*sizeof(float), cudaMemcpyHostToDevice);
      cudaMemcpy(GAD_forceZ_d, GAD_forceZ, (Nxp+2*Nh)*(Nyp+2*Nh)*(Nzp+2*Nh)*sizeof(float), cudaMemcpyHostToDevice);
    }
   }


   return(errorCode);
} //end cuda_GADDeviceSetup()

/*----->>>>> extern "C" int cuda_GADDeviceCleanup();  -----------------------------------------------------------
Used to free all malloced memory by the GAD submodule.
*/

extern "C" int cuda_GADDeviceCleanup(){
   int errorCode = CUDA_GAD_SUCCESS;

   /* Free any GAD submodule arrays */
   if(GADSelector > 0){
     cudaFree(GAD_turbineType_d);
     cudaFree(GAD_turbineRank_d);
     cudaFree(GAD_turbineRefi_d);
     cudaFree(GAD_turbineRefj_d);
     cudaFree(GAD_turbineRefk_d);
     cudaFree(GAD_turbineYawing_d);
     cudaFree(GAD_turbineRefMag_d);
     cudaFree(GAD_turbineRefDir_d);
     cudaFree(GAD_turbineUseries_d);
     cudaFree(GAD_turbineVseries_d);
     cudaFree(u_sampAvg_d);
     cudaFree(v_sampAvg_d);
     cudaFree(GAD_yawError);
     cudaFree(GAD_anFactor_d);
     cudaFree(GAD_Xcoords_d);
     cudaFree(GAD_Ycoords_d);
     cudaFree(GAD_rotorTheta_d);
     
     cudaFree(GAD_hubHeights_d);
     cudaFree(GAD_rotorD_d);
     cudaFree(GAD_nacelleD_d);
     
     cudaFree(turbinePolyTwist_d);
     cudaFree(turbinePolyChord_d);
     cudaFree(turbinePolyPitch_d);
     cudaFree(turbinePolyOmega_d);

     cudaFree(GAD_turbineVolMask_d);
     if (GADoutputForces == 1){
       cudaFree(GAD_forceX_d);
       cudaFree(GAD_forceY_d);
       cudaFree(GAD_forceZ_d);
     }
   }

   return(errorCode);

}//end cuda_GADDeviceCleanup()

/*----->>>>> __global__ void  cudaDevice_GADinter();  --------------------------------------------------
* This function is the global entry kernel for computing reference values for GAD yawing and other turbine characteristics
*/
__global__ void cudaDevice_GADinter(float* xPos_d, float* yPos_d, float* zPos_d, float* topoPos_d,
		                    int simTime_it, int timeStage, int numRKstages, float dt,
		                    float* hydroFlds_d, int* GAD_turbineType_d, float* GAD_turbineVolMask_d,
                                    float* GAD_Xcoords_d, float* GAD_Ycoords_d, float* GAD_rotorTheta_d,
                                    float* GAD_hubHeights_d, float* GAD_rotorD_d, float* GAD_nacelleD_d,
                                    float* turbinePolyTwist_d, float* turbinePolyChord_d,
                                    float* turbinePolyPitch_d, float* turbinePolyOmega_d,
                                    float* rnorm_vect_d, float* alpha_minmax_vect_d,
                                    float* turbinePolyCl_d, float* turbinePolyCd_d,
		                    int* GAD_turbineRank_d, int* GAD_turbineRefi_d, int* GAD_turbineRefj_d, int* GAD_turbineRefk_d,
                                    float* u_sampAvg_d, float* v_sampAvg_d,
                                    float* GAD_turbineUseries_d, float* GAD_turbineVseries_d,
                                    float* GAD_turbineRefMag_d, float* GAD_turbineRefDir_d,
				    int* GAD_turbineYawing_d, float* GAD_yawError_d, float* GAD_anFactor_d){

   int i,j,k,ijk,ij;
   int fldStride;
   int iStride,jStride,kStride;
   int iturb;
   int sampleIndex;
   float cell_inRotor;
   float cell_rVector;
   float cell_twistAngle;
   float cell_chordLength;
   float cell_betaAngle;
   float cell_omegaRot;
   float tiltAngle=0.0;

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
      ij = i*(Ny_d+2*Nh_d) + j*(1);
      if(GAD_turbineVolMask_d[ijk] > 0.0){
        iturb = __float2int_rn( GAD_turbineVolMask_d[ijk] ) - 1;
        if((timeStage == numRKstages) &&
           (mpi_rank_world_d == GAD_turbineRank_d[iturb]) &&
           (i == GAD_turbineRefi_d[iturb]) &&
           (j == GAD_turbineRefj_d[iturb]) &&
           (k == GAD_turbineRefk_d[iturb])){
           if(simTime_it%GADsamplingAvgLength_d == 0){
             sampleIndex = (simTime_it/GADsamplingAvgLength_d)%GADrefSeriesLength_d;
             //update the corresponding series element and refMag and refDir values
             update_turbineRefMagDir(sampleIndex, u_sampAvg_d[iturb],v_sampAvg_d[iturb],
                                     &GAD_turbineUseries_d[iturb*GADrefSeriesLength_d], &GAD_turbineVseries_d[iturb*GADrefSeriesLength_d], &GAD_turbineRefMag_d[iturb], &GAD_turbineRefDir_d[iturb]);
#ifdef DEBUG_GAD
             printf("%d/%d:simTime_it=%d, iturb--%d @ (%d,%d,%d): rotTheta = %f, u_sA=%f, v_sA=%f, RefMag=%f, RefDir=%f \n",
                mpi_rank_world_d,mpi_size_world_d,simTime_it,iturb,i,j,k, GAD_rotorTheta_d[iturb], u_sampAvg_d[iturb],v_sampAvg_d[iturb],GAD_turbineRefMag_d[iturb],GAD_turbineRefDir_d[iturb]);
#endif

             //reset the sampleAVG values to zero
             u_sampAvg_d[iturb] = 0.0;
             v_sampAvg_d[iturb] = 0.0;
	     // compute normal/axial induction factor at hub height from time-averaged hub-hight wind speed
	     if (simTime_it > 0){
             cudaDevice_cellInRotor(&cell_inRotor, &cell_rVector, iturb, GAD_Xcoords_d[iturb], GAD_Ycoords_d[iturb],
                                    GAD_rotorTheta_d[iturb], GAD_hubHeights_d[GAD_turbineType_d[iturb]], tiltAngle,
                                    GAD_rotorD_d[GAD_turbineType_d[iturb]], GAD_nacelleD_d[GAD_turbineType_d[iturb]],
                                    xPos_d[ijk], yPos_d[ijk],
                                    zPos_d[ijk]-topoPos_d[ij],
                                    dX_d,dY_d);
	     cudaDevice_GADtwistChord(&turbinePolyTwist_d[GAD_turbineType_d[iturb]*turbinePolyOrderMax_d], &turbinePolyChord_d[GAD_turbineType_d[iturb]*turbinePolyOrderMax_d],
                                      GAD_rotorD_d[GAD_turbineType_d[iturb]], cell_rVector, &cell_twistAngle, &cell_chordLength);
	     cudaDevice_GADbetaOmega(GAD_turbineRefMag_d[iturb], GAD_anFactor_d[iturb],
                                     &turbinePolyPitch_d[GAD_turbineType_d[iturb]*turbinePolyOrderMax_d], &turbinePolyOmega_d[GAD_turbineType_d[iturb]*turbinePolyOrderMax_d],
                                     GAD_rotorD_d[GAD_turbineType_d[iturb]], cell_rVector, cell_twistAngle, &cell_betaAngle, &cell_omegaRot);

	     compute_normalInduction(GAD_turbineRefMag_d[iturb], GAD_rotorD_d[GAD_turbineType_d[iturb]], GAD_nacelleD_d[GAD_turbineType_d[iturb]],
                                     cell_rVector, cell_betaAngle, cell_omegaRot, cell_chordLength,
				     &rnorm_vect_d[GAD_turbineType_d[iturb]*(turbinePolyClCdrNormSegments_d+1)],
                                     &alpha_minmax_vect_d[GAD_turbineType_d[iturb]*alphaBounds_d],
                                     &turbinePolyCl_d[GAD_turbineType_d[iturb]*turbinePolyClCdrNormSegments_d*turbinePolyOrderMax_d],
                                     &turbinePolyCd_d[GAD_turbineType_d[iturb]*turbinePolyClCdrNormSegments_d*turbinePolyOrderMax_d],
				     &GAD_anFactor_d[iturb]);
#ifdef DEBUG_GAD
	     printf("GAD_turbineRefMag_d[iturb]=%f,GAD_anFactor_d[iturb]=%f \n",GAD_turbineRefMag_d[iturb],GAD_anFactor_d[iturb]);
#endif
	     } // if (simTime_it > 0)

           }//endif beginning/end of a sample window
           // accumulate this timestep instance into the sampling window average
           update_sampleRefVel(hydroFlds_d[fldStride*U_INDX+ijk], hydroFlds_d[fldStride*V_INDX+ijk], hydroFlds_d[fldStride*RHO_INDX+ijk], &u_sampAvg_d[iturb], &v_sampAvg_d[iturb]);

	   if ((simTime_it%GADsamplingAvgLength_d == 0) && (simTime_it >= GADsamplingAvgLength_d*GADrefSeriesLength_d)){
	     update_yawError(&GAD_turbineRefDir_d[iturb], &GAD_rotorTheta_d[iturb], &GAD_yawError_d[iturb], &GAD_turbineYawing_d[iturb], dt);
#ifdef DEBUG_GAD
	   printf("%d/%d:simTime_it=%d, iturb--%d @ (%d,%d,%d) [after update_yawError]: GAD_turbineYawing_d[iturb]=%d, GAD_yawError_d[iturb]=%f \n",
		  mpi_rank_world_d,mpi_size_world_d,simTime_it,iturb,i,j,k,GAD_turbineYawing_d[iturb], GAD_yawError_d[iturb]);
#endif
           }
	   if (GAD_turbineYawing_d[iturb] == 1){
	     update_rotorTheta(&GAD_turbineRefDir_d[iturb], &GAD_rotorTheta_d[iturb], &GAD_yawError_d[iturb], &GAD_turbineYawing_d[iturb], dt);
#ifdef DEBUG_GAD
           printf("%d/%d:simTime_it=%d, iturb--%d @ (%d,%d,%d) [after update_rotorTheta]: GAD_turbineYawing_d[iturb]=%d, GAD_rotorTheta_d[iturb]=%f \n",
		  mpi_rank_world_d,mpi_size_world_d,simTime_it,iturb,i,j,k,GAD_turbineYawing_d[iturb], GAD_rotorTheta_d[iturb]);
#endif
	   }

        }

      } // end if(GAD_turbineVolMask_d[ijk] > 0.0){
   }//end if in the range of non-halo cells

} // end cudaDevice_GADinter()

/*----->>>>> __global__ void cudaDevice_GADfinal();  --------------------------------------------------
* This function is the global entry kernel for computing GAD forcing from turbines
*/
__global__ void cudaDevice_GADfinal(float* xPos_d, float* yPos_d, float* zPos_d, float* topoPos_d,
                                    float* hydroFlds_d, float* hydroFldsFrhs_d, int simTime_it, float dt,
                                    int* GAD_turbineType_d, float* GAD_turbineVolMask_d,
                                    float* GAD_Xcoords_d, float* GAD_Ycoords_d, float* GAD_rotorTheta_d,
                                    float* GAD_hubHeights_d, float* GAD_rotorD_d, float* GAD_nacelleD_d,
                                    float* turbinePolyTwist_d, float* turbinePolyChord_d,
                                    float* turbinePolyPitch_d, float* turbinePolyOmega_d,
                                    float* rnorm_vect_d, float* alpha_minmax_vect_d,
                                    float* turbinePolyCl_d, float* turbinePolyCd_d,
				    float* GAD_turbineRefMag_d, float* GAD_anFactor_d,
                                    float* GAD_forceX_d, float* GAD_forceY_d, float* GAD_forceZ_d){

   int i,j,k,ijk,ij;
   int fldStride;
   int iStride,jStride,kStride;
   int iturb;
   float cell_inRotor;
   float cell_rVector;
   float cell_twistAngle;
   float cell_chordLength;
   float cell_betaAngle;
   float cell_omegaRot;
   float cell_forceN;
   float cell_forceT;

   float tiltAngle=0.0;

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
      ij = i*(Ny_d+2*Nh_d) + j*(1);
      if(GAD_turbineVolMask_d[ijk] > 0.0){
        iturb = __float2int_rn( GAD_turbineVolMask_d[ijk] ) - 1;
        cudaDevice_cellInRotor(&cell_inRotor, &cell_rVector, iturb, GAD_Xcoords_d[iturb], GAD_Ycoords_d[iturb], 
                               GAD_rotorTheta_d[iturb], GAD_hubHeights_d[GAD_turbineType_d[iturb]], tiltAngle,
                               GAD_rotorD_d[GAD_turbineType_d[iturb]], GAD_nacelleD_d[GAD_turbineType_d[iturb]],
                               xPos_d[ijk], yPos_d[ijk], 
                               zPos_d[ijk]-topoPos_d[ij],
                               dX_d,dY_d);
        if(cell_inRotor > 0.0){ //Compute the momentum Frhs forces from the GAD blade element momentum theory for this cell
          cudaDevice_GADtwistChord(&turbinePolyTwist_d[GAD_turbineType_d[iturb]*turbinePolyOrderMax_d], &turbinePolyChord_d[GAD_turbineType_d[iturb]*turbinePolyOrderMax_d],
                                   GAD_rotorD_d[GAD_turbineType_d[iturb]], cell_rVector, &cell_twistAngle, &cell_chordLength);
          cudaDevice_GADbetaOmega(GAD_turbineRefMag_d[iturb], GAD_anFactor_d[iturb],
                                  &turbinePolyPitch_d[GAD_turbineType_d[iturb]*turbinePolyOrderMax_d], &turbinePolyOmega_d[GAD_turbineType_d[iturb]*turbinePolyOrderMax_d],
                                  GAD_rotorD_d[GAD_turbineType_d[iturb]], cell_rVector, cell_twistAngle, &cell_betaAngle, &cell_omegaRot);
//#define DEBUG_GAD
#ifdef DEBUG_GAD
      if(fabsf(cell_rVector-0.5*(GAD_rotorD_d[GAD_turbineType_d[iturb]]-GAD_nacelleD_d[GAD_turbineType_d[iturb]]))<5.0){
        printf("At i,j,k (%d,%d,%d): cell_rVector = %f, twist=%f, chord=%f, pitch=%f, omega=%f, U_ijk=%f\n",
                i,j,k,cell_rVector, cell_twistAngle, cell_chordLength,cell_betaAngle, cell_omegaRot, 
                sqrtf(powf(hydroFlds_d[fldStride*U_INDX+ijk]/hydroFlds_d[fldStride*RHO_INDX+ijk],2.0)+powf(hydroFlds_d[fldStride*V_INDX+ijk]/hydroFlds_d[fldStride*RHO_INDX+ijk],2.0)));        
      }
#endif
          cudaDevice_GADforcesCompute(hydroFlds_d[fldStride*U_INDX+ijk], hydroFlds_d[fldStride*V_INDX+ijk], hydroFlds_d[fldStride*RHO_INDX+ijk],
                                      GAD_rotorD_d[GAD_turbineType_d[iturb]], GAD_nacelleD_d[GAD_turbineType_d[iturb]],
                                      cell_rVector, cell_betaAngle, cell_omegaRot, cell_chordLength,
                                      &rnorm_vect_d[GAD_turbineType_d[iturb]*(turbinePolyClCdrNormSegments_d+1)],
                                      &alpha_minmax_vect_d[GAD_turbineType_d[iturb]*alphaBounds_d],
                                      &turbinePolyCl_d[GAD_turbineType_d[iturb]*turbinePolyClCdrNormSegments_d*turbinePolyOrderMax_d],
                                      &turbinePolyCd_d[GAD_turbineType_d[iturb]*turbinePolyClCdrNormSegments_d*turbinePolyOrderMax_d],
                                      &cell_forceN, &cell_forceT);
	  if (simTime_it >= (GADsamplingAvgLength_d*GADrefSeriesLength_d)){ // prevents use of potentially unrealistic initial values...
          cudaDevice_GADforcesApply(hydroFlds_d[fldStride*RHO_INDX+ijk], GAD_Xcoords_d[iturb], GAD_Ycoords_d[iturb],
                                    GAD_hubHeights_d[GAD_turbineType_d[iturb]], GAD_rotorTheta_d[iturb], GAD_rotorD_d[GAD_turbineType_d[iturb]],
                                    xPos_d[ijk], yPos_d[ijk], (zPos_d[ijk]-topoPos_d[ij]),
                                    cell_forceN, cell_forceT, &hydroFldsFrhs_d[fldStride*U_INDX+ijk], &hydroFldsFrhs_d[fldStride*V_INDX+ijk],
                                    &hydroFldsFrhs_d[fldStride*W_INDX+ijk],&GAD_forceX_d[ijk],&GAD_forceY_d[ijk],&GAD_forceZ_d[ijk],
                                    cell_rVector, GAD_nacelleD_d[GAD_turbineType_d[iturb]]);
	  }
        }
      }//if this is a cell in the yaw-swept volume sphere
   }//end if in the range of non-halo cells

} // end cudaDevice_GADfinal()

/*----->>>>> __device__ void  cudaDevice_cellInRotor();  --------------------------------------------------
* This functions calculates a radial vector and setes a flag to detrmine if a cell is in a rotor disk area
*/
__device__ void cudaDevice_cellInRotor(float* cell_inRotor, float* cell_rVector, 
                                       int iturb, float turbX, float turbY, 
                                       float turbTheta, float turbHubHgt, float tiltAngle,
                                       float rotorD, float nacelleD,   
                                       float xLoc, float yLoc, float zLoc, float dx, float dy){

    float x_hat[3];
    float dr[3];
   
    float perpDist;
    float perpdx_rot;
    float pi = 3.1415926535;

    /*Initialize the cell flag value to 0.0 (False) */
    *cell_inRotor = 0.0;

    //Unit horizontal vector normal to the rotor-disk plane 
    x_hat[0] = cosf(tiltAngle*pi/180.0)*cosf(turbTheta*pi/180.0); 
    x_hat[1] = cosf(tiltAngle*pi/180.0)*sinf(turbTheta*pi/180.0); 
    x_hat[2] = -sinf(tiltAngle*pi/180.0); 
    
    //Vector from nacelle center to current grid point
    dr[0] = xLoc-turbX;
    dr[1] = yLoc-turbY;
    dr[2] = zLoc-turbHubHgt;

    //Perpendicular distance from nacelle-center to current grid point (normal to the rotor-disk plane)
    perpDist = dr[0]*x_hat[0] + dr[1]*x_hat[1] + dr[2]*x_hat[2];
    //Proper radial distance of blade segment 
    *cell_rVector = sqrtf( powf(dr[0]-perpDist*x_hat[0],2.0)
                          +powf(dr[1]-perpDist*x_hat[1],2.0) 
                          +powf(dr[2]-perpDist*x_hat[2],2.0) );

    /*Perpendicular "dx" in the rotated "x-y" plane  */
    perpdx_rot =  fabsf(dx*cosf(0.5*pi+turbTheta*pi/180.0)) + fabsf(dy*sinf(0.5*pi+turbTheta*pi/180.0));

    if(   (perpDist < __int2float_rn(numgridCells_away_d)*perpdx_rot)
       && *cell_rVector <= (0.5*rotorD)     //Not in notebook but needed?
       && *cell_rVector > (0.5*nacelleD) ){
       *cell_inRotor = 1.0;
    }

} // end cudaDevice_cellInRotor()
/*----->>>>> __device__ void cudaDevice_GADtwistChord();  --------------------------------------------------
*/
__device__ void cudaDevice_GADtwistChord(float* turbinePolyTwist_d, float* turbinePolyChord_d,
                                         float rotorD, float turbineRadius, float* twist_angle, float* chord_length){

  int nn;
  float r_norm,blade_length;
  float pi = acosf(-1.0);
  float temp_expo;

  blade_length = 0.5*rotorD;
  r_norm = turbineRadius/blade_length;

  /* twist angle */
  temp_expo = __int2float_rn(turbinePolyOrderMax_d);
  *twist_angle = 0.0;
  for (nn=0; nn<turbinePolyOrderMax_d; nn++){
     temp_expo = temp_expo - 1.0;
     *twist_angle = *twist_angle + powf(r_norm,temp_expo)*turbinePolyTwist_d[nn];
  }
  *twist_angle = *twist_angle*(pi/180.0); // rad

  /* chord length */
  temp_expo = __int2float_rn(turbinePolyOrderMax_d);
  *chord_length = 0.0;
  for (nn=0; nn<turbinePolyOrderMax_d; nn++){
     temp_expo = temp_expo - 1.0;
     *chord_length = *chord_length + powf(r_norm,temp_expo)*turbinePolyChord_d[nn];
  }
  //JAS 4-19-2023 Opting not to use chord polynomial fits normalized by blade length
  //*chord_length = *chord_length*blade_length; // rad S  

} // end cudaDevice_GADtwistChord()

/*----->>>>> __device__ void cudaDevice_GADbetaOmega();  --------------------------------------------------
*/
__device__ void cudaDevice_GADbetaOmega(float turbineRefMag, float anFactor, float* turbinePolyPitch_d, float* turbinePolyOmega_d,
                                        float rotorD, float turbineRadius, float twist_angle, float* beta_angle, float* omega_rot){

  int nn;
  float pitch_angle;
  float U_ijk;
  float pi = acosf(-1.0);
  float temp_expo;
/*  Removed use of this limiters
  float pitch_umin = 11.0; // perhaps to be read in from netCDF turbine characteristics file...
  float omega_umin = 4.0; //  perhaps to be read in from netCDF turbine characteristics file...
  float omega_umax = 9.0; //  perhaps to be read in from netCDF turbine characteristics file...
*/
  if(GADrefSwitch_d == 1){
    U_ijk = GADrefU_d;
  }else{
    U_ijk = turbineRefMag/(1.0-anFactor); // should this include vertical velcoty too - w???
  }

  /* pitch angle */
  temp_expo = __int2float_rn(turbinePolyOrderMax_d);
  for (nn=0; nn<turbinePolyOrderMax_d; nn++){
     temp_expo = temp_expo - 1.0;
     pitch_angle = pitch_angle + powf(U_ijk,temp_expo)*turbinePolyPitch_d[nn];
  }
  pitch_angle = pitch_angle*(pi/180.0); // rad
  *beta_angle = twist_angle + pitch_angle; // total twist angle

  /* rotational speed */
  temp_expo = __int2float_rn(turbinePolyOrderMax_d);
  *omega_rot = 0.0;
  for (nn=0; nn<turbinePolyOrderMax_d; nn++){
     temp_expo = temp_expo - 1.0;
     *omega_rot = *omega_rot + powf(U_ijk,temp_expo)*turbinePolyOmega_d[nn];
  }
  *omega_rot = *omega_rot*(2.0*pi/60.0); // rad s-1

} // end cudaDevice_GADbetaOmega()

/*----->>>>> __device__ void cudaDevice_GADforcesCompute();  --------------------------------------------------
*/
__device__ void cudaDevice_GADforcesCompute(float u, float v, float rho, float rotorD, float nacelleD,
                                            float turbineRadius, float beta_angle, float omega_rot, float chord_length,
                                            float *rnorm_vect, float *alpha_minmax_vect, float *turbinePolyCl, float *turbinePolyCd,
                                            float *GADforce_n, float *GADforce_t){

  float U_ijk,U_rel,phi_rel,alpha_angle;
  float a_tol;
  float a_tol_min = 1.0e-5; // minimum tolerance for converged induction factors
  float at_it,at_0;
  float an_it,an_0;
  float pi = acosf(-1.0);
  float C_l,C_d;
  float r_norm,blade_length;
  float B_num = 3.0; // number of blades
  float sigma,f_tip,f_hub,F_tot;
  float r_hub;
  float c_n,c_t;
  float L_f,D_f;
  int iter_cnt;
  int max_iter = 50;
  float U_ijk_tmp; 
  float switchFactor;

  blade_length = 0.5*rotorD;
  r_norm = turbineRadius/blade_length;
  r_hub = 0.5*nacelleD;

  if(GADForcingSwitch_d == 1){
    U_ijk = GADrefU_d; 
    switchFactor = 1.0;
  }else{  
    U_ijk = sqrtf(powf(u/rho,2.0)+powf(v/rho,2.0));
    switchFactor = 0.0;
  }//end if-else  GADForcingSwitch_d == 1

  if (GADaxialInduction_d == 1){
    an_it = 0.0;
  }else{
    an_it = GADaxialIndVal_d;
  }
  at_it = 0.0;
  an_0 = an_it;
  at_0 = at_it;
  // iterative solve for induction factor(s)
  a_tol = a_tol_min + 1.0; //Initialize a_tol to get into the while loop
  iter_cnt = 0;
  while((a_tol > a_tol_min) && (iter_cnt < max_iter)){

    U_ijk_tmp = (1.0-switchFactor)*U_ijk + switchFactor*U_ijk*(1.0-an_it); 
    U_rel = sqrtf(powf(U_ijk_tmp,2.0) + powf(omega_rot*turbineRadius*(1.0+at_it),2.0)); // wind/blade relative velocity
    phi_rel = atanf(U_ijk_tmp/(omega_rot*turbineRadius*(1.0+at_it))); // angle between relative velocity and plane of rotation    
    alpha_angle = (phi_rel - beta_angle)*(180.0/pi); // angle of attack

    alpha_angle = fmaxf(fminf(alpha_angle,alpha_minmax_vect[1]),alpha_minmax_vect[0]);
    compute_ClCd_incoeff(rnorm_vect,turbinePolyCl,turbinePolyCd,alpha_angle,r_norm,&C_l,&C_d); // lift and drag coefficients
    sigma = B_num*chord_length/(2.0*pi*turbineRadius); // solidity factor
    f_tip = B_num*(0.5*rotorD-turbineRadius)/(2.0*turbineRadius*sin(phi_rel)); // blade tip losses
    f_hub = B_num*(turbineRadius-r_hub)/(2.0*turbineRadius*sin(phi_rel)); // blade hub losses
    F_tot = (2.0/pi)*acosf(expf(-f_tip))*(2.0/pi)*acosf(expf(-f_hub)); // blade total losses

    if (GADaxialInduction_d == 1){
      c_n = C_l*cosf(phi_rel)+C_d*sinf(phi_rel);
      an_it = 1.0/(1.0 + 4.0*F_tot*sinf(phi_rel)*sinf(phi_rel)/(sigma*c_n)); // normal induction factor
    }

    c_t = C_l*sinf(phi_rel)-C_d*cosf(phi_rel);
    at_it = 1.0/(4.0*F_tot*sinf(phi_rel)*cosf(phi_rel)/(sigma*c_t) - 1.0); // tangential induction factor
    __syncthreads();

    a_tol = sqrtf(powf(an_it-an_0,2.0) + powf(at_it-at_0,2.0));
    an_0 = an_it;
    at_0 = at_it;

    __syncthreads();
    iter_cnt++;
  } //while(a_tol > a_tol_min) --end of iterative process

  // calculate lift and drag forces (N m-1)
  L_f = 0.5*rho*U_rel*U_rel*chord_length*C_l;
  D_f = 0.5*rho*U_rel*U_rel*chord_length*C_d;
  // calculate normal and tangential forces (N m-2)
  *GADforce_n = fmaxf((B_num/(2.0*pi*turbineRadius))*(L_f*cosf(phi_rel)+D_f*sinf(phi_rel)),0.0);
  *GADforce_t = fmaxf((B_num/(2.0*pi*turbineRadius))*(L_f*sinf(phi_rel)-D_f*cosf(phi_rel)),0.0);
#ifdef DEBUG_GAD
  if(fabsf(turbineRadius-0.5*(rotorD-nacelleD))<5.0){
        printf("rVector = %f, phi_rel=%f, alpha_angle=%f, f_tip=%f, f_hub=%f, F_tot=%f, sigma=%f, c_n=%f, an_it=%f, at_it=%f a_tol=%14.12f\n\t C_l=%f, C_d=%f, U_rel=%f, L=%f, D=%f, F_n=%f, F_t=%f\n",
                turbineRadius, phi_rel, alpha_angle, f_tip, f_hub, F_tot, sigma, c_n, an_it, at_it, a_tol, C_l, C_d, U_rel, L_f, D_f, *GADforce_n, *GADforce_t);
  }
#endif
} // end cudaDevice_GADforcesCompute()

/*----->>>>> __device__ void cudaDevice_GADforcesApply();  --------------------------------------------------
*/
__device__ void cudaDevice_GADforcesApply(float rho, float turb_Xcoord, float turb_Ycoord, float hubHeight, float rotorTheta, float rotorD, 
                                          float xLoc, float yLoc, float zLoc,
                                          float GADforce_n, float GADforce_t, float* GADforce_x, float* GADforce_y, float* GADforce_z,
                                          float* GAD_fX, float* GAD_fY, float* GAD_fZ, float turbineRadius, float nacelleD){

  float hor_dist,sign_quadrant;
  float dist_x,dist_y,y_prime; // x_prime
  float epsilon;
  float F_x,F_y,F_z;
  float F_factor;
  float pi = acosf(-1.0);
  float rotorTheta_tmp;

  rotorTheta_tmp = rotorTheta - 180.0;
  dist_x = xLoc - turb_Xcoord;
  dist_y = yLoc - turb_Ycoord;
  hor_dist = sqrtf(powf(dist_x,2.0) + powf(dist_y,2.0));
  sign_quadrant = 0.0;
//  x_prime =  dist_x*cosf(rotorTheta_tmp*pi/180.0) + dist_y*sinf(rotorTheta_tmp*pi/180.0);
  y_prime = -dist_x*sinf(rotorTheta_tmp*pi/180.0) + dist_y*cosf(rotorTheta_tmp*pi/180.0);
  sign_quadrant = copysign(1.0,y_prime);
  epsilon = atan2f(zLoc-hubHeight,sign_quadrant*hor_dist);

  // project turbine forces into cartesian grid coordinates
  F_x = -GADforce_n*cosf(rotorTheta_tmp*pi/180.0) - GADforce_t*sinf(epsilon)*sinf(rotorTheta_tmp*pi/180.0);
  F_y = -GADforce_n*sinf(rotorTheta_tmp*pi/180.0) + GADforce_t*sinf(epsilon)*cosf(rotorTheta_tmp*pi/180.0);
  F_z = -GADforce_t*cosf(epsilon);
  distribute_GADforces(xLoc, yLoc, turb_Xcoord, turb_Ycoord, rotorTheta_tmp, rotorD, &F_factor);

#ifdef DEBUG_GAD
  if(fabsf(turbineRadius-0.5*(rotorD-nacelleD))<5.0){
        printf("rVector = %f, (xLoc,yLoc)=(%f,%f), (xT,yT)=(%f,%f), rotorTheta_tmp=%f, \n\t dist_x=%f, dist_y=%f,zLoc=%f, epsilon=%f, F_factor=%f\n\t F_x=%f, F_y=%f, F_z=%f\n",
                turbineRadius, xLoc,yLoc,turb_Xcoord,turb_Ycoord, rotorTheta_tmp, dist_x,dist_y,zLoc,epsilon, F_factor, F_x, F_y, F_z);
  }
#endif
  // forces exerted by the turbine on the flow have opposite sign
  if (GADofflineForces_d==0){
    *GADforce_x = *GADforce_x - rho*F_factor*F_x;
    *GADforce_y = *GADforce_y - rho*F_factor*F_y;
    *GADforce_z = *GADforce_z - rho*F_factor*F_z;
  }
  // save forces for output
  if (GADoutputForces_d==1){
    *GAD_fX = -rho*F_factor*F_x;
    *GAD_fY = -rho*F_factor*F_y;
    *GAD_fZ = -rho*F_factor*F_z;
  }

} // end cudaDevice_GADforcesApply()

/*----->>>>> __device__ void compute_ClCd_incoeff();  --------------------------------------------------
*/
__device__ void compute_ClCd_incoeff(float* rnorm_vect, float* turbinePolyCl, float* turbinePolyCd, float alpha, float r_norm, float* C_l, float* C_d){

  int rn,rn_seg;
  float rn_min,rn_max;
  int ind_ps,ind_pe,nn;
  float temp_expo;

  *C_l = 0.0;
  *C_d = 0.0;

  // determine the segment of normalized radius where the point belongs
  rn_seg = -1;
  for (rn=0; rn<turbinePolyClCdrNormSegments_d; rn++){
     rn_min = rnorm_vect[rn];
     rn_max = rnorm_vect[rn+1];
     if ((r_norm>rn_min)&&(rn_min<=rn_max)){
       rn_seg = rn;
     }
  }

  if ((rn_seg>=0)&&(rn_seg<turbinePolyClCdrNormSegments_d)){
    temp_expo = __int2float_rn(turbinePolyOrderMax_d);
    ind_ps = rn_seg*turbinePolyOrderMax_d;
    ind_pe = ind_ps + turbinePolyOrderMax_d;
    for (nn=ind_ps; nn<ind_pe; nn++){
       temp_expo = temp_expo - 1.0;
       *C_l = *C_l + turbinePolyCl[nn]*powf(alpha,temp_expo);
       *C_d = *C_d + turbinePolyCd[nn]*powf(alpha,temp_expo);
    }
  }

} // end compute_ClCd_incoeff()


/*----->>>>> __device__ void distribute_GADforces();  --------------------------------------------------
*/
__device__ void distribute_GADforces(float xLoc, float yLoc, float x_turb, float y_turb, float theta_turb, float rotorD, float* F_dist_fact){

  float x1,x2,y1,y2;
  float dist_grid_rotor,proj_dx;
  float a_coeff,b_coeff;
  float pi = acosf(-1.0);

  x1 = x_turb - 0.5*rotorD*cosf(0.5*pi + theta_turb*pi/180.0);
  x2 = x_turb + 0.5*rotorD*cosf(0.5*pi + theta_turb*pi/180.0);
  y1 = y_turb - 0.5*rotorD*sinf(0.5*pi + theta_turb*pi/180.0);
  y2 = y_turb + 0.5*rotorD*sinf(0.5*pi + theta_turb*pi/180.0);

  dist_grid_rotor = fabsf((x2-x1)*(y1-yLoc)-(x1-xLoc)*(y2-y1))/sqrtf(powf(x2-x1,2.0)+powf(y2-y1,2.0));
  proj_dx = fabsf(dX_d*cosf(0.5*pi+theta_turb*pi/180.0)) + fabsf(dY_d*sinf(0.5*pi+theta_turb*pi/180.0));
  a_coeff = sqrtf(2.0*pi)*proj_dx;
  b_coeff = 2.0*proj_dx*proj_dx;

  *F_dist_fact = (1.0/a_coeff)*expf(-dist_grid_rotor*dist_grid_rotor/b_coeff);

} // end distribute_GADforces()

/*----->>>>> __device__ void update_sampleRefVel();  --------------------------------------------------
*/
__device__ void update_sampleRefVel(float u, float v, float rho, float* u_sampAvg, float* v_sampAvg){

  *u_sampAvg = *u_sampAvg+GADsamplingAvgWeight_d*u/rho;
  *v_sampAvg = *v_sampAvg+GADsamplingAvgWeight_d*v/rho;

} // update_sampleRefVel()

/*----->>>>> __device__ void update_turbineRefMagDir();  --------------------------------------------------
*/
__device__ void update_turbineRefMagDir(int sampleIndex, float u_sampAvg, float v_sampAvg,
		                        float* uSeries, float* vSeries, float* turbineRefMag, float* turbineRefDir){
  
  int idx;
  float u_seriesAvg;
  float v_seriesAvg;
  float pi = acosf(-1.0);

  uSeries[sampleIndex] = u_sampAvg;
  vSeries[sampleIndex] = v_sampAvg;
  u_seriesAvg = 0.0;
  v_seriesAvg = 0.0;

  for(idx=0; idx < GADrefSeriesLength_d; idx++){
    u_seriesAvg = u_seriesAvg + uSeries[idx];
    v_seriesAvg = v_seriesAvg + vSeries[idx];
  }
  u_seriesAvg = u_seriesAvg*GADrefSeriesWeight_d;  
  v_seriesAvg = v_seriesAvg*GADrefSeriesWeight_d;  

  *turbineRefMag = sqrtf(powf(u_seriesAvg,2.0)+powf(v_seriesAvg,2.0));
  *turbineRefDir = 180.0 + atan2f(u_seriesAvg,v_seriesAvg)*180.0/pi;

} // update_turbineRefMagDir()

/*----->>>>> __device__ void update_yawError();  --------------------------------------------------
*/
__device__ void update_yawError(float* turbineRefDir, float* rotorTheta, float* yawError, int* turbineYawing, float dt){

  float diff_angle;
  float t_refresh;
  float yawErr_max = 10000.0; // (deg)^2 s -> threshold to start yawing to align with incoming wind

  if (*turbineYawing == 0){ // turbine currently not yawing
    t_refresh = GADsamplingAvgLength_d*dt;
    Angle_TurbWind(*turbineRefDir, *rotorTheta, &diff_angle);
    *yawError = *yawError + copysign(1.0,diff_angle)*powf(diff_angle,2.0)*t_refresh;
    if (fabsf(*yawError) >= yawErr_max){
      *turbineYawing = 1;
    }
  }

} // update_yawError()

/*----->>>>> __device__ void update_rotorTheta();  --------------------------------------------------
*/
__device__ void update_rotorTheta(float* turbineRefDir, float* rotorTheta, float* yawError, int* turbineYawing, float dt){

  float diff_angle;
  float yawing_angle;
  float yawing_rate = 2.0; // deg s-1 -> turbine's yawing rate
  float ref360 = 360.0;

  yawing_angle = copysign(1.0,*yawError)*yawing_rate*dt;
  *rotorTheta = *rotorTheta + yawing_angle;
  *rotorTheta = fmodf(*rotorTheta,ref360);
  Angle_TurbWind(*turbineRefDir, *rotorTheta, &diff_angle);

  if (fabsf(diff_angle) <  fabsf(yawing_angle)){
    *turbineYawing = 0;
    *yawError = 0.0;
  }

#ifdef DEBUG_GAD
  printf("[in update_rotorTheta]: GAD_turbineYawing_d[iturb]=%d, GAD_rotorTheta_d[iturb]=%f, diff_angle=%f, yawing_angle=%f \n",
          *turbineYawing, *rotorTheta,diff_angle,yawing_angle);
#endif

} // update_rotorTheta()

/*----->>>>> __device__ void Angle_TurbWind();  --------------------------------------------------
*/
__device__ void Angle_TurbWind(float turbineRefDir, float rotorTheta, float* diff_angle){

  float sign_diff;
  *diff_angle = 0;

  *diff_angle = fmodf(270.0 - turbineRefDir,360.0) - rotorTheta;
  if (fabsf(*diff_angle) > 180.0){
    sign_diff = -copysign(1.0,*diff_angle);
    *diff_angle = sign_diff*(360.0 - fabsf(*diff_angle));
  }

} // Angle_TurbWind()

/*----->>>>> __device__ void compute_normalInduction();  --------------------------------------------------
*/
__device__ void compute_normalInduction(float turbineRefMag, float rotorD, float nacelleD,
                                        float turbineRadius, float beta_angle, float omega_rot, float chord_length,
                                        float *rnorm_vect, float *alpha_minmax_vect, float *turbinePolyCl, float *turbinePolyCd,
					float *turbineRefAn){

  float U_ijk,phi_rel,alpha_angle;
  float a_tol;
  float a_tol_min = 1.0e-5; // minimum tolerance for converged induction factors
  float at_it,at_0;
  float an_it,an_0;
  float pi = acosf(-1.0);
  float C_l,C_d;
  float r_norm,blade_length;
  float B_num = 3.0; // number of blades
  float sigma,f_tip,f_hub,F_tot;
  float r_hub;
  float c_n,c_t;
  int iter_cnt;
  int max_iter = 50;
  float U_ijk_tmp;
  float switchFactor;

  blade_length = 0.5*rotorD;
  r_norm = turbineRadius/blade_length;
  r_hub = 0.5*nacelleD;

  if(GADForcingSwitch_d == 1){
    U_ijk = GADrefU_d;
    switchFactor = 1.0;
  }else{
    U_ijk = turbineRefMag; // hub-height local velocity (time averaged)
    switchFactor = 0.0;
  }

  an_it = 0.0;
  at_it = 0.0;
  an_0 = an_it;
  at_0 = at_it;
  // iterative solve for induction factor(s)
  a_tol = a_tol_min + 1.0; //Initialize a_tol to get into the while loop
  iter_cnt = 0;
  while((a_tol > a_tol_min) && (iter_cnt < max_iter)){

    U_ijk_tmp = (1.0-switchFactor)*U_ijk + switchFactor*U_ijk*(1.0-an_it);
    phi_rel = atanf(U_ijk_tmp/(omega_rot*turbineRadius*(1.0+at_it))); // angle between relative velocity and plane of rotation
    alpha_angle = (phi_rel - beta_angle)*(180.0/pi); // angle of attack

    alpha_angle = fmaxf(fminf(alpha_angle,alpha_minmax_vect[1]),alpha_minmax_vect[0]);
    compute_ClCd_incoeff(rnorm_vect,turbinePolyCl,turbinePolyCd,alpha_angle,r_norm,&C_l,&C_d); // lift and drag coefficients
    sigma = B_num*chord_length/(2.0*pi*turbineRadius); // solidity factor
    f_tip = B_num*(0.5*rotorD-turbineRadius)/(2.0*turbineRadius*sin(phi_rel)); // blade tip losses
    f_hub = B_num*(turbineRadius-r_hub)/(2.0*turbineRadius*sin(phi_rel)); // blade hub losses
    F_tot = (2.0/pi)*acosf(expf(-f_tip))*(2.0/pi)*acosf(expf(-f_hub)); // blade total losses

    c_n = C_l*cosf(phi_rel)+C_d*sinf(phi_rel);
    an_it = 1.0/(1.0 + 4.0*F_tot*sinf(phi_rel)*sinf(phi_rel)/(sigma*c_n)); // normal induction factor

    c_t = C_l*sinf(phi_rel)-C_d*cosf(phi_rel);
    at_it = 1.0/(4.0*F_tot*sinf(phi_rel)*cosf(phi_rel)/(sigma*c_t) - 1.0); // tangential induction factor
    __syncthreads();

    a_tol = sqrtf(powf(an_it-an_0,2.0) + powf(at_it-at_0,2.0));
    an_0 = an_it;
    at_0 = at_it;

    __syncthreads();
    iter_cnt++;
  } //while(a_tol > a_tol_min) --end of iterative process

  *turbineRefAn = fmaxf(fminf(an_it,0.5),0.0);

} // end compute_normalInduction()
