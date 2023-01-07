/* FastEddy®: SRC/HYDRO_CORE/CUDA/cuda_surfaceLayerDevice.cu 
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
/*---SURFACE LAYER*/
__constant__ int surflayerSelector_d;    /*Monin-Obukhov surface layer selector: 0= off, 1= on */
__constant__ float surflayer_z0_d;       /* roughness length (momentum)*/
__constant__ float surflayer_z0t_d;      /* roughness length (temperature)*/
__constant__ float surflayer_wth_d;       /* kinematic sensible heat flux at the surface */
__constant__ float surflayer_tr_d;       /* surface cooling rate K h-1 */
__constant__ float surflayer_wq_d;       /* kinematic latent heat flux at the surface */
__constant__ float surflayer_qr_d;       /* surface water vapor rate (g/kg) h-1 */
__constant__ int surflayer_qskin_input_d;/* selector to use file input (restart) value for qskin under surflayerSelector == 2 */
__constant__ float temp_grnd_d;       /* initial surface temperature */
__constant__ float pres_grnd_d;       /* initial surface pressure */
__constant__ int surflayer_stab_d;    /* exchange coeffcient stability correction selector: 0= on, 1= off */
float* cdFld_d;            /*Base address for momentum exchange coefficient (2d-array)*/ 
float* chFld_d;            /*Base address for sensible heat exchange coefficient (2d-array)*/
float* cqFld_d;            /*Base address for latent heat exchange coefficient (2d-array)*/
float* fricVel_d;                      /*Base address for friction velocity*/ 
float* htFlux_d;                       /*Base address for sensible heat flux*/
float* qFlux_d;                        /*Base address for latent heat flux*/
float* tskin_d;                        /*Base address for skin temperature*/
float* qskin_d;                        /*Base address for skin moisture*/
float* invOblen_d;                     /*Base address for Monin-Obukhov length*/
float* z0m_d;                          /*Base address for roughness length (momentum)*/
float* z0t_d;                          /*Base address for roughness length (temperature)*/
__constant__ int surflayer_idealsine_d;   /*selector for idealized sinusoidal surface heat flux or skin temperature forcing*/
__constant__ float surflayer_ideal_ts_d;  /*start time in seconds for the idealized sinusoidal surface forcing*/
__constant__ float surflayer_ideal_te_d;  /*end time in seconds for the idealized sinusoidal surface forcing*/
__constant__ float surflayer_ideal_amp_d; /*maximum amplitude of the idealized sinusoidal surface forcing*/
__constant__ float surflayer_ideal_qts_d;  /*start time in seconds for the idealized sinusoidal surface forcing of latent heat flux*/
__constant__ float surflayer_ideal_qte_d;  /*end time in seconds for the idealized sinusoidal surface forcing of latent heat flux*/
__constant__ float surflayer_ideal_qamp_d; /*maximum amplitude of the idealized sinusoidal surface forcing of latent heat flux*/

/*----->>>>> int cuda_surfaceLayerDeviceSetup();       -------------------------------------------------------------
 * Used to cudaMalloc and cudaMemcpy parameters and coordinate arrays, and for the SURFLAYER HC-Submodule.
*/
extern "C" int cuda_surfaceLayerDeviceSetup(){
   int errorCode = CUDA_SURFLAYER_SUCCESS;
   int Nelems2d;

   cudaMemcpyToSymbol(surflayerSelector_d, &surflayerSelector, sizeof(int));
   cudaMemcpyToSymbol(surflayer_z0_d, &surflayer_z0, sizeof(float));
   cudaMemcpyToSymbol(surflayer_z0t_d, &surflayer_z0t, sizeof(float));
   cudaMemcpyToSymbol(surflayer_wth_d, &surflayer_wth, sizeof(float));
   cudaMemcpyToSymbol(surflayer_wq_d, &surflayer_wq, sizeof(float));
   cudaMemcpyToSymbol(surflayer_tr_d, &surflayer_tr, sizeof(float));
   cudaMemcpyToSymbol(surflayer_qr_d, &surflayer_qr, sizeof(float));
   cudaMemcpyToSymbol(surflayer_qskin_input_d, &surflayer_qskin_input, sizeof(int));
   cudaMemcpyToSymbol(surflayer_stab_d, &surflayer_stab, sizeof(int));
   cudaMemcpyToSymbol(surflayer_idealsine_d, &surflayer_idealsine, sizeof(int));
   cudaMemcpyToSymbol(surflayer_ideal_ts_d, &surflayer_ideal_ts, sizeof(float));
   cudaMemcpyToSymbol(surflayer_ideal_te_d, &surflayer_ideal_te, sizeof(float));
   cudaMemcpyToSymbol(surflayer_ideal_amp_d, &surflayer_ideal_amp, sizeof(float));
   cudaMemcpyToSymbol(surflayer_ideal_qts_d, &surflayer_ideal_qts, sizeof(float));
   cudaMemcpyToSymbol(surflayer_ideal_qte_d, &surflayer_ideal_qte, sizeof(float));
   cudaMemcpyToSymbol(surflayer_ideal_qamp_d, &surflayer_ideal_qamp, sizeof(float));

   Nelems2d = (Nxp+2*Nh)*(Nyp+2*Nh);  //2-d element count

   fecuda_DeviceMalloc(Nelems2d*sizeof(float), &cdFld_d);
   fecuda_DeviceMalloc(Nelems2d*sizeof(float), &chFld_d);
   fecuda_DeviceMalloc(Nelems2d*sizeof(float), &cqFld_d);
   fecuda_DeviceMalloc(Nelems2d*sizeof(float), &fricVel_d);
   fecuda_DeviceMalloc(Nelems2d*sizeof(float), &htFlux_d);
   fecuda_DeviceMalloc(Nelems2d*sizeof(float), &qFlux_d);
   fecuda_DeviceMalloc(Nelems2d*sizeof(float), &tskin_d);
   fecuda_DeviceMalloc(Nelems2d*sizeof(float), &qskin_d);
   fecuda_DeviceMalloc(Nelems2d*sizeof(float), &invOblen_d);
   fecuda_DeviceMalloc(Nelems2d*sizeof(float), &z0m_d);
   fecuda_DeviceMalloc(Nelems2d*sizeof(float), &z0t_d);

   /* Done */
   return(errorCode);
} //end cuda_surfaceLayerDeviceSetup

/*----->>>>> extern "C" int cuda_surfaceLayerDeviceCleanup();  -------------------------------------------------------
Used to free all malloced memory by the SURFLAYER HC-Submodule.
*/
extern "C" int cuda_surfaceLayerDeviceCleanup(){
   int errorCode = CUDA_SURFLAYER_SUCCESS;

     cudaFree(cdFld_d);
     cudaFree(chFld_d);
     cudaFree(cqFld_d);
     cudaFree(fricVel_d);
     cudaFree(htFlux_d);
     cudaFree(qFlux_d);
     cudaFree(tskin_d);
     cudaFree(qskin_d);
     cudaFree(invOblen_d);
     cudaFree(z0m_d);
     cudaFree(z0t_d);

   return(errorCode);
}//end cuda_surfaceLayerDeviceCleanup()

/*----->>>>> __device__ void cudaDevice_SurfaceLayerLSMdry();  --------------------------------------------------
*/ 
__device__ void cudaDevice_SurfaceLayerLSMdry(float simTime, int simTime_it, int simTime_itRestart,
                                              float dt, int timeStage, int maxSteps, int ijk,
                                              float* u, float* v, float* rho, float* theta,
                                              float* cd_iter, float* ch_iter, float* fricVel, float* htFlux, float* tskin,
                                              float* z0m, float* z0t, float* J33_d){

   float z0,z1,z1oz0,z1ozt0;
   float U1,u1,v1,th1,th0;
   float pi = acosf(-1.0);
   int temp_freq;
   float temp_freq_f;
   int tskin_update;
   float tsk_p,tsk_c;
   float tsk_inc,simTimePrev;
   float z0temp;

   temp_freq = roundf(10.0/dt); // make it so temp_freq is ~ 10 seconds
   temp_freq_f = __int2float_rn(temp_freq); // make it so temp_freq is ~ 10 seconds

   z0 = *z0m;
   z1 = 0.5/(dZi_d*J33_d[ijk]); // = zPos[ij,k = surface cell];
   z1oz0 = (z1+z0)/z0;
   z0temp = *z0t;
   z1ozt0 = (z1+z0temp)/z0temp;

   u1 = *u/ *rho;
   v1 = *v/ *rho;
   U1 = sqrtf(powf(u1,2.0)+powf(v1,2.0));
   th1 = (*theta)/(*rho);

   // cd, ch, tskin and htFlux initialization/restart
   if((simTime_it==simTime_itRestart) && (simTime_it==0)){ // first time step or pseudo-restart from modified initial condition
     *cd_iter = powf(kappa_d,2.0)/powf(logf(z1oz0),2.0); //  move this initialization to the CPU
     *ch_iter = powf(kappa_d,2.0)/powf(logf(z1ozt0),2.0); // move this initialization to the CPU
   } else if((simTime_it==simTime_itRestart) && (simTime_it!=0)){ // restart
     *cd_iter = powf(*fricVel,2.0)/(powf(U1,2.0));
     th0 = *tskin*powf((refPressure_d/pres_grnd_d),R_cp_d);
     *ch_iter = *htFlux/(U1*(th0-th1));
   } // otherwise uses values from previous time step

   if (surflayerSelector_d==1){ // heat flux formulation
     if (surflayer_idealsine_d==1){ // idealized time evolution (sine function)
       if ((simTime>surflayer_ideal_ts_d) && (simTime<=surflayer_ideal_te_d)){
         *htFlux = surflayer_wth_d + surflayer_ideal_amp_d*sin(pi*(simTime-surflayer_ideal_ts_d)/(surflayer_ideal_te_d-surflayer_ideal_ts_d));
       }else{
         *htFlux = surflayer_wth_d;
       }
     }else{
       // reuse *htFlux array values
     }
   }else if (surflayerSelector_d==2){ // skin temperature formulation
     tskin_update = simTime_it%temp_freq;

     if ((tskin_update==0)&&(timeStage==maxSteps)){ // update skin temperature
       if (surflayer_idealsine_d==1){ // idealized time evolution (sine function)
         if ((simTime>surflayer_ideal_ts_d) && (simTime<surflayer_ideal_te_d+dt*temp_freq_f)){
           // temperature
           tsk_p = *tskin;
           simTimePrev = simTime-temp_freq_f*dt;
           simTimePrev = fmaxf(simTimePrev,surflayer_ideal_ts_d);
           tsk_inc = surflayer_ideal_amp_d*(sin(pi*(simTime-surflayer_ideal_ts_d)/(surflayer_ideal_te_d-surflayer_ideal_ts_d))
                                           -sin(pi*(simTimePrev-surflayer_ideal_ts_d)/(surflayer_ideal_te_d-surflayer_ideal_ts_d)));
           tsk_c = tsk_p + tsk_inc;
           *tskin = tsk_c;
         }else{
           // temperature
           tsk_p = *tskin;
           tsk_c = tsk_p+surflayer_tr_d*dt*temp_freq_f/3600.0; // surflayer_tr_d < 0 is cooling
           *tskin = tsk_c;
         }
       }else{ // linear evolution
         // temperature
         tsk_p = *tskin;
         tsk_c = tsk_p+surflayer_tr_d*dt*temp_freq_f/3600.0; // surflayer_tr_d < 0 is cooling
         *tskin = tsk_c;
       }
     }else{ // keep skin temperature from previous time step
       tsk_c = *tskin;
     }
     th0 = tsk_c*powf((refPressure_d/pres_grnd_d),R_cp_d);
     *htFlux = *ch_iter*U1*(th0-th1);
   }//end if (surflayerSelector_d==1), elseif (surflayerSelector_d==2) 

} //end cudaDevice_SurfaceLayerLSMdry(...

/*----->>>>> __device__ void cudaDevice_SurfaceLayerLSMmoist();  --------------------------------------------------
*/ 
__device__ void cudaDevice_SurfaceLayerLSMmoist(float simTime, int simTime_it, int simTime_itRestart,
                                                float dt, int timeStage, int maxSteps, int ijk,
                                                float* u, float* v, float* rho, float* theta, float* qv,
                                                float* cd_iter, float* ch_iter, float* cq_iter, float* fricVel,
                                                float* htFlux, float* tskin, float* qFlux, float* qskin,
                                                float* z0m, float* z0t, float* J33_d){

   float z0,z1,z1oz0,z1ozt0;
   float U1,u1,v1,th1,th0;
   float pi = acosf(-1.0);
   int temp_freq;
   float temp_freq_f;
   int tskin_update;
   float tsk_p,tsk_c;
   float tsk_inc,simTimePrev;
   float z0temp;
   float qsk_p,qsk_c,qsk_inc;
   float q0,q1,qsk_input;

   temp_freq = roundf(10.0/dt); // make it so temp_freq is ~ 10 seconds
   temp_freq_f = __int2float_rn(temp_freq); // make it so temp_freq is ~ 10 seconds

   z0 = *z0m;
   z1 = 0.5/(dZi_d*J33_d[ijk]); // = zPos[ij,k=surface cell];
   z1oz0 = (z1+z0)/z0;
   z0temp = *z0t;
   z1ozt0 = (z1+z0temp)/z0temp;

   u1 = *u/ *rho;
   v1 = *v/ *rho;
   U1 = sqrtf(powf(u1,2.0)+powf(v1,2.0));
   th1 = (*theta)/(*rho);
   q1 = (*qv)/(*rho);
   if (surflayer_qskin_input_d == 1){
     qsk_input = *qskin;
   }

   // cd, ch, cq, tskin, htFlux, qskin, qFlux initialization/restart
   if((simTime_it==simTime_itRestart) && (simTime_it==0)){ // first time step or pseudo-restart from modified initial condition
     *cd_iter = powf(kappa_d,2.0)/powf(logf(z1oz0),2.0); // move this initialization to the CPU
     *ch_iter = powf(kappa_d,2.0)/powf(logf(z1ozt0),2.0); // move this initialization to the CPU
     *cq_iter = powf(kappa_d,2.0)/powf(logf(z1ozt0),2.0); // move this initialization to the CPU
   } else if((simTime_it==simTime_itRestart) && (simTime_it!=0)){ // restart
     *cd_iter = powf(*fricVel,2.0)/(powf(U1,2.0));
     th0 = *tskin*powf((refPressure_d/pres_grnd_d),R_cp_d);
     *ch_iter = *htFlux/(U1*(th0-th1));
     q0 = *qskin;
     *cq_iter = *qFlux/(U1*(q0-q1));
   } // otherwise uses values from previous time step

   if (surflayerSelector_d==1){ // heat flux formulation
     if (surflayer_idealsine_d==1){ // idealized time evolution (sine function)
       if ((simTime>surflayer_ideal_ts_d) && (simTime<=surflayer_ideal_te_d)){
         *htFlux = surflayer_wth_d + surflayer_ideal_amp_d*sin(pi*(simTime-surflayer_ideal_ts_d)/(surflayer_ideal_te_d-surflayer_ideal_ts_d));
       }else{
         *htFlux = surflayer_wth_d;
       }
       if ((simTime>surflayer_ideal_qts_d) && (simTime<=surflayer_ideal_qte_d)){
         *qFlux = surflayer_ideal_qamp_d*sin(pi*(simTime-surflayer_ideal_qts_d)/(surflayer_ideal_qte_d-surflayer_ideal_qts_d));
       }else{
         *qFlux = surflayer_wq_d;
       }
     }else{
       // reuse *htFlux array values
       // reuse *qFlux array values
     }
   } else if (surflayerSelector_d==2){ // skin temperature formulation
     tskin_update = simTime_it%temp_freq;

     if ((tskin_update==0)&&(timeStage==maxSteps)){ // update skin temperature
       if (surflayer_idealsine_d==1){ // idealized time evolution (sine function)
         if ((simTime>surflayer_ideal_ts_d) && (simTime<surflayer_ideal_te_d+dt*temp_freq_f)){
           // temperature
           tsk_p = *tskin;
           simTimePrev = simTime-temp_freq_f*dt;
           simTimePrev = fmaxf(simTimePrev,surflayer_ideal_ts_d);
           tsk_inc = surflayer_ideal_amp_d*(sin(pi*(simTime-surflayer_ideal_ts_d)/(surflayer_ideal_te_d-surflayer_ideal_ts_d))
                                           -sin(pi*(simTimePrev-surflayer_ideal_ts_d)/(surflayer_ideal_te_d-surflayer_ideal_ts_d)));
           tsk_c = tsk_p + tsk_inc;
           *tskin = tsk_c;
           // moisture
           qsk_p = *qskin;
           qsk_inc = surflayer_ideal_qamp_d*(sin(pi*(simTime-surflayer_ideal_qts_d)/(surflayer_ideal_qte_d-surflayer_ideal_qts_d))
                                            -sin(pi*(simTimePrev-surflayer_ideal_qts_d)/(surflayer_ideal_qte_d-surflayer_ideal_qts_d)));
           qsk_c = qsk_p + qsk_inc;
           *qskin = qsk_c;
         }else{
           // temperature
           tsk_p = *tskin;
           tsk_c = tsk_p+surflayer_tr_d*dt*temp_freq_f/3600.0; // surflayer_tr_d < 0 is cooling
           *tskin = tsk_c;
           // moisture
           qsk_p = *qskin;
           qsk_c = qsk_p+surflayer_qr_d*dt*temp_freq_f/3600.0;
           *qskin = qsk_c;
         }
       }else{ // linear evolution
         // temperature
         tsk_p = *tskin;
         tsk_c = tsk_p+surflayer_tr_d*dt*temp_freq_f/3600.0; // surflayer_tr_d < 0 is cooling
         *tskin = tsk_c;
         // moisture
         qsk_p = *qskin;
         qsk_c = qsk_p+surflayer_qr_d*dt*temp_freq_f/3600.0;
         *qskin = qsk_c;
       }
     }else{ // keep skin temperature from previous time step
       tsk_c = *tskin;
       qsk_c = *qskin;
     }
     th0 = tsk_c*powf((refPressure_d/pres_grnd_d),R_cp_d);
     *htFlux = *ch_iter*U1*(th0-th1);
     q0 = qsk_c;
     *qFlux = *cq_iter*U1*(q0-q1); // M factor here as well
     if (surflayer_qskin_input_d == 1){
       *qskin = qsk_input;
     }
   }//end if (surflayerSelector_d==1), elseif (surflayerSelector_d==2) 

} //end cudaDevice_SurfaceLayerLSMmoist(...

/*----->>>>> __device__ void cudaDevice_SurfaceLayerMOSTdry();  --------------------------------------------------
*/ 
__device__ void cudaDevice_SurfaceLayerMOSTdry(int ijk, float* u, float* v, float* rho, float* theta,
                                               float* tau31, float* tau32, float* tauTH3,
                                               float* cd_iter, float* ch_iter, float* fricVel,
                                               float* htFlux, float* tskin, float* invOblen, float* z0m, float* z0t, float* J33_d){

   float cd_i,ch_i,cd_0;
   float z0,z1,z1oz0,z1ozt0;
   float U1,u1,v1,th1,th0;
   float tauxz,tauyz,tauthz;
   float zol;
   float xi;
   float psi_m;
   float psi_h;
   float beta = 5.0;
   float pi = acosf(-1.0);
   float ol_lim = 1.0; // limit Obukhov length (in meters)
   int it_n;
   float z0temp;
   float it_max;
   if (surflayer_stab_d==0){
     it_max = 5;
   }else{
     it_max = 1;
   }

   z0 = *z0m;
   z1 = 0.5/(dZi_d*J33_d[ijk]); // = zPos[ij,k=surface cell];
   z1oz0 = (z1+z0)/z0;
   z0temp = *z0t;
   z1ozt0 = (z1+z0temp)/z0temp;

   u1 = *u/ *rho;
   v1 = *v/ *rho;
   U1 = sqrtf(powf(u1,2.0)+powf(v1,2.0));
   th1 = (*theta)/(*rho);
   th0 = (*tskin)*powf((refPressure_d/pres_grnd_d),R_cp_d);
   cd_0 = *cd_iter;

   it_n = 0;
   // iterative solve for exchange coefficients
   do {

     tauxz = -cd_0*U1*u1;
     tauyz = -cd_0*U1*v1;
     *fricVel = powf(powf(tauxz,2.0)+powf(tauyz,2.0),0.25);
     it_n = it_n + 1;

     if (surflayer_stab_d==0){
       // calculate inverse Obukhov length
       if (*fricVel > 0.0){
          *invOblen = -(kappa_d*accel_g_d*(*htFlux))/(powf((*fricVel),3.0)*th1);
          *invOblen = fmaxf(fminf(*invOblen,ol_lim),-ol_lim);
       }else{            //ust < 0.0...
          *invOblen = -ol_lim;  //Technically this would be infinite, but we will use iol_fc...
       }
     }else{
       *invOblen = 0.0;
     }
     zol = (*invOblen)*(z1+z0);

     if (zol < 0.0) { // convective ABL
       xi = powf(1.0-16.0*zol,0.25);
       psi_m = logf(0.5*(1.0+powf(xi,2.0))*powf(0.5*(1.0+xi),2.0)) - 2.0*atanf(xi)+0.5*pi;
       psi_h = 2.0*logf(0.5*(1.0+powf(xi,2.0)));
     } else { // stable ABL
       psi_m = -beta*zol;
       psi_h = -beta*zol;
     }

     cd_i = powf(kappa_d,2.0)/powf(logf(z1oz0)-psi_m,2.0);
     ch_i = powf(kappa_d,2.0)/((logf(z1ozt0)-psi_m)*(logf(z1ozt0)-psi_h));

     if (surflayerSelector_d > 1){
        *htFlux = ch_i*U1*(th0-th1);
     }//endif surflayerSelector_d==2

     cd_0 = cd_i;

   } while(it_n<=it_max);
   // end of iterative process

   *cd_iter = cd_i;
   *ch_iter = ch_i;
   tauxz = -cd_i*sqrtf(powf(*u/ *rho,2.0)+powf(*v/ *rho,2.0))*(*u);
   tauyz = -cd_i*sqrtf(powf(*u/ *rho,2.0)+powf(*v/ *rho,2.0))*(*v);
   *tau31 = tauxz;
   *tau32 = tauyz;
   *fricVel = powf(powf(tauxz,2.0)+powf(tauyz,2.0),0.25);
   tauthz = (*htFlux)*(*rho);
   *tauTH3 = tauthz;
   *invOblen = -(kappa_d*accel_g_d*(*htFlux))/(powf((*fricVel),3.0)*th1);

} //end cudaDevice_SurfaceLayerMOSTdry(...

/*----->>>>> __device__ void cudaDevice_SurfaceLayerMOSTmoist();  --------------------------------------------------
*/
__device__ void cudaDevice_SurfaceLayerMOSTmoist(int ijk, float* u, float* v, float* rho, float* theta, float* qv,
                                                 float* tau31, float* tau32, float* tauTH3, float* tauQ3,
                                                 float* cd_iter, float* ch_iter, float* cq_iter, float* fricVel,
                                                 float* htFlux, float* tskin, float* qFlux, float* qskin,
                                                 float* invOblen, float* z0m, float* z0t, float* J33_d){

   float cd_i,ch_i,cd_0;
   float z0,z1,z1oz0,z1ozt0;
   float U1,u1,v1,th1,th0;
   float tauxz,tauyz,tauthz;
   float zol;
   float xi;
   float psi_m;
   float psi_h;
   float beta = 5.0;
   float pi = acosf(-1.0);
   float ol_lim = 1.0; // limit Obukhov length (in meters)
   int it_n;
   float z0temp;
   float q0,q1,cq_i,psi_q,tauqz;
   int it_max;
   if (surflayer_stab_d==0){
     it_max = 5;
   }else{
     it_max = 1;
   }

   z0 = *z0m;
   z1 = 0.5/(dZi_d*J33_d[ijk]); // = zPos[ij,k=surface cell];
   z1oz0 = (z1+z0)/z0;
   z0temp = *z0t;
   z1ozt0 = (z1+z0temp)/z0temp;

   u1 = *u/ *rho;
   v1 = *v/ *rho;
   U1 = sqrtf(powf(u1,2.0)+powf(v1,2.0));
   th1 = (*theta)/(*rho);
   q1 = (*qv)/(*rho);
   th0 = (*tskin)*powf((refPressure_d/pres_grnd_d),R_cp_d);
   q0 = *qskin;
   cd_0 = *cd_iter;

   it_n = 0;
   // iterative solve for exchange coefficients
   do {

     tauxz = -cd_0*U1*u1;
     tauyz = -cd_0*U1*v1;
     *fricVel = powf(powf(tauxz,2.0)+powf(tauyz,2.0),0.25);
     it_n = it_n + 1;

     if (surflayer_stab_d==0){
       // calculate inverse Obukhov length
       if (*fricVel > 0.0){
          *invOblen = -(kappa_d*accel_g_d*(*htFlux))/(powf((*fricVel),3.0)*th1);
          *invOblen = fmaxf(fminf(*invOblen,ol_lim),-ol_lim);
       }else{            //ust < 0.0...
          *invOblen = -ol_lim;  //Technically this would be infinite, but we will use iol_fc...
       }
     }else{
       *invOblen = 0.0;
     }
     zol = (*invOblen)*(z1+z0);

     if (zol < 0.0) { // convective ABL
       xi = powf(1.0-16.0*zol,0.25);
       psi_m = logf(0.5*(1.0+powf(xi,2.0))*powf(0.5*(1.0+xi),2.0)) - 2.0*atanf(xi)+0.5*pi;
       psi_h = 2.0*logf(0.5*(1.0+powf(xi,2.0)));
       psi_q = psi_h;
     } else { // stable ABL
       psi_m = -beta*zol;
       psi_h = -beta*zol;
       psi_q = psi_h;
     }

     cd_i = powf(kappa_d,2.0)/powf(logf(z1oz0)-psi_m,2.0);
     ch_i = powf(kappa_d,2.0)/((logf(z1ozt0)-psi_m)*(logf(z1ozt0)-psi_h));
     cq_i = powf(kappa_d,2.0)/((logf(z1ozt0)-psi_m)*(logf(z1ozt0)-psi_q));

     if (surflayerSelector_d > 1){
        *htFlux = ch_i*U1*(th0-th1);
        *qFlux = cq_i*U1*(q0-q1);
     }//endif surflayerSelector_d==2

     cd_0 = cd_i;

   } while(it_n<=it_max);
   // end of iterative process

   *cd_iter = cd_i;
   *ch_iter = ch_i;
   *cq_iter = cq_i;
   tauxz = -cd_i*sqrtf(powf(*u/ *rho,2.0)+powf(*v/ *rho,2.0))*(*u);
   tauyz = -cd_i*sqrtf(powf(*u/ *rho,2.0)+powf(*v/ *rho,2.0))*(*v);
   *tau31 = tauxz;
   *tau32 = tauyz;
   *fricVel = powf(powf(tauxz,2.0)+powf(tauyz,2.0),0.25);
   tauthz = (*htFlux)*(*rho);
   *tauTH3 = tauthz;
   tauqz = (*qFlux)*(*rho); // specified qflux or delta-qv-based flux assumes qv units of g/kg
   *tauQ3 = tauqz;
   *invOblen = -(kappa_d*accel_g_d*(*htFlux))/(powf((*fricVel),3.0)*th1);

} //end cudaDevice_SurfaceLayerMOSTmoist(...
