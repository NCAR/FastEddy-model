/* FastEddy®: SRC/EXTENSIONS/URBAN/urban.c 
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
/*Urban parameters*/
int urbanSelector;          /* urban selector: 0=off, 1=on, 2=on with thermal relaxation towards base state */
float cd_build;             /* c_d coefficient (m-1) used by the drag-based building formulation: -c_d|u_i|u_i */
float ct_build;             /* c_t coefficient (s-1) used by the drag-based building formulation: -c_t(rho*theta-rho_b*theta_b) & -c_t(rho-rho_b) */
float *building_mask;       /* Base Address of memory containing building mask 0,1 field */
float delta_aware_bdg;      /* scale-aware correction for building forcing and limiters */
int urban_heatRedis;        /* selector to activate surface heat redistribution */
float *urban_heat_redis;    /* Base Address of memory containing 2d map of heat redistribution coefficient in urban areas */

/*----->>>>> int URBANGetParams();   ----------------------------------------------------------------------
 * Obtain parameters for the URBAN sub-module
*/
int URBANGetParams(){
   int errorCode = URBAN_SUCCESS;

   urbanSelector = 0; // Default to off
   errorCode = queryIntegerParameter("urbanSelector", &urbanSelector, 0, 2, PARAM_OPTIONAL);
   if(urbanSelector > 0){
     cd_build = 100.0; // Default to 100.0
     errorCode = queryFloatParameter("cd_build", &cd_build, 0.0, 1e+8, PARAM_OPTIONAL);
     ct_build = 10.0; // Default to 0.0
     errorCode = queryFloatParameter("ct_build", &ct_build, 0.0, 1e+8, PARAM_OPTIONAL);
     urban_heatRedis = 0; // Default off
     errorCode = queryIntegerParameter("urban_heatRedis", &urban_heatRedis, 0, 1, PARAM_OPTIONAL);
   } // end if(urbanSelector > 0)

   return(errorCode);
} //end URBANGetParams()

/*----->>>>> int URBANPrintParams();   ----------------------------------------------------------------------
* Print parameters for the URBAN sub-module
*/
int URBANPrintParams(){
   int errorCode = URBAN_SUCCESS;
   if(mpi_rank_world == 0){
     printParameter("urbanSelector", "urban selector: 0=off, 1=on, 2=on with thermal relaxation towards base state");	   
     if(urbanSelector > 0){
      printParameter("cd_build", "drag coefficient for buildings when urbanSelector > 0");
      printParameter("ct_build", "temperature and density damping coefficient for buildings when urbanSelector > 0");
      printParameter("urban_heatRedis", "selector to activate surface heat redistribution");
     }
   } //end if(mpi_rank_world == 0)
   return(errorCode);
} //end URBANPrintParams()

/*----->>>>> int URBANInit();   ----------------------------------------------------------------------
 * Used to broadcast parameters, allocate memory, and initialize configuration settings 
 * for the URBAN sub-module.
*/
int URBANInit(){
   int errorCode = URBAN_SUCCESS;
   char fldName[MAX_HC_FLDNAME_LENGTH];

   MPI_Bcast(&urbanSelector, 1, MPI_INT, 0, MPI_COMM_WORLD);
   if(urbanSelector > 0){
     MPI_Bcast(&cd_build, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
     MPI_Bcast(&ct_build, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
     MPI_Bcast(&urban_heatRedis, 1, MPI_INT, 0, MPI_COMM_WORLD);
   }

   if(urbanSelector > 0){
     delta_aware_bdg = 1.0/fmin(pow(d_xi*d_eta*d_zeta,1.0/3.0),1.0);
     printf("urban:delta_aware_bdg = %f\n",delta_aware_bdg);
     building_mask = memAllocateFloat3DField(Nxp, Nyp, Nzp, Nh, "building_mask");
     errorCode = sprintf(&fldName[0],"BuildingMask");
     errorCode = ioRegisterVar(&fldName[0], "float", 4, dims4d, building_mask);
     errorCode = ioAddStandardAttrs("BuildingMask", "-", "Building Mask", NULL);
     printf("urban:Field = %s stored at %p, has been registered with IO.\n",
            &fldName[0],building_mask);
     fflush(stdout);
     if(urban_heatRedis > 0){
       urban_heat_redis = memAllocateFloat2DField(Nxp, Nyp, Nh, "urban_heat_redis");
       errorCode = sprintf(&fldName[0],"UrbanHeatRedis");
       errorCode = ioRegisterVar(&fldName[0], "float", 3, dims2dTD, urban_heat_redis);
       errorCode = ioAddStandardAttrs("UrbanHeatRedis", "-", "Urban Heat Redistribution Coefficient", NULL);
       printf("urban:Field = %s stored at %p, has been registered with IO.\n",
              &fldName[0],urban_heat_redis);
       fflush(stdout);
     }

   } // end of urbanSelector > 0

   return(errorCode);
} //end URBANInit()

/*----->>>>> int URBANCleanup();  ----------------------------------------------------------------------
* Used to free all malloced memory by the URBAN module.
*/
int URBANCleanup(){
   int errorCode = URBAN_SUCCESS;

   if(urbanSelector > 0){
     free(building_mask);
     if(urban_heatRedis > 0){
       free(urban_heat_redis);
     }
   } //end if urbanSelector > 0

   return(errorCode);
}//end URBANCleanup()
