/* FastEddy®: SRC/IO/io_binary.c 
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
/*----->>>>> int ioWriteBinaryoutFileSingleTime();  ---------------------------------------------------------------
 * Used to have N-ranks write N-binary files of registered variables for a single timestep.
*/
#ifdef GAD_EXT
int ioWriteBinaryoutFileSingleTime(int tstep, int Nx, int Ny, int Nz, int Nh, int Nturbines){
#else
int ioWriteBinaryoutFileSingleTime(int tstep, int Nx, int Ny, int Nz, int Nh){
#endif
   int errorCode = IO_SUCCESS;
   FILE *output_ptr;

   /* build the subString tag */
   sprintf(outSubString, "_rank_%d.%d",mpi_rank_world,tstep);
   /* concatenate the fileName components */
   sprintf(outFileName, "%s%s%s",outPath,outFileBase,outSubString);
   /*Open the output file*/
   output_ptr = fopen(outFileName,"wb");
   /*Write the IO-registered variables to the output file*/
#ifdef GAD_EXT
   errorCode = ioPutBinaryoutFileVars(output_ptr, Nx, Ny, Nz, Nh, Nturbines);
#else
   errorCode = ioPutBinaryoutFileVars(output_ptr, Nx, Ny, Nz, Nh);
#endif
   /*Close the output file*/
   fclose(output_ptr);
   return(errorCode);
} //end ioWriteBinaryoutFileSingleTime

/*----->>>>> int ioPutBinaryoutFileVars();    ---------------------------------------------------------------------
* Used to put(write) all variables in the register list in(to) the Binary file. 
*/
#ifdef GAD_EXT
int ioPutBinaryoutFileVars(FILE *outptr, int Nx, int Ny, int Nz, int Nh, int Nturbines){
#else
int ioPutBinaryoutFileVars(FILE *outptr, int Nx, int Ny, int Nz, int Nh){
#endif
   int errorCode = IO_SUCCESS;
   ioVar_t *ptr;
   ioVar_t *rhoptr;
   float * field;
   int binary_nDims;
   int numElems;
   int nameLen;
   int extent;
   int i,j,k;
   int ijk;
   int rhoDivideSwitch = 0;
   int verbose_log = 0;
   float * rhofield;
   int *intField;
   int typeLen;

   /* For each entry in the ioVarsList, "put" the var */
   ptr = getFirstVarFromList();
   while(ptr != NULL){
      if( (!strcmp(ptr->name,"u"))||
          (!strcmp(ptr->name,"v"))||
          (!strcmp(ptr->name,"w"))||
          (!strcmp(ptr->name,"theta")) ||
          (!strcmp(ptr->name,"TKE_0")) ||
          (!strcmp(ptr->name,"TKE_1")) ||
          (!strcmp(ptr->name,"qv")) ||
          (!strcmp(ptr->name,"ql")) ||
          (!strcmp(ptr->name,"qr")) ){
          rhoDivideSwitch=1;
          rhoptr = getNamedVarFromList("rho");
          if(rhoptr != NULL){
            rhofield = (float *) rhoptr->varMemAddress;
            if(verbose_log == 1){
              printf("ioPutNetCDFoutFileVars: rhofield identified at 0x%p !\n", (void *) rhofield);
              fflush(stdout);
            }
          }else{
            printf("ioPutNetCDFoutFileVars: Couldn't find rho!!! Catastrophinc Error!!!!!!!!!!!!!!!!!\n");
            fflush(stdout);
          } //end if
      }else{ // do not divide by rho
        rhoDivideSwitch=0;
      } //end if name is u,v,w, or theta
      /*Write the output variable name*/ 
      nameLen=strlen(ptr->name);
      fwrite(&nameLen,sizeof(int),1,outptr); 
      fwrite(ptr->name,nameLen*sizeof(char),1,outptr);
      /*Write the output variable type*/
      typeLen=strlen(ptr->type);
      fwrite(&typeLen,sizeof(int),1,outptr);
      fwrite(ptr->type,typeLen*sizeof(char),1,outptr); 
      //If this registered field is of type float
      if (!strcmp(ptr->type,"float")){
         field = (float *) ptr->varMemAddress;
         if((ptr->nDims > 2)&&(ptr->dimids[1] == 1)){  //Should be a 4D with time,z,y,x...
           if(rhoDivideSwitch==1){
             for(i=0; i < Nx+2*Nh; i++){
               for(j=0; j < Ny+2*Nh; j++){
                 for(k=0; k < Nz+2*Nh; k++){
                   ijk = (i)*(Ny+2*Nh)*(Nz+2*Nh)+(j)*(Nz+2*Nh)+(k); //Account for halo presence in the raw field
                   field[ijk] = field[ijk]/rhofield[ijk]; //in-place divide by rho to write strictly non-flux-conservative field (GPU still holds flux-conservative form)
                 } //end for(k...
               } // end for(j...
             } // end for(i...
           }//end if rhoDivideSwitch
           numElems=(Nx+2*Nh)*(Ny+2*Nh)*(Nz+2*Nh);
           binary_nDims=3;
           fwrite(&binary_nDims,sizeof(int),1,outptr);
           extent=Nx+2*Nh; 
           fwrite(&extent,sizeof(int),1,outptr); 
           extent=Ny+2*Nh; 
           fwrite(&extent,sizeof(int),1,outptr); 
           extent=Nz+2*Nh; 
           fwrite(&extent,sizeof(int),1,outptr); 
           fwrite(field,numElems*sizeof(float),1,outptr);
         }else if((ptr->nDims == 3)&&(ptr->dimids[1] == 2)){
           numElems=(Nx+2*Nh)*(Ny+2*Nh);
           binary_nDims=2;
           fwrite(&binary_nDims,sizeof(int),1,outptr);
           extent=Nx+2*Nh; 
           fwrite(&extent,sizeof(int),1,outptr);
           extent=Ny+2*Nh; 
           fwrite(&extent,sizeof(int),1,outptr); 
           fwrite(field,numElems*sizeof(float),1,outptr);
         }else if((ptr->nDims == 2)&&(ptr->dimids[1] == 2)){
           numElems=(Nx+2*Nh)*(Ny+2*Nh);
           binary_nDims=2;
           fwrite(&binary_nDims,sizeof(int),1,outptr);
           extent=Nx+2*Nh; 
           fwrite(&extent,sizeof(int),1,outptr);
           extent=Ny+2*Nh; 
           fwrite(&extent,sizeof(int),1,outptr);
           fwrite(field,numElems*sizeof(float),1,outptr);
#ifdef GAD_EXT
         }else if((ptr->nDims == 2)&&(ptr->dimids[1] == 4)){
           numElems=(Nturbines);
           binary_nDims=1;
           fwrite(&binary_nDims,sizeof(int),1,outptr);
           extent=Nturbines;
           fwrite(&extent,sizeof(int),1,outptr);
           fwrite(field,numElems*sizeof(float),1,outptr);
#endif
	 }else if((ptr->nDims == 1)&&(ptr->dimids[0] == 0)){
           numElems=1;
           binary_nDims=1;
           fwrite(&binary_nDims,sizeof(int),1,outptr);
           extent=1;
           fwrite(&extent,sizeof(int),1,outptr);
           fwrite(field,numElems*sizeof(float),1,outptr);
         }// end if ndims > 2  && ptr->dimids[1] == 1 -else
      }else if (!strcmp(ptr->type,"int")){
         intField = (int *) ptr->varMemAddress;
         if((ptr->nDims == 1)&&(ptr->dimids[0] == 0)){
           numElems=1;
           binary_nDims=1;
           fwrite(&binary_nDims,sizeof(int),1,outptr);
           extent=1;
           fwrite(&extent,sizeof(int),1,outptr);
           fwrite(intField,numElems*sizeof(int),1,outptr);
#ifdef GAD_EXT
	 }else if((ptr->nDims == 2)&&(ptr->dimids[1] == 4)){
           numElems=(Nturbines);
           binary_nDims=1;
           fwrite(&binary_nDims,sizeof(int),1,outptr);
           extent=Nturbines;
           fwrite(&extent,sizeof(int),1,outptr);
           fwrite(intField,numElems*sizeof(int),1,outptr);
#endif
         }// end if ndims == 1  && ptr->dimids[0] == 0  
      }// if (ptr.type == "float") else if (ptr.type == "int")...
      ptr = ptr->next;
   } //end while ptr != NULL

   return(errorCode);
} //ioPutBinaryoutFileVars()

/*----->>>>> int ioWriteBinaryTowerFileSingleBatch();  ---------------------------------------------------------------
 * Used to have N-ranks write N-binary files of virtual tower data structures for a batch of timesteps.
 */
int ioWriteBinaryTowerFileSingleBatch(int tstep, int batchSize, int Nz, float *batchTimes, float *towersData, float *towersSurfData, 
		                      int *towerIDs, int rank_nTowers, int towerInstanceSize, int towerSurfInstanceSize){
   int errorCode = IO_SUCCESS;
   FILE *output_ptr;
   char towerSubString[64];
   char towerFileName[256];
   int itower;

   for(itower = 0; itower < rank_nTowers; itower++){
     //--------------Tower profile-variables file 
     /* build the subString tag */
     sprintf(towerSubString, "tower_%d.%d",towerIDs[itower],tstep-batchSize);
     /* concatenate the fileName components */
     sprintf(towerFileName, "%s%s",towerPath,towerSubString);
     /*Open the output file*/
     output_ptr = fopen(towerFileName,"wb");
     /*Write the batch of tower instances to the output file*/
     fwrite(&towerInstanceSize,sizeof(int),1,output_ptr);
     fwrite(&batchSize,sizeof(int),1,output_ptr);
     fwrite(&batchTimes[0],batchSize*sizeof(float),1,output_ptr);
     fwrite(&towersData[itower*batchSize*towerInstanceSize],batchSize*towerInstanceSize*sizeof(float),1,output_ptr);
     /*Close the output file*/
     fclose(output_ptr);

     //--------------Tower surface-variables file 
     /* build the subString tag */
     sprintf(towerSubString, "tower_sv_%d.%d",towerIDs[itower],tstep-batchSize);
     /* concatenate the fileName components */
     sprintf(towerFileName, "%s%s",towerPath,towerSubString);
     /*Open the output file*/
     output_ptr = fopen(towerFileName,"wb");
     /*Write the batch of tower instances to the output file*/
     fwrite(&towerSurfInstanceSize,sizeof(int),1,output_ptr);
     fwrite(&batchSize,sizeof(int),1,output_ptr);
     fwrite(&batchTimes[0],batchSize*sizeof(float),1,output_ptr);
     fwrite(&towersSurfData[itower*batchSize*towerSurfInstanceSize],batchSize*towerSurfInstanceSize*sizeof(float),1,output_ptr);
     /*Close the output file*/
     fclose(output_ptr);
   } //end for itower


   return(errorCode);
} //end ioWriteBinaryTowerFileSingleBatch()

/*----->>>>> int ioWriteBinaryTowerInitialFile();  ---------------------------------------------------------------
 * Used to have N-ranks write N-binary files of virtual tower data structures for the initial timestep and time-independent fields.
 */
int ioWriteBinaryTowerInitialFile(float dt, int itStart, int Nx, int Ny, int Nz, int Nh, 
		                  float *towersData, float *towersSurfData,
                                  int *towerIDs, int rank_nTowers, int *tower_iInds, int *tower_jInds, 
				  int coordType, float *tower_xOffs, float *tower_yOffs, double *tower_LonOffs, double *tower_LatOffs, 
				  int batchSize, int towerInstanceSize, int towerSurfInstanceSize, 
				  float *zCoords, float *yCoords, float *xCoords, float *topoFld, int surflayer_offshore, float *seamask){
    int errorCode = IO_SUCCESS;
    FILE *output_ptr;
    char towerSubString[64];
    char towerFileName[256];
    int itower;
    int i,j,k,ijk,ij;
    int iStride,jStride,kStride;
    int tmpOne = 1;
    float timeStart;
    double tmpDbleWE;
    double tmpDbleSN;

    iStride = (Ny+2*Nh)*(Nz+2*Nh);
    jStride = (Nz+2*Nh);
    kStride = 1;

    timeStart = itStart*dt;

    for(itower = 0; itower < rank_nTowers; itower++){
       i = tower_iInds[itower];
       j = tower_jInds[itower];
       k = Nh;
       ijk = i*iStride + j*jStride + k*kStride;
       ij = i*(Ny+2*Nh) + j; 
       //--------------Tower profile-variables  
       /* build the subString tag */
       sprintf(towerSubString, "tower_ic_%d.%d",towerIDs[itower],itStart);
       /* concatenate the fileName components */
       sprintf(towerFileName, "%s%s",towerPath,towerSubString);
       /*Open the output file*/
       output_ptr = fopen(towerFileName,"wb");
     
       //--------------Tower time-independent variables  
       fwrite(&Nz,sizeof(int),1,output_ptr);
       fwrite(&zCoords[ijk],Nz*sizeof(float),1,output_ptr);
       fwrite(&yCoords[ijk],sizeof(float),1,output_ptr);
       fwrite(&xCoords[ijk],sizeof(float),1,output_ptr);
       fwrite(&topoFld[ij],sizeof(float),1,output_ptr);
       if(surflayer_offshore > 0){
          fwrite(&seamask[ij],sizeof(float),1,output_ptr);
       }//end if surfacelayer_offshore
       if(coordType == 0){
         tmpDbleSN = tower_LatOffs[itower];
         tmpDbleWE = tower_LonOffs[itower];
       }else{
         tmpDbleSN = (double) tower_yOffs[itower];
         tmpDbleWE = (double) tower_xOffs[itower];
       }
       fwrite(&tmpDbleSN,sizeof(double),1,output_ptr);
       fwrite(&tmpDbleWE,sizeof(double),1,output_ptr);

       /*Write the batch of tower instances to the output file*/
       fwrite(&towerInstanceSize,sizeof(int),1,output_ptr);
       fwrite(&tmpOne,sizeof(int),1,output_ptr);
       fwrite(&timeStart,sizeof(float),1,output_ptr);
       fwrite(&towersData[itower*batchSize*towerInstanceSize],1*towerInstanceSize*sizeof(float),1,output_ptr);
       
       //--------------Tower surface-variables  
       fwrite(&towerSurfInstanceSize,sizeof(int),1,output_ptr);
       fwrite(&tmpOne,sizeof(int),1,output_ptr);
       fwrite(&timeStart,sizeof(float),1,output_ptr);
       fwrite(&towersSurfData[itower*batchSize*towerSurfInstanceSize],1*towerSurfInstanceSize*sizeof(float),1,output_ptr);
     
       /*Close the output file*/
       fclose(output_ptr);
    } //end for itower

     
    return(errorCode);
} //end ioWriteBinaryTowerInitialFile()
