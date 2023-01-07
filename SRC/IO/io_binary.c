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
int ioWriteBinaryoutFileSingleTime(int tstep, int Nx, int Ny, int Nz, int Nh){
   int errorCode = IO_SUCCESS;
   FILE *output_ptr;

   /* build the subString tag */
   sprintf(outSubString, "_rank_%d.%d",mpi_rank_world,tstep);
   /* concatenate the fileName components */
   sprintf(outFileName, "%s%s%s",outPath,outFileBase,outSubString);
   /*Open the output file*/
   output_ptr = fopen(outFileName,"wb");
   /*Write the IO-registered variables to the output file*/
   errorCode = ioPutBinaryoutFileVars(output_ptr, Nx, Ny, Nz, Nh);
   /*Close the output file*/
   fclose(output_ptr);
   return(errorCode);
} //end ioWriteBinaryoutFileSingleTime

/*----->>>>> int ioPutBinaryoutFileVars();    ---------------------------------------------------------------------
* Used to put(write) all variables in the register list in(to) the Binary file. 
*/
int ioPutBinaryoutFileVars(FILE *outptr, int Nx, int Ny, int Nz, int Nh){
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
      //If this registered field is of type float
      if (!strcmp(ptr->type,"float")){
         field = (float *) ptr->varMemAddress;
         nameLen=strlen(ptr->name);
         fwrite(&nameLen,sizeof(int),1,outptr); 
         fwrite(ptr->name,nameLen*sizeof(char),1,outptr); 
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
         }else{
           numElems=(Nx+2*Nh)*(Ny+2*Nh);
           binary_nDims=2;
           fwrite(&binary_nDims,sizeof(int),1,outptr);
           extent=Nx+2*Nh; 
           fwrite(&extent,sizeof(int),1,outptr);
           extent=Ny+2*Nh; 
           fwrite(&extent,sizeof(int),1,outptr);
           fwrite(field,numElems*sizeof(float),1,outptr);
         }// end if ndims > 2  && ptr->dimids[1] == 1 -else 
      }// if (ptr.type == "float") else ...
      ptr = ptr->next;
   } //end while ptr != NULL

   return(errorCode);
} //ioPutBinaryoutFileVars()
