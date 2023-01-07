/* FastEddy®: SRC/IO/io_netcdf.c
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
#define MAXLEN 256
int dimids[MAXDIMS];
size_t count[MAXDIMS];
size_t start[MAXDIMS];
size_t count2d[MAXDIMS];
size_t start2d[MAXDIMS];
size_t count2dTD[MAXDIMS];
size_t start2dTD[MAXDIMS];

int dims4d[] = {0,1,2,3};
int dims3d[] = {1,2,3};  
int dims2dTD[] = {0,2,3};  
int dims2d[] = {2,3}; 

//////////***********************  INPUT FUNCTIONS  *********************************////////
/*----->>>>> int ioReadNetCDFgridFile();  ---------------------------------------------------------------
* Used to read a NetCDF file of registered "GRID" variables.
*/
int ioReadNetCDFgridFile(char* gridFile, int Nx, int Ny, int Nz, int Nh){
   int errorCode = IO_SUCCESS;
   int ncid;

   /* Open the input file.*/
   printf("Attempting to open gridFile = %s\n",gridFile);
   errorCode = ioOpenNetCDFinFile(gridFile, &ncid);
   printf("Opened gridFile = %s with ncid = %d\n",gridFile,ncid);
   fflush(stdout);
   /* Inquire for the dimension-ids*/
   if ((errorCode = nc_inq_dimid(ncid, "time", &dimids[0]))){
      ERR(errorCode);
   }
   if ((errorCode = nc_inq_dimid(ncid, "zIndex", &dimids[1]))){
      ERR(errorCode);
   }
   if ((errorCode = nc_inq_dimid(ncid, "yIndex", &dimids[2]))){
      ERR(errorCode);
   }
   if ((errorCode = nc_inq_dimid(ncid, "xIndex", &dimids[3]))){
      ERR(errorCode);
   }
   printf("Established dimension ids of xIndex,yIndex,zIndex = %d, %d, %d\n",dimids[3],dimids[2],dimids[1]);
   fflush(stdout);
   /*Attempt to read all of the variables in the IO Registry list*/
   /* These are precisely the same as Nxp, Nyp, and Nzp calculated in GRID/grid.c:grid_init(). */
   count[dimids[0]] = 1;
   count[dimids[1]] = Nz;
   count[dimids[2]] = Ny;
   count[dimids[3]] = Nx;
   if ((errorCode = nc_inq_dimlen(ncid, dimids[0], &count[dimids[0]]))){
      ERR(errorCode);
   }
   if ((errorCode = nc_inq_dimlen(ncid, dimids[1], &count[dimids[1]]))){
      ERR(errorCode);
   }
   if ((errorCode = nc_inq_dimlen(ncid, dimids[2], &count[dimids[2]]))){
      ERR(errorCode);
   }
   if ((errorCode = nc_inq_dimlen(ncid, dimids[3], &count[dimids[3]]))){
      ERR(errorCode);
   }
   /* Check for consistency between inputFile dimLens and parameter file inputs Nx,Ny,Nz.*/
   if((count[dimids[1]]!=Nz)||(count[dimids[2]]!=Ny)||(count[dimids[3]]!=Nx)){
       printf("ERROR: gridFile = %s dimension lengths for x,y,z = %lu,%lu,%lu\n",
               gridFile, count[dimids[3]], count[dimids[2]], count[dimids[1]]);
       printf("       do not match Nx,Ny,Nz = %d,%d,%d parameter file settings!\n",
               Nx,Ny,Nz);
       printf("       No values will be read from the file!\n");
       fflush(stdout);
       errorCode = IO_ERROR_DIMLEN;
       return(errorCode);
   }
   /*These are the starting element in the full domain space. */
   start[dimids[0]] = 0;   
   start[dimids[1]] = 0;
   start[dimids[2]] = 0;
   start[dimids[3]] = 0;

   printf("Reading IO-registered variable fields from gridFile = %s\n",gridFile);
   errorCode = ioGetNetCDFinFileVars(ncid, Nx, Ny, Nz, Nh);
   printf("Done Reading IO-registered variable fields from gridFile = %s\n",gridFile);
   /* close the file */
   errorCode = ioCloseNetCDFfile(ncid);
   printf("Success in reading coordinates/topography from gridfile = %s\n",gridFile);

   return(errorCode);
} //end ioReadNetCDFgridFile

/*----->>>>> int ioReadNetCDFinFileSingleTime();  ---------------------------------------------------------------
 * Used to read a NetCDF file of registered variables for a single timestep.
*/
int ioReadNetCDFinFileSingleTime(int tstep, int Nx, int Ny, int Nz, int Nh){
   int errorCode = IO_SUCCESS;
   int ncid;
   /* concatenate the fileName components */
   sprintf(inFileName, "%s%s",inPath,inFile);
   /* Open the input file.*/
   printf("Attempting to open inFileName = %s\n",inFileName);
   errorCode = ioOpenNetCDFinFile(inFileName, &ncid);
   /* Inquire for the dimension-ids*/
   if ((errorCode = nc_inq_dimid(ncid, "time", &dimids[0]))){
      ERR(errorCode);
   }
   if ((errorCode = nc_inq_dimid(ncid, "zIndex", &dimids[1]))){
      ERR(errorCode);
   }
   if ((errorCode = nc_inq_dimid(ncid, "yIndex", &dimids[2]))){
      ERR(errorCode);
   }
   if ((errorCode = nc_inq_dimid(ncid, "xIndex", &dimids[3]))){
      ERR(errorCode);
   }
   printf("Opened inFileName = %s with ncid = %d\n",inFileName,ncid);
   printf("Established dimension ids of xIndex,yIndex,zIndex = %d, %d, %d\n",dimids[3],dimids[2],dimids[1]);
  
   /*Attempt to read all of the variables in the IO Registry list*/
   /* These are precisely the same as Nxp, Nyp, and Nzp calculated in GRID/grid.c:grid_init(). */
#define NOMPI
   count[dimids[0]] = 1;
   count[dimids[1]] = Nz;
   count[dimids[2]] = Ny;
   count[dimids[3]] = Nx;
   count2d[dimids[0]] = Ny;
   count2d[dimids[1]] = Nx;
   count2dTD[dimids[0]] = 1;
   count2dTD[dimids[1]] = Ny;
   count2dTD[dimids[2]] = Nx;

   if ((errorCode = nc_inq_dimlen(ncid, dimids[0], &count[0]))){
      ERR(errorCode);
   }
   if ((errorCode = nc_inq_dimlen(ncid, dimids[1], &count[1]))){
      ERR(errorCode);
   }
   if ((errorCode = nc_inq_dimlen(ncid, dimids[2], &count[2]))){
      ERR(errorCode);
   }
   if ((errorCode = nc_inq_dimlen(ncid, dimids[3], &count[3]))){
      ERR(errorCode);
   }
   /* Check for consistency between inputFile dimLens and parameter file inputs Nx,Ny,Nz.*/
   if((count[1]!=Nz)||(count[2]!=Ny)||(count[3]!=Nx)){
       printf("ERROR: inFileName = %s dimension lengths for x,y,z = %lu,%lu,%lu\n", 
               inFileName, count[3], count[2], count[1]); 
       printf("       do not match Nx,Ny,Nz = %d,%d,%d parameter file settings!\n",
               Nx,Ny,Nz);
       printf("       No values will be read from the file!\n");
       errorCode = IO_ERROR_DIMLEN;
       return(errorCode); 
   }
   //count2d
   if ((errorCode = nc_inq_dimlen(ncid, dimids[2], &count2d[0]))){
      ERR(errorCode);
   } 
   if ((errorCode = nc_inq_dimlen(ncid, dimids[3], &count2d[1]))){
      ERR(errorCode);
   }
   if((count2d[0]!=Ny)||(count2d[1]!=Nx)){
       printf("ERROR: inFileName = %s, count2d dimension lengths for x,y = %lu,%lu\n",
               inFileName, count2d[1], count2d[0]);
       printf("       do not match Nx,Ny,Nz = %d,%d parameter file settings!\n",
               Nx,Ny);
       printf("       No values will be read from the file!\n");
       errorCode = IO_ERROR_DIMLEN;
       return(errorCode);
   } 
   //count2dTD
   if ((errorCode = nc_inq_dimlen(ncid, dimids[0], &count2dTD[0]))){
      ERR(errorCode);
   }
   if ((errorCode = nc_inq_dimlen(ncid, dimids[2], &count2dTD[1]))){
      ERR(errorCode);
   } 
   if ((errorCode = nc_inq_dimlen(ncid, dimids[3], &count2dTD[2]))){
      ERR(errorCode);
   } 
   if((count2dTD[1]!=Ny)||(count2dTD[2]!=Nx)){
       printf("ERROR: inFileName = %s, count2dTD dimension lengths for t,x,y = %lu,%lu\n",
               inFileName, count2d[2], count2d[1]);
       printf("       do not match Nx,Ny = %d,%d parameter file settings!\n",
               Nx,Ny);
       printf("       No values will be read from the file!\n");
       errorCode = IO_ERROR_DIMLEN;
       return(errorCode);  
   } 

   /*These are the starting location in the full domain space.*/ 
   start[dimids[0]] = 0;   
   start[dimids[1]] = 0;  
   start[dimids[2]] = 0;  
   start[dimids[3]] = 0;  
   
   printf("Reading IO-registered variable fields from inFileName = %s\n",inFileName);
   errorCode = ioGetNetCDFinFileVars(ncid, Nx, Ny, Nz, Nh);
   printf("Done Reading IO-registered variable fields from inFileName = %s\n",inFileName);
   /* close the file */
   errorCode = ioCloseNetCDFfile(ncid);    
   printf("Success in reading from inFileName = %s\n",inFileName);

   return(errorCode);
} //end ioReadNetCDFinFileSingleTime 

/*----->>>>> int ioOpenNetCDFinFile();    ---------------------------------------------------------------------
* Used to open a NetCDF file for reading. This routine is called internally 
* by the main IO routine. Returns the IO private ncid for the file in the call to nc_open.
*/
int ioOpenNetCDFinFile(char *fileName, int *ncidptr){
   int errorCode = IO_SUCCESS;

   /* Open the file. */
   /* If using NetCDF-serial */
   if ((errorCode = nc_open(fileName, NC_NOWRITE, ncidptr))){
           ERR(errorCode);
   }

   return(errorCode);
} //end ioOpenNetCDFinFile()

/*----->>>>> int ioGetNetCDFinFileVars();    ---------------------------------------------------------------------
* Used to get(read) all variables from the register list into the appropriately registered module memory blocks. 
*/
int ioGetNetCDFinFileVars(int ncid, int Nx, int Ny, int Nz, int Nh){
   int errorCode = IO_SUCCESS;
   int varFound;
   size_t *countPtr;
   ioVar_t *ptr;
   ioVar_t *rhoptr;
   float * field;
   float * rhofield;
   int i,j,k;
   int ijk,kji;
   int nDims;
   int tmpDimids[MAXDIMS];
   int rhoMultSwitch=0;

   /* For each entry in the ioVarsList, "get" the var */
   ptr = getFirstVarFromList();
   while(ptr != NULL){
      if(!strcmp(ptr->type,"float")){
         field = (float *) ptr->varMemAddress;  //All ranks set pointer to local memory location 
                                                //for this registered field
         if(mpi_rank_world==0){ 
           if((strcmp(ptr->name,"Tau11")==0)||(strcmp(ptr->name,"Tau21")==0)||(strcmp(ptr->name,"Tau31")==0)||
              (strcmp(ptr->name,"Tau32")==0)||(strcmp(ptr->name,"Tau22")==0)||(strcmp(ptr->name,"Tau33")==0)||
              (strcmp(ptr->name,"TauTH1")==0)||(strcmp(ptr->name,"TauTH2")==0)||(strcmp(ptr->name,"TauTH3")==0)
             ){
             varFound = 0;    
           }else{   
             varFound = 0;
             /*inquire for the varid for this variable name*/
             if( (errorCode = nc_inq_varid(ncid, ptr->name, &ptr->ncvarid)) ){
               printf("Error ioGetNetCDFinFileVars(): Variable field = %s was not found in this file,!\n",ptr->name);
               fflush(stdout);
               ERR(errorCode);
             }else{
               varFound=1;
             }
             printf("Next registered var in list to get is ptr->name = %s, from ptr->ncvarid = %d\n",ptr->name,ptr->ncvarid);
             fflush(stdout);
           
             /*Allocate a tmp buffer and get the pointer to the Register var*/
             if((ptr->nDims == 2)||(ptr->nDims==3)){
               if(ptr->nDims == 2){
                 countPtr = count2d;
               }else if(ptr->nDims == 3){
                 countPtr = count2dTD;
               }//end if,else ptr->nDims == 2,3
             }else if(ptr->nDims==4){
               countPtr = count;
             }
           
             /*read the variable */
             printf("nc_get_vara() for  ptr->name = %s, from ptr-ncvarid = %d,\n into ioBuffField = 0x%p, for transpose into field = 0x%p \n",ptr->name,ptr->ncvarid,(void *) ioBuffField, (void *)field);
             fflush(stdout);
             if ((errorCode = nc_inq_varndims(ncid, ptr->ncvarid, &nDims))){
                ERR(errorCode);
             }
             printf("Variable field = %s has nDims = %d\n",ptr->name,nDims);
             if ((errorCode = nc_inq_vardimid(ncid, ptr->ncvarid, tmpDimids))){
                ERR(errorCode);
             }
             for(i = 0; i< nDims; i++){
               printf("Variable field = %s has dimid(%d) = %d\n",ptr->name,i,tmpDimids[i]);
             }
             //Read in the field
             printf("Attempting for field = %s with start = %lu,%lu,%lu,%lu and count = %lu,%lu,%lu,%lu\n",
                    ptr->name,start[0],start[1],start[2],start[3],countPtr[0],countPtr[1],countPtr[2],countPtr[3]);
             if ((errorCode = nc_get_vara_float(ncid, ptr->ncvarid, start, countPtr, ioBuffField))){
                  ERR(errorCode);
             }
             /* Transpose the data */
             if((nDims == 2)||(nDims == 3)){
               for(i=0; i < Nx; i++){
                 for(j=0; j < Ny; j++){
                   ijk = i*(Ny)+j;  //Note ijk is only 2-d here
                   kji = j*(Nx)+i;  //Note kji is only 2-d here
                   ioBuffFieldTransposed2D[ijk] = ioBuffField[kji]; //out-of-place transpose the array elements
                 } // end for(j...
               } // end for(i...
             }else{
               for(i=0; i < Nx; i++){
                 for(j=0; j < Ny; j++){
                   for(k=0; k < Nz; k++){
                     ijk = i*Ny*Nz+j*Nz+k;
                     kji = k*Ny*Nx+j*Nx+i;
                     ioBuffFieldTransposed[ijk] = ioBuffField[kji]; //out-of-place transpose 
                   } //end for(k...
                 } // end for(j...
               } // end for(i...
             }//end if(nDims==2)-else
           }// if this var is Tau* -else 
         }//end if mpi_rank_world == 0
         MPI_Barrier(MPI_COMM_WORLD);
         //Broadcast the varFound flag for this variable
         MPI_Bcast(&varFound, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);
         if(varFound==1){
           //Broadcast the nDims read by the rrot rank for this variable
           MPI_Bcast(&nDims, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);
           //Now scatter the field across ranks
           if((nDims == 2)||(nDims == 3)){
             errorCode = fempi_ScatterVariable(Nx,Ny,1,Nxp,Nyp,1,Nh,ioBuffFieldTransposed2D,field);
           }else{
             errorCode = fempi_ScatterVariable(Nx,Ny,Nz,Nxp,Nyp,Nzp,Nh,ioBuffFieldTransposed,field);
           }//end if(nDims==2)-else
           //Now multiply by rho for flux conservative when appropriate for registered variable field...
#define NORHO    //TODO define another attribute of the ioVarsList struictures that indicates whether the variable is "flux-conservative form"
#ifdef NORHO     // in which case we multiple by rho upon reading from an input file.
           if( (!strcmp(ptr->name,"u"))||
               (!strcmp(ptr->name,"v"))||
               (!strcmp(ptr->name,"w"))||
               (!strcmp(ptr->name,"theta")) || 
               (!strcmp(ptr->name,"TKE_0")) || 
               (!strcmp(ptr->name,"TKE_1")) || 
               (!strcmp(ptr->name,"qv")) || 
               (!strcmp(ptr->name,"ql")) ||
               (!strcmp(ptr->name,"qr")) ){
               rhoMultSwitch=1;
               rhoptr = getNamedVarFromList("rho");
               if(rhoptr != NULL){
                 rhofield = (float *) rhoptr->varMemAddress;
               }else{
                 printf("ioGetNetCDFinFileVars: Couldn't find rho!!! Catastrophinc Error!!!!!!!!!!!!!!!!!\n");
                 fflush(stdout);
               } //end if
           }else{  //Not a flux conservative field so don't mult by rho
               rhoMultSwitch=0;
           } //end if name is u,v,w, or theta
           if((nDims == 2)||(nDims == 3)){
               //Do nothing, there are no flux-conservative prognostic 2-d fields :-)
           }else{
             if(rhoMultSwitch==1){
               for(i=Nh; i < Nxp+Nh; i++){
                 for(j=Nh; j < Nyp+Nh; j++){
                   for(k=Nh; k < Nzp+Nh; k++){
                     ijk = i*(Nyp+2*Nh)*(Nzp+2*Nh)+j*(Nzp+2*Nh)+k;
                     field[ijk] = field[ijk]*rhofield[ijk]; //out-of-place transpose the array elements
                   } //end for(k...
                 } // end for(j...
               } // end for(i...
             }//end if-else rhoMultSwitch
#endif //if-else NORHO
           }//end if(nDims==2)-else
           MPI_Barrier(MPI_COMM_WORLD);
         } //end if varFound == 1
      } else {
        printf("Cannot 'get' a NetCDF variable with var.type = %s\n",ptr->type);
      }// if (ptr.type == "float") else ...
      ptr = ptr->next;
   }//end while

   return(errorCode);   
} //ioGetNetCDFinFileVars()

//////////***********************  OUTPUT FUNCTIONS  *********************************////////
/*----->>>>> int ioWriteNetCDFoutFileSingleTime();  ---------------------------------------------------------------
 * Used to write a NetCDF file of registered variables for a single timestep.
*/
int ioWriteNetCDFoutFileSingleTime(int tstep, int Nx, int Ny, int Nz, int Nh){
   int errorCode = IO_SUCCESS;
   int ncid;

#ifdef DEBUG 
   printf("mpi_rank_world--%d/%d Beginning ioWriteNetCDFoutFileSingleTime...\n",mpi_rank_world,mpi_size_world);
   fflush(stdout);
#endif
   /* build the subString tag */
   sprintf(outSubString, ".%d",tstep);
   /* concatenate the fileName components */
   sprintf(outFileName, "%s%s%s",outPath,outFileBase,outSubString);
   if(mpi_rank_world==0){
     /* Open and set  the file into "define mode" */
     errorCode = ioCreateNetCDFoutFile(outFileName, &ncid);
     errorCode = ioDefineNetCDFoutFileDims(ncid, Nx, Ny, Nz, Nh);
     errorCode = ioDefineNetCDFoutFileVars(ncid);
     errorCode = ioEndNetCDFdefineMode(ncid,Nx, Ny, Nz, Nh);
     /*Write all of the variables in the IO Registry list*/
   } //endif mpi_rank_world==0
   //Broadcast the ncid...
   MPI_Bcast(&ncid, 1, MPI_INT, 0, MPI_COMM_WORLD);
   errorCode = ioPutNetCDFoutFileVars(ncid, Nx, Ny, Nz, Nh);
   /* close the file */
   if(mpi_rank_world==0){
     errorCode = ioCloseNetCDFfile(ncid);    
   } //endif mpi_rank_world==0

   return(errorCode);
} //end ioWriteNetCDFoutFileSingleTime 

/*----->>>>> int ioCreateNetCDFoutFile();    ---------------------------------------------------------------------
* Used to create NetCDF file for writing. This routine is called internally 
* by the main IO routine. The IO private ncid for the file in the call to nc_create.
*/
int ioCreateNetCDFoutFile(char *outFileName, int *ncidptr){
   int errorCode = IO_SUCCESS;

   /* Create the file. */
   if ((errorCode = nc_create(outFileName, NC_NETCDF4, ncidptr))){
           ERR(errorCode);
   }

   return(errorCode);
} //end ioCreateNetCDFoutFile()

/*----->>>>> int ioDefineNetCDFoutFileDims();    ---------------------------------------------------------------------
* Used to complete the sequence of steps involved in "define mode" for a NetCDF file to be written.
*/

int ioDefineNetCDFoutFileDims(int ncid, int Nx, int Ny, int Nz, int Nh){
   int errorCode = IO_SUCCESS;


   /* The supplied values of dimids used in ioRegisterVar();
 * will always assume the dimensions are defined in this order (time), X, Y, Z.
 * For now time is omitted,so the dimids for a 3-D field in our 
 * X,Y,Z space are 0,1,2 respectively*/

   //"time" is the unlimited record length dimension. 11-1-17 
   if ((errorCode = nc_def_dim(ncid, "time", NC_UNLIMITED, &dimids[0]))){
      ERR(errorCode);
   }
   /* Define the dimensions (in column-major order so we can use all the visulaisation tools). */
   if ((errorCode = nc_def_dim(ncid, "zIndex", Nz, &dimids[1]))){
      ERR(errorCode);
   }
   if ((errorCode = nc_def_dim(ncid, "yIndex", Ny, &dimids[2]))){
      ERR(errorCode);
   }
   if ((errorCode = nc_def_dim(ncid, "xIndex", Nx, &dimids[3]))){
      ERR(errorCode);
   }

   count[dimids[0]] = 1;
   count[dimids[1]] = Nz;
   count[dimids[2]] = Ny;
   count[dimids[3]] = Nx;
   count2d[dimids[0]] = Ny;
   count2d[dimids[1]] = Nx;
   count2dTD[dimids[0]] = 1;
   count2dTD[dimids[1]] = Ny;
   count2dTD[dimids[2]] = Nx;

   /*These are the starting location in the full domain space. */
   start[dimids[0]] = 0; 
   start[dimids[1]] = 0; 
   start[dimids[2]] = 0; 
   start[dimids[3]] = 0; 
   
   return(errorCode);
} //end ioDefineNetCDFoutFileDims()

/*----->>>>> int ioDefineNetCDFoutFileVars();    ---------------------------------------------------------------------
 * Used to complete the sequence of variable definitions involved in "define mode" for a NetCDF file to be written.
*/
int ioDefineNetCDFoutFileVars(int ncid){
   int errorCode = IO_SUCCESS;
   ioVar_t *ptr;

   /*define the dimension-index (aka coordinate variables)*/
   if((errorCode = nc_def_var(ncid, "zIndex", NC_INT, 1, &dimids[1], &nz_varid))){
      ERR(errorCode);
   }
   if((errorCode = nc_def_var(ncid, "yIndex", NC_INT, 1, &dimids[2], &ny_varid))){
      ERR(errorCode);
   }
   if((errorCode = nc_def_var(ncid, "xIndex", NC_INT, 1, &dimids[3], &nx_varid))){
      ERR(errorCode);
   }
   /* For each entry in the ioVarsList, def the var */
   ptr = getFirstVarFromList();
   while(ptr != NULL){
      /* define the variable */
      if (!strcmp(ptr->type,"float")){ 
         if ((errorCode = nc_def_var(ncid, ptr->name, NC_FLOAT, ptr->nDims, ptr->dimids, &ptr->ncvarid))){
            ERR(errorCode);
         }
      } else {
        printf("Cannot define a NetCDF variable with var->type = %s\n",ptr->type);
      }// if (ptr.type == "float") else ...
      /* define any variable attributes. */
/*TODO      
      if ((errorCode = nc_put_att_text(ncid, ptr->ncvarid, ptr->attname, strlen(ptr->attval), ptr->attval))){
           ERR(errorCode);
      } 
*/
      ptr = ptr->next;
   }
   
   return(errorCode);
} //end 

/*----->>>>> int ioEndNetCDFdefineMode();    ---------------------------------------------------------------------
 * Used to close the sequence steps involved in "define mode" for a NetCDF file to be written.
 */
int ioEndNetCDFdefineMode(int ncid, int Nx, int Ny, int Nz, int Nh){
   int *dimIndexVec;
   int  dimIndexCnt;
   int idx;

   //End the define Mode for this netcdf file
   int errorCode = IO_SUCCESS;
        if ((errorCode = nc_enddef(ncid))){
           ERR(errorCode);
        }
   //Write the time-constant dimension-index vectors
   //Find the longest dimension 
   dimIndexCnt = Nz;
   if(dimIndexCnt < Ny){
     dimIndexCnt = Ny;
   } 
   if(dimIndexCnt < Nx){
     dimIndexCnt = Nx;
   } 
   //Malloc an index vector with dimIndexCnt elements;
   dimIndexVec= (int *) malloc(dimIndexCnt*sizeof(int)); /* 3 for each part of path/base.subString */ 
   //Initialize the index vector
   for(idx=0; idx < dimIndexCnt; idx++){
     dimIndexVec[idx] = idx;
   } 
   if ((errorCode = nc_put_vara_int(ncid, nz_varid, &start[dimids[1]], &count[dimids[1]], dimIndexVec))){
       ERR(errorCode);
   } 
   if ((errorCode = nc_put_vara_int(ncid, ny_varid, &start[dimids[2]], &count[dimids[2]], dimIndexVec))){
       ERR(errorCode);
   } 
   if ((errorCode = nc_put_vara_int(ncid, nx_varid, &start[dimids[3]], &count[dimids[3]], dimIndexVec))){
       ERR(errorCode);
   } 
   free(dimIndexVec);
   
   return(errorCode);
}//end ioEndNetCDFdefineMode

/*----->>>>> int ioPutNetCDFoutFileVars();    ---------------------------------------------------------------------
* Used to put(write) all variables in the regiter list in(to) the NetCDF file. 
*/
int ioPutNetCDFoutFileVars(int ncid, int Nx, int Ny, int Nz, int Nh){
   int errorCode = IO_SUCCESS;
   size_t *countPtr;
   ioVar_t *ptr;
   ioVar_t *rhoptr;
   float * field;
   int numElems;
   int i,j,k;
   int ijk,kji,ijkTransposed;
   int rhoDivideSwitch = 0;
   int verbose_log = 0;

   /* For each entry in the ioVarsList, "put" the var */
   ptr = getFirstVarFromList();
   while(ptr != NULL){
//#define NORHO  //if defined, it was in ioGetNetCDFinFileVars up above
#ifdef NORHO
   float * rhofield;
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
#endif //NORHO
      if (!strcmp(ptr->type,"float")){ 
         field = (float *) ptr->varMemAddress;
         /* Trim-halos and Transpose the row-major ordered internal field 
         *  to a column major order for writing the netcdf file 
         */
         if((ptr->nDims > 2)&&(ptr->dimids[1] == 1)){  //Should be a 4D with time,z,y,x...
           countPtr=count;
           numElems = Nx*Ny*Nz;
           //Gather the variable field...
           errorCode = fempi_GatherVariable(Nxp,Nyp,Nzp,Nh,Nx,Ny,Nz,field,ioBuffFieldTransposed);
#ifdef NORHO
           if(mpi_rank_world==0){
             if(rhoDivideSwitch==1){
               for(i=0; i < Nx; i++){
                 for(j=0; j < Ny; j++){
                   for(k=0; k < Nz; k++){
                     ijk = (i+Nh)*(Ny+2*Nh)*(Nz+2*Nh)+(j+Nh)*(Nz+2*Nh)+(k+Nh); //Account for halo presence in the raw field
                     kji = k*(Ny)*(Nx)+j*(Nx)+i;  //Do not include halos in the destination array
                     ijkTransposed = i*(Ny)*(Nz)+j*(Nz)+k;  //Do not include halos in the destination array
                     ioBuffField[kji] = ioBuffFieldTransposed[ijkTransposed]/ioBuffFieldRho[kji]; //out-of-place trim and transpose if the array elements
                   } //end for(k...
                 } // end for(j...
               } // end for(i...
             }else{
               for(i=0; i < Nx; i++){
                 for(j=0; j < Ny; j++){
                   for(k=0; k < Nz; k++){
                     ijk = (i+Nh)*(Ny+2*Nh)*(Nz+2*Nh)+(j+Nh)*(Nz+2*Nh)+(k+Nh); //Account for halo presence in the raw field
                     kji = k*(Ny)*(Nx)+j*(Nx)+i;  //Do not include halos in the destination array
                     ijkTransposed = i*(Ny)*(Nz)+j*(Nz)+k;  //Do not include halos in the destination array
                     ioBuffField[kji] = ioBuffFieldTransposed[ijkTransposed]; //out-of-place trim and transpose if the array elements
                   } //end for(k...
                 } // end for(j...
               } // end for(i...
             }//end if-else rhoDivideSwitch 
             /*If this field is rho, store the global field for subsequent reuse to convert out of 
               flux-conservative form...*/
             if (!strcmp(ptr->name,"rho")){
                memcpy(ioBuffFieldRho,ioBuffField,numElems*sizeof(float));
             } //endif this field was rho
           } //endif mpi_rank_world==0
#else  // write the raw variable field (except and notably the flux conservative raw forms of u,v,w,theta) variable fields
           for(i=0; i < Nx; i++){
             for(j=0; j < Ny; j++){
               for(k=0; k < Nz; k++){
                 ijk = (i+Nh)*(Ny+2*Nh)*(Nz+2*Nh)+(j+Nh)*(Nz+2*Nh)+(k+Nh); //Account for halo presence in the raw field
                 kji = k*(Ny)*(Nx)+j*(Nx)+i;  //Do not include halos in the destination array
                 ijkTransposed = i*(Ny)*(Nz)+j*(Nz)+k;  //Do not include halos in the destination array
                 ioBuffField[kji] = ioBuffFieldTransposed[ijkTransposed]; //out-of-place trim and transpose if the array elements
               } //end for(k...
             } // end for(j...
           } // end for(i...
#endif //NORHO
         }else if((ptr->nDims == 3)&&(ptr->dimids[1] == 2)){
           countPtr=count2dTD;
           numElems = Nx*Ny;
           //Gather the variable field...
           errorCode = fempi_GatherVariable(Nxp,Nyp,1,Nh,Nx,Ny,1,field,ioBuffFieldTransposed2D);
           if(mpi_rank_world==0){
             for(i=0; i < Nx; i++){
               for(j=0; j < Ny; j++){
                 ijk = (i+Nh)*(Ny+2*Nh)+(j+Nh); //Account for halo presence in the raw field
                 kji = j*(Nx)+i;  //Do not include halos in the destination array
                 ijkTransposed = i*(Ny)+j;  //Do not include halos in the destination array
                 ioBuffField[kji] = ioBuffFieldTransposed2D[ijkTransposed]; //out-of-place trim and transpose if the array elements
               } // end for(j...
             } // end for(i...
           } //endif mpi_Rank_world==0
         }else{
           countPtr=count2d;
           numElems = Nx*Ny;
           /* Set the coordinate bounds */
           for(i=0; i < Nx; i++){
             for(j=0; j < Ny; j++){
                 ijk = (i+Nh)*(Ny+2*Nh)+(j+Nh); //Account for halo presence in the raw field
                 kji = j*(Nx)+i;  //Do not include halos in the destination array
                 ioBuffField[kji] = field[ijk]; //out-of-place trim and transpose if the array elements
             } // end for(j...
           } // end for(i...
         }// end if ndims==3  && time,y,z... -else        

         /*write the variable */
#ifdef DEBUG
   printf("mpi_rank_world--%d/%d Putting variable field %s...\n",mpi_rank_world,mpi_size_world,ptr->name);
   fflush(stdout);
#endif
         if (mpi_rank_world==0){
           if ((errorCode = nc_put_vara_float(ncid, ptr->ncvarid, start, countPtr, ioBuffField))){
              ERR(errorCode);
              printf("ioPutNetCDFoutFileVars: Error writing field = %s\n",ptr->name);
           }
         }//endif mpi_Rank_world==0
      } else {
        printf("Cannot 'put' a NetCDF variable with var.type = %s\n",ptr->type);
      }// if (ptr.type == "float") else ...
      ptr = ptr->next;
   } //end while ptr != NULL

   return(errorCode);   
} //ioPutNetCDFoutFileVars()

/*----->>>>> int ioCloseNetCDFfile();    ---------------------------------------------------------------------
 * Used to close a netCDF file
 * */
int ioCloseNetCDFfile(int ncid){
   int errorCode = IO_SUCCESS;
   /* Close the file. */
   if ((errorCode = nc_close(ncid))){
      ERR(errorCode);
   } 

   return(errorCode);
}  //end ioCloseNetCDFfile()
