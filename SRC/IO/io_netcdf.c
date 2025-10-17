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

#include <stdlib.h>

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
int dims1dTD[] = {0};
#ifdef GAD_EXT
   size_t count1dTD_GAD[MAXDIMS];
   size_t start1dTD_GAD[MAXDIMS];
   int dims1dTD_GAD[] = {0,4};
#endif
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
#ifdef GAD_EXT
   errorCode = ioGetNetCDFinFileVars(ncid, Nx, Ny, Nz, Nh, 0);
#else
   errorCode = ioGetNetCDFinFileVars(ncid, Nx, Ny, Nz, Nh);
#endif
   printf("Done Reading IO-registered variable fields from gridFile = %s\n",gridFile);
   /* close the file */
   errorCode = ioCloseNetCDFfile(ncid);
   printf("Success in reading coordinates/topography from gridfile = %s\n",gridFile);

   return(errorCode);
} //end ioReadNetCDFgridFile

/*----->>>>> int ioReadNetCDFinFileSingleTime();  ---------------------------------------------------------------
 * Used to read a NetCDF file of registered variables for a single timestep.
*/
#ifdef GAD_EXT
int ioReadNetCDFinFileSingleTime(int tstep, int Nx, int Ny, int Nz, int Nh, int Nturbines){
#else
int ioReadNetCDFinFileSingleTime(int tstep, int Nx, int Ny, int Nz, int Nh){
#endif
   int errorCode = IO_SUCCESS;
   int ncid;
   int ncdims;
   /* concatenate the fileName components */
   sprintf(inFileName, "%s%s",inPath,inFile);
   /* Open the input file.*/
   printf("Attempting to open inFileName = %s\n",inFileName);
   errorCode = ioOpenNetCDFinFile(inFileName, &ncid);
   /* Inquire for the inumber of dimensions*/
   if ((errorCode = nc_inq_ndims(ncid, &ncdims))){
      ERR(errorCode);
   }
   printf("inFileName = %s as %d dimensions\n",inFileName,ncdims);

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
#ifdef GAD_EXT
   if(ncdims > 4){
     if ((errorCode = nc_inq_dimid(ncid, "GADNumTurbines", &dimids[4]))){
        ERR(errorCode);
     }
     printf("Established GAD dimension id of GADNumTurbines as %d\n",dimids[4]);
   }//endif ncdims > 4
#endif
  
   /*Attempt to read all of the variables in the IO Registry list*/
   /* These are precisely the same as Nxp, Nyp, and Nzp calculated in GRID/grid.c:grid_init(). */
   count[dimids[0]] = 1;
   count[dimids[1]] = Nz;
   count[dimids[2]] = Ny;
   count[dimids[3]] = Nx;
   count2d[dimids[0]] = Ny;
   count2d[dimids[1]] = Nx;
   count2dTD[dimids[0]] = 1;
   count2dTD[dimids[1]] = Ny;
   count2dTD[dimids[2]] = Nx;
#ifdef GAD_EXT
   count1dTD_GAD[dimids[0]] = 1;
   count1dTD_GAD[dimids[1]] = Nturbines;
#endif
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
#ifdef GAD_EXT
   if(ncdims > 4){
     //count1dTD_GAD
     if ((errorCode = nc_inq_dimlen(ncid, dimids[0], &count1dTD_GAD[0]))){
        ERR(errorCode);
     }
     if ((errorCode = nc_inq_dimlen(ncid, dimids[4], &count1dTD_GAD[1]))){
        ERR(errorCode);
     }
     if(count1dTD_GAD[1]!=Nturbines){
        printf("ERROR: inFileName = %s, count1dTD_GAD dimension lengths for t,GADNumTurbines = %lu\n",
               inFileName, count1dTD_GAD[1]);
        printf("       does not match GADNumTurbines = %d turbineSpecsFile parameter!\n",
               Nturbines);
        printf("       No values will be read from the file!\n");
        errorCode = IO_ERROR_DIMLEN;
        return(errorCode);
     }
   }
#endif
   /*These are the starting location in the full domain space.*/ 
   start[dimids[0]] = 0;   
   start[dimids[1]] = 0;  
   start[dimids[2]] = 0;  
   start[dimids[3]] = 0;  
#ifdef GAD_EXT
   start[dimids[4]] = 0;
#endif
   
   printf("Reading IO-registered variable fields from inFileName = %s\n",inFileName);
#ifdef GAD_EXT
   errorCode = ioGetNetCDFinFileVars(ncid, Nx, Ny, Nz, Nh, Nturbines);
#else
   errorCode = ioGetNetCDFinFileVars(ncid, Nx, Ny, Nz, Nh);
#endif
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
#ifdef GAD_EXT
int ioGetNetCDFinFileVars(int ncid, int Nx, int Ny, int Nz, int Nh, int Nturbines){
#else
int ioGetNetCDFinFileVars(int ncid, int Nx, int Ny, int Nz, int Nh){
#endif
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
   int *intField;

   /* For each entry in the ioVarsList, "get" the var */
   ptr = getFirstVarFromList();
   while(ptr != NULL){
      if(mpi_rank_world==0){ 
        printf("Checking for %s...\n",ptr->name);
        fflush(stdout);
      }//end if mpi_rank==0
      varFound = 0;
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
               printf("Next registered var in list to get is ptr->name = %s, from ptr->ncvarid = %d\n",ptr->name,ptr->ncvarid);
               fflush(stdout);
               varFound=1;
             }
          
	     if(varFound == 1){  //Only do this section if a var was found 
             /*Allocate a tmp buffer and get the pointer to the Register var*/
             if((ptr->nDims == 2)||(ptr->nDims==3)){
               if(ptr->nDims == 2){
		 if(ptr->dimids[1] == 2){
                   countPtr = count2d;
#ifdef GAD_EXT
		 }else if(ptr->dimids[1] == 4){
                   countPtr = count1dTD_GAD;
#endif
                 }
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
             fflush(stdout);
             if ((errorCode = nc_inq_vardimid(ncid, ptr->ncvarid, tmpDimids))){
                ERR(errorCode);
             }
             for(i = 0; i< nDims; i++){
	       printf("Variable field = %s has dimid(%d) = %d: start, count => %lu, %lu\n",ptr->name,i,tmpDimids[i],start[tmpDimids[i]],countPtr[tmpDimids[i]]);
             }
	     fflush(stdout);
             //Read in the field
             printf("Attempting for field = %s with start = %lu,%lu,%lu,%lu and count = %lu,%lu,%lu,%lu\n",
                    ptr->name,start[0],start[1],start[2],start[3],countPtr[0],countPtr[1],countPtr[2],countPtr[3]);
             if ((errorCode = nc_get_vara_float(ncid, ptr->ncvarid, start, countPtr, ioBuffField))){
                  ERR(errorCode);
             }
	     fflush(stdout);
             /* Transpose the data */
             if((nDims == 2)||(nDims == 3)){
#ifdef GAD_EXT
	       if(ptr->dimids[1] == 4){
                 for(i=0; i < Nturbines; i++){
                    field[i] = ioBuffField[i];
		 }
	       }else{
#endif
                 for(i=0; i < Nx; i++){
                   for(j=0; j < Ny; j++){
                     ijk = i*(Ny)+j;  //Note ijk is only 2-d here
                     kji = j*(Nx)+i;  //Note kji is only 2-d here
                     ioBuffFieldTransposed2D[ijk] = ioBuffField[kji]; //out-of-place transpose the array elements
                   } // end for(j...
                 } // end for(i...
#ifdef GAD_EXT
	       }//end if (ptr->dimids[1] == 4) else...
#endif
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
	 }//end if varFound==1
         }//end if mpi_rank_world == 0
         MPI_Barrier(MPI_COMM_WORLD);
         //Broadcast the varFound flag for this variable
         MPI_Bcast(&varFound, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);
         if(varFound==1){
           //Broadcast the nDims read by the rrot rank for this variable
           MPI_Bcast(&nDims, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);
           //Now scatter the field across ranks
           if((nDims == 2)||(nDims == 3)){
#ifdef GAD_EXT
             if(ptr->dimids[1] == 4){
	       MPI_Bcast(field, Nturbines, MPI_FLOAT, 0, MPI_COMM_WORLD);
             }else{
#endif
               errorCode = fempi_ScatterVariable(Nx,Ny,1,Nxp,Nyp,1,Nh,ioBuffFieldTransposed2D,field);
#ifdef GAD_EXT
	     }//end if (ptr->dimids[1] == 4) else...
#endif
           }else if(nDims == 4){
             errorCode = fempi_ScatterVariable(Nx,Ny,Nz,Nxp,Nyp,Nzp,Nh,ioBuffFieldTransposed,field);
           }else if(nDims == 1){  // A scalar float variable was read, it shoud be simply broadcast to all ranks rather than "scattered"
             if(mpi_rank_world==0){
               *field=ioBuffField[0];
             }
             MPI_Bcast(field, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
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
      }else if(!strcmp(ptr->type,"int")){
        intField = (int *) ptr->varMemAddress;  //All ranks set pointer to local memory location 
	if(ptr->nDims == 1){					
          countPtr = count;
#ifdef GAD_EXT
	}else if(ptr->nDims == 2){
          countPtr = count1dTD_GAD;
#endif
        }//end if ptr->nDims ==1
        if(mpi_rank_world==0){
         varFound = 0;
         /*inquire for the varid for this variable name*/
         if ( (errorCode = nc_inq_varid(ncid, ptr->name, &ptr->ncvarid)) ){
           printf("Error ioGetNetCDFinFileVars(): Variable field = %s was not found in this file,!\n",ptr->name);
           fflush(stdout);
           ERR(errorCode);
         }else{
           printf("Next registered var in list to get is ptr->name = %s, from ptr->ncvarid = %d\n",ptr->name,ptr->ncvarid);
           fflush(stdout);
           varFound=1;
         }

         if(varFound==1){
           /*read the variable */
           printf("nc_get_vara() for  ptr->name = %s, from ptr-ncvarid = %d,\n into intField = 0x%p \n",ptr->name,ptr->ncvarid,(void *) intField);
           fflush(stdout);
           if ((errorCode = nc_inq_varndims(ncid, ptr->ncvarid, &nDims))){
              ERR(errorCode);
           }
           printf("Variable field = %s has nDims = %d\n",ptr->name,nDims);
           if ((errorCode = nc_inq_vardimid(ncid, ptr->ncvarid, tmpDimids))){
              ERR(errorCode);
           }
           for(i = 0; i< nDims; i++){
              //printf("Variable field = %s has dimid(%d) = %d\n",ptr->name,i,tmpDimids[i]);
              printf("Variable field = %s has dimid(%d) = %d: start, count => %lu, %lu\n",ptr->name,i,tmpDimids[i],start[tmpDimids[i]],countPtr[tmpDimids[i]]);
           }
           //Actually read in the field
           //printf("Attempting for field = %s with start = %lu,%lu,%lu,%lu and count = %lu,%lu,%lu,%lu\n",
           //        ptr->name,start[0],start[1],start[2],start[3],countPtr[0],countPtr[1],countPtr[2],countPtr[3]);
           if ((errorCode = nc_get_vara_int(ncid, ptr->ncvarid, start, countPtr, intField))){
                ERR(errorCode);
           }
         } //end if varFound == 1
        }//end if mpi_rank_world==0
        MPI_Bcast(&varFound, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);
        if(varFound==1){
          //Broadcast the nDims read by the rrot rank for this variable
          MPI_Bcast(&nDims, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);
          if(nDims == 1){
            MPI_Bcast(intField, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);
#ifdef GAD_EXT
	  }else if(nDims == 2){
            MPI_Bcast(intField, Nturbines, MPI_INTEGER, 0, MPI_COMM_WORLD);
#endif
          }//end if nDims == 1
        }//end if varFound == 1
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
#ifdef GAD_EXT
int ioWriteNetCDFoutFileSingleTime(int tstep, int Nx, int Ny, int Nz, int Nh, int Nturbines){
#else
int ioWriteNetCDFoutFileSingleTime(int tstep, int Nx, int Ny, int Nz, int Nh){
#endif
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
#ifdef GAD_EXT
     errorCode = ioDefineNetCDFoutFileDims(ncid, Nx, Ny, Nz, Nh, Nturbines);
#else
     errorCode = ioDefineNetCDFoutFileDims(ncid, Nx, Ny, Nz, Nh);
#endif
     errorCode = ioDefineNetCDFoutFileVars(ncid);
     /* Define dimension coordinate variable attributes */
     errorCode = ioDefineNetCDFcoordVarAttrs(ncid);
     /* Define variable attributes */
     errorCode = ioDefineNetCDFoutFileAttrs(ncid);
#ifdef GAD_EXT
     errorCode = ioEndNetCDFdefineMode(ncid,Nx, Ny, Nz, Nh, Nturbines);
#else
     errorCode = ioEndNetCDFdefineMode(ncid,Nx, Ny, Nz, Nh);
#endif
     /*Write all of the variables in the IO Registry list*/
   } //endif mpi_rank_world==0
   //Broadcast the ncid...
   MPI_Bcast(&ncid, 1, MPI_INT, 0, MPI_COMM_WORLD);
#ifdef GAD_EXT
   errorCode = ioPutNetCDFoutFileVars(ncid, Nx, Ny, Nz, Nh, Nturbines);
#else
   errorCode = ioPutNetCDFoutFileVars(ncid, Nx, Ny, Nz, Nh);
#endif
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
#ifdef GAD_EXT
int ioDefineNetCDFoutFileDims(int ncid, int Nx, int Ny, int Nz, int Nh, int Nturbines){
#else
int ioDefineNetCDFoutFileDims(int ncid, int Nx, int Ny, int Nz, int Nh){
#endif
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
   
#ifdef GAD_EXT
   if ((errorCode = nc_def_dim(ncid, "GADNumTurbines", Nturbines, &dimids[4]))){
      ERR(errorCode);
   }
   count1dTD_GAD[dimids[0]] = 1;
   count1dTD_GAD[dimids[4]] = Nturbines;
#endif
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
      }else if (!strcmp(ptr->type,"int")){
         if ((errorCode = nc_def_var(ncid, ptr->name, NC_INT, ptr->nDims, ptr->dimids, &ptr->ncvarid))){
            ERR(errorCode);
         }
      } else {
        printf("Cannot define a NetCDF variable with var->type = %s\n",ptr->type);
      }// if (ptr.type == "float") else ...
      /* define any variable attributes. */
      ptr = ptr->next;
   }
   
   return(errorCode);
} //end 

/*----->>>>> int ioEndNetCDFdefineMode();    ---------------------------------------------------------------------
 * Used to close the sequence steps involved in "define mode" for a NetCDF file to be written.
 */
#ifdef GAD_EXT
int ioEndNetCDFdefineMode(int ncid, int Nx, int Ny, int Nz, int Nh, int Nturbines){
#else
int ioEndNetCDFdefineMode(int ncid, int Nx, int Ny, int Nz, int Nh){
#endif
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
#ifdef GAD_EXT
   if(dimIndexCnt < Nturbines){
     dimIndexCnt = Nturbines;
   }
#endif

   //Malloc an index vector with dimIndexCnt elements;
   dimIndexVec= (int *) malloc(dimIndexCnt*sizeof(int)); 
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
#ifdef GAD_EXT
   if ((errorCode = nc_put_vara_int(ncid, nx_varid, &start[dimids[4]], &count[dimids[4]], dimIndexVec))){
       ERR(errorCode);
   }
#endif 
   free(dimIndexVec);
   
   return(errorCode);
}//end ioEndNetCDFdefineMode

/*----->>>>> int ioPutNetCDFoutFileVars();    ---------------------------------------------------------------------
* Used to put(write) all variables in the regiter list in(to) the NetCDF file. 
*/
#ifdef GAD_EXT
int ioPutNetCDFoutFileVars(int ncid, int Nx, int Ny, int Nz, int Nh, int Nturbines){
#else
int ioPutNetCDFoutFileVars(int ncid, int Nx, int Ny, int Nz, int Nh){
#endif
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
   int *intField;

#ifdef GAD_EXT
   void* memsetReturnVal;
#endif

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
         }else if((ptr->nDims == 2)&&(ptr->dimids[1] == 2)){
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
#ifdef GAD_EXT
         }else if((ptr->nDims == 2)&&(ptr->dimids[1] == 4)){
           countPtr=count1dTD_GAD;
           /* Reset to zero and then Reduce(MPI_SUM op) into the write buffer */
           if(mpi_rank_world==0){
	     memsetReturnVal = memset(ioBuffField,0,(Nturbines)*sizeof(float));  //Just sets the first Nturbines elements to zero
	   }//end if mpi_rank ==0
           MPI_Reduce(field, ioBuffField, Nturbines, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
#endif
         }else if((ptr->nDims == 1)&&(ptr->dimids[0] == 0)){
           countPtr=count;
           if(mpi_rank_world==0){
             ioBuffField[0] = *field;
           } //endif mpi_Rank_world==0
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
      } else if(!strcmp(ptr->type,"int")){
#ifdef DEBUG
   printf("mpi_rank_world--%d/%d Putting variable field %s...\n",mpi_rank_world,mpi_size_world,ptr->name);
   fflush(stdout);
#endif
         intField = (int *) ptr->varMemAddress;
         if((ptr->nDims == 1)&&(ptr->dimids[0] == 0)){
           countPtr=count;
           if (mpi_rank_world==0){
             if ((errorCode = nc_put_vara_int(ncid, ptr->ncvarid, start, countPtr, intField))){
                ERR(errorCode);
                printf("ioPutNetCDFoutFileVars: Error writing field = %s\n",ptr->name);
                fflush(stdout);
             }
           }//endif mpi_Rank_world==0
#ifdef GAD_EXT
	 }else if((ptr->nDims == 2)&&(ptr->dimids[1] == 4)){
           countPtr=count1dTD_GAD;
           if (mpi_rank_world==0){
	      memsetReturnVal = memset(ioBuffFieldInt,0,(Nturbines)*sizeof(int));  //Just sets the first Nturbines elements to zero
	      if (memsetReturnVal==NULL){
                printf("ioPutNetCDFoutFileVars: Error in call to memset for var = %s\n",ptr->name);
                fflush(stdout);
	      }//end if 
           }//endif mpi_Rank_world==0
           MPI_Reduce(intField, ioBuffFieldInt, Nturbines, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
           if (mpi_rank_world==0){
             if ((errorCode = nc_put_vara_int(ncid, ptr->ncvarid, start, countPtr, ioBuffFieldInt))){
                ERR(errorCode);
                printf("ioPutNetCDFoutFileVars: Error writing field = %s\n",ptr->name);
                fflush(stdout);
             }
           }//endif mpi_Rank_world==0
#endif         
	 }// end if ndims==1  && dimids[0]=0 (time), #ifdef GAD_EXT-- else if(nDims==2 && dimids[1]=4) #endif
      } else {
        printf("Cannot 'put' a NetCDF variable with var.type = %s\n",ptr->type);
      }// if (ptr->type == "float") else if(ptr->type == "int")...
      MPI_Barrier(MPI_COMM_WORLD);
      ptr = ptr->next;
   } //end while ptr != NULL

   return(errorCode);   
} //ioPutNetCDFoutFileVars()

/*----->>>>> int ioDefineNetCDFoutFileAttrs();    ---------------------------------------------------------------------
* Used to define NetCDF variable attributes.
*/
int ioDefineNetCDFoutFileAttrs(int ncid){
   int errorCode = IO_SUCCESS;
   ioVar_t *ptr;
   int i;

   /* For each entry in the ioVarsList, define attributes if they exist */
   ptr = getFirstVarFromList();
   while(ptr != NULL){
      /* Check if variable has attributes defined and loop through them */
      for(i = 0; i < ptr->nAttrs; i++){
         if(strlen(ptr->attrs[i].name) > 0 && strlen(ptr->attrs[i].value) > 0){
            /* Determine the appropriate NetCDF function based on attribute type */
            if(strcmp(ptr->attrs[i].type, "text") == 0){
               if ((errorCode = nc_put_att_text(ncid, ptr->ncvarid, ptr->attrs[i].name,
                                              strlen(ptr->attrs[i].value), ptr->attrs[i].value))){
                  ERR(errorCode);
               }
            }
            else if(strcmp(ptr->attrs[i].type, "float") == 0){
               float val = atof(ptr->attrs[i].value);
               if ((errorCode = nc_put_att_float(ncid, ptr->ncvarid, ptr->attrs[i].name, NC_FLOAT, 1, &val))){
                  ERR(errorCode);
               }
            }
            else if(strcmp(ptr->attrs[i].type, "double") == 0){
               double val = atof(ptr->attrs[i].value);
               if ((errorCode = nc_put_att_double(ncid, ptr->ncvarid, ptr->attrs[i].name, NC_DOUBLE, 1, &val))){
                  ERR(errorCode);
               }
            }
            else if(strcmp(ptr->attrs[i].type, "int") == 0){
               int val = atoi(ptr->attrs[i].value);
               if ((errorCode = nc_put_att_int(ncid, ptr->ncvarid, ptr->attrs[i].name, NC_INT, 1, &val))){
                  ERR(errorCode);
               }
            }
            else {
               /* Default to text if type is unrecognized */
               if ((errorCode = nc_put_att_text(ncid, ptr->ncvarid, ptr->attrs[i].name,
                                              strlen(ptr->attrs[i].value), ptr->attrs[i].value))){
                  ERR(errorCode);
               }
            }
         }
      }

      ptr = ptr->next;
   }
   
   return(errorCode);
} //end ioDefineNetCDFoutFileAttrs

/*----->>>>> int ioDefineNetCDFcoordVarAttrs();    ---------------------------------------------------------------------
* Used to define attributes for dimension coordinate variables.
*/
int ioDefineNetCDFcoordVarAttrs(int ncid){
   int errorCode = IO_SUCCESS;
   
   /* Define attributes for xIndex coordinate variable */
   if ((errorCode = nc_put_att_text(ncid, nx_varid, "long_name", 
                                   strlen("x-coordinate index"), "x-coordinate index"))){
      ERR(errorCode);
   }
   if ((errorCode = nc_put_att_text(ncid, nx_varid, "units", strlen("1"), "1"))){
      ERR(errorCode);
   }
   if ((errorCode = nc_put_att_text(ncid, nx_varid, "axis", strlen("X"), "X"))){
      ERR(errorCode);
   }
   
   /* Define attributes for yIndex coordinate variable */
   if ((errorCode = nc_put_att_text(ncid, ny_varid, "long_name", 
                                   strlen("y-coordinate index"), "y-coordinate index"))){
      ERR(errorCode);
   }
   if ((errorCode = nc_put_att_text(ncid, ny_varid, "units", strlen("1"), "1"))){
      ERR(errorCode);
   }
   if ((errorCode = nc_put_att_text(ncid, ny_varid, "axis", strlen("Y"), "Y"))){
      ERR(errorCode);
   }
   
   /* Define attributes for zIndex coordinate variable */
   if ((errorCode = nc_put_att_text(ncid, nz_varid, "long_name", 
                                   strlen("z-coordinate index"), "z-coordinate index"))){
      ERR(errorCode);
   }
   if ((errorCode = nc_put_att_text(ncid, nz_varid, "units", strlen("1"), "1"))){
      ERR(errorCode);
   }
   if ((errorCode = nc_put_att_text(ncid, nz_varid, "axis", strlen("Z"), "Z"))){
      ERR(errorCode);
   }
   if ((errorCode = nc_put_att_text(ncid, nz_varid, "positive", strlen("up"), "up"))){
      ERR(errorCode);
   }

   return(errorCode);
} //end ioDefineNetCDFcoordVarAttrs


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
