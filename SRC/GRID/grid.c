/* FastEddy®: SRC/GRID/grid.c 
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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <float.h>
#include <math.h>
#include <fempi.h>
#include <parameters.h>
#include <mem_utils.h>
#include <io.h>
#include <grid.h>

/*##################------------------- GRID module variable definitions ---------------------#################*/
char *gridFile = NULL;
char *topoFile = NULL;
int Nx = 1;
int Ny = 1;
int Nz = 1;
int Nh = 0;
float d_xi = 1.0;
float d_eta = 1.0;
float d_zeta = 1.0;
int coordHorizHalos = 1; //switch to setup coordiante halos as periodic, or gradient following
int iMin, iMax; //Constant min and max bounds of i-index accounting for only non-halos cells of the mpi_rank subdomain
int jMin, jMax; //Constant min and max bounds of j-index accounting for only non-halos cells of the mpi_rank subdomain
int kMin, kMax; //Constant min and max bounds of k-index accounting for only non-halos cells of the mpi_rank subdomain
int verticalDeformSwitch; //switch to use vertical coordinate deformation
float verticalDeformFactor; //factor to used under vertical deformation (0.0-1.0)
float verticalDeformQuadCoeff; // quadratic term coefficient in the deformtion scheme (default = 0.0)

float dX, dY, dZ; //reference computational model coordinate resolution
float dXi, dYi, dZi; //inverse of the reference computational model coordinate resolution

/* array fields */
float *xPos;  /* Cell-center position in x (meters) */
float *yPos;  /* Cell-center position in y (meters) */
float *zPos;  /* Cell-center position in z (meters) */
float *topoPosGlobal; /*Topography elevation (z in meters) at the cell center position in x and y. (Global domain) */
float *topoPos; /*Topography elevation (z in meters) at the cell center position in x and y. (per-rank domain) */

float *J31;      // dz/d_xi
float *J32;      // dz/d_eta
float *J33;      // dz/d_zeta

float *D_Jac;    //Determinant of the Jacobian  (called scale factor i.e. if d_xi=d_eta=d_zeta=1, then cell volume)
float *invD_Jac; //inverse Determinant of the Jacobian 
 
/*######################------------------- GRID module function definitions ---------------------#################*/

/*----->>>>> int gridGetParams();       ----------------------------------------------------------------------
Obtain the complete set of parameters for the GRID module
*/
int gridGetParams(){
   int errorCode = GRID_SUCCESS;

   /*query for each GRID parameter */
   errorCode = queryFileParameter("gridFile", &gridFile, PARAM_OPTIONAL);
   errorCode = queryFileParameter("topoFile", &topoFile, PARAM_OPTIONAL);
   errorCode = queryIntegerParameter("Nx", &Nx, 1, INT_MAX, PARAM_MANDATORY);
   errorCode = queryIntegerParameter("Ny", &Ny, 1, INT_MAX, PARAM_MANDATORY);
   errorCode = queryIntegerParameter("Nz", &Nz, 1, INT_MAX, PARAM_MANDATORY);
   errorCode = queryIntegerParameter("Nh", &Nh, 0, INT_MAX, PARAM_MANDATORY);

   errorCode = queryFloatParameter("d_xi", &d_xi, FLT_MIN, FLT_MAX, PARAM_MANDATORY);
   errorCode = queryFloatParameter("d_eta", &d_eta, FLT_MIN, FLT_MAX, PARAM_MANDATORY);
   errorCode = queryFloatParameter("d_zeta", &d_zeta, FLT_MIN, FLT_MAX, PARAM_MANDATORY);
   errorCode = queryIntegerParameter("coordHorizHalos", &coordHorizHalos, 0, 1, PARAM_MANDATORY);
   errorCode = queryIntegerParameter("verticalDeformSwitch", &verticalDeformSwitch, 0, 1, PARAM_MANDATORY);
   errorCode = queryFloatParameter("verticalDeformFactor", &verticalDeformFactor, 0.0, 1.0, PARAM_MANDATORY);
   errorCode = queryFloatParameter("verticalDeformQuadCoeff", &verticalDeformQuadCoeff, -2.0, 2.0, PARAM_MANDATORY);
   
   return(errorCode);
} //end gridGetParams()

/*----->>>>> int gridInit();       ----------------------------------------------------------------------
Used to broadcast and print parameters, allocate memory, and initialize configuration settings for  the GRID module.
*/
int gridInit(){
   int errorCode = GRID_SUCCESS;
   int strLength;
   int ioerrorCode = 0; 
    
   if(mpi_rank_world == 0){
      printComment("GRID parameters---");
      printParameter("gridFile", "A file containing a complete grid specification");
      printParameter("topoFile", "A file containing topography (surface elevation in meters ASL)");
      printParameter("Nx", "Number of discretised domain elements in the x (zonal) direction.");
      printParameter("Ny", "Number of discretised domain elements in the y (meridional) direction.");
      printParameter("Nz", "Number of discretised domain elements in the z (vertical) direction.");
      printParameter("Nh", "Number of halo cells to be used (dependent on largest stencil extent)."); 
      printParameter("d_xi", "Computational domain fixed resolution in the 'i' direction."); 
      printParameter("d_eta", "Computational domain fixed resolution in the 'j' direction."); 
      printParameter("d_zeta", "Computational domain fixed resolution in the 'k' direction."); 
      printParameter("coordHorizHalos", "switch to setup coordiante halos as periodic=1 or gradient-following=0."); 
      printParameter("verticalDeformSwitch", "switch to use vertical coordinate deformation 0=off, 1=on"); 
      printParameter("verticalDeformFactor", "deformation factor (0.0=max compression,  1.0=no compression)"); 
      printParameter("verticalDeformQuadCoeff", "deformation factor (0.0=max compression,  1.0=no compression)"); 
   } //end if(mpi_rank_world == 0)
   /*Broadcast the parameters across mpi_ranks*/
   // gridFile-------------- 
#ifdef DEBUG
  MPI_Barrier(MPI_COMM_WORLD);  
  printf("mpi_rank_world--%d/%d BCAST-ing from  gridInit!\n",mpi_rank_world, mpi_size_world);
   fflush(stdout);
#endif
   if(mpi_rank_world == 0){
      if(gridFile != NULL){
         strLength = strlen(gridFile)+1;
      }else{
         strLength = 0;
      }
   } //end if(mpi_rank_world == 0)
   MPI_Bcast(&strLength, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);
   if(mpi_rank_world != 0){
      gridFile = (char *) malloc(strLength*sizeof(char));
   } //if a non-root mpi_rank
   if(strLength > 0){
     MPI_Bcast(gridFile, strLength, MPI_CHARACTER, 0, MPI_COMM_WORLD);
   }else{
     if(mpi_rank_world != 0){
       gridFile = NULL;
     } //if a non-root mpi_rank
   }
   // topoFile --------------
   if(mpi_rank_world == 0){
      if(topoFile != NULL){
         strLength = strlen(topoFile)+1;
      }
   } //end if(mpi_rank_world == 0)
   MPI_Bcast(&strLength, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);
   if(mpi_rank_world != 0){
      topoFile = (char *) malloc(strLength*sizeof(char));
   } //if a non-root mpi_rank
   if(strLength > 0){
     MPI_Bcast(topoFile, strLength, MPI_CHARACTER, 0, MPI_COMM_WORLD);
   }else{
     if(mpi_rank_world != 0){
       topoFile = NULL;
     } //if a non-root mpi_rank
   }
   // non-string parameters   --------------------- 
   MPI_Bcast(&Nx, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);
   MPI_Bcast(&Ny, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);
   MPI_Bcast(&Nz, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);
   MPI_Bcast(&Nh, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);
   MPI_Bcast(&d_xi, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&d_eta, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&d_zeta, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&coordHorizHalos, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);
   MPI_Bcast(&verticalDeformSwitch, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);
   MPI_Bcast(&verticalDeformFactor, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&verticalDeformQuadCoeff, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

   /* Sanity check the implicit domain decomposition rules that numprocX/Y evenly divides Nx/y */
   if((Nx % numProcsX !=0)||(Ny % numProcsY !=0)){
     if(Nx % numProcsX !=0){
       printf("CRITICAL ERROR: Nx is not an exact multiple of numProcsX, Nx = %d, numProcsX = %d\n",Nx,numProcsX);
     }
     if(Ny % numProcsY !=0){
       printf("CRITICAL ERROR: Ny is not an exact multiple of numProcsY, Ny = %d, numProcsY = %d\n",Ny,numProcsY);
     }
    fflush(stdout);
    errorCode = GRID_DECOMPOSE_FAIL;
   }

   /*set the reference computational domain resolutions*/
   dX = d_xi;
   dY = d_eta;
   dZ = d_zeta;
   /* Set the inverse values */
   dXi = 1.0/dX;
   dYi = 1.0/dY;
   dZi = 1.0/dZ;

   /* Set the per-rank domain decomposition loop indexing limits */
   Nxp = Nx/numProcsX + ( Nx % numProcsX !=0);
   Nyp = Ny/numProcsY + ( Ny % numProcsY !=0);
   Nzp = Nz;
//#ifdef DEBUG
#if 1
   printf("Nxp = %d, Nyp = %d, Nzp = %d\n",Nxp,Nyp,Nzp);
   fflush(stdout);
#endif
   /* Set the constant index bounds for i,j, and k */
   iMin = Nh;
   iMax = Nxp+Nh;
   jMin = Nh;
   jMax = Nyp+Nh;
   kMin = Nh;
   kMax = Nzp+Nh;
//#ifdef DEBUG
#if 1
   printf("****Spatial loop-index Mins & Maxs ****  (i,j,k)-min = (%d,%d,%d) & (i,j,k)-max = (%d,%d,%d)\n",
                                                                              iMin,jMin,kMin,iMax,jMax,kMax);
#endif
   
   /* Allocate the GRID arrays */
     /* Coordinate Arrays */
   xPos = memAllocateFloat3DField(Nxp, Nyp, Nzp, Nh, "xPos");
   yPos = memAllocateFloat3DField(Nxp, Nyp, Nzp, Nh, "yPos");
   zPos = memAllocateFloat3DField(Nxp, Nyp, Nzp, Nh, "zPos");
   topoPos = memAllocateFloat2DField(Nxp, Nyp, Nh, "topoPos");
   topoPosGlobal = memAllocateFloat2DField(Nx, Ny, 0, "topoPos");
     /* Metric Tensors Fields */
   J31 = memAllocateFloat3DField(Nxp, Nyp, Nzp, Nh, "J31");
   J32 = memAllocateFloat3DField(Nxp, Nyp, Nzp, Nh, "J32");
   J33 = memAllocateFloat3DField(Nxp, Nyp, Nzp, Nh, "J33");
   D_Jac = memAllocateFloat3DField(Nxp, Nyp, Nzp, Nh, "D_Jac");
   invD_Jac = memAllocateFloat3DField(Nxp, Nyp, Nzp, Nh, "invD_Jac");

   /*Register these fields with the IO module*/
   /********* FOR THE MOMENT THESE SHOULD BE STRICTLY GLOBAL DOMAIN VARIABLE FIELDS ********/
   if(errorCode == GRID_SUCCESS){ 
     ioerrorCode = ioRegisterVar("xPos", "float", 4, dims4d, xPos);
     ioerrorCode = ioRegisterVar("yPos", "float", 4, dims4d, yPos);
     ioerrorCode = ioRegisterVar("zPos", "float", 4, dims4d, zPos);
     ioerrorCode = ioRegisterVar("topoPos", "float", 3, dims2dTD, topoPos);
     printf("gridInit:topoPos stored at %p, has been registered with IO.\n",
            &topoPosGlobal);
     fflush(stdout);
     if(ioerrorCode!=0){
       printf("Error in registering GRID module coordinate fields with IO.\n");
       fflush(stdout);
       errorCode = GRID_IO_CALL_FAIL;
     }
   } // end if errorCode indicates no errors thus far
#ifdef DEBUG
//#if 1
   errorCode = ioRegisterVar("D_Jac", "float", 4, dims4d, D_Jac);
   errorCode = ioRegisterVar("invD_Jac", "float", 4, dims4d, invD_Jac);
   errorCode = ioRegisterVar("J31", "float", 4, dims4d, J31);
   errorCode = ioRegisterVar("J32", "float", 4, dims4d, J32);
   errorCode = ioRegisterVar("J33", "float", 4, dims4d, J33);
#endif 
  
#ifdef DEBUG
//#if 1
   printf("mpi_rank_world %d/%d: Finished gridInit()!\n",mpi_rank_world,mpi_size_world);
   fflush(stdout);
#endif
   /* Done */
   return(errorCode);
} //end gridInit()

/*----->>>>> int gridSecondaryPreperations();   -------------------------------------------------------------------
 * Used to read a gridFile and/or calculate the metric tensor fields.
 * */
int gridSecondaryPreparations(){
   int errorCode = GRID_SUCCESS;
   int i, j, k, ij, ji, ijk;
   int iGlobal,jGlobal,ijGlobal;
   int ijkTarg, ijkTargm1, ijkTargp1;
   int iStride, jStride, kStride;
   float dX_tmp,dY_tmp,dZ_tmp;
   float zbot,ztop;
   FILE* fptrGrid;
   int tmpNx,tmpNy;
   float *tmpInField;
#ifdef DEBUG
   if((gridFile != NULL)&&(inFile == NULL)){  //If a gridFile exists but no inFile 
     /* Open the gridFile if one has been specified 
     *and read the appropriate grid coordinate/topo fields on each mpi_rank */
     //Call something to read only the grid-related fields (xPos, yPos, zPos, and topoPos)
     errorCode = ioReadNetCDFgridFile(gridFile, Nx, Ny, Nz, Nh); 
     //Do something to fill halos
     
//#if 1
//i = 128;
//j = 128;
//k = 1;
for(i=iMin; i < iMax; i++){
  for(j=jMin; j < jMax; j++){
    for(k=kMin; k < kMax; k++){
      ijk = i*(Nyp+2*Nh)*(Nzp+2*Nh)+j*(Nzp+2*Nh)+k;
      ij = i*(Nyp+2*Nh)+j;
      printf("(xPos,yPos,zPos,topo)[%d,%d,%d] = (%f,%f,%f,%f) \n",i,j,k,xPos[ijk],yPos[ijk],zPos[ijk],topoPos[ij]); 
    } //end for(k...
  } // end for(j...
} // end for(i...

   } //end if gridFile != NULL
#endif

   if(topoFile != NULL){  //If a topoFile exists read it. 
     //Perform a serial read of the topography with the root mpi rank
     if(mpi_rank_world == 0){
       //Malloc a temporary array of floats
       tmpInField = (float *) malloc(Nx*Ny*sizeof(float));
       // Open the binary file for reading
       if((fptrGrid=fopen(topoFile, "rb")) == NULL){
         printf("ERROR: Falied to open topoFile, %s\n", topoFile);
         fflush(stdout);
         return(TOPOFILE_GRID_FAIL);
       }//end if unable to open the file 
       // Read the x,y extents
       printf("Reading topoFile extents...\n");
       fflush(stdout);
       fread(&tmpNx, sizeof(int), 1, fptrGrid);
       fread(&tmpNy, sizeof(int), 1, fptrGrid);
       printf("topoFile extents (tmpNx,tmpNy) = (%d,%d).\n", tmpNx,tmpNy);
       fflush(stdout);
       if((tmpNx == Nx) && (tmpNy == Ny)){
         // Read the data
         fread(tmpInField, sizeof(float), tmpNx*tmpNy, fptrGrid);
         printf("Successful topography read.\n");
         fflush(stdout);
         for(i=0; i < Nx; i++){
           for(j=0; j < Ny; j++){
             ij = i*(Ny)+j;
             ji =j*(Nx)+i;
             topoPosGlobal[ij] = tmpInField[ji]; //out-of-place transpose the array elements 
#ifdef DEBUG
             printf("topoPosGlobal[%d,%d] = %f\n", i,j,topoPosGlobal[ij]);
#endif
           } //end for j
         } //end for i
          
       }else{
         printf("ERROR: topoFile extents (tmpNx,tmpNy) = (%d,%d) != (Nx,Ny) = (%d,%d).\n", tmpNx,tmpNy,Nx,Ny);
         fflush(stdout);
         return(TOPOFILE_GRID_FAIL);
       } //end if extents are correct --else

       // Close the file 
       fclose(fptrGrid);
       //Free the temporary field used to read the data.
       free(tmpInField);
   
     } //end if (mpi_rank_world == 0)
     //Broadcast the topoPos array.
     MPI_Bcast(topoPosGlobal, Nx*Ny, MPI_FLOAT, 0, MPI_COMM_WORLD);
   }else{ //Set topography = 0 everywhere
     for(i=0; i < Nx; i++){
       for(j=0; j < Ny; j++){
          ij = (i)*(Ny)+(j);
         topoPosGlobal[ij] = 0.0;
       } // end for(j...
     } // end for(i...
   } //if(topoFile != NULL) --else--
   /* Initialize the per-Rank topoPos array From the Global version and setup halos*/
   for(i=iMin-Nh; i < iMax+Nh; i++){
     for(j=jMin-Nh; j < jMax+Nh; j++){
        ij = (i)*(Nyp+2*Nh)+j;
        iGlobal = rankXid*Nxp+i-Nh;
        if(iGlobal < 0){
          iGlobal=0;
        }
        if(iGlobal > Nx-1){
          iGlobal=Nx-1;
        }
        jGlobal = rankYid*Nyp+j-Nh;
        if(jGlobal < 0){
          jGlobal=0;
        }
        if(jGlobal > Ny-1){
          jGlobal=Ny-1;
        }
        ijGlobal = iGlobal*Ny+jGlobal;
        topoPos[ij] = topoPosGlobal[ijGlobal];
#ifdef GRID_DEBUG
        if((iGlobal>=126)&&(iGlobal<=129)&&(jGlobal>=123)&&(jGlobal<=126)){
         printf("mpi_rank_world %d/%d: topoGlobal(%d,%d) & topo(%d,%d) =  (%f & %f)\n",
                mpi_rank_world,mpi_size_world,iGlobal,jGlobal,i,j,topoPosGlobal[ijGlobal],topoPos[ij]);
         fflush(stdout);
        }
#endif
     } //end for i
   } //end for j
 
   /*If no coordinate frame work is specified through a gridFile or inputFile, setup xPos,yPos,zPos */
   if((gridFile == NULL)&&(inFile == NULL)){
     printf("**** Coordinate generation ****  (i,j,k)-extents = (%d,%d,%d) & (i,j,k)-deltas = (%f,%f,%f)\n",Nx,Ny,Nz,d_xi,d_eta,d_zeta);
     fflush(stdout);

     /* Set the coordinate bounds */
     ztop = ((float) Nz)*d_zeta-0.5*d_zeta; //Note the rectilinear vertical resolution max
     zbot = 0.0;    //Note the vertical resolution min
     for(i=iMin; i < iMax; i++){
       for(j=jMin; j < jMax; j++){
         for(k=kMin; k < kMax; k++){
           ij = i*(Nyp+2*Nh)+j;
           ijk = i*(Nyp+2*Nh)*(Nzp+2*Nh)+j*(Nzp+2*Nh)+k;
           xPos[ijk] = ((float)((rankXid)*Nxp)
                       +(float)(i-Nh))*d_xi+0.5*d_xi;
           yPos[ijk] = ((float)((rankYid)*Nyp)
                       +(float)(j-Nh))*d_eta+0.5*d_eta;
           zPos[ijk] = ((float) k-Nh)*d_zeta+0.5*d_zeta;    //For fixed vertical resolution
           if(verticalDeformSwitch == 1){  //For non-uniform vertical resiolution  
             zPos[ijk] = zDeform(zPos[ijk],zbot,ztop);
           } //endif verticalDeformSwitch == 1
           zPos[ijk] = zPos[ijk]*((ztop-topoPos[ij])/ztop)+topoPos[ij]; //Compress this column for terrain-following
         } //end for(k...
       } // end for(j...
     } // end for(i...
   } //end if no gridFile or inputFile
   if(coordHorizHalos==0){
     printf("mpi_rank_world--%d/%d grid-Coord-gen(), coordHorizHalos = %d!\n",mpi_rank_world, mpi_size_world,coordHorizHalos);
     fflush(stdout);
     /*setup non-periodic, interior gradient following coordinates out each boundary*/
     iStride = (Ny+2*Nh)*(Nz+2*Nh);
     jStride = (Nz+2*Nh);
     kStride = 1;
     for(j=jMin; j < jMax; j++){
       for(k=kMin; k < kMax; k++){
         for(i=iMin-Nh; i < iMin; i++){
            ijk = i*(Nyp+2*Nh)*(Nzp+2*Nh)+j*(Nzp+2*Nh)+k;
            ijkTarg = (iMin)*iStride + j*jStride + k*kStride;
            ijkTargp1 = ijkTarg+iStride;
            dX_tmp = xPos[ijkTargp1]-xPos[ijkTarg];
            xPos[ijk] = xPos[ijkTarg]-(Nh-i)*dX_tmp;
            yPos[ijk] = yPos[ijkTarg];
            zPos[ijk] = zPos[ijkTarg];
         } //end for(i...
         for(i=iMax; i < iMax+Nh; i++){
            ijk = i*(Nyp+2*Nh)*(Nzp+2*Nh)+j*(Nzp+2*Nh)+k;
            ijkTarg = ((iMax-1))*iStride + j*jStride + k*kStride;
            ijkTargm1 = ijkTarg-iStride;
            dX_tmp = xPos[ijkTarg]-xPos[ijkTargm1];
            xPos[ijk] = xPos[ijk-iStride]+dX_tmp;
            yPos[ijk] = yPos[ijkTarg];
            zPos[ijk] = zPos[ijkTarg];
         } //end for(i...
       } // end for(k...
     } // end for(j...
     for(i=iMin-Nh; i < iMax+Nh; i++){
       for(k=kMin; k < kMax; k++){
         for(j=jMin-Nh; j < jMin; j++){
            ijk = i*(Nyp+2*Nh)*(Nzp+2*Nh)+j*(Nzp+2*Nh)+k;
            ijkTarg = i*iStride + (jMin)*jStride + k*kStride;
            ijkTargp1 = ijkTarg+jStride;
            dY_tmp = yPos[ijkTargp1]-yPos[ijkTarg];
            yPos[ijk] = yPos[ijkTarg]-(Nh-j)*dY_tmp;
            xPos[ijk] = xPos[ijkTarg];
            zPos[ijk] = zPos[ijkTarg];
         } //end for(j...
         for(j=jMax; j < jMax+Nh; j++){
            ijk = i*(Nyp+2*Nh)*(Nzp+2*Nh)+j*(Nzp+2*Nh)+k;
            ijkTarg = i*iStride + (jMax-1)*jStride + k*kStride; 
            ijkTargm1 = ijkTarg-jStride;
            dY_tmp = yPos[ijkTarg]-yPos[ijkTargm1];
            yPos[ijk] = yPos[ijk-jStride]+dY_tmp;
            xPos[ijk] = xPos[ijkTarg];
            zPos[ijk] = zPos[ijkTarg];
         } //end for(j...
       } // end for(k...
     } // end for(i...
   }else{ // then coordHorizHalos == 1
     printf("mpi_rank_world--%d/%d grid-Coord-gen(), else-coordHorizHalos = %d!\n",mpi_rank_world, mpi_size_world,coordHorizHalos);
     fflush(stdout);
     if(mpi_size_world==1){ 
       errorCode = singleRankGridHaloInit();
     } // if mpi_size_world==1 
   }//end if-else coordHorizHalos == 0

   /*Update the halos for the coordinate fields */
   errorCode = fempi_XdirHaloExchange(Nxp, Nyp, Nzp, Nh,xPos); 
   errorCode = fempi_YdirHaloExchange(Nxp, Nyp, Nzp, Nh,xPos); 
   errorCode = fempi_XdirHaloExchange(Nxp, Nyp, Nzp, Nh,yPos); 
   errorCode = fempi_YdirHaloExchange(Nxp, Nyp, Nzp, Nh,yPos); 
   errorCode = fempi_XdirHaloExchange(Nxp, Nyp, Nzp, Nh,zPos); 
   errorCode = fempi_YdirHaloExchange(Nxp, Nyp, Nzp, Nh,zPos); 
   /*Force global domain boundary ranks to adjust said boundary's halo cells appropriately 
   *  since coordinates are never "periodic".
   */
   //Western boundary
   if(mpi_XloBndyRank==1){
     for(i=iMin-Nh; i < iMin; i++){
       for(j=jMin-Nh; j < jMax+Nh; j++){
         for(k=kMin-Nh; k < kMax+Nh; k++){
           ijk   = i*(Nyp+2*Nh)*(Nzp+2*Nh)+j*(Nzp+2*Nh)+k;
           ijkTarg   = Nh*(Nyp+2*Nh)*(Nzp+2*Nh)+j*(Nzp+2*Nh)+k;
           xPos[ijk] = xPos[ijkTarg]-(Nh-i)*d_xi;  
           yPos[ijk] = yPos[ijkTarg];  
           zPos[ijk] = zPos[ijkTarg];  
         } //end for(k...
       } // end for(j...
     } // end for(i...
   } //end if(mpi_XloBndyRank==1)
   //Eastern boundary
   if(mpi_XhiBndyRank==1){
     for(i=iMax; i < iMax+Nh; i++){
       for(j=jMin-Nh; j < jMax+Nh; j++){
         for(k=kMin-Nh; k < kMax+Nh; k++){
           ijk   = i*(Nyp+2*Nh)*(Nzp+2*Nh)+j*(Nzp+2*Nh)+k;
           ijkTarg   = (iMax-1)*(Nyp+2*Nh)*(Nzp+2*Nh)+j*(Nzp+2*Nh)+k;
           xPos[ijk] = xPos[ijkTarg]+(i-iMax+1)*d_xi;  
           yPos[ijk] = yPos[ijkTarg];  
           zPos[ijk] = zPos[ijkTarg];  
         } //end for(k...
       } // end for(j...
     } // end for(i...
   } //end if(mpi_XhiBndyRank==1)
   //Southern boundary
   if(mpi_YloBndyRank==1){
     for(i=iMin-Nh; i < iMax+Nh; i++){
       for(j=jMin-Nh; j < jMin; j++){
         for(k=kMin-Nh; k < kMax+Nh; k++){
           ijk   = i*(Nyp+2*Nh)*(Nzp+2*Nh)+j*(Nzp+2*Nh)+k;
           ijkTarg   = i*(Nyp+2*Nh)*(Nzp+2*Nh)+Nh*(Nzp+2*Nh)+k;
           yPos[ijk] = yPos[ijkTarg]-(Nh-j)*d_eta;  
           xPos[ijk] = xPos[ijkTarg];  
           zPos[ijk] = zPos[ijkTarg];  
         } //end for(k...
       } // end for(j...
     } // end for(i...
   } //end if(mpi_XloBndyRank==1)
   //Northern boundary
   if(mpi_YhiBndyRank==1){
     for(i=iMin-Nh; i < iMax+Nh; i++){
       for(j=jMax; j < jMax+Nh; j++){
         for(k=kMin-Nh; k < kMax+Nh; k++){
           ijk   = i*(Nyp+2*Nh)*(Nzp+2*Nh)+j*(Nzp+2*Nh)+k;
           ijkTarg   = i*(Nyp+2*Nh)*(Nzp+2*Nh)+(jMax-1)*(Nzp+2*Nh)+k;
           yPos[ijk] = yPos[ijkTarg]+(j-jMax+1)*d_eta;  
           xPos[ijk] = xPos[ijkTarg];  
           zPos[ijk] = zPos[ijkTarg];  
         } //end for(k...
       } // end for(j...
     } // end for(i...
   } //end if(mpi_YhiBndyRank==1)
   
   /*Finally, set the below-ground, and above-ceiling halos as interior gradient matching...*/
#ifdef DEBUG
   printf("mpi_rank_world--%d/%d coordHalos, ground-ceiling Boundaries!\n",mpi_rank_world, mpi_size_world);
   fflush(stdout);
#endif
   iStride = (Nyp+2*Nh)*(Nzp+2*Nh);
   jStride = (Nzp+2*Nh);
   kStride = 1;
   for(i=iMin-Nh; i < iMax+Nh; i++){
     for(j=jMin-Nh; j < jMax+Nh; j++){
       for(k=kMin-Nh; k < kMin; k++){
         ijk = i*(Nyp+2*Nh)*(Nzp+2*Nh)+j*(Nzp+2*Nh)+k;
         ijkTarg = i*iStride + j*jStride + (kMin)*kStride;
         ijkTargp1 = ijkTarg+kStride;
         dZ_tmp = zPos[ijkTargp1]-zPos[ijkTarg];
         zPos[ijk] = zPos[ijkTarg]-(Nh-k)*dZ_tmp;
         xPos[ijk] = xPos[ijkTarg];
         yPos[ijk] = yPos[ijkTarg];
       } //end for(k...
       for(k=kMax; k < kMax+Nh; k++){
         ijk = i*(Nyp+2*Nh)*(Nzp+2*Nh)+j*(Nzp+2*Nh)+k;
         ijkTarg = i*iStride + j*jStride + (kMax-1)*kStride;
         ijkTargm1 = ijkTarg-kStride;
         dZ_tmp = zPos[ijkTarg]-zPos[ijkTargm1];
         zPos[ijk] = zPos[ijkTarg]+(k-kMax+1)*dZ_tmp;
         xPos[ijk] = xPos[ijkTarg];
         yPos[ijk] = yPos[ijkTarg];
       } //end for(k...
     } // end for(j...
   } // end for(i...
#ifdef DEBUG
   printf("xpos[%d,%d,%d] = %f, ((float) %d-Nh)*d_eta = %f \n",i,j,k,xPos[ijk],i,((float) i-Nh)*d_xi); 
   fflush(stdout);
#endif


   /* Compute the metric tensor fields (Jacobians), scale factor, and inverse scale factor */
   errorCode = calculateJacobians(); 

   /* Done */
   return(errorCode);

} // end gridSecondaryPreparations()

/*----->>>>> int calculateJacobians();       ----------------------------------------------------------------------
Used to calculate the metric tensor elements for a generalized curvilinear coordinate system
*/
int calculateJacobians(){
   int errorCode = GRID_SUCCESS;
   float T[3][3]; //The metric tensor at each i,j,k location
   float determinant;  //The determinant of the metric tensor
   int i, j, k, ijk, ip1jk, im1jk, ijp1k, ijm1k, ijkp1, ijkm1;
   int iBnd, iT,jT,kT, ilo,jlo,klo, ihi,jhi,khi;
   int ijkDest,ijkTarget;
   float inv_2dxi;
   float inv_2deta;
   float inv_2dzeta;

#ifdef DEBUG
   printf("mpi_rank_world--%d/%d coordHalos, calculateJacaobians()!\n",mpi_rank_world, mpi_size_world);
   fflush(stdout);
#endif
   /* setup constants */
   inv_2dxi = 1.0/(2.0*d_xi);
   inv_2deta = 1.0/(2.0*d_eta);
   inv_2dzeta = 1.0/(2.0*d_zeta);

   /*For each i,j,k gridpoint caluclate the Jacobian elements and 
   * both the determinant and inverse determinant of the metric tensor 
   */
   for(i=iMin; i < iMax; i++){
     for(j=jMin; j < jMax; j++){
       for(k=kMin; k < kMax; k++){
         /* Build the metric tensor via second-order centered finite difference */
         ijk   = i*(Nyp+2*Nh)*(Nzp+2*Nh)+j*(Nzp+2*Nh)+k;
         ip1jk = (i+1)*(Nyp+2*Nh)*(Nzp+2*Nh)+j*(Nzp+2*Nh)+k;
         im1jk = (i-1)*(Nyp+2*Nh)*(Nzp+2*Nh)+j*(Nzp+2*Nh)+k;
         ijp1k = i*(Nyp+2*Nh)*(Nzp+2*Nh)+(j+1)*(Nzp+2*Nh)+k;
         ijm1k = i*(Nyp+2*Nh)*(Nzp+2*Nh)+(j-1)*(Nzp+2*Nh)+k;
         ijkp1 = i*(Nyp+2*Nh)*(Nzp+2*Nh)+j*(Nzp+2*Nh)+(k+1);
         ijkm1 = i*(Nyp+2*Nh)*(Nzp+2*Nh)+j*(Nzp+2*Nh)+(k-1);
         
         
         T[0][0] = (xPos[ip1jk]-xPos[im1jk])*inv_2dxi;        
         T[0][1] = (yPos[ip1jk]-yPos[im1jk])*inv_2dxi;        
         T[0][2] = (zPos[ip1jk]-zPos[im1jk])*inv_2dxi;        
         T[1][0] = (xPos[ijp1k]-xPos[ijm1k])*inv_2deta;        
         T[1][1] = (yPos[ijp1k]-yPos[ijm1k])*inv_2deta;        
         T[1][2] = (zPos[ijp1k]-zPos[ijm1k])*inv_2deta;        
         T[2][0] = (xPos[ijkp1]-xPos[ijkm1])*inv_2dzeta;        
         T[2][1] = (yPos[ijkp1]-yPos[ijkm1])*inv_2dzeta;        
         T[2][2] = (zPos[ijkp1]-zPos[ijkm1])*inv_2dzeta;        
         /*Calculate the determinant */
         determinant = (T[1][1]*T[2][2] - T[1][2]*T[2][1])*T[0][0]
                      -(T[0][1]*T[2][2] - T[0][2]*T[2][1])*T[1][0]
                      +(T[0][1]*T[1][2] - T[0][2]*T[1][1])*T[2][0];
         /* ensure determinant is not zero */
         if(fabsf(determinant) < 1e-6){
#ifdef DEBUG
            printf("determinant < 0 for %d\n",ijk);
            printf("(xPos,yPos,zPos)[%d,%d,%d] = (%f,%f,%f)\n",i,j,k,xPos[ijk],yPos[ijk],zPos[ijk]);
#endif
           determinant = 1e-6; 
         }
         /* Set the determinant and inverse determinant for this ijk location */    
         D_Jac[ijk] = determinant;
         invD_Jac[ijk] = 1.0/determinant;
         //d(x,y,z)/d_zeta
         J31[ijk] =  (T[1][0]*T[2][1] - T[1][1]*T[2][0])*invD_Jac[ijk];
         J32[ijk] = -(T[0][0]*T[2][1] - T[0][1]*T[2][0])*invD_Jac[ijk];
         J33[ijk] =  (T[0][0]*T[1][1] - T[0][1]*T[1][0])*invD_Jac[ijk];
#ifdef DEBUG
         if(isnan(D_Jac[ijk])){
            printf("D_Jac[%d,%d,%d] = %f\n",i,j,k,D_Jac[ijk]);
         }
         /*if(isnan(invD_Jac[ijk])){
            printf("invD_Jac[%d,%d,%d] = %f\n",i,j,k,D_Jac[ijk]);
         }*/
#endif
       } //end for(k...
     } // end for(j...
   } // end for(i...

#ifdef DEBUG
   printf("mpi_rank_world--%d/%d coordHalos, calculateJacaobians()-Boundaries!\n",mpi_rank_world, mpi_size_world);
   fflush(stdout);
#endif
/* TODO technically these only work for periodic domain.  Since we currently use strictly constant horizontal and vertical spacing they work for now. */
   /*lower i-index halos*/
   for(iBnd=0; iBnd < 6; iBnd++){
#ifdef DEBUG
      printf("mpi_rank_world--%d/%d calculateJacaobians()-Boundaries, iBnd= %d!\n",mpi_rank_world, mpi_size_world,iBnd);
      fflush(stdout);
#endif
     ilo = iMin-Nh; 
     ihi = iMax+Nh;
     jlo = jMin-Nh; 
     jhi = jMax+Nh;
     klo = kMin-Nh; 
     khi = kMax+Nh;
     switch(iBnd){
       case 0:
         ihi = iMin;
         break;
       case 1:
         ilo = iMax;
         break;
       case 2:
         jhi = jMin;
         break;
       case 3:
         jlo = jMax;
         break;
       case 4:
         khi = kMin;
         break;
       case 5:
         klo = kMax;
         break;
       default:
         printf("calculateJacobians: switch(iBnd) for iBnd = INVALID\n");
         break;
     }// end switch iBnd
     for(i=ilo; i < ihi; i++){
       for(j=jlo; j < jhi; j++){
         for(k=klo; k < khi; k++){
           iT = i;
           jT = j;
           kT = k;
           switch(iBnd){
             case 0:
               iT = iMin;
               break;
             case 1:
               iT = iMax-1;
               break;
             case 2:
               jT = jMin;
               break;
             case 3:
               jT = jMax-1;
               break;
             case 4:
               kT = kMin;
               break;
             case 5:
               kT = kMax-1;
               break;
             default:
               printf("calculateJacobians: switch(iBnd) for iBnd = INVALID\n");
               break;
           }// end switch iBnd
           ijkDest   = i*(Nyp+2*Nh)*(Nzp+2*Nh)+j*(Nzp+2*Nh)+k;
           ijkTarget = iT*(Nyp+2*Nh)*(Nzp+2*Nh)+jT*(Nzp+2*Nh)+kT;
           /* Set the determinant and inverse determinant for this ijk location */    
           D_Jac[ijkDest] = D_Jac[ijkTarget];
           invD_Jac[ijkDest] = invD_Jac[ijkTarget];
           //d(x,y,z)/d_zeta
           J31[ijkDest] = J31[ijkTarget];
           J32[ijkDest] = J32[ijkTarget];
           J33[ijkDest] = J33[ijkTarget];
         } //end for(k...
       } // end for(j...
     } // end for(i...
   } // end for(iBnd...
#ifdef DEBUG
   printf("mpi_rank_world--%d/%d calculateJacaobians()-Boundaries, complete!\n",mpi_rank_world, mpi_size_world);
   fflush(stdout);
#endif
#ifdef DEBUG
   i = 2;
   j = 1;
   k = 1;
   ijk   = i*(Nyp+2*Nh)*(Nzp+2*Nh)+j*(Nzp+2*Nh)+k;
   ip1jk = (i+1)*(Nyp+2*Nh)*(Nzp+2*Nh)+j*(Nzp+2*Nh)+k;
   im1jk = (i-1)*(Nyp+2*Nh)*(Nzp+2*Nh)+j*(Nzp+2*Nh)+k;
   ijp1k = i*(Nyp+2*Nh)*(Nzp+2*Nh)+(j+1)*(Nzp+2*Nh)+k;
   ijm1k = i*(Nyp+2*Nh)*(Nzp+2*Nh)+(j-1)*(Nzp+2*Nh)+k;
   ijkp1 = i*(Nyp+2*Nh)*(Nzp+2*Nh)+j*(Nzp+2*Nh)+(k+1);
   ijkm1 = i*(Nyp+2*Nh)*(Nzp+2*Nh)+j*(Nzp+2*Nh)+(k-1);
   printf("\n");
   printf("For i =%d, j= %d, k=%d, ijk = %d, we have...\n",i,j,k,ijk);
   printf("(xPos,yPos,zPos)[%d,%d,%d] = (%f,%f,%f)\n",i,j,k,xPos[ijk],yPos[ijk],zPos[ijk]);
   printf("(xPos,yPos,zPos)[%d,%d,%d] = (%f,%f,%f)\n",i-1,j,k,xPos[im1jk],yPos[im1jk],zPos[im1jk]);
   printf("(xPos,yPos,zPos)[%d,%d,%d] = (%f,%f,%f)\n",i+1,j,k,xPos[ip1jk],yPos[ip1jk],zPos[ip1jk]);
   printf("(xPos,yPos,zPos)[%d,%d,%d] = (%f,%f,%f)\n",i,j-1,k,xPos[ijm1k],yPos[ijm1k],zPos[ijm1k]);
   printf("(xPos,yPos,zPos)[%d,%d,%d] = (%f,%f,%f)\n",i,j+1,k,xPos[ijp1k],yPos[ijp1k],zPos[ijp1k]);
   printf("(xPos,yPos,zPos)[%d,%d,%d] = (%f,%f,%f)\n",i,j,k-1,xPos[ijkm1],yPos[ijkm1],zPos[ijkm1]);
   printf("(xPos,yPos,zPos)[%d,%d,%d] = (%f,%f,%f)\n",i,j,k+1,xPos[ijkp1],yPos[ijkp1],zPos[ijkp1]);
   printf("D_Jac[%d] = %f\n",ijk,D_Jac[ijk]);
   printf("invD_Jac[%d] = %f\n",ijk,invD_Jac[ijk]);
   printf("J31[%d] = %f\n",ijk,J31[ijk]);
   printf("J32[%d] = %f\n",ijk,J32[ijk]);
   printf("J33[%d] = %f\n",ijk,J33[ijk]);
   printf("T[%d] = (%f,%f,%f),(%f,%f,%f),(%f,%f,%f)\n",ijk,T[0][0],T[0][1],T[0][2],
                                                           T[1][0],T[1][1],T[1][2],
                                                           T[2][0],T[2][1],T[2][2]);
   printf("\n");
   printf("i coordinate dependence--------------------------------\n");
   for(i=iMin-Nh; i < iMax+Nh; i++){
     j = jMin-1;
     k = kMin-1;
     ijk   = i*(Nyp+2*Nh)*(Nzp+2*Nh)+j*(Nzp+2*Nh)+k;
     printf("(xPos,yPos,zPos)[%d,%d,%d] = (%f,%f,%f)\n",i,j,k,xPos[ijk],yPos[ijk],zPos[ijk]);

   } // end for(i...
   printf("\n");
   printf("j coordinate dependence--------------------------------\n");
   for(j=jMin-Nh; j < jMax+Nh; j++){
     i = iMin-1;
     k = kMin-1;
     ijk   = i*(Nyp+2*Nh)*(Nzp+2*Nh)+j*(Nzp+2*Nh)+k;
     printf("(xPos,yPos,zPos)[%d,%d,%d] = (%f,%f,%f)\n",i,j,k,xPos[ijk],yPos[ijk],zPos[ijk]);
   } // end for(j...
   printf("\n");
   printf("k coordinate dependence--------------------------------\n");
   for(k=kMin-Nh; k < kMax+Nh; k++){
     i = iMin-1;
     j = jMin-1;
      ijk   = i*(Nyp+2*Nh)*(Nzp+2*Nh)+j*(Nzp+2*Nh)+k;
      printf("(xPos,yPos,zPos)[%d,%d,%d] = (%f,%f,%f)\n",i,j,k,xPos[ijk],yPos[ijk],zPos[ijk]);
   } //end for(k...

#endif
   /* Done */
   return(errorCode);
} //end calculateJacobians

/*----->>>>> int singleRankGridHaloInit();    ------------------------------------------------------------
* Used to setup xPos,yPos,zPos halos on all x-y boundaries 
* when under single-rank setup (i.e. mpi_size_world ==1).
*/
int singleRankGridHaloInit(){
   int errorCode = GRID_SUCCESS;
   int i,j,k,ijk,ijkTarg,ijkTargp1,ijkTargm1;
   int iStride,jStride,kStride;
   float dX_tmp,dY_tmp;

   /*Create horizontally periodic zPos and provide gradient matching xPos, yPos*/
   iStride = (Ny+2*Nh)*(Nz+2*Nh);
   jStride = (Nz+2*Nh);
   kStride = 1;
   /*Western/Eastern boundaries*/
   printf("mpi_rank_world--%d/%d coordHalos, Western-Eastern Boundaries!\n",mpi_rank_world, mpi_size_world);
   fflush(stdout);
   for(j=jMin; j < jMax; j++){
     for(k=kMin; k < kMax; k++){
       /*Western*/
       for(i=iMin-Nh; i < iMin; i++){
          ijk = i*iStride+j*jStride+k*kStride;
          ijkTarg = (iMax-Nh+i)*iStride + j*jStride+k*kStride;
          ijkTargm1 = ijkTarg-iStride;
          dX_tmp = xPos[ijkTarg]-xPos[ijkTargm1];
          xPos[ijk] = xPos[ijk+(Nh-i+1)*iStride]-(float)(Nh-i+1)*dX_tmp;
          yPos[ijk] = yPos[ijkTarg];  /*Take the real y-coordinate from the x-periodic neighbor*/
          zPos[ijk] = zPos[ijkTarg];  /*Take the real z-coordinate from the x-periodic neighbor*/
       } //end for(i...
       /*Eastern*/
       for(i=iMax; i < iMax+Nh; i++){
          ijk = i*iStride+j*jStride+k*kStride;
          ijkTarg = (iMin+(i-iMax))*iStride + j*jStride+k*kStride;
          ijkTargp1 = ijkTarg+iStride;
          dX_tmp = xPos[ijkTargp1]-xPos[ijkTarg];
          xPos[ijk] = xPos[ijk-iStride]+dX_tmp;
          yPos[ijk] = yPos[ijkTarg];  /*Take the real y-coordinate from the x-periodic neighbor*/
          zPos[ijk] = zPos[ijkTarg];  /*Take the real z-coordinate from the x-periodic neighbor*/
       } //end for(i...
     } // end for(k...
   } // end for(j...
   /*Southern/Northern boundaries*/
   printf("mpi_rank_world--%d/%d coordHalos, Southern-Northern Boundaries!\n",mpi_rank_world, mpi_size_world);
   fflush(stdout);
   for(i=iMin-Nh; i < iMax+Nh; i++){
     for(k=kMin; k < kMax; k++){
       /*Southern*/
       for(j=jMin-Nh; j < jMin; j++){
         ijk = i*iStride+j*jStride+k*kStride;
         ijkTarg = i*iStride + (jMax-Nh+j)*jStride+k*kStride;
         ijkTargm1 = ijkTarg-jStride;
         dY_tmp = yPos[ijkTarg]-yPos[ijkTargm1];
         xPos[ijk] = xPos[ijkTarg];  /*Take the real y-coordinate from the x-periodic neighbor*/
         yPos[ijk] = yPos[ijk+(Nh-j+1)*jStride]-(float)(Nh-j+1)*dY_tmp;
         zPos[ijk] = zPos[ijkTarg];  /*Take the real z-coordinate from the x-periodic neighbor*/
       } //end for(i...
       /*Northern*/
       for(j=jMax; j < jMax+Nh; j++){
         ijk = i*iStride+j*jStride+k*kStride;
         ijkTarg = i*iStride + (jMin+(j-jMax))*jStride+k*kStride;
         ijkTargp1 = ijkTarg+jStride;
         dY_tmp = yPos[ijkTargp1]-yPos[ijkTarg];
         xPos[ijk] = xPos[ijkTarg];  /*Take the real y-coordinate from the x-periodic neighbor*/
         yPos[ijk] = yPos[ijk-jStride]+dY_tmp;  /*Take the real y-coordinate from the x-periodic neighbor*/
         zPos[ijk] = zPos[ijkTarg];  /*Take the real z-coordinate from the x-periodic neighbor*/
       } //end for(i...
     } // end for(k...
   } // end for(i...
   
   return(errorCode);
}//end singleRankGridHaloInit()

/*----->>>>> float zDeform();       ----------------------------------------------------------------------
 * Used to calculate non-uniform resolution vertical coordinates.
 */
float zDeform(float zRect, float zGround, float zCeiling){
   float returnValue;
   float fCoeff;
   float c1,c2,c3;
   
   c1 = verticalDeformFactor;
   fCoeff = verticalDeformQuadCoeff;
   c2 = fCoeff*(1.0-c1)/zCeiling;
   c3 = (1.0-c2*zCeiling-c1)/(pow(zCeiling,2.0));
   returnValue = (c3*pow(zRect,3.0)+c2*pow(zRect,2.0)+c1*zRect)*(zCeiling-zGround)/zCeiling+zGround;

   return(returnValue);
} //end zDeform() 

/*----->>>>> int gridCleanup();       ----------------------------------------------------------------------
Used to free all malloced memory by the GRID module.
*/
int gridCleanup(){
   int errorCode = GRID_SUCCESS;

   /* Free any GRID module arrays */
    /* metric tensor fields */
   memReleaseFloat(J31); 
   memReleaseFloat(J32); 
   memReleaseFloat(J33); 
   memReleaseFloat(D_Jac); 
   memReleaseFloat(invD_Jac); 
    /* coordinate fields */
   memReleaseFloat(xPos); 
   memReleaseFloat(yPos); 
   memReleaseFloat(zPos);
   memReleaseFloat(topoPos);
   memReleaseFloat(topoPosGlobal);

   if(mpi_rank_world != 0){
     free(gridFile);
     free(topoFile);
   } 
   return(errorCode);

}//end gridCleanup()
