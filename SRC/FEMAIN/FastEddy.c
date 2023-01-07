/* FastEddy®: SRC/FEMAIN/FastEddy.c 
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
//#define NOTCUDA
//#define IO_OFF
//#define DEBUG_INITIALIZATION 

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <parameters.h>
#include <fempi.h>
#include <mem_utils.h>
#include <io.h>
#include <fecuda.h>
#include <grid.h>
#include <cuda_grid.h>
#include <hydro_core.h>
#include <cuda_hydroCore.h>
#include <time_integration.h>
#include <cuda_timeInt.h>

/***    main.c    ***/
int main(int argc, char **argv){
  int errorCode;
  char paramFile[MAXLEN];
  double mpi_t1, mpi_t2, mpi_t3, mpi_t4;
  char *desc; 
  int it, itTmp;

  
  /*** ---------------------------------------------------------------------------------------------- ***/
  /*** ---------------- Launch the MPI environment & Read the command line arguments  --------------- ***/
  /*** ---------------------------------------------------------------------------------------------- ***/
  /* Launch the FEMPI environment */
  errorCode = fempi_LaunchMPI(argc, argv);

  /* A synchronization point */
  MPI_Barrier(MPI_COMM_WORLD);
  mpi_t1 = MPI_Wtime();    //Mark the walltime to measure duration of initializations.
  
  if(mpi_rank_world == 0){
     /* Parse the command line arguments */
     if(argc != 2){
        printf("usage: %s paramFile \n",argv[0]);
        fflush(stdout);
        exit(0);
     }else{
        sscanf(argv[1],"%s", paramFile);
        printf("Obtaining parameters from %s\n", paramFile);
        fflush(stdout);
     }
  } //end if(mpi_rank == 0 )
 
  /*** ---------------------------------------------------------------------------------------------- ***/
  /*** -------------------- Read & Set FastEddy's Runtime Parameters by module ---------------------- ***/
  /*** ---------------------------------------------------------------------------------------------- ***/
  /* Initialize the PARAMETERS module */
  parameters_init();

  /* Root MPI-rank reads the parameters file */
  if(mpi_rank_world == 0){
    errorCode = parameters_readFile(paramFile);
    if(errorCode==PARAM_ERROR_FILE_NOT_FOUND){
      printf("Bailing out after failing to find the parameters file: %s\n", paramFile);
      fflush(stdout);
      exit(errorCode);
    }
    /*check the simulation description parameter*/
    errorCode = queryStringParameter("Description", &desc, PARAM_MANDATORY);
    /* Obtain the FEMPI module parameters */
    errorCode = fempi_GetParams();
    /* Obtain the MEM_UTILS module parameters */
    errorCode = mem_utilsGetParams();
    /*Obtain the  the IO module parameters*/
    errorCode = ioGetParams();
    /* Obtain the CUDA_AALES module parameters */
    errorCode = fecuda_GetParams();
    /* Obtain the GRID module parameters */
    errorCode = gridGetParams();
    /* Obtain the HYDRO_CORE module parameters */
    errorCode = hydro_coreGetParams();
    /* Obtain the TIME_INTEGRATION module parameters */
    errorCode = timeGetParams();
  }  //endif  mpi_rank == 0

  /* Print the description parameter and initialize the various modules used here*/
  if(mpi_rank_world == 0){
     printComment("time_test");
     printParameter("Description", "A description of this input file.");
  } //end if mpi_rank_world == 0   

  /*Seed the random number generator*/
  srand(mpi_rank_world+12345);

  /*** ---------------------------------------------------------------------------------------------- ***/
  /*** ------------------------- Initialize C-layer (Host/CPU layer) modules  ----------------------- ***/
  /*** ---------------------------------------------------------------------------------------------- ***/
  /*Initialize the FEMPI module*/
  errorCode = fempi_Init();
  /*Initialize the MEM_UTILS module*/
  errorCode = mem_utilsInit();
  /*Initialize the IO module*/
  errorCode = ioInit();
  /*** ++++ Initialize the mixed (C & CUDA)-layer module for exposing device/GPU functionality  +++++ ***/
#ifndef NOTCUDA 
  /*Initialize the FECUDA module*/
  errorCode = fecuda_Init();
#endif
  /*Initialize the GRID module*/
  MPI_Barrier(MPI_COMM_WORLD);  
#ifdef DEBUG_INITIALIZATION 
  printf("mpi_rank_world--%d/%d calling gridInit!\n",mpi_rank_world, mpi_size_world);
  fflush(stdout);
#endif
  errorCode = gridInit();
  MPI_Barrier(MPI_COMM_WORLD);
  if(errorCode != 0){
    printf("mpi_rank_world--%d/%d: ABORTING SIMULATION due to error returned by gridInit(): errorCode = %d\n", 
           mpi_rank_world, mpi_size_world, errorCode);
    fflush(stdout);
    exit(errorCode);
  } 
  /*Initialize the HYDRO_CORE module*/
#ifdef DEBUG_INITIALIZATION 
  printf("mpi_rank_world--%d/%d calling hydro_coreInit!\n",mpi_rank_world, mpi_size_world);
  fflush(stdout);
#endif
  errorCode = hydro_coreInit();
  /*Initialize the TIME_INTEGRATION module*/
  MPI_Barrier(MPI_COMM_WORLD); 
#ifdef DEBUG_INITIALIZATION 
  printf("mpi_rank_world--%d/%d calling timeInit!\n",mpi_rank_world, mpi_size_world);
  fflush(stdout);
#endif
  errorCode = timeInit();

  MPI_Barrier(MPI_COMM_WORLD); 
#ifdef DEBUG_INITIALIZATION 
  printf("mpi_rank_world--%d/%d Dumping parameter check results!\n",mpi_rank_world, mpi_size_world);
  fflush(stdout);
#endif
  
  /*** ---------------------------------------------------------------------------------------------- ***/
  /*** ---------------------- Log & Error-check the runtime parameters by module  ------------------- ***/
  /*** ---------------------------------------------------------------------------------------------- ***/
  /*Standard parameter reporting block -------------------------------------------------*/
  if(mpi_rank_world == 0){
     /* print the PARAMETERS outputBuffer to stdout */
     outputParameters(stdout);
     fflush(stdout);

     /* print out any extra parameters from the file that are not used. */
     printUnusedParameters();
     fflush(stdout);

     errorCode = getParameterErrors();
     if(errorCode != 0) {
        printf("There were %d error(s) in the input parameters that need to be corrected.\n", errorCode);
        fflush(stdout);
        exit(errorCode);
     } else {
        printf("Input parameters passed verification.\n");
        fflush(stdout);
     }// end if-else errorCode

  } // end if(mpi_rank...
  MPI_Barrier(MPI_COMM_WORLD); 
#ifdef DEBUG_INITIALIZATION 
  printf("mpi_rank_world--%d/%d Parameter Checking complete!!\n",mpi_rank_world,mpi_size_world);
  fflush(stdout);
#endif
  /*END  Standard parameter reporting block -------------------------------------------------*/

  /*** ---------------------------------------------------------------------------------------------- ***/
  /*** -------  Secondary (GRID runtime parameter dependent) allocations for MPI & IO buffers  ------ ***/
  /*** ---------------------------------------------------------------------------------------------- ***/
  /* Initialize any buffer arrays required for fempi_ Collective operations (scatters/gathers)...*/
  errorCode = fempi_AllocateBuffers(Nxp,Nyp,Nzp,Nh); 
  /* Initialize any buffer arrays required for IO reading/writing ...*/
  errorCode = ioAllocateBuffers(Nx,Ny,Nz); 

  /*** ---------------------------------------------------------------------------------------------- ***/
  /*** ----------------- Read initial condition or domain (grid) specification files -----------------***/
  /*** ---------------------------------------------------------------------------------------------- ***/
  /*If there is an initial conditions file read it*/
  if(inFile != NULL){
#ifdef DEBUG_INITIALIZATION 
    printf("mpi_rank_world--%d/%d inFile != NULL !!\n",mpi_rank_world,mpi_size_world);
#endif
    printf("Reading coordinates and input conditions from %s\n",inFile);
    fflush(stdout);
    errorCode = ioReadNetCDFinFileSingleTime(0, Nx, Ny, Nz, Nh);
  }else{
#ifdef DEBUG_INITIALIZATION 
    printf("mpi_rank_world--%d/%d inFile == NULL !!\n",mpi_rank_world,mpi_size_world);
    fflush(stdout);
#endif
    if(gridFile != NULL){
      printf("Reading coordinates-only from %s\n",gridFile);
      fflush(stdout);
    }else{
     printf("No input file has been specified, no gridFile has been specified.\n Using default grid and initial condition schemes.\n");
     fflush(stdout);
    }//endif gridFile != NULL
  }//end if inFile !=NULL
 
  /*** ---------------------------------------------------------------------------------------------- ***/
  /*** ----------------- Read initial condition or domain (grid) specification files -----------------***/
  /*** ---------------------------------------------------------------------------------------------- ***/
#ifdef DEBUG_INITIALIZATION 
  MPI_Barrier(MPI_COMM_WORLD); 
  printf("mpi_rank_world--%d/%d Beginning secondary and CUDA preparations!\n",mpi_rank_world,mpi_size_world);
  fflush(stdout);
#endif
  /*Allow for Secondary Preparations*/
  errorCode = gridSecondaryPreparations();
#ifdef DEBUG_INITIALIZATION 
  printf("mpi_rank_world--%d/%d Setting hydro_core Base State!\n",mpi_rank_world,mpi_size_world);
  fflush(stdout);
#endif
  /*Now that the grid is definitely defined, setup the base state  */
  errorCode = hydro_coreSetBaseState();

  /* inFile exists, allow HYDRO_CORE to preparations specifically from initial conditions */
  if(inFile != NULL){
    errorCode = hydro_corePrepareFromInitialConditions();
  }//end if inFile !=NULL

  /*** ---------------------------------------------------------------------------------------------- ***/
  /*** ----------------- Initialize the CUDA-layer of each model-component module --------------------***/
  /*** ---------------------------------------------------------------------------------------------- ***/
#ifndef NOTCUDA /* THIS SECTION PREPARES FOR CUDA FASTEDDY SIMULATION */

#ifdef DEBUG_INITIALIZATION 
  printf("mpi_rank_world--%d/%d calling cuda_gridInit!\n",mpi_rank_world,mpi_size_world);
  fflush(stdout);
#endif
  /*Perform CUDA preparations*/
  /*GRID/CUDA preparations*/
  errorCode = cuda_gridInit();

#ifdef DEBUG_INITIALIZATION 
  printf("mpi_rank_world--%d/%d calling fecuda_SetBlocksPerGrid!\n",mpi_rank_world,mpi_size_world);
  fflush(stdout);
#endif
  /*CUDA_AALES secondary (post-gridInit) preparations*/
  errorCode = fecuda_SetBlocksPerGrid((Nxp+2*Nh),(Nyp+2*Nh),(Nzp+2*Nh));
  if(numProcsY>1){
    errorCode = fecuda_AllocateHaloBuffers(Nxp,Nyp,Nzp,Nh);
  }//endif halo exchange buffers need to be allocated

#ifdef DEBUG_INITIALIZATION 
  printf("mpi_rank_world--%d/%d calling cuda_hydroCoreInit!\n",mpi_rank_world,mpi_size_world);
  fflush(stdout);
#endif
  /*HYDRO_CORE/CUDA preparations*/
  errorCode = cuda_hydroCoreInit();

#ifdef DEBUG_INITIALIZATION 
  printf("mpi_rank_world--%d/%d calling cuda_timeIntInit!\n",mpi_rank_world,mpi_size_world);
  fflush(stdout);
#endif
  /*TIME_INTEGRATION/CUDA preparations*/
  errorCode = cuda_timeIntInit();

#endif /* ifndef NOTCUDA: THIS SECTION PREPARED FOR CUDA FASTEDDY SIMULATION */
   
  MPI_Barrier(MPI_COMM_WORLD); 

  /*** ---------------------------------------------------------------------------------------------- ***/
  /*** ------- Final pre-check logging and initialization before entering the main time-loop ---------***/
  /*** ---------------------------------------------------------------------------------------------- ***/
#ifdef DEBUG_INITIALIZATION 
  int iRank;
  int i,j,k,ijk;
  /*Reset i,j,k, to the first non-halo cell)*/
  i = Nh; //Nxp/2+Nh-1;
  j = Nh; //Nyp/2+Nh-1;
  k = Nh; //Nzp/2+Nh-1;

  ijk = i*(Nyp+2*Nh)*(Nzp+2*Nh)+j*(Nzp+2*Nh)+k;
  for(iRank=0; iRank<mpi_size_world; iRank++){
    if(mpi_rank_world==iRank){
       printf("Rank %d/%d:  xMin, yMin, zMin = %f, %f, %f\n", 
            mpi_rank_world, mpi_size_world, xPos[ijk], yPos[ijk], zPos[ijk]);
       fflush(stdout);
       i = Nxp+Nh-1; //Nxp/2+Nh-1;
       j = Nyp+Nh-1; //Nyp/2+Nh-1;
       k = Nzp+Nh-1; //Nzp/2+Nh-1;
       ijk = i*(Nyp+2*Nh)*(Nzp+2*Nh)+j*(Nzp+2*Nh)+k;
 
       printf("Rank %d/%d:  xMax, yMax, zMax = %f, %f, %f\n", 
              mpi_rank_world, mpi_size_world, xPos[ijk], yPos[ijk], zPos[ijk]);
       fflush(stdout);
    }//end if mpi_rank_world==iRank 
    MPI_Barrier(MPI_COMM_WORLD); 
  } //end for iRank
#endif  //ifdef DEBUG_INITIALIZATION

#ifdef DEBUG_INITIALIZATION 
  printf("Rank %d/%d: Parameters-- numProcsX = %d, numProcsY = %d, Nh = %d\n\t\t gridFile = %s\n\t\t Nx/Nxp = %d/%d, Ny/Nyp = %d/%d, Nz/Nzp = %d/%d.\n", 
         mpi_rank_world, mpi_size_world, numProcsX, numProcsY, Nh, gridFile, Nx, Nxp, Ny, Nyp, Nz, Nzp);
  fflush(stdout);
#endif

  MPI_Barrier(MPI_COMM_WORLD);
  mpi_t2 = MPI_Wtime();    //Mark the walltime to measure duration of initializations.
  if(mpi_rank_world == 0){
     printf("Initializations complete after %8.4f (s).\n", (mpi_t2-mpi_t1));
     fflush(stdout);
  } //if mpi_rank_world

  /*** ---------------------------------------------------------------------------------------------- ***/
  /*** ------------------------------------ Main Timestep Loop: Begin --------------------------------***/
  /*** ---------------------------------------------------------------------------------------------- ***/
  /*Main Timestep Loop*/
  for(it=simTime_it; it < Nt; it=it+NtBatch){
     mpi_t1 = MPI_Wtime();    //Mark the walltime to measure duration of timestep.
     if(mpi_rank_world == 0){
       printf("Main Time: Advancing simulation timestep = %d... and time = %10.5f\n",it, it*dt);
       fflush(stdout);
     }//endif mpi_rank_world==0

     itTmp = it; 
     if(it%frqOutput == 0){
       MPI_Barrier(MPI_COMM_WORLD); 
       if(mpi_rank_world == 0){
         printf("\n_____________________#######_________  STATE-SUMMARY @ it = %d _________#######____________________ \n", it);
         printf("\n\t\t\t!!!!!\t  HYDRO_CORE \t !!!!! \n");
         fflush(stdout);
       } //if mpi_rank_world
  
       MPI_Barrier(MPI_COMM_WORLD); 

       /*Every rank calls the StateLogDump*/
       hydro_coreStateLogDump();
       MPI_Barrier(MPI_COMM_WORLD);
       fflush(stdout);
       MPI_Barrier(MPI_COMM_WORLD);
       if(mpi_rank_world == 0){
         printf("Dumping state at timestep = %d...\n",it);
         fflush(stdout);
       } //if mpi_rank_world

       mpi_t3 = MPI_Wtime();    //Mark the walltime to measure IO duration.
       /* Dump the root output file. */
#ifndef IO_OFF
       if(ioOutputMode==0){
         errorCode = ioWriteNetCDFoutFileSingleTime(it, Nx, Ny, Nz, Nh);
       }else if(ioOutputMode==1){
         errorCode = ioWriteBinaryoutFileSingleTime(it, Nxp, Nyp, Nzp, Nh);
       }
#endif
       mpi_t4 = MPI_Wtime();    //Mark the walltime to measure IO duration
       if(mpi_rank_world == 0){
         printf("Dumped state at timestep = %d...\n",it);
       } //if mpi_rank_world
     } //end if (it%frqOutput == 0) ....   (We log summary info and dump outputs)
#ifdef NOTCUDA 
     /* OBSELETE!!!!! There is longer any CPU model integration functionality */
#else  /* ---------------  CUDA FASTEDDY !!!!! -------------------------  */
     /*Launch the GPU batch timestep kernel*/
     errorCode = cuda_timeIntCommence(itTmp);
           /*Build an Frhs*/
           /*Update the prognostic variables*/
           /*Do any necessary halo exchange*/
     /*Kernel return*/
     itTmp = itTmp+NtBatch;  
#endif
     mpi_t2 = MPI_Wtime();    //Mark the walltime to measure duration of  a batch of timesteps.
     if(mpi_rank_world == 0){
        printf("\n\t\t\t!!!!!\t  TIMESTEP PERFORMANCE  \t !!!!! \n");
        printf(" Total Time (s) | Batch Steps \t| Time/step (s) | Comp./step (s) | IO Time (s)\n");
        printf("---------------------------------------------------------------------------------------------------------\n");
        printf("   %8.4f \t| %8d \t|  %8.4f \t|  %8.4f \t |  %9.6f \n", (mpi_t2-mpi_t1), NtBatch, 
                (mpi_t2-mpi_t1)/NtBatch, (mpi_t2-mpi_t1-(mpi_t4-mpi_t3))/NtBatch, (mpi_t4-mpi_t3));
        printf("\n********************************************************************************************************\n");
        fflush(stdout);
     } //if mpi_rank_world
  } //end Main Timestep Loop
  /*** ---------------------------------------------------------------------------------------------- ***/
  /*** -------------------------------------- Main Timestep Loop: End --------------------------------***/
  /*** ---------------------------------------------------------------------------------------------- ***/
  
  /*** ---------------------------------------------------------------------------------------------- ***/
  /*** ------------------------------------ Final Simulation Logging & IO ----------------------------***/
  /*** ---------------------------------------------------------------------------------------------- ***/
  mpi_t1 = MPI_Wtime();    //Mark the walltime to measure final timestep summary and performance.
  /*Summarize the final state */
  if(mpi_rank_world == 0){
    printf("\n_____________________#######_________  STATE-SUMMARY @ it = %d _________#######____________________ \n", it);
    printf("\n\t\t\t!!!!!\t  HYDRO_CORE \t !!!!! \n");
    fflush(stdout);
  } //if mpi_rank_world
  
  MPI_Barrier(MPI_COMM_WORLD); 
  /*Every Rank calls the Dump*/ 
  hydro_coreStateLogDump();
  MPI_Barrier(MPI_COMM_WORLD); 

  if(mpi_rank_world == 0){
    printf("Dumping state at timestep = %d...\n",it);
    fflush(stdout);
  } //if mpi_rank_world
  mpi_t3 = MPI_Wtime();    //Mark the walltime to measure IO duration.
  /*Dump the final timestep*/
#ifndef IO_OFF
  if(ioOutputMode==0){
    errorCode = ioWriteNetCDFoutFileSingleTime(it, Nx, Ny, Nz, Nh);
  }else if(ioOutputMode==1){
    errorCode = ioWriteBinaryoutFileSingleTime(it, Nxp, Nyp, Nzp, Nh);
  }
#endif
  mpi_t4 = MPI_Wtime();    //Mark the walltime to measure IO duration
  mpi_t2 = MPI_Wtime();    //Mark the walltime to measure final timestep summary and performance.
  if(mpi_rank_world == 0){
    printf("Dumped state at timestep = %d...\n",it);
    printf("\n\t\t\t!!!!!\t  TIMESTEP PERFORMANCE  \t !!!!! \n");
    printf(" Total Time (s) | Batch Steps \t| Time/step (s) | Comp./step (s) | IO Time (s)\n");
    printf("---------------------------------------------------------------------------------------------------------\n");
    printf("   %8.4f \t| %8d \t|  %8.4f \t|  %8.4f \t |  %9.6f \n", (mpi_t2-mpi_t1), 0, 
            (mpi_t2-mpi_t1)/NtBatch, (mpi_t2-mpi_t1-(mpi_t4-mpi_t3))/NtBatch, (mpi_t4-mpi_t3));
    printf("\n********************************************************************************************************\n");
    fflush(stdout);
    printf("Your FastEddy simulation is complete!\n");
    printf("Cleaning up...\n");
    fflush(stdout);
  } //if mpi_rank_world == 0

  /*** ---------------------------------------------------------------------------------------------- ***/
  /*** ------------------------------------ Final cleanup by module ----------------------------------***/
  /*** ---------------------------------------------------------------------------------------------- ***/
  /*Cleanup local memory*/ 
  /*Cleanup the TIME_INTEGRATION module*/
  errorCode = cuda_timeIntCleanup();
  /*Cleanup the TIME_INTEGRATION module*/
  errorCode = timeCleanup();
  /*Cleanup the HYDRO_CORE/CUDA module*/
  errorCode = cuda_hydroCoreCleanup();
  /*Cleanup the HYDRO_CORE module*/
  errorCode = hydro_coreCleanup();
  /*Cleanup FECUDA module*/
  errorCode = fecuda_DeallocateHaloBuffers();
  /*Cleanup the GRID/CUDA module*/
  errorCode = cuda_gridCleanup();
  /*Cleanup the GRID module*/
  errorCode = gridCleanup();
  /*Cleanup the IO module*/
  errorCode = ioCleanup();
  /*Cleanup the MEM_UTILS module*/
  errorCode = mem_utilsCleanup();
  /*Cleanup the FEMPI module*/
  errorCode = fempi_Cleanup();
  
  /* Cleanup the PARAMETERS module */
  parameters_clean();

  /*** ---------------------------------------------------------------------------------------------- ***/
  /*** ------------------------------------ Shut down the MPI environment and END --------------------***/
  /*** ---------------------------------------------------------------------------------------------- ***/
  /* Finalize the FEMPI environment */
  if(mpi_rank_world == 0){
    printf("Shutting down MPI...\n Goodbye!\n");
  } //if mpi_rank_world == 0
  MPI_Barrier(MPI_COMM_WORLD);
  errorCode = fempi_FinalizeMPI(argc, argv);

  return(errorCode);
} //end main{}
