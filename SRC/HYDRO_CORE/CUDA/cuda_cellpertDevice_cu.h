#ifndef _CELLPERT_CUDADEV_CU_H
#define _CELLPERT_CUDADEV_CU_H

/*cellpert_ return codes */
#define CUDA_CELLPERT_SUCCESS    0

/*##############------------------- CELLPERT submodule variable declarations ---------------------#################*/
/* array fields */
extern __constant__ int cellpertSelector_d;   /*CP method selector: 0= off, 1= on */
extern __constant__ int cellpert_sw2b_d;      /* switch to do: 0= all four lateral boundaries, 1= only south & west boundaries, 2= only south boundary */
extern __constant__ float cellpert_amp_d;     /* maximum amplitude for the potential temperature perturbations */
extern __constant__ int cellpert_nts_d;       /* number of time steps after which perturbations are seeded */
extern __constant__ int cellpert_gppc_d;      /* number of grid points conforming the cell */
extern __constant__ int cellpert_ndbc_d;      /* number of cells normal to domain lateral boundaries */
extern __constant__ int cellpert_zbottom_d;   /* z-grid point where the perturbations start */
extern __constant__ int cellpert_ztop_d;      /* z-grid point where the perturbations end */
extern __constant__ float cellpert_eckert_d;  /* Eckert number for the potential temperature perturbations (hydroBCs == 5) */
extern __constant__ float cellpert_tsfact_d;  /* factor on the refreshing perturbation time scale (hydroBCs == 5) */
extern float* randcp_d;                       /*Base address for pseudo-random numbers used for cell perturbations (1d-array) */

/*##############-------------- CELLPERT_CUDADEV submodule function declarations ------------------############*/

/*----->>>>> int cuda_cellpertDeviceSetup();      -----------------------------------------------------------------
* Used to cudaMalloc and cudaMemcpy parameters and coordinate arrays for the CELLPERT_CUDADEV submodule.
*/
extern "C" int cuda_cellpertDeviceSetup();

/*----->>>>> int cuda_cellpertDeviceCleanup();    ---------------------------------------------------------------
* Used to free all malloced memory by the CELLPERT_CUDADEV submodule.
*/
extern "C" int cuda_cellpertDeviceCleanup();

/*----->>>>> extern "C" int cuda_hydroCoreDeviceBuildCPmethod();  -------------------------------------------------- 
* This routine provides the externally callable cuda-kernel call to perform a call to cell perturbation method */
extern "C" int cuda_hydroCoreDeviceBuildCPmethod(int simTime_it);

/*----->>>>> extern "C" int cuda_hydroCoreTVCP();  -----------------------------------------------------------
* Updates device-sided parameters used by the CELLPERT submodule from dynamic lateral BNDY conditions 
*/
extern "C" int cuda_hydroCoreTVCP();

__global__ void cudaDevice_hydroCoreCompleteCellPerturbation(float* hydroFlds, float* randcp_d, int my_mpi, int numpx, int numpy);

/*----->>>>> __device__ void  cudaDevice_CellPerturbation();  --------------------------------------------------
 *  */ // This cuda kerne lsets up the cells and their id in the CP method
__device__ void cudaDevice_CellPerturbation(int i_ind, int j_ind, int k_ind, int Nx, int Ny, int Nz, int Nh, int my_mpi, int numpx, int numpy, float* rho, float* theta, float *rand_1darray);

#endif // _CELLPERT_CUDADEV_CU_H
