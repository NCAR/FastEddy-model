/* FastEddy®: SRC/FECUDA/fecuda_Device.h 
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
/*This header file acts as a surrogate CPU-centric 'C' header-wrapper file to the _cu.h version of this 
 * header file enabling mixed C and CUDA compilation/linking*/
#ifndef _FECUDA_DEV_H
#define _FECUDA_DEV_H

/*fecuda return codes */

/*fecuda includes*/
#include <cuda.h>
#include <cuda_runtime.h>

/*##################------------------- FECUDA module variable declarations ---------------------#################*/
/* Parameters */

/*##################------------------- FECUDA module function declarations ---------------------#################*/

/*----->>>>> int fecuda_DeviceSetup();       ----------------------------------------------------------------------
* Used to set the "dim3 tblock" module variable that is passed to any device kernel to specify the number of 
* threads per block in each dimension 
*/
int fecuda_DeviceSetup(int tBx, int tBy, int tBz);

/*----->>>>> int fecuda_SetBlocksPerGrid();   ------------------------------------------------------------------
* Used to set the "dim3 grid" module variable that is passed to any device kernel 
* to specify the number of blocks per grid in each dimenaion
*/
int fecuda_DeviceSetBlocksPerGrid(int Nx, int Ny, int Nz);

/*----->>>>> int fecuda_RunTestKernel();    ------------------------------------------------------------------
 Used as placeholder routine for testing the launch of a cuda kernel
*/
int fecuda_DeviceRunTestKernel(int Nx, int Ny, int Nz);

/*----->>>>> int fecuda_DeviceCleanup();       ----------------------------------------------------------------------
Used to free all malloced memory by the FECUDA module.
*/
int fecuda_DeviceCleanup();

#endif // _FECUDA_DEV_H
