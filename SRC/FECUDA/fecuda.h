/* FastEddy®: SRC/FECUDA/fecuda.h 
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
#ifndef _FECUDA_H
#define _FECUDA_H

/*fecuda return codes */
#define FECUDA_SUCCESS               0

/*fecuda includes*/

/*#################------------------- FECUDA module variable declarations ---------------------#################*/
/* Parameters */
extern int tBx;
extern int tBy;
extern int tBz;

/*#################------------------- FECUDA module function declarations ---------------------#################*/

/*----->>>>> int fecuda_GetParams();       ----------------------------------------------------------------------
* Used to populate the set of parameters for the FECUDA module
*/
int fecuda_GetParams();

/*#########------------------- FECUDA module CUDA-C wrapper function declarations ------------#################*/
/* Each of these functions has a corrseponding "Device" function called by the wrapper. 
 * This allows us to mix MPI and CUDA. */

/*----->>>>> int fecuda_Init();       ----------------------------------------------------------------------
* Used to broadcast and print parameters, allocate memory, and initialize configuration settings for FECUDA.
*/
int fecuda_Init();

/*----->>>>> int fecuda_Cleanup();       ----------------------------------------------------------------------
* Used to free all malloced memory by the FECUDA module.
*/
int fecuda_Cleanup();

/*----->>>>> int fecuda_SetBlocksPerGrid();    -----------------------------------------------------------
* Used to set the grid and tBlock (threadBlock) configuration parameters for the device..
*/
int fecuda_SetBlocksPerGrid(int Nx, int Ny, int Nz);

/*----->>>>> int fecuda_AllocateHaloBuffers(); -------------------------------------------------------------
* Used to allocate device memory buffers for coalesced halo exchanges in the FECUDA module.
*/
int fecuda_AllocateHaloBuffers(int Nxp, int Nyp, int Nzp, int Nh);

/*----->>>>> int fecuda_DeallocateHaloBuffers(); ---------------------------------------------------------
* Used to free device memory buffers for coalesced halo exchanges in the FECUDA module.
*/
int fecuda_DeallocateHaloBuffers();

#endif // _FECUDA_H
