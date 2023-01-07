/* FastEddy®: SRC/GRID/CUDA/cuda_grid.h 
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
#ifndef _CUDA_GRID_H
#define _CUDA_GRID_H

/*cuda_grid_ return codes */
#define CUDA_GRID_SUCCESS               0

/*######################------------------- CUDA_GRID module variable declarations ---------------------#################*/
/* Parameters */

/*##################------------------- CUDA_GRID module function declarations ---------------------#################*/

/*----->>>>> int cuda_gridInit();       ----------------------------------------------------------------------
Used to broadcast and print parameters, allocate memory, and initialize configuration settings for  the CUDA_GRID module.
*/
int cuda_gridInit();

/*----->>>>> int cuda_gridCleanup();       ----------------------------------------------------------------------
Used to free all malloced memory by the CUDA_GRID module.
*/
int cuda_gridCleanup();


#endif // _CUDA_GRID_H
