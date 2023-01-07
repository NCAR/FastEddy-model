/* FastEddy®: SRC/MEM_UTILS/mem_utils.h 
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
#ifndef _MEM_UTILS_H
#define _MEM_UTILS_H

/*mem_utils_ return codes */
#define MEM_UTILS_SUCCESS               0


/*######################------------------- MEM_UTILS module variable declarations ---------------------#################*/
/* Parameters */

/*######################------------------- MEM_UTILS module function declarations ---------------------#################*/

/*----->>>>> int mem_utilsGetParams();       ----------------------------------------------------------------------
Used to populate the set of parameters for the MEM_UTILS module
*/

int mem_utilsGetParams();

/*----->>>>> int mem_utilsInit();       ----------------------------------------------------------------------
Used to broadcast and print parameters, allocate memory, and initialize configuration settings for  the MEM_UTILS module.
*/
int mem_utilsInit();

/*----->>>>> float * memAllocateFloat2DField(iN, jN, halo_extent, fieldName); -------------------------------
 * Used to allocate memory for a 2-Dmensional field through the MEM_UTILS module. This field-memory space will
 * be aligned on ALIGN_SIZE bytes using posix_memalign.
 * */
float * memAllocateFloat2DField(int iN, int jN, int halo_extent, char *fieldName); // DME

/*----->>>>> float * memAllocateFloat2DFieldN1D(nN, kN, halo_extent, fieldName); -------------------------------
 * Used to allocate memory for a 2-Dmensional field made of n vector of one spatial dimention through the MEM_UTILS module. This field-memory space will
 * be aligned on ALIGN_SIZE bytes using posix_memalign.
 * */
float * memAllocateFloat2DFieldN1D(int nN, int kN, int halo_extent, char *fieldName);

/*----->>>>> float * memAllocateFloat3DField(iN, jN, kN, halo_extent, fieldName); -------------------------------
 * Used to allocate memory for a 3-Dmensional field through the MEM_UTILS module. This field-memory space will
 * be aligned on ALIGN_SIZE bytes using posix_memalign.
 * */
float * memAllocateFloat3DField(int iN, int jN, int kN, int halo_extent, char *fieldName);

/*----->>>>> float * memAllocateFloat4DField(Nfields, iN, jN, kN, halo_extent, fieldName); --------------------------
 * Used to allocate memory for a 4-Dmensional block of memory through the MEM_UTILS module. This multifield-memory space
 * will be aligned on ALIGN_SIZE bytes using posix_memalign.
 * */
float * memAllocateFloat4DField(int Nfields, int iN, int jN, int kN, int halo_extent, char *fieldName);

/*----->>>>> int memReleaseFloat(float * mem_block); --------------------------------------------------------
 * Used to release memory for a block of floats.
 * */
int memReleaseFloat(float * mem_block);

/*----->>>>> int mem_utilsCleanup();       ----------------------------------------------------------------------
Used to free all malloced memory by the MEM_UTILS module.
*/
int mem_utilsCleanup();


#endif // _MEM_UTILS_H
