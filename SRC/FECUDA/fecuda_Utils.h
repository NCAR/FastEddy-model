/* FastEddy®: SRC/FECUDA/fecuda_Utils.h 
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
#ifndef _FECUDA_UTILS_H
#define _FECUDA_UTILS_H

/*----->>>>> int fecuda_UtilsAllocateHaloBuffers(); ----------------------------------------------------------
* Used to allocate device memory buffers for coalesced halo exchanges in the FECUDA module.
*/
int fecuda_UtilsAllocateHaloBuffers(int Nxp, int Nyp, int Nzp, int Nh);

/*----->>>>> int fecuda_UtilsDeallocateHaloBuffers(); ---------------------------------------------------------
* Used to free device memory buffers for coalesced halo exchanges in the FECUDA module.
*/
int fecuda_UtilsDeallocateHaloBuffers();
#endif //_FECUDA_UTILS_H
