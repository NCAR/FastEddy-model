/* FastEddy®: SRC/FECUDA/fecuda_Utils_cu.h 
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
#ifndef _FECUDA_UTILS_CU_H
#define _FECUDA_UTILS_CU_H
/*##############--------- FECUDA Utility (fecuda_Utils.cu) variable declarations -------------#################*/
extern float *haloSendBuff_d;  //Send-Buffer for coalesced halo exchanges
extern float *haloRecvBuff_d;  //Recieve-Buffer for coalesced halo exchanges

/*##########---------- FECUDA Utility (fecuda_Utils.cu) function declarations -----------#################*/
void createAndStartEvent(cudaEvent_t *startE, cudaEvent_t *stopE);
void stopSynchReportDestroyEvent(cudaEvent_t *startE, cudaEvent_t *stopE, float *elapsedTime);

__global__ void fecuda_UtilsPackLoSideHaloBuff(float* sendFld_d,float* haloSendBuff_d,int Nxp,int Nyp,int Nzp,int Nh);
__global__ void fecuda_UtilsUnpackLoSideHaloBuff(float* recvFld_d,float* haloRecvBuff_d,int Nxp,int Nyp,int Nzp,int Nh);
__global__ void fecuda_UtilsPackHiSideHaloBuff(float* sendFld_d,float* haloSendBuff_d,int Nxp,int Nyp,int Nzp,int Nh);
__global__ void fecuda_UtilsUnpackHiSideHaloBuff(float* recvFld_d,float* haloRecvBuff_d,int Nxp,int Nyp,int Nzp,int Nh);

/*----->>>>> int fecuda_UtilsAllocateHaloBuffers(); ----------------------------------------------------------
* Used to allocate device memory buffers for coalesced halo exchanges in the FECUDA module.
*/
extern "C" int fecuda_UtilsAllocateHaloBuffers(int Nxp, int Nyp, int Nzp, int Nh);

/*----->>>>> int fecuda_UtilsDeallocateHaloBuffers(); ---------------------------------------------------------
* Used to free device memory buffers for coalesced halo exchanges in the FECUDA module.
*/
extern "C" int fecuda_UtilsDeallocateHaloBuffers();

/*----->>>>> void fecuda_DeviceMalloc();    -----------------------------------------------------------
* Used to allocate device memory float blocks and set the  host memory addresses of device memory pointers.
*/
extern "C" void fecuda_DeviceMalloc(int Nelems, float** memBlock_d);

/*----->>>>> int fecuda_SendRecvWestEast(); -------------------------------------------------------------------
Used to perform western/eastern device domain halo exchange for an arbitrary field.
*/
extern "C" int fecuda_SendRecvWestEast(float* sendFld_d, float* recvFld_d, int hydroBCs);

/*----->>>>> int fecuda_SendRecvEastWest(); -------------------------------------------------------------------
Used to perform eastern/western device domain halo exchange for an arbitrary field.
*/
extern "C" int fecuda_SendRecvEastWest(float* sendFld_d, float* recvFld_d, int hydroBCs);

/*----->>>>> int fecuda_SendRecvSouthNorth(); -------------------------------------------------------------------
Used to perform southern/northern device domain halo exchange for an arbitrary field.
*/
extern "C" int fecuda_SendRecvSouthNorth(float* sendFld_d, float* recvFld_d, int hydroBCs);

/*----->>>>> int fecuda_SendRecvNorthSouth(); -------------------------------------------------------------------
Used to perform northern/southern device domain halo exchange for an arbitrary field.
*/
extern "C" int fecuda_SendRecvNorthSouth(float* sendFld_d, float* recvFld_d, int hydroBCs);
#endif //_FECUDA_UTILS_CU_H
