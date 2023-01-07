/* FastEddy®: SRC/MEM_UTILS/mem_utils.c 
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
#include <parameters.h>
#include <fempi.h>
#include <mem_utils.h>

#define ALIGN_SIZE 128
/*######################------------------- MEM_UTILS module function definitions ---------------------#################*/

/*----->>>>> int mem_utilsGetParams();       ----------------------------------------------------------------------

Obtain the complete set of parameters for the MEM_UTILS module

*/
int mem_utilsGetParams(){
   int errorCode = MEM_UTILS_SUCCESS;

   /*query for each MEM_UTILS parameter */

   return(errorCode);
} //end mem_utilsGetParams()

/*----->>>>> int mem_utilsInit();       ----------------------------------------------------------------------
Used to broadcast and print parameters, allocate memory, and initialize configuration settings for  the MEM_UTILS module.
*/
int mem_utilsInit(){
   int errorCode = MEM_UTILS_SUCCESS;
  
   return(errorCode);
} //end mem_utilsInit()

/*----->>>>> float * memAllocateFloat2DField(iN, jN, halo_extent, fieldName); -------------------------------
 * Used to allocate memory for a 2-Dmensional field through the MEM_UTILS module. This field-memory space will
 * be aligned on ALIGN_SIZE bytes using posix_memalign.
 * */
float * memAllocateFloat2DField(int iN, int jN, int halo_extent, char *fieldName){ // DME
  float *field;
  void  *m_field;
  void  *memsetReturnVal;

  if(posix_memalign(&m_field, ALIGN_SIZE, (iN+2*halo_extent)*(jN+2*halo_extent)*sizeof(float))) {
     fprintf(stderr, "Rank %d/%d memAllocateFloat2DField(%s): Memory Allocation of m_field failed!\n",
             mpi_rank_world,mpi_size_world,fieldName);
     exit(1);
  } // if

  /*initialize the allocated space to zero everywhere*/
  memsetReturnVal = memset(m_field,0,(iN+2*halo_extent)*(jN+2*halo_extent)*sizeof(float));

  field = (float *) m_field;

  if(memsetReturnVal == NULL){
     fprintf(stderr, "Rank %d/%d memAllocateFloat2DField():WARNING memsetReturnVal == NULL!\n",
                mpi_rank_world,mpi_size_world);
  }
  return(field);
} // end memAllocateFloat2DField();

/*----->>>>> float * memAllocateFloat2DFieldN1D(nN, kN, halo_extent, fieldName); -------------------------------
 * Used to allocate memory for a 2-Dmensional field made of n vector of one spatial dimention through the MEM_UTILS module. This field-memory space will
 * be aligned on ALIGN_SIZE bytes using posix_memalign.
 * */
float * memAllocateFloat2DFieldN1D(int nN, int kN, int halo_extent, char *fieldName){ // DME
  float *field;
  void  *m_field;
  void  *memsetReturnVal;

  if(posix_memalign(&m_field, ALIGN_SIZE, nN*(kN+2*halo_extent)*sizeof(float))) {
     fprintf(stderr, "Rank %d/%d memAllocateFloat2DField(%s): Memory Allocation of m_field failed!\n",
             mpi_rank_world,mpi_size_world,fieldName);
     exit(1);
  } // if

  /*initialize the allocated space to zero everywhere*/
  memsetReturnVal = memset(m_field,0,nN*(kN+2*halo_extent)*sizeof(float));

  field = (float *) m_field;

  if(memsetReturnVal == NULL){
     fprintf(stderr, "Rank %d/%d memAllocateFloat2DField():WARNING memsetReturnVal == NULL!\n",
                mpi_rank_world,mpi_size_world);
  }

  return(field);
} // end memAllocateFloat2DField();

/*----->>>>> float * memAllocateFloat3DField(iN, jN, kN, halo_extent, fieldName); -------------------------------
Used to allocate memory for a 3-Dmensional field through the MEM_UTILS module. This field-memory space will
be aligned on ALIGN_SIZE bytes using posix_memalign.
*/
float * memAllocateFloat3DField(int iN, int jN, int kN, int halo_extent, char *fieldName){
  float *field;
  void  *m_field;
  void  *memsetReturnVal;

  if(posix_memalign(&m_field, ALIGN_SIZE, (iN+2*halo_extent)*(jN+2*halo_extent)*(kN+2*halo_extent)*sizeof(float))) {
     fprintf(stderr, "Rank %d/%d memAllocateFloat3DField(%s): Memory Allocation of m_field failed!\n",
             mpi_rank_world,mpi_size_world,fieldName);
     exit(1);
  } // if

  /*initialize the allocated space to zero everywhere*/
  memsetReturnVal = memset(m_field,0,(iN+2*halo_extent)*(jN+2*halo_extent)*(kN+2*halo_extent)*sizeof(float));

  field = (float *) m_field;

  if(memsetReturnVal == NULL){
     fprintf(stderr, "Rank %d/%d memAllocateFloat3DField():WARNING memsetReturnVal == NULL!\n",
                mpi_rank_world,mpi_size_world);
  }
  return(field);
} // end memAllocateFloat3DField();

/*----->>>>> float * memAllocateFloat4DField(Nfields, iN, jN, kN, halo_extent, fieldName); --------------------------
* Used to allocate memory for a 4-Dmensional block of memory through the MEM_UTILS module. This multifield-memory space
* will be aligned on ALIGN_SIZE bytes using posix_memalign.
* */
float * memAllocateFloat4DField(int Nfields, int iN, int jN, int kN, int halo_extent, char *fieldName){
  float *blockOfFields;
  void  *m_field;
  void  *memsetReturnVal;

  if(posix_memalign(&m_field, ALIGN_SIZE, 
                    (Nfields)*(iN+2*halo_extent)*(jN+2*halo_extent)*(kN+2*halo_extent)*sizeof(float))) {
     fprintf(stderr, "Rank %d/%d memAllocateFloat4DField(%s): Memory Allocation of m_field failed!\n",
             mpi_rank_world,mpi_size_world,fieldName);
     exit(1);
  } // if
  
  /*initialize the allocated space to zero everywhere*/
  memsetReturnVal = memset(m_field,0,(Nfields)*(iN+2*halo_extent)*(jN+2*halo_extent)*(kN+2*halo_extent)*sizeof(float));

  blockOfFields = (float *) m_field;

  if(memsetReturnVal == NULL){
     fprintf(stderr, "Rank %d/%d memAllocateFloat4DField():WARNING memsetReturnVal == NULL!\n",
                mpi_rank_world,mpi_size_world);
  }
  return(blockOfFields);
} // end memAllocateFloat4DField();

/*----->>>>> int memReleaseFloat(float * mem_block); --------------------------------------------------------
Used to release memory for a block of floats.
*/
int memReleaseFloat(float * mem_block){
   int errorCode = MEM_UTILS_SUCCESS;
 
   free(mem_block);
   
   return(errorCode); 
} //end memReleaseFloat();

/*----->>>>> int mem_utilsCleanup();       ----------------------------------------------------------------------
Used to free all malloced memory by the MEM_UTILS module.
*/
int mem_utilsCleanup(){
   int errorCode = MEM_UTILS_SUCCESS;

   /* Free any MEM_UTILS module arrays */

   return(errorCode);

}//end mem_utilsCleanup()
