/* FastEddy®: SRC/FECUDA/fecuda_PlugIns.cu 
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
//Include cub for the atomic functions
#include <cub/cub.cuh> 

/* These constant compile time constants are needed for the cuda_singleRankHorizSlabMeans function*/
dim3 grid_red;
dim3 tBlock_red;

/*----->>>>> int cuda_singleRankHorizSlabMeans(); -------------------------------------------------------------
*/
__global__ void cuda_singleRankHorizSlabMeans(float* inputRho, float* inputFld, float* blockMean_d, float* slabMeans_d){
    // Block wise reduction so that one thread in each block holds sum of thread results
    typedef cub::BlockReduce<float, tBx_red, cub::BLOCK_REDUCE_RAKING, tBy_red, tBz_red> BlockReduce;
    int i,j,k,ijk;
    i = (blockIdx.x)*blockDim.x + threadIdx.x;
    j = (blockIdx.y)*blockDim.y + threadIdx.y;
    k = (blockIdx.z)*blockDim.z + threadIdx.z;

    float meanFactor = 1.0/((float)(gridDim.x*blockDim.x*gridDim.y*blockDim.y));
    ijk = ijk = i*(gridDim.y*blockDim.y)*(gridDim.z*blockDim.z) + j*(gridDim.z*blockDim.z) + k*1;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    float aggregate = BlockReduce(temp_storage).Sum(inputFld[ijk]);
    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0){
        blockMean_d[blockIdx.x*gridDim.y*gridDim.z + blockIdx.y*gridDim.z+blockIdx.z] = aggregate;
    }
    if (threadIdx.x == 0 && threadIdx.y == 0 && (blockIdx.x < gridDim.x && blockIdx.y < gridDim.y)){
        atomicAdd(&slabMeans_d[k],meanFactor*blockMean_d[blockIdx.x*gridDim.y*gridDim.z + blockIdx.y*gridDim.z+blockIdx.z]);
    }
}

