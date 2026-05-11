/* FastEddy®: SRC/IO/io_binary.h
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
//////////***********************  INPUT FUNCTIONS  *********************************////////

//////////***********************  OUTPUT FUNCTIONS  *********************************////////
/*----->>>>> int ioWriteBinaryoutFileSingleTime();  ---------------------------------------------------------------
 * Used to have N-ranks write N-binary files of registered variables for a single timestep.
 */
#ifdef GAD_EXT
int ioWriteBinaryoutFileSingleTime(int tstep, int Nx, int Ny, int Nz, int Nh, int Nturbines);
#else
int ioWriteBinaryoutFileSingleTime(int tstep, int Nx, int Ny, int Nz, int Nh);
#endif
/*----->>>>> int ioPutBinaryoutFileVars();    ---------------------------------------------------------------------
 * Used to put(write) all variables in the register list in(to) the Binary file. 
*/
#ifdef GAD_EXT
int ioPutBinaryoutFileVars(FILE *outptr, int Nx, int Ny, int Nz, int Nh, int Nturbines);
#else
int ioPutBinaryoutFileVars(FILE *outptr, int Nx, int Ny, int Nz, int Nh);
#endif
/*----->>>>> int ioWriteBinaryTowerFileSingleBatch();  ---------------------------------------------------------------
 * Used to have N-ranks write N-binary files of virtual tower data structures for a batch of timesteps.
 */
int ioWriteBinaryTowerFileSingleBatch(int tstep, int batchSize, int Nz, float *batchTimes, float *towersData, float *towersSurfData, 
		                      int *towerIDs, int rank_nTowers, int towerInstanceSize, int towerSurfInstanceSize);

/*----->>>>> int ioWriteBinaryTowerInitialFile();  ---------------------------------------------------------------
 * Used to have N-ranks write N-binary files of virtual tower data structures for the initial timestep and time-independent fields.
 */
int ioWriteBinaryTowerInitialFile(float dt, int itStart, int Nx, int Ny, int Nz, int Nh,
                                  float *towersData, float *towersSurfData,
                                  int *towerIDs, int rank_nTowers, int *tower_iInds, int *tower_jInds, 
                                  int coordType, float *tower_xOffs, float *tower_yOffs, double *tower_LonOffs, double *tower_LatOffs,
    				  int batchSize, int towerInstanceSize, int towerSurfInstanceSize,
                                  float *zCoords, float *yCoords, float *xCoords, float *topoFld, int surflayer_offshore, float *seamask);
