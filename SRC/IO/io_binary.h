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
/*----->>>>> int ioWriteiBinaryoutFileSingleTime();  ---------------------------------------------------------------
 * Used to have N-ranks write N-binary files of registered variables for a single timestep.
 */
int ioWriteBinaryoutFileSingleTime(int tstep, int Nx, int Ny, int Nz, int Nh);
/*----->>>>> int ioPutBinaryoutFileVars();    ---------------------------------------------------------------------
 * Used to put(write) all variables in the register list in(to) the Binary file. 
*/
int ioPutBinaryoutFileVars(FILE *outptr, int Nx, int Ny, int Nz, int Nh);

