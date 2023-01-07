/* FastEddy®: SRC/IO/io_netcdf.h
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
#include <netcdf.h>

/* Handle errors by printing an error message and exiting with a
 * non-zero status. */
#define ERR(e) {printf("Error: %s\n", nc_strerror(e));}

/*######################------------------- IO module variable declarations ---------------------#################*/
//netCDF dimension arrays
extern int dims4d[];
extern int dims3d[];  
extern int dims2dTD[];  
extern int dims2d[];  


/*######################------------------- IO module function declarations ---------------------#################*/

//////////***********************  INPUT FUNCTIONS  *********************************////////
/*----->>>>> int ioReadNetCDFgridFile();  ---------------------------------------------------------------
* Used to read a NetCDF file of registered "GRID" variables.
*/
int ioReadNetCDFgridFile(char* gridFile, int Nx, int Ny, int Nz, int Nh);

/*----->>>>> int ioReadNetCDFinFileSingleTime();  ---------------------------------------------------------------
* Used to read a NetCDF file for a single timestep.
*/
int ioReadNetCDFinFileSingleTime(int tstep, int Nx, int Ny, int Nz, int Nh);

/*----->>>>> int ioOpenNetCDFoutFile();    ---------------------------------------------------------------------
* Used to open a NetCDF file for reading.
*/
int ioOpenNetCDFinFile(char *fileName, int *ncidptr);

/*----->>>>> int ioGetNetCDFinFileVars();    ---------------------------------------------------------------------
* Used to get(read) all variables in the regiter list in(to) the appropriately registered module memory. 
*/
int ioGetNetCDFinFileVars(int ncid, int Nx, int Ny, int Nz, int Nh);

//////////***********************  OUTPUT FUNCTIONS  *********************************////////
/*----->>>>> int ioWriteNetCDFoutFileSingleTime();  ---------------------------------------------------------------
* Used to write a NetCDF file for a single timestep.
*/
int ioWriteNetCDFoutFileSingleTime(int tstep, int Nx, int Ny, int Nz, int Nh);

/*----->>>>> int ioCreateNetCDFoutFile();    ---------------------------------------------------------------------
* Used to create NetCDF file for writing.
*/
int ioCreateNetCDFoutFile(char *outFileName, int *ncidptr);

/*----->>>>> int ioDefineNetCDFoutFileDims();    ---------------------------------------------------------------------
* Used to complete the sequence of dimension definitions involved in "define mode" for a NetCDF file to be written.
*/
int ioDefineNetCDFoutFileDims(int ncid, int Nx, int Ny, int Nz, int Nh);

/*----->>>>> int ioDefineNetCDFoutFileVars();    ---------------------------------------------------------------------
* Used to complete the sequence of variable definitions involved in "define mode" for a NetCDF file to be written.
*/
int ioDefineNetCDFoutFileVars(int ncid);

/*----->>>>> int ioEndNetCDFdefineMode();    ---------------------------------------------------------------------
* Used to close the sequence steps involved in "define mode" for a NetCDF file to be written.
*/
int ioEndNetCDFdefineMode(int ncid, int Nx, int Ny, int Nz, int Nh);

/*----->>>>> int ioiPutNetCDFoutFileVars();    ---------------------------------------------------------------------
* Used to put(write) all variables in the regiter list in(to) the NetCDF file.
*/
int ioPutNetCDFoutFileVars(int ncid, int Nx, int Ny, int Nz, int Nh);

/*----->>>>> int ioCloseNetCDFfile();    ---------------------------------------------------------------------
* Used to close a netCDF file
*/
int ioCloseNetCDFfile(int ncid);

