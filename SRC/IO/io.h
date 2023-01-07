/* FastEddy®: SRC/IO/io.h 
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
#ifndef _IO_H
#define _IO_H

/*io_ return codes */
#define IO_SUCCESS  0
#define MAX_LEN     256   //used for static allocation of dimids length. Could be made dynamic.
#define MAXDIMS     16   //used for static allocation of dimids length. Could be made dynamic.

#define IO_ERROR_DIMLEN          200

/*io includes*/
#include <io_netcdf.h>
#include <io_binary.h>

/*######################------------------- IO module variable declarations ---------------------#################*/
/* Parameters */
extern int ioOutputMode;  /*0: N-to-1 gather and write to a netcdf file, 1:N-to-N writes of FastEddy binary files*/
extern char *outPath;     /* Directory Path where output files are to be written */
extern char *outFileBase; /* Base name of the output file series as in (outFileBase).element-in-series */ 
extern char *inPath;      /* Directory Path where input files are to be read from */
extern char *inFile;      /* Name of the input file */ 
extern int frqOutput;     /*frequency in timesteps to produce output*/

/*static Variables*/
extern char *outSubString; /*subString portion of outFile holding element-in-series as in path/base.substring */
extern char *outFileName;      /*full name instance of outFileName =  path/base.substring */
extern char *inFileName;      /*full name instance of inFileName =  path/infile */
extern int nz_varid;
extern int ny_varid;
extern int nx_varid;

/*IO-Buffers*/
extern float *ioBuffField;
extern float *ioBuffFieldTransposed;
extern float *ioBuffFieldRho;
extern float *ioBuffFieldTransposed2D;

/*######################------------------- IO module function declarations ---------------------#################*/

/*----->>>>> int ioGetParams();       ----------------------------------------------------------------------
Used to populate the set of parameters for the IO module
*/
int ioGetParams();

/*----->>>>> int ioInit();     ----------------------------------------------------------------------
Used to broadcast and print parameters, allocate memory, and initialize configuration settings for  the IO module.
*/
int ioInit();

/*----->>>>> int ioAllocateBuffers();   -----------------------------------
* Allocate memory to io-buffers for reading/writing IO-registered fields
*/
int ioAllocateBuffers(int globalNx, int globalNy, int globalNz);

/*----->>>>> int ioCleanup();       ----------------------------------------------------------------------
Used to free all malloced memory by the IO module.
*/
int ioCleanup();

/*----->>>>> int ioRegisterVar();    ---------------------------------------------------------------------
* Used by other modules to register a variable in the IO module list of variables to read/write as input/output.
*/
int ioRegisterVar(char *name, char *type, int nDims, int *dimids, void *varMemAddress);

#endif // _IO_H
