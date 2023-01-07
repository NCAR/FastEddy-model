/* FastEddy®: SRC/GRID/grid.h 
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
#ifndef _GRID_H
#define _GRID_H

/*grid_ return codes */
#define GRID_SUCCESS               0
#define TOPOFILE_GRID_FAIL         1
#define GRID_DECOMPOSE_FAIL        2
#define GRID_IO_CALL_FAIL          3


/*######################------------------- GRID module variable declarations ---------------------#################*/
/* Parameters */
extern char *gridFile;  //A file containing a complete grid specification
extern char *topoFile;  //A file containing a complete grid specification
extern int Nh;          //Number of halo cells to be used (dependent on largest stencil extent)
extern int Nx, Ny, Nz;  //Complete Cartesian Domain extents in the x, y, and z directions 
extern float d_xi, d_eta, d_zeta; //Computational Domain fixed resolutions (i, j, k respectively)
extern int coordHorizHalos; //switch to setup coordiante halos as periodic, or gradient following
extern int verticalDeformSwitch; //switch to use vertical coordinate deformation
extern float verticalDeformFactor; // factor used under vertical deformation (0.0-1.0)
extern float verticalDeformQuadCoeff; // quadratic term coefficient in the deformtion scheme (default = 0.0)
extern int iMin, iMax; //Constant min and max bounds of i-index accounting for only non-halos cells of the mpi_rank subdomain
extern int jMin, jMax; //Constant min and max bounds of j-index accounting for only non-halos cells of the mpi_rank subdomain
extern int kMin, kMax; //Constant min and max bounds of k-index accounting for only non-halos cells of the mpi_rank subdomain

/*static scalars*/
extern float dX, dY, dZ; //reference computational model coordinate resolution
extern float dXi, dYi, dZi; //inverse of the reference computational model coordinate resolution
 
/* array fields */
extern float *xPos;  /* Cell-center position in x (meters) */
extern float *yPos;  /* Cell-center position in y (meters) */
extern float *zPos;  /* Cell-center position in z (meters) */
extern float *topoPos; /*Topography elevation (z in meters) at the cell center position in x and y. */
extern float *topoPosGlobal; /*Topography elevation (z in meters) at the cell center position in x and y. (Global domain) */

extern float *J31;      // dz/d_xi
extern float *J32;      // dz/d_eta
extern float *J33;      // dz/d_zeta

extern float *D_Jac;    //Determinant of the Jacbian  (called scale factor i.e. if d_xi=d_eta=d_zeta=1, then cell volume)
extern float *invD_Jac; //inverse Determinant of the Jacbian 

/*######################------------------- GRID module function declarations ---------------------#################*/

/*----->>>>> int gridGetParams();       ----------------------------------------------------------------------
*Used to populate the set of parameters for the GRID module
*/

int gridGetParams();

/*----->>>>> int gridInit();       ----------------------------------------------------------------------
*Used to broadcast and print parameters, allocate memory, and initialize configuration settings for  the GRID module.
*/
int gridInit();

/*----->>>>> int gridSecondaryPreperations();   -------------------------------------------------------------------
* Used to read a gridFile and/or calculate the metric tensor fields.
*/
int gridSecondaryPreparations();

/*----->>>>> int calculateJacobians();       ----------------------------------------------------------------------
* Used to calculate the metric tensor elements for a generalized curvilinear coordinate system
*/
int calculateJacobians();

/*----->>>>> int singleRankGridHaloInit();    ------------------------------------------------------------
* Used to setup xPos,yPos,zPos halos on all x-y boundaries 
* when under single-rank setup (i.e. mpi_size_world ==1).
*/
int singleRankGridHaloInit();

/*----->>>>> float zDeform();       ----------------------------------------------------------------------
* Used to calculate non-uniform resolution vertical coordinates.
*/
float zDeform(float zRect, float zGround, float zCeiling);

/*----->>>>> int gridCleanup();       ----------------------------------------------------------------------
* Used to free all malloced memory by the GRID module.
*/
int gridCleanup();


#endif // _GRID_H
