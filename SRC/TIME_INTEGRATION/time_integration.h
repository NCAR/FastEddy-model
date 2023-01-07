/* FastEddy®: SRC/TIME_INTEGRATION/time_integration.h 
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
#ifndef _TIME_INTEGRATION_H
#define _TIME_INTEGRATION_H

/*time_ return codes */
#define TIME_INTEGRATION_SUCCESS               0

/*######################------------------- TIME_INTEGRATION module variable declarations ---------------------#################*/
/* Parameters */
extern int timeMethod;  // Selector for time integration method. [0=RK1, 1=RK3-WS2002 (default)]
extern int Nt;          // Number of timesteps to perform
extern float dt;        // timestep resolution in seconds
extern int NtBatch;     // Number of timesteps in a batch to perform in a CUDA kernel launch

/*static scalars*/
extern float simTime; /*Master simulation time*/
extern int simTime_it; /*Master simulation time step*/ 
extern int simTime_itRestart; /*Master simulation 'Restart' time step*/ 
extern int numRKstages; /* number of stages in the time scheme */

/* array fields */

/*############------------------- TIME_INTEGRATION module function declarations ---------------------############*/

/*----->>>>> int timeGetParams();       ----------------------------------------------------------------------
Used to populate the set of parameters for the TIME_INTEGRATION module
*/
int timeGetParams();

/*----->>>>> int timeInit();       ----------------------------------------------------------------------
* Used to broadcast and print parameters, allocate memory, and initialize configuration settings for  
* the TIME_INTEGRATION module.
*/
int timeInit();

/*----->>>>> int timeInitBdyPlaneUpdates();     -------------------------------------------------------------
*/
int timeIntBdyPlaneUpdates();

/*----->>>>> int timeCleanup();       ----------------------------------------------------------------------
Used to free all malloced memory by the TIME_INTEGRATION module.
*/
int timeCleanup();

#endif // _TIME_INTEGRATION_H
