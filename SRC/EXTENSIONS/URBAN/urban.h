/* FastEddy®: SRC/EXTENSIONS/URBAN/urban.h
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
#ifndef _URBAN_H
#define _URBAN_H

/*URBAN return codes */
#define URBAN_SUCCESS    0
#define URBAN_FAIL       10

/*---URBAN parameters*/
extern int urbanSelector;      /* urban selector: 0=off, 1=on, 2=on with thermal relaxation towards base state */
extern float cd_build;         /* c_d coefficient used by the drag-based building formulation: -c_d|u_i|u_i */
extern float ct_build;         /* c_t coefficient (s-1) used by the drag-based building formulation: -c_t(rho*theta-rho_b*theta_b) & -c_t(rho-rho_b) */
extern float *building_mask;   /* Base Address of memory containing building mask 0,1 field */
extern float delta_aware_bdg;  /* scale-aware correction for building forcing and limiters */
extern int urban_heatRedis;        /* selector to activate surface heat redistribution */
extern float *urban_heat_redis;    /* Base Address of memory containing 2d map of heat redistribution coefficient in urban areas */

/*----->>>>> int URBANGetParams();   ----------------------------------------------------------------------
 * Obtain parameters for the URBAN sub-module
*/
int URBANGetParams();

/*----->>>>> int URBANPrintParams();   ----------------------------------------------------------------------
* Print parameters for the URBAN sub-module
*/
int URBANPrintParams();

/*----->>>>> int URBANInit();   ----------------------------------------------------------------------
 * Used to broadcast parameters, allocate memory, and initialize configuration settings
 * for the URBAN sub-module.
*/
int URBANInit();

/*----->>>>> int URBANCleanup();  ----------------------------------------------------------------------
* Used to free all malloced memory by the URBAN module.
*/
int URBANCleanup();

#endif // _URBAN_H
