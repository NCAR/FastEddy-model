/* FastEddy®: SRC/PARAMETERS/parameters.h 
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

#ifndef _PARAMETERS_H
#define _PARAMETERS_H

/*parameters return codes */
#define PARAM_SUCCESS               0
#define PARAM_WARN_DEFAULTED        1000 /* use default */

#define PARAM_ERROR_MALLOC          200
#define PARAM_ERROR_FILE_NOT_FOUND  201
#define PARAM_ERROR_PATH_NOT_FOUND  202
#define PARAM_ERROR_INVALID_FLOAT   203
#define PARAM_ERROR_INVALID_DOUBLE  204
#define PARAM_ERROR_INVALID_INTEGER 205
#define PARAM_ERROR_ABSENT          206
#define PARAM_ERROR_DUPLICATE       207  
#define PARAM_ERROR_OVERWRITE_FAILED 208  
#define PARAM_ERROR_NOT_FOUND       209 /* not found - internal code error */

/* parameters requirement states */
#define PARAM_NOT_REQUIRED   0   /* must be 0 to match switch off */
#define PARAM_OPTIONAL       1   /* must be 1 to match switch on */
#define PARAM_MANDATORY      2

/*parameters requirements rules */
#define USE_MANDATORY(f) ((f) ? PARAM_MANDATORY : PARAM_NOT_REQUIRED)
#define USE_OPTIONAL(f)  ((f) ? PARAM_OPTIONAL  : PARAM_NOT_REQUIRED)

#include <stdio.h>

void parameters_init(void);
void parameters_clean(void);
int parameters_readFile(char *paramFile);
void printParameter(char *name, char *description);
void printComment(char *comment);
int getParameterErrors(void);
void outputParameters(FILE *out);
int queryFloatParameter(char *name, float *var, float min, float max, int reqStatus);
int queryDoubleParameter(char *name, double *var, double min, double max, int reqStatus);
int queryIntegerParameter(char *name, int *var, int min, int max, int reqStatus);
int queryStringParameter(char *name, char **str, int reqStatus);
int queryFileParameter(char *name, char **file, int reqStatus);
int queryPathParameter(char *name, char **path, int reqStatus);
int invalidateParameter(char *name);
void printUnusedParameters(void);
int overwriteFloatParameter(char *name, float *var, float newVal);
int overwriteDoubleParameter(char *name, double *var, double newVal);
int overwriteIntegerParameter(char *name, int *var, int newVal);

#endif // _PARAMETERS_H
