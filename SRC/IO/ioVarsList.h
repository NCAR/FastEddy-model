/* FastEddy®: SRC/IO/ioVarsList.h
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
#ifndef _IOVARSLIST_H
#define _IOVARSLIST_H

#define MAXDIMS         16   //used for static allocation of dimids array. Could be made dynamic.
#define MAX_NAME_LENGTH 128  //used for static allocation of name char array. Could be made dynamic
#define MAX_TYPE_LENGTH 16   //used for static allocation of type char array. Could be made dynamic
#define MAX_ATTR_LENGTH 256  //used for static allocation of attribute strings
#define MAX_ATTRS       10   //maximum number of attributes per variable

/*attribute structure*/
typedef struct _ioAttr_t {
   char    name[MAX_NAME_LENGTH];
   char    type[MAX_TYPE_LENGTH];  /* "text", "float", "double", "int" */
   char    value[MAX_ATTR_LENGTH]; /* string representation of value */
} ioAttr_t;

/*type definitions*/
typedef struct _ioVar_t {
   char    name[MAX_NAME_LENGTH];
   char    type[MAX_TYPE_LENGTH]; /* 1 = float, 2 = double, 3 = int, 4 = short, 5 = char, 6 = byte */
   int     nDims;
   int     dimids[MAXDIMS];
   void    *varMemAddress;
   int     ncvarid;

  /* Attribute support */
   int     nAttrs;                /* number of attributes */
   ioAttr_t attrs[MAX_ATTRS];     /* array of attributes */
   
   struct _ioVar_t *next;
} ioVar_t;

ioVar_t *createList();
ioVar_t *getFirstVarFromList();
ioVar_t *getNamedVarFromList(char* name);
int addVarToList(char *name, char *type, int nDims, int *dimids, void *varMemAddress);
int printList();
void destroyList();

/* Add a single NetCDF attribute to an existing variable in the list
 * Parameters:
 *   varName - name of the variable to add attribute to
 *   attrName - name of the attribute
 *   attrType - type of attribute ("text", "float", "double", "int")
 *   attrValue - string representation of the attribute value
 * Returns: error code (0 = success, non-zero = error)
 */   
int addAttrToVar(char *varName, char *attrName, char *attrType, char *attrValue);

/* Add standard CF convention attributes to a variable (units, long_name, standard_name)
 * This is a convenience function for adding the most commonly used CF attributes
 * Parameters:
 *   varName - name of the variable to add attributes to
 *   units - units string (e.g., "m/s", "K", "kg/m^3") - can be NULL
 *   longName - descriptive long name for the variable - can be NULL
 *   standardName - CF standard name if applicable - can be NULL
 * Returns: error code (0 = success, non-zero = error)
 */
int addStandardAttrsToVar(char *varName, char *units, char *longName, char *standardName);

#endif // _IOVARSLIST_H
