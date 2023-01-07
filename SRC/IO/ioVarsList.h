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
#define MAX_TYPE_LENGTH 16  //used for static allocation of type char array. Could be made dynamic

/*type definitions*/
typedef struct _ioVar_t {
   char    name[MAX_NAME_LENGTH];
   char    type[MAX_TYPE_LENGTH]; /* 1 = float, 2 = double, 3 = int, 4 = short, 5 = char, 6 = byte */
   int     nDims;
   int     dimids[MAXDIMS];
   void    *varMemAddress;
   int     ncvarid;
   struct _ioVar_t *next;
} ioVar_t;

ioVar_t *createList();
ioVar_t *getFirstVarFromList();
ioVar_t *getNamedVarFromList(char* name);
int addVarToList(char *name, char *type, int nDims, int *dimids, void *varMemAddress);
int printList();
void destroyList();

#endif // _IOVARSLIST_H
