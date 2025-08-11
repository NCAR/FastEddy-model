/* FastEddy®: SRC/IO/ioVarsList.c
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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ioVarsList.h>


static ioVar_t *head = NULL;
static ioVar_t *curr = NULL;
static int listSize = 0;

ioVar_t *createList(){
   ioVar_t *ptr;

   ptr = (ioVar_t *)malloc(sizeof(ioVar_t));
   if(ptr == NULL){
      printf("Failed to create an entry for tmpEntry.\n");
      exit(0);
   }
   ptr->next = NULL;
   ptr->nAttrs = 0;  /* Initialize attribute count */
   head = ptr;
   curr = ptr;

   return(ptr);
} //end create_ioVarsList()

ioVar_t *getFirstVarFromList(){
   ioVar_t *retVal = NULL;
   
   retVal = head;

   return(retVal);
} //end getFirstVarFromList()

ioVar_t *getNamedVarFromList(char* name){
   ioVar_t *retVal = NULL;
   
   retVal = head;
   while((strcmp(retVal->name,name)!=0)&&(retVal->next != NULL)){ //this variable doesn't match check the next variable
    retVal = retVal -> next;
   }
   if(strcmp(retVal->name,name)!=0){  //if we haven't found this named variable
     return(NULL);
   }else{     //we did find it, return it...
     return(retVal);
   }
} //end getNamedVarFromList()

int addVarToList(char *name, char *type, int nDims, int *dimids, void *varMemAddress){
    ioVar_t *ptr;
    int i;

    if(head == NULL){
       ptr = createList();
    }else{
       ptr = (ioVar_t *)malloc(sizeof(ioVar_t));
       if(ptr == NULL){
          printf("Failed to create an entry for tmpEntry.\n");
          exit(0);
       }
    }
    strcpy(ptr->name,name);
    strcpy(ptr->type,type);
    ptr->nDims = nDims;
    for(i=0; i < nDims; i++){
       ptr->dimids[i] = dimids[i];       
    }//end for i
    ptr->varMemAddress = varMemAddress;
    ptr->nAttrs = 0;  /* Initialize attribute count */
    ptr->next = NULL;
    curr->next = ptr;
    curr = ptr;
    listSize++;

    return(0);
} //end addVarToList

int addAttrToVar(char *varName, char *attrName, char *attrType, char *attrValue){
    ioVar_t *ptr = getNamedVarFromList(varName);
    
    if(ptr == NULL){
        printf("addAttrToVar: Variable %s not found.\n", varName);
        return(-1);
    }
    
    if(ptr->nAttrs >= MAX_ATTRS){
        printf("addAttrToVar: Maximum attributes (%d) reached for variable %s.\n", MAX_ATTRS, varName);
        return(-1);
    }
    
    strcpy(ptr->attrs[ptr->nAttrs].name, attrName);
    strcpy(ptr->attrs[ptr->nAttrs].type, attrType);
    strcpy(ptr->attrs[ptr->nAttrs].value, attrValue);
    ptr->nAttrs++;
    
    return(0);
} //end addAttrToVar

int addStandardAttrsToVar(char *varName, char *units, char *longName, char *standardName){
    int result = 0;
    
    if(units != NULL && strlen(units) > 0){
        result |= addAttrToVar(varName, "units", "text", units);
    }
    
    if(longName != NULL && strlen(longName) > 0){
        result |= addAttrToVar(varName, "long_name", "text", longName);
    }
    
    if(standardName != NULL && strlen(standardName) > 0){
        result |= addAttrToVar(varName, "standard_name", "text", standardName);
    }
    
    return result;
} //end addStandardAttrsToVar

int printList(){
  int i, j;
   ioVar_t *tmp;
   /*print the contents of the list from beginning to end*/
   i = 0;
   tmp = head;
   printf("Entry #: name, type, nDims, [dimids]:\n");
   while(tmp != NULL){
     switch (tmp->nDims){
       case 1:
         printf("%d: %s, %s, %d, [%d], %d attrs\n",
                 i,tmp->name,tmp->type,tmp->nDims,tmp->dimids[0],tmp->nAttrs);
         break;
       case 2:
         printf("%d: %s, %s, %d, [%d %d], %d attrs\n",
                 i,tmp->name,tmp->type,tmp->nDims,tmp->dimids[0],tmp->dimids[1],tmp->nAttrs);
         break;
       case 3:
         printf("%d: %s, %s, %d, [%d %d %d], %d attrs\n",
                 i,tmp->name,tmp->type,tmp->nDims,tmp->dimids[0],tmp->dimids[1],tmp->dimids[2],tmp->nAttrs);
         break;
       case 4:
         printf("%d: %s, %s, %d, [%d %d %d %d], %d attrs\n",
                 i,tmp->name,tmp->type,tmp->nDims,tmp->dimids[0],tmp->dimids[1],tmp->dimids[2],tmp->dimids[3],tmp->nAttrs);
         break;
       case 5:
         printf("%d: %s, %s, %d, [%d %d %d %d %d], %d attrs\n",
           i,tmp->name,tmp->type,tmp->nDims,tmp->dimids[0],tmp->dimids[1],tmp->dimids[2],tmp->dimids[3],tmp->dimids[4],tmp->nAttrs);
         break;
       default:
          printf("%d has nDims< 1 or nDims >5, no printing...\n", i);
         break;
      }//end switch tmp->nDims

      /* Print attributes */
      for(j = 0; j < tmp->nAttrs; j++){
          printf("    Attr %d: %s (%s) = %s\n", j, tmp->attrs[j].name, tmp->attrs[j].type, tmp->attrs[j].value);
      }
           
      tmp = tmp->next;
      i++;
   }// end while

   return(i);
} //end printList

void destroyList(){
   
   ioVar_t *tmp;
   ioVar_t *toDelete;
   /*free the list*/
#ifdef DEBUG
   int i;
   i = 0;
#endif
   tmp = head;
   while(tmp != NULL){
      toDelete = tmp;
      tmp = tmp->next;
#ifdef DEBUG
      printf("Deleting Entry %d: name = %s\n",i,toDelete->name);
      i++;
#endif
      free(toDelete);
   }// end while

   return;
} //end destroyList()

