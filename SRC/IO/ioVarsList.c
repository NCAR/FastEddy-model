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
    ptr->next = NULL;
    curr->next = ptr;
    curr = ptr;
    listSize++;

    return(0);
} //end addVarToList

int printList(){
   int i;
   ioVar_t *tmp;
   /*print the contents of the list from beginning to end*/
   i = 0;
   tmp = head;
   printf("Entry #: name, type, nDims, [dimids]:\n");
   while(tmp != NULL){
     switch (tmp->nDims){
       case 1:
         printf("%d: %s, %s, %d, [%d]\n",
                 i,tmp->name,tmp->type,tmp->nDims,tmp->dimids[0]);
         break;
       case 2:
         printf("%d: %s, %s, %d, [%d %d]\n",
                 i,tmp->name,tmp->type,tmp->nDims,tmp->dimids[0],tmp->dimids[1]);
         break;
       case 3:
         printf("%d: %s, %s, %d, [%d %d %d]\n",
                 i,tmp->name,tmp->type,tmp->nDims,tmp->dimids[0],tmp->dimids[1],tmp->dimids[2]);
         break;
       case 4:
         printf("%d: %s, %s, %d, [%d %d %d %d]\n",
                 i,tmp->name,tmp->type,tmp->nDims,tmp->dimids[0],tmp->dimids[1],tmp->dimids[2],tmp->dimids[3]);
         break;
       case 5:
         printf("%d: %s, %s, %d, [%d %d %d %d %d]\n",
           i,tmp->name,tmp->type,tmp->nDims,tmp->dimids[0],tmp->dimids[1],tmp->dimids[2],tmp->dimids[3],tmp->dimids[4]);
         break;
       default:
          printf("%d has nDims< 1 or nDims >5, no printing...\n", i);
         break;
      }//end switch tmp->nDims
      tmp = tmp->next;
      i++;
   }// end while

   return(i);
} //end printList

void destroyList(){
   int i;
   ioVar_t *tmp;
   ioVar_t *toDelete;
   /*free the list*/
   i = 0;
   tmp = head;
   while(tmp != NULL){
      toDelete = tmp;
      tmp = tmp->next;
#ifdef dEBUG
      printf("Deleting Entry %d: name = %s\n",i,toDelete->name);
#endif
      free(toDelete);
      i++;
   }// end while

   return;
} //end destroyList()

