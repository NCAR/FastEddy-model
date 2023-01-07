/* FastEddy®: SRC/PARAMETERS/parameters.c 
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
#include <stdint.h>
#include <sys/stat.h>
#include <errno.h>
#include <string.h>
#include <strings.h>
#include <hashTable.h>
#include <parameters.h>

/* __________________________ Macro definitions ____________________________________*/
#define MAXLEN 256

/* value type definitions */
#define VALUE_TYPE_UNKNOWN 0
#define VALUE_TYPE_INTEGER 1
#define VALUE_TYPE_FLOAT   2
#define VALUE_TYPE_DOUBLE  3
#define VALUE_TYPE_STRING  4
#define VALUE_TYPE_FILE    5
#define VALUE_TYPE_PATH    6

/* value state definitions */
#define VALUE_STATE_NOT_USED     0x0  /*Base state*/
#define VALUE_STATE_VALID        0x1  /*present and valid*/
#define VALUE_STATE_INVALID      0x2  /*present but invalid*/
#define VALUE_STATE_DEFAULTED    0x3  /*not-found, made present with default value*/
#define VALUE_STATE_ABSENT       0x4  /*not-found, but required*/
#define VALUE_STATE_OVERWRITTEN  0x5  /*value was overwritten*/

/* Macros for parameter name-value pair and description printing */
#define SPACES "                                        "
#define WIDTH  18

#define OUTPUT_SIZE 100000
/*-------------------------------------- static local functions ----------------------------------------*/
static void printParametersWithState(int state);
/*-------------------------------------- static local variables ----------------------------------------*/
static int numErrors;
static hash_table_t *parameters_table;
static char *outputBuffer;
static char *top;
static char *valueState[] = { "NOT VALIDATED",
                              "IN VALUE",
                              "INVALID",
                              "DEFAULTED",
                              "ABSENT",
                              "OVERWRITTEN" };

/*-------------------------------------- module procedure definitions  ----------------------------------------*/
void parameters_init(void){
   /* Allocate/Initialize the static data strucutres used within the PARAMETERS module */
   numErrors=0;
   /* initialize output buffer */
   outputBuffer = (char *)malloc(OUTPUT_SIZE);
   if (outputBuffer == NULL) {
     printf("ERROR: Cannot allocate memory for output buffer.\n");
     exit(PARAM_ERROR_MALLOC);
   }
   top = outputBuffer;
   outputBuffer[0] = 0;
 
}//end parameters_init()

int parameters_readFile(char *paramFile){
  int returnCode = PARAM_SUCCESS;
  int paramLines;
  FILE *fptrParam;
  char strBuff[MAXLEN];
  int size_of_table;
  char *paramBuff, *valueBuff; //pointer to the name and value sections of a given line

  /* Open the parameters file */
  if((fptrParam=fopen(paramFile, "r"))== NULL){
    printf("ERROR: Falied to open parameters file, %s\n", paramFile);
    fflush(stdout);
    numErrors++;
    return(PARAM_ERROR_FILE_NOT_FOUND);
  }//end if unable to open parameters file

  /* 1st pass-- Count the lines in the parameters file */
  paramLines=0;
  while (fgets(strBuff, MAXLEN, fptrParam)!=NULL){
     paramLines = paramLines+1;
  }
  printf("Successfully opened %s and read %d lines.\n", paramFile, paramLines);
  /* Return the file pointer back to the beginning of the file. */
  fseek(fptrParam, 0, SEEK_SET);
 
  /* Create the parameters hash table */
  /*Using the number of elements squared to hopefully ensure unique keys for everything*/
  /*This shouldn't become an issue unless the number of lines gets incredibly massive*/
  size_of_table = paramLines*paramLines;
  if(size_of_table > 10000){
     printf("Warning more than 10000 entries were requested for the parameters hash table.\n");
     printf("Consider changing the requested table size in the PARAMETERS module.\n");
  }
  parameters_table = create_hash_table(size_of_table);
 
    /* 2nd pass-- Read and Parse parameters */
  while (fgets(strBuff, MAXLEN, fptrParam)!=NULL){
     /* Look for a comment character and terminate the line there */
     valueBuff = strchr(strBuff, '#');
     if(valueBuff != NULL){
       *valueBuff = 0; //Set the termination of this string at the starting point of any comment (i.e. #)
     }
     /* If the entire line was not a comment parse the name-value pair */
     if(strlen(strBuff)>0){
        valueBuff = strchr(strBuff, '=');
        /*if there is an '=' set the valueBuff to begin one character beyond the '=' */
        if(valueBuff != NULL){
          *valueBuff++ = 0;
        }
        /*trim the delimited parameter name*/
        paramBuff = str_trim(strBuff);
        /*trim the delimited parameter value*/
        valueBuff = str_trim(valueBuff);

        /* Search the hash table for this parameter */
        if(lookup_pair(parameters_table, paramBuff) != NULL){
          /* Ooops Found a duplicate parameter name!! Bail Out! */
          returnCode = PARAM_ERROR_DUPLICATE;
          numErrors++;
          printf("ERROR: Duplicate entries for parameter name--%s.\n",paramBuff);
        }else{ /*This parameter name is unique and does not exist in the hash table yet */
           /* Allocate memory for the parameter name */
           pair_name_t *name = (pair_name_t *) malloc((strlen(paramBuff)+1)*sizeof(pair_name_t));
           if(name == NULL){
              printf("ERROR: Cannot allocate memory for parameter name--%s.\n",paramBuff);
              exit(PARAM_ERROR_MALLOC);
           }
           strcpy(name, paramBuff);

           /* Allocate memory for the parameter value (string phase) */
           pair_value_t *value = (pair_value_t *) malloc(sizeof(pair_value_t));
           if(value == NULL){
              printf("ERROR: Cannot allocate memory for the value segment of parameter name--%s.\n",paramBuff);
              exit(PARAM_ERROR_MALLOC);
           }

           /* If non-blank allocate and copy otherwise set to NULL*/
           if(strlen(valueBuff) > 0){
             value->inputStr = (char *) malloc(strlen(valueBuff)+1);
             if(value->inputStr == NULL){
               printf("ERROR: Cannot allocate memory for the value->inputStr field of parameter name--%s.\n",paramBuff); 
               exit(PARAM_ERROR_MALLOC);
             }
             strcpy(value->inputStr, valueBuff);
           }else{
               value->inputStr = NULL;
           }

           /* set the remaining value_t fields */
           value->type = VALUE_TYPE_UNKNOWN;
           value->state = VALUE_STATE_NOT_USED;
           /* add the name-value pair to the hash table */
           add_pair(parameters_table, name, value);
#ifdef DEBUG
           int hashval;
           /*Query the hashval of this entry */
           hashval = hash(parameters_table, paramBuff);
           printf("Adding Parameter = %s to parameters_table->table[%d]\n",paramBuff,hashval);
#endif
        }//end if(duplicate)-else
     }//Non-comment section has length greater than 0
  } //end 2nd pass-- while all lines have not yet been read 

  /* Close the paramFile */
  fclose(fptrParam);

#ifdef DEBUG
  int i;
  /*Print all the parameters */ 
  for(i=0; i<parameters_table->size; i++) {
    if(parameters_table->table[i] != NULL){
      printf("Added to table[%d]: Parameter = %s, Value = %s\n",
             i,parameters_table->table[i]->name,parameters_table->table[i]->val->inputStr);
    }
  }//for i 
#endif

  return(returnCode);

} //end parameters_readFile()

void printParameter(char *name, char *description){
  uint32_t len = 0;

  /* find parameter to print */
  entry_t *entry = lookup_pair(parameters_table, name);
  if ( entry != NULL ) {

   /* print when parameter in all states except VALUE_STATE_NOT_USED */
   if ((entry->val != NULL) && (entry->val->state != VALUE_STATE_NOT_USED)) {

     /* print name */
     sprintf(top+len, "%s=", name);
     len = strlen(top);

     /* print value */
     if (entry->val->type == VALUE_TYPE_FLOAT) {
       sprintf(top+len, "%g", entry->val->value.fltVal);
       len = strlen(top);
     }else if (entry->val->type == VALUE_TYPE_DOUBLE) {
       sprintf(top+len, "%g", entry->val->value.dblVal);
       len = strlen(top);
     } else if (entry->val->type == VALUE_TYPE_INTEGER) {
       sprintf(top+len, "%d", entry->val->value.intVal);
       len = strlen(top);
     } else {
       if (entry->val->value.strVal != NULL) {
         sprintf(top+len, "%s", entry->val->value.strVal);
         len = strlen(top);
       }
     }
     top += len;

     /* insert more spaces */
     while (len < WIDTH) {
       sprintf(top, " ");
       len++;
       top++;
     }

     /* add comment indicator */
     sprintf(top," # ");
     top += strlen(top);

     /* print string for status if not in value */
     if (entry->val->state != VALUE_STATE_VALID) {
        sprintf(top, "(%s) ", valueState[entry->val->state]);
       top += strlen(top);
     }

     /* print description */
     printComment(description);
  } /* if we found the parameter  and value to print */
 } else { /*we didn't find a table entry for the parameter so return nothing */
   /* print name */
   printf("WARNING: parameter %s was not found but was requested for printing.\n", name);
 } //end if-else we found the parameter entry in the hash table 
} //end PrintParameter()

void printComment(char *comment){
  sprintf(top, "## %s\n", comment);
  top += strlen(top);
} // end printComment()

int queryFloatParameter(char *name, float *var, float min, float max, int reqStatus){
   int retCode = PARAM_SUCCESS;

   /* find parameter to print */
   entry_t *entry = lookup_pair(parameters_table, name);
   if(entry == NULL) { /*failed to find the parameter in the hash table*/
      /*if this isn't a MANDATORY parameter it is not a problem so just return*/
      if(reqStatus == PARAM_NOT_REQUIRED){
        return(retCode);
      }
      
      /* Create a pair entry */
      pair_value_t *newValue = (pair_value_t *) malloc(sizeof(pair_value_t));
      if(newValue == NULL){
         printf("ERROR: Cannot allocate memory for the parameter name--%s.\n", name);
         exit(PARAM_ERROR_MALLOC);
      }
      /* add the name-value pair to the hash table */
      newValue->inputStr = NULL; //Not going to be needed so NULL is appropriate
      add_pair(parameters_table, name, newValue);
      
      /* If optional, allow a default value */
      if(reqStatus == PARAM_OPTIONAL){
         retCode = PARAM_WARN_DEFAULTED;
         newValue->type = VALUE_TYPE_FLOAT;
         newValue->value.fltVal = *var;
         newValue->state = VALUE_STATE_DEFAULTED;
         printf("WARNING: optional parameter %s defaulted to %g.\n", name, *var);
      } else {
      /*else if mandatory log the error set the type and state and exit*/
         retCode = PARAM_ERROR_ABSENT;
         numErrors++;
         printf("ERROR: mandatory parameter %s has not been specified.\n", name);
         newValue->type = VALUE_TYPE_STRING;
         newValue->value.strVal = "??";
         newValue->state = VALUE_STATE_ABSENT;
      }// end if optional or mandatory 
   } else {  /* The name was found in the hash table */
      /* cast the inputStr to a double value. */
      float val;
      sscanf(entry->val->inputStr, "%f", &val);
      entry->val->type = VALUE_TYPE_FLOAT;

      /*Ensure the value is within the appropriate range limits*/
      if((val < min) || (val > max)){
         /* Outside limits so log the error and set the state to invalid */
         entry->val->state = VALUE_STATE_INVALID;
         retCode = PARAM_ERROR_INVALID_FLOAT;
         numErrors++;
         printf("ERROR: parameter '%s' value %g is outside limits [%g,%g].\n",
                 name, val, min, max);
      }else {
         /* the value checks out, return it */
         entry->val->value.fltVal = val;
         entry->val->state = VALUE_STATE_VALID;
         *var = val; 
      }//is the value in range?
   } //if this pair is in the hash table
   return(retCode);
} // end queryFloatParameter()

int queryDoubleParameter(char *name, double *var, double min, double max, int reqStatus){
   int retCode = PARAM_SUCCESS;

   /* find parameter to print */
   entry_t *entry = lookup_pair(parameters_table, name);
   if(entry == NULL) { /*failed to find the parameter in the hash table*/
      /*if this isn't a MANDATORY parameter it is not a problem so just return*/
      if(reqStatus == PARAM_NOT_REQUIRED){
        return(retCode);
      }
      
      /* Create a pair entry */
      pair_value_t *newValue = (pair_value_t *) malloc(sizeof(pair_value_t));
      if(newValue == NULL){
         printf("ERROR: Cannot allocate memory for the parameter name--%s.\n", name);
         exit(PARAM_ERROR_MALLOC);
      }
      /* add the name-value pair to the hash table */
      newValue->inputStr = NULL; //Not going to be needed so NULL is appropriate
      add_pair(parameters_table, name, newValue);
      
      /* If optional, allow a default value */
      if(reqStatus == PARAM_OPTIONAL){
         retCode = PARAM_WARN_DEFAULTED;
         newValue->type = VALUE_TYPE_DOUBLE;
         newValue->value.dblVal = *var;
         newValue->state = VALUE_STATE_DEFAULTED;
         printf("WARNING: optional parameter %s defaulted to %g.\n", name, *var);
      } else {
      /*else if mandatory log the error set the type and state and exit*/
         retCode = PARAM_ERROR_ABSENT;
         numErrors++;
         printf("ERROR: mandatory parameter %s has not been specified.\n", name);
         newValue->type = VALUE_TYPE_STRING;
         newValue->value.strVal = "??";
         newValue->state = VALUE_STATE_ABSENT;
      }// end if optional or mandatory 
   } else {  /* The name was found in the hash table */
      /* cast the inputStr to a double value. */
      double val;
      sscanf(entry->val->inputStr, "%lf", &val);
      entry->val->type = VALUE_TYPE_DOUBLE;

      /*Ensure the value is within the appropriate range limits*/
      if((val < min) || (val > max)){
         /* Outside limits so log the error and set the state to invalid */
         entry->val->state = VALUE_STATE_INVALID;
         retCode = PARAM_ERROR_INVALID_DOUBLE;
         numErrors++;
         printf("ERROR: parameter '%s' value %g is outside limits [%g,%g].\n",
                 name, val, min, max);
      }else {
         /* the value checks out, return it */
         entry->val->value.dblVal = val;
         entry->val->state = VALUE_STATE_VALID;
         *var = val; 
      }//is the value in range?
   } //if this pair is in the hash table
   return(retCode);
} // end queryDoubleParameter()

int queryIntegerParameter(char *name, int *var, int min, int max, int reqStatus){
   int retCode = PARAM_SUCCESS;

   /* find parameter to print */
   entry_t *entry = lookup_pair(parameters_table, name);
   if(entry == NULL) { /*failed to find the parameter in the hash table*/
      /*if this isn't a MANDATORY parameter it is not a problem so just return*/
      if(reqStatus == PARAM_NOT_REQUIRED){
        return(retCode);
      }
      
      /* Create a pair entry */
      pair_value_t *newValue = (pair_value_t *) malloc(sizeof(pair_value_t));
      if(newValue == NULL){
         printf("ERROR: Cannot allocate memory for the parameter name--%s.\n", name);
         exit(PARAM_ERROR_MALLOC);
      }
      /* add the name-value pair to the hash table */
      newValue->inputStr = NULL; //Not going to be needed so NULL is appropriate
      add_pair(parameters_table, name, newValue);
      
      /* If optional, allow a default value */
      if(reqStatus == PARAM_OPTIONAL){
         retCode = PARAM_WARN_DEFAULTED;
         newValue->type = VALUE_TYPE_INTEGER;
         newValue->value.intVal = *var;
         newValue->state = VALUE_STATE_DEFAULTED;
         printf("WARNING: optional parameter %s defaulted to %d.\n", name, *var);
      } else {
      /*else if mandatory log the error set the type and state and exit*/
         retCode = PARAM_ERROR_ABSENT;
         numErrors++;
         printf("ERROR: mandatory parameter %s has not been specified.\n", name);
         newValue->type = VALUE_TYPE_STRING;
         newValue->value.strVal = "??";
         newValue->state = VALUE_STATE_ABSENT;
      }// end if optional or mandatory 
   } else {  /* The name was found in the hash table */
      /* cast the inputStr to a int value. */
      int val;
      sscanf(entry->val->inputStr, "%d", &val);
      entry->val->type = VALUE_TYPE_INTEGER;

      /*Ensure the value is within the appropriate range limits*/
      if((val < min) || (val > max)){
         /* Outside limits so log the error and set the state to invalid */
         entry->val->state = VALUE_STATE_INVALID;
         retCode = PARAM_ERROR_INVALID_INTEGER;
         numErrors++;
         printf("ERROR: parameter '%s' value %d is outside limits [%d,%d].\n",
                 name, val, min, max);
      }else {
         /* the value checks out, return it */
         entry->val->value.intVal = val;
         entry->val->state = VALUE_STATE_VALID;
         *var = val; 
      }//is the value in range?
   } //if this pair is in the hash table
   return(retCode);
} // end queryIntegerParameter()

int queryStringParameter(char *name, char **string, int reqStatus){
   int retCode = PARAM_SUCCESS;
   *string = NULL;
  
   /* find parameter to query */
   entry_t *entry = lookup_pair(parameters_table, name);
   if(entry == NULL) { /*failed to find the parameter in the hash table*/
      /*if this isn't a MANDATORY parameter it is not a problem so just return*/
      if(reqStatus == PARAM_NOT_REQUIRED){
        retCode = PARAM_ERROR_NOT_FOUND;
        return(retCode);
      }

      /* Create a pair entry */
      pair_value_t *newValue = (pair_value_t *) malloc(sizeof(pair_value_t));
      if(newValue == NULL){
         printf("ERROR: Cannot allocate memory for the parameter name--%s.\n", name);
         exit(PARAM_ERROR_MALLOC);
      }
      /* add the name-value pair to the hash table */
      newValue->inputStr = NULL; //Not going to be needed so NULL is appropriate
      add_pair(parameters_table, name, newValue);

      /* If optional, allow a default value */
      if(reqStatus == PARAM_OPTIONAL){
         retCode = PARAM_WARN_DEFAULTED;
         newValue->type = VALUE_TYPE_STRING;
         newValue->value.strVal = NULL;
         newValue->state = VALUE_STATE_DEFAULTED;
         printf("WARNING: optional parameter %s defaulted to NULL.\n", name);
      } else {
      /*else if mandatory log the error set the type and state and exit*/
         retCode = PARAM_ERROR_ABSENT;
         numErrors++;
         printf("ERROR: mandatory parameter %s has not been specified.\n", name);
         newValue->type = VALUE_TYPE_STRING;
         newValue->value.strVal = "??";
         newValue->state = VALUE_STATE_ABSENT;
      }// end if optional or mandatory 
   } else {  /* The name was found in the hash table */
      /* Set the value type. */
      entry->val->type = VALUE_TYPE_STRING;
      entry->val->value.strVal = entry->val->inputStr;
      
      /*malloc the space in "string" and copy from the hashtable regitered value */ 
      entry->val->state = VALUE_STATE_VALID;
      if(entry->val->inputStr != NULL){
        *string = (char *) malloc(strlen(entry->val->inputStr)+1);
        strcpy(*string, entry->val->inputStr);
      } else {
        *string = NULL;
      }//if-else the entry inputStr was NULL

      if((*string == NULL) && (reqStatus == PARAM_MANDATORY)){
        retCode = PARAM_ERROR_ABSENT;
        numErrors++;
        entry->val->state = VALUE_STATE_ABSENT;
        printf("ERROR: mandatory parameter %s has not been specified.\n", name);
      }// end if the string was required but got set to NULL
     
   } //if-else the parameter entry was found in the hashtable 

   return(retCode);
} //end queryStringParameter()

int queryFileParameter(char *name, char **file, int reqStatus){
   int retCode = PARAM_SUCCESS;

   /* Use the string query */
   retCode = queryStringParameter(name, file, reqStatus);
   
   /*if the string query return success validate the file existence */
   if(retCode != PARAM_ERROR_ABSENT){
      /* find parameter to query and set the type to file*/
      entry_t *entry = lookup_pair(parameters_table, name);
      entry->val->type = VALUE_TYPE_FILE;

      /* ensure the file exists */
      if(entry->val->value.strVal != NULL){
         struct stat statbuf;
         if((stat(entry->val->value.strVal, &statbuf) == -1) && (errno == ENOENT)){
            if(reqStatus == PARAM_MANDATORY){  /* The file doesn't exist and this is a mandatory parameter */
               retCode = PARAM_ERROR_FILE_NOT_FOUND;
               numErrors++;
               printf("ERROR: could not find file %s.\n", entry->val->value.strVal);
               entry->val->state = VALUE_STATE_INVALID;
            } else {
               printf("WARNING: could not find file %s.\n", entry->val->value.strVal);
            } //if-else mandatory
         } //if file doesn't exist
      }//end if the strVal is not NULL
   } //if string query was successful 
   return(retCode);
}//end queryFileParameter()

int queryPathParameter(char *name, char **path, int reqStatus){
   int retCode = PARAM_SUCCESS;
   char *pathPortion;
   char *last;

   /* Use the string query */
   retCode = queryStringParameter(name, path, reqStatus);

   /*if the string query return success validate the file existence */
   if(retCode != PARAM_ERROR_ABSENT){
      /* find parameter to query and set the type to path*/
      entry_t *entry = lookup_pair(parameters_table, name);
      entry->val->type = VALUE_TYPE_PATH;

      /* ensure the path exists */
      if(entry->val->value.strVal != NULL){
         pathPortion = (char *)malloc(strlen(entry->val->value.strVal)+1);                 
         last = strrchr(pathPortion, '/');
         if(last != NULL){
           *last = 0;
         }// end if last not NULL
         struct stat statbuf;
         if((stat(entry->val->value.strVal, &statbuf) == -1) && (errno == ENOENT)){
            if(reqStatus == PARAM_MANDATORY){  /* The file doesn't exist and this is a mandatory parameter */
               retCode = PARAM_ERROR_PATH_NOT_FOUND;
               numErrors++;
               printf("ERROR: could not find path %s.\n", entry->val->value.strVal);
               entry->val->state = VALUE_STATE_INVALID;
            } else {
               printf("WARNING: could not find path %s.\n", entry->val->value.strVal);
            } //if-else mandatory
         } //if path doesn't exist
      }//end if the strVal is not NULL
   }//if string query was successful 
   return(retCode);
} //end queryPathParameter()

int overwriteFloatParameter(char *name, float *var, float newVal){
   int retCode = PARAM_SUCCESS;

   /* find parameter to ovverride*/
   entry_t *entry = lookup_pair(parameters_table, name);
   if(entry == NULL){
      retCode = PARAM_ERROR_OVERWRITE_FAILED;
      numErrors++;
      printf("ERROR: could not overwritte parameter %s since it was not found.\n", name);
   } else {
     printf("WARNING: parameter %s, value = %f overwritten with value = %f.\n", name, entry->val->value.fltVal, newVal);
     entry->val->value.fltVal = newVal;
     entry->val->state = VALUE_STATE_OVERWRITTEN;
     *var = newVal;
   }//end if-else parameter not found
   return(retCode);
}//end overwriteFloatParameter()

int overwriteDoubleParameter(char *name, double *var, double newVal){
   int retCode = PARAM_SUCCESS;

   /* find parameter to ovverride*/
   entry_t *entry = lookup_pair(parameters_table, name);
   if(entry == NULL){
      retCode = PARAM_ERROR_OVERWRITE_FAILED;
      numErrors++;
      printf("ERROR: could not overwritte parameter %s since it was not found.\n", name);
   } else {
     printf("WARNING: parameter %s, value = %lf overwritten with value = %lf.\n", name, entry->val->value.dblVal, newVal);
     entry->val->value.dblVal = newVal;
     entry->val->state = VALUE_STATE_OVERWRITTEN;
     *var = newVal;
   }//end if-else parameter not found
   return(retCode);
}//end overwriteDoubleParameter()

int overwriteIntegerParameter(char *name, int *var, int newVal){
   int retCode = PARAM_SUCCESS;

   /* find parameter to ovverride*/
   entry_t *entry = lookup_pair(parameters_table, name);
   if(entry == NULL){
      retCode = PARAM_ERROR_OVERWRITE_FAILED;
      numErrors++;
      printf("ERROR: could not overwritte parameter %s since it was not found.\n", name);
   } else {
     printf("WARNING: parameter %s, value = %d overwritten with value = %d.\n", name, entry->val->value.intVal, newVal);
     entry->val->value.intVal = newVal;
     entry->val->state = VALUE_STATE_OVERWRITTEN;
     *var = newVal;
   }//end if-else parameter not found
   return(retCode);
}//end overwriteIntegerParameter()

int invalidateParameter(char *name){
   int retCode = PARAM_SUCCESS;
   
   /* find parameter to query and set the type to path*/
   entry_t *entry = lookup_pair(parameters_table, name);
   if(entry == NULL){
      retCode = PARAM_ERROR_ABSENT;
      numErrors++;
      printf("ERROR: could not invalidate parameter %s since it was not found.\n", name); 
   } else {
     entry->val->state = VALUE_STATE_INVALID;
     numErrors++;
   }//end if-else parameter not found
   return(retCode);
}// end invalidateParameter()

void outputParameters(FILE *out) {
  fprintf(out, "\n%s\n", outputBuffer);
} //end outputParameters()

int getParameterErrors(void) {
  return numErrors;
} // end getParameterErrors()

static void printParametersWithState(int state){
  int i;
  /*iterate through all the  parameter table entries */
    for(i=0; i<parameters_table->size; i++) {
       if(parameters_table->table[i] != NULL){
         if(parameters_table->table[i]->val->state == state){
            printf("%s = %s   # %s\n", parameters_table->table[i]->name,
                    parameters_table->table[i]->val->inputStr, valueState[parameters_table->table[i]->val->state]);
         }//parameter has the specified state
       } //this table entry exists
    }//for i 
}//end printParameterWithState()

void printUnusedParameters(void){
  printParametersWithState(VALUE_STATE_NOT_USED);
}//end printUnusedParameters(void);

void parameters_clean(void){
   /* Cleanup the static data strucutres used within the PARAMETERS module */
   free(outputBuffer);
   top = NULL;
   free_table(parameters_table);
   parameters_table = NULL;

}//end parameters_clean()
