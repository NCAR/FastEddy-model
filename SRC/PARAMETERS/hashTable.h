/* FastEddy®: SRC/PARAMETERS/hashTable.h 
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

#ifndef _HASHTABLE_H
#define _HASHTABLE_H

typedef char pair_name_t;

typedef union _value_t {
   float  fltVal;
   double dblVal;
   int    intVal;
   char   *strVal;
} value_t;

typedef struct _pair_value_t {
   char    *inputStr;
   value_t value;
   int     type;
   int     state;
} pair_value_t;

typedef struct _entry_t_ {
    pair_name_t *name;
    pair_value_t *val;
    struct _entry_t_ *next;
} entry_t;

typedef struct _hash_value_t {
    unsigned int hashval;
} hash_value_t;

typedef struct _hash_table_t_ {
    int size;       /* the size of the table */
    entry_t **table; /* the table elements */
} hash_table_t;

hash_table_t *create_hash_table(int size);
void free_table(hash_table_t *hashtable);
unsigned int hash(hash_table_t *hashtable, char *str);
entry_t *lookup_pair(hash_table_t *hashtable, char *str);
int add_pair(hash_table_t *hashtable, pair_name_t *str, pair_value_t *value);
int next_pr(int num);
int is_prime(int num);
char *str_trim(char *str);

#endif // _HASHTABLE_H
