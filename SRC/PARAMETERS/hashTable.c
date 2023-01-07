/* FastEddy®: SRC/PARAMETERS/hashTable.c 
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
#include <math.h>
#include <ctype.h>
#include <hashTable.h>


hash_table_t *create_hash_table(int size){
    hash_table_t *new_table;
    int i;
    int prime_size;

    if (size<1) return NULL; /* invalid size for table */

    /* ensure the suggested hashtable size is a prime number */
    prime_size=next_pr(size);        
    /* Attempt to allocate memory for the table structure */
    if ((new_table = malloc(sizeof(hash_value_t))) == NULL) {
        return NULL;
    }
    
    /* Attempt to allocate memory for the table itself */
    if ((new_table->table = malloc(sizeof(entry_t *) * prime_size)) == NULL) {
        return NULL;
    }

    /* Initialize the elements of the table */
    for(i=0; i<prime_size; i++) new_table->table[i] = NULL;

    /* Set the table's size */
    new_table->size = prime_size;
#ifdef DEBUG
    printf("size = %d requested, prime_size = %d created.\n",size,prime_size);
#endif
    return new_table;
} /* end create_hash_table() */

unsigned int hash(hash_table_t *hashtable, char *str){
   int i,c;
   unsigned int hashval;

   /* we start our hash out at 0 */
   hashval = 0;

   /* Bruce McKenzie et al paper recommended hash function */
   for (i=0; i<strlen(str); i++) {
     c = str[i];
     if ((c>=97) && (c<=122)) {
       /* fold alpha case 97-122 onto 65-90 */
       c -= 32;
     }
     hashval = (hashval << 1) + c;
   }

   return hashval % hashtable->size;
}


entry_t *lookup_pair(hash_table_t *hashtable, char *str){
   entry_t *entry;
   unsigned int hashval = hash(hashtable, str);

   /* Go to the correct entry based on the hash value and see if str is
   *      * in the entry.  If it is, return return a pointer to the entry element.
   *           * If it isn't, the item isn't in the table, so return NULL.
   *                */
   for(entry = hashtable->table[hashval]; entry != NULL; entry = entry->next) {
        if (strcmp(str, entry->name) == 0) return entry;
   }
   return NULL;
}

int add_pair(hash_table_t *hashtable, pair_name_t *str, pair_value_t *value){
    entry_t *new_entry;
    entry_t *current_entry;
    unsigned int hashval = hash(hashtable, str);

    /* Attempt to allocate memory for entry */
    if ((new_entry = malloc(sizeof(entry_t))) == NULL) return 1;

    /* Does item already exist? */
    current_entry = lookup_pair(hashtable, str);
    /* item already exists, don't insert it again. */
    if (current_entry != NULL) return 2;
    /* Insert into entry */
    new_entry->name = strdup(str);
    new_entry->val = value;
    new_entry->next = hashtable->table[hashval];
    hashtable->table[hashval] = new_entry;

    return 0;
}

void free_table(hash_table_t *hashtable){
    int i;
    entry_t *entry, *temp;

    if (hashtable==NULL) return;

    /* Free the memory for every item in the table, including the 
    *      * strings themselves.
    *           */
    for(i=0; i<hashtable->size; i++) {
        entry = hashtable->table[i];
        while(entry!=NULL) {
            temp = entry;
            entry = entry->next;
            free(temp->name);
            free(temp->val);
            free(temp);
        }
    }

    /* Free the table itself */
    free(hashtable->table);
    free(hashtable);
}

/*function to find the next prime number greater than an argument */
int next_pr(int num){
    int c;
    if(num < 2)
        c = 2;
    else if (num == 2)
        c = 3;
    else if(num & 1){
        num += 2;
        c = is_prime(num) ? num : next_pr(num);
    } else
        c = next_pr(num-1);

    return c;
}

/* function to check if a number is prime */
int is_prime(int num){
    if((num & 1)==0)
        return num == 2;
    else {
        int i, limit = sqrt(num);
        for (i = 3; i <= limit; i+=2){
            if (num % i == 0)
                return 0;
        }
    }
    return 1;
}

/* ----------------------------------------------------------------------------
 *  * Trim the leading and trailing spaces from a string, str (return a modified str)
 *   * recipe:
 *    * 1 - skip leading spaces, using sbuf1 
 *     * 2 - shift remaining *sbuf1's to the left, using sbuf2
 *      * 3 - mark a new end of string
 *       * 4 - replace trailing spaces with '\0', using sbuf2
 *        * 5 - return the trimmed str
 *         *         */
char *str_trim(char *str){
  char *sbuf1;    /* for parsing the whole string */
  char *sbuf2;    /* for shifting & terminating/padding */

  /* skip leading spaces, shift remaining chars */
  for (sbuf1=str; isspace(*sbuf1); sbuf1++ );

  /* shift left remaining chars, via sbuf2 */
  for (sbuf2=str; *sbuf1; sbuf1++, sbuf2++) {
    /* shift left remaining chars, via sbuf2 */
    *sbuf2 = *sbuf1;
  }

  /* mark new end of string for str */
  *sbuf2-- = 0;

  /* replace trailing spaces with '\0' */
  while ( sbuf2 > str && isspace(*sbuf2) ) {
    /* pad with '\0's */
    *sbuf2-- = 0;
  }

  return str;
}

