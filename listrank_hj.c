#include "listutils.h"
#include <omp.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>


bool sublist_check(long* heads, long rando, int to_index)
{
	//returns true if the random index is in the list
	
	bool innit = false; //guvnah
	long *checka;
	int q=0;
	// A FOR-LOOP WITH AN 'IF'/break IS JUST A WHILE LOOP WITH OVERHEAD
	
	do{	checka = heads+q;
		if (*checka == rando)
		{	innit = true;}
		q++;
	} while(!innit && q<to_index);
	
	return innit;
}

void parallelListRanks (long head, const long* next, long* rank, size_t n)
{
	/* source : https://downey.io/blog/helman-jaja-list-ranking-explained/ How to make Helmann & JaJa list ranker*/
	/* source : https://www.openmp.org/spec-html/5.0/openmp.html  OpenMP API Specification */
	/* source : https://courses.cs.washington.edu/courses/cse332/21sp/lectures/19-ParallelPrefix.pdf Parallel Prefix-Sum */

	/* The first parameter is the head node location. */
	/* The second parameter is an array of the next locations. */
	/* The third is an array for your list rankings. */
	/* The fourth parameter is the size of the arrays. */

	// https://www.openmp.org/spec-html/5.0/openmpsu112.html

	/*
	1. Sequentially, partition the input list into *s* sublists, by randomly choosing 
		*s* sublist head nodes.
	2. In parallel, traverse each sublist computing the list ranking of each node 
		within the sublists.
	3. Sequentially, compute the list ranking of the head nodes overall 
		(by doing a prefix sum of the head node ranks).
	4. In parallel, traverse the sublists again to convert the sublist ranking 
		to the complete list ranking (by adding the previously determined prefix sum values).
	*/

	/* INITIALIZE VARIABLES */
	
	/* num_sublists = number of sublists to cut the main list into
	
	sublist_heads = heads of each sublist, including the main head
	sublist_tails = end of each sublist (next head), paired with heads
	
	rank = overall ranks

	n1 = indexing of next
	*/
	
	int num_sublists = omp_get_max_threads() + ceil(log2(n)); //because why not give a little extra
	if (num_sublists%2 == 1)
	{	num_sublists++;} // handles prefix sum odd# case

	long *sublist_heads = malloc(num_sublists * sizeof(long));
	long *sublist_heads_begin = sublist_heads;
	
	*sublist_heads = head; //assign initial head value
	
	long *rank_begin = rank;
	long *n1 = next;
	


	// generate random head values, doesn't point to *next = -1
		
	int range = n/num_sublists;
	
	#pragma omp parallel for private(sublist_heads, n1)
	for (int l=1; l<num_sublists; l++)
	{
		int rand_head; 

		sublist_heads = sublist_heads_begin + l;
		do
		{	rand_head = (range*(l)) + rand()%range;
			n1 = next + rand_head;
		} while (*n1 < 0 || rand_head == head);
		*sublist_heads = rand_head;
	}
	

	/* PARALLEL RANKINGS */
	
	long *sublist_tails = malloc(num_sublists * sizeof(long));
	long *sublist_tails_begin = sublist_tails;

	long rank_count = 0;
	long tail;

	rank = rank_begin + head;
	*rank = rank_count; //assign 0 count to head

	#pragma omp parallel for private(n1, rank, sublist_heads, sublist_tails, rank_count, tail)
	for (int w=0; w<num_sublists ;w++)
	{
		//set index equal for all sublist trackers
		sublist_heads = sublist_heads_begin + w;		
		
				
		// index to current head
		n1 = next + (*sublist_heads);//points at head index, value is next index

		// set counting trackers
		rank_count = 1;
				
		while(!(sublist_check(sublist_heads_begin, *n1, num_sublists) || *n1<0 ) )
		{
			// check if next index is the head of another sublist, therefore end of this one
			// if its another head or NIL, this is the last loop
			rank = rank_begin+(*n1); // goes to next index
			*rank = rank_count;
			rank_count++; 

			tail = *n1;
			n1 = next+(*n1); //sends *next pointer to the next index, making *n1's value the next index from the next index
		}

		sublist_tails = sublist_tails_begin + w;

		if (*n1 >= 0)
		{	rank = rank_begin+(*n1); //goes to tail location
			*rank = rank_count; //writes to tail, gives heads prefix sum value
			*sublist_tails = *n1; //which is a tail
		}
		else
		{	*rank = rank_count-1; //BUG CORRECTION because it is at the location of -1, instead of pointing from the previous locaiton
			*sublist_tails = tail; //points to end
		}

	}
		

	/* SEQUENTIAL PREFIX SUM */
	/*
	int indice = 0;

	long *rank_head;
	long *rank_tail;
	//printf("sublist tails in order\n");
	for (int v = 0; v<num_sublists; v++)
	{
		sublist_heads = sublist_heads_begin+indice;
		sublist_tails = sublist_tails_begin+indice;
		rank_head = rank_begin + (*sublist_heads);
		rank_tail = rank_begin + (*sublist_tails);

		//printf("%d ", *sublist_tails);
		//printf("%d ", *(sublist_tails_begin+indice));
		//printf("%d ", indice);
		
		*rank_tail = *rank_tail + (*rank_head);
		
				
		
		indice = 0;
		while (*sublist_heads != (*sublist_tails) && indice < num_sublists-1)
		{	indice++; 
			sublist_heads = sublist_heads_begin+indice;
		}

		
		
	} 
	
		
	/* PARALLEL PREFIX SUM */
	

	// get your heads in order
	long *next_head_ordered = malloc(num_sublists*sizeof(long)); //points to locations in sublist_tails in order instead of swapping, contains all tails 
	long *next_head_ordered_begin = next_head_ordered;

	*next_head_ordered = *sublist_tails_begin; //points to first tail

	//#pragma omp parallel for private (sublist_heads, sublist_tails, next_head_ordered) //INHERENTLY UNPARALLELIZABLE? needs previous head oh fudge
	for (int e = 0; e<(num_sublists-1); e++)
	{			
		int g = 0;
		next_head_ordered = next_head_ordered_begin+e;
				
		do
		{	g++;
			sublist_heads = sublist_heads_begin + g; 
		}while (*sublist_heads != (*next_head_ordered));
		
		
		next_head_ordered = next_head_ordered_begin + (e+1); //set location in array to accept address of sublist tail
		
		sublist_tails = sublist_tails_begin+g;
		
		*next_head_ordered = *sublist_tails;
		
	} 
	
	
	int iter_limit = ceil(log2(num_sublists)); //use to keep track of k's. num_sublists = first duration
	
	int *k = malloc((iter_limit)*sizeof(int));
	int *k_begin = k;
	
	int *k_next;
	
	int k_total;
	
	//populate sublist indices, needed to go backwards
	//9> {5>3>2>1}, {length} = ceil(log2(num_sublists))+1
	
	*k = num_sublists/2; // creates the index counter for iteration,  9>5>3>2>1 from n=18 ALWAYS EVEN
								// ?? make into indices 0 to 9>14>17>19>20  - each index is the beginning of the next chunk
	k_total = *k;
	// sequential no matter what
	for (int u=0; u<iter_limit-1; u++) //iterate over each value, and take half of the previous value
	{	
		k = k_begin+u;
		k_next = k_begin+(u+1);
		*k_next = ((*k)+1)/2; //ceil((double)(*k)/2); dang integer arithmetic
		k_total = k_total+(*k_next);
	}

	long *num_store = malloc((k_total)*sizeof(long)); // stores added values up tree for prefix rank, 18 >9+5+3+2+1 = 20 values = num_sublists+2 maximum, could be less
	long *num_store_begin = num_store;

	
	
	//going UP

	//first round, needed to get values from rank n indexed at next_head_ordered
	
	#pragma omp parallel for private(num_store)
	for (int d=0; d<(*k_begin); d++) //initialize values over first chunk
		{
		num_store = num_store_begin + d;
		
		*num_store = *(rank_begin+*(next_head_ordered_begin+(2*d))) + (*(rank_begin+(*(next_head_ordered_begin+(2*d+1))))); 		
		}				//value at the rank stored in next_head + value at the next rank in next_head
		
	
	int k_start = *k_begin; // sets num_store index to beginning of current block 9>*5>3>2>1 from n=18
	int k_prev = 0;	//sets num_store index of beginning of previous block

	//sequential
	for (int a = 1; a<iter_limit; a++) // do it qty(blocks) times
	{
		k = k_begin+a; //pointer for indices 9>*5>3>2>1 from n=18
					
		#pragma omp parallel for private(num_store) //WORKS only parallel here, due to sequential nature 
		for (int j=0; j<(*k); j++) //iterates over current block
		{	
			num_store = num_store_begin + k_start + j; //go to num_store index for current addition
			
			if (j == ((*k)-1) && k_prev%2 == 1) //LAST ITERATION, check for an odd number of previous values
			{
				*num_store = *(num_store_begin + ( k_prev+(2*j))); //single value
			}
			else
			{	*num_store = *(num_store_begin + ( k_prev+(2*j)) ) + ( *(num_store_begin + (k_prev+(2*j)+1) ) ); //adds two adjacent previous values together
			}

		
		}
		k_prev = k_start; //sets beginning of current block to beginning of previous block
		k_start = k_start+(*(k)); //cumulative index of block sizes, final value is k_total, one over the last index
			
	}
	
	
	//going DOWN
	// count backwards 
	// values can NOT be overwritten in original block, causes race condition

	num_store = num_store_begin + (k_total-1); //goes to final index
	*num_store = 0; //first from-left value

	
	long *num_store_2 = malloc((k_total)*sizeof(long)); // stores added values up tree for prefix rank, 18 >9+5+3+2+1 = 20 values = num_sublists+2 maximum, could be less
	long *num_store_2_begin = num_store_2;

	num_store_2 = num_store_2_begin + (k_total - 1);
	*num_store_2 = 0;
	num_store_2 = num_store_2_begin;

	int helper;
	
	k_start = k_total-3; //points to left-most index of (2) block, then iterates backwards from right-most index
	//k_prev = k_total-1; //location of (1)) index, left_most index of previous index
	
	//sequential
	for (int a = iter_limit-2; a>-1; a--) // from 2 to the zero index 9>5>3>*2>1 to n=18, final index needs special processing
	{
		// adds previous transit cumulatively to start index, since it is a length value it is the beginning index
		k = k_begin+a; //pointer for indices, starts at next to last
		
		helper = k_start +(*k)-1;
		
		#pragma omp parallel for private(num_store_2, k_prev) //WORKS only parallel here - race condition due to overwrite, declared second array to fix
		for (int j=(*k)-1; j>-1; j--)
		{				
			k_prev = helper + (j+2)/2;
			
			num_store_2 = num_store_2_begin + k_start + j; //go to added value location
						
			if (j%2 == 1)
			{
				*num_store_2 = *(num_store_begin + k_start + j - 1) + ( *(num_store_2_begin + k_prev) );
							// value to left of current value + previous value, overwrites value
			}
			else
			{
				*num_store_2 = *(num_store_2_begin + k_prev);
							//previous value, overwrites index
			}

			//printf("%d at %d, k_prev = %d, k = %d, k_start = %d, ceil = %d, (j+2)/2 = %d\n",*num_store_2, j, *(num_store_begin + k_prev), *k, k_start, helper , (j+2)/2);
			

		}
		
		if (a!=0)
			{k_start = k_start-(*(k-1));}
			
	}

	free(num_store_begin); 
	free(k);

	long *rank_holder = malloc(num_sublists*sizeof(long));
	long *rank_holder_begin = rank_holder; 
	
	//last round, need to get values from & write to rank
	
	#pragma omp parallel for private(num_store_2, rank, k_prev, rank_holder) // WORKS race condition writing to rank
	for (int d=num_sublists-1; d>-1; d--) //initial values
	{
		k_prev = (d+2)/2 - 1 ; //sets index

		num_store_2 = num_store_2_begin + k_prev;

		rank = rank_begin + (*(next_head_ordered_begin + d)); //goes to rank index, last to first

		rank_holder = rank_holder_begin + d;

		if (d%2 == 1)
		{
			*rank_holder = *rank + (*num_store_2) + ( *(rank_begin + (*(next_head_ordered_begin + d - 1)) ) );
				//current + previous     + rank to left in head index
		}
		else
		{
			*rank_holder = *rank + *num_store_2;
			//current + previous
		}		
	}
	
	#pragma omp parallel for private(rank, rank_holder) //WORKS
	for (int h=0;h<num_sublists; h++)
	{
		rank_holder = rank_holder_begin+h;
		rank = rank_begin + (*(next_head_ordered_begin + h));
		*rank = *rank_holder;
	}


	free(rank_holder_begin);
	free(next_head_ordered_begin);
	
	free(num_store_2_begin);
	


	/* PARALLEL RANK-OFFSET ADDITION */

	#pragma omp parallel for private(n1, rank, sublist_heads, sublist_tails, rank_count)
	for (int u=0; u<num_sublists ;u++) 
	{
		sublist_heads = sublist_heads_begin+u;
		sublist_tails = sublist_tails_begin+u;
		
		// index to current head
		rank = rank_begin + (*sublist_heads);
		n1 = next + (*sublist_heads);

		// set counting trackers
		
		rank_count = *rank; //takes value at head of sublist and increments
		rank_count++; 
		
		while(*n1 != (*sublist_tails) || *n1<0 )
		{
			rank = rank_begin+(*n1); 
			*rank = rank_count;
			rank_count++;
			n1 = next + (*n1);
		}

	}

	/* WRAP IT UP */
	
	rank = rank_begin;
	
	free(sublist_heads_begin); free(sublist_tails_begin); 

}
