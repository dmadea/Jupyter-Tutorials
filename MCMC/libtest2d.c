// libtest2d.c
#include <stdlib.h> // for malloc and free
#include <stdio.h>  // for printf

// create a 2d array from the 1d one
double ** convert2d(unsigned long len1, unsigned long len2, double * arr) {
    double ** ret_arr;

    // allocate the additional memory for the additional pointers
    ret_arr = (double **)malloc(sizeof(double*)*len1);

    // set the pointers to the correct address within the array
    for (int i = 0; i < len1; i++) {
        ret_arr[i] = &arr[i*len2];
    }

    // return the 2d-array
    return ret_arr;
}

// print the 2d array
void mul_10(unsigned long len1, unsigned long len2, double * array) {
	
	for (unsigned long i = 0; i < len1 * len2; i++) {
        
        array[i] *= 10.0;
    }
	
	
	/*

    // call the 1d-to-2d-conversion function
    double ** list2d = convert2d(len1, len2, list);

    // print the array just to show it works
    for (unsigned long index1 = 0; index1 < len1; index1++) {
        for (unsigned long index2 = 0; index2 < len2; index2++) {
            printf("%1.1f ", list2d[index1][index2]);
        }
        printf("\n");
    }

    // free the pointers (only)
    free(list2d);*/
}


// print the 2d array
void mul_10_fix(double * array) {
	
	for (unsigned long i = 0; i < 150 * 500; i++) {
        
        array[i] *= 10.0;
    }
	
	
	/*

    // call the 1d-to-2d-conversion function
    double ** list2d = convert2d(len1, len2, list);

    // print the array just to show it works
    for (unsigned long index1 = 0; index1 < len1; index1++) {
        for (unsigned long index2 = 0; index2 < len2; index2++) {
            printf("%1.1f ", list2d[index1][index2]);
        }
        printf("\n");
    }

    // free the pointers (only)
    free(list2d);*/
}



















