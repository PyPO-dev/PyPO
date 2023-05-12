#include <iostream>

#ifndef __Debug_h
#define __Debug_h

/*! \file Debug.h
    \brief Methods for printing complex or real arrays of length 3 for GPU.
*/

__host__ __device__ inline void _debugArray(cuFloatComplex arr[3]);
__host__ __device__ inline void _debugArray(float arr[3]);

#endif

/**
 * Debug complex array.
 *
 * Print complex array of size 3.
 *      Useful for debugging.

 * @param arr Array of 3 cuFloatComplex.
 */
__host__ __device__ inline void _debugArray(cuFloatComplex arr[3])
{
    printf("%e + %ej, %e + %ej, %e + %ej\n", arr[0].x, arr[0].y, arr[1].x, arr[1].y, arr[2].x, arr[2].y);
}

/**
 * Debug real array.
 *
 * Print real valued array of size 3.
 *      Useful for debugging.

 * @param arr Array of 3 float.
 */
__host__ __device__ inline void _debugArray(float arr[3])
{
    printf("%e, %e, %e\n", arr[0], arr[1], arr[2]);
}

