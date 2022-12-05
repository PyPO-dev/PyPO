#include <iostream>
#include <chrono>
#include <string>
#include <cmath>
#include <array>

//#include <cuda.h>
//#include <cuComplex.h>
//#include <cuda_runtime.h>

#include "GUtils.h"
#include "Structs.h"
#include "InterfaceReflector.h"

#define CSIZE 10
#define CSIZERT 5
#define MILLISECOND 1000

#ifdef _WIN32
#   define POPPY_DLL __declspec(dllexport);
#else
#   define POPPY_DLL
#endif

#ifndef __InterfaceCUDA_h
#define __InterfaceCUDA_h
/* Kernels for single precision PO.
 * Author: Arend Moerman
 * For questions, contact: arendmoerman@gmail.com
 */

extern "C"
{
    POPPY_DLL void callKernelf_JM(c2Bundlef *res, reflparamsf source, reflparamsf target,
               reflcontainerf *cs, reflcontainerf *ct,
               c2Bundlef *currents,
               float k, float epsilon,
               float t_direction, int nBlocks, int nThreads);

    POPPY_DLL void callKernelf_EH(c2Bundlef *res, reflparamsf source, reflparamsf target,
               reflcontainerf *cs, reflcontainerf *ct,
               c2Bundlef *currents,
               float k, float epsilon,
               float t_direction, int nBlocks, int nThreads);

    POPPY_DLL void callKernelf_JMEH(c4Bundlef *res, reflparamsf source, reflparamsf target,
               reflcontainerf *cs, reflcontainerf *ct,
               c2Bundlef *currents,
               float k, float epsilon,
               float t_direction, int nBlocks, int nThreads);

    POPPY_DLL void callKernelf_EHP(c2rBundlef *res, reflparamsf source, reflparamsf target,
               reflcontainerf *cs, reflcontainerf *ct,
               c2Bundlef *currents,
               float k, float epsilon,
               float t_direction, int nBlocks, int nThreads);

    POPPY_DLL void callKernelf_FF(c2Bundlef *res, reflparamsf source, reflparamsf target,
               reflcontainerf *cs, reflcontainerf *ct,
               c2Bundlef *currents,
               float k, float epsilon,
               float t_direction, int nBlocks, int nThreads);
    
    POPPY_DLL void callRTKernel(reflparamsf ctp, cframef *fr_in,
               cframef *fr_out, float epsilon, float t0,
               int nBlocks, int nThreads);
}

#endif
