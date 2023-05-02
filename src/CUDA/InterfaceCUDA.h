#include <iostream>
#include <string>
#include <cmath>
#include <array>

#include "GUtils.h"
#include "Structs.h"
#include "InterfaceReflector.h"
#include "MemUtils.h"
#include "Debug.h"

#define CSIZE 10
#define CSIZERT 5
#define MILLISECOND 1000

#ifdef _WIN32
#   define PYPO_DLL __declspec(dllexport);
#else
#   define PYPO_DLL
#endif

#ifndef __InterfaceCUDA_h
#define __InterfaceCUDA_h
/*! \file InterfaceCUDA.h
    \brief Declarations of PO and RT library for GPU.

    Provides double and single precision interface for NVIDIA GPUs running CUDA.
*/

extern "C"
{
    PYPO_DLL void callKernelf_JM(c2Bundlef *res, reflparamsf source, reflparamsf target,
               reflcontainerf *cs, reflcontainerf *ct,
               c2Bundlef *currents,
               float k, float epsilon,
               float t_direction, int nBlocks, int nThreads);

    PYPO_DLL void callKernelf_EH(c2Bundlef *res, reflparamsf source, reflparamsf target,
               reflcontainerf *cs, reflcontainerf *ct,
               c2Bundlef *currents,
               float k, float epsilon,
               float t_direction, int nBlocks, int nThreads);

    PYPO_DLL void callKernelf_JMEH(c4Bundlef *res, reflparamsf source, reflparamsf target,
               reflcontainerf *cs, reflcontainerf *ct,
               c2Bundlef *currents,
               float k, float epsilon,
               float t_direction, int nBlocks, int nThreads);

    PYPO_DLL void callKernelf_EHP(c2rBundlef *res, reflparamsf source, reflparamsf target,
               reflcontainerf *cs, reflcontainerf *ct,
               c2Bundlef *currents,
               float k, float epsilon,
               float t_direction, int nBlocks, int nThreads);

    PYPO_DLL void callKernelf_FF(c2Bundlef *res, reflparamsf source, reflparamsf target,
               reflcontainerf *cs, reflcontainerf *ct,
               c2Bundlef *currents,
               float k, float epsilon,
               float t_direction, int nBlocks, int nThreads);
    
    PYPO_DLL void callKernelf_scalar(arrC1f *res, reflparamsf source, reflparamsf target,
               reflcontainerf *cs, reflcontainerf *ct, arrC1f *inp,
               float k, float epsilon,
               float t_direction, int nBlocks, int nThreads);
    
    PYPO_DLL void callRTKernel(reflparamsf ctp, cframef *fr_in,
               cframef *fr_out, float epsilon, float t0,
               int nBlocks, int nThreads);
}

#endif
