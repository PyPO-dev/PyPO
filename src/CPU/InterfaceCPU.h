#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <string>

#include "Propagation.h"
#include "RayTrace.h"

#ifdef _WIN32
#   define PYPO_DLL __declspec(dllexport)
#else
#   define PYPO_DLL
#endif

#ifndef __InterfaceCPU_h
#define __InterfaceCPU_h

/*! \file InterfaceCPU.h
    \brief Declarations of PO and RT library for CPU.

    Provides double and single precision interface for CPU PO and RT.
*/
extern "C"
{
    PYPO_DLL void propagateToGrid_JM(c2Bundle *res, reflparams source, reflparams target,
                         reflcontainer *cs, reflcontainer *ct,
                         c2Bundle *currents,
                         double k, int numThreads, double epsilon,
                         double t_direction);

    PYPO_DLL void propagateToGrid_EH(c2Bundle *res, reflparams source, reflparams target,
                         reflcontainer *cs, reflcontainer *ct,
                         c2Bundle *currents,
                         double k, int numThreads, double epsilon,
                         double t_direction);

    PYPO_DLL void propagateToGrid_JMEH(c4Bundle *res, reflparams source, reflparams target,
                         reflcontainer *cs, reflcontainer *ct,
                         c2Bundle *currents,
                         double k, int numThreads, double epsilon,
                         double t_direction);

    PYPO_DLL void propagateToGrid_EHP(c2rBundle *res, reflparams source, reflparams target,
                         reflcontainer *cs, reflcontainer *ct,
                         c2Bundle *currents,
                         double k, int numThreads, double epsilon,
                         double t_direction);

    PYPO_DLL void propagateToGrid_scalar(arrC1 *res, reflparams source, reflparams target,
                         reflcontainer *cs, reflcontainer *ct,
                         arrC1 *field,
                         double k, int numThreads, double epsilon,
                         double t_direction);

    PYPO_DLL void propagateToFarField(c2Bundle *res, reflparams source, reflparams target,
                         reflcontainer *cs, reflcontainer *ct,
                         c2Bundle *currents,
                         double k, int numThreads, double epsilon,
                         double t_direction);

    PYPO_DLL void propagateRays(reflparams ctp, cframe *fr_in, cframe *fr_out,
                       int numThreads, double epsilon, double t0);

    // SINGLE INTERFACE
    PYPO_DLL void propagateToGridf_JM(c2Bundlef *res, reflparamsf source, reflparamsf target,
                         reflcontainerf *cs, reflcontainerf *ct,
                         c2Bundlef *currents,
                         float k, int numThreads, float epsilon,
                         float t_direction);

    PYPO_DLL void propagateToGridf_EH(c2Bundlef *res, reflparamsf source, reflparamsf target,
                         reflcontainerf *cs, reflcontainerf *ct,
                         c2Bundlef *currents,
                         float k, int numThreads, float epsilon,
                         float t_direction);

    PYPO_DLL void propagateToGridf_JMEH(c4Bundlef *res, reflparamsf source, reflparamsf target,
                         reflcontainerf *cs, reflcontainerf *ct,
                         c2Bundlef *currents,
                         float k, int numThreads, float epsilon,
                         float t_direction);

    PYPO_DLL void propagateToGridf_EHP(c2rBundlef *res, reflparamsf source, reflparamsf target,
                         reflcontainerf *cs, reflcontainerf *ct,
                         c2Bundlef *currents,
                         float k, int numThreads, float epsilon,
                         float t_direction);

    PYPO_DLL void propagateToGridf_scalar(arrC1f *res, reflparamsf source, reflparamsf target,
                         reflcontainerf *cs, reflcontainerf *ct,
                         arrC1f *field,
                         float k, int numThreads, float epsilon,
                         float t_direction);

    PYPO_DLL void propagateToFarFieldf(c2Bundlef *res, reflparamsf source, reflparamsf target,
                                    reflcontainerf *cs, reflcontainerf *ct,
                                    c2Bundlef *currents,
                                    float k, int numThreads, float epsilon,
                                    float t_direction);
}

#endif
