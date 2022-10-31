#include <iostream>
#include <vector>
#include <complex>
#include <chrono>
#include <cmath>
#include <string>
#include <iterator>
#include <new>
#include "stdlib.h"

#include "Propagation.h"
#include "RayTrace.h"

#ifdef __cplusplus
    extern "C"
#endif

// DOUBLE PRECISION INTERFACE

extern "C" void propagateToGrid_JM(c2Bundle *res, reflparams source, reflparams target,
                                reflcontainer *cs, reflcontainer *ct,
                                c2Bundle *currents,
                                double k, int numThreads, double epsilon,
                                double t_direction)
{
    Propagation<double, c2Bundle, reflcontainer, c2Bundle> prop(k, numThreads, cs->size, ct->size, epsilon, t_direction);

    // Generate source and target grids
    generateGrid(source, cs);
    generateGrid(target, ct);

    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;

    printf("Calculating J and M...\n");
    begin = std::chrono::steady_clock::now();

    prop.parallelProp_JM(cs, ct, currents, res);

    end = std::chrono::steady_clock::now();

    std::cout << "Elapsed time : "
              << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count()
              << " [s]" << std::endl;
}

extern "C" void propagateToGrid_EH(c2Bundle *res, reflparams source, reflparams target,
                                reflcontainer *cs, reflcontainer *ct,
                                c2Bundle *currents,
                                double k, int numThreads, double epsilon,
                                double t_direction)
{
    Propagation<double, c2Bundle, reflcontainer, c2Bundle> prop(k, numThreads, cs->size, ct->size, epsilon, t_direction);

    // Generate source and target grids
    generateGrid(source, cs);
    generateGrid(target, ct);

    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;

    printf("Calculating E and H...\n");
    begin = std::chrono::steady_clock::now();

    prop.parallelProp_EH(cs, ct, currents, res);

    end = std::chrono::steady_clock::now();

    std::cout << "Elapsed time : "
              << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count()
              << " [s]" << std::endl;
}

extern "C" void propagateToGrid_JMEH(c4Bundle *res, reflparams source, reflparams target,
                                reflcontainer *cs, reflcontainer *ct,
                                c2Bundle *currents,
                                double k, int numThreads, double epsilon,
                                double t_direction)
{
    Propagation<double, c4Bundle, reflcontainer, c2Bundle> prop(k, numThreads, cs->size, ct->size, epsilon, t_direction);

    // Generate source and target grids
    generateGrid(source, cs);
    generateGrid(target, ct);

    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;

    printf("Calculating J, M, E and H...\n");
    begin = std::chrono::steady_clock::now();

    prop.parallelProp_JMEH(cs, ct, currents, res);

    end = std::chrono::steady_clock::now();
    std::cout << "Elapsed time : "
              << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count()
              << " [s]" << std::endl;
}

extern "C" void propagateToGrid_EHP(c2rBundle *res, reflparams source, reflparams target,
                                reflcontainer *cs, reflcontainer *ct,
                                c2Bundle *currents,
                                double k, int numThreads, double epsilon,
                                double t_direction)
{
    Propagation<double, c2rBundle, reflcontainer, c2Bundle> prop(k, numThreads, cs->size, ct->size, epsilon, t_direction);

    // Generate source and target grids
    generateGrid(source, cs);
    generateGrid(target, ct);

    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;

    printf("Calculating E, H and Pr...\n");
    begin = std::chrono::steady_clock::now();

    prop.parallelProp_EHP(cs, ct, currents, res);

    end = std::chrono::steady_clock::now();
    std::cout << "Elapsed time : "
              << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count()
              << " [s]" << std::endl;
}

extern "C" void propagateToGrid_scalar(arrC1 *res, reflparams source, reflparams target,
                                reflcontainer *cs, reflcontainer *ct,
                                arrC1 *field,
                                double k, int numThreads, double epsilon,
                                double t_direction)
{
    Propagation<double, arrC1, reflcontainer, arrC1> prop(k, numThreads, cs->size, ct->size, epsilon, t_direction);

    // Generate source and target grids
    generateGrid(source, cs);
    generateGrid(target, ct);

    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;

    printf("Calculating scalar field...\n");
    begin = std::chrono::steady_clock::now();

    prop.parallelPropScalar(cs, ct, field, res);

    end = std::chrono::steady_clock::now();
    std::cout << "Elapsed time : "
              << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count()
              << " [s]" << std::endl;
}

extern "C" void propagateToFarField(c2Bundle *res, reflparams source, reflparams target,
                                reflcontainer *cs, reflcontainer *ct,
                                c2Bundle *currents,
                                double k, int numThreads, double epsilon,
                                double t_direction)
{
    Propagation<double, c2Bundle, reflcontainer, c2Bundle> prop(k, numThreads, cs->size, ct->size, epsilon, t_direction);

    // Generate source and target grids
    generateGrid(source, cs);
    generateGrid(target, ct);

    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;

    printf("Calculating far-field...\n");
    begin = std::chrono::steady_clock::now();

    prop.parallelFarField(cs, ct, currents, res);

    end = std::chrono::steady_clock::now();
    std::cout << "Elapsed time : "
              << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count()
              << " [s]" << std::endl;
}

extern "C" void propagateRays(reflparams ctp, cframe *fr_in, cframe *fr_out,
                              int numThreads, double epsilon)
{
    int nTot = fr_in->size;
    RayTracer<reflparams, cframe, double> RT(numThreads, nTot, epsilon);

    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;

    printf("Calculating ray-trace...\n");
    begin = std::chrono::steady_clock::now();
    RT.parallelRays(ctp, fr_in, fr_out);

    end = std::chrono::steady_clock::now();
    std::cout << "Elapsed time : "
              << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count()
              << " [s]" << std::endl;
}

// SINGLE PRECISION INTERFACE
extern "C" void propagateToGridf_JM(c2Bundlef *res, reflparamsf source, reflparamsf target,
                                reflcontainerf *cs, reflcontainerf *ct,
                                c2Bundlef *currents,
                                float k, int numThreads, float epsilon,
                                float t_direction)
{
    Propagation<float, c2Bundlef, reflcontainerf, c2Bundlef> prop(k, numThreads, cs->size, ct->size, epsilon, t_direction);

    // Generate source and target grids
    generateGridf(source, cs);
    generateGridf(target, ct);

    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;

    printf("Calculating J and M...\n");
    begin = std::chrono::steady_clock::now();
    prop.parallelProp_JM(cs, ct, currents, res);

    end = std::chrono::steady_clock::now();

    std::cout << "Elapsed time : "
              << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count()
              << " [s]" << std::endl;
}

extern "C" void propagateToGridf_EH(c2Bundlef *res, reflparamsf source, reflparamsf target,
                                reflcontainerf *cs, reflcontainerf *ct,
                                c2Bundlef *currents,
                                float k, int numThreads, float epsilon,
                                float t_direction)
{
    Propagation<float, c2Bundlef, reflcontainerf, c2Bundlef> prop(k, numThreads, cs->size, ct->size, epsilon, t_direction);

    // Generate source and target grids
    generateGridf(source, cs);
    generateGridf(target, ct);

    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;

    printf("Calculating E and H...\n");
    begin = std::chrono::steady_clock::now();
    prop.parallelProp_EH(cs, ct, currents, res);

    end = std::chrono::steady_clock::now();

    std::cout << "Elapsed time : "
              << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count()
              << " [s]" << std::endl;
}

extern "C" void propagateToGridf_JMEH(c4Bundlef *res, reflparamsf source, reflparamsf target,
                                reflcontainerf *cs, reflcontainerf *ct,
                                c2Bundlef *currents,
                                float k, int numThreads, float epsilon,
                                float t_direction)
{
    Propagation<float, c4Bundlef, reflcontainerf, c2Bundlef> prop(k, numThreads, cs->size, ct->size, epsilon, t_direction);

    // Generate source and target grids
    generateGridf(source, cs);
    generateGridf(target, ct);

    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;

    printf("Calculating J, M, E and H...\n");
    begin = std::chrono::steady_clock::now();

    prop.parallelProp_JMEH(cs, ct, currents, res);

    end = std::chrono::steady_clock::now();
    std::cout << "Elapsed time : "
              << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count()
              << " [s]" << std::endl;
}

extern "C" void propagateToGridf_EHP(c2rBundlef *res, reflparamsf source, reflparamsf target,
                                reflcontainerf *cs, reflcontainerf *ct,
                                c2Bundlef *currents,
                                float k, int numThreads, float epsilon,
                                float t_direction)
{
    Propagation<float, c2rBundlef, reflcontainerf, c2Bundlef> prop(k, numThreads, cs->size, ct->size, epsilon, t_direction);

    // Generate source and target grids
    generateGridf(source, cs);
    generateGridf(target, ct);

    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;

    printf("Calculating E, H and Pr...\n");
    begin = std::chrono::steady_clock::now();

    prop.parallelProp_EHP(cs, ct, currents, res);

    end = std::chrono::steady_clock::now();
    std::cout << "Elapsed time : "
              << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count()
              << " [s]" << std::endl;
}

extern "C" void propagateToGridf_scalar(arrC1f *res, reflparamsf source, reflparamsf target,
                                reflcontainerf *cs, reflcontainerf *ct,
                                arrC1f *field,
                                float k, int numThreads, float epsilon,
                                float t_direction)
{
    Propagation<float, arrC1f, reflcontainerf, arrC1f> prop(k, numThreads, cs->size, ct->size, epsilon, t_direction);

    // Generate source and target grids
    generateGridf(source, cs);
    generateGridf(target, ct);

    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;

    printf("Calculating scalar field...\n");
    begin = std::chrono::steady_clock::now();

    prop.parallelPropScalar(cs, ct, field, res);

    end = std::chrono::steady_clock::now();
    std::cout << "Elapsed time : "
              << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count()
              << " [s]" << std::endl;
}

extern "C" void propagateToFarFieldf(c2Bundlef *res, reflparamsf source, reflparamsf target,
                                reflcontainerf *cs, reflcontainerf *ct,
                                c2Bundlef *currents,
                                float k, int numThreads, float epsilon,
                                float t_direction)
{
    Propagation<float, c2Bundlef, reflcontainerf, c2Bundlef> prop(k, numThreads, cs->size, ct->size, epsilon, t_direction);

    // Generate source and target grids
    generateGridf(source, cs);
    generateGridf(target, ct);

    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;

    printf("Calculating scalar field...\n");
    begin = std::chrono::steady_clock::now();

    prop.parallelFarField(cs, ct, currents, res);

    end = std::chrono::steady_clock::now();
    std::cout << "Elapsed time : "
              << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count()
              << " [s]" << std::endl;
}
