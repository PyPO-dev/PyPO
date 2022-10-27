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
    Propagation<double, c2Bundle> prop(k, numThreads, cs->size, ct->size, epsilon, t_direction);

    // Generate source and target grids
    generateGridf(source, cs);
    generateGridf(target, ct);

    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;

    printf("Starting mode 0...\n");
    begin = std::chrono::steady_clock::now();

    prop.parallelProp_JM(cs, ct, currents, res);

    end = std::chrono::steady_clock::now();

    std::cout << "Calculation time "
              << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count()
              << " [s]" << std::endl;
}

extern "C" void propagateToGrid_EH(c2Bundle *res, double *xt, double *yt, double *zt,
                                double *xs, double *ys, double *zs,
                                double *rJxs, double *rJys, double *rJzs,
                                double *iJxs, double *iJys, double *iJzs,
                                double *rMxs, double *rMys, double *rMzs,
                                double *iMxs, double *iMys, double *iMzs,
                                double *area, int gt, int gs,
                                double k, int numThreads, double epsilon,
                                double t_direction)
{
    res->r1x = (double*)calloc(gt, sizeof(double));
    res->r1y = (double*)calloc(gt, sizeof(double));
    res->r1z = (double*)calloc(gt, sizeof(double));
    res->i1x = (double*)calloc(gt, sizeof(double));
    res->i1y = (double*)calloc(gt, sizeof(double));
    res->i1z = (double*)calloc(gt, sizeof(double));

    res->r2x = (double*)calloc(gt, sizeof(double));
    res->r2y = (double*)calloc(gt, sizeof(double));
    res->r2z = (double*)calloc(gt, sizeof(double));
    res->i2x = (double*)calloc(gt, sizeof(double));
    res->i2y = (double*)calloc(gt, sizeof(double));
    res->i2z = (double*)calloc(gt, sizeof(double));

    Propagation<double, c2Bundle> prop(k, numThreads, gs, gt, epsilon, t_direction);

    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;

    printf("Starting mode 1...\n");
    begin = std::chrono::steady_clock::now();
    prop.parallelProp_EH(xt, yt, zt,
                        xs, ys, zs,
                        rJxs, rJys, rJzs,
                        iJxs, iJys, iJzs,
                        rMxs, rMys, rMzs,
                        iMxs, iMys, iMzs,
                        area, res);

    end = std::chrono::steady_clock::now();

    std::cout << "Calculation time "
              << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count()
              << " [s]" << std::endl;
}

extern "C" void propagateToGrid_JMEH(c4Bundle *res,
                                double *xt, double *yt, double *zt,
                                double *xs, double *ys, double *zs,
                                double *nxt, double *nyt, double *nzt,
                                double *rJxs, double *rJys, double *rJzs,
                                double *iJxs, double *iJys, double *iJzs,
                                double *rMxs, double *rMys, double *rMzs,
                                double *iMxs, double *iMys, double *iMzs,
                                double *area, int gt, int gs,
                                double k, int numThreads, double epsilon,
                                double t_direction)
{
    res->r1x = (double*)calloc(gt, sizeof(double));
    res->r1y = (double*)calloc(gt, sizeof(double));
    res->r1z = (double*)calloc(gt, sizeof(double));
    res->i1x = (double*)calloc(gt, sizeof(double));
    res->i1y = (double*)calloc(gt, sizeof(double));
    res->i1z = (double*)calloc(gt, sizeof(double));

    res->r2x = (double*)calloc(gt, sizeof(double));
    res->r2y = (double*)calloc(gt, sizeof(double));
    res->r2z = (double*)calloc(gt, sizeof(double));
    res->i2x = (double*)calloc(gt, sizeof(double));
    res->i2y = (double*)calloc(gt, sizeof(double));
    res->i2z = (double*)calloc(gt, sizeof(double));

    res->r3x = (double*)calloc(gt, sizeof(double));
    res->r3y = (double*)calloc(gt, sizeof(double));
    res->r3z = (double*)calloc(gt, sizeof(double));
    res->i3x = (double*)calloc(gt, sizeof(double));
    res->i3y = (double*)calloc(gt, sizeof(double));
    res->i3z = (double*)calloc(gt, sizeof(double));

    res->r4x = (double*)calloc(gt, sizeof(double));
    res->r4y = (double*)calloc(gt, sizeof(double));
    res->r4z = (double*)calloc(gt, sizeof(double));
    res->i4x = (double*)calloc(gt, sizeof(double));
    res->i4y = (double*)calloc(gt, sizeof(double));
    res->i4z = (double*)calloc(gt, sizeof(double));

    Propagation<double, c4Bundle> prop(k, numThreads, gs, gt, epsilon, t_direction);

    prop.parallelProp_JMEH(xt, yt, zt,
                        xs, ys, zs,
                        nxt, nyt, nzt,
                        rJxs, rJys, rJzs,
                        iJxs, iJys, iJzs,
                        rMxs, rMys, rMzs,
                        iMxs, iMys, iMzs,
                        area, res);


}

extern "C" void propagateToGrid_EHP(c2rBundle *res,
                                double *xt, double *yt, double *zt,
                                double *xs, double *ys, double *zs,
                                double *nxt, double *nyt, double *nzt,
                                double *rJxs, double *rJys, double *rJzs,
                                double *iJxs, double *iJys, double *iJzs,
                                double *rMxs, double *rMys, double *rMzs,
                                double *iMxs, double *iMys, double *iMzs,
                                double *area, int gt, int gs,
                                double k, int numThreads, double epsilon,
                                double t_direction)
{
    res->r1x = (double*)calloc(gt, sizeof(double));
    res->r1y = (double*)calloc(gt, sizeof(double));
    res->r1z = (double*)calloc(gt, sizeof(double));
    res->i1x = (double*)calloc(gt, sizeof(double));
    res->i1y = (double*)calloc(gt, sizeof(double));
    res->i1z = (double*)calloc(gt, sizeof(double));

    res->r2x = (double*)calloc(gt, sizeof(double));
    res->r2y = (double*)calloc(gt, sizeof(double));
    res->r2z = (double*)calloc(gt, sizeof(double));
    res->i2x = (double*)calloc(gt, sizeof(double));
    res->i2y = (double*)calloc(gt, sizeof(double));
    res->i2z = (double*)calloc(gt, sizeof(double));

    res->r3x = (double*)calloc(gt, sizeof(double));
    res->r3y = (double*)calloc(gt, sizeof(double));
    res->r3z = (double*)calloc(gt, sizeof(double));

    Propagation<double, c2rBundle> prop(k, numThreads, gs, gt, epsilon, t_direction);

    prop.parallelProp_EHP(xt, yt, zt,
                        xs, ys, zs,
                        nxt, nyt, nzt,
                        rJxs, rJys, rJzs,
                        iJxs, iJys, iJzs,
                        rMxs, rMys, rMzs,
                        iMxs, iMys, iMzs,
                        area, res);
}

extern "C" void propagateToGrid_scalar(arrC1 *res, double *xt, double *yt, double *zt,
                                double *xs, double *ys, double *zs,
                                double *rEs, double *iEs,
                                double *area, int gt, int gs,
                                double k, int numThreads, double epsilon,
                                double t_direction)
{
    res->rx = (double*)calloc(gt, sizeof(double));
    res->ix = (double*)calloc(gt, sizeof(double));

    Propagation<double, arrC1> prop(k, numThreads, gs, gt, epsilon, t_direction);

    prop.parallelPropScalar(xt, yt, zt,
                            xs, ys, zs,
                            rEs, iEs,
                            area, res);
}

extern "C" void propagateToFarField(c2Bundle *res,
                                double *xt, double *yt,
                                double *xs, double *ys, double *zs,
                                double *rJxs, double *rJys, double *rJzs,
                                double *iJxs, double *iJys, double *iJzs,
                                double *rMxs, double *rMys, double *rMzs,
                                double *iMxs, double *iMys, double *iMzs,
                                double *area, int gt, int gs,
                                double k, int numThreads, double epsilon,
                                double t_direction)
{
    res->r1x = (double*)calloc(gt, sizeof(double));
    res->r1y = (double*)calloc(gt, sizeof(double));
    res->r1z = (double*)calloc(gt, sizeof(double));
    res->i1x = (double*)calloc(gt, sizeof(double));
    res->i1y = (double*)calloc(gt, sizeof(double));
    res->i1z = (double*)calloc(gt, sizeof(double));

    res->r2x = (double*)calloc(gt, sizeof(double));
    res->r2y = (double*)calloc(gt, sizeof(double));
    res->r2z = (double*)calloc(gt, sizeof(double));
    res->i2x = (double*)calloc(gt, sizeof(double));
    res->i2y = (double*)calloc(gt, sizeof(double));
    res->i2z = (double*)calloc(gt, sizeof(double));

    Propagation<double, c2Bundle> prop(k, numThreads, gs, gt, epsilon, t_direction);

    prop.parallelFarField(xt, yt,
                        xs, ys, zs,
                        rJxs, rJys, rJzs,
                        iJxs, iJys, iJzs,
                        rMxs, rMys, rMzs,
                        iMxs, iMys, iMzs,
                        area, res);
}

// SINGLE PRECISION INTERFACE
extern "C" void propagateToGridf_JM(c2Bundlef *res, float *xt, float *yt, float *zt,
                                float *xs, float *ys, float *zs,
                                float *nxt, float *nyt, float *nzt,
                                float *rJxs, float *rJys, float *rJzs,
                                float *iJxs, float *iJys, float *iJzs,
                                float *rMxs, float *rMys, float *rMzs,
                                float *iMxs, float *iMys, float *iMzs,
                                float *area, int gt, int gs,
                                float k, int numThreads, float epsilon,
                                float t_direction)
{
    // Since res contains null pointers to arrays of floats, assign new pointers to dynamic arrays
    res->r1x = (float*)calloc(gt, sizeof(float));
    res->r1y = (float*)calloc(gt, sizeof(float));
    res->r1z = (float*)calloc(gt, sizeof(float));
    res->i1x = (float*)calloc(gt, sizeof(float));
    res->i1y = (float*)calloc(gt, sizeof(float));
    res->i1z = (float*)calloc(gt, sizeof(float));

    res->r2x = (float*)calloc(gt, sizeof(float));
    res->r2y = (float*)calloc(gt, sizeof(float));
    res->r2z = (float*)calloc(gt, sizeof(float));
    res->i2x = (float*)calloc(gt, sizeof(float));
    res->i2y = (float*)calloc(gt, sizeof(float));
    res->i2z = (float*)calloc(gt, sizeof(float));

    Propagation<float, c2Bundlef> prop(k, numThreads, gs, gt, epsilon, t_direction);

    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;

    printf("Starting mode 0...\n");
    begin = std::chrono::steady_clock::now();
    prop.parallelProp_JM(xt, yt, zt,
                        xs, ys, zs,
                        nxt, nyt, nzt,
                        rJxs, rJys, rJzs,
                        iJxs, iJys, iJzs,
                        rMxs, rMys, rMzs,
                        iMxs, iMys, iMzs,
                        area, res);

    end = std::chrono::steady_clock::now();

    std::cout << "Calculation time "
              << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count()
              << " [s]" << std::endl;
}

extern "C" void propagateToGridf_EH(c2Bundlef *res, float *xt, float *yt, float *zt,
                                float *xs, float *ys, float *zs,
                                float *rJxs, float *rJys, float *rJzs,
                                float *iJxs, float *iJys, float *iJzs,
                                float *rMxs, float *rMys, float *rMzs,
                                float *iMxs, float *iMys, float *iMzs,
                                float *area, int gt, int gs,
                                float k, int numThreads, float epsilon,
                                float t_direction)
{
    res->r1x = (float*)calloc(gt, sizeof(float));
    res->r1y = (float*)calloc(gt, sizeof(float));
    res->r1z = (float*)calloc(gt, sizeof(float));
    res->i1x = (float*)calloc(gt, sizeof(float));
    res->i1y = (float*)calloc(gt, sizeof(float));
    res->i1z = (float*)calloc(gt, sizeof(float));

    res->r2x = (float*)calloc(gt, sizeof(float));
    res->r2y = (float*)calloc(gt, sizeof(float));
    res->r2z = (float*)calloc(gt, sizeof(float));
    res->i2x = (float*)calloc(gt, sizeof(float));
    res->i2y = (float*)calloc(gt, sizeof(float));
    res->i2z = (float*)calloc(gt, sizeof(float));

    Propagation<float, c2Bundlef> prop(k, numThreads, gs, gt, epsilon, t_direction);

    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;

    printf("Starting mode 1...\n");
    begin = std::chrono::steady_clock::now();
    prop.parallelProp_EH(xt, yt, zt,
                        xs, ys, zs,
                        rJxs, rJys, rJzs,
                        iJxs, iJys, iJzs,
                        rMxs, rMys, rMzs,
                        iMxs, iMys, iMzs,
                        area, res);

    end = std::chrono::steady_clock::now();

    std::cout << "Calculation time "
              << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count()
              << " [s]" << std::endl;
}

extern "C" void propagateToGridf_JMEH(c4Bundlef *res,
                                float *xt, float *yt, float *zt,
                                float *xs, float *ys, float *zs,
                                float *nxt, float *nyt, float *nzt,
                                float *rJxs, float *rJys, float *rJzs,
                                float *iJxs, float *iJys, float *iJzs,
                                float *rMxs, float *rMys, float *rMzs,
                                float *iMxs, float *iMys, float *iMzs,
                                float *area, int gt, int gs,
                                float k, int numThreads, float epsilon,
                                float t_direction)
{
    res->r1x = (float*)calloc(gt, sizeof(float));
    res->r1y = (float*)calloc(gt, sizeof(float));
    res->r1z = (float*)calloc(gt, sizeof(float));
    res->i1x = (float*)calloc(gt, sizeof(float));
    res->i1y = (float*)calloc(gt, sizeof(float));
    res->i1z = (float*)calloc(gt, sizeof(float));

    res->r2x = (float*)calloc(gt, sizeof(float));
    res->r2y = (float*)calloc(gt, sizeof(float));
    res->r2z = (float*)calloc(gt, sizeof(float));
    res->i2x = (float*)calloc(gt, sizeof(float));
    res->i2y = (float*)calloc(gt, sizeof(float));
    res->i2z = (float*)calloc(gt, sizeof(float));

    res->r3x = (float*)calloc(gt, sizeof(float));
    res->r3y = (float*)calloc(gt, sizeof(float));
    res->r3z = (float*)calloc(gt, sizeof(float));
    res->i3x = (float*)calloc(gt, sizeof(float));
    res->i3y = (float*)calloc(gt, sizeof(float));
    res->i3z = (float*)calloc(gt, sizeof(float));

    res->r4x = (float*)calloc(gt, sizeof(float));
    res->r4y = (float*)calloc(gt, sizeof(float));
    res->r4z = (float*)calloc(gt, sizeof(float));
    res->i4x = (float*)calloc(gt, sizeof(float));
    res->i4y = (float*)calloc(gt, sizeof(float));
    res->i4z = (float*)calloc(gt, sizeof(float));

    Propagation<float, c4Bundlef> prop(k, numThreads, gs, gt, epsilon, t_direction);

    prop.parallelProp_JMEH(xt, yt, zt,
                        xs, ys, zs,
                        nxt, nyt, nzt,
                        rJxs, rJys, rJzs,
                        iJxs, iJys, iJzs,
                        rMxs, rMys, rMzs,
                        iMxs, iMys, iMzs,
                        area, res);


}

extern "C" void propagateToGridf_EHP(c2rBundlef *res,
                                float *xt, float *yt, float *zt,
                                float *xs, float *ys, float *zs,
                                float *nxt, float *nyt, float *nzt,
                                float *rJxs, float *rJys, float *rJzs,
                                float *iJxs, float *iJys, float *iJzs,
                                float *rMxs, float *rMys, float *rMzs,
                                float *iMxs, float *iMys, float *iMzs,
                                float *area, int gt, int gs,
                                float k, int numThreads, float epsilon,
                                float t_direction)
{
    res->r1x = (float*)calloc(gt, sizeof(float));
    res->r1y = (float*)calloc(gt, sizeof(float));
    res->r1z = (float*)calloc(gt, sizeof(float));
    res->i1x = (float*)calloc(gt, sizeof(float));
    res->i1y = (float*)calloc(gt, sizeof(float));
    res->i1z = (float*)calloc(gt, sizeof(float));

    res->r2x = (float*)calloc(gt, sizeof(float));
    res->r2y = (float*)calloc(gt, sizeof(float));
    res->r2z = (float*)calloc(gt, sizeof(float));
    res->i2x = (float*)calloc(gt, sizeof(float));
    res->i2y = (float*)calloc(gt, sizeof(float));
    res->i2z = (float*)calloc(gt, sizeof(float));

    res->r3x = (float*)calloc(gt, sizeof(float));
    res->r3y = (float*)calloc(gt, sizeof(float));
    res->r3z = (float*)calloc(gt, sizeof(float));

    Propagation<float, c2rBundlef> prop(k, numThreads, gs, gt, epsilon, t_direction);

    prop.parallelProp_EHP(xt, yt, zt,
                        xs, ys, zs,
                        nxt, nyt, nzt,
                        rJxs, rJys, rJzs,
                        iJxs, iJys, iJzs,
                        rMxs, rMys, rMzs,
                        iMxs, iMys, iMzs,
                        area, res);
}

extern "C" void propagateToGridf_scalar(arrC1f *res, float *xt, float *yt, float *zt,
                                float *xs, float *ys, float *zs,
                                float *rEs, float *iEs,
                                float *area, int gt, int gs,
                                float k, int numThreads, float epsilon,
                                float t_direction)
{
    res->rx = (float*)calloc(gt, sizeof(float));
    res->ix = (float*)calloc(gt, sizeof(float));

    Propagation<float, arrC1f> prop(k, numThreads, gs, gt, epsilon, t_direction);

    prop.parallelPropScalar(xt, yt, zt,
                            xs, ys, zs,
                            rEs, iEs,
                            area, res);
}

extern "C" void propagateToFarFieldf(c2Bundlef *res,
                                float *xt, float *yt,
                                float *xs, float *ys, float *zs,
                                float *rJxs, float *rJys, float *rJzs,
                                float *iJxs, float *iJys, float *iJzs,
                                float *rMxs, float *rMys, float *rMzs,
                                float *iMxs, float *iMys, float *iMzs,
                                float *area, int gt, int gs,
                                float k, int numThreads, float epsilon,
                                float t_direction)
{
    res->r1x = (float*)calloc(gt, sizeof(float));
    res->r1y = (float*)calloc(gt, sizeof(float));
    res->r1z = (float*)calloc(gt, sizeof(float));
    res->i1x = (float*)calloc(gt, sizeof(float));
    res->i1y = (float*)calloc(gt, sizeof(float));
    res->i1z = (float*)calloc(gt, sizeof(float));

    res->r2x = (float*)calloc(gt, sizeof(float));
    res->r2y = (float*)calloc(gt, sizeof(float));
    res->r2z = (float*)calloc(gt, sizeof(float));
    res->i2x = (float*)calloc(gt, sizeof(float));
    res->i2y = (float*)calloc(gt, sizeof(float));
    res->i2z = (float*)calloc(gt, sizeof(float));

    Propagation<float, c2Bundlef> prop(k, numThreads, gs, gt, epsilon, t_direction);

    prop.parallelFarField(xt, yt,
                        xs, ys, zs,
                        rJxs, rJys, rJzs,
                        iJxs, iJys, iJzs,
                        rMxs, rMys, rMzs,
                        iMxs, iMys, iMzs,
                        area, res);
}
