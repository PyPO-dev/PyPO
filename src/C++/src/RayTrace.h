#include <iostream>
#include <vector>
#include <complex>
#include <array>
#define _USE_MATH_DEFINES
#include <cmath>
#include <thread>
#include <iomanip>
#include <algorithm>
#include <new>

#include "Utils.h"
#include "Structs.h"
#include "RTRefls.h"

#ifndef __RayTracer_h
#define __RayTracer_h

template<class T, class U, class V>
class RayTracer
{
    int numThreads;
    int step;
    int nTot;

    V epsilon;

public:
    std::vector<std::thread> threadPool;

    RayTracer(int numThreads, int nTot, V epsilon);

    void propagateRaysToP(int start, int stop,
                      T ctp, U *fr_in, U *fr_out);
    void propagateRaysToH(int start, int stop,
                      T ctp, U *fr_in, U *fr_out);
    void propagateRaysToE(int start, int stop,
                      T ctp, U *fr_in, U *fr_out);
    void propagateRaysToPl(int start, int stop,
                      T ctp, U *fr_in, U *fr_out);

    void parallelRays(T ctp, U *fr_in, U *fr_out);

    void joinThreads();
};

template<class T, class U, class V>
RayTracer<T, U, V>::RayTracer(int numThreads, int nTot, V epsilon)
{
    this->numThreads = numThreads;
    this->step = ceil(nTot / numThreads);
    this->nTot = nTot;
    this->threadPool.resize(numThreads);
    this->epsilon = epsilon;

    printf("<<<--------- RT info --------->>>\n");
    printf("--- Rays          :   %d\n", nTot);
    printf("--- Threads       :   %d\n", numThreads);
    printf("--- Device        :   CPU\n");
    printf("<<<--------- RT info --------->>>\n");
    printf("\n");
}

template<class T, class U, class V>
void RayTracer<T, U, V>::propagateRaysToP(int start, int stop,
                  T ctp, U *fr_in, U *fr_out)
{
    RTRefls<V> refls(ctp.coeffs[0], ctp.coeffs[1], ctp.coeffs[2]);

    int flip = 1;

    if (ctp.flip)
    {
        flip = -1;
    }

    std::array<V, 3> norms;

    int jc = 0; // Counter

    for (int i=start; i<stop; i++)
    {
        V t0 = 100;
        V t1 = 1e99;

        V check = fabs(t1 - t0);

        V x = fr_in->x[i];
        V y = fr_in->y[i];
        V z = fr_in->z[i];

        V dx = fr_in->dx[i];
        V dy = fr_in->dy[i];
        V dz = fr_in->dz[i];

        while (check > epsilon)
        {
            t1 = refls.gp(t0, x, y, z, dx, dy, dz);


            check = fabs(t1 - t0);

            t0 = t1;
        }

        fr_out->x[i] = x + t0*dx;
        fr_out->y[i] = y + t0*dy;
        fr_out->z[i] = z + t0*dz;

        norms = refls.np(fr_out->x[i], fr_out->y[i], fr_out->z[i], flip);
        check = (dx*norms[0] + dy*norms[1] + dz*norms[2]);

        fr_out->dx[i] = dx - 2*check*norms[0];
        fr_out->dy[i] = dy - 2*check*norms[1];
        fr_out->dz[i] = dz - 2*check*norms[2];

        if((i * 100 / this->step) > jc and start == 0 * this->step)
        {
            std::cout << jc << " / 100" << '\r';
            std::cout.flush();
            jc++;
        }
    }
}

template<class T, class U, class V>
void RayTracer<T, U, V>::propagateRaysToH(int start, int stop,
                  T ctp, U *fr_in, U *fr_out)
{
    RTRefls<V> refls(ctp.coeffs[0], ctp.coeffs[1], ctp.coeffs[2]);

    int flip = 1;

    if (ctp.flip)
    {
        flip = -1;
    }

    std::array<V, 3> norms;

    for (int i=start; i<stop; i++)
    {
        V t0 = 100;
        V t1 = 1e99;

        V check = fabs(t1 - t0);

        V x = fr_in->x[i];
        V y = fr_in->y[i];
        V z = fr_in->z[i];

        V dx = fr_in->dx[i];
        V dy = fr_in->dy[i];
        V dz = fr_in->dz[i];
        while (check > epsilon)
        {
            t1 = refls.gh(t0, fr_in->x[i], fr_in->y[i], fr_in->z[i],
                      fr_in->dx[i], fr_in->dy[i], fr_in->dz[i]);

            check = fabs(t1 - t0);

            t0 = t1;
        }

        fr_out->x[i] = x + t0*dx;
        fr_out->y[i] = y + t0*dy;
        fr_out->z[i] = z + t0*dz;

        norms = refls.nhe(fr_out->x[i], fr_out->y[i], fr_out->z[i], flip);
        check = (dx*norms[0] + dy*norms[1] + dz*norms[2]);

        fr_out->dx[i] = dx - 2*check*norms[0];
        fr_out->dy[i] = dy - 2*check*norms[1];
        fr_out->dz[i] = dz - 2*check*norms[2];
    }
}

template<class T, class U, class V>
void RayTracer<T, U, V>::propagateRaysToE(int start, int stop,
                  T ctp, U *fr_in, U *fr_out)
{
    RTRefls<V> refls(ctp.coeffs[0], ctp.coeffs[1], ctp.coeffs[2]);

    int flip = 1;

    if (ctp.flip)
    {
        flip = -1;
    }

    std::array<V, 3> norms;

    for (int i=start; i<stop; i++)
    {
        V t0 = 100;
        V t1 = 1e99;

        V check = fabs(t1 - t0);

        V x = fr_in->x[i];
        V y = fr_in->y[i];
        V z = fr_in->z[i];

        V dx = fr_in->dx[i];
        V dy = fr_in->dy[i];
        V dz = fr_in->dz[i];
        while (check > epsilon)
        {
            t1 = refls.ge(t0, fr_in->x[i], fr_in->y[i], fr_in->z[i],
                      fr_in->dx[i], fr_in->dy[i], fr_in->dz[i]);

            check = fabs(t1 - t0);

            t0 = t1;
        }

        fr_out->x[i] = x + t0*dx;
        fr_out->y[i] = y + t0*dy;
        fr_out->z[i] = z + t0*dz;

        norms = refls.nhe(fr_out->x[i], fr_out->y[i], fr_out->z[i], flip);
        check = (dx*norms[0] + dy*norms[1] + dz*norms[2]);

        fr_out->dx[i] = dx - 2*check*norms[0];
        fr_out->dy[i] = dy - 2*check*norms[1];
        fr_out->dz[i] = dz - 2*check*norms[2];
    }
}

template<class T, class U, class V>
void RayTracer<T, U, V>::propagateRaysToPl(int start, int stop,
                  T ctp, U *fr_in, U *fr_out)
{
    RTRefls<V> refls(ctp.coeffs[0], ctp.coeffs[1], ctp.coeffs[2]);

    int flip = 1;

    if (ctp.flip)
    {
        flip = -1;
    }

    for (int i=start; i<stop; i++)
    {
        V t0 = 100;
        V t1 = 1e99;

        V check = fabs(t1 - t0);
        std::array<V, 3> norms;

        V x = fr_in->x[i];
        V y = fr_in->y[i];
        V z = fr_in->z[i];

        V dx = fr_in->dx[i];
        V dy = fr_in->dy[i];
        V dz = fr_in->dz[i];
        while (check > epsilon)
        {
            t1 = refls.gpl(t0, fr_in->x[i], fr_in->y[i], fr_in->z[i],
                      fr_in->dx[i], fr_in->dy[i], fr_in->dz[i]);

            check = fabs(t1 - t0);

            t0 = t1;
        }

        fr_out->x[i] = x + t0*dx;
        fr_out->y[i] = y + t0*dy;
        fr_out->z[i] = z + t0*dz;

        norms = refls.npl(fr_out->x[i], fr_out->y[i], fr_out->z[i], flip);
        check = (dx*norms[0] + dy*norms[1] + dz*norms[2]);

        fr_out->dx[i] = dx - 2*check*norms[0];
        fr_out->dy[i] = dy - 2*check*norms[1];
        fr_out->dz[i] = dz - 2*check*norms[2];
    }
}


template <class T, class U, class V>
void RayTracer<T, U, V>::parallelRays(T ctp, U *fr_in, U *fr_out)
{
    int final_step;

    for(int n=0; n<numThreads; n++)
    {
        if(n == (numThreads-1))
        {
            final_step = nTot;
        }

        else
        {
            final_step = (n+1) * step;
        }

        if (ctp.type == 0)
        {
            threadPool[n] = std::thread(&RayTracer::propagateRaysToP,
                                        this, n * step, final_step,
                                        ctp, fr_in, fr_out);
        }

        else if (ctp.type == 1)
        {
            threadPool[n] = std::thread(&RayTracer::propagateRaysToH,
                                        this, n * step, final_step,
                                        ctp, fr_in, fr_out);
        }

        else if (ctp.type == 2)
        {
            threadPool[n] = std::thread(&RayTracer::propagateRaysToE,
                                        this, n * step, final_step,
                                        ctp, fr_in, fr_out);
        }

        else if (ctp.type == 3)
        {
            threadPool[n] = std::thread(&RayTracer::propagateRaysToPl,
                                        this, n * step, final_step,
                                        ctp, fr_in, fr_out);
        }
    }
    joinThreads();
}

template <class T, class U, class V>
void RayTracer<T, U, V>::joinThreads()
{
    for (std::thread &t : threadPool)
    {
        if (t.joinable())
        {
            t.join();
        }
    }
}
#endif
