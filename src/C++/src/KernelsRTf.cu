#include <iostream>
#include <chrono>
#include <string>
#include <iterator>
#include <cmath>
#include <array>
#include <iomanip>

#include <cuda.h>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <unistd.h>

#include "GUtils.h"
#include "Structs.h"
#include "InterfaceReflector.h"
//#include "CompOverload.h"

#define CSIZE 5
#define MILLISECOND 1000

/* Kernels for single precision PO.
 * Author: Arend Moerman
 * For questions, contact: arendmoerman@gmail.com
 */

// Declare constant memory for Device
__constant__ float con[CSIZE]; // a, b, c, t0, epsilon
__constant__ float mat[16]; //
__constant__ int nTot;
__constant__ int cflip;

/**
 * Wrapper for finding errors in CUDA API calls.
 *
 * @param code The errorcode returned from failed API call.
 * @param file The file in which failure occured.
 * @param line The line in file in which error occured.
 * @param abort Exit code upon error.
 */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
__host__ inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__host__ __device__ void _debugArrayf(float arr[3])
{
    printf("%f, %f, %f\n", arr[0], arr[1], arr[2]);
}

__device__ __inline__ float common1(float t, float xr, float yr, float dxr, float dyr)
{
    return (xr + t*dxr)*(xr + t*dxr)/(con[0]*con[0]) + (yr + t*dyr)*(yr + t*dyr)/(con[1]*con[1]);
}


__device__ __inline__ float common2(float t, float xr, float yr, float dxr, float dyr)
{
    return (xr + t*dxr)*2*dxr/(con[0]*con[0]) + (yr + t*dyr)*2*dyr/(con[1]*con[1]);
}


__device__ __inline__ float gp(float t, float xr, float yr, float zr, float dxr, float dyr, float dzr)
{
    return t - (zr + t*dzr - common1(t, xr, yr, dxr, dyr)) /
                (dzr - common2(t, xr, yr, dxr, dyr));
}


__device__ __inline__ float gh(float t, float xr, float yr, float zr, float dxr, float dyr, float dzr)
{
    return t - (zr + t*dzr - con[2]*sqrt(common1(t, xr, yr, dxr, dyr) + 1)) /
                (dzr - con[2]/(2*sqrt(common1(t, xr, yr, dxr, dyr) + 1)) *
                common2(t, xr, yr, dxr, dyr));
}


__device__ __inline__ float ge(float t, float xr, float yr, float zr, float dxr, float dyr, float dzr)
{
    return t - (zr + t*dzr - con[2]*sqrt(1 - common1(t, xr, yr, dxr, dyr))) /
                (dzr + con[2]/(2*sqrt(1 - common1(t, xr, yr, dxr, dyr))) *
                common2(t, xr, yr, dxr, dyr));
}


__device__ __inline__ float gpl(float t, float xr, float yr, float zr, float dxr, float dyr, float dzr)
{
    return t - (zr + t*dzr) / dzr;
}


__device__ __inline__ void np(float xr, float yr, float zr, float (&out)[3])
{
    out[0] = -2 * xr / (con[0]*con[0]) * cflip;
    out[1] = -2 * yr / (con[1]*con[1]) * cflip;
    out[2] = cflip;

    float norm = sqrt(out[0]*out[0] + out[1]*out[1] + out[2]*out[2]);

    out[0] = out[0] / norm;
    out[1] = out[1] / norm;
    out[2] = out[2] / norm;
}


__device__ __inline__ void nhe(float xr, float yr, float zr, float (&out)[3])
{
    out[0] = -2 * xr / (con[0]*con[0]) * cflip;
    out[1] = -2 * yr / (con[1]*con[1]) * cflip;
    out[2] = 2 * zr / (con[2]*con[2]) * cflip;

    float norm = sqrt(out[0]*out[0] + out[1]*out[1] + out[2]*out[2]);

    out[0] = out[0] / norm;
    out[1] = out[1] / norm;
    out[2] = out[2] / norm;
}


__device__ __inline__ void npl(float xr, float yr, float zr, float (&out)[3])
{
    out[0] = 0;
    out[1] = 0;
    out[2] = cflip;
}

// Not placed in GUtils.h, because want to place rotMat in constant memory
__device__ __inline__ void matVec4(float (&cv1)[3], float (&out)[3], bool vec = false)
{
    if (vec)
    {
        for(int n=0; n<3; n++)
        {
            out[n] = mat[n*4] * cv1[0] + mat[1+n*4] * cv1[1] + mat[2+n*4] * cv1[2];
        }
    }

    else
    {
        for(int n=0; n<3; n++)
        {
            out[n] = mat[n*4] * cv1[0] + mat[1+n*4] * cv1[1] + mat[2+n*4] * cv1[2] + mat[3+n*4];
        }
    }
}

__device__ __inline__ void invmatVec4(float (&cv1)[3], float (&out)[3], bool vec = false)
{
    if (vec)
    {
        for(int n=0; n<3; n++)
        {
            out[n] = mat[n] * cv1[0] + mat[n+4] * cv1[1] + mat[n+8] * cv1[2];
        }
    }

    else
    {
        float temp;
        for(int n=0; n<3; n++)
        {
            temp = -mat[n]*mat[3] - mat[n+4]*mat[7] - mat[n+8]*mat[11];
            out[n] = mat[n] * cv1[0] + mat[n+4] * cv1[1] + mat[n+8] * cv1[2] + temp;
        }
    }
}

__device__ __inline__ void transfRays(float *x, float *y, float *z,
                                  float *dx, float *dy, float *dz,
                                  int i, bool inv = false)
{
    bool vec = true;
    float inp[3], out[3];

    inp[0] = x[i];
    inp[1] = y[i];
    inp[2] = z[i];

    if (inv) {invmatVec4(inp, out);}
    else {matVec4(inp, out);}

    x[i] = out[0];
    y[i] = out[1];
    z[i] = out[2];

    inp[0] = dx[i];
    inp[1] = dy[i];
    inp[2] = dz[i];

    if (inv) {invmatVec4(inp, out, vec);}
    else {matVec4(inp, out, vec);}

    dx[i] = out[0];
    dy[i] = out[1];
    dz[i] = out[2];

}

__host__ std::array<dim3, 2> _initCUDA(reflparamsf ctp, float epsilon, float t0,
                                      int nTot, int nBlocks, int nThreads)
{
    // Calculate nr of blocks per grid and nr of threads per block
    dim3 nrb(nBlocks); dim3 nrt(nThreads);

    // Pack constant array
    cuFloatComplex _con[CSIZE] = {ctp.coeffs[0], ctp.coeffs[1],
                                  ctp.coeffs[2], nTot, t0, epsilon};

    int iflip = 1;
    if (ctp.flip) {iflip = -1;}

    // Copy constant array to Device constant memory
    gpuErrchk( cudaMemcpyToSymbol(con, &_con, CSIZE * sizeof(float)) );
    gpuErrchk( cudaMemcpyToSymbol(mat, &ctp.transf, 16 * sizeof(float)) );
    gpuErrchk( cudaMemcpyToSymbol(cflip, &iflip, sizeof(int)) );

    std::array<dim3, 2> BT;
    BT[0] = nrb;
    BT[1] = nrt;

    return BT;
}

__global__ void propagateRaysToP(float *xs, float *ys, float *zs,
                                float *dxs, float *dys, float *dzs,
                                float *xt, float *yt, float *zt,
                                float *dxt, float *dyt, float *dzt)
{
    float norms[3];
    bool inv = true;

    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    if (idx < nTot)
    {
        transfRays(xs, ys, zs, dxs, dys, dzs, idx, inv);

        float _t = con[3];
        float t1 = 1e99;

        float check = fabs(t1 - _t);

        float x = xs[idx];
        float y = ys[idx];
        float z = zs[idx];

        float dx = dxs[idx];
        float dy = dys[idx];
        float dz = dzs[idx];

        while (check > con[4])
        {
            t1 = gp(_t, x, y, z, dx, dy, dz);

            check = fabs(t1 - _t);

            _t = t1;
        }
        xt[idx] = x + _t*dx;
        yt[idx] = y + _t*dy;
        zt[idx] = z + _t*dz;

        np(xt[idx], yt[idx], zt[idx], norms);
        check = (dxt[idx]*norms[0] + dyt[idx]*norms[1] + dzt[idx]*norms[2]);

        dxt[idx] = dx - 2*check*norms[0];
        dyt[idx] = dy - 2*check*norms[1];
        dzt[idx] = dz - 2*check*norms[2];

        transfRays(xs, ys, zs, dxs, dys, dzs, idx);
        transfRays(xt, yt, zt, dxt, dyt, dzt, idx);
    }
}

__global__ void propagateRaysToH(float *xs, float *ys, float *zs,
                                float *dxs, float *dys, float *dzs,
                                float *xt, float *yt, float *zt,
                                float *dxt, float *dyt, float *dzt)
{
    float norms[3];
    bool inv = true;

    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    if (idx < nTot)
    {
        transfRays(xs, ys, zs, dxs, dys, dzs, idx, inv);

        float _t = con[3];
        float t1 = 1e99;

        float check = fabs(t1 - _t);

        float x = xs[idx];
        float y = ys[idx];
        float z = zs[idx];

        float dx = dxs[idx];
        float dy = dys[idx];
        float dz = dzs[idx];

        while (check > con[4])
        {
            t1 = gh(_t, x, y, z, dx, dy, dz);

            check = fabs(t1 - _t);

            _t = t1;
        }
        xt[idx] = x + _t*dx;
        yt[idx] = y + _t*dy;
        zt[idx] = z + _t*dz;

        nhe(xt[idx], yt[idx], zt[idx], norms);
        check = (dxt[idx]*norms[0] + dyt[idx]*norms[1] + dzt[idx]*norms[2]);

        dxt[idx] = dx - 2*check*norms[0];
        dyt[idx] = dy - 2*check*norms[1];
        dzt[idx] = dz - 2*check*norms[2];

        transfRays(xs, ys, zs, dxs, dys, dzs, idx);
        transfRays(xt, yt, zt, dxt, dyt, dzt, idx);
    }
}

__global__ void propagateRaysToE(float *xs, float *ys, float *zs,
                                float *dxs, float *dys, float *dzs,
                                float *xt, float *yt, float *zt,
                                float *dxt, float *dyt, float *dzt)
{
    float norms[3];
    bool inv = true;

    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    if (idx < nTot)
    {
        transfRays(xs, ys, zs, dxs, dys, dzs, idx, inv);

        float _t = con[3];
        float t1 = 1e99;

        float check = fabs(t1 - _t);

        float x = xs[idx];
        float y = ys[idx];
        float z = zs[idx];

        float dx = dxs[idx];
        float dy = dys[idx];
        float dz = dzs[idx];

        while (check > con[4])
        {
            t1 = ge(_t, x, y, z, dx, dy, dz);

            check = fabs(t1 - _t);

            _t = t1;
        }
        xt[idx] = x + _t*dx;
        yt[idx] = y + _t*dy;
        zt[idx] = z + _t*dz;

        nhe(xt[idx], yt[idx], zt[idx], norms);
        check = (dxt[idx]*norms[0] + dyt[idx]*norms[1] + dzt[idx]*norms[2]);

        dxt[idx] = dx - 2*check*norms[0];
        dyt[idx] = dy - 2*check*norms[1];
        dzt[idx] = dz - 2*check*norms[2];

        transfRays(xs, ys, zs, dxs, dys, dzs, idx);
        transfRays(xt, yt, zt, dxt, dyt, dzt, idx);
    }
}

__global__ void propagateRaysToPl(float *xs, float *ys, float *zs,
                                float *dxs, float *dys, float *dzs,
                                float *xt, float *yt, float *zt,
                                float *dxt, float *dyt, float *dzt)
{
    float norms[3];
    bool inv = true;

    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    if (idx < nTot)
    {
        transfRays(xs, ys, zs, dxs, dys, dzs, idx, inv);

        float _t = con[3];
        float t1 = 1e99;

        float check = fabs(t1 - _t);

        float x = xs[idx];
        float y = ys[idx];
        float z = zs[idx];

        float dx = dxs[idx];
        float dy = dys[idx];
        float dz = dzs[idx];

        while (check > con[4])
        {
            t1 = gpl(_t, x, y, z, dx, dy, dz);

            check = fabs(t1 - _t);

            _t = t1;
        }
        xt[idx] = x + _t*dx;
        yt[idx] = y + _t*dy;
        zt[idx] = z + _t*dz;

        npl(xt[idx], yt[idx], zt[idx], norms);
        check = (dxt[idx]*norms[0] + dyt[idx]*norms[1] + dzt[idx]*norms[2]);

        dxt[idx] = dx - 2*check*norms[0];
        dyt[idx] = dy - 2*check*norms[1];
        dzt[idx] = dz - 2*check*norms[2];

        transfRays(xs, ys, zs, dxs, dys, dzs, idx);
        transfRays(xt, yt, zt, dxt, dyt, dzt, idx);
    }
}

extern "C" void callRTKernel(reflparamsf ctp, cframef *fr_in,
                            cframef *fr_out, float epsilon, float t0,
                            int nBlocks, int nThreads)
{
    std::array<dim3, 2> BT;
    BT = _initCUDA(ctp, epsilon, t0, fr_in->size, nBlocks, nThreads);

    float *d_xs, *d_ys, *d_zs;
    gpuErrchk( cudaMalloc((void**)&d_xs, fr_in->size * sizeof(float)) );
    gpuErrchk( cudaMalloc((void**)&d_ys, fr_in->size * sizeof(float)) );
    gpuErrchk( cudaMalloc((void**)&d_zs, fr_in->size * sizeof(float)) );

    gpuErrchk( cudaMemcpy(d_xs, fr_in->x, fr_in->size * sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_ys, fr_in->y, fr_in->size * sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_zs, fr_in->z, fr_in->size * sizeof(float), cudaMemcpyHostToDevice) );

    float *d_dxs, *d_dys, *d_dzs;
    gpuErrchk( cudaMalloc((void**)&d_dxs, fr_in->size * sizeof(float)) );
    gpuErrchk( cudaMalloc((void**)&d_dys, fr_in->size * sizeof(float)) );
    gpuErrchk( cudaMalloc((void**)&d_dzs, fr_in->size * sizeof(float)) );

    gpuErrchk( cudaMemcpy(d_dxs, fr_in->dx, fr_in->size * sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_dys, fr_in->dy, fr_in->size * sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_dzs, fr_in->dz, fr_in->size * sizeof(float), cudaMemcpyHostToDevice) );

    float *d_xt, *d_yt, *d_zt;
    gpuErrchk( cudaMalloc((void**)&d_xt, fr_in->size * sizeof(float)) );
    gpuErrchk( cudaMalloc((void**)&d_yt, fr_in->size * sizeof(float)) );
    gpuErrchk( cudaMalloc((void**)&d_zt, fr_in->size * sizeof(float)) );

    float *d_dxt, *d_dyt, *d_dzt;
    gpuErrchk( cudaMalloc((void**)&d_dxt, fr_in->size * sizeof(float)) );
    gpuErrchk( cudaMalloc((void**)&d_dyt, fr_in->size * sizeof(float)) );
    gpuErrchk( cudaMalloc((void**)&d_dzt, fr_in->size * sizeof(float)) );

    if (ctp.type == 0)
    {
        propagateRaysToP<<<BT[0], BT[1]>>>(d_xs, d_ys, d_zs, d_dxs, d_dys, d_dzs,
                                          d_xt, d_yt, d_zt, d_dxt, d_dyt, d_dzt);
    }

    else if (ctp.type == 1)
    {
        propagateRaysToH<<<BT[0], BT[1]>>>(d_xs, d_ys, d_zs, d_dxs, d_dys, d_dzs,
                                          d_xt, d_yt, d_zt, d_dxt, d_dyt, d_dzt);
    }

    else if (ctp.type == 2)
    {
        propagateRaysToE<<<BT[0], BT[1]>>>(d_xs, d_ys, d_zs, d_dxs, d_dys, d_dzs,
                                          d_xt, d_yt, d_zt, d_dxt, d_dyt, d_dzt);
    }

    else if (ctp.type == 3)
    {
        propagateRaysToPl<<<BT[0], BT[1]>>>(d_xs, d_ys, d_zs, d_dxs, d_dys, d_dzs,
                                          d_xt, d_yt, d_zt, d_dxt, d_dyt, d_dzt);
    }

    gpuErrchk( cudaMemcpy(fr_out->x, d_xt, fr_in->size * sizeof(float), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(fr_out->y, d_yt, fr_in->size * sizeof(float), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(fr_out->z, d_zt, fr_in->size * sizeof(float), cudaMemcpyDeviceToHost) );

    gpuErrchk( cudaMemcpy(fr_out->dx, d_dxt, fr_in->size * sizeof(float), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(fr_out->dy, d_dyt, fr_in->size * sizeof(float), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(fr_out->dz, d_dzt, fr_in->size * sizeof(float), cudaMemcpyDeviceToHost) );
}
