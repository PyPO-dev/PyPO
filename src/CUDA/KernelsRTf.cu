#include "InterfaceCUDA.h"

/*! \file KernelsRTf.cu
    \brief Kernels for CUDA RT calculations.
    
    Contains kernels for RT calculations. Multiple kernels are defined, each one optimized for a certain calculation.
*/

// Declare constant memory for Device
__constant__ float conrt[CSIZERT]; // a, b, c, t0, epsilon
__constant__ float mat[16]; //
__constant__ int nTot;
__constant__ int cflip;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

/**
 * Check CUDA call.
 *
 * Wrapper for finding errors in CUDA API calls.
 *
 * @param code The errorcode returned from failed API call.
 * @param file The file in which failure occured.
 * @param line The line in file in which error occured.
 * @param abort Exit code upon error.
 */
__host__ inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/**
 * Debug real array.
 *
 * Print real valued array of size 3.
 *      Useful for debugging.

 * @param arr Array of 3 float.
 */
__host__ __device__ void _debugArrayf(float arr[3])
{
    printf("%f, %f, %f\n", arr[0], arr[1], arr[2]);
}

/**
  Calculate common factor 1.

  @param t Scaling factor.
  @param xr x co-ordinate of ray.
  @param yr y co-ordinate of ray.
  @param dxr x co-ordinate of ray direction
  @param dyr y co-ordinate of ray direction
  */
__device__ __inline__ float common1(float t, float xr, float yr, float dxr, float dyr)
{
    return (xr + t*dxr)*(xr + t*dxr)/(conrt[0]*conrt[0]) + (yr + t*dyr)*(yr + t*dyr)/(conrt[1]*conrt[1]);
}

/**
  Calculate common factor 2.

  @param t Scaling factor.
  @param xr x co-ordinate of ray.
  @param yr y co-ordinate of ray.
  @param dxr x co-ordinate of ray direction
  @param dyr y co-ordinate of ray direction
  */
__device__ __inline__ float common2(float t, float xr, float yr, float dxr, float dyr)
{
    return (xr + t*dxr)*2*dxr/(conrt[0]*conrt[0]) + (yr + t*dyr)*2*dyr/(conrt[1]*conrt[1]);
}

/**
  Calculate ray-paraboloid intersection.

  @param t Scaling factor.
  @param xr x co-ordinate of ray.
  @param yr y co-ordinate of ray.
  @param zr z co-ordinate of ray.
  @param dxr x co-ordinate of ray direction
  @param dyr y co-ordinate of ray direction
  @param dzr z co-ordinate of ray direction
  */
__device__ __inline__ float gp(float t, float xr, float yr, float zr, float dxr, float dyr, float dzr)
{
    return t - (zr + t*dzr - common1(t, xr, yr, dxr, dyr)) /
                (dzr - common2(t, xr, yr, dxr, dyr));
}


/**
  Calculate ray-hyperboloid intersection.

  @param t Scaling factor.
  @param xr x co-ordinate of ray.
  @param yr y co-ordinate of ray.
  @param zr z co-ordinate of ray.
  @param dxr x co-ordinate of ray direction
  @param dyr y co-ordinate of ray direction
  @param dzr z co-ordinate of ray direction
  */
__device__ __inline__ float gh(float t, float xr, float yr, float zr, float dxr, float dyr, float dzr)
{
    return t - (zr + t*dzr - conrt[2]*sqrt(common1(t, xr, yr, dxr, dyr) + 1)) /
                (dzr - conrt[2]/(2*sqrt(common1(t, xr, yr, dxr, dyr) + 1)) *
                common2(t, xr, yr, dxr, dyr));
}


/**
  Calculate ray-ellipsoid intersection.

  @param t Scaling factor.
  @param xr x co-ordinate of ray.
  @param yr y co-ordinate of ray.
  @param zr z co-ordinate of ray.
  @param dxr x co-ordinate of ray direction
  @param dyr y co-ordinate of ray direction
  @param dzr z co-ordinate of ray direction
  */
__device__ __inline__ float ge(float t, float xr, float yr, float zr, float dxr, float dyr, float dzr)
{
    return t - (zr + t*dzr - conrt[2]*sqrt(1 - common1(t, xr, yr, dxr, dyr))) /
                (dzr + conrt[2]/(2*sqrt(1 - common1(t, xr, yr, dxr, dyr))) *
                common2(t, xr, yr, dxr, dyr));
}


/**
  Calculate ray-plane intersection.

  @param t Scaling factor.
  @param xr x co-ordinate of ray.
  @param yr y co-ordinate of ray.
  @param zr z co-ordinate of ray.
  @param dxr x co-ordinate of ray direction
  @param dyr y co-ordinate of ray direction
  @param dzr z co-ordinate of ray direction
  */
__device__ __inline__ float gpl(float t, float xr, float yr, float zr, float dxr, float dyr, float dzr)
{
    return t - (zr + t*dzr) / dzr;
}


/**
  Calculate paraboloid normals.

  @param xr x co-ordinate of ray.
  @param yr y co-ordinate of ray.
  @param zr z co-ordinate of ray.
  @param out Array of 3 float.
  */
__device__ __inline__ void np(float xr, float yr, float zr, float (&out)[3])
{
    out[0] = -2 * xr / (conrt[0]*conrt[0]) * cflip;
    out[1] = -2 * yr / (conrt[1]*conrt[1]) * cflip;
    out[2] = cflip;

    float norm = sqrt(out[0]*out[0] + out[1]*out[1] + out[2]*out[2]);

    out[0] = out[0] / norm;
    out[1] = out[1] / norm;
    out[2] = out[2] / norm;
}


/**
  Calculate hyperboloid normals.

  @param xr x co-ordinate of ray.
  @param yr y co-ordinate of ray.
  @param zr z co-ordinate of ray.
  @param out Array of 3 float.
  */
__device__ __inline__ void nh(float xr, float yr, float zr, float (&out)[3])
{
    out[0] = -2 * xr / (conrt[0]*conrt[0]) * cflip;
    out[1] = -2 * yr / (conrt[1]*conrt[1]) * cflip;
    out[2] = 2 * zr / (conrt[2]*conrt[2]) * cflip;

    float norm = sqrt(out[0]*out[0] + out[1]*out[1] + out[2]*out[2]);

    out[0] = out[0] / norm;
    out[1] = out[1] / norm;
    out[2] = out[2] / norm;
}

/**
  Calculate ellipsoid normals.

  @param xr x co-ordinate of ray.
  @param yr y co-ordinate of ray.
  @param zr z co-ordinate of ray.
  @param out Array of 3 float.
  */
__device__ __inline__ void ne(float xr, float yr, float zr, float (&out)[3])
{
    out[0] = 2 * xr / (conrt[0]*conrt[0]) * cflip;
    out[1] = 2 * yr / (conrt[1]*conrt[1]) * cflip;
    out[2] = 2 * zr / (conrt[2]*conrt[2]) * cflip;

    float norm = sqrt(out[0]*out[0] + out[1]*out[1] + out[2]*out[2]);

    out[0] = out[0] / norm;
    out[1] = out[1] / norm;
    out[2] = out[2] / norm;
}


/**
  Calculate plane normals.

  @param xr x co-ordinate of ray.
  @param yr y co-ordinate of ray.
  @param zr z co-ordinate of ray.
  @param out Array of 3 float.
  */
__device__ __inline__ void npl(float xr, float yr, float zr, float (&out)[3])
{
    out[0] = 0;
    out[1] = 0;
    out[2] = cflip;
}

/**
  Matrix-vector multiplication.

  Uses mat from constant memory.

  @param cv1 Array of 3 float.
  @param out Array of 3 float.
  @param vec Whether to rotate as a vector or as a point.
  */
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

/**
  Matrix-vector multiplication.

  Uses mat from constant memory.

  @param cv1 Array of 3 float.
  @param out Array of 3 float.
  @param vec Whether to rotate as a vector or as a point.
  */
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

/**
  Transform rays to surface restframe.

  @param x Array of ray x co-ordinates.
  @param y Array of ray y co-ordinates.
  @param z Array of ray z co-ordinates.
  @param dx Array of ray x directions.
  @param dy Array of ray y directions.
  @param dz Array of ray z directions.
  @param i Index of co-ordinate.
  @param inv Whether to apply inverse transformation.
  */
__device__ __inline__ void transfRays(float *x, float *y, float *z,
                                  float *dx, float *dy, float *dz,
                                  int i, bool inv = false)
{
    bool vec = true;
    float inp[3], out[3];

    //if (i == 300) {printf("%f\n", mat[0]);}

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

/**
 * Initialize CUDA.
 *
 * Instantiate program and populate constant memory.
 *
 * @param ctp reflparamsf object containing target surface parameters.
 * @param epsilon Precision of NR method.
 * @param t0 Starting guess for NR method.
 * @param _nTot Total number of rays in beam.
 * @param nBlocks Number of blocks per grid.
 * @param nThreads Number of threads per block.
 *
 * @return BT Array of two dim3 objects.
 */
__host__ std::array<dim3, 2> _initCUDA(reflparamsf ctp, float epsilon, float t0,
                                      int _nTot, int nBlocks, int nThreads)
{
    // Calculate nr of blocks per grid and nr of threads per block
    dim3 nrb(nBlocks); dim3 nrt(nThreads);

    // Pack constant array
    cuFloatComplex _conrt[CSIZERT] = {ctp.coeffs[0], ctp.coeffs[1],
                                  ctp.coeffs[2], t0, epsilon};

    int iflip = 1;
    if (ctp.flip) {iflip = -1;}

    // Copy constant array to Device constant memory
    gpuErrchk( cudaMemcpyToSymbol(conrt, &_conrt, CSIZERT * sizeof(float)) );
    gpuErrchk( cudaMemcpyToSymbol(mat, ctp.transf, 16 * sizeof(float)) );
    gpuErrchk( cudaMemcpyToSymbol(nTot, &_nTot, sizeof(int)) );
    gpuErrchk( cudaMemcpyToSymbol(cflip, &iflip, sizeof(int)) );

    std::array<dim3, 2> BT;
    BT[0] = nrb;
    BT[1] = nrt;

    return BT;
}

/**
  Optimize ray-paraboloid distance.

  Uses a Newton Rhapson (NR) method to find the point of ray-surface intersection.

  @param xs Array of ray x co-ordinates.
  @param ys Array of ray y co-ordinates.
  @param zs Array of ray z co-ordinates.
  @param dxs Array of ray x directions.
  @param dys Array of ray y directions.
  @param dzs Array of ray z directions.
  @param xt Array of ray x co-ordinates, to be filled.
  @param yt Array of ray y co-ordinates, to be filled.
  @param zt Array of ray z co-ordinates, to be filled.
  @param dxt Array of ray x directions, to be filled.
  @param dyt Array of ray y directions, to be filled.
  @param dzt Array of ray z directions, to be filled.
  */
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

        float _t = conrt[3];
        float t1 = 1e99;

        float check = fabs(t1 - _t);

        float x = xs[idx];
        float y = ys[idx];
        float z = zs[idx];

        float dx = dxs[idx];
        float dy = dys[idx];
        float dz = dzs[idx];

        while (check > conrt[4])
        {
            t1 = gp(_t, x, y, z, dx, dy, dz);

            check = fabs(t1 - _t);

            _t = t1;
        }

        xt[idx] = x + _t*dx;
        yt[idx] = y + _t*dy;
        zt[idx] = z + _t*dz;

        np(xt[idx], yt[idx], zt[idx], norms);
        check = (dx*norms[0] + dy*norms[1] + dz*norms[2]);

        dxt[idx] = dx - 2*check*norms[0];
        dyt[idx] = dy - 2*check*norms[1];
        dzt[idx] = dz - 2*check*norms[2];

        transfRays(xs, ys, zs, dxs, dys, dzs, idx);
        transfRays(xt, yt, zt, dxt, dyt, dzt, idx);
    }
}

/**
  Optimize ray-hyperboloid distance.

  Uses a Newton Rhapson (NR) method to find the point of ray-surface intersection.

  @param xs Array of ray x co-ordinates.
  @param ys Array of ray y co-ordinates.
  @param zs Array of ray z co-ordinates.
  @param dxs Array of ray x directions.
  @param dys Array of ray y directions.
  @param dzs Array of ray z directions.
  @param xt Array of ray x co-ordinates, to be filled.
  @param yt Array of ray y co-ordinates, to be filled.
  @param zt Array of ray z co-ordinates, to be filled.
  @param dxt Array of ray x directions, to be filled.
  @param dyt Array of ray y directions, to be filled.
  @param dzt Array of ray z directions, to be filled.
  */
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

        float _t = conrt[3];
        float t1 = 1e99;

        float check = fabs(t1 - _t);

        float x = xs[idx];
        float y = ys[idx];
        float z = zs[idx];

        float dx = dxs[idx];
        float dy = dys[idx];
        float dz = dzs[idx];

        while (check > conrt[4])
        {
            t1 = gh(_t, x, y, z, dx, dy, dz);

            check = fabs(t1 - _t);

            _t = t1;
        }
        xt[idx] = x + _t*dx;
        yt[idx] = y + _t*dy;
        zt[idx] = z + _t*dz;

        nh(xt[idx], yt[idx], zt[idx], norms);
        check = (dx*norms[0] + dy*norms[1] + dz*norms[2]);

        dxt[idx] = dx - 2*check*norms[0];
        dyt[idx] = dy - 2*check*norms[1];
        dzt[idx] = dz - 2*check*norms[2];

        transfRays(xs, ys, zs, dxs, dys, dzs, idx);
        transfRays(xt, yt, zt, dxt, dyt, dzt, idx);
    }
}

/**
  Optimize ray-ellipsoid distance.

  Uses a Newton Rhapson (NR) method to find the point of ray-surface intersection.

  @param xs Array of ray x co-ordinates.
  @param ys Array of ray y co-ordinates.
  @param zs Array of ray z co-ordinates.
  @param dxs Array of ray x directions.
  @param dys Array of ray y directions.
  @param dzs Array of ray z directions.
  @param xt Array of ray x co-ordinates, to be filled.
  @param yt Array of ray y co-ordinates, to be filled.
  @param zt Array of ray z co-ordinates, to be filled.
  @param dxt Array of ray x directions, to be filled.
  @param dyt Array of ray y directions, to be filled.
  @param dzt Array of ray z directions, to be filled.
  */
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

        float _t = conrt[3];
        float t1 = 1e99;

        float check = fabs(t1 - _t);

        float x = xs[idx];
        float y = ys[idx];
        float z = zs[idx];

        float dx = dxs[idx];
        float dy = dys[idx];
        float dz = dzs[idx];

        while (check > conrt[4])
        {
            t1 = ge(_t, x, y, z, dx, dy, dz);

            check = fabs(t1 - _t);

            _t = t1;
        }
        xt[idx] = x + _t*dx;
        yt[idx] = y + _t*dy;
        zt[idx] = z + _t*dz;

        ne(xt[idx], yt[idx], zt[idx], norms);
        check = (dx*norms[0] + dy*norms[1] + dz*norms[2]);

        dxt[idx] = dx - 2*check*norms[0];
        dyt[idx] = dy - 2*check*norms[1];
        dzt[idx] = dz - 2*check*norms[2];

        transfRays(xs, ys, zs, dxs, dys, dzs, idx);
        transfRays(xt, yt, zt, dxt, dyt, dzt, idx);
    }
}

/**
  Optimize ray-plane distance.

  Uses a Newton Rhapson (NR) method to find the point of ray-surface intersection.

  @param xs Array of ray x co-ordinates.
  @param ys Array of ray y co-ordinates.
  @param zs Array of ray z co-ordinates.
  @param dxs Array of ray x directions.
  @param dys Array of ray y directions.
  @param dzs Array of ray z directions.
  @param xt Array of ray x co-ordinates, to be filled.
  @param yt Array of ray y co-ordinates, to be filled.
  @param zt Array of ray z co-ordinates, to be filled.
  @param dxt Array of ray x directions, to be filled.
  @param dyt Array of ray y directions, to be filled.
  @param dzt Array of ray z directions, to be filled.
  */
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
      //if (idx == 0) {printf("%f\n", conrt[2]);}
        transfRays(xs, ys, zs, dxs, dys, dzs, idx, inv);

        float _t = conrt[3];
        float t1 = 1e99;

        float check = fabs(t1 - _t);

        float x = xs[idx];
        float y = ys[idx];
        float z = zs[idx];
        //printf("%f\n", x);
        float dx = dxs[idx];
        float dy = dys[idx];
        float dz = dzs[idx];

        while (check > conrt[4])
        {
            t1 = gpl(_t, x, y, z, dx, dy, dz);

            check = fabs(t1 - _t);

            _t = t1;
        }
        //printf("%.14f\n", check);

        xt[idx] = x + _t*dx;
        yt[idx] = y + _t*dy;
        zt[idx] = z + _t*dz;

        npl(xt[idx], yt[idx], zt[idx], norms);
        check = (dx*norms[0] + dy*norms[1] + dz*norms[2]);

        dxt[idx] = dx - 2*check*norms[0];
        dyt[idx] = dy - 2*check*norms[1];
        dzt[idx] = dz - 2*check*norms[2];

        transfRays(xs, ys, zs, dxs, dys, dzs, idx);
        transfRays(xt, yt, zt, dxt, dyt, dzt, idx);
    }
}

/**
  Call ray-trace Kernel.

  Calculate a new frame of rays. Several kernels can be called, depending on surface type.

  @param ctp reflparamsf object containing target surface parameters.
  @param fr_in Pointer to input cframef object.
  @param fr_out Pointer to output cframef object.
  @param epsilon Precision for NR method.
  @param t0 Starting guess for NR method.
  @param nBlocks Number of blocks in GPU grid.
  @param nThreads Number of threads in block.
  
  @see reflparamsf
  @see cframef
  */
void callRTKernel(reflparamsf ctp, cframef *fr_in,
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

    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;

    printf("Calculating ray-trace...\n");
    begin = std::chrono::steady_clock::now();

    if (ctp.type == 0)
    {
        propagateRaysToP<<<BT[0], BT[1]>>>(d_xs, d_ys, d_zs, d_dxs, d_dys, d_dzs,
                                          d_xt, d_yt, d_zt, d_dxt, d_dyt, d_dzt);
        //printf("made it\n");
        gpuErrchk( cudaDeviceSynchronize() );

    }

    else if (ctp.type == 1)
    {
        propagateRaysToH<<<BT[0], BT[1]>>>(d_xs, d_ys, d_zs, d_dxs, d_dys, d_dzs,
                                          d_xt, d_yt, d_zt, d_dxt, d_dyt, d_dzt);
        gpuErrchk( cudaDeviceSynchronize() );
    }

    else if (ctp.type == 2)
    {
        propagateRaysToE<<<BT[0], BT[1]>>>(d_xs, d_ys, d_zs, d_dxs, d_dys, d_dzs,
                                          d_xt, d_yt, d_zt, d_dxt, d_dyt, d_dzt);
        gpuErrchk( cudaDeviceSynchronize() );
    }

    else if (ctp.type == 3)
    {
        propagateRaysToPl<<<BT[0], BT[1]>>>(d_xs, d_ys, d_zs, d_dxs, d_dys, d_dzs,
                                          d_xt, d_yt, d_zt, d_dxt, d_dyt, d_dzt);
        gpuErrchk( cudaDeviceSynchronize() );
    }

    end = std::chrono::steady_clock::now();
    std::cout << "Elapsed time : "
              << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()
              << " [us]\n" << std::endl;

    gpuErrchk( cudaMemcpy(fr_out->x, d_xt, fr_in->size * sizeof(float), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(fr_out->y, d_yt, fr_in->size * sizeof(float), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(fr_out->z, d_zt, fr_in->size * sizeof(float), cudaMemcpyDeviceToHost) );

    gpuErrchk( cudaMemcpy(fr_out->dx, d_dxt, fr_in->size * sizeof(float), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(fr_out->dy, d_dyt, fr_in->size * sizeof(float), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(fr_out->dz, d_dzt, fr_in->size * sizeof(float), cudaMemcpyDeviceToHost) );
    
    gpuErrchk( cudaDeviceReset() );
}
