#include "InterfaceCUDA.h"

/*! \file KernelsRTf.cu
    \brief Kernel for CUDA RT calculations.
    
    Contains kernel for RT calculations.
 */

// Declare constant memory for Device
__constant__ float conrt[CSIZERT]; // a, b, c, t0, epsilon
__constant__ float mat[16]; //
__constant__ int nTot;
__constant__ int cflip;
__constant__ int ctype;

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

    if (inv) {invmatVec4(mat, inp, out);}
    else {matVec4(mat, inp, out);}

    x[i] = out[0];
    y[i] = out[1];
    z[i] = out[2];

    inp[0] = dx[i];
    inp[1] = dy[i];
    inp[2] = dz[i];

    if (inv) {invmatVec4(mat, inp, out, vec);}
    else {matVec4(mat, inp, out, vec);}

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
    gpuErrchk( cudaMemcpyToSymbol(ctype, &(ctp.type), sizeof(int)) );

    std::array<dim3, 2> BT;
    BT[0] = nrb;
    BT[1] = nrt;

    return BT;
}

/**
  Optimize ray-target distance.

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
__global__ void propagateRaysToTarget(float *xs, float *ys, float *zs,
                                float *dxs, float *dys, float *dzs,
                                float *xt, float *yt, float *zt,
                                float *dxt, float *dyt, float *dzt)
{
    float norms[3];
    bool inv = true;

    float (*refl_func_ptr)(float, float, float, float, float, float, float);
    void (*refl_norm_ptr)(float, float, float, float (&arr)[3]);

    if (ctype == 0) 
    {
        refl_func_ptr = &gp;
        refl_norm_ptr = &np;
    }

    if (ctype == 1) 
    {
        refl_func_ptr = &gh;
        refl_norm_ptr = &nh;
    }
    
    else if (ctype == 2) 
    {
        refl_func_ptr = &ge;
        refl_norm_ptr = &ne;
    }
    
    else if (ctype == 3) 
    {
        refl_func_ptr = &gpl;
        refl_norm_ptr = &npl;
    }

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
            t1 = refl_func_ptr(_t, x, y, z, dx, dy, dz);

            check = fabs(t1 - _t);

            _t = t1;
        }
        if ((abs(round(dx)) == 0 && abs(round(dy)) == 0 && abs(round(dz)) == 0) || isnan(_t)) 
        {
            xt[idx] = x;
            yt[idx] = y;
            zt[idx] = z;
        
            dxt[idx] = 0; // Set at 2: since beta should be normalized, can select on 2
            dyt[idx] = 0;
            dzt[idx] = 0;
        } 
       
        else
        {
            xt[idx] = x + _t*dx;
            yt[idx] = y + _t*dy;
            zt[idx] = z + _t*dz;

            refl_norm_ptr(xt[idx], yt[idx], zt[idx], norms);
            check = (dx*norms[0] + dy*norms[1] + dz*norms[2]);

            dxt[idx] = dx - 2*check*norms[0];
            dyt[idx] = dy - 2*check*norms[1];
            dzt[idx] = dz - 2*check*norms[2];
        }
        transfRays(xs, ys, zs, dxs, dys, dzs, idx);
        transfRays(xt, yt, zt, dxt, dyt, dzt, idx);
    }
}
/**
  Call ray-trace Kernel.

  Calculate a new frame of rays on a target, given an input frame of rays.

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
    
    MemUtils memutil;

    int n_ds = 6;
    int n_dt = 6;
     
    std::vector<float*> vec_frdat = {fr_in->x, fr_in->y, fr_in->z, fr_in->dx, fr_in->dy, fr_in->dz};
    std::vector<float*> vec_frout = {fr_out->x, fr_out->y, fr_out->z, fr_out->dx, fr_out->dy, fr_out->dz};

    std::vector<float*> vec_frs = memutil.cuMallFloat(n_ds, fr_in->size);
    memutil.cuMemCpFloat(vec_frs, vec_frdat, fr_in->size); 
    
    std::vector<float*> vec_fro = memutil.cuMallFloat(n_dt, fr_in->size);

    propagateRaysToTarget<<<BT[0], BT[1]>>>(vec_frs[0], vec_frs[1], vec_frs[2], 
                                            vec_frs[3], vec_frs[4], vec_frs[5],
                                            vec_fro[0], vec_fro[1], vec_fro[2], 
                                            vec_fro[3], vec_fro[4], vec_fro[5]);
    
    gpuErrchk( cudaDeviceSynchronize() );

    memutil.cuMemCpFloat(vec_frout, vec_fro, fr_in->size, false);
    
    gpuErrchk( cudaDeviceReset() );
}
