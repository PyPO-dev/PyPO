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

#define CSIZE 10
#define MILLISECOND 1000

/* This program calculates the PO propagation between a source and a target plane.
 * NOTE: This file contains the CUDA version of PhysBeam
 *
 * In order to run, the presence of the following .txt files in the POPPy/src/C++/input/ is required:
 * - s_Jr_(x,y,z).txt the real x,y,z components of the source electric current distribution
 * - s_Ji_(x,y,z).txt the imaginary x,y,z components of the source electric current distribution
 * - s_Mr_(x,y,z).txt the real x,y,z components of the source magnetic current distribution
 * - s_Mi_(x,y,z).txt the imaginary x,y,z components of the source magnetic current distribution
 *
 * - s_(x,y,z).txt the source x,y,z grids
 * - s_n(x,y,z).txt the source nx,ny,nz normal grids
 * - A_s the source area elements corresponding to points x,y,z
 *
 * - t_(x,y,z).txt the target x,y,z grids
 * - t_n(x,y,z).txt the target nx,ny,nz normal grids
 *
 *
 * Author: Arend Moerman
 * For questions, contact: arendmoerman@gmail.com
 */

// Declare constant memory for Device
__constant__ cuFloatComplex con[CSIZE];     // Contains: k, eps, mu0, zeta0, pi, C_l, Time direction, unit, zero, c4 as complex numbers
//__constant__ cuDoubleComplex con[CSIZE];

__constant__ float eye[3][3];      // Identity matrix
__constant__ int g_s;               // Gridsize on source
__constant__ int g_t;               // Gridsize on target

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

__device__ __inline__ cuFloatComplex expCo(cuFloatComplex z)
{
    cuFloatComplex res;
    float t = exp(z.x);
    float ys = sin(z.y);
    float yc = cos(z.y);
    res.x = t*yc;
    res.y = t*ys;

    return res;
}

/**
 * (PRIVATE)
 * Instantiate program and populate constant memory.
 *
 * @param k Wavenumber of incoming field.
 * @param epsilon Relative permittivity of source.
 * @param ct->size Number of elements in target.
 * @param cs->size Number of elements in source.
 * @param t_direction Sign of exponent in Green function.
 * @param nBlock Number of blocks per grid.
 * @param nThreads Number of threads per block.
 *
 * @return BT Array of two dim3 objects.
 */

 __host__ std::array<dim3, 2> _initCUDA(float k, float epsilon, int gt, int gs, float t_direction, int nBlocks, int nThreads)
 {
     // Calculate nr of blocks per grid and nr of threads per block
     dim3 nrb(nBlocks); dim3 nrt(nThreads);

     float M_PIf = 3.1415926; /* pi */
     float C_L = 2.9979246e11; // mm s^-1
     float MU_0 = 1.256637e-3; // kg mm s^-2 A^-2
     float EPS_VAC = 1 / (MU_0 * C_L*C_L);
     float ZETA_0_INV = 1 / (C_L * MU_0);

     // Calculate permittivity of target
     float EPS = EPS_VAC * epsilon;

     // Fill ID matrix
     float _eye[3][3];
     _eye[0][0] = 1.;
     _eye[1][1] = 1.;
     _eye[2][2] = 1.;

     _eye[0][1] = 0.;
     _eye[0][2] = 0.;
     _eye[1][0] = 0.;
     _eye[1][2] = 0.;
     _eye[2][0] = 0.;
     _eye[2][1] = 0.;

     // Pack constant array
     cuFloatComplex _con[CSIZE] = {make_cuFloatComplex(k, 0.),
                                     make_cuFloatComplex(EPS, 0.),
                                     make_cuFloatComplex(MU_0, 0.),
                                     make_cuFloatComplex(ZETA_0_INV, 0.),
                                     make_cuFloatComplex(M_PIf, 0.),
                                     make_cuFloatComplex(C_L, 0.),
                                     make_cuFloatComplex(t_direction, 0.),
                                     make_cuFloatComplex(0., 1.),
                                     make_cuFloatComplex(0., 0.),
                                     make_cuFloatComplex(4., 0.)};

     // Copy constant array to Device constant memory
     gpuErrchk( cudaMemcpyToSymbol(con, &_con, CSIZE * sizeof(cuFloatComplex)) );
     gpuErrchk( cudaMemcpyToSymbol(eye, &_eye, sizeof(_eye)) );
     gpuErrchk( cudaMemcpyToSymbol(g_s, &gs, sizeof(int)) );
     gpuErrchk( cudaMemcpyToSymbol(g_t, &gt, sizeof(int)) );

     std::array<dim3, 2> BT;
     BT[0] = nrb;
     BT[1] = nrt;

     return BT;
 }

/**
 * Calculate total field at point on target.
 *
 * @param d_xs C-style array containing source points x-coordinate.
 * @param d_ys C-style array containing source points y-coordinate.
 * @param d_zs C-style array containing source points z-coordinate.
 * @param d_Jx C-style array containing source J x-component.
 * @param d_Jy C-style array containing source J y-component.
 * @param d_Jz C-style array containing source J z-component.
 * @param d_Mx C-style array containing source M x-component.
 * @param d_My C-style array containing source M y-component.
 * @param d_Mz C-style array containing source M z-component.
 * @param point C-style array of lenct->sizeh 3 containing xyz coordinates of target point.
 * @param d_A C-style array containing area elements.
 * @param d_ei C-style array of lenct->sizeh 3 to be filled with E-field at point.
 * @param d_hi C-style array of lenct->sizeh 3 to be filled with H-field at point.
 */
__device__ void fieldAtPoint(float *d_xs, float *d_ys, float*d_zs,
                    cuFloatComplex *d_Jx, cuFloatComplex *d_Jy, cuFloatComplex *d_Jz,
                    cuFloatComplex *d_Mx, cuFloatComplex *d_My, cuFloatComplex *d_Mz,
                    float (&point)[3], float *d_A,
                    cuFloatComplex (&d_ei)[3], cuFloatComplex (&d_hi)[3])
{
    // Scalars (float & complex float)
    float r;                           // Distance between source and target points
    float r_inv;                       // 1 / r
    cuFloatComplex omega;                       // Angular frequency of field
    cuFloatComplex Green;         // Container for Green's function
    cuFloatComplex r_in_s;        // Container for inner products between wavevctor and currents
    cuFloatComplex rc;

    // Arrays of floats
    float source_point[3]; // Container for xyz co-ordinates
    float r_vec[3];        // Distance vector between source and target points
    float k_hat[3];        // Unit wavevctor
    float k_arr[3];        // Wavevector

    // Arrays of complex floats
    cuFloatComplex e_field[3] = {con[8], con[8], con[8]}; // Electric field on target
    cuFloatComplex h_field[3] = {con[8], con[8], con[8]}; // Magnetic field on target
    cuFloatComplex js[3];             // Electric current at source point
    cuFloatComplex ms[3];             // Magnetic current at source point
    cuFloatComplex e_vec_thing[3];    // Electric current contribution to e-field
    cuFloatComplex h_vec_thing[3];    // Magnetic current contribution to h-field
    cuFloatComplex k_out_ms[3];       // Outer product between k and ms
    cuFloatComplex k_out_js[3];       // Outer product between k and js
    cuFloatComplex temp[3];           // Temporary container for intermediate values

    //e_field = {con[8], con[8], con[8]};
    //h_field = {con[8], con[8], con[8]};

    omega = cuCmulf(con[5], con[0]); // C_L * k

    for(int i=0; i<g_s; i++)

    {
        js[0] = d_Jx[i];
        js[1] = d_Jy[i];
        js[2] = d_Jz[i];

        //if (i == 2){printf("%.16f\n", d_My[i].x);}
        //printf("hello from fieldatpoint %d\n", i);
        ms[0] = d_Mx[i];
        ms[1] = d_My[i];
        ms[2] = d_Mz[i];

        source_point[0] = d_xs[i];
        source_point[1] = d_ys[i];
        source_point[2] = d_zs[i];

        diff(point, source_point, r_vec);
        abs(r_vec, r);

        rc = make_cuFloatComplex(r, 0.);
        r_inv = 1 / r;

        s_mult(r_vec, r_inv, k_hat);
        s_mult(k_hat, con[0].x, k_arr);

        // e-field
        dot(k_hat, js, r_in_s);
        s_mult(k_hat, r_in_s, temp);
        diff(js, temp, e_vec_thing);

        ext(k_arr, ms, k_out_ms);

        // h-field
        dot(k_hat, ms, r_in_s);
        s_mult(k_hat, r_in_s, temp);
        diff(ms, temp, h_vec_thing);

        ext(k_arr, js, k_out_js);

        cuFloatComplex d_Ac = make_cuFloatComplex(d_A[i], 0.);

        Green = cuCmulf(cuCdivf(expCo(cuCmulf(con[6], cuCmulf(con[7], cuCmulf(con[0], rc)))),
                (cuCmulf(con[9], cuCmulf(con[4], rc)))), cuCmulf(d_Ac, con[7]));

        for( int n=0; n<3; n++)
        {
            e_field[n] = cuCsubf(e_field[n], cuCmulf(cuCsubf(cuCmulf(omega, cuCmulf(con[2], e_vec_thing[n])), k_out_ms[n]), Green));
            h_field[n] = cuCsubf(h_field[n], cuCmulf(cuCaddf(cuCmulf(omega, cuCmulf(con[1], h_vec_thing[n])), k_out_js[n]), Green));

            //if (i == 2){printf("%.16f\n", e_vec_thing[n].x);}
            //if (i == 2){printf("%.16f\n", h_vec_thing[n].x);}

            //if (i == 2){printf("%.16f\n", k_out_ms[n].x);}
            //if (i == 2){printf("%.16f\n", k_out_js[n].x);}
        }
        //if (i == 2){printf("%.16f\n", k_out_js[1].x);}

    }

    d_ei[0] = e_field[0];
    d_ei[1] = e_field[1];
    d_ei[2] = e_field[2];

    d_hi[0] = h_field[0];
    d_hi[1] = h_field[1];
    d_hi[2] = h_field[2];


}

__device__ void farfieldAtPoint(float *d_xs, float *d_ys, float *d_zs,
                                cuFloatComplex *d_Jx, cuFloatComplex *d_Jy, cuFloatComplex *d_Jz,
                                cuFloatComplex *d_Mx, cuFloatComplex *d_My, cuFloatComplex *d_Mz,
                                float (&r_hat)[3], float *d_A, cuFloatComplex (&e)[3])
{
    // Scalars (float & complex float)
    float omega_mu;                       // Angular frequency of field times mu
    float r_hat_in_rp;                 // r_hat dot product r_prime

    // Arrays of floats
    float source_point[3]; // Container for xyz co-ordinates

    // Arrays of complex floats
    cuFloatComplex js[3];      // Build radiation integral
    cuFloatComplex ms[3];      // Build radiation integral

    cuFloatComplex _ctemp[3];
    cuFloatComplex js_tot_factor[3];
    cuFloatComplex ms_tot_factor[3];

    // Matrices
    float rr_dyad[3][3];       // Dyadic product between r_hat - r_hat
    float eye_min_rr[3][3];    // I - rr

    //omega_mu = C_L * k * MU_0;
    omega_mu = con[5].x * con[0].x * con[2].x;
    dyad(r_hat, r_hat, rr_dyad);
    matDiff(eye, rr_dyad, eye_min_rr);

    e[0] = con[8];
    e[1] = con[8];
    e[2] = con[8];

    for(int i=0; i<g_s; i++)
    {
        source_point[0] = d_xs[i];
        source_point[1] = d_ys[i];
        source_point[2] = d_zs[i];

        dot(r_hat, source_point, r_hat_in_rp);

        cuFloatComplex cfact = cuCmulf(con[7], make_cuFloatComplex((con[0].x * r_hat_in_rp), 0.));
        cuFloatComplex expo = expCo(cfact);
        cfact = cuCmulf(expo, make_cuFloatComplex(d_A[i], 0.));

        js[0] = cuCaddf(js[0], cuCmulf(d_Jx[i], cfact));
        js[1] = cuCaddf(js[1], cuCmulf(d_Jy[i], cfact));
        js[2] = cuCaddf(js[2], cuCmulf(d_Jz[i], cfact));

        ms[0] = cuCaddf(ms[0], cuCmulf(d_Mx[i], cfact));
        ms[1] = cuCaddf(ms[1], cuCmulf(d_My[i], cfact));
        ms[2] = cuCaddf(ms[2], cuCmulf(d_Mz[i], cfact));

    }
    matVec(eye_min_rr, js, _ctemp);
    s_mult(_ctemp, omega_mu, js_tot_factor);

    ext(r_hat, ms, _ctemp);
    s_mult(_ctemp, con[0].x, ms_tot_factor);

    for (int n=0; n<3; n++)
    {
        e[n] = cuCsubf(ms_tot_factor[n], js_tot_factor[n]);
    }
}

/**
 * Kernel for toPrint == 0: save J and M.
 *
 * @param d_xs C-style array containing source points x-coordinate.
 * @param d_ys C-style array containing source points y-coordinate.
 * @param d_zs C-style array containing source points z-coordinate.
 * @param d_A C-style array containing area elements.
 * @param d_xt C-style array containing target points x-coordinate.
 * @param d_yt C-style array containing target points y-coordinate.
 * @param d_zt C-style array containing target points z-coordinate.
 * @param d_nxt C-style array containing target norms x-component.
 * @param d_nyt C-style array containing target norms y-component.
 * @param d_nzt C-style array containing target norms z-component.
 * @param d_Jx C-style array containing source J x-component.
 * @param d_Jy C-style array containing source J y-component.
 * @param d_Jz C-style array containing source J z-component.
 * @param d_Mx C-style array containing source M x-component.
 * @param d_My C-style array containing source M y-component.
 * @param d_Mz C-style array containing source M z-component.
 * @param d_Jxt C-style array to be filled with target J x-component.
 * @param d_Jyt C-style array to be filled with target J y-component.
 * @param d_Jzt C-style array to be filled with target J z-component.
 * @param d_Mxt C-style array to be filled with target M x-component.
 * @param d_Myt C-style array to be filled with target M y-component.
 * @param d_Mzt C-style array to be filled with target M z-component.
 */
__global__ void GpropagateBeam_0(float *d_xs, float *d_ys, float *d_zs,
                                float *d_A, float *d_xt, float *d_yt, float *d_zt,
                                float *d_nxt, float *d_nyt, float *d_nzt,
                                cuFloatComplex *d_Jx, cuFloatComplex *d_Jy, cuFloatComplex *d_Jz,
                                cuFloatComplex *d_Mx, cuFloatComplex *d_My, cuFloatComplex *d_Mz,
                                cuFloatComplex *d_Jxt, cuFloatComplex *d_Jyt, cuFloatComplex *d_Jzt,
                                cuFloatComplex *d_Mxt, cuFloatComplex *d_Myt, cuFloatComplex *d_Mzt)
{

    // Scalars (float & complex float)
    cuFloatComplex e_dot_p_r_perp;    // E-field - perpendicular reflected POI polarization vector dot product
    cuFloatComplex e_dot_p_r_parr;    // E-field - parallel reflected POI polarization vector dot product

    // Arrays of floats
    float S_i_norm[3];         // Normalized incoming Poynting vector
    float p_i_perp[3];         // Perpendicular incoming POI polarization vector
    float p_i_parr[3];         // Parallel incoming POI polarization vector
    float S_r_norm[3];         // Normalized reflected Poynting vector
    float p_r_perp[3];         // Perpendicular reflected POI polarization vector
    float p_r_parr[3];         // Parallel reflected POI polarization vector
    float S_out_n[3];          // Container for Poynting-normal ext products
    float point[3];            // Point on target
    float norms[3];            // Normal vector at point
    float e_out_h_r[3];        // Real part of E-field - H-field ext product

    // Arrays of complex floats
    cuFloatComplex e_r[3];            // Reflected E-field
    cuFloatComplex h_r[3];            // Reflected H-field
    cuFloatComplex n_out_e_i_r[3];    // Electric current
    cuFloatComplex temp1[3];          // Temporary container 1 for intermediate irrelevant values
    cuFloatComplex temp2[3];          // Temporary container 2
    cuFloatComplex temp3[3];          // Temporary container 3

    // Return containers
    cuFloatComplex d_ei[3];
    cuFloatComplex d_hi[3];

    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    //printf("%d\n", idx);
    if (idx < g_t)
    {
        point[0] = d_xt[idx];
        point[1] = d_yt[idx];
        point[2] = d_zt[idx];

        norms[0] = d_nxt[idx];
        norms[1] = d_nyt[idx];
        norms[2] = d_nzt[idx];

        // Calculate total incoming E and H field at point on target
        fieldAtPoint(d_xs, d_ys, d_zs,
                    d_Jx, d_Jy, d_Jz,
                    d_Mx, d_My, d_Mz,
                    point, d_A, d_ei, d_hi);

        // Calculate normalised incoming poynting vector.
        conja(d_hi, temp1);                        // h_conj
        ext(d_ei, temp1, temp2);                  // e_out_h

        for (int n=0; n<3; n++)
        {
            e_out_h_r[n] = temp2[n].x;                      // e_out_h_r
        }

        normalize(e_out_h_r, S_i_norm);                       // S_i_norm

        // Calculate incoming polarization vectors
        ext(S_i_norm, norms, S_out_n);                      // S_i_out_n
        normalize(S_out_n, p_i_perp);                       // p_i_perp
        ext(p_i_perp, S_i_norm, p_i_parr);               // p_i_parr

        // Now calculate reflected poynting vector.
        snell(S_i_norm, norms, S_r_norm);                // S_r_norm

        // Calculate normalised reflected polarization vectors
        ext(S_r_norm, norms, S_out_n);                      // S_r_out_n
        normalize(S_out_n, p_r_perp);                       // p_r_perp
        ext(S_r_norm, p_r_perp, p_r_parr);               // p_r_parr

        // Now, calculate reflected field from target
        dot(d_ei, p_r_perp, e_dot_p_r_perp);      // e_dot_p_r_perp
        dot(d_ei, p_r_parr, e_dot_p_r_parr);      // e_dot_p_r_parr

        // Calculate reflected field from reflection matrix
        for(int n=0; n<3; n++)
        {
            e_r[n] = cuCsubf(cuCmulf(e_dot_p_r_perp, make_cuFloatComplex(-p_i_perp[n], 0.)), cuCmulf(e_dot_p_r_parr, make_cuFloatComplex(p_i_parr[n], 0.)));
        }

        ext(S_r_norm, e_r, temp1);                       // h_r_temp
        s_mult(temp1, con[3], h_r);                     // ZETA_0_INV, h_r

        //Calculate and store J and M only
        for(int n=0; n<3; n++)
        {
            temp1[n] = cuCaddf(e_r[n], d_ei[n]); // e_i_r
            temp2[n] = cuCaddf(h_r[n], d_hi[n]); // h_i_r
        }

        ext(norms, temp2, temp3);

        d_Jxt[idx] = temp3[0];
        d_Jyt[idx] = temp3[1];
        d_Jzt[idx] = temp3[2];

        ext(norms, temp1, n_out_e_i_r);
        s_mult(n_out_e_i_r, -1., temp3);

        d_Mxt[idx] = temp3[0];
        d_Myt[idx] = temp3[1];
        d_Mzt[idx] = temp3[2];
    }
}

/**
 * Kernel for toPrint == 1: save Ei and Hi.
 *
 * @param d_xs C-style array containing source points x-coordinate.
 * @param d_ys C-style array containing source points y-coordinate.
 * @param d_zs C-style array containing source points z-coordinate.
 * @param d_A C-style array containing area elements.
 * @param d_xt C-style array containing target points x-coordinate.
 * @param d_yt C-style array containing target points y-coordinate.
 * @param d_zt C-style array containing target points z-coordinate.
 * @param d_Jx C-style array containing source J x-component.
 * @param d_Jy C-style array containing source J y-component.
 * @param d_Jz C-style array containing source J z-component.
 * @param d_Mx C-style array containing source M x-component.
 * @param d_My C-style array containing source M y-component.
 * @param d_Mz C-style array containing source M z-component.
 * @param d_Ext C-style array to be filled with target Ei x-component.
 * @param d_Eyt C-style array to be filled with target Ei y-component.
 * @param d_Ezt C-style array to be filled with target Ei z-component.
 * @param d_Hxt C-style array to be filled with target Hi x-component.
 * @param d_Hyt C-style array to be filled with target Hi y-component.
 * @param d_Hzt C-style array to be filled with target Hi z-component.
 */
__global__ void GpropagateBeam_1(float *d_xs, float *d_ys, float *d_zs,
                                float *d_A, float *d_xt, float *d_yt, float *d_zt,
                                cuFloatComplex *d_Jx, cuFloatComplex *d_Jy, cuFloatComplex *d_Jz,
                                cuFloatComplex *d_Mx, cuFloatComplex *d_My, cuFloatComplex *d_Mz,
                                cuFloatComplex *d_Ext, cuFloatComplex *d_Eyt, cuFloatComplex *d_Ezt,
                                cuFloatComplex *d_Hxt, cuFloatComplex *d_Hyt, cuFloatComplex *d_Hzt)
{
    // Arrays of floats
    float point[3];            // Point on target

    // Return containers for call to fieldAtPoint
    cuFloatComplex d_ei[3];
    cuFloatComplex d_hi[3];

    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    if (idx < g_t)
    {
        point[0] = d_xt[idx];
        point[1] = d_yt[idx];
        point[2] = d_zt[idx];

        // Calculate total incoming E and H field at point on target
        fieldAtPoint(d_xs, d_ys, d_zs,
                    d_Jx, d_Jy, d_Jz,
                    d_Mx, d_My, d_Mz,
                    point, d_A, d_ei, d_hi);

        d_Ext[idx] = d_ei[0];
        d_Eyt[idx] = d_ei[1];
        d_Ezt[idx] = d_ei[2];

        //printf("%f\n", d_ei[0].x);

        d_Hxt[idx] = d_hi[0];
        d_Hyt[idx] = d_hi[1];
        d_Hzt[idx] = d_hi[2];
    }
}

/**
 * Kernel for toPrint == 2: save J, M, Ei and Hi.
 *
 * @param d_xs C-style array containing source points x-coordinate.
 * @param d_ys C-style array containing source points y-coordinate.
 * @param d_zs C-style array containing source points z-coordinate.
 * @param d_A C-style array containing area elements.
 * @param d_xt C-style array containing target points x-coordinate.
 * @param d_yt C-style array containing target points y-coordinate.
 * @param d_zt C-style array containing target points z-coordinate.
 * @param d_nxt C-style array containing target norms x-component.
 * @param d_nyt C-style array containing target norms y-component.
 * @param d_nzt C-style array containing target norms z-component.
 * @param d_Jx C-style array containing source J x-component.
 * @param d_Jy C-style array containing source J y-component.
 * @param d_Jz C-style array containing source J z-component.
 * @param d_Mx C-style array containing source M x-component.
 * @param d_My C-style array containing source M y-component.
 * @param d_Mz C-style array containing source M z-component.
 * @param d_Jxt C-style array to be filled with target J x-component.
 * @param d_Jyt C-style array to be filled with target J y-component.
 * @param d_Jzt C-style array to be filled with target J z-component.
 * @param d_Mxt C-style array to be filled with target M x-component.
 * @param d_Myt C-style array to be filled with target M y-component.
 * @param d_Mzt C-style array to be filled with target M z-component.
 * @param d_Ext C-style array to be filled with target Ei x-component.
 * @param d_Eyt C-style array to be filled with target Ei y-component.
 * @param d_Ezt C-style array to be filled with target Ei z-component.
 * @param d_Hxt C-style array to be filled with target Hi x-component.
 * @param d_Hyt C-style array to be filled with target Hi y-component.
 * @param d_Hzt C-style array to be filled with target Hi z-component.
 */
__global__ void GpropagateBeam_2(float *d_xs, float *d_ys, float *d_zs,
                                float *d_A, float *d_xt, float *d_yt, float *d_zt,
                                float *d_nxt, float *d_nyt, float *d_nzt,
                                cuFloatComplex *d_Jx, cuFloatComplex *d_Jy, cuFloatComplex *d_Jz,
                                cuFloatComplex *d_Mx, cuFloatComplex *d_My, cuFloatComplex *d_Mz,
                                cuFloatComplex *d_Jxt, cuFloatComplex *d_Jyt, cuFloatComplex *d_Jzt,
                                cuFloatComplex *d_Mxt, cuFloatComplex *d_Myt, cuFloatComplex *d_Mzt,
                                cuFloatComplex *d_Ext, cuFloatComplex *d_Eyt, cuFloatComplex *d_Ezt,
                                cuFloatComplex *d_Hxt, cuFloatComplex *d_Hyt, cuFloatComplex *d_Hzt)
{

    // Scalars (float & complex float)
    cuFloatComplex e_dot_p_r_perp;    // E-field - perpendicular reflected POI polarization vector dot product
    cuFloatComplex e_dot_p_r_parr;    // E-field - parallel reflected POI polarization vector dot product

    // Arrays of floats
    float S_i_norm[3];         // Normalized incoming Poynting vector
    float p_i_perp[3];         // Perpendicular incoming POI polarization vector
    float p_i_parr[3];         // Parallel incoming POI polarization vector
    float S_r_norm[3];         // Normalized reflected Poynting vector
    float p_r_perp[3];         // Perpendicular reflected POI polarization vector
    float p_r_parr[3];         // Parallel reflected POI polarization vector
    float S_out_n[3];          // Container for Poynting-normal ext products
    float point[3];            // Point on target
    float norms[3];            // Normal vector at point
    float e_out_h_r[3];        // Real part of E-field - H-field ext product

    // Arrays of complex floats
    cuFloatComplex e_r[3];            // Reflected E-field
    cuFloatComplex h_r[3];            // Reflected H-field
    cuFloatComplex n_out_e_i_r[3];    // Electric current
    cuFloatComplex temp1[3];          // Temporary container 1 for intermediate irrelevant values
    cuFloatComplex temp2[3];          // Temporary container 2
    cuFloatComplex temp3[3];          // Temporary container 3

    // Return containers
    cuFloatComplex d_ei[3];
    cuFloatComplex d_hi[3];

    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    if (idx < g_t)
    {
        point[0] = d_xt[idx];
        point[1] = d_yt[idx];
        point[2] = d_zt[idx];

        norms[0] = d_nxt[idx];
        norms[1] = d_nyt[idx];
        norms[2] = d_nzt[idx];

        // Calculate total incoming E and H field at point on target
        fieldAtPoint(d_xs, d_ys, d_zs,
                    d_Jx, d_Jy, d_Jz,
                    d_Mx, d_My, d_Mz,
                    point, d_A, d_ei, d_hi);

        d_Ext[idx] = d_ei[0];
        d_Eyt[idx] = d_ei[1];
        d_Ezt[idx] = d_ei[2];

        d_Hxt[idx] = d_hi[0];
        d_Hyt[idx] = d_hi[1];
        d_Hzt[idx] = d_hi[2];

        // Calculate normalised incoming poynting vector.
        conja(d_hi, temp1);                        // h_conj
        ext(d_ei, temp1, temp2);                  // e_out_h

        for (int n=0; n<3; n++)
        {
            e_out_h_r[n] = temp2[n].x;                      // e_out_h_r
        }

        normalize(e_out_h_r, S_i_norm);                       // S_i_norm

        // Calculate incoming polarization vectors
        ext(S_i_norm, norms, S_out_n);                      // S_i_out_n
        normalize(S_out_n, p_i_perp);                       // p_i_perp
        ext(p_i_perp, S_i_norm, p_i_parr);               // p_i_parr

        // Now calculate reflected poynting vector.
        snell(S_i_norm, norms, S_r_norm);                // S_r_norm

        // Calculate normalised reflected polarization vectors
        ext(S_r_norm, norms, S_out_n);                      // S_r_out_n
        normalize(S_out_n, p_r_perp);                       // p_r_perp
        ext(S_r_norm, p_r_perp, p_r_parr);               // p_r_parr

        // Now, calculate reflected field from target
        dot(d_ei, p_r_perp, e_dot_p_r_perp);      // e_dot_p_r_perp
        dot(d_ei, p_r_parr, e_dot_p_r_parr);      // e_dot_p_r_parr

        // Calculate reflected field from reflection matrix
        for(int n=0; n<3; n++)
        {
            e_r[n] = cuCsubf(cuCmulf(e_dot_p_r_perp, make_cuFloatComplex(-p_i_perp[n], 0.)), cuCmulf(e_dot_p_r_parr, make_cuFloatComplex(p_i_parr[n], 0.)));
        }

        ext(S_r_norm, e_r, temp1);                       // h_r_temp
        s_mult(temp1, con[3], h_r);                     // ZETA_0_INV, h_r

        //Calculate and store J and M only
        for(int n=0; n<3; n++)
        {
            temp1[n] = cuCaddf(e_r[n], d_ei[n]); // e_i_r
            temp2[n] = cuCaddf(h_r[n], d_hi[n]); // h_i_r
        }

        ext(norms, temp2, temp3);

        d_Jxt[idx] = temp3[0];
        d_Jyt[idx] = temp3[1];
        d_Jzt[idx] = temp3[2];

        ext(norms, temp1, n_out_e_i_r);
        s_mult(n_out_e_i_r, -1., temp3);

        d_Mxt[idx] = temp3[0];
        d_Myt[idx] = temp3[1];
        d_Mzt[idx] = temp3[2];
    }
}

/**
 * Kernel for toPrint == 3: save Pr, Er and Hr.
 *
 * @param d_xs C-style array containing source points x-coordinate.
 * @param d_ys C-style array containing source points y-coordinate.
 * @param d_zs C-style array containing source points z-coordinate.
 * @param d_A C-style array containing area elements.
 * @param d_xt C-style array containing target points x-coordinate.
 * @param d_yt C-style array containing target points y-coordinate.
 * @param d_zt C-style array containing target points z-coordinate.
 * @param d_nxt C-style array containing target norms x-component.
 * @param d_nyt C-style array containing target norms y-component.
 * @param d_nzt C-style array containing target norms z-component.
 * @param d_Jx C-style array containing source J x-component.
 * @param d_Jy C-style array containing source J y-component.
 * @param d_Jz C-style array containing source J z-component.
 * @param d_Mx C-style array containing source M x-component.
 * @param d_My C-style array containing source M y-component.
 * @param d_Mz C-style array containing source M z-component.
 * @param d_Prxt C-style array to be filled with Pr x-component.
 * @param d_Pryt C-style array to be filled with Pr y-component.
 * @param d_Przt C-style array to be filled with Pr z-component.
 * @param d_Ext C-style array to be filled with target Er x-component.
 * @param d_Eyt C-style array to be filled with target Er y-component.
 * @param d_Ezt C-style array to be filled with target Er z-component.
 * @param d_Hxt C-style array to be filled with target Hr x-component.
 * @param d_Hyt C-style array to be filled with target Hr y-component.
 * @param d_Hzt C-style array to be filled with target Hr z-component.
 */
__global__ void GpropagateBeam_3(float *d_xs, float *d_ys, float *d_zs,
                                float *d_A, float *d_xt, float *d_yt, float *d_zt,
                                float *d_nxt, float *d_nyt, float *d_nzt,
                                cuFloatComplex *d_Jx, cuFloatComplex *d_Jy, cuFloatComplex *d_Jz,
                                cuFloatComplex *d_Mx, cuFloatComplex *d_My, cuFloatComplex *d_Mz,
                                cuFloatComplex *d_Ext, cuFloatComplex *d_Eyt, cuFloatComplex *d_Ezt,
                                cuFloatComplex *d_Hxt, cuFloatComplex *d_Hyt, cuFloatComplex *d_Hzt,
                                float *d_Prxt, float *d_Pryt, float *d_Przt)
{

    // Scalars (float & complex float)
    cuFloatComplex e_dot_p_r_perp;    // E-field - perpendicular reflected POI polarization vector dot product
    cuFloatComplex e_dot_p_r_parr;    // E-field - parallel reflected POI polarization vector dot product

    // Arrays of floats
    float S_i_norm[3];         // Normalized incoming Poynting vector
    float p_i_perp[3];         // Perpendicular incoming POI polarization vector
    float p_i_parr[3];         // Parallel incoming POI polarization vector
    float S_r_norm[3];         // Normalized reflected Poynting vector
    float p_r_perp[3];         // Perpendicular reflected POI polarization vector
    float p_r_parr[3];         // Parallel reflected POI polarization vector
    float S_out_n[3];          // Container for Poynting-normal ext products
    float point[3];            // Point on target
    float norms[3];            // Normal vector at point
    float e_out_h_r[3];        // Real part of E-field - H-field ext product

    // Arrays of complex floats
    cuFloatComplex e_r[3];            // Reflected E-field
    cuFloatComplex h_r[3];            // Reflected H-field
    cuFloatComplex temp1[3];          // Temporary container 1 for intermediate irrelevant values
    cuFloatComplex temp2[3];          // Temporary container 2

    // Return containers
    cuFloatComplex d_ei[3];
    cuFloatComplex d_hi[3];

    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    if (idx < g_t)
    {
        point[0] = d_xt[idx];
        point[1] = d_yt[idx];
        point[2] = d_zt[idx];

        norms[0] = d_nxt[idx];
        norms[1] = d_nyt[idx];
        norms[2] = d_nzt[idx];

        // Calculate total incoming E and H field at point on target
        fieldAtPoint(d_xs, d_ys, d_zs,
                    d_Jx, d_Jy, d_Jz,
                    d_Mx, d_My, d_Mz,
                    point, d_A, d_ei, d_hi);

        // Calculate normalised incoming poynting vector.
        conja(d_hi, temp1);                        // h_conj
        ext(d_ei, temp1, temp2);                  // e_out_h

        for (int n=0; n<3; n++)
        {
            e_out_h_r[n] = temp2[n].x;                      // e_out_h_r
        }

        normalize(e_out_h_r, S_i_norm);                       // S_i_norm

        // Calculate incoming polarization vectors
        ext(S_i_norm, norms, S_out_n);                      // S_i_out_n
        normalize(S_out_n, p_i_perp);                       // p_i_perp
        ext(p_i_perp, S_i_norm, p_i_parr);               // p_i_parr

        // Now calculate reflected poynting vector.
        snell(S_i_norm, norms, S_r_norm);                // S_r_norm

        // Store REFLECTED Pynting vectors
        d_Prxt[idx] = S_r_norm[0];
        d_Pryt[idx] = S_r_norm[1];
        d_Przt[idx] = S_r_norm[2];

        // Calculate normalised reflected polarization vectors
        ext(S_r_norm, norms, S_out_n);                      // S_r_out_n
        normalize(S_out_n, p_r_perp);                       // p_r_perp
        ext(S_r_norm, p_r_perp, p_r_parr);               // p_r_parr

        // Now, calculate reflected field from target
        dot(d_ei, p_r_perp, e_dot_p_r_perp);      // e_dot_p_r_perp
        dot(d_ei, p_r_parr, e_dot_p_r_parr);      // e_dot_p_r_parr

        // Calculate reflected field from reflection matrix
        for(int n=0; n<3; n++)
        {
            e_r[n] = cuCsubf(cuCmulf(e_dot_p_r_perp, make_cuFloatComplex(-p_i_perp[n], 0.)), cuCmulf(e_dot_p_r_parr, make_cuFloatComplex(p_i_parr[n], 0.)));
        }

        ext(S_r_norm, e_r, temp1);                       // h_r_temp
        s_mult(temp1, con[3], h_r);                     // ZETA_0_INV, h_r

        // Store REFLECTED fields
        d_Ext[idx] = e_r[0];
        d_Eyt[idx] = e_r[1];
        d_Ezt[idx] = e_r[2];

        d_Hxt[idx] = h_r[0];
        d_Hyt[idx] = h_r[1];
        d_Hzt[idx] = h_r[2];
    }
}
/*
 * Kernel 4, propagation to far field
 * */
void __global__ GpropagateBeam_4(float *d_xs, float *d_ys, float *d_zs,
                                float *d_A, float *d_xt, float *d_yt,
                                cuFloatComplex *d_Jx, cuFloatComplex *d_Jy, cuFloatComplex *d_Jz,
                                cuFloatComplex *d_Mx, cuFloatComplex *d_My, cuFloatComplex *d_Mz,
                                cuFloatComplex *d_Ext, cuFloatComplex *d_Eyt, cuFloatComplex *d_Ezt)
{
    // Scalars (float & complex float)
    float theta;
    float phi;

    // Arrays of floats
    float r_hat[3];                // Unit vector in far-field point direction

    // Arrays of complex floats
    cuFloatComplex e[3];            // far-field E-field

    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    if (idx < g_t)
    {
        theta   = d_xt[idx];
        phi     = d_yt[idx];

        r_hat[0] = cos(theta) * sin(phi);
        r_hat[1] = sin(theta) * sin(phi);
        r_hat[2] = cos(phi);

        // Calculate total incoming E field at point on far-field
        farfieldAtPoint(d_xs, d_ys, d_zs, d_Jx, d_Jy, d_Jz, d_Mx, d_My, d_Mz, r_hat, d_A, e);

        d_Ext[idx] = e[0];
        d_Eyt[idx] = e[1];
        d_Ezt[idx] = e[2];
    }
}

/**
 * (PRIVATE)
 * Allocate and copy grid in R1 from Host to Device.
 *
 * @param d_x Pointer for array on device.
 * @param h_x Pointer for array on host.
 * @param size Number of elements of h_x/d_x.
 */
__host__ void _allocateGridR1ToGPU(float *d_x, float *h_x, int size)
{
    gpuErrchk( cudaMalloc((void**)&d_x, size * sizeof(float)) );
    gpuErrchk( cudaMemcpy(d_x, h_x, size * sizeof(float), cudaMemcpyHostToDevice) );
}

/**
 * (PRIVATE)
 * Allocate R3 to Device.
 *
 * @param d_x Pointer for x array on device.
 * @param d_y Pointer for y array on device.
 * @param d_z Pointer for z array on device.
 * */
__host__ void _allocateGridR3(float *d_x, float *d_y, float *d_z, int size)
{
    gpuErrchk( cudaMalloc((void**)&d_x, size * sizeof(float)) );
    gpuErrchk( cudaMalloc((void**)&d_y, size * sizeof(float)) );
    gpuErrchk( cudaMalloc((void**)&d_z, size * sizeof(float)) );
}

__host__ void _allocateGridR2ToGPU(float *d_x, float *d_y,
                                  float *h_x, float *h_y,
                                    int size)
{
    gpuErrchk( cudaMalloc((void**)&d_x, size * sizeof(float)) );
    gpuErrchk( cudaMalloc((void**)&d_y, size * sizeof(float)) );

    gpuErrchk( cudaMemcpy(d_x, h_x, size * sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_y, h_y, size * sizeof(float), cudaMemcpyHostToDevice) );
}

/**
 * (PRIVATE)
 * Allocate and copy grid in R3 from Host to Device.
 *
 * @param d_x Pointer for x array on device.
 * @param d_y Pointer for y array on device.
 * @param d_z Pointer for z array on device.
 * @param h_x Pointer for x array on host.
 * @param h_y Pointer for y array on host.
 * @param h_z Pointer for z array on host.
 * @param size Number of elements of h_x/d_x.
 */
__host__ void _allocateGridR3ToGPU(float *d_x, float *d_y, float *d_z,
                                  float *h_x, float *h_y, float *h_z,
                                    int size)
{
    gpuErrchk( cudaMalloc((void**)&d_x, size * sizeof(float)) );
    gpuErrchk( cudaMalloc((void**)&d_y, size * sizeof(float)) );
    gpuErrchk( cudaMalloc((void**)&d_z, size * sizeof(float)) );

    gpuErrchk( cudaMemcpy(d_x, h_x, size * sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_y, h_y, size * sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_z, h_z, size * sizeof(float), cudaMemcpyHostToDevice) );
}

__host__ void _allocateGridGPUToR3(float *h_x, float *h_y, float *h_z,
                                   float *d_x, float *d_y, float *d_z,
                                   int size)
{
    gpuErrchk( cudaMemcpy(h_x, d_x, size * sizeof(float), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(h_y, d_y, size * sizeof(float), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(h_z, d_z, size * sizeof(float), cudaMemcpyDeviceToHost) );
}

/**
 * (PRIVATE)
 * Allocate C3 to Device.
 *
 * @param d_x Pointer for x array on device.
 * @param d_y Pointer for y array on device.
 * @param d_z Pointer for z array on device.
 * */
__host__ void _allocateGridC3(cuFloatComplex *d_x, cuFloatComplex *d_y, cuFloatComplex *d_z, int size)
{
    gpuErrchk( cudaMalloc((void**)&d_x, size * sizeof(cuFloatComplex)) );
    gpuErrchk( cudaMalloc((void**)&d_y, size * sizeof(cuFloatComplex)) );
    gpuErrchk( cudaMalloc((void**)&d_z, size * sizeof(cuFloatComplex)) );
}

/**
 * (PRIVATE)
 * Allocate and copy grid in C3 from Host to Device.
 *
 * @param d_x Pointer for x array on device.
 * @param d_y Pointer for y array on device.
 * @param d_z Pointer for z array on device.
 * @param h_x Pointer for x array on host.
 * @param h_y Pointer for y array on host.
 * @param h_z Pointer for z array on host.
 * @param size Number of elements of h_x/d_x.
 */
__host__ void _allocateGridC3ToGPU(cuFloatComplex *d_x, cuFloatComplex *d_y, cuFloatComplex *d_z,
                                  cuFloatComplex *h_x, cuFloatComplex *h_y, cuFloatComplex *h_z,
                                    int size)
{
    gpuErrchk( cudaMalloc((void**)&d_x, size * sizeof(cuFloatComplex)) );
    gpuErrchk( cudaMalloc((void**)&d_y, size * sizeof(cuFloatComplex)) );
    gpuErrchk( cudaMalloc((void**)&d_z, size * sizeof(cuFloatComplex)) );

    gpuErrchk( cudaMemcpy(d_x, h_x, size * sizeof(cuFloatComplex), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_y, h_y, size * sizeof(cuFloatComplex), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_y, h_z, size * sizeof(cuFloatComplex), cudaMemcpyHostToDevice) );
}

__host__ void _allocateGridGPUToC3(cuFloatComplex *h_x, cuFloatComplex *h_y, cuFloatComplex *h_z,
                                  cuFloatComplex *d_x, cuFloatComplex *d_y, cuFloatComplex *d_z,
                                    int size)
{
    gpuErrchk( cudaMemcpy(h_x, d_x, size * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(h_y, d_y, size * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(h_y, d_z, size * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost) );
}

/**
 * (PRIVATE)
 * Convert two arrays of floats to array of cuComplex
 *
 * @param rarr Real part of complex array.
 * @param iarr Imaginary part of complex array.
 * @param carr Array of cuFloatComplex, to be filled.
 * @param size Size of arrays.
 */
__host__ void _arrToCUDAC(float *rarr, float *iarr, cuFloatComplex* carr, int size)
{
    for (int i=0; i<size; i++)
    {
        carr[i] = make_cuFloatComplex(rarr[i], iarr[i]);
    }
}

__host__ void _arrC3ToCUDAC(float *r1arr, float *r2arr, float *r3arr,
                            float *i1arr, float *i2arr, float *i3arr,
                            cuFloatComplex* c1arr, cuFloatComplex* c2arr, cuFloatComplex* c3arr,
                            int size)
{
    for (int i=0; i<size; i++)
    {
        c1arr[i] = make_cuFloatComplex(r1arr[i], i1arr[i]);
        c2arr[i] = make_cuFloatComplex(r2arr[i], i2arr[i]);
        c3arr[i] = make_cuFloatComplex(r3arr[i], i3arr[i]);

    }
}

/**
 * (PRIVATE)
 * Convert array of cuComplex to two arrays of floats.
 *
 * @param carr Array of cuFloatComplex.
 * @param rarr Real part of complex array, to be filled.
 * @param iarr Imaginary part of complex array, to be filled.
 * @param size Size of arrays.
 */
__host__ void _arrCUDACToC(cuFloatComplex* carr, float *rarr, float *iarr, int size)
{
    for (int i=0; i<size; i++)
    {
        rarr[i] = carr[i].x;
        iarr[i] = carr[i].x;
    }
}

__host__ void _arrCUDACToC3(cuFloatComplex* c1arr, cuFloatComplex* c2arr, cuFloatComplex* c3arr,
                           float *r1arr, float *r2arr, float *r3arr,
                           float *i1arr, float *i2arr, float *i3arr,
                           int size)
{
    for (int i=0; i<size; i++)
    {
        r1arr[i] = c1arr[i].x;
        i1arr[i] = c1arr[i].y;

        r2arr[i] = c2arr[i].x;
        i2arr[i] = c2arr[i].y;

        r3arr[i] = c3arr[i].x;
        i3arr[i] = c3arr[i].y;
    }
}

/**
 * (PUBLIC)
 * Wrapper for calling kernel 0.
 *
 * @param d_x Pointer for x array on device.
 * @param d_y Pointer for y array on device.
 * @param d_z Pointer for z array on device.
 * @param h_x Pointer for x array on host.
 * @param h_y Pointer for y array on host.
 * @param h_z Pointer for z array on host.
 * @param size Number of elements of h_x/d_x.
 */
extern "C" void callKernelf_JM(c2Bundlef *res, reflparamsf source, reflparamsf target,
                                reflcontainerf *cs, reflcontainerf *ct,
                                c2Bundlef *currents,
                                float k, float epsilon,
                                float t_direction, int nBlocks, int nThreads)
{
    // Generate source and target grids
    generateGridf(source, cs);
    generateGridf(target, ct);

    // Initialize and copy constant memory to device
    std::array<dim3, 2> BT;
    BT = _initCUDA(k, epsilon, ct->size, cs->size, t_direction, nBlocks, nThreads);

    // Create pointers to device arrays and allocate/copy source grid and area.
    float *d_xs, *d_ys, *d_zs, *d_A;

    gpuErrchk( cudaMalloc((void**)&d_xs, cs->size * sizeof(float)) );
    gpuErrchk( cudaMalloc((void**)&d_ys, cs->size * sizeof(float)) );
    gpuErrchk( cudaMalloc((void**)&d_zs, cs->size * sizeof(float)) );
    gpuErrchk( cudaMalloc((void**)&d_A, cs->size * sizeof(float)) );

    gpuErrchk( cudaMemcpy(d_xs, cs->x, cs->size * sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_ys, cs->y, cs->size * sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_zs, cs->z, cs->size * sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_A, cs->area, cs->size * sizeof(float), cudaMemcpyHostToDevice) );

    // Create pointers to target grid and normal vectors
    float *d_xt, *d_yt, *d_zt, *d_nxt, *d_nyt, *d_nzt;

    // Allocate target co-ordinate and normal grids
    gpuErrchk( cudaMalloc((void**)&d_xt, ct->size * sizeof(float)) );
    gpuErrchk( cudaMalloc((void**)&d_yt, ct->size * sizeof(float)) );
    gpuErrchk( cudaMalloc((void**)&d_zt, ct->size * sizeof(float)) );

    gpuErrchk( cudaMemcpy(d_xt, ct->x, ct->size * sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_yt, ct->y, ct->size * sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_zt, ct->z, ct->size * sizeof(float), cudaMemcpyHostToDevice) );

    gpuErrchk( cudaMalloc((void**)&d_nxt, ct->size * sizeof(float)) );
    gpuErrchk( cudaMalloc((void**)&d_nyt, ct->size * sizeof(float)) );
    gpuErrchk( cudaMalloc((void**)&d_nzt, ct->size * sizeof(float)) );

    gpuErrchk( cudaMemcpy(d_nxt, ct->nx, ct->size * sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_nyt, ct->ny, ct->size * sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_nzt, ct->nz, ct->size * sizeof(float), cudaMemcpyHostToDevice) );

    cuFloatComplex *h_Jx = new cuFloatComplex[cs->size];
    cuFloatComplex *h_Jy = new cuFloatComplex[cs->size];
    cuFloatComplex *h_Jz = new cuFloatComplex[cs->size];

    cuFloatComplex *h_Mx = new cuFloatComplex[cs->size];
    cuFloatComplex *h_My = new cuFloatComplex[cs->size];
    cuFloatComplex *h_Mz = new cuFloatComplex[cs->size];

    cuFloatComplex *d_Jx, *d_Jy, *d_Jz;
    cuFloatComplex *d_Mx, *d_My, *d_Mz;

    _arrC3ToCUDAC(currents->r1x, currents->r1y, currents->r1z,
                  currents->i1x, currents->i1y, currents->i1z,
                  h_Jx, h_Jy, h_Jz, cs->size);

    _arrC3ToCUDAC(currents->r2x, currents->r2y, currents->r2z,
                  currents->i2x, currents->i2y, currents->i2z,
                  h_Mx, h_My, h_Mz, cs->size);

    // Allocate and copy J and M currents
    gpuErrchk( cudaMalloc((void**)&d_Jx, cs->size * sizeof(cuFloatComplex)) );
    gpuErrchk( cudaMalloc((void**)&d_Jy, cs->size * sizeof(cuFloatComplex)) );
    gpuErrchk( cudaMalloc((void**)&d_Jz, cs->size * sizeof(cuFloatComplex)) );

    gpuErrchk( cudaMemcpy(d_Jx, h_Jx, cs->size * sizeof(cuFloatComplex), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_Jy, h_Jy, cs->size * sizeof(cuFloatComplex), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_Jz, h_Jz, cs->size * sizeof(cuFloatComplex), cudaMemcpyHostToDevice) );

    gpuErrchk( cudaMalloc((void**)&d_Mx, cs->size * sizeof(cuFloatComplex)) );
    gpuErrchk( cudaMalloc((void**)&d_My, cs->size * sizeof(cuFloatComplex)) );
    gpuErrchk( cudaMalloc((void**)&d_Mz, cs->size * sizeof(cuFloatComplex)) );

    gpuErrchk( cudaMemcpy(d_Mx, h_Mx, cs->size * sizeof(cuFloatComplex), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_My, h_My, cs->size * sizeof(cuFloatComplex), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_Mz, h_Mz, cs->size * sizeof(cuFloatComplex), cudaMemcpyHostToDevice) );

    // Delete host arrays for source currents - not needed anymore
    delete h_Jx;
    delete h_Jy;
    delete h_Jz;

    delete h_Mx;
    delete h_My;
    delete h_Mz;

    cuFloatComplex *d_Jxt, *d_Jyt, *d_Jzt;
    cuFloatComplex *d_Mxt, *d_Myt, *d_Mzt;

    gpuErrchk( cudaMalloc((void**)&d_Jxt, ct->size * sizeof(cuFloatComplex)) );
    gpuErrchk( cudaMalloc((void**)&d_Jyt, ct->size * sizeof(cuFloatComplex)) );
    gpuErrchk( cudaMalloc((void**)&d_Jzt, ct->size * sizeof(cuFloatComplex)) );

    gpuErrchk( cudaMalloc((void**)&d_Mxt, ct->size * sizeof(cuFloatComplex)) );
    gpuErrchk( cudaMalloc((void**)&d_Myt, ct->size * sizeof(cuFloatComplex)) );
    gpuErrchk( cudaMalloc((void**)&d_Mzt, ct->size * sizeof(cuFloatComplex)) );

    // Create stopping event for kernel
    cudaEvent_t event;
    gpuErrchk( cudaEventCreateWithFlags(&event, cudaEventDisableTiming) );

    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;

    // Call to KERNEL 0
    printf("Calculating J and M...\n");
    begin = std::chrono::steady_clock::now();
    GpropagateBeam_0<<<BT[0], BT[1]>>>(d_xs, d_ys, d_zs,
                                d_A, d_xt, d_yt, d_zt,
                                d_nxt, d_nyt, d_nzt,
                                d_Jx, d_Jy, d_Jz,
                                d_Mx, d_My, d_Mz,
                                d_Jxt, d_Jyt, d_Jzt,
                                d_Mxt, d_Myt, d_Mzt);
    /*
    while(cudaEventQuery(event) != cudaSuccess)
    {
        //printf("nothing to see\n");
        //std::this_thread::sleep_for(std::chrono::microseconds(wsleep));
    }

    printf("finished\n");
    */

    gpuErrchk( cudaDeviceSynchronize() );

    end = std::chrono::steady_clock::now();

    std::cout << "Elapsed time : "
            << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count()
            << " [s]" << std::endl;

    // Allocate Host arrays for J and M
    cuFloatComplex *h_Jxt = new cuFloatComplex[ct->size];
    cuFloatComplex *h_Jyt = new cuFloatComplex[ct->size];
    cuFloatComplex *h_Jzt = new cuFloatComplex[ct->size];

    cuFloatComplex *h_Mxt = new cuFloatComplex[ct->size];
    cuFloatComplex *h_Myt = new cuFloatComplex[ct->size];
    cuFloatComplex *h_Mzt = new cuFloatComplex[ct->size];

    // Copy data back from Device to Host
    gpuErrchk( cudaMemcpy(h_Jxt, d_Jxt, ct->size * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(h_Jyt, d_Jyt, ct->size * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(h_Jzt, d_Jzt, ct->size * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost) );

    gpuErrchk( cudaMemcpy(h_Mxt, d_Mxt, ct->size * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(h_Myt, d_Myt, ct->size * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(h_Mzt, d_Mzt, ct->size * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost) );

    // Free Device memory
    gpuErrchk( cudaDeviceReset() );

    _arrCUDACToC3(h_Jxt, h_Jyt, h_Jzt, res->r1x, res->r1y, res->r1z, res->i1x, res->i1y, res->i1z, ct->size);
    _arrCUDACToC3(h_Mxt, h_Myt, h_Mzt, res->r2x, res->r2y, res->r2z, res->i2x, res->i2y, res->i2z, ct->size);

    // Delete host arrays for target currents
    delete h_Jxt;
    delete h_Jyt;
    delete h_Jzt;

    delete h_Mxt;
    delete h_Myt;
    delete h_Mzt;
}

/**
 * (PUBLIC)
 * Wrapper for calling kernel 1.
 *
 * @param d_x Pointer for x array on device.
 * @param d_y Pointer for y array on device.
 * @param d_z Pointer for z array on device.
 * @param h_x Pointer for x array on host.
 * @param h_y Pointer for y array on host.
 * @param h_z Pointer for z array on host.
 * @param size Number of elements of h_x/d_x.
 */
extern "C" void callKernelf_EH(c2Bundlef *res, reflparamsf source, reflparamsf target,
                                reflcontainerf *cs, reflcontainerf *ct,
                                c2Bundlef *currents,
                                float k, float epsilon,
                                float t_direction, int nBlocks, int nThreads)
{
    // Generate source and target grids
    generateGridf(source, cs);
    generateGridf(target, ct);

    // Initialize and copy constant memory to device
    std::array<dim3, 2> BT;
    BT = _initCUDA(k, epsilon, ct->size, cs->size, t_direction, nBlocks, nThreads);
    //alsd
    // Create pointers to device arrays and allocate/copy source grid and area.
    float *d_xs, *d_ys, *d_zs, *d_A;

    gpuErrchk( cudaMalloc((void**)&d_xs, cs->size * sizeof(float)) );
    gpuErrchk( cudaMalloc((void**)&d_ys, cs->size * sizeof(float)) );
    gpuErrchk( cudaMalloc((void**)&d_zs, cs->size * sizeof(float)) );
    gpuErrchk( cudaMalloc((void**)&d_A, cs->size * sizeof(float)) );

    gpuErrchk( cudaMemcpy(d_xs, cs->x, cs->size * sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_ys, cs->y, cs->size * sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_zs, cs->z, cs->size * sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_A, cs->area, cs->size * sizeof(float), cudaMemcpyHostToDevice) );

    // Create pointers to target grid
    float *d_xt, *d_yt, *d_zt;
    gpuErrchk( cudaMalloc((void**)&d_xt, ct->size * sizeof(float)) );
    gpuErrchk( cudaMalloc((void**)&d_yt, ct->size * sizeof(float)) );
    gpuErrchk( cudaMalloc((void**)&d_zt, ct->size * sizeof(float)) );

    gpuErrchk( cudaMemcpy(d_xt, ct->x, ct->size * sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_yt, ct->y, ct->size * sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_zt, ct->z, ct->size * sizeof(float), cudaMemcpyHostToDevice) );

    cuFloatComplex *h_Jx = new cuFloatComplex[cs->size];
    cuFloatComplex *h_Jy = new cuFloatComplex[cs->size];
    cuFloatComplex *h_Jz = new cuFloatComplex[cs->size];

    cuFloatComplex *h_Mx = new cuFloatComplex[cs->size];
    cuFloatComplex *h_My = new cuFloatComplex[cs->size];
    cuFloatComplex *h_Mz = new cuFloatComplex[cs->size];

    _arrC3ToCUDAC(currents->r1x, currents->r1y, currents->r1z,
                  currents->i1x, currents->i1y, currents->i1z,
                  h_Jx, h_Jy, h_Jz, cs->size);

    _arrC3ToCUDAC(currents->r2x, currents->r2y, currents->r2z,
                  currents->i2x, currents->i2y, currents->i2z,
                  h_Mx, h_My, h_Mz, cs->size);

    cuFloatComplex *d_Jx, *d_Jy, *d_Jz;
    cuFloatComplex *d_Mx, *d_My, *d_Mz;

    // Allocate and copy J and M currents
    gpuErrchk( cudaMalloc((void**)&d_Jx, cs->size * sizeof(cuFloatComplex)) );
    gpuErrchk( cudaMalloc((void**)&d_Jy, cs->size * sizeof(cuFloatComplex)) );
    gpuErrchk( cudaMalloc((void**)&d_Jz, cs->size * sizeof(cuFloatComplex)) );

    gpuErrchk( cudaMemcpy(d_Jx, h_Jx, cs->size * sizeof(cuFloatComplex), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_Jy, h_Jy, cs->size * sizeof(cuFloatComplex), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_Jz, h_Jz, cs->size * sizeof(cuFloatComplex), cudaMemcpyHostToDevice) );

    gpuErrchk( cudaMalloc((void**)&d_Mx, cs->size * sizeof(cuFloatComplex)) );
    gpuErrchk( cudaMalloc((void**)&d_My, cs->size * sizeof(cuFloatComplex)) );
    gpuErrchk( cudaMalloc((void**)&d_Mz, cs->size * sizeof(cuFloatComplex)) );

    gpuErrchk( cudaMemcpy(d_Mx, h_Mx, cs->size * sizeof(cuFloatComplex), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_My, h_My, cs->size * sizeof(cuFloatComplex), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_Mz, h_Mz, cs->size * sizeof(cuFloatComplex), cudaMemcpyHostToDevice) );

    // Delete host arrays for source currents - not needed anymore
    delete h_Jx;
    delete h_Jy;
    delete h_Jz;

    delete h_Mx;
    delete h_My;
    delete h_Mz;

    cuFloatComplex *d_Ext, *d_Eyt, *d_Ezt;
    cuFloatComplex *d_Hxt, *d_Hyt, *d_Hzt;

    gpuErrchk( cudaMalloc((void**)&d_Ext, ct->size * sizeof(cuFloatComplex)) );
    gpuErrchk( cudaMalloc((void**)&d_Eyt, ct->size * sizeof(cuFloatComplex)) );
    gpuErrchk( cudaMalloc((void**)&d_Ezt, ct->size * sizeof(cuFloatComplex)) );

    gpuErrchk( cudaMalloc((void**)&d_Hxt, ct->size * sizeof(cuFloatComplex)) );
    gpuErrchk( cudaMalloc((void**)&d_Hyt, ct->size * sizeof(cuFloatComplex)) );
    gpuErrchk( cudaMalloc((void**)&d_Hzt, ct->size * sizeof(cuFloatComplex)) );

    // Create stopping event for kernel
    cudaEvent_t event;
    gpuErrchk( cudaEventCreateWithFlags(&event, cudaEventDisableTiming) );

    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;

    // Call to KERNEL 1
    printf("Calculating E and H...\n");
    begin = std::chrono::steady_clock::now();
    GpropagateBeam_1<<<BT[0], BT[1]>>>(d_xs, d_ys, d_zs,
                                d_A, d_xt, d_yt, d_zt,
                                d_Jx, d_Jy, d_Jz,
                                d_Mx, d_My, d_Mz,
                                d_Ext, d_Eyt, d_Ezt,
                                d_Hxt, d_Hyt, d_Hzt);

    gpuErrchk( cudaDeviceSynchronize() );

    end = std::chrono::steady_clock::now();

    std::cout << "Elapsed time : "
            << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count()
            << " [s]" << std::endl;

    // Allocate Host arrays for E and H
    cuFloatComplex *h_Ext = new cuFloatComplex[ct->size];
    cuFloatComplex *h_Eyt = new cuFloatComplex[ct->size];
    cuFloatComplex *h_Ezt = new cuFloatComplex[ct->size];

    cuFloatComplex *h_Hxt = new cuFloatComplex[ct->size];
    cuFloatComplex *h_Hyt = new cuFloatComplex[ct->size];
    cuFloatComplex *h_Hzt = new cuFloatComplex[ct->size];

    // Copy data back from Device to Host
    gpuErrchk( cudaMemcpy(h_Ext, d_Ext, ct->size * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(h_Eyt, d_Eyt, ct->size * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(h_Ezt, d_Ezt, ct->size * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost) );

    gpuErrchk( cudaMemcpy(h_Hxt, d_Hxt, ct->size * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(h_Hyt, d_Hyt, ct->size * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(h_Hzt, d_Hzt, ct->size * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost) );

    // Free Device memory
    gpuErrchk( cudaDeviceReset() );

    _arrCUDACToC3(h_Ext, h_Eyt, h_Ezt, res->r1x, res->r1y, res->r1z, res->i1x, res->i1y, res->i1z, ct->size);
    _arrCUDACToC3(h_Hxt, h_Hyt, h_Hzt, res->r2x, res->r2y, res->r2z, res->i2x, res->i2y, res->i2z, ct->size);

    // Delete host arrays for target fields
    delete h_Ext;
    delete h_Eyt;
    delete h_Ezt;

    delete h_Hxt;
    delete h_Hyt;
    delete h_Hzt;
}

/**
 * (PUBLIC)
 * Wrapper for calling kernel 2.
 *
 * @param d_x Pointer for x array on device.
 * @param d_y Pointer for y array on device.
 * @param d_z Pointer for z array on device.
 * @param h_x Pointer for x array on host.
 * @param h_y Pointer for y array on host.
 * @param h_z Pointer for z array on host.
 * @param size Number of elements of h_x/d_x.
 */
extern "C" void callKernelf_JMEH(c4Bundlef *res, reflparamsf source, reflparamsf target,
                                reflcontainerf *cs, reflcontainerf *ct,
                                c2Bundlef *currents,
                                float k, float epsilon,
                                float t_direction, int nBlocks, int nThreads)
{
    // Generate source and target grids
    generateGridf(source, cs);
    generateGridf(target, ct);

    // Initialize and copy constant memory to device
    std::array<dim3, 2> BT;
    BT = _initCUDA(k, epsilon, ct->size, cs->size, t_direction, nBlocks, nThreads);

    // Create pointers to device arrays and allocate/copy source grid and area.
    float *d_xs, *d_ys, *d_zs, *d_A;

    gpuErrchk( cudaMalloc((void**)&d_xs, cs->size * sizeof(float)) );
    gpuErrchk( cudaMalloc((void**)&d_ys, cs->size * sizeof(float)) );
    gpuErrchk( cudaMalloc((void**)&d_zs, cs->size * sizeof(float)) );
    gpuErrchk( cudaMalloc((void**)&d_A, cs->size * sizeof(float)) );

    gpuErrchk( cudaMemcpy(d_xs, cs->x, cs->size * sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_ys, cs->y, cs->size * sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_zs, cs->z, cs->size * sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_A, cs->area, cs->size * sizeof(float), cudaMemcpyHostToDevice) );

    // Create pointers to target grid and normal vectors
    float *d_xt, *d_yt, *d_zt, *d_nxt, *d_nyt, *d_nzt;

    // Allocate target co-ordinate and normal grids
    gpuErrchk( cudaMalloc((void**)&d_xt, ct->size * sizeof(float)) );
    gpuErrchk( cudaMalloc((void**)&d_yt, ct->size * sizeof(float)) );
    gpuErrchk( cudaMalloc((void**)&d_zt, ct->size * sizeof(float)) );

    gpuErrchk( cudaMemcpy(d_xt, ct->x, ct->size * sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_yt, ct->y, ct->size * sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_zt, ct->z, ct->size * sizeof(float), cudaMemcpyHostToDevice) );

    gpuErrchk( cudaMalloc((void**)&d_nxt, ct->size * sizeof(float)) );
    gpuErrchk( cudaMalloc((void**)&d_nyt, ct->size * sizeof(float)) );
    gpuErrchk( cudaMalloc((void**)&d_nzt, ct->size * sizeof(float)) );

    gpuErrchk( cudaMemcpy(d_nxt, ct->nx, ct->size * sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_nyt, ct->ny, ct->size * sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_nzt, ct->nz, ct->size * sizeof(float), cudaMemcpyHostToDevice) );

    cuFloatComplex *h_Jx = new cuFloatComplex[cs->size];
    cuFloatComplex *h_Jy = new cuFloatComplex[cs->size];
    cuFloatComplex *h_Jz = new cuFloatComplex[cs->size];

    cuFloatComplex *h_Mx = new cuFloatComplex[cs->size];
    cuFloatComplex *h_My = new cuFloatComplex[cs->size];
    cuFloatComplex *h_Mz = new cuFloatComplex[cs->size];

    cuFloatComplex *d_Jx, *d_Jy, *d_Jz;
    cuFloatComplex *d_Mx, *d_My, *d_Mz;

    _arrC3ToCUDAC(currents->r1x, currents->r1y, currents->r1z,
                  currents->i1x, currents->i1y, currents->i1z,
                  h_Jx, h_Jy, h_Jz, cs->size);

    _arrC3ToCUDAC(currents->r2x, currents->r2y, currents->r2z,
                  currents->i2x, currents->i2y, currents->i2z,
                  h_Mx, h_My, h_Mz, cs->size);

    // Allocate and copy J and M currents
    gpuErrchk( cudaMalloc((void**)&d_Jx, cs->size * sizeof(cuFloatComplex)) );
    gpuErrchk( cudaMalloc((void**)&d_Jy, cs->size * sizeof(cuFloatComplex)) );
    gpuErrchk( cudaMalloc((void**)&d_Jz, cs->size * sizeof(cuFloatComplex)) );

    gpuErrchk( cudaMemcpy(d_Jx, h_Jx, cs->size * sizeof(cuFloatComplex), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_Jy, h_Jy, cs->size * sizeof(cuFloatComplex), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_Jz, h_Jz, cs->size * sizeof(cuFloatComplex), cudaMemcpyHostToDevice) );

    gpuErrchk( cudaMalloc((void**)&d_Mx, cs->size * sizeof(cuFloatComplex)) );
    gpuErrchk( cudaMalloc((void**)&d_My, cs->size * sizeof(cuFloatComplex)) );
    gpuErrchk( cudaMalloc((void**)&d_Mz, cs->size * sizeof(cuFloatComplex)) );

    gpuErrchk( cudaMemcpy(d_Mx, h_Mx, cs->size * sizeof(cuFloatComplex), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_My, h_My, cs->size * sizeof(cuFloatComplex), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_Mz, h_Mz, cs->size * sizeof(cuFloatComplex), cudaMemcpyHostToDevice) );

    // Delete host arrays for source currents - not needed anymore
    delete h_Jx;
    delete h_Jy;
    delete h_Jz;

    delete h_Mx;
    delete h_My;
    delete h_Mz;

    cuFloatComplex *d_Jxt, *d_Jyt, *d_Jzt;
    cuFloatComplex *d_Mxt, *d_Myt, *d_Mzt;
    cuFloatComplex *d_Ext, *d_Eyt, *d_Ezt;
    cuFloatComplex *d_Hxt, *d_Hyt, *d_Hzt;

    gpuErrchk( cudaMalloc((void**)&d_Ext, ct->size * sizeof(cuFloatComplex)) );
    gpuErrchk( cudaMalloc((void**)&d_Eyt, ct->size * sizeof(cuFloatComplex)) );
    gpuErrchk( cudaMalloc((void**)&d_Ezt, ct->size * sizeof(cuFloatComplex)) );

    gpuErrchk( cudaMalloc((void**)&d_Hxt, ct->size * sizeof(cuFloatComplex)) );
    gpuErrchk( cudaMalloc((void**)&d_Hyt, ct->size * sizeof(cuFloatComplex)) );
    gpuErrchk( cudaMalloc((void**)&d_Hzt, ct->size * sizeof(cuFloatComplex)) );

    gpuErrchk( cudaMalloc((void**)&d_Jxt, ct->size * sizeof(cuFloatComplex)) );
    gpuErrchk( cudaMalloc((void**)&d_Jyt, ct->size * sizeof(cuFloatComplex)) );
    gpuErrchk( cudaMalloc((void**)&d_Jzt, ct->size * sizeof(cuFloatComplex)) );

    gpuErrchk( cudaMalloc((void**)&d_Mxt, ct->size * sizeof(cuFloatComplex)) );
    gpuErrchk( cudaMalloc((void**)&d_Myt, ct->size * sizeof(cuFloatComplex)) );
    gpuErrchk( cudaMalloc((void**)&d_Mzt, ct->size * sizeof(cuFloatComplex)) );

    // Create stopping event for kernel
    cudaEvent_t event;
    gpuErrchk( cudaEventCreateWithFlags(&event, cudaEventDisableTiming) );

    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;

    // Call to KERNEL 2
    printf("Calculating J, M, E and H...\n");
    begin = std::chrono::steady_clock::now();
    GpropagateBeam_2<<<BT[0], BT[1]>>>(d_xs, d_ys, d_zs,
                                   d_A, d_xt, d_yt, d_zt,
                                   d_nxt, d_nyt, d_nzt,
                                   d_Jx, d_Jy, d_Jz,
                                   d_Mx, d_My, d_Mz,
                                   d_Jxt, d_Jyt, d_Jzt,
                                   d_Mxt, d_Myt, d_Mzt,
                                   d_Ext, d_Eyt, d_Ezt,
                                   d_Hxt, d_Hyt, d_Hzt);

    gpuErrchk( cudaDeviceSynchronize() );

    end = std::chrono::steady_clock::now();

    std::cout << "Elapsed time : "
            << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count()
            << " [s]" << std::endl;

    // Allocate Host arrays for J and M
    cuFloatComplex *h_Jxt = new cuFloatComplex[ct->size];
    cuFloatComplex *h_Jyt = new cuFloatComplex[ct->size];
    cuFloatComplex *h_Jzt = new cuFloatComplex[ct->size];

    cuFloatComplex *h_Mxt = new cuFloatComplex[ct->size];
    cuFloatComplex *h_Myt = new cuFloatComplex[ct->size];
    cuFloatComplex *h_Mzt = new cuFloatComplex[ct->size];

    // Allocate Host arrays for E and H
    cuFloatComplex *h_Ext = new cuFloatComplex[ct->size];
    cuFloatComplex *h_Eyt = new cuFloatComplex[ct->size];
    cuFloatComplex *h_Ezt = new cuFloatComplex[ct->size];

    cuFloatComplex *h_Hxt = new cuFloatComplex[ct->size];
    cuFloatComplex *h_Hyt = new cuFloatComplex[ct->size];
    cuFloatComplex *h_Hzt = new cuFloatComplex[ct->size];

    // Copy data back from Device to Host
    gpuErrchk( cudaMemcpy(h_Jxt, d_Jxt, ct->size * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(h_Jyt, d_Jyt, ct->size * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(h_Jzt, d_Jzt, ct->size * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost) );

    gpuErrchk( cudaMemcpy(h_Mxt, d_Mxt, ct->size * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(h_Myt, d_Myt, ct->size * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(h_Mzt, d_Mzt, ct->size * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost) );

    gpuErrchk( cudaMemcpy(h_Ext, d_Ext, ct->size * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(h_Eyt, d_Eyt, ct->size * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(h_Ezt, d_Ezt, ct->size * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost) );

    gpuErrchk( cudaMemcpy(h_Hxt, d_Hxt, ct->size * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(h_Hyt, d_Hyt, ct->size * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(h_Hzt, d_Hzt, ct->size * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost) );

    // Free Device memory
    gpuErrchk( cudaDeviceReset() );

    _arrCUDACToC3(h_Jxt, h_Jyt, h_Jzt, res->r1x, res->r1y, res->r1z, res->i1x, res->i1y, res->i1z, ct->size);
    _arrCUDACToC3(h_Mxt, h_Myt, h_Mzt, res->r2x, res->r2y, res->r2z, res->i2x, res->i2y, res->i2z, ct->size);

    _arrCUDACToC3(h_Ext, h_Eyt, h_Ezt, res->r3x, res->r3y, res->r3z, res->i3x, res->i3y, res->i3z, ct->size);
    _arrCUDACToC3(h_Hxt, h_Hyt, h_Hzt, res->r4x, res->r4y, res->r4z, res->i4x, res->i4y, res->i4z, ct->size);

    // Delete host arrays for target currents
    delete h_Jxt;
    delete h_Jyt;
    delete h_Jzt;

    delete h_Mxt;
    delete h_Myt;
    delete h_Mzt;

    // Delete host arrays for target fields
    delete h_Ext;
    delete h_Eyt;
    delete h_Ezt;

    delete h_Hxt;
    delete h_Hyt;
    delete h_Hzt;
}

/**
 * (PUBLIC)
 * Wrapper for calling kernel 3.
 *
 * @param d_x Pointer for x array on device.
 * @param d_y Pointer for y array on device.
 * @param d_z Pointer for z array on device.
 * @param h_x Pointer for x array on host.
 * @param h_y Pointer for y array on host.
 * @param h_z Pointer for z array on host.
 * @param size Number of elements of h_x/d_x.
 */
extern "C" void callKernelf_EHP(c2rBundlef *res, reflparamsf source, reflparamsf target,
                                reflcontainerf *cs, reflcontainerf *ct,
                                c2Bundlef *currents,
                                float k, float epsilon,
                                float t_direction, int nBlocks, int nThreads)
{
    // Generate source and target grids
    generateGridf(source, cs);
    generateGridf(target, ct);

    // Initialize and copy constant memory to device
    std::array<dim3, 2> BT;
    BT = _initCUDA(k, epsilon, ct->size, cs->size, t_direction, nBlocks, nThreads);

    // Create pointers to device arrays and allocate/copy source grid and area.
    float *d_xs, *d_ys, *d_zs, *d_A;

    gpuErrchk( cudaMalloc((void**)&d_xs, ct->size * sizeof(float)) );
    gpuErrchk( cudaMalloc((void**)&d_ys, ct->size * sizeof(float)) );
    gpuErrchk( cudaMalloc((void**)&d_zs, ct->size * sizeof(float)) );
    gpuErrchk( cudaMalloc((void**)&d_A, ct->size * sizeof(float)) );

    gpuErrchk( cudaMemcpy(d_xs, cs->x, cs->size * sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_ys, cs->y, cs->size * sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_zs, cs->z, cs->size * sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_A, cs->area, cs->size * sizeof(float), cudaMemcpyHostToDevice) );

    // Create pointers to target grid and normal vectors
    float *d_xt, *d_yt, *d_zt, *d_nxt, *d_nyt, *d_nzt;
    gpuErrchk( cudaMalloc((void**)&d_xt, ct->size * sizeof(float)) );
    gpuErrchk( cudaMalloc((void**)&d_yt, ct->size * sizeof(float)) );
    gpuErrchk( cudaMalloc((void**)&d_zt, ct->size * sizeof(float)) );

    gpuErrchk( cudaMemcpy(d_xt, ct->x, cs->size * sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_yt, ct->y, cs->size * sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_zt, ct->z, cs->size * sizeof(float), cudaMemcpyHostToDevice) );

    gpuErrchk( cudaMalloc((void**)&d_nxt, ct->size * sizeof(float)) );
    gpuErrchk( cudaMalloc((void**)&d_nyt, ct->size * sizeof(float)) );
    gpuErrchk( cudaMalloc((void**)&d_nzt, ct->size * sizeof(float)) );

    gpuErrchk( cudaMemcpy(d_nxt, ct->nx, cs->size * sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_nyt, ct->ny, cs->size * sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_nzt, ct->nz, cs->size * sizeof(float), cudaMemcpyHostToDevice) );

    cuFloatComplex *h_Jx = new cuFloatComplex[cs->size];
    cuFloatComplex *h_Jy = new cuFloatComplex[cs->size];
    cuFloatComplex *h_Jz = new cuFloatComplex[cs->size];

    cuFloatComplex *h_Mx = new cuFloatComplex[cs->size];
    cuFloatComplex *h_My = new cuFloatComplex[cs->size];
    cuFloatComplex *h_Mz = new cuFloatComplex[cs->size];

    cuFloatComplex *d_Jx, *d_Jy, *d_Jz;
    cuFloatComplex *d_Mx, *d_My, *d_Mz;

    _arrC3ToCUDAC(currents->r1x, currents->r1y, currents->r1z,
                  currents->i1x, currents->i1y, currents->i1z,
                  h_Jx, h_Jy, h_Jz, cs->size);

    _arrC3ToCUDAC(currents->r2x, currents->r2y, currents->r2z,
                  currents->i2x, currents->i2y, currents->i2z,
                  h_Mx, h_My, h_Mz, cs->size);

    gpuErrchk( cudaMalloc((void**)&d_Jx, cs->size * sizeof(cuFloatComplex)) );
    gpuErrchk( cudaMalloc((void**)&d_Jy, cs->size * sizeof(cuFloatComplex)) );
    gpuErrchk( cudaMalloc((void**)&d_Jz, cs->size * sizeof(cuFloatComplex)) );

    gpuErrchk( cudaMemcpy(d_Jx, h_Jx, cs->size * sizeof(cuFloatComplex), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_Jy, h_Jy, cs->size * sizeof(cuFloatComplex), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_Jz, h_Jz, cs->size * sizeof(cuFloatComplex), cudaMemcpyHostToDevice) );

    gpuErrchk( cudaMalloc((void**)&d_Mx, cs->size * sizeof(cuFloatComplex)) );
    gpuErrchk( cudaMalloc((void**)&d_My, cs->size * sizeof(cuFloatComplex)) );
    gpuErrchk( cudaMalloc((void**)&d_Mz, cs->size * sizeof(cuFloatComplex)) );

    gpuErrchk( cudaMemcpy(d_Mx, h_Mx, cs->size * sizeof(cuFloatComplex), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_My, h_My, cs->size * sizeof(cuFloatComplex), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_Mz, h_Mz, cs->size * sizeof(cuFloatComplex), cudaMemcpyHostToDevice) );

    // Delete host arrays for source currents
    delete h_Jx;
    delete h_Jy;
    delete h_Jz;

    delete h_Mx;
    delete h_My;
    delete h_Mz;

    cuFloatComplex *d_Ext, *d_Eyt, *d_Ezt;
    cuFloatComplex *d_Hxt, *d_Hyt, *d_Hzt;
    float *d_Prxt, *d_Pryt, *d_Przt;

    gpuErrchk( cudaMalloc((void**)&d_Ext, ct->size * sizeof(cuFloatComplex)) );
    gpuErrchk( cudaMalloc((void**)&d_Eyt, ct->size * sizeof(cuFloatComplex)) );
    gpuErrchk( cudaMalloc((void**)&d_Ezt, ct->size * sizeof(cuFloatComplex)) );

    gpuErrchk( cudaMalloc((void**)&d_Hxt, ct->size * sizeof(cuFloatComplex)) );
    gpuErrchk( cudaMalloc((void**)&d_Hyt, ct->size * sizeof(cuFloatComplex)) );
    gpuErrchk( cudaMalloc((void**)&d_Hzt, ct->size * sizeof(cuFloatComplex)) );

    gpuErrchk( cudaMalloc((void**)&d_Prxt, ct->size * sizeof(float)) );
    gpuErrchk( cudaMalloc((void**)&d_Pryt, ct->size * sizeof(float)) );
    gpuErrchk( cudaMalloc((void**)&d_Przt, ct->size * sizeof(float)) );

    // Create stopping event for kernel
    cudaEvent_t event;
    gpuErrchk( cudaEventCreateWithFlags(&event, cudaEventDisableTiming) );

    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;

    // Call to KERNEL 3
    printf("Calculating E, H and Pr...\n");
    begin = std::chrono::steady_clock::now();
    GpropagateBeam_3<<<BT[0], BT[1]>>>(d_xs, d_ys, d_zs,
                                d_A, d_xt, d_yt, d_zt,
                                d_nxt, d_nyt, d_nzt,
                                d_Jx, d_Jy, d_Jz,
                                d_Mx, d_My, d_Mz,
                                d_Ext, d_Eyt, d_Ezt,
                                d_Hxt, d_Hyt, d_Hzt,
                                d_Prxt, d_Pryt, d_Przt);

    gpuErrchk( cudaDeviceSynchronize() );

    end = std::chrono::steady_clock::now();

    std::cout << "Elapsed time : "
            << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count()
            << " [s]" << std::endl;

    // Allocate Host arrays for E, H and P
    cuFloatComplex *h_Ext = new cuFloatComplex[ct->size];
    cuFloatComplex *h_Eyt = new cuFloatComplex[ct->size];
    cuFloatComplex *h_Ezt = new cuFloatComplex[ct->size];

    cuFloatComplex *h_Hxt = new cuFloatComplex[ct->size];
    cuFloatComplex *h_Hyt = new cuFloatComplex[ct->size];
    cuFloatComplex *h_Hzt = new cuFloatComplex[ct->size];

    float *h_Prxt = new float[ct->size];
    float *h_Pryt = new float[ct->size];
    float *h_Przt = new float[ct->size];

    // Copy data back from Device to Host
    gpuErrchk( cudaMemcpy(h_Ext, d_Ext, ct->size * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(h_Eyt, d_Eyt, ct->size * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(h_Ezt, d_Ezt, ct->size * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost) );

    gpuErrchk( cudaMemcpy(h_Hxt, d_Hxt, ct->size * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(h_Hyt, d_Hyt, ct->size * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(h_Hzt, d_Hzt, ct->size * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost) );

    gpuErrchk( cudaMemcpy(h_Prxt, d_Prxt, ct->size * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(h_Pryt, d_Pryt, ct->size * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(h_Przt, d_Przt, ct->size * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost) );

    // Free Device memory
    gpuErrchk( cudaDeviceReset() );

    _arrCUDACToC3(h_Ext, h_Eyt, h_Ezt, res->r1x, res->r1y, res->r1z, res->i1x, res->i1y, res->i1z, ct->size);
    _arrCUDACToC3(h_Hxt, h_Hyt, h_Hzt, res->r2x, res->r2y, res->r2z, res->i2x, res->i2y, res->i2z, ct->size);

    res->r3x = h_Prxt;
    res->r3y = h_Pryt;
    res->r3z = h_Przt;

    // Delete host arrays for target fields
    delete h_Ext;
    delete h_Eyt;
    delete h_Ezt;

    delete h_Hxt;
    delete h_Hyt;
    delete h_Hzt;

    delete h_Prxt;
    delete h_Pryt;
    delete h_Przt;
}

/**
 * (PUBLIC)
 * Wrapper for calling kernel 4 (far-field).
 *
 * @param d_x Pointer for x array on device.
 * @param d_y Pointer for y array on device.
 * @param d_z Pointer for z array on device.
 * @param h_x Pointer for x array on host.
 * @param h_y Pointer for y array on host.
 * @param h_z Pointer for z array on host.
 * @param size Number of elements of h_x/d_x.
 */
/*
extern "C" void callKernelf_FF(arrC3f *res, float *xt, float *yt,
                                float *xs, float *ys, float *zs,
                                float *rJxs, float *rJys, float *rJzs,
                                float *iJxs, float *iJys, float *iJzs,
                                float *rMxs, float *rMys, float *rMzs,
                                float *iMxs, float *iMys, float *iMzs,
                                float *area, float k, float epsilon, int ct->size, int cs->size,
                                float t_direction, int nBlocks, int nThreads)
{
    // Initialize and copy constant memory to device
    std::array<dim3, 2> BT;
    BT = _initCUDA(k, epsilon, ct->size, cs->size, t_direction, nBlocks, nThreads);

    // Create pointers to device arrays and allocate/copy source grid and area.
    float *d_xs, *d_ys, *d_zs, *d_A;
    _allocateGridR3ToGPU(d_xs, d_ys, d_zs, xs, ys, zs, cs->size);
    _allocateGridR1ToGPU(d_A, area, cs->size);

    // Create pointers to target grid and normal vectors
    float *d_xt, *d_yt;
    _allocateGridR2ToGPU(d_xt, d_yt, xt, yt, ct->size);

    cuFloatComplex *h_Jx, *h_Jy, *h_Jz;
    cuFloatComplex *h_Mx, *h_My, *h_Mz;

    cuFloatComplex *d_Jx, *d_Jy, *d_Jz;
    cuFloatComplex *d_Mx, *d_My, *d_Mz;

    _arrC3ToCUDAC(rJxs, rJys, rJzs, iJxs, iJys, iJzs,
                    h_Jx, h_Jy, h_Jz, cs->size);

    _arrC3ToCUDAC(rMxs, rMys, rMzs, iMxs, iMys, iMzs,
                    h_Mx, h_My, h_Mz, cs->size);

    _allocateGridC3ToGPU(d_Jx, d_Jy, d_Jz, h_Jx, h_Jy, h_Jz, cs->size);
    _allocateGridC3ToGPU(d_Mx, d_My, d_Mz, h_Mx, h_My, h_Mz, cs->size);

    cuFloatComplex *d_Ext, *d_Eyt, *d_Ezt;

    _allocateGridC3(d_Ext, d_Eyt, d_Ezt, cs->size);

    // Create stopping event for kernel
    cudaEvent_t event;
    gpuErrchk( cudaEventCreateWithFlags(&event, cudaEventDisableTiming) );

    // Call to KERNEL 4
    GpropagateBeam_4<<<BT[0], BT[1]>>>(d_xs, d_ys, d_zs,
                                d_A, d_xt, d_yt,
                                d_Jx, d_Jy, d_Jz,
                                d_Mx, d_My, d_Mz,
                                d_Ext, d_Eyt, d_Ezt);

    gpuErrchk( cudaDeviceSynchronize() );

    // Allocate, on stackframe, Host arrays for E and H
    cuFloatComplex h_Ext[ct->size], h_Eyt[ct->size], h_Ezt[ct->size];

    // Copy data back from Device to Host
    _allocateGridGPUToC3(h_Ext, h_Eyt, h_Ezt, d_Ext, d_Eyt, d_Ezt, ct->size);

    // Free Device memory
    gpuErrchk( cudaDeviceReset() );

    // Create arrays of floats, for transferring back output
    float *rExt, *rEyt, *rEzt, *iExt, *iEyt, *iEzt;

    _arrCUDACToC3(h_Ext, h_Eyt, h_Ezt, rExt, rEyt, rEzt, iExt, iEyt, iEzt, ct->size);

    arrC3 *out = new arrC3;
    if (out == NULL) exit (1); //EXIT_FAILURE

    fill_arrC3(out, rExt, rEyt, rEzt, iExt, iEyt, iEzt);

    return out;
}
*/
