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

#define M_PI            3.14159265358979323846  /* pi */
#define C_L             2.99792458e11 // mm s^-1
#define MU_0            1.2566370614e-3 // kg mm s^-2 A^-2
#define EPS_VAC         1 / (MU_0 * C_L*C_L)
#define ZETA_0_INV      1 / (C_L * MU_0)

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
__constant__ cuDoubleComplex con[CSIZE];     // Contains: k, eps, mu0, zeta0, pi, C_l, Time direction, unit, zero, c4 as complex numbers
__constant__ double eye[3][3];      // Identity matrix
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


/**
 * Function to calculate complex exponential with
 * CUDA type cuDoubleComplex.
 *
 * @param z Complex number.
 */
__device__ __inline__ cuDoubleComplex my_cexp(cuDoubleComplex z)
{
    cuDoubleComplex res;
    double t = exp(z.x);
    double ys = sin(z.y);
    double yc = cos(z.y);
    res = cuCmul(make_cuDoubleComplex(t, 0.), make_cuDoubleComplex(yc, ys));
    return res;
}



/**
 * (PRIVATE)
 * Instantiate program and populate constant memory.
 *
 * @param k Wavenumber of incoming field.
 * @param epsilon Relative permittivity of source.
 * @param gt Number of elements in target.
 * @param gs Number of elements in source.
 * @param t_direction Sign of exponent in Green function.
 * @param nBlock Number of blocks per grid.
 * @param nThreads Number of threads per block.
 *
 * @return BT Array of two dim3 objects.
 */

__host__ std::array<dim3, 2> _initCUDA(double k, double epsilon, int gt, int gs, double t_direction, int nBlocks, int nThreads)
{
    // Calculate nr of blocks per grid and nr of threads per block
    dim3 nrb(nBlocks); dim3 nrt(nThreads);

    // Calculate permittivity of target
    double EPS = EPS_VAC * epsilon;

    // Fill ID matrix
    double _eye[3][3];
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
    cuDoubleComplex _con[CSIZE] = {make_cuDoubleComplex(k, 0.),
                                    make_cuDoubleComplex(EPS, 0.),
                                    make_cuDoubleComplex(MU_0, 0.),
                                    make_cuDoubleComplex(ZETA_0_INV, 0.),
                                    make_cuDoubleComplex(M_PI, 0.),
                                    make_cuDoubleComplex(C_L, 0.),
                                    make_cuDoubleComplex(t_direction, 0.),
                                    make_cuDoubleComplex(0., 1.),
                                    make_cuDoubleComplex(0., 0.),
                                    make_cuDoubleComplex(4., 0.)};

    // Copy constant array to Device constant memory
    gpuErrchk( cudaMemcpyToSymbol(con, &_con, CSIZE * sizeof(cuDoubleComplex)) );
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
 * @param point C-style array of length 3 containing xyz coordinates of target point.
 * @param d_A C-style array containing area elements.
 * @param d_ei C-style array of length 3 to be filled with E-field at point.
 * @param d_hi C-style array of length 3 to be filled with H-field at point.
 */
__device__ void fieldAtPoint(double *d_xs, double *d_ys, double*d_zs,
                    cuDoubleComplex *d_Jx, cuDoubleComplex *d_Jy, cuDoubleComplex *d_Jz,
                    cuDoubleComplex *d_Mx, cuDoubleComplex *d_My, cuDoubleComplex *d_Mz,
                    double (&point)[3], double *d_A,
                    cuDoubleComplex (&d_ei)[3], cuDoubleComplex (&d_hi)[3])
{
    // Scalars (double & complex double)
    double r;                           // Distance between source and target points
    double r_inv;                       // 1 / r
    cuDoubleComplex omega;                       // Angular frequency of field
    cuDoubleComplex Green;         // Container for Green's function
    cuDoubleComplex r_in_s;        // Container for inner products between wavevctor and currents
    cuDoubleComplex rc;

    // Arrays of doubles
    double source_point[3]; // Container for xyz co-ordinates
    double r_vec[3];        // Distance vector between source and target points
    double k_hat[3];        // Unit wavevctor
    double k_arr[3];        // Wavevector

    // Arrays of complex doubles
    cuDoubleComplex e_field[3] = {con[8], con[8], con[8]}; // Electric field on target
    cuDoubleComplex h_field[3] = {con[8], con[8], con[8]}; // Magnetic field on target
    cuDoubleComplex js[3];             // Electric current at source point
    cuDoubleComplex ms[3];             // Magnetic current at source point
    cuDoubleComplex e_vec_thing[3];    // Electric current contribution to e-field
    cuDoubleComplex h_vec_thing[3];    // Magnetic current contribution to h-field
    cuDoubleComplex k_out_ms[3];       // Outer product between k and ms
    cuDoubleComplex k_out_js[3];       // Outer product between k and js
    cuDoubleComplex temp[3];           // Temporary container for intermediate values

    //e_field = {con[8], con[8], con[8]};
    //h_field = {con[8], con[8], con[8]};

    omega = cuCmul(con[5], con[0]); // C_L * k

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

        rc = make_cuDoubleComplex(r, 0.);
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



        cuDoubleComplex d_Ac = make_cuDoubleComplex(d_A[i], 0.);

        Green = cuCmul(cuCdiv(my_cexp(cuCmul(con[6], cuCmul(con[7], cuCmul(con[0], rc)))),
                (cuCmul(con[9], cuCmul(con[4], rc)))), cuCmul(d_Ac, con[7]));

        for( int n=0; n<3; n++)
        {
            e_field[n] = cuCsub(e_field[n], cuCmul(cuCsub(cuCmul(omega, cuCmul(con[2], e_vec_thing[n])), k_out_ms[n]), Green));
            h_field[n] = cuCsub(h_field[n], cuCmul(cuCadd(cuCmul(omega, cuCmul(con[1], h_vec_thing[n])), k_out_js[n]), Green));

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

__device__ void farfieldAtPoint(double *d_xs, double *d_ys, double *d_zs,
                                cuDoubleComplex *d_Jx, cuDoubleComplex *d_Jy, cuDoubleComplex *d_Jz,
                                cuDoubleComplex *d_Mx, cuDoubleComplex *d_My, cuDoubleComplex *d_Mz,
                                double (&r_hat)[3], double *d_A, cuDoubleComplex (&e)[3])
{
    // Scalars (double & complex double)
    double omega_mu;                       // Angular frequency of field times mu
    double r_hat_in_rp;                 // r_hat dot product r_prime

    // Arrays of doubles
    double source_point[3]; // Container for xyz co-ordinates

    // Arrays of complex doubles
    cuDoubleComplex js[3];      // Build radiation integral
    cuDoubleComplex ms[3];      // Build radiation integral

    cuDoubleComplex _ctemp[3];
    cuDoubleComplex js_tot_factor[3];
    cuDoubleComplex ms_tot_factor[3];

    // Matrices
    double rr_dyad[3][3];       // Dyadic product between r_hat - r_hat
    double eye_min_rr[3][3];    // I - rr

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

        cuDoubleComplex cfact = cuCmul(con[7], make_cuDoubleComplex((con[0].x * r_hat_in_rp), 0.));
        cuDoubleComplex expo = my_cexp(cfact);
        cfact = cuCmul(expo, make_cuDoubleComplex(d_A[i], 0.));

        js[0] = cuCadd(js[0], cuCmul(d_Jx[i], cfact));
        js[1] = cuCadd(js[1], cuCmul(d_Jy[i], cfact));
        js[2] = cuCadd(js[2], cuCmul(d_Jz[i], cfact));

        ms[0] = cuCadd(ms[0], cuCmul(d_Mx[i], cfact));
        ms[1] = cuCadd(ms[1], cuCmul(d_My[i], cfact));
        ms[2] = cuCadd(ms[2], cuCmul(d_Mz[i], cfact));

    }
    matVec(eye_min_rr, js, _ctemp);
    s_mult(_ctemp, omega_mu, js_tot_factor);

    ext(r_hat, ms, _ctemp);
    s_mult(_ctemp, con[0].x, ms_tot_factor);

    for (int n=0; n<3; n++)
    {
        e[n] = cuCsub(ms_tot_factor[n], js_tot_factor[n]);
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
__global__ void GpropagateBeam_0(double *d_xs, double *d_ys, double *d_zs,
                                double *d_A, double *d_xt, double *d_yt, double *d_zt,
                                double *d_nxt, double *d_nyt, double *d_nzt,
                                cuDoubleComplex *d_Jx, cuDoubleComplex *d_Jy, cuDoubleComplex *d_Jz,
                                cuDoubleComplex *d_Mx, cuDoubleComplex *d_My, cuDoubleComplex *d_Mz,
                                cuDoubleComplex *d_Jxt, cuDoubleComplex *d_Jyt, cuDoubleComplex *d_Jzt,
                                cuDoubleComplex *d_Mxt, cuDoubleComplex *d_Myt, cuDoubleComplex *d_Mzt)
{

    // Scalars (double & complex double)
    cuDoubleComplex e_dot_p_r_perp;    // E-field - perpendicular reflected POI polarization vector dot product
    cuDoubleComplex e_dot_p_r_parr;    // E-field - parallel reflected POI polarization vector dot product

    // Arrays of doubles
    double S_i_norm[3];         // Normalized incoming Poynting vector
    double p_i_perp[3];         // Perpendicular incoming POI polarization vector
    double p_i_parr[3];         // Parallel incoming POI polarization vector
    double S_r_norm[3];         // Normalized reflected Poynting vector
    double p_r_perp[3];         // Perpendicular reflected POI polarization vector
    double p_r_parr[3];         // Parallel reflected POI polarization vector
    double S_out_n[3];          // Container for Poynting-normal ext products
    double point[3];            // Point on target
    double norms[3];            // Normal vector at point
    double e_out_h_r[3];        // Real part of E-field - H-field ext product

    // Arrays of complex doubles
    cuDoubleComplex e_r[3];            // Reflected E-field
    cuDoubleComplex h_r[3];            // Reflected H-field
    cuDoubleComplex n_out_e_i_r[3];    // Electric current
    cuDoubleComplex temp1[3];          // Temporary container 1 for intermediate irrelevant values
    cuDoubleComplex temp2[3];          // Temporary container 2
    cuDoubleComplex temp3[3];          // Temporary container 3

    // Return containers
    cuDoubleComplex d_ei[3];
    cuDoubleComplex d_hi[3];

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
        conja(d_ei, temp1);                        // h_conj
        ext(d_hi, temp1, temp2);                  // e_out_h

        for (int n=0; n<3; n++)
        {
            e_out_h_r[n] = cuCreal(temp2[n]);                      // e_out_h_r
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
            e_r[n] = cuCsub(cuCmul(e_dot_p_r_perp, make_cuDoubleComplex(-p_i_perp[n], 0.)), cuCmul(e_dot_p_r_parr, make_cuDoubleComplex(p_i_parr[n], 0.)));
        }

        ext(S_r_norm, e_r, temp1);                       // h_r_temp
        s_mult(temp1, con[3], h_r);                     // ZETA_0_INV, h_r

        //Calculate and store J and M only
        for(int n=0; n<3; n++)
        {
            temp1[n] = cuCadd(e_r[n], d_ei[n]); // e_i_r
            temp2[n] = cuCadd(h_r[n], d_hi[n]); // h_i_r
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
__global__ void GpropagateBeam_1(double *d_xs, double *d_ys, double *d_zs,
                                double *d_A, double *d_xt, double *d_yt, double *d_zt,
                                cuDoubleComplex *d_Jx, cuDoubleComplex *d_Jy, cuDoubleComplex *d_Jz,
                                cuDoubleComplex *d_Mx, cuDoubleComplex *d_My, cuDoubleComplex *d_Mz,
                                cuDoubleComplex *d_Ext, cuDoubleComplex *d_Eyt, cuDoubleComplex *d_Ezt,
                                cuDoubleComplex *d_Hxt, cuDoubleComplex *d_Hyt, cuDoubleComplex *d_Hzt)
{
    // Arrays of doubles
    double point[3];            // Point on target

    // Return containers for call to fieldAtPoint
    cuDoubleComplex d_ei[3];
    cuDoubleComplex d_hi[3];

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
__global__ void GpropagateBeam_2(double *d_xs, double *d_ys, double *d_zs,
                                double *d_A, double *d_xt, double *d_yt, double *d_zt,
                                double *d_nxt, double *d_nyt, double *d_nzt,
                                cuDoubleComplex *d_Jx, cuDoubleComplex *d_Jy, cuDoubleComplex *d_Jz,
                                cuDoubleComplex *d_Mx, cuDoubleComplex *d_My, cuDoubleComplex *d_Mz,
                                cuDoubleComplex *d_Jxt, cuDoubleComplex *d_Jyt, cuDoubleComplex *d_Jzt,
                                cuDoubleComplex *d_Mxt, cuDoubleComplex *d_Myt, cuDoubleComplex *d_Mzt,
                                cuDoubleComplex *d_Ext, cuDoubleComplex *d_Eyt, cuDoubleComplex *d_Ezt,
                                cuDoubleComplex *d_Hxt, cuDoubleComplex *d_Hyt, cuDoubleComplex *d_Hzt)
{

    // Scalars (double & complex double)
    cuDoubleComplex e_dot_p_r_perp;    // E-field - perpendicular reflected POI polarization vector dot product
    cuDoubleComplex e_dot_p_r_parr;    // E-field - parallel reflected POI polarization vector dot product

    // Arrays of doubles
    double S_i_norm[3];         // Normalized incoming Poynting vector
    double p_i_perp[3];         // Perpendicular incoming POI polarization vector
    double p_i_parr[3];         // Parallel incoming POI polarization vector
    double S_r_norm[3];         // Normalized reflected Poynting vector
    double p_r_perp[3];         // Perpendicular reflected POI polarization vector
    double p_r_parr[3];         // Parallel reflected POI polarization vector
    double S_out_n[3];          // Container for Poynting-normal ext products
    double point[3];            // Point on target
    double norms[3];            // Normal vector at point
    double e_out_h_r[3];        // Real part of E-field - H-field ext product

    // Arrays of complex doubles
    cuDoubleComplex e_r[3];            // Reflected E-field
    cuDoubleComplex h_r[3];            // Reflected H-field
    cuDoubleComplex n_out_e_i_r[3];    // Electric current
    cuDoubleComplex temp1[3];          // Temporary container 1 for intermediate irrelevant values
    cuDoubleComplex temp2[3];          // Temporary container 2
    cuDoubleComplex temp3[3];          // Temporary container 3

    // Return containers
    cuDoubleComplex d_ei[3];
    cuDoubleComplex d_hi[3];

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
        conja(d_ei, temp1);                        // h_conj
        ext(d_hi, temp1, temp2);                  // e_out_h

        for (int n=0; n<3; n++)
        {
            e_out_h_r[n] = cuCreal(temp2[n]);                      // e_out_h_r
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
            e_r[n] = cuCsub(cuCmul(e_dot_p_r_perp, make_cuDoubleComplex(-p_i_perp[n], 0.)), cuCmul(e_dot_p_r_parr, make_cuDoubleComplex(p_i_parr[n], 0.)));
        }

        ext(S_r_norm, e_r, temp1);                       // h_r_temp
        s_mult(temp1, con[3], h_r);                     // ZETA_0_INV, h_r

        //Calculate and store J and M only
        for(int n=0; n<3; n++)
        {
            temp1[n] = cuCadd(e_r[n], d_ei[n]); // e_i_r
            temp2[n] = cuCadd(h_r[n], d_hi[n]); // h_i_r
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
__global__ void GpropagateBeam_3(double *d_xs, double *d_ys, double *d_zs,
                                double *d_A, double *d_xt, double *d_yt, double *d_zt,
                                double *d_nxt, double *d_nyt, double *d_nzt,
                                cuDoubleComplex *d_Jx, cuDoubleComplex *d_Jy, cuDoubleComplex *d_Jz,
                                cuDoubleComplex *d_Mx, cuDoubleComplex *d_My, cuDoubleComplex *d_Mz,
                                cuDoubleComplex *d_Ext, cuDoubleComplex *d_Eyt, cuDoubleComplex *d_Ezt,
                                cuDoubleComplex *d_Hxt, cuDoubleComplex *d_Hyt, cuDoubleComplex *d_Hzt,
                                double *d_Prxt, double *d_Pryt, double *d_Przt)
{

    // Scalars (double & complex double)
    cuDoubleComplex e_dot_p_r_perp;    // E-field - perpendicular reflected POI polarization vector dot product
    cuDoubleComplex e_dot_p_r_parr;    // E-field - parallel reflected POI polarization vector dot product

    // Arrays of doubles
    double S_i_norm[3];         // Normalized incoming Poynting vector
    double p_i_perp[3];         // Perpendicular incoming POI polarization vector
    double p_i_parr[3];         // Parallel incoming POI polarization vector
    double S_r_norm[3];         // Normalized reflected Poynting vector
    double p_r_perp[3];         // Perpendicular reflected POI polarization vector
    double p_r_parr[3];         // Parallel reflected POI polarization vector
    double S_out_n[3];          // Container for Poynting-normal ext products
    double point[3];            // Point on target
    double norms[3];            // Normal vector at point
    double e_out_h_r[3];        // Real part of E-field - H-field ext product

    // Arrays of complex doubles
    cuDoubleComplex e_r[3];            // Reflected E-field
    cuDoubleComplex h_r[3];            // Reflected H-field
    cuDoubleComplex temp1[3];          // Temporary container 1 for intermediate irrelevant values
    cuDoubleComplex temp2[3];          // Temporary container 2

    // Return containers
    cuDoubleComplex d_ei[3];
    cuDoubleComplex d_hi[3];

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
        conja(d_ei, temp1);                        // h_conj
        ext(d_hi, temp1, temp2);                  // e_out_h

        for (int n=0; n<3; n++)
        {
            e_out_h_r[n] = cuCreal(temp2[n]);                      // e_out_h_r
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
            e_r[n] = cuCsub(cuCmul(e_dot_p_r_perp, make_cuDoubleComplex(-p_i_perp[n], 0.)), cuCmul(e_dot_p_r_parr, make_cuDoubleComplex(p_i_parr[n], 0.)));
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
void __global__ GpropagateBeam_4(double *d_xs, double *d_ys, double *d_zs,
                                double *d_A, double *d_xt, double *d_yt,
                                cuDoubleComplex *d_Jx, cuDoubleComplex *d_Jy, cuDoubleComplex *d_Jz,
                                cuDoubleComplex *d_Mx, cuDoubleComplex *d_My, cuDoubleComplex *d_Mz,
                                cuDoubleComplex *d_Ext, cuDoubleComplex *d_Eyt, cuDoubleComplex *d_Ezt)
{
    // Scalars (double & complex double)
    double theta;
    double phi;

    // Arrays of doubles
    double r_hat[3];                // Unit vector in far-field point direction

    // Arrays of complex doubles
    cuDoubleComplex e[3];            // far-field E-field

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
__host__ void _allocateGridR1ToGPU(double *d_x, double *h_x, int size)
{
    gpuErrchk( cudaMalloc((void**)&d_x, size * sizeof(double)) );
    gpuErrchk( cudaMemcpy(d_x, h_x, size * sizeof(double), cudaMemcpyHostToDevice) );
}

/**
 * (PRIVATE)
 * Allocate R3 to Device.
 *
 * @param d_x Pointer for x array on device.
 * @param d_y Pointer for y array on device.
 * @param d_z Pointer for z array on device.
 * */
__host__ void _allocateGridR3(double *d_x, double *d_y, double *d_z, int size)
{
    gpuErrchk( cudaMalloc((void**)&d_x, size * sizeof(double)) );
    gpuErrchk( cudaMalloc((void**)&d_y, size * sizeof(double)) );
    gpuErrchk( cudaMalloc((void**)&d_z, size * sizeof(double)) );
}

__host__ void _allocateGridR2ToGPU(double *d_x, double *d_y,
                                  double *h_x, double *h_y,
                                    int size)
{
    gpuErrchk( cudaMalloc((void**)&d_x, size * sizeof(double)) );
    gpuErrchk( cudaMalloc((void**)&d_y, size * sizeof(double)) );

    gpuErrchk( cudaMemcpy(d_x, h_x, size * sizeof(double), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_y, h_y, size * sizeof(double), cudaMemcpyHostToDevice) );
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
__host__ void _allocateGridR3ToGPU(double *d_x, double *d_y, double *d_z,
                                  double *h_x, double *h_y, double *h_z,
                                    int size)
{
    gpuErrchk( cudaMalloc((void**)&d_x, size * sizeof(double)) );
    gpuErrchk( cudaMalloc((void**)&d_y, size * sizeof(double)) );
    gpuErrchk( cudaMalloc((void**)&d_z, size * sizeof(double)) );

    gpuErrchk( cudaMemcpy(d_x, h_x, size * sizeof(double), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_y, h_y, size * sizeof(double), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_z, h_z, size * sizeof(double), cudaMemcpyHostToDevice) );
}

__host__ void _allocateGridGPUToR3(double *h_x, double *h_y, double *h_z,
                                   double *d_x, double *d_y, double *d_z,
                                   int size)
{
    gpuErrchk( cudaMemcpy(h_x, d_x, size * sizeof(double), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(h_y, d_y, size * sizeof(double), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(h_z, d_z, size * sizeof(double), cudaMemcpyDeviceToHost) );
}

/**
 * (PRIVATE)
 * Allocate C3 to Device.
 *
 * @param d_x Pointer for x array on device.
 * @param d_y Pointer for y array on device.
 * @param d_z Pointer for z array on device.
 * */
__host__ void _allocateGridC3(cuDoubleComplex *d_x, cuDoubleComplex *d_y, cuDoubleComplex *d_z, int size)
{
    gpuErrchk( cudaMalloc((void**)&d_x, size * sizeof(cuDoubleComplex)) );
    gpuErrchk( cudaMalloc((void**)&d_y, size * sizeof(cuDoubleComplex)) );
    gpuErrchk( cudaMalloc((void**)&d_z, size * sizeof(cuDoubleComplex)) );
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
__host__ void _allocateGridC3ToGPU(cuDoubleComplex *d_x, cuDoubleComplex *d_y, cuDoubleComplex *d_z,
                                  cuDoubleComplex *h_x, cuDoubleComplex *h_y, cuDoubleComplex *h_z,
                                    int size)
{
    gpuErrchk( cudaMalloc((void**)&d_x, size * sizeof(cuDoubleComplex)) );
    gpuErrchk( cudaMalloc((void**)&d_y, size * sizeof(cuDoubleComplex)) );
    gpuErrchk( cudaMalloc((void**)&d_z, size * sizeof(cuDoubleComplex)) );

    gpuErrchk( cudaMemcpy(d_x, h_x, size * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_y, h_y, size * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_y, h_z, size * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice) );
}

__host__ void _allocateGridGPUToC3(cuDoubleComplex *h_x, cuDoubleComplex *h_y, cuDoubleComplex *h_z,
                                  cuDoubleComplex *d_x, cuDoubleComplex *d_y, cuDoubleComplex *d_z,
                                    int size)
{
    gpuErrchk( cudaMemcpy(h_x, d_x, size * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(h_y, d_y, size * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(h_y, d_z, size * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost) );
}

/**
 * (PRIVATE)
 * Convert two arrays of doubles to array of cuComplex
 *
 * @param rarr Real part of complex array.
 * @param iarr Imaginary part of complex array.
 * @param carr Array of cuDoubleComplex, to be filled.
 * @param size Size of arrays.
 */
__host__ void _arrToCUDAC(double *rarr, double *iarr, cuDoubleComplex* carr, int size)
{
    for (int i=0; i<size; i++)
    {
        carr[i] = make_cuDoubleComplex(rarr[i], iarr[i]);
    }
}

__host__ void _arrC3ToCUDAC(double *r1arr, double *r2arr, double *r3arr,
                            double *i1arr, double *i2arr, double *i3arr,
                            cuDoubleComplex* c1arr, cuDoubleComplex* c2arr, cuDoubleComplex* c3arr,
                            int size)
{
    for (int i=0; i<size; i++)
    {
        c1arr[i] = make_cuDoubleComplex(r1arr[i], i1arr[i]);
        c2arr[i] = make_cuDoubleComplex(r2arr[i], i2arr[i]);
        c3arr[i] = make_cuDoubleComplex(r3arr[i], i3arr[i]);

    }
}

/**
 * (PRIVATE)
 * Convert array of cuComplex to two arrays of doubles.
 *
 * @param carr Array of cuDoubleComplex.
 * @param rarr Real part of complex array, to be filled.
 * @param iarr Imaginary part of complex array, to be filled.
 * @param size Size of arrays.
 */
__host__ void _arrCUDACToC(cuDoubleComplex* carr, double *rarr, double *iarr, int size)
{
    for (int i=0; i<size; i++)
    {
        rarr[i] = cuCreal(carr[i]);
        iarr[i] = cuCimag(carr[i]);
    }
}

__host__ void _arrCUDACToC3(cuDoubleComplex* c1arr, cuDoubleComplex* c2arr, cuDoubleComplex* c3arr,
                           double *r1arr, double *r2arr, double *r3arr,
                           double *i1arr, double *i2arr, double *i3arr,
                           int size)
{
    for (int i=0; i<size; i++)
    {
        r1arr[i] = cuCreal(c1arr[i]);
        i1arr[i] = cuCimag(c1arr[i]);

        r2arr[i] = cuCreal(c2arr[i]);
        i2arr[i] = cuCimag(c2arr[i]);

        r3arr[i] = cuCreal(c3arr[i]);
        i3arr[i] = cuCimag(c3arr[i]);
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
extern "C" void callKernel_JM(c2Bundle *res,
                                double *xt, double *yt, double *zt,
                                double *xs, double *ys, double *zs,
                                double *nxt, double *nyt, double *nzt,
                                double *rJxs, double *rJys, double *rJzs,
                                double *iJxs, double *iJys, double *iJzs,
                                double *rMxs, double *rMys, double *rMzs,
                                double *iMxs, double *iMys, double *iMzs,
                                double *area, double k, double epsilon, int gt, int gs,
                                double t_direction, int nBlocks, int nThreads)
{
    // Allocate memory for struct arrays. Then pass the pointers to receive data from gpu
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

    // Initialize and copy constant memory to device
    std::array<dim3, 2> BT;
    BT = _initCUDA(k, epsilon, gt, gs, t_direction, nBlocks, nThreads);

    // Create pointers to device arrays and allocate/copy source grid and area.
    double *d_xs, *d_ys, *d_zs, *d_A;

    gpuErrchk( cudaMalloc((void**)&d_xs, gs * sizeof(double)) );
    gpuErrchk( cudaMalloc((void**)&d_ys, gs * sizeof(double)) );
    gpuErrchk( cudaMalloc((void**)&d_zs, gs * sizeof(double)) );
    gpuErrchk( cudaMalloc((void**)&d_A, gs * sizeof(double)) );

    gpuErrchk( cudaMemcpy(d_xs, xs, gs * sizeof(double), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_ys, ys, gs * sizeof(double), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_zs, zs, gs * sizeof(double), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_A, area, gs * sizeof(double), cudaMemcpyHostToDevice) );

    // Create pointers to target grid and normal vectors
    double *d_xt, *d_yt, *d_zt, *d_nxt, *d_nyt, *d_nzt;

    // Allocate target co-ordinate and normal grids
    gpuErrchk( cudaMalloc((void**)&d_xt, gt * sizeof(double)) );
    gpuErrchk( cudaMalloc((void**)&d_yt, gt * sizeof(double)) );
    gpuErrchk( cudaMalloc((void**)&d_zt, gt * sizeof(double)) );

    gpuErrchk( cudaMemcpy(d_xt, xt, gt * sizeof(double), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_yt, yt, gt * sizeof(double), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_zt, zt, gt * sizeof(double), cudaMemcpyHostToDevice) );

    gpuErrchk( cudaMalloc((void**)&d_nxt, gt * sizeof(double)) );
    gpuErrchk( cudaMalloc((void**)&d_nyt, gt * sizeof(double)) );
    gpuErrchk( cudaMalloc((void**)&d_nzt, gt * sizeof(double)) );

    gpuErrchk( cudaMemcpy(d_nxt, nxt, gt * sizeof(double), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_nyt, nyt, gt * sizeof(double), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_nzt, nzt, gt * sizeof(double), cudaMemcpyHostToDevice) );

    cuDoubleComplex *h_Jx = new cuDoubleComplex[gs];
    cuDoubleComplex *h_Jy = new cuDoubleComplex[gs];
    cuDoubleComplex *h_Jz = new cuDoubleComplex[gs];

    cuDoubleComplex *h_Mx = new cuDoubleComplex[gs];
    cuDoubleComplex *h_My = new cuDoubleComplex[gs];
    cuDoubleComplex *h_Mz = new cuDoubleComplex[gs];

    cuDoubleComplex *d_Jx, *d_Jy, *d_Jz;
    cuDoubleComplex *d_Mx, *d_My, *d_Mz;

    _arrC3ToCUDAC(rJxs, rJys, rJzs, iJxs, iJys, iJzs,
                    h_Jx, h_Jy, h_Jz, gs);

    _arrC3ToCUDAC(rMxs, rMys, rMzs, iMxs, iMys, iMzs,
                    h_Mx, h_My, h_Mz, gs);

    // Allocate and copy J and M currents
    gpuErrchk( cudaMalloc((void**)&d_Jx, gs * sizeof(cuDoubleComplex)) );
    gpuErrchk( cudaMalloc((void**)&d_Jy, gs * sizeof(cuDoubleComplex)) );
    gpuErrchk( cudaMalloc((void**)&d_Jz, gs * sizeof(cuDoubleComplex)) );

    gpuErrchk( cudaMemcpy(d_Jx, h_Jx, gs * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_Jy, h_Jy, gs * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_Jz, h_Jz, gs * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice) );

    gpuErrchk( cudaMalloc((void**)&d_Mx, gs * sizeof(cuDoubleComplex)) );
    gpuErrchk( cudaMalloc((void**)&d_My, gs * sizeof(cuDoubleComplex)) );
    gpuErrchk( cudaMalloc((void**)&d_Mz, gs * sizeof(cuDoubleComplex)) );

    gpuErrchk( cudaMemcpy(d_Mx, h_Mx, gs * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_My, h_My, gs * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_Mz, h_Mz, gs * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice) );

    // Delete host arrays for source currents - not needed anymore
    delete h_Jx;
    delete h_Jy;
    delete h_Jz;

    delete h_Mx;
    delete h_My;
    delete h_Mz;

    cuDoubleComplex *d_Jxt, *d_Jyt, *d_Jzt;
    cuDoubleComplex *d_Mxt, *d_Myt, *d_Mzt;

    gpuErrchk( cudaMalloc((void**)&d_Jxt, gt * sizeof(cuDoubleComplex)) );
    gpuErrchk( cudaMalloc((void**)&d_Jyt, gt * sizeof(cuDoubleComplex)) );
    gpuErrchk( cudaMalloc((void**)&d_Jzt, gt * sizeof(cuDoubleComplex)) );

    gpuErrchk( cudaMalloc((void**)&d_Mxt, gt * sizeof(cuDoubleComplex)) );
    gpuErrchk( cudaMalloc((void**)&d_Myt, gt * sizeof(cuDoubleComplex)) );
    gpuErrchk( cudaMalloc((void**)&d_Mzt, gt * sizeof(cuDoubleComplex)) );

    // Create stopping event for kernel
    cudaEvent_t event;
    gpuErrchk( cudaEventCreateWithFlags(&event, cudaEventDisableTiming) );

    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;

    // Call to KERNEL 0
    printf("Starting kernel 0...\n");
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

    std::cout << "Calculation time "
            << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count()
            << " [s]" << std::endl;

    // Allocate Host arrays for J and M
    cuDoubleComplex *h_Jxt = new cuDoubleComplex[gt];
    cuDoubleComplex *h_Jyt = new cuDoubleComplex[gt];
    cuDoubleComplex *h_Jzt = new cuDoubleComplex[gt];

    cuDoubleComplex *h_Mxt = new cuDoubleComplex[gt];
    cuDoubleComplex *h_Myt = new cuDoubleComplex[gt];
    cuDoubleComplex *h_Mzt = new cuDoubleComplex[gt];

    // Copy data back from Device to Host
    gpuErrchk( cudaMemcpy(h_Jxt, d_Jxt, gt * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(h_Jyt, d_Jyt, gt * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(h_Jzt, d_Jzt, gt * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost) );

    gpuErrchk( cudaMemcpy(h_Mxt, d_Mxt, gt * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(h_Myt, d_Myt, gt * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(h_Mzt, d_Mzt, gt * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost) );

    // Free Device memory
    gpuErrchk( cudaDeviceReset() );

    _arrCUDACToC3(h_Jxt, h_Jyt, h_Jzt, res->r1x, res->r1y, res->r1z, res->i1x, res->i1y, res->i1z, gt);
    _arrCUDACToC3(h_Mxt, h_Myt, h_Mzt, res->r2x, res->r2y, res->r2z, res->i2x, res->i2y, res->i2z, gt);

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
extern "C" void callKernel_EH(c2Bundle *res, double *xt, double *yt, double *zt,
                                double *xs, double *ys, double *zs,
                                double *rJxs, double *rJys, double *rJzs,
                                double *iJxs, double *iJys, double *iJzs,
                                double *rMxs, double *rMys, double *rMzs,
                                double *iMxs, double *iMys, double *iMzs,
                                double *area, double k, double epsilon, int gt, int gs,
                                double t_direction, int nBlocks, int nThreads)
{
    // Allocate memory for struct arrays. Then pass the pointers to receive data from gpu
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

    // Initialize and copy constant memory to device
    std::array<dim3, 2> BT;
    BT = _initCUDA(k, epsilon, gt, gs, t_direction, nBlocks, nThreads);

    // Create pointers to device arrays and allocate/copy source grid and area.
    double *d_xs, *d_ys, *d_zs, *d_A;

    gpuErrchk( cudaMalloc((void**)&d_xs, gs * sizeof(double)) );
    gpuErrchk( cudaMalloc((void**)&d_ys, gs * sizeof(double)) );
    gpuErrchk( cudaMalloc((void**)&d_zs, gs * sizeof(double)) );
    gpuErrchk( cudaMalloc((void**)&d_A, gs * sizeof(double)) );

    gpuErrchk( cudaMemcpy(d_xs, xs, gs * sizeof(double), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_ys, ys, gs * sizeof(double), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_zs, zs, gs * sizeof(double), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_A, area, gs * sizeof(double), cudaMemcpyHostToDevice) );

    // Create pointers to target grid and normal vectors
    double *d_xt, *d_yt, *d_zt;
    gpuErrchk( cudaMalloc((void**)&d_xt, gt * sizeof(double)) );
    gpuErrchk( cudaMalloc((void**)&d_yt, gt * sizeof(double)) );
    gpuErrchk( cudaMalloc((void**)&d_zt, gt * sizeof(double)) );

    gpuErrchk( cudaMemcpy(d_xt, xt, gt * sizeof(double), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_yt, yt, gt * sizeof(double), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_zt, zt, gt * sizeof(double), cudaMemcpyHostToDevice) );

    cuDoubleComplex *h_Jx = new cuDoubleComplex[gs];
    cuDoubleComplex *h_Jy = new cuDoubleComplex[gs];
    cuDoubleComplex *h_Jz = new cuDoubleComplex[gs];

    cuDoubleComplex *h_Mx = new cuDoubleComplex[gs];
    cuDoubleComplex *h_My = new cuDoubleComplex[gs];
    cuDoubleComplex *h_Mz = new cuDoubleComplex[gs];

    _arrC3ToCUDAC(rJxs, rJys, rJzs, iJxs, iJys, iJzs,
                    h_Jx, h_Jy, h_Jz, gs);

    _arrC3ToCUDAC(rMxs, rMys, rMzs, iMxs, iMys, iMzs,
                    h_Mx, h_My, h_Mz, gs);

    cuDoubleComplex *d_Jx, *d_Jy, *d_Jz;
    cuDoubleComplex *d_Mx, *d_My, *d_Mz;

    // Allocate and copy J and M currents
    gpuErrchk( cudaMalloc((void**)&d_Jx, gs * sizeof(cuDoubleComplex)) );
    gpuErrchk( cudaMalloc((void**)&d_Jy, gs * sizeof(cuDoubleComplex)) );
    gpuErrchk( cudaMalloc((void**)&d_Jz, gs * sizeof(cuDoubleComplex)) );

    gpuErrchk( cudaMemcpy(d_Jx, h_Jx, gs * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_Jy, h_Jy, gs * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_Jz, h_Jz, gs * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice) );

    gpuErrchk( cudaMalloc((void**)&d_Mx, gs * sizeof(cuDoubleComplex)) );
    gpuErrchk( cudaMalloc((void**)&d_My, gs * sizeof(cuDoubleComplex)) );
    gpuErrchk( cudaMalloc((void**)&d_Mz, gs * sizeof(cuDoubleComplex)) );

    gpuErrchk( cudaMemcpy(d_Mx, h_Mx, gs * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_My, h_My, gs * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_Mz, h_Mz, gs * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice) );

    // Delete host arrays for source currents - not needed anymore
    delete h_Jx;
    delete h_Jy;
    delete h_Jz;

    delete h_Mx;
    delete h_My;
    delete h_Mz;

    cuDoubleComplex *d_Ext, *d_Eyt, *d_Ezt;
    cuDoubleComplex *d_Hxt, *d_Hyt, *d_Hzt;

    gpuErrchk( cudaMalloc((void**)&d_Ext, gt * sizeof(cuDoubleComplex)) );
    gpuErrchk( cudaMalloc((void**)&d_Eyt, gt * sizeof(cuDoubleComplex)) );
    gpuErrchk( cudaMalloc((void**)&d_Ezt, gt * sizeof(cuDoubleComplex)) );

    gpuErrchk( cudaMalloc((void**)&d_Hxt, gt * sizeof(cuDoubleComplex)) );
    gpuErrchk( cudaMalloc((void**)&d_Hyt, gt * sizeof(cuDoubleComplex)) );
    gpuErrchk( cudaMalloc((void**)&d_Hzt, gt * sizeof(cuDoubleComplex)) );

    // Create stopping event for kernel
    cudaEvent_t event;
    gpuErrchk( cudaEventCreateWithFlags(&event, cudaEventDisableTiming) );

    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;

    // Call to KERNEL 1
    printf("Starting kernel 1...\n");
    begin = std::chrono::steady_clock::now();
    GpropagateBeam_1<<<BT[0], BT[1]>>>(d_xs, d_ys, d_zs,
                                d_A, d_xt, d_yt, d_zt,
                                d_Jx, d_Jy, d_Jz,
                                d_Mx, d_My, d_Mz,
                                d_Ext, d_Eyt, d_Ezt,
                                d_Hxt, d_Hyt, d_Hzt);

    gpuErrchk( cudaDeviceSynchronize() );

    end = std::chrono::steady_clock::now();

    std::cout << "Calculation time "
            << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count()
            << " [s]" << std::endl;

    // Allocate Host arrays for E and H
    cuDoubleComplex *h_Ext = new cuDoubleComplex[gt];
    cuDoubleComplex *h_Eyt = new cuDoubleComplex[gt];
    cuDoubleComplex *h_Ezt = new cuDoubleComplex[gt];

    cuDoubleComplex *h_Hxt = new cuDoubleComplex[gt];
    cuDoubleComplex *h_Hyt = new cuDoubleComplex[gt];
    cuDoubleComplex *h_Hzt = new cuDoubleComplex[gt];

    // Copy data back from Device to Host
    gpuErrchk( cudaMemcpy(h_Ext, d_Ext, gt * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(h_Eyt, d_Eyt, gt * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(h_Ezt, d_Ezt, gt * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost) );

    gpuErrchk( cudaMemcpy(h_Hxt, d_Hxt, gt * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(h_Hyt, d_Hyt, gt * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(h_Hzt, d_Hzt, gt * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost) );

    // Free Device memory
    gpuErrchk( cudaDeviceReset() );

    _arrCUDACToC3(h_Ext, h_Eyt, h_Ezt, res->r1x, res->r1y, res->r1z, res->i1x, res->i1y, res->i1z, gt);
    _arrCUDACToC3(h_Hxt, h_Hyt, h_Hzt, res->r2x, res->r2y, res->r2z, res->i2x, res->i2y, res->i2z, gt);

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
extern "C" void callKernel_JMEH(c4Bundle *res, double *xt, double *yt, double *zt,
                                double *xs, double *ys, double *zs,
                                double *nxt, double *nyt, double *nzt,
                                double *rJxs, double *rJys, double *rJzs,
                                double *iJxs, double *iJys, double *iJzs,
                                double *rMxs, double *rMys, double *rMzs,
                                double *iMxs, double *iMys, double *iMzs,
                                double *area, double k, double epsilon, int gt, int gs,
                                double t_direction, int nBlocks, int nThreads)
{
    // Allocate memory for struct arrays. Then pass the pointers to receive data from gpu
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

    // Initialize and copy constant memory to device
    std::array<dim3, 2> BT;
    BT = _initCUDA(k, epsilon, gt, gs, t_direction, nBlocks, nThreads);

    // Create pointers to device arrays and allocate/copy source grid and area.
    double *d_xs, *d_ys, *d_zs, *d_A;
    _allocateGridR3ToGPU(d_xs, d_ys, d_zs, xs, ys, zs, gs);
    _allocateGridR1ToGPU(d_A, area, gs);

    // Create pointers to target grid and normal vectors
    double *d_xt, *d_yt, *d_zt, *d_nxt, *d_nyt, *d_nzt;
    _allocateGridR3ToGPU(d_xt, d_yt, d_zt, xt, yt, zt, gt);
    _allocateGridR3ToGPU(d_nxt, d_nyt, d_nzt, nxt, nyt, nzt, gt);

    cuDoubleComplex *h_Jx = new cuDoubleComplex[gs];
    cuDoubleComplex *h_Jy = new cuDoubleComplex[gs];
    cuDoubleComplex *h_Jz = new cuDoubleComplex[gs];

    cuDoubleComplex *h_Mx = new cuDoubleComplex[gs];
    cuDoubleComplex *h_My = new cuDoubleComplex[gs];
    cuDoubleComplex *h_Mz = new cuDoubleComplex[gs];

    cuDoubleComplex *d_Jx, *d_Jy, *d_Jz;
    cuDoubleComplex *d_Mx, *d_My, *d_Mz;

    _arrC3ToCUDAC(rJxs, rJys, rJzs, iJxs, iJys, iJzs,
                    h_Jx, h_Jy, h_Jz, gs);

    _arrC3ToCUDAC(rMxs, rMys, rMzs, iMxs, iMys, iMzs,
                    h_Mx, h_My, h_Mz, gs);

    // Delete host arrays for source currents
    delete h_Jx;
    delete h_Jy;
    delete h_Jz;

    delete h_Mx;
    delete h_My;
    delete h_Mz;

    _allocateGridC3ToGPU(d_Jx, d_Jy, d_Jz, h_Jx, h_Jy, h_Jz, gs);
    _allocateGridC3ToGPU(d_Mx, d_My, d_Mz, h_Mx, h_My, h_Mz, gs);

    cuDoubleComplex *d_Jxt, *d_Jyt, *d_Jzt;
    cuDoubleComplex *d_Mxt, *d_Myt, *d_Mzt;
    cuDoubleComplex *d_Ext, *d_Eyt, *d_Ezt;
    cuDoubleComplex *d_Hxt, *d_Hyt, *d_Hzt;

    _allocateGridC3(d_Jxt, d_Jyt, d_Jzt, gs);
    _allocateGridC3(d_Mxt, d_Myt, d_Mzt, gs);
    _allocateGridC3(d_Ext, d_Eyt, d_Ezt, gs);
    _allocateGridC3(d_Hxt, d_Hyt, d_Hzt, gs);

    // Create stopping event for kernel
    cudaEvent_t event;
    gpuErrchk( cudaEventCreateWithFlags(&event, cudaEventDisableTiming) );

    // Call to KERNEL 2
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

    // Allocate Host arrays for J and M
    cuDoubleComplex *h_Jxt = new cuDoubleComplex[gt];
    cuDoubleComplex *h_Jyt = new cuDoubleComplex[gt];
    cuDoubleComplex *h_Jzt = new cuDoubleComplex[gt];

    cuDoubleComplex *h_Mxt = new cuDoubleComplex[gt];
    cuDoubleComplex *h_Myt = new cuDoubleComplex[gt];
    cuDoubleComplex *h_Mzt = new cuDoubleComplex[gt];

    // Allocate Host arrays for E and H
    cuDoubleComplex *h_Ext = new cuDoubleComplex[gt];
    cuDoubleComplex *h_Eyt = new cuDoubleComplex[gt];
    cuDoubleComplex *h_Ezt = new cuDoubleComplex[gt];

    cuDoubleComplex *h_Hxt = new cuDoubleComplex[gt];
    cuDoubleComplex *h_Hyt = new cuDoubleComplex[gt];
    cuDoubleComplex *h_Hzt = new cuDoubleComplex[gt];

    // Copy data back from Device to Host
    _allocateGridGPUToC3(h_Jxt, h_Jyt, h_Jzt, d_Jxt, d_Jyt, d_Jzt, gt);
    _allocateGridGPUToC3(h_Mxt, h_Myt, h_Mzt, d_Mxt, d_Myt, d_Mzt, gt);
    _allocateGridGPUToC3(h_Ext, h_Eyt, h_Ezt, d_Ext, d_Eyt, d_Ezt, gt);
    _allocateGridGPUToC3(h_Hxt, h_Hyt, h_Hzt, d_Hxt, d_Hyt, d_Hzt, gt);

    // Free Device memory
    gpuErrchk( cudaDeviceReset() );

    _arrCUDACToC3(h_Jxt, h_Jyt, h_Jzt, res->r1x, res->r1y, res->r1z, res->i1x, res->i1y, res->i1z, gt);
    _arrCUDACToC3(h_Mxt, h_Myt, h_Mzt, res->r2x, res->r2y, res->r2z, res->i2x, res->i2y, res->i2z, gt);

    _arrCUDACToC3(h_Ext, h_Eyt, h_Ezt, res->r3x, res->r3y, res->r3z, res->i3x, res->i3y, res->i3z, gt);
    _arrCUDACToC3(h_Hxt, h_Hyt, h_Hzt, res->r4x, res->r4y, res->r4z, res->i4x, res->i4y, res->i4z, gt);

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
extern "C" void callKernel_EHP(c2rBundle *res, double *xt, double *yt, double *zt,
                                double *xs, double *ys, double *zs,
                                double *nxt, double *nyt, double *nzt,
                                double *rJxs, double *rJys, double *rJzs,
                                double *iJxs, double *iJys, double *iJzs,
                                double *rMxs, double *rMys, double *rMzs,
                                double *iMxs, double *iMys, double *iMzs,
                                double *area, double k, double epsilon, int gt, int gs,
                                double t_direction, int nBlocks, int nThreads)
{
    // Allocate memory for struct arrays. Then pass the pointers to receive data from gpu
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

    // Initialize and copy constant memory to device
    std::array<dim3, 2> BT;
    BT = _initCUDA(k, epsilon, gt, gs, t_direction, nBlocks, nThreads);

    // Create pointers to device arrays and allocate/copy source grid and area.
    double *d_xs, *d_ys, *d_zs, *d_A;
    _allocateGridR3ToGPU(d_xs, d_ys, d_zs, xs, ys, zs, gs);
    _allocateGridR1ToGPU(d_A, area, gs);

    // Create pointers to target grid and normal vectors
    double *d_xt, *d_yt, *d_zt, *d_nxt, *d_nyt, *d_nzt;
    _allocateGridR3ToGPU(d_xt, d_yt, d_zt, xt, yt, zt, gt);
    _allocateGridR3ToGPU(d_nxt, d_nyt, d_nzt, nxt, nyt, nzt, gt);

    cuDoubleComplex *h_Jx = new cuDoubleComplex[gs];
    cuDoubleComplex *h_Jy = new cuDoubleComplex[gs];
    cuDoubleComplex *h_Jz = new cuDoubleComplex[gs];

    cuDoubleComplex *h_Mx = new cuDoubleComplex[gs];
    cuDoubleComplex *h_My = new cuDoubleComplex[gs];
    cuDoubleComplex *h_Mz = new cuDoubleComplex[gs];

    cuDoubleComplex *d_Jx, *d_Jy, *d_Jz;
    cuDoubleComplex *d_Mx, *d_My, *d_Mz;

    _arrC3ToCUDAC(rJxs, rJys, rJzs, iJxs, iJys, iJzs,
                    h_Jx, h_Jy, h_Jz, gs);

    _arrC3ToCUDAC(rMxs, rMys, rMzs, iMxs, iMys, iMzs,
                    h_Mx, h_My, h_Mz, gs);

    // Delete host arrays for source currents
    delete h_Jx;
    delete h_Jy;
    delete h_Jz;

    delete h_Mx;
    delete h_My;
    delete h_Mz;

    _allocateGridC3ToGPU(d_Jx, d_Jy, d_Jz, h_Jx, h_Jy, h_Jz, gs);
    _allocateGridC3ToGPU(d_Mx, d_My, d_Mz, h_Mx, h_My, h_Mz, gs);

    cuDoubleComplex *d_Ext, *d_Eyt, *d_Ezt;
    cuDoubleComplex *d_Hxt, *d_Hyt, *d_Hzt;
    double *d_Prxt, *d_Pryt, *d_Przt;

    _allocateGridC3(d_Ext, d_Eyt, d_Ezt, gs);
    _allocateGridC3(d_Hxt, d_Hyt, d_Hzt, gs);
    _allocateGridR3(d_Prxt, d_Pryt, d_Przt, gs);

    // Create stopping event for kernel
    cudaEvent_t event;
    gpuErrchk( cudaEventCreateWithFlags(&event, cudaEventDisableTiming) );

    // Call to KERNEL 3
    GpropagateBeam_3<<<BT[0], BT[1]>>>(d_xs, d_ys, d_zs,
                                d_A, d_xt, d_yt, d_zt,
                                d_nxt, d_nyt, d_nzt,
                                d_Jx, d_Jy, d_Jz,
                                d_Mx, d_My, d_Mz,
                                d_Ext, d_Eyt, d_Ezt,
                                d_Hxt, d_Hyt, d_Hzt,
                                d_Prxt, d_Pryt, d_Przt);

    gpuErrchk( cudaDeviceSynchronize() );

    // Allocate Host arrays for E, H and P
    cuDoubleComplex *h_Ext = new cuDoubleComplex[gt];
    cuDoubleComplex *h_Eyt = new cuDoubleComplex[gt];
    cuDoubleComplex *h_Ezt = new cuDoubleComplex[gt];

    cuDoubleComplex *h_Hxt = new cuDoubleComplex[gt];
    cuDoubleComplex *h_Hyt = new cuDoubleComplex[gt];
    cuDoubleComplex *h_Hzt = new cuDoubleComplex[gt];

    double *h_Prxt = new double[gt];
    double *h_Pryt = new double[gt];
    double *h_Przt = new double[gt];

    // Copy data back from Device to Host
    _allocateGridGPUToC3(h_Ext, h_Eyt, h_Ezt, d_Ext, d_Eyt, d_Ezt, gt);
    _allocateGridGPUToC3(h_Hxt, h_Hyt, h_Hzt, d_Hxt, d_Hyt, d_Hzt, gt);
    _allocateGridGPUToR3(h_Prxt, h_Pryt, h_Przt, d_Prxt, d_Pryt, d_Przt, gt);

    // Free Device memory
    gpuErrchk( cudaDeviceReset() );

    _arrCUDACToC3(h_Ext, h_Eyt, h_Ezt, res->r1x, res->r1y, res->r1z, res->i1x, res->i1y, res->i1z, gt);
    _arrCUDACToC3(h_Hxt, h_Hyt, h_Hzt, res->r2x, res->r2y, res->r2z, res->i2x, res->i2y, res->i2z, gt);

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
extern "C" void callKernel_FF(arrC3 *res, double *xt, double *yt,
                                double *xs, double *ys, double *zs,
                                double *rJxs, double *rJys, double *rJzs,
                                double *iJxs, double *iJys, double *iJzs,
                                double *rMxs, double *rMys, double *rMzs,
                                double *iMxs, double *iMys, double *iMzs,
                                double *area, double k, double epsilon, int gt, int gs,
                                double t_direction, int nBlocks, int nThreads)
{
    // Initialize and copy constant memory to device
    std::array<dim3, 2> BT;
    BT = _initCUDA(k, epsilon, gt, gs, t_direction, nBlocks, nThreads);

    // Create pointers to device arrays and allocate/copy source grid and area.
    double *d_xs, *d_ys, *d_zs, *d_A;
    _allocateGridR3ToGPU(d_xs, d_ys, d_zs, xs, ys, zs, gs);
    _allocateGridR1ToGPU(d_A, area, gs);

    // Create pointers to target grid and normal vectors
    double *d_xt, *d_yt;
    _allocateGridR2ToGPU(d_xt, d_yt, xt, yt, gt);

    cuDoubleComplex *h_Jx, *h_Jy, *h_Jz;
    cuDoubleComplex *h_Mx, *h_My, *h_Mz;

    cuDoubleComplex *d_Jx, *d_Jy, *d_Jz;
    cuDoubleComplex *d_Mx, *d_My, *d_Mz;

    _arrC3ToCUDAC(rJxs, rJys, rJzs, iJxs, iJys, iJzs,
                    h_Jx, h_Jy, h_Jz, gs);

    _arrC3ToCUDAC(rMxs, rMys, rMzs, iMxs, iMys, iMzs,
                    h_Mx, h_My, h_Mz, gs);

    _allocateGridC3ToGPU(d_Jx, d_Jy, d_Jz, h_Jx, h_Jy, h_Jz, gs);
    _allocateGridC3ToGPU(d_Mx, d_My, d_Mz, h_Mx, h_My, h_Mz, gs);

    cuDoubleComplex *d_Ext, *d_Eyt, *d_Ezt;

    _allocateGridC3(d_Ext, d_Eyt, d_Ezt, gs);

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
    cuDoubleComplex h_Ext[gt], h_Eyt[gt], h_Ezt[gt];

    // Copy data back from Device to Host
    _allocateGridGPUToC3(h_Ext, h_Eyt, h_Ezt, d_Ext, d_Eyt, d_Ezt, gt);

    // Free Device memory
    gpuErrchk( cudaDeviceReset() );

    // Create arrays of doubles, for transferring back output
    double *rExt, *rEyt, *rEzt, *iExt, *iEyt, *iEzt;

    _arrCUDACToC3(h_Ext, h_Eyt, h_Ezt, rExt, rEyt, rEzt, iExt, iEyt, iEzt, gt);

    arrC3 *out = new arrC3;
    if (out == NULL) exit (1); //EXIT_FAILURE

    fill_arrC3(out, rExt, rEyt, rEzt, iExt, iEyt, iEzt);

    return out;
}
*/
