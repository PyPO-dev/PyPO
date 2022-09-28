#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <string>
#include <iterator>
#include <array>
#include <cmath>

#include <cuda.h>
#include <cuComplex.h>

#include "GDataHandlerf.h"
#include "GUtilsf.h"

#define CSIZE 10

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

/**
 * Function to calculate complex exponential with
 * CUDA type cuFloatComplex.
 * 
 * @param z Complex number.
 */
__device__ __inline__ cuFloatComplex my_cexp(cuFloatComplex z)
{
    cuFloatComplex res;
    float t = exp(z.x);
    float ys = sin(z.y);
    float yc = cos(z.y);
    res = cuCmulf(make_cuFloatComplex(t, 0.), make_cuFloatComplex(yc, ys));
    return res;
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
        
        Green = cuCmulf(cuCdivf(my_cexp(cuCmulf(con[6], cuCmulf(con[7], cuCmulf(con[0], rc)))), (cuCmulf(con[9], cuCmulf(con[4], rc)))), cuCmulf(d_Ac, con[7]));

        for( int n=0; n<3; n++)
        {
            e_field[n] = cuCsubf(e_field[n], cuCmulf(cuCsubf(cuCmulf(omega, cuCmulf(con[2], e_vec_thing[n])), k_out_ms[n]), Green));
            h_field[n] = cuCsubf(h_field[n], cuCmulf(cuCaddf(cuCmulf(omega, cuCmulf(con[1], h_vec_thing[n])), k_out_js[n]), Green));
        }  
    }

    d_ei[0] = e_field[0];
    d_ei[1] = e_field[1];
    d_ei[2] = e_field[2];
    
    d_hi[0] = h_field[0];
    d_hi[1] = h_field[1];
    d_hi[2] = h_field[2];

    
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
            e_out_h_r[n] = cuCrealf(temp2[n]);                      // e_out_h_r
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
        conja(d_ei, temp1);                        // h_conj
        ext(d_hi, temp1, temp2);                  // e_out_h
        
        for (int n=0; n<3; n++) 
        {
            e_out_h_r[n] = cuCrealf(temp2[n]);                      // e_out_h_r
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
                                float *d_Prxt, float *d_Pryt, float *d_Przt,
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
        conja(d_ei, temp1);                        // h_conj
        ext(d_hi, temp1, temp2);                  // e_out_h
        
        for (int n=0; n<3; n++) 
        {
            e_out_h_r[n] = cuCrealf(temp2[n]);                      // e_out_h_r
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

int main(int argc, char *argv [])
{
    int numThreads  = atoi(argv[1]); // Number of GPU threads per block
    int numBlocks   = atoi(argv[2]); // Number of execution blocks
    float k        = atof(argv[3]); // Wavenumber of field to be propagated
    int toPrint     = atoi(argv[4]); // 0 for printing J and M, 1 for E and H and 2 for all fields
    
    float epsilon  = atof(argv[5]); // Relative electric permeability
    int prop_mode   = atoi(argv[6]); // Whether to propagate to surface or to far-field
    float t_direction = atof(argv[7]); // Whether to propagate forward or back in time
    
    int gridsize_s  = atoi(argv[8]); // Source gridsize, flattened
    int gridsize_t  = atoi(argv[9]); // Target gridsize, flattened
    
    float M_PIf = 3.1415926;
    float C_L = 2.9979245; // mm s^-1
    float MU_0 = 1.2566370e-3; // kg mm s^-2 A^-2
    float EPS_VAC = 1 / (MU_0 * C_L*C_L);
    float ZETA_0_INV = 1 / (C_L * MU_0);
    
    // Calculate nr of blocks per grid and nr of threads per block
    dim3 nrb(numBlocks); dim3 nrt(numThreads);
    
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
    cudaMemcpyToSymbol(con, &_con, CSIZE * sizeof(cuFloatComplex));
    cudaMemcpyToSymbol(eye, &_eye, sizeof(_eye));
    cudaMemcpyToSymbol(g_s, &gridsize_s, sizeof(int));
    cudaMemcpyToSymbol(g_t, &gridsize_t, sizeof(int));

    std::string source = "s"; 
    std::string target = "t"; 
    
    GDataHandlerf ghandler;
    
    float source_area[gridsize_s];
    
    // Obtain source area elements
    ghandler.cppToCUDA_area(source_area);
    
    std::array<float*, 3> grid_source = ghandler.cppToCUDA_3DGrid(source);
    std::array<float*, 3> grid_target3D;
    std::array<float*, 2> grid_target2D;
    std::array<float*, 3> norm_target;
    
    // Allocate source grid and area on Device
    float *d_xs; cudaMalloc( (void**)&d_xs, gridsize_s * sizeof(float) );
    float *d_ys; cudaMalloc( (void**)&d_ys, gridsize_s * sizeof(float) );
    float *d_zs; cudaMalloc( (void**)&d_zs, gridsize_s * sizeof(float) );
    float *d_A; cudaMalloc( (void**)&d_A, gridsize_s * sizeof(float) );
    
    // Copy data from Host to Device
    cudaMemcpy(d_xs, grid_source[0], gridsize_s * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ys, grid_source[1], gridsize_s * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_zs, grid_source[2], gridsize_s * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, source_area, gridsize_s * sizeof(float), cudaMemcpyHostToDevice);
    
    // Declare pointers to Device arrays. No return for kernel calls
    float *d_xt; float *d_yt; float *d_zt; float *d_nxt; float *d_nyt; float *d_nzt;
    
    if (prop_mode == 0)
    {
        // Convert .txt files to CUDA arrays
        grid_target3D = ghandler.cppToCUDA_3DGrid(target);
        norm_target = ghandler.cppToCUDA_3Dnormals();
        
        // Allocate memory on Device for 3D grids and normals
        cudaMalloc( (void**)&d_xt, gridsize_t * sizeof(float) );
        cudaMalloc( (void**)&d_yt, gridsize_t * sizeof(float) );
        cudaMalloc( (void**)&d_zt, gridsize_t * sizeof(float) );
        
        cudaMalloc( (void**)&d_nxt, gridsize_t * sizeof(float) );
        cudaMalloc( (void**)&d_nyt, gridsize_t * sizeof(float) );
        cudaMalloc( (void**)&d_nzt, gridsize_t * sizeof(float) );
        
        // Copy grids and normals from Host to Device
        cudaMemcpy(d_xt, grid_target3D[0], gridsize_t * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_yt, grid_target3D[1], gridsize_t * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_zt, grid_target3D[2], gridsize_t * sizeof(float), cudaMemcpyHostToDevice);
        
        cudaMemcpy(d_nxt, norm_target[0], gridsize_t * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_nyt, norm_target[1], gridsize_t * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_nzt, norm_target[2], gridsize_t * sizeof(float), cudaMemcpyHostToDevice);
    }
    
    else if (prop_mode == 1)
    {
        // Convert .txt files to CUDA arrays
        grid_target2D = ghandler.cppToCUDA_2DGrid();
        
        // Allocate memory on Device for 2D grids
        cudaMalloc( (void**)&d_xt, gridsize_t * sizeof(float) );
        cudaMalloc( (void**)&d_yt, gridsize_t * sizeof(float) );
        
        // Copy to GPU from Host
        cudaMemcpy(d_xt, grid_target2D[0], gridsize_t * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_yt, grid_target2D[1], gridsize_t * sizeof(float), cudaMemcpyHostToDevice);
    }
    
    // Read source currents from .txt and convert to CUDA array
    std::array<cuFloatComplex*, 3> Js = ghandler.cppToCUDA_Js();
    std::array<cuFloatComplex*, 3> Ms = ghandler.cppToCUDA_Ms();

    // Allocate memory on Device for source currents
    cuFloatComplex *d_Jx; cudaMalloc( (void**)&d_Jx, gridsize_s * sizeof(cuFloatComplex) );
    cuFloatComplex *d_Jy; cudaMalloc( (void**)&d_Jy, gridsize_s * sizeof(cuFloatComplex) );
    cuFloatComplex *d_Jz; cudaMalloc( (void**)&d_Jz, gridsize_s * sizeof(cuFloatComplex) );
    
    cuFloatComplex *d_Mx; cudaMalloc( (void**)&d_Mx, gridsize_s * sizeof(cuFloatComplex) );
    cuFloatComplex *d_My; cudaMalloc( (void**)&d_My, gridsize_s * sizeof(cuFloatComplex) );
    cuFloatComplex *d_Mz; cudaMalloc( (void**)&d_Mz, gridsize_s * sizeof(cuFloatComplex) );
    
    // Copy source currents from Host to Device
    cudaMemcpy(d_Jx, Js[0], gridsize_s * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Jy, Js[1], gridsize_s * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Jz, Js[2], gridsize_s * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
        
    cudaMemcpy(d_Mx, Ms[0], gridsize_s * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_My, Ms[1], gridsize_s * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Mz, Ms[2], gridsize_s * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
    
    
    // Create Device arrays for storing the data and call the kernel
    // Which kernel is called is determined by toPrint
    if (toPrint == 0)
    {
        // Allocate memory for J and M arrays on Device
        cuFloatComplex *d_Jxt; cudaMalloc( (void**)&d_Jxt, gridsize_t * sizeof(cuFloatComplex) );
        cuFloatComplex *d_Jyt; cudaMalloc( (void**)&d_Jyt, gridsize_t * sizeof(cuFloatComplex) );
        cuFloatComplex *d_Jzt; cudaMalloc( (void**)&d_Jzt, gridsize_t * sizeof(cuFloatComplex) );
        
        cuFloatComplex *d_Mxt; cudaMalloc( (void**)&d_Mxt, gridsize_t * sizeof(cuFloatComplex) );
        cuFloatComplex *d_Myt; cudaMalloc( (void**)&d_Myt, gridsize_t * sizeof(cuFloatComplex) );
        cuFloatComplex *d_Mzt; cudaMalloc( (void**)&d_Mzt, gridsize_t * sizeof(cuFloatComplex) );

        // Call to KERNEL 0
        GpropagateBeam_0<<<nrb, nrt>>>(d_xs, d_ys, d_zs, 
                                   d_A, d_xt, d_yt, d_zt,
                                   d_nxt, d_nyt, d_nzt,
                                   d_Jx, d_Jy, d_Jz,
                                   d_Mx, d_My, d_Mz,
                                   d_Jxt, d_Jyt, d_Jzt,
                                   d_Mxt, d_Myt, d_Mzt);
        cudaDeviceSynchronize();
        
        // Allocate, on stackframe, Host arrays for J and M
        cuFloatComplex h_Jxt[gridsize_t];
        cuFloatComplex h_Jyt[gridsize_t];
        cuFloatComplex h_Jzt[gridsize_t];
        
        cuFloatComplex h_Mxt[gridsize_t];
        cuFloatComplex h_Myt[gridsize_t];
        cuFloatComplex h_Mzt[gridsize_t];
        
        // Copy content of Device J,M arrays into Host J,M arrays
        cudaMemcpy(h_Jxt, d_Jxt, gridsize_t * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_Jyt, d_Jyt, gridsize_t * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_Jzt, d_Jzt, gridsize_t * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
        
        cudaMemcpy(h_Mxt, d_Mxt, gridsize_t * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_Myt, d_Myt, gridsize_t * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_Mzt, d_Mzt, gridsize_t * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
        
        // Free Device memory
        cudaDeviceReset();
        
        // Pack CUDA arrays into cpp array for processing
        std::array<cuFloatComplex*, 3> CJ;
        std::array<cuFloatComplex*, 3> CM;
        
        // Fill the C++ std::array with C-style arrays
        CJ[0] = h_Jxt;
        CJ[1] = h_Jyt;
        CJ[2] = h_Jzt;
        
        CM[0] = h_Mxt;
        CM[1] = h_Myt;
        CM[2] = h_Mzt;
        
        // Convert the CUDA style arrays to format compatible with CPU functions
        std::vector<std::array<std::complex<float>, 3>> Jt = ghandler.CUDAToCpp_C(CJ, gridsize_t);
        std::vector<std::array<std::complex<float>, 3>> Mt = ghandler.CUDAToCpp_C(CM, gridsize_t);

        std::string Jt_file = "Jt";
        std::string Mt_file = "Mt";
        
        // Write using standard CPU DataHandler object
        ghandler.dh.writeOutC(Jt, Jt_file);
        ghandler.dh.writeOutC(Mt, Mt_file);
    }

    else if (toPrint == 1)
    {
        // Allocate memory for E and H arrays on Device
        cuFloatComplex *d_Ext; cudaMalloc( (void**)&d_Ext, gridsize_t * sizeof(cuFloatComplex) );
        cuFloatComplex *d_Eyt; cudaMalloc( (void**)&d_Eyt, gridsize_t * sizeof(cuFloatComplex) );
        cuFloatComplex *d_Ezt; cudaMalloc( (void**)&d_Ezt, gridsize_t * sizeof(cuFloatComplex) );
        
        cuFloatComplex *d_Hxt; cudaMalloc( (void**)&d_Hxt, gridsize_t * sizeof(cuFloatComplex) );
        cuFloatComplex *d_Hyt; cudaMalloc( (void**)&d_Hyt, gridsize_t * sizeof(cuFloatComplex) );
        cuFloatComplex *d_Hzt; cudaMalloc( (void**)&d_Hzt, gridsize_t * sizeof(cuFloatComplex) );
        
        // Call to KERNEL 1
        GpropagateBeam_1<<<nrb, nrt>>>(d_xs, d_ys, d_zs, 
                                   d_A, d_xt, d_yt, d_zt,
                                   d_Jx, d_Jy, d_Jz,
                                   d_Mx, d_My, d_Mz,
                                   d_Ext, d_Eyt, d_Ezt,
                                   d_Hxt, d_Hyt, d_Hzt);
        cudaDeviceSynchronize();
        
        // Allocate, on stackframe, Host arrays for E and H
        cuFloatComplex h_Ext[gridsize_t];
        cuFloatComplex h_Eyt[gridsize_t];
        cuFloatComplex h_Ezt[gridsize_t];
        
        cuFloatComplex h_Hxt[gridsize_t];
        cuFloatComplex h_Hyt[gridsize_t];
        cuFloatComplex h_Hzt[gridsize_t];
        
        // Copy content of Device E,H arrays into Host E,H arrays
        cudaMemcpy(h_Ext, d_Ext, gridsize_t * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_Eyt, d_Eyt, gridsize_t * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_Ezt, d_Ezt, gridsize_t * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
        
        cudaMemcpy(h_Hxt, d_Hxt, gridsize_t * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_Hyt, d_Hyt, gridsize_t * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_Hzt, d_Hzt, gridsize_t * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
        
        // Free Device memory
        cudaDeviceReset();
        
        // Pack CUDA arrays into cpp array for processing
        std::array<cuFloatComplex*, 3> CE;
        std::array<cuFloatComplex*, 3> CH;
        
        // Fill the C++ std::array with C-style arrays
        CE[0] = h_Ext;
        CE[1] = h_Eyt;
        CE[2] = h_Ezt;
        
        CH[0] = h_Hxt;
        CH[1] = h_Hyt;
        CH[2] = h_Hzt;
        
        // Convert the CUDA style arrays to format compatible with CPU functions
        std::vector<std::array<std::complex<float>, 3>> Et = ghandler.CUDAToCpp_C(CE, gridsize_t);
        std::vector<std::array<std::complex<float>, 3>> Ht = ghandler.CUDAToCpp_C(CH, gridsize_t);

        std::string Et_file = "Et";
        std::string Ht_file = "Ht";
        
        // Write using standard CPU DataHandler object
        ghandler.dh.writeOutC(Et, Et_file);
        ghandler.dh.writeOutC(Ht, Ht_file);
    }
    
    else if (toPrint == 2)
    {
        // Allocate memory for J, M, E and H arrays on Device
        cuFloatComplex *d_Jxt; cudaMalloc( (void**)&d_Jxt, gridsize_t * sizeof(cuFloatComplex) );
        cuFloatComplex *d_Jyt; cudaMalloc( (void**)&d_Jyt, gridsize_t * sizeof(cuFloatComplex) );
        cuFloatComplex *d_Jzt; cudaMalloc( (void**)&d_Jzt, gridsize_t * sizeof(cuFloatComplex) );
        
        cuFloatComplex *d_Mxt; cudaMalloc( (void**)&d_Mxt, gridsize_t * sizeof(cuFloatComplex) );
        cuFloatComplex *d_Myt; cudaMalloc( (void**)&d_Myt, gridsize_t * sizeof(cuFloatComplex) );
        cuFloatComplex *d_Mzt; cudaMalloc( (void**)&d_Mzt, gridsize_t * sizeof(cuFloatComplex) );
        
        cuFloatComplex *d_Ext; cudaMalloc( (void**)&d_Ext, gridsize_t * sizeof(cuFloatComplex) );
        cuFloatComplex *d_Eyt; cudaMalloc( (void**)&d_Eyt, gridsize_t * sizeof(cuFloatComplex) );
        cuFloatComplex *d_Ezt; cudaMalloc( (void**)&d_Ezt, gridsize_t * sizeof(cuFloatComplex) );
        
        cuFloatComplex *d_Hxt; cudaMalloc( (void**)&d_Hxt, gridsize_t * sizeof(cuFloatComplex) );
        cuFloatComplex *d_Hyt; cudaMalloc( (void**)&d_Hyt, gridsize_t * sizeof(cuFloatComplex) );
        cuFloatComplex *d_Hzt; cudaMalloc( (void**)&d_Hzt, gridsize_t * sizeof(cuFloatComplex) );
        
        // Call to KERNEL 2
        GpropagateBeam_2<<<nrb, nrt>>>(d_xs, d_ys, d_zs, 
                                   d_A, d_xt, d_yt, d_zt,
                                   d_nxt, d_nyt, d_nzt,
                                   d_Jx, d_Jy, d_Jz,
                                   d_Mx, d_My, d_Mz,
                                   d_Jxt, d_Jyt, d_Jzt,
                                   d_Mxt, d_Myt, d_Mzt,
                                   d_Ext, d_Eyt, d_Ezt,
                                   d_Hxt, d_Hyt, d_Hzt);
        cudaDeviceSynchronize();
        
        // Allocate, on stackframe, Host arrays for J, M, E and H
        cuFloatComplex h_Jxt[gridsize_t];
        cuFloatComplex h_Jyt[gridsize_t];
        cuFloatComplex h_Jzt[gridsize_t];
        
        cuFloatComplex h_Mxt[gridsize_t];
        cuFloatComplex h_Myt[gridsize_t];
        cuFloatComplex h_Mzt[gridsize_t];
        
        cuFloatComplex h_Ext[gridsize_t];
        cuFloatComplex h_Eyt[gridsize_t];
        cuFloatComplex h_Ezt[gridsize_t];
        
        cuFloatComplex h_Hxt[gridsize_t];
        cuFloatComplex h_Hyt[gridsize_t];
        cuFloatComplex h_Hzt[gridsize_t];
        
        // Copy content of Device J,M,E,H arrays into Host J,M,E,H arrays
        cudaMemcpy(h_Jxt, d_Jxt, gridsize_t * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_Jyt, d_Jyt, gridsize_t * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_Jzt, d_Jzt, gridsize_t * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
        
        cudaMemcpy(h_Mxt, d_Mxt, gridsize_t * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_Myt, d_Myt, gridsize_t * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_Mzt, d_Mzt, gridsize_t * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
        
        cudaMemcpy(h_Ext, d_Ext, gridsize_t * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_Eyt, d_Eyt, gridsize_t * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_Ezt, d_Ezt, gridsize_t * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
        
        cudaMemcpy(h_Hxt, d_Hxt, gridsize_t * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_Hyt, d_Hyt, gridsize_t * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_Hzt, d_Hzt, gridsize_t * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
        
        // Free Device memory
        cudaDeviceReset();
        
        // Pack CUDA arrays into cpp array for processing
        std::array<cuFloatComplex*, 3> CJ;
        std::array<cuFloatComplex*, 3> CM;
        
        std::array<cuFloatComplex*, 3> CE;
        std::array<cuFloatComplex*, 3> CH;
        
        // Fill the C++ std::array with C-style arrays
        CJ[0] = h_Jxt;
        CJ[1] = h_Jyt;
        CJ[2] = h_Jzt;
        
        CM[0] = h_Mxt;
        CM[1] = h_Myt;
        CM[2] = h_Mzt;
        
        CE[0] = h_Ext;
        CE[1] = h_Eyt;
        CE[2] = h_Ezt;
        
        CH[0] = h_Hxt;
        CH[1] = h_Hyt;
        CH[2] = h_Hzt;
        
        // Convert the CUDA style arrays to format compatible with CPU functions
        std::vector<std::array<std::complex<float>, 3>> Jt = ghandler.CUDAToCpp_C(CJ, gridsize_t);
        std::vector<std::array<std::complex<float>, 3>> Mt = ghandler.CUDAToCpp_C(CM, gridsize_t);
        
        std::vector<std::array<std::complex<float>, 3>> Et = ghandler.CUDAToCpp_C(CE, gridsize_t);
        std::vector<std::array<std::complex<float>, 3>> Ht = ghandler.CUDAToCpp_C(CH, gridsize_t);

        std::string Jt_file = "Jt";
        std::string Mt_file = "Mt";
        
        std::string Et_file = "Et";
        std::string Ht_file = "Ht";
        
        // Write using standard CPU DataHandler object
        ghandler.dh.writeOutC(Jt, Jt_file);
        ghandler.dh.writeOutC(Mt, Mt_file);
        
        ghandler.dh.writeOutC(Et, Et_file);
        ghandler.dh.writeOutC(Ht, Ht_file);
    }
    
    else if (toPrint == 3)
    {
        // Allocate memory for Pr, Er and Hr arrays on Device
        float *d_Prxt; cudaMalloc( (void**)&d_Prxt, gridsize_t * sizeof(float) );
        float *d_Pryt; cudaMalloc( (void**)&d_Pryt, gridsize_t * sizeof(float) );
        float *d_Przt; cudaMalloc( (void**)&d_Przt, gridsize_t * sizeof(float) );
        
        cuFloatComplex *d_Ext; cudaMalloc( (void**)&d_Ext, gridsize_t * sizeof(cuFloatComplex) );
        cuFloatComplex *d_Eyt; cudaMalloc( (void**)&d_Eyt, gridsize_t * sizeof(cuFloatComplex) );
        cuFloatComplex *d_Ezt; cudaMalloc( (void**)&d_Ezt, gridsize_t * sizeof(cuFloatComplex) );
        
        cuFloatComplex *d_Hxt; cudaMalloc( (void**)&d_Hxt, gridsize_t * sizeof(cuFloatComplex) );
        cuFloatComplex *d_Hyt; cudaMalloc( (void**)&d_Hyt, gridsize_t * sizeof(cuFloatComplex) );
        cuFloatComplex *d_Hzt; cudaMalloc( (void**)&d_Hzt, gridsize_t * sizeof(cuFloatComplex) );
        
        // Call to KERNEL 3
        GpropagateBeam_3<<<nrb, nrt>>>(d_xs, d_ys, d_zs, 
                                   d_A, d_xt, d_yt, d_zt,
                                   d_nxt, d_nyt, d_nzt,
                                   d_Jx, d_Jy, d_Jz,
                                   d_Mx, d_My, d_Mz,
                                   d_Prxt, d_Pryt, d_Przt,
                                   d_Ext, d_Eyt, d_Ezt,
                                   d_Hxt, d_Hyt, d_Hzt);
        cudaDeviceSynchronize();
        
        // Allocate, on stackframe, Host arrays for Pr, Er and Hr
        float h_Prxt[gridsize_t];
        float h_Pryt[gridsize_t];
        float h_Przt[gridsize_t];
        
        cuFloatComplex h_Ext[gridsize_t];
        cuFloatComplex h_Eyt[gridsize_t];
        cuFloatComplex h_Ezt[gridsize_t];
        
        cuFloatComplex h_Hxt[gridsize_t];
        cuFloatComplex h_Hyt[gridsize_t];
        cuFloatComplex h_Hzt[gridsize_t];
        
        // Copy content of Device Pr,Er,Hr arrays into Host Pr,Er,Hr arrays
        cudaMemcpy(h_Prxt, d_Prxt, gridsize_t * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_Pryt, d_Pryt, gridsize_t * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_Przt, d_Przt, gridsize_t * sizeof(float), cudaMemcpyDeviceToHost);
        
        cudaMemcpy(h_Ext, d_Ext, gridsize_t * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_Eyt, d_Eyt, gridsize_t * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_Ezt, d_Ezt, gridsize_t * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
        
        cudaMemcpy(h_Hxt, d_Hxt, gridsize_t * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_Hyt, d_Hyt, gridsize_t * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_Hzt, d_Hzt, gridsize_t * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
        
        // Free Device memory
        cudaDeviceReset();
        
        // Pack CUDA arrays into cpp array for processing
        std::array<float*, 3> CP;
        std::array<cuFloatComplex*, 3> CE;
        std::array<cuFloatComplex*, 3> CH;
        
        // Fill the C++ std::array with C-style arrays
        CP[0] = h_Prxt;
        CP[1] = h_Pryt;
        CP[2] = h_Przt;
        
        CE[0] = h_Ext;
        CE[1] = h_Eyt;
        CE[2] = h_Ezt;
        
        CH[0] = h_Hxt;
        CH[1] = h_Hyt;
        CH[2] = h_Hzt;
        
        // Convert the CUDA style arrays to format compatible with CPU functions
        std::vector<std::array<float, 3>> Pr = ghandler.CUDAToCpp_R(CP, gridsize_t);
        std::vector<std::array<std::complex<float>, 3>> Et = ghandler.CUDAToCpp_C(CE, gridsize_t);
        std::vector<std::array<std::complex<float>, 3>> Ht = ghandler.CUDAToCpp_C(CH, gridsize_t);
        
        std::string Pr_file = "Pr";
        std::string Et_file = "Et";
        std::string Ht_file = "Ht";
        
        // Write using standard CPU DataHandler object
        ghandler.dh.writeOutR(Pr, Pr_file);
        ghandler.dh.writeOutC(Et, Et_file);
        ghandler.dh.writeOutC(Ht, Ht_file);
    }
    return 0;
}
 
