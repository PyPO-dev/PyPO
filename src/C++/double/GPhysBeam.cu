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

#include "GDataHandler.h"
#include "GUtils.h"

#define M_PI            3.14159265358979323846  /* pi */
#define C_L             2.99792458e11 // mm s^-1
#define MU_0            1.2566370614e-3 // kg mm s^-2 A^-2
#define EPS_VAC         1 / (MU_0 * C_L*C_L)
#define ZETA_0_INV      1 / (C_L * MU_0)

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
__constant__ cuDoubleComplex con[CSIZE];     // Contains: k, eps, mu0, zeta0, pi, C_l, Time direction, unit, zero, c4 as complex numbers
__constant__ double eye[3][3];      // Identity matrix
__constant__ int g_s;               // Gridsize on source
__constant__ int g_t;               // Gridsize on target

__device__ __inline__ cuDoubleComplex my_cexp(cuDoubleComplex z)
{
    cuDoubleComplex res;
    double t = exp(z.x);
    double ys = sin(z.y);
    double yc = cos(z.y);
    res = cuCmul(make_cuDoubleComplex(t, 0.), make_cuDoubleComplex(yc, ys));
    return res;
}

// Calculate field at target point
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
        
        Green = cuCmul(cuCdiv(my_cexp(cuCmul(con[6], cuCmul(con[7], cuCmul(con[0], rc)))), (cuCmul(con[9], cuCmul(con[4], rc)))), cuCmul(d_Ac, con[7]));
        
        //printf("%.16f, %.16f\n", rc.x, r);
        
        for( int n=0; n<3; n++)
        {
            e_field[n] = cuCsub(e_field[n], cuCmul(cuCsub(cuCmul(omega, cuCmul(con[2], e_vec_thing[n])), k_out_ms[n]), Green));
            h_field[n] = cuCsub(h_field[n], cuCmul(cuCadd(cuCmul(omega, cuCmul(con[1], h_vec_thing[n])), k_out_js[n]), Green));
        }  
        //printf("%.16g, %.16g\n", cuCreal(Green), cuCimag(Green)); // %s is format specifier
    }

    d_ei[0] = e_field[0];
    d_ei[1] = e_field[1];
    d_ei[2] = e_field[2];
    
    d_hi[0] = h_field[0];
    d_hi[1] = h_field[1];
    d_hi[2] = h_field[2];

    
}

// Propagate to next surface.
// Kernel for toPrint == 0
__global__ void GpropagateBeam(double *d_xs, double *d_ys, double *d_zs,
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

    int jc = 0; // Counter
    
    //for(int i=start; i<stop; i++)
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    if (idx < g_t)
    {
        
        
        point[0] = d_xt[idx];
        point[1] = d_yt[idx];
        point[2] = d_zt[idx];
        
        //printf("%.16f, %.16f, %.16f\n", point[0], point[1], point[2]);
        
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
            
        //std::cout << this->Et_container.size() << std::endl;
        
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
            
            //this->Et_container[k][i] = e_r[k];
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
        /*
        if((i * 100 / this->step) > jc and start == 0 * this->step)
        {
            std::cout << jc << " / 100" << '\r';
            std::cout.flush();
            jc++;
        }
        */
        if (idx == 1000)
        {
            printf("%.16f", d_Myt[idx].x);
        }
    }
}

int main(int argc, char *argv [])
{
    int numThreads  = atoi(argv[1]); // Number of GPU threads per block
    int numBlocks   = atoi(argv[2]); // Threshold in dB for propagation performance
    double k        = atof(argv[3]); // Wavenumber of field to be propagated
    int toPrint     = atoi(argv[4]); // 0 for printing J and M, 1 for E and H and 2 for all fields
    
    double epsilon  = atof(argv[5]); // Relative electric permeability
    int prop_mode   = atoi(argv[6]); // Whether to propagate to surface or to far-field
    double t_direction = atof(argv[7]); // Whether to propagate forward or back in time
    
    int gridsize_s  = atoi(argv[8]); // Source gridsize, flattened
    int gridsize_t  = atoi(argv[9]); // Target gridsize, flattened
    
    // Calculate nr of blocks per grid and nr of threads per block
    dim3 nrb(numBlocks); dim3 nrt(numThreads);
    
    // Calculate permittivity of target
    double EPS = EPS_VAC * epsilon;
    
    // Fill ID
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
    
    // Allocate constant memory on Device
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

    // Copy to constant memory
    cudaMemcpyToSymbol(con, &_con, CSIZE * sizeof(cuDoubleComplex));
    cudaMemcpyToSymbol(eye, &_eye, sizeof(_eye));
    cudaMemcpyToSymbol(g_s, &gridsize_s, sizeof(int));
    cudaMemcpyToSymbol(g_t, &gridsize_t, sizeof(int));

    // Initialize timer to assess performance
    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end; 
    
    std::string source = "s"; 
    std::string target = "t"; 
    
    GDataHandler ghandler;
    
    double source_area[gridsize_s];
    
    ghandler.cppToCUDA_area(source_area);
    
    std::array<double*, 3> grid_source = ghandler.cppToCUDA_3DGrid(source);
    std::array<double*, 3> grid_target3D;
    std::array<double*, 2> grid_target2D;
    std::array<double*, 3> norm_target;
    
    // Allocate source grid and area on GPU
    double *d_xs; cudaMalloc( (void**)&d_xs, gridsize_s * sizeof(double) );
    double *d_ys; cudaMalloc( (void**)&d_ys, gridsize_s * sizeof(double) );
    double *d_zs; cudaMalloc( (void**)&d_zs, gridsize_s * sizeof(double) );
    double *d_A; cudaMalloc( (void**)&d_A, gridsize_s * sizeof(double) );
    
    // Copy data to Device
    cudaMemcpy(d_xs, grid_source[0], gridsize_s * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ys, grid_source[1], gridsize_s * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_zs, grid_source[2], gridsize_s * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, source_area, gridsize_s * sizeof(double), cudaMemcpyHostToDevice);
    
    
    
    
    double *d_xt; double *d_yt; double *d_zt; double *d_nxt; double *d_nyt; double *d_nzt;
    
    if (prop_mode == 0)
    {
        grid_target3D = ghandler.cppToCUDA_3DGrid(target);
        norm_target = ghandler.cppToCUDA_3Dnormals();
        
        // Allocate memory for grids
        cudaMalloc( (void**)&d_xt, gridsize_t * sizeof(double) );
        cudaMalloc( (void**)&d_yt, gridsize_t * sizeof(double) );
        cudaMalloc( (void**)&d_zt, gridsize_t * sizeof(double) );
        
        cudaMalloc( (void**)&d_nxt, gridsize_t * sizeof(double) );
        cudaMalloc( (void**)&d_nyt, gridsize_t * sizeof(double) );
        cudaMalloc( (void**)&d_nzt, gridsize_t * sizeof(double) );
        
        // Copy to GPU from Host
        cudaMemcpy(d_xt, grid_target3D[0], gridsize_t * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_yt, grid_target3D[1], gridsize_t * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_zt, grid_target3D[2], gridsize_t * sizeof(double), cudaMemcpyHostToDevice);
        
        cudaMemcpy(d_nxt, norm_target[0], gridsize_t * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_nyt, norm_target[1], gridsize_t * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_nzt, norm_target[2], gridsize_t * sizeof(double), cudaMemcpyHostToDevice);
    }
    
    else if (prop_mode == 1)
    {
        grid_target2D = ghandler.cppToCUDA_2DGrid();
        
        // Allocate memory for grids
        cudaMalloc( (void**)&d_xt, gridsize_t * sizeof(double) );
        cudaMalloc( (void**)&d_yt, gridsize_t * sizeof(double) );
        
        // Copy to GPU from Host
        cudaMemcpy(d_xt, grid_target3D[0], gridsize_t * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_yt, grid_target3D[1], gridsize_t * sizeof(double), cudaMemcpyHostToDevice);
    }

    std::array<cuDoubleComplex*, 3> Js = ghandler.cppToCUDA_Js();
    std::array<cuDoubleComplex*, 3> Ms = ghandler.cppToCUDA_Ms();
    

    std::cout << Ms[1][0].x << std::endl;

    
    // Allocate memory on GPU for currents
    cuDoubleComplex *d_Jx; cudaMalloc( (void**)&d_Jx, gridsize_s * sizeof(cuDoubleComplex) );
    cuDoubleComplex *d_Jy; cudaMalloc( (void**)&d_Jy, gridsize_s * sizeof(cuDoubleComplex) );
    cuDoubleComplex *d_Jz; cudaMalloc( (void**)&d_Jz, gridsize_s * sizeof(cuDoubleComplex) );
    
    cuDoubleComplex *d_Mx; cudaMalloc( (void**)&d_Mx, gridsize_s * sizeof(cuDoubleComplex) );
    cuDoubleComplex *d_My; cudaMalloc( (void**)&d_My, gridsize_s * sizeof(cuDoubleComplex) );
    cuDoubleComplex *d_Mz; cudaMalloc( (void**)&d_Mz, gridsize_s * sizeof(cuDoubleComplex) );
    
    // Copy currents from Host to GPU
    cudaMemcpy(d_Jx, Js[0], gridsize_s * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Jy, Js[1], gridsize_s * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Jz, Js[2], gridsize_s * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
        
    cudaMemcpy(d_Mx, Ms[0], gridsize_s * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_My, Ms[1], gridsize_s * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Mz, Ms[2], gridsize_s * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    
    
    // Create Device arrays for storing the data.
    // Which kernel is called is determined by toPrint
    if (toPrint == 0)
    {
        cuDoubleComplex *d_Jxt; cudaMalloc( (void**)&d_Jxt, gridsize_t * sizeof(cuDoubleComplex) );
        cuDoubleComplex *d_Jyt; cudaMalloc( (void**)&d_Jyt, gridsize_t * sizeof(cuDoubleComplex) );
        cuDoubleComplex *d_Jzt; cudaMalloc( (void**)&d_Jzt, gridsize_t * sizeof(cuDoubleComplex) );
        
        cuDoubleComplex *d_Mxt; cudaMalloc( (void**)&d_Mxt, gridsize_t * sizeof(cuDoubleComplex) );
        cuDoubleComplex *d_Myt; cudaMalloc( (void**)&d_Myt, gridsize_t * sizeof(cuDoubleComplex) );
        cuDoubleComplex *d_Mzt; cudaMalloc( (void**)&d_Mzt, gridsize_t * sizeof(cuDoubleComplex) );

        std::cout << numBlocks << std::endl;

        // Call to KERNEL 1
        GpropagateBeam<<<nrb, nrt>>>(d_xs, d_ys, d_zs, 
                                   d_A, d_xt, d_yt, d_zt,
                                   d_nxt, d_nyt, d_nzt,
                                   d_Jx, d_Jy, d_Jz,
                                   d_Mx, d_My, d_Mz,
                                   d_Jxt, d_Jyt, d_Jzt,
                                   d_Mxt, d_Myt, d_Mzt);
        cudaDeviceSynchronize();
        cudaDeviceReset();
        
        
        // Pack CUDA arrays into cpp array for processing
        std::array<cuDoubleComplex*, 3> CJ;
        std::array<cuDoubleComplex*, 3> CM;
        
        cuDoubleComplex *h_Jxt = new cuDoubleComplex[gridsize_t];
        cuDoubleComplex *h_Jyt = new cuDoubleComplex[gridsize_t];
        cuDoubleComplex *h_Jzt = new cuDoubleComplex[gridsize_t];
        
        cuDoubleComplex *h_Mxt = new cuDoubleComplex[gridsize_t];
        cuDoubleComplex *h_Myt = new cuDoubleComplex[gridsize_t];
        cuDoubleComplex *h_Mzt = new cuDoubleComplex[gridsize_t];
        
        
        
        cudaMemcpy(h_Jxt, d_Jxt, gridsize_t * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_Jyt, d_Jyt, gridsize_t * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_Jzt, d_Jzt, gridsize_t * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
        
        cudaMemcpy(h_Mxt, d_Mxt, gridsize_t * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_Myt, d_Myt, gridsize_t * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_Mzt, d_Mzt, gridsize_t * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
        
        int size_arr = sizeof(h_Jxt);
        
        CJ[0] = h_Jxt;
        CJ[1] = h_Jyt;
        CJ[2] = h_Jzt;
        
        CM[0] = h_Mxt;
        CM[1] = h_Myt;
        CM[2] = h_Mzt;
        
        
        std::cout << h_Myt[0].x << std::endl;
        
        

        std::vector<std::array<std::complex<double>, 3>> Jt = ghandler.CUDAToCpp_C(CJ);
        std::vector<std::array<std::complex<double>, 3>> Mt = ghandler.CUDAToCpp_C(CM);
    
        std::string Jt_file = "Jt";
        std::string Mt_file = "Mt";
        
        ghandler.dh.writeOutC(Jt, Jt_file);
        ghandler.dh.writeOutC(Mt, Mt_file);
        
        
    }
    /*
    else if (toPrint == 1)
    {
        std::vector<std::array<std::complex<double>, 3>> Et = prop.Et_container;
        std::vector<std::array<std::complex<double>, 3>> Ht = prop.Ht_container;
    
        std::string Et_file = "Et";
        std::string Ht_file = "Ht";
        handler.writeOutC(Et, Et_file);
        handler.writeOutC(Ht, Ht_file);
    }
    
    else if (toPrint == 2)
    {
        std::vector<std::array<std::complex<double>, 3>> Jt = prop.Jt_container;
        std::vector<std::array<std::complex<double>, 3>> Mt = prop.Mt_container;
    
        std::string Jt_file = "Jt";
        std::string Mt_file = "Mt";
        handler.writeOutC(Jt, Jt_file);
        handler.writeOutC(Mt, Mt_file);
        
        std::vector<std::array<std::complex<double>, 3>> Et = prop.Et_container;
        std::vector<std::array<std::complex<double>, 3>> Ht = prop.Ht_container;
    
        std::string Et_file = "Et";
        std::string Ht_file = "Ht";
        handler.writeOutC(Et, Et_file);
        handler.writeOutC(Ht, Ht_file);
    }
    
    else if (toPrint == 3)
    {
        std::vector<std::array<double, 3>> Pr = prop.Pr_container;
        std::vector<std::array<std::complex<double>, 3>> Et = prop.Et_container;
        std::vector<std::array<std::complex<double>, 3>> Ht = prop.Ht_container;
        
        std::string Pr_file = "Pr";
        std::string Et_file = "Et";
        std::string Ht_file = "Ht";
        handler.writeOutR(Pr, Pr_file);
        handler.writeOutC(Et, Et_file);
        handler.writeOutC(Ht, Ht_file);
    }
    */
    // End timer
    end = std::chrono::steady_clock::now();
    
    std::cout << "Elapsed time: " 
        << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() 
        << " [s]\n" << std::endl;
    
    return 0;
}
 
