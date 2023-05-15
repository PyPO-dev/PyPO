#include "InterfaceCUDA.h"

/*! \file Kernelsf.cu
    \brief Kernels for CUDA PO calculations.
    
    Contains kernels for PO calculations. Multiple kernels are defined, each one optimized for a certain calculation.
*/

// Declare constant memory for Device
__constant__ cuFloatComplex con[CSIZE];     // Contains: k, eps, mu0, zeta0, pi, C_l, Time direction, unit, zero, c4 as complex numbers

__constant__ float eye[3][3];      // Identity matrix
__constant__ int g_s;               // Gridsize on source
__constant__ int g_t;               // Gridsize on target

/**
 * Initialize CUDA.
 *
 * Instantiate program and populate constant memory.
 *
 * @param k Wavenumber of incoming field in 1 / mm.
 * @param epsilon Relative electric permittivity of source.
 * @param gt Number of cells on target.
 * @param gs Number of cells on source.
 * @param t_direction Time direction (experimental!).
 * @param nBlocks Number of blocks per grid.
 * @param nThreads Number of threads per block.
 *
 * @return BT Array of two dim3 objects, containing number of blocks per grid and number of threads per block.
 */

 __host__ std::array<dim3, 2> initCUDA(float k, float epsilon, int gt, int gs, float t_direction, int nBlocks, int nThreads)
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
 * Calculate total E and H field at point on target.
 *
 * @param d_xs Array containing source points x-coordinate.
 * @param d_ys Array containing source points y-coordinate.
 * @param d_zs Array containing source points z-coordinate.
 * @param d_nxs Array containing source normals x-component.
 * @param d_nys Array containing source normals y-component.
 * @param d_nzs Array containing source normals z-component.
 * @param d_Jx Array containing source J x-component.
 * @param d_Jy Array containing source J y-component.
 * @param d_Jz Array containing source J z-component.
 * @param d_Mx Array containing source M x-component.
 * @param d_My Array containing source M y-component.
 * @param d_Mz Array containing source M z-component.
 * @param point Array of 3 float, containing xyz coordinates of target point.
 * @param d_A Array containing area elements.
 * @param d_ei Array of 3 cuFloatComplex, to be filled with E-field at point.
 * @param d_hi Array of 3 cuFloatComplex, to be filled with H-field at point.
 */
__device__ void fieldAtPoint(float *d_xs, float *d_ys, float*d_zs,
                    float *d_nxs, float *d_nys, float *d_nzs,
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
    float source_norm[3];  // Container for xyz source normals
    float norm_dot_k_hat;  // Source normal dotted with wavevector direction
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
        
        source_norm[0] = d_nxs[i];
        source_norm[1] = d_nys[i];
        source_norm[2] = d_nzs[i];


        diff(point, source_point, r_vec);
        abs(r_vec, r);

        rc = make_cuFloatComplex(r, 0.);
        r_inv = 1 / r;

        s_mult(r_vec, r_inv, k_hat);

        dot(source_norm, k_hat, norm_dot_k_hat);
        if (norm_dot_k_hat < 0) {continue;}

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
 * Calculate JM on target.
 *
 * Kernel for calculating J, M currents on target surface.
 *
 * @param d_xs Array containing source points x-coordinate.
 * @param d_ys Array containing source points y-coordinate.
 * @param d_zs Array containing source points z-coordinate.
 * @param d_A Array containing area elements.
 * @param d_xt Array containing target points x-coordinate.
 * @param d_yt Array containing target points y-coordinate.
 * @param d_zt Array containing target points z-coordinate.
 * @param d_nxs Array containing source normals x-component.
 * @param d_nys Array containing source normals y-component.
 * @param d_nzs Array containing source normals z-component.
 * @param d_nxt Array containing target norms x-component.
 * @param d_nyt Array containing target norms y-component.
 * @param d_nzt Array containing target norms z-component.
 * @param d_Jx Array containing source J x-component.
 * @param d_Jy Array containing source J y-component.
 * @param d_Jz Array containing source J z-component.
 * @param d_Mx Array containing source M x-component.
 * @param d_My Array containing source M y-component.
 * @param d_Mz Array containing source M z-component.
 * @param d_Jxt Array to be filled with target J x-component.
 * @param d_Jyt Array to be filled with target J y-component.
 * @param d_Jzt Array to be filled with target J z-component.
 * @param d_Mxt Array to be filled with target M x-component.
 * @param d_Myt Array to be filled with target M y-component.
 * @param d_Mzt Array to be filled with target M z-component.
 */
__global__ void GpropagateBeam_0(float *d_xs, float *d_ys, float *d_zs,
                                float *d_A, float *d_xt, float *d_yt, float *d_zt,
                                float *d_nxs, float *d_nys, float *d_nzs,
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
                    d_nxs, d_nys, d_nzs,
                    d_Jx, d_Jy, d_Jz,
                    d_Mx, d_My, d_Mz,
                    point, d_A, d_ei, d_hi);

        // Calculate normalised incoming Poynting vector.
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

        // Now calculate reflected Poynting vector.
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
            e_r[n] = cuCsubf(cuCmulf(e_dot_p_r_perp, make_cuFloatComplex(-p_i_perp[n], 0.)), 
                            cuCmulf(e_dot_p_r_parr, make_cuFloatComplex(p_i_parr[n], 0.)));
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
 * Calculate EH on target.
 *
 * Kernel for calculating E, H fields on target surface.
 *
 * @param d_xs Array containing source points x-coordinate.
 * @param d_ys Array containing source points y-coordinate.
 * @param d_zs Array containing source points z-coordinate.
 * @param d_A Array containing area elements.
 * @param d_xt Array containing target points x-coordinate.
 * @param d_yt Array containing target points y-coordinate.
 * @param d_zt Array containing target points z-coordinate.
 * @param d_nxs Array containing source normals x-component.
 * @param d_nys Array containing source normals y-component.
 * @param d_nzs Array containing source normals z-component.
 * @param d_Jx Array containing source J x-component.
 * @param d_Jy Array containing source J y-component.
 * @param d_Jz Array containing source J z-component.
 * @param d_Mx Array containing source M x-component.
 * @param d_My Array containing source M y-component.
 * @param d_Mz Array containing source M z-component.
 * @param d_Ext Array to be filled with target E x-component.
 * @param d_Eyt Array to be filled with target E y-component.
 * @param d_Ezt Array to be filled with target E z-component.
 * @param d_Hxt Array to be filled with target H x-component.
 * @param d_Hyt Array to be filled with target H y-component.
 * @param d_Hzt Array to be filled with target H z-component.
 */
__global__ void GpropagateBeam_1(float* d_xs, float* d_ys, float* d_zs,
                                float *d_A, float *d_xt, float *d_yt, float *d_zt,
                                float *d_nxs, float *d_nys, float *d_nzs,
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
                    d_nxs, d_nys, d_nzs,
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
 * Calculate JM on target.
 *
 * Kernel for calculating J, M currents on target surface.
 *
 * @param d_xs Array containing source points x-coordinate.
 * @param d_ys Array containing source points y-coordinate.
 * @param d_zs Array containing source points z-coordinate.
 * @param d_A Array containing area elements.
 * @param d_xt Array containing target points x-coordinate.
 * @param d_yt Array containing target points y-coordinate.
 * @param d_zt Array containing target points z-coordinate.
 * @param d_nxs Array containing source normals x-component.
 * @param d_nys Array containing source normals y-component.
 * @param d_nzs Array containing source normals z-component.
 * @param d_nxt Array containing target norms x-component.
 * @param d_nyt Array containing target norms y-component.
 * @param d_nzt Array containing target norms z-component.
 * @param d_Jx Array containing source J x-component.
 * @param d_Jy Array containing source J y-component.
 * @param d_Jz Array containing source J z-component.
 * @param d_Mx Array containing source M x-component.
 * @param d_My Array containing source M y-component.
 * @param d_Mz Array containing source M z-component.
 * @param d_Jxt Array to be filled with target J x-component.
 * @param d_Jyt Array to be filled with target J y-component.
 * @param d_Jzt Array to be filled with target J z-component.
 * @param d_Mxt Array to be filled with target M x-component.
 * @param d_Myt Array to be filled with target M y-component.
 * @param d_Mzt Array to be filled with target M z-component.
 * @param d_Ext Array to be filled with target E x-component.
 * @param d_Eyt Array to be filled with target E y-component.
 * @param d_Ezt Array to be filled with target E z-component.
 * @param d_Hxt Array to be filled with target H x-component.
 * @param d_Hyt Array to be filled with target H y-component.
 * @param d_Hzt Array to be filled with target H z-component.
 */
__global__ void GpropagateBeam_2(float *d_xs, float *d_ys, float *d_zs,
                                float *d_A, float *d_xt, float *d_yt, float *d_zt,
                                float *d_nxs, float *d_nys, float *d_nzs,
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
                    d_nxs, d_nys, d_nzs,
                    d_Jx, d_Jy, d_Jz,
                    d_Mx, d_My, d_Mz,
                    point, d_A, d_ei, d_hi);

        d_Ext[idx] = d_ei[0];
        d_Eyt[idx] = d_ei[1];
        d_Ezt[idx] = d_ei[2];

        d_Hxt[idx] = d_hi[0];
        d_Hyt[idx] = d_hi[1];
        d_Hzt[idx] = d_hi[2];

        // Calculate normalised incoming Poynting vector.
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

        // Now calculate reflected Poynting vector.
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
            e_r[n] = cuCsubf(cuCmulf(e_dot_p_r_perp, make_cuFloatComplex(-p_i_perp[n], 0.)), 
                            cuCmulf(e_dot_p_r_parr, make_cuFloatComplex(p_i_parr[n], 0.)));
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
 * Calculate reflected EH and P on target.
 *
 * Kernel for calculating reflected E, H fields and P, the reflected Poynting vector field, on target surface.
 *
 * @param d_xs Array containing source points x-coordinate.
 * @param d_ys Array containing source points y-coordinate.
 * @param d_zs Array containing source points z-coordinate.
 * @param d_A Array containing area elements.
 * @param d_xt Array containing target points x-coordinate.
 * @param d_yt Array containing target points y-coordinate.
 * @param d_zt Array containing target points z-coordinate.
 * @param d_nxs Array containing source normals x-component.
 * @param d_nys Array containing source normals y-component.
 * @param d_nzs Array containing source normals z-component.
 * @param d_nxt Array containing target norms x-component.
 * @param d_nyt Array containing target norms y-component.
 * @param d_nzt Array containing target norms z-component.
 * @param d_Jx Array containing source J x-component.
 * @param d_Jy Array containing source J y-component.
 * @param d_Jz Array containing source J z-component.
 * @param d_Mx Array containing source M x-component.
 * @param d_My Array containing source M y-component.
 * @param d_Mz Array containing source M z-component.
 * @param d_Ext Array to be filled with target E x-component.
 * @param d_Eyt Array to be filled with target E y-component.
 * @param d_Ezt Array to be filled with target E z-component.
 * @param d_Hxt Array to be filled with target H x-component.
 * @param d_Hyt Array to be filled with target H y-component.
 * @param d_Hzt Array to be filled with target H z-component.
 * @param d_Prxt Array to be filled with target P x-component.
 * @param d_Pryt Array to be filled with target P y-component.
 * @param d_Przt Array to be filled with target P z-component.
 */
__global__ void GpropagateBeam_3(float *d_xs, float *d_ys, float *d_zs,
                                float *d_A, float *d_xt, float *d_yt, float *d_zt,
                                float *d_nxs, float *d_nys, float *d_nzs,
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
                    d_nxs, d_nys, d_nzs,
                    d_Jx, d_Jy, d_Jz,
                    d_Mx, d_My, d_Mz,
                    point, d_A, d_ei, d_hi);

        // Calculate normalised incoming Poynting vector.
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

        // Now calculate reflected Poynting vector.
        snell(S_i_norm, norms, S_r_norm);                // S_r_norm

        // Store REFLECTED Poynting vectors
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
            e_r[n] = cuCsubf(cuCmulf(e_dot_p_r_perp, make_cuFloatComplex(-p_i_perp[n], 0.)), 
                            cuCmulf(e_dot_p_r_parr, make_cuFloatComplex(p_i_parr[n], 0.)));
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

/**
 * Calculate total E and H field at point on far-field target.
 *
 * @param d_xs Array containing source points x-coordinate.
 * @param d_ys Array containing source points y-coordinate.
 * @param d_zs Array containing source points z-coordinate.
 * @param d_Jx Array containing source J x-component.
 * @param d_Jy Array containing source J y-component.
 * @param d_Jz Array containing source J z-component.
 * @param d_Mx Array containing source M x-component.
 * @param d_My Array containing source M y-component.
 * @param d_Mz Array containing source M z-component.
 * @param r_hat Array of 3 float, containing xyz coordinates of target point direction.
 * @param d_A Array containing area elements.
 * @param e Array of 3 cuFloatComplex, to be filled with E-field at point.
 */
__device__ void farfieldAtPoint(float *d_xs, float *d_ys, float *d_zs,
                                cuFloatComplex *d_Jx, cuFloatComplex *d_Jy, cuFloatComplex *d_Jz,
                                cuFloatComplex *d_Mx, cuFloatComplex *d_My, cuFloatComplex *d_Mz,
                                float (&r_hat)[3], float *d_A, cuFloatComplex (&e)[3], cuFloatComplex (&h)[3])
{
    // Scalars (float & complex float)
    float omega_mu;                       // Angular frequency of field times mu
    float omega_eps;                       // Angular frequency of field times eps
    float r_hat_in_rp;                 // r_hat dot product r_prime

    // Arrays of floats
    float source_point[3]; // Container for xyz co-ordinates

    // Arrays of complex floats
    cuFloatComplex js[3] = {con[8], con[8], con[8]};      // Build radiation integral
    cuFloatComplex ms[3] = {con[8], con[8], con[8]};      // Build radiation integral

    cuFloatComplex _ctemp[3];
    cuFloatComplex js_tot_factor[3];
    cuFloatComplex ms_tot_factor[3];
    cuFloatComplex js_tot_factor_h[3];
    cuFloatComplex ms_tot_factor_h[3];
    cuFloatComplex expo;
    cuFloatComplex cfact;

    // Matrices
    float rr_dyad[3][3];       // Dyadic product between r_hat - r_hat
    float eye_min_rr[3][3];    // I - rr

    omega_mu = con[5].x * con[0].x * con[2].x;
    dyad(r_hat, r_hat, rr_dyad);
    matDiff(eye, rr_dyad, eye_min_rr);

    e[0] = con[8];
    e[1] = con[8];
    e[2] = con[8];
    
    h[0] = con[8];
    h[1] = con[8];
    h[2] = con[8];

    for(int i=0; i<g_s; i++)
    {
        source_point[0] = d_xs[i];
        source_point[1] = d_ys[i];
        source_point[2] = d_zs[i];

        dot(r_hat, source_point, r_hat_in_rp);

        expo = expCo(cuCmulf(con[7], make_cuFloatComplex((con[0].x * r_hat_in_rp), 0.)));

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

    omega_eps = con[5].x * con[0].x * con[1].x;
    
    matVec(eye_min_rr, ms, _ctemp);
    s_mult(_ctemp, omega_eps, ms_tot_factor_h);

    ext(r_hat, js, _ctemp);
    s_mult(_ctemp, -con[0].x, js_tot_factor_h);
    
    for (int n=0; n<3; n++)
    {
        e[n] = cuCsubf(ms_tot_factor[n], js_tot_factor[n]);
        h[n] = cuCsubf(js_tot_factor_h[n], ms_tot_factor_h[n]);
    }
}

/**
 * Calculate EH on far-field target.
 *
 * Kernel for calculating E, H fields on a far-field target.
 *
 * @param d_xs Array containing source points x-coordinate.
 * @param d_ys Array containing source points y-coordinate.
 * @param d_zs Array containing source points z-coordinate.
 * @param d_A Array containing area elements.
 * @param d_xt Array containing target direction x-coordinate.
 * @param d_yt Array containing target direction y-coordinate.
 * @param d_Jx Array containing source J x-component.
 * @param d_Jy Array containing source J y-component.
 * @param d_Jz Array containing source J z-component.
 * @param d_Mx Array containing source M x-component.
 * @param d_My Array containing source M y-component.
 * @param d_Mz Array containing source M z-component.
 * @param d_Ext Array to be filled with target E x-component.
 * @param d_Eyt Array to be filled with target E y-component.
 * @param d_Ezt Array to be filled with target E z-component.
 * @param d_Hxt Array to be filled with target H x-component.
 * @param d_Hyt Array to be filled with target H y-component.
 * @param d_Hzt Array to be filled with target H z-component.
 */
void __global__ GpropagateBeam_4(float *d_xs, float *d_ys, float *d_zs,
                                float *d_A, float *d_xt, float *d_yt,
                                cuFloatComplex *d_Jx, cuFloatComplex *d_Jy, cuFloatComplex *d_Jz,
                                cuFloatComplex *d_Mx, cuFloatComplex *d_My, cuFloatComplex *d_Mz,
                                cuFloatComplex *d_Ext, cuFloatComplex *d_Eyt, cuFloatComplex *d_Ezt,
                                cuFloatComplex *d_Hxt, cuFloatComplex *d_Hyt, cuFloatComplex *d_Hzt)
{
    // Scalars (float & complex float)
    float theta;
    float phi;

    // Arrays of floats
    float r_hat[3];                // Unit vector in far-field point direction

    // Arrays of complex floats
    cuFloatComplex e[3];            // far-field E-field
    cuFloatComplex h[3];            // far-field H-field

    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    if (idx < g_t)
    {
        phi     = d_xt[idx];
        theta   = d_yt[idx];

        r_hat[0] = cos(theta) * sin(phi);
        r_hat[1] = sin(theta) * sin(phi);
        r_hat[2] = cos(phi);

        // Calculate total incoming E field at point on far-field
        farfieldAtPoint(d_xs, d_ys, d_zs,
                      d_Jx, d_Jy, d_Jz,
                      d_Mx, d_My, d_Mz,
                      r_hat, d_A, e, h);

        d_Ext[idx] = e[0];
        d_Eyt[idx] = e[1];
        d_Ezt[idx] = e[2];

        d_Hxt[idx] = h[0];
        d_Hyt[idx] = h[1];
        d_Hzt[idx] = h[2];
    }
}


/**
 * Calculate scalarfield on target.
 *
 * Kernel for calculating scalarfield on a target.
 *
 * @param d_xs Array containing source points x-coordinate.
 * @param d_ys Array containing source points y-coordinate.
 * @param d_zs Array containing source points z-coordinate.
 * @param d_sfs Array containing source scalarfield.
 * @param point Array containing target point.
 * @param d_A Array containing area elements.
 * @param e Array to be filled with results.
 */
void __device__ scalarfieldAtPoint(float *d_xs, float *d_ys, float *d_zs,
                                   cuFloatComplex *d_sfs, float (&point)[3], float *d_A, cuFloatComplex &e)
{
    float r;
    float r_vec[3];
    float source_point[3];
    
    e = con[8];
    cuFloatComplex expo;
    cuFloatComplex cfact;

    for(int i=0; i<g_s; i++)
    {
        source_point[0] = d_xs[i];
        source_point[1] = d_ys[i];
        source_point[2] = d_zs[i];

        diff(point, source_point, r_vec);
        abs(r_vec, r);

        expo = expCo(cuCmulf(con[7], make_cuFloatComplex(con[6].x * con[0].x * r, 0)));
        cfact = make_cuFloatComplex(-con[0].x * con[0].x / (4 * r * con[4].x) * d_A[i], 0);
        
        e = cuCaddf(cuCmulf(cuCmulf(cfact, expo), d_sfs[i]), e);
    }
}

/**
 * Calculate scalar field on target.
 *
 * Kernel for calculating scalar field on a target, given a scattered field and source surface..
 *
 * @param d_xs Array containing source points x-coordinate.
 * @param d_ys Array containing source points y-coordinate.
 * @param d_zs Array containing source points z-coordinate.
 * @param d_A Array containing area elements.
 * @param d_xt Array containing target direction x-coordinate.
 * @param d_yt Array containing target direction y-coordinate.
 * @param d_zt Array containing target direction z-coordinate.
 * @param d_sfs Array containing source scalar field.
 * @param d_sft Array to be filled with target scalar field.
 */
void __global__ GpropagateBeam_5(float *d_xs, float *d_ys, float *d_zs,
                                float *d_A, float *d_xt, float *d_yt, float *d_zt,
                                cuFloatComplex *d_sfs, cuFloatComplex *d_sft)
{
    // Arrays of floats
    float point[3];                // Unit vector in far-field point direction

    // Complex floats
    cuFloatComplex e;            // scalar field at point

    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    if (idx < g_t)
    {
        point[0] = d_xt[idx];
        point[1] = d_yt[idx];
        point[2] = d_zt[idx];

        // Calculate total incoming E field at point on far-field
        scalarfieldAtPoint(d_xs, d_ys, d_zs,
                      d_sfs, point, d_A, e);

        d_sft[idx] = e;
    }
}

/**
 * Convert 2 arrays of floats to 1 array of cuComplex
 *
 * @param rarr Real part of complex array.
 * @param iarr Real part of complex array.
 * @param carr Array of cuFloatComplex, to be filled.
 * @param size Size of arrays.
 */
__host__ void _arrC1ToCUDAC(float *rarr, float *iarr, cuFloatComplex* carr,  int size)
{
    for (int i=0; i<size; i++)
    {
        carr[i] = make_cuFloatComplex(rarr[i], iarr[i]);
    }
}

/**
 * Convert 6 arrays of floats to 3 arrays of cuComplex
 *
 * @param r1arr Real part of complex array.
 * @param r2arr Real part of complex array.
 * @param r3arr Real part of complex array.
 * @param i1arr Imaginary part of complex array.
 * @param i2arr Imaginary part of complex array.
 * @param i3arr Imaginary part of complex array.
 * @param c1arr Array of cuFloatComplex, to be filled.
 * @param c2arr Array of cuFloatComplex, to be filled.
 * @param c3arr Array of cuFloatComplex, to be filled.
 * @param size Size of arrays.
 */
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
 * Convert 1 array of cuComplex to 2 arrays of floats.
 *
 * @param carr Array of cuFloatComplex.
 * @param rarr Real part of complex array, to be filled.
 * @param iarr Imaginary part of complex array, to be filled.
 * @param size Size of arrays.
 */
__host__ void _arrCUDACToC1(cuFloatComplex* carr, float *rarr, float *iarr, int size)
{
    for (int i=0; i<size; i++)
    {
        rarr[i] = carr[i].x;
        iarr[i] = carr[i].y;
    }
}

/**
 * Convert 3 arrays of cuComplex to 6 arrays of floats.
 *
 * @param c1arr Array of cuFloatComplex.
 * @param c2arr Array of cuFloatComplex.
 * @param c3arr Array of cuFloatComplex.
 * @param r1arr Real part of complex array, to be filled.
 * @param r2arr Real part of complex array, to be filled.
 * @param r3arr Real part of complex array, to be filled.
 * @param i1arr Imaginary part of complex array, to be filled.
 * @param i2arr Imaginary part of complex array, to be filled.
 * @param i3arr Imaginary part of complex array, to be filled.
 * @param size Size of arrays.
 */
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
 * Call JM kernel.
 *
 * Calculate J, M currents on a target surface using CUDA.
 *
 * @param res Pointer to c2Bundlef object.
 * @param source reflparamsf object containing source surface parameters.
 * @param target reflparamsf object containing target surface parameters.
 * @param cs Pointer to reflcontainerf object containing source grids.
 * @param ct Pointer to reflcontainerf object containing target grids.
 * @param currents Pointer to c2Bundlef object containing source currents.
 * @param k Wavenumber of radiation in 1 /mm.
 * @param epsilon Relative permittivity of source surface.
 * @param t_direction Time direction (experimental!).
 * @param nBlocks Number of blocks in GPU grid.
 * @param nThreads Number of threads in a block.
 *
 * @see c2Bundlef
 * @see reflparamsf
 * @see reflcontainerf
 */
void callKernelf_JM(c2Bundlef *res, reflparamsf source, reflparamsf target,
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
    BT = initCUDA(k, epsilon, ct->size, cs->size, t_direction, nBlocks, nThreads);

    MemUtils memutil;

    int n_ds = 7;
    int n_dt = 6;
     
    std::vector<float*> vec_csdat = {cs->x, cs->y, cs->z, cs->area, cs->nx, cs->ny, cs->nz};
    std::vector<float*> vec_ctdat = {ct->x, ct->y, ct->z, ct->nx, ct->ny, ct->nz};

    std::vector<float*> vec_ds = memutil.cuMallFloat(n_ds, cs->size);
    memutil.cuMemCpFloat(vec_ds, vec_csdat, cs->size); 
    
    std::vector<float*> vec_dt = memutil.cuMallFloat(n_dt, ct->size);
    memutil.cuMemCpFloat(vec_dt, vec_ctdat, ct->size); 

    int n_in = 6;
    int n_out = 6;
    
    std::vector<cuFloatComplex*> vec_hin = memutil.cuMallComplexStack(n_in, cs->size);

    _arrC3ToCUDAC(currents->r1x, currents->r1y, currents->r1z,
                  currents->i1x, currents->i1y, currents->i1z,
                  vec_hin[0], vec_hin[1], vec_hin[2], cs->size);

    _arrC3ToCUDAC(currents->r2x, currents->r2y, currents->r2z,
                  currents->i2x, currents->i2y, currents->i2z,
                  vec_hin[3], vec_hin[4], vec_hin[5], cs->size);
     
    std::vector<cuFloatComplex*> vec_din = memutil.cuMallComplex(n_in, cs->size);
    std::vector<cuFloatComplex*> vec_dout = memutil.cuMallComplex(n_out, ct->size);
    
    memutil.cuMemCpComplex(vec_din, vec_hin, cs->size); 
    memutil.deallocComplexHost(vec_hin);

    // Call to KERNEL 0
    GpropagateBeam_0<<<BT[0], BT[1]>>>(vec_ds[0], vec_ds[1], vec_ds[2], vec_ds[3],
                                       vec_dt[0], vec_dt[1], vec_dt[2],
                                       vec_ds[4], vec_ds[5], vec_ds[6],
                                       vec_dt[3], vec_dt[4], vec_dt[5],
                                       vec_din[0], vec_din[1], vec_din[2],
                                       vec_din[3], vec_din[4], vec_din[5],
                                       vec_dout[0], vec_dout[1], vec_dout[2],
                                       vec_dout[3], vec_dout[4], vec_dout[5]);
    
    gpuErrchk( cudaDeviceSynchronize() );
    
    std::vector<cuFloatComplex*> vec_hout = memutil.cuMallComplexStack(n_out, ct->size);
    memutil.cuMemCpComplex(vec_hout, vec_dout, ct->size, false);

    gpuErrchk( cudaDeviceReset() );

    _arrCUDACToC3(vec_hout[0], vec_hout[1], vec_hout[2], res->r1x, res->r1y, res->r1z, res->i1x, res->i1y, res->i1z, ct->size);
    _arrCUDACToC3(vec_hout[3], vec_hout[4], vec_hout[5], res->r2x, res->r2y, res->r2z, res->i2x, res->i2y, res->i2z, ct->size);

    memutil.deallocComplexHost(vec_hout);
}

/**
 * Call EH kernel.
 *
 * Calculate E, H fields on a target surface using CUDA.
 *
 * @param res Pointer to c2Bundlef object.
 * @param source reflparamsf object containing source surface parameters.
 * @param target reflparamsf object containing target surface parameters.
 * @param cs Pointer to reflcontainerf object containing source grids.
 * @param ct Pointer to reflcontainerf object containing target grids.
 * @param currents Pointer to c2Bundlef object containing source currents.
 * @param k Wavenumber of radiation in 1 /mm.
 * @param epsilon Relative permittivity of source surface.
 * @param t_direction Time direction (experimental!).
 * @param nBlocks Number of blocks in GPU grid.
 * @param nThreads Number of threads in a block.
 *
 * @see c2Bundlef
 * @see reflparamsf
 * @see reflcontainerf
 */
void callKernelf_EH(c2Bundlef *res, reflparamsf source, reflparamsf target,
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
    BT = initCUDA(k, epsilon, ct->size, cs->size, t_direction, nBlocks, nThreads);

    MemUtils memutil;

    int n_ds = 7;
    int n_dt = 3;
     
    std::vector<float*> vec_csdat = {cs->x, cs->y, cs->z, cs->area, cs->nx, cs->ny, cs->nz};
    std::vector<float*> vec_ctdat = {ct->x, ct->y, ct->z};

    std::vector<float*> vec_ds = memutil.cuMallFloat(n_ds, cs->size);
    memutil.cuMemCpFloat(vec_ds, vec_csdat, cs->size); 
    
    std::vector<float*> vec_dt = memutil.cuMallFloat(n_dt, ct->size);
    memutil.cuMemCpFloat(vec_dt, vec_ctdat, ct->size); 

    int n_in = 6;
    int n_out = 6;
    
    std::vector<cuFloatComplex*> vec_hin = memutil.cuMallComplexStack(n_in, cs->size);

    _arrC3ToCUDAC(currents->r1x, currents->r1y, currents->r1z,
                  currents->i1x, currents->i1y, currents->i1z,
                  vec_hin[0], vec_hin[1], vec_hin[2], cs->size);

    _arrC3ToCUDAC(currents->r2x, currents->r2y, currents->r2z,
                  currents->i2x, currents->i2y, currents->i2z,
                  vec_hin[3], vec_hin[4], vec_hin[5], cs->size);
     
    std::vector<cuFloatComplex*> vec_din = memutil.cuMallComplex(n_in, cs->size);
    std::vector<cuFloatComplex*> vec_dout = memutil.cuMallComplex(n_out, ct->size);
    
    memutil.cuMemCpComplex(vec_din, vec_hin, cs->size); 
    memutil.deallocComplexHost(vec_hin);

    GpropagateBeam_1<<<BT[0], BT[1]>>>(vec_ds[0], vec_ds[1], vec_ds[2], vec_ds[3],
                                       vec_dt[0], vec_dt[1], vec_dt[2],
                                       vec_ds[4], vec_ds[5], vec_ds[6],
                                       vec_din[0], vec_din[1], vec_din[2],
                                       vec_din[3], vec_din[4], vec_din[5],
                                       vec_dout[0], vec_dout[1], vec_dout[2],
                                       vec_dout[3], vec_dout[4], vec_dout[5]);
    
    gpuErrchk( cudaDeviceSynchronize() );
    
    std::vector<cuFloatComplex*> vec_hout = memutil.cuMallComplexStack(n_out, ct->size);
    memutil.cuMemCpComplex(vec_hout, vec_dout, ct->size, false);

    gpuErrchk( cudaDeviceReset() );

    _arrCUDACToC3(vec_hout[0], vec_hout[1], vec_hout[2], res->r1x, res->r1y, res->r1z, res->i1x, res->i1y, res->i1z, ct->size);
    _arrCUDACToC3(vec_hout[3], vec_hout[4], vec_hout[5], res->r2x, res->r2y, res->r2z, res->i2x, res->i2y, res->i2z, ct->size);

    memutil.deallocComplexHost(vec_hout);
}

/**
 * Call JMEH kernel.
 *
 * Calculate J, M currents and E, H fields on a target surface using CUDA.
 *
 * @param res Pointer to c4Bundlef object.
 * @param source reflparamsf object containing source surface parameters.
 * @param target reflparamsf object containing target surface parameters.
 * @param cs Pointer to reflcontainerf object containing source grids.
 * @param ct Pointer to reflcontainerf object containing target grids.
 * @param currents Pointer to c2Bundlef object containing source currents.
 * @param k Wavenumber of radiation in 1 /mm.
 * @param epsilon Relative permittivity of source surface.
 * @param t_direction Time direction (experimental!).
 * @param nBlocks Number of blocks in GPU grid.
 * @param nThreads Number of threads in a block.
 *
 * @see c4Bundlef
 * @see c2Bundlef
 * @see reflparamsf
 * @see reflcontainerf
 */
void callKernelf_JMEH(c4Bundlef *res, reflparamsf source, reflparamsf target,
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
    BT = initCUDA(k, epsilon, ct->size, cs->size, t_direction, nBlocks, nThreads);

    MemUtils memutil;

    int n_ds = 7;
    int n_dt = 6;
     
    std::vector<float*> vec_csdat = {cs->x, cs->y, cs->z, cs->area, cs->nx, cs->ny, cs->nz};
    std::vector<float*> vec_ctdat = {ct->x, ct->y, ct->z, ct->nx, ct->ny, ct->nz};

    std::vector<float*> vec_ds = memutil.cuMallFloat(n_ds, cs->size);
    memutil.cuMemCpFloat(vec_ds, vec_csdat, cs->size); 
    
    std::vector<float*> vec_dt = memutil.cuMallFloat(n_dt, ct->size);
    memutil.cuMemCpFloat(vec_dt, vec_ctdat, ct->size); 

    int n_in = 6;
    int n_out = 12;
    
    std::vector<cuFloatComplex*> vec_hin = memutil.cuMallComplexStack(n_in, cs->size);

    _arrC3ToCUDAC(currents->r1x, currents->r1y, currents->r1z,
                  currents->i1x, currents->i1y, currents->i1z,
                  vec_hin[0], vec_hin[1], vec_hin[2], cs->size);

    _arrC3ToCUDAC(currents->r2x, currents->r2y, currents->r2z,
                  currents->i2x, currents->i2y, currents->i2z,
                  vec_hin[3], vec_hin[4], vec_hin[5], cs->size);
     
    std::vector<cuFloatComplex*> vec_din = memutil.cuMallComplex(n_in, cs->size);
    std::vector<cuFloatComplex*> vec_dout = memutil.cuMallComplex(n_out, ct->size);
    
    memutil.cuMemCpComplex(vec_din, vec_hin, cs->size); 
    memutil.deallocComplexHost(vec_hin);

    // Call to KERNEL 2
    GpropagateBeam_2<<<BT[0], BT[1]>>>(vec_ds[0], vec_ds[1], vec_ds[2], vec_ds[3],
                                       vec_dt[0], vec_dt[1], vec_dt[2],
                                       vec_ds[4], vec_ds[5], vec_ds[6],
                                       vec_dt[3], vec_dt[4], vec_dt[5],
                                       vec_din[0], vec_din[1], vec_din[2],
                                       vec_din[3], vec_din[4], vec_din[5],
                                       vec_dout[0], vec_dout[1], vec_dout[2],
                                       vec_dout[3], vec_dout[4], vec_dout[5],
                                       vec_dout[6], vec_dout[7], vec_dout[8],
                                       vec_dout[9], vec_dout[10], vec_dout[11]);
    
    gpuErrchk( cudaDeviceSynchronize() );
    
    std::vector<cuFloatComplex*> vec_hout = memutil.cuMallComplexStack(n_out, ct->size);
    memutil.cuMemCpComplex(vec_hout, vec_dout, ct->size, false);

    gpuErrchk( cudaDeviceReset() );

    _arrCUDACToC3(vec_hout[0], vec_hout[1], vec_hout[2], res->r1x, res->r1y, res->r1z, res->i1x, res->i1y, res->i1z, ct->size);
    _arrCUDACToC3(vec_hout[3], vec_hout[4], vec_hout[5], res->r2x, res->r2y, res->r2z, res->i2x, res->i2y, res->i2z, ct->size);
    _arrCUDACToC3(vec_hout[6], vec_hout[7], vec_hout[8], res->r3x, res->r3y, res->r3z, res->i3x, res->i3y, res->i3z, ct->size);
    _arrCUDACToC3(vec_hout[9], vec_hout[10], vec_hout[11], res->r4x, res->r4y, res->r4z, res->i4x, res->i4y, res->i4z, ct->size);

    memutil.deallocComplexHost(vec_hout);
}

/**
 * Call EHP kernel.
 *
 * Calculate reflected E, H fields and P, the reflected Poynting vectorfield, on a target surface using CUDA.
 *
 * @param res Pointer to c2rBundlef object.
 * @param source reflparamsf object containing source surface parameters.
 * @param target reflparamsf object containing target surface parameters.
 * @param cs Pointer to reflcontainerf object containing source grids.
 * @param ct Pointer to reflcontainerf object containing target grids.
 * @param currents Pointer to c2Bundlef object containing source currents.
 * @param k Wavenumber of radiation in 1 /mm.
 * @param epsilon Relative permittivity of source surface.
 * @param t_direction Time direction (experimental!).
 * @param nBlocks Number of blocks in GPU grid.
 * @param nThreads Number of threads in a block.
 *
 * @see c2rBundlef
 * @see c2Bundlef
 * @see reflparamsf
 * @see reflcontainerf
 */
void callKernelf_EHP(c2rBundlef *res, reflparamsf source, reflparamsf target,
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
    BT = initCUDA(k, epsilon, ct->size, cs->size, t_direction, nBlocks, nThreads);

    MemUtils memutil;

    int n_ds = 7;
    int n_dt = 6;
     
    std::vector<float*> vec_csdat = {cs->x, cs->y, cs->z, cs->area, cs->nx, cs->ny, cs->nz};
    std::vector<float*> vec_ctdat = {ct->x, ct->y, ct->z, ct->nx, ct->ny, ct->nz};

    std::vector<float*> vec_ds = memutil.cuMallFloat(n_ds, cs->size);
    memutil.cuMemCpFloat(vec_ds, vec_csdat, cs->size); 
    
    std::vector<float*> vec_dt = memutil.cuMallFloat(n_dt, ct->size);
    memutil.cuMemCpFloat(vec_dt, vec_ctdat, ct->size); 

    int n_in = 6;
    int n_out = 6;
    int n_out_poynt = 3;
    
    std::vector<cuFloatComplex*> vec_hin = memutil.cuMallComplexStack(n_in, cs->size);

    _arrC3ToCUDAC(currents->r1x, currents->r1y, currents->r1z,
                  currents->i1x, currents->i1y, currents->i1z,
                  vec_hin[0], vec_hin[1], vec_hin[2], cs->size);

    _arrC3ToCUDAC(currents->r2x, currents->r2y, currents->r2z,
                  currents->i2x, currents->i2y, currents->i2z,
                  vec_hin[3], vec_hin[4], vec_hin[5], cs->size);
     
    std::vector<cuFloatComplex*> vec_din = memutil.cuMallComplex(n_in, cs->size);
    std::vector<cuFloatComplex*> vec_dout = memutil.cuMallComplex(n_out, ct->size);
    std::vector<float*> vec_dpoynt = memutil.cuMallFloat(n_out_poynt, ct->size);
    
    memutil.cuMemCpComplex(vec_din, vec_hin, cs->size); 
    memutil.deallocComplexHost(vec_hin);

    // Call to KERNEL 3
    GpropagateBeam_3<<<BT[0], BT[1]>>>(vec_ds[0], vec_ds[1], vec_ds[2], vec_ds[3],
                                       vec_dt[0], vec_dt[1], vec_dt[2],
                                       vec_ds[4], vec_ds[5], vec_ds[6],
                                       vec_dt[3], vec_dt[4], vec_dt[5],
                                       vec_din[0], vec_din[1], vec_din[2],
                                       vec_din[3], vec_din[4], vec_din[5],
                                       vec_dout[0], vec_dout[1], vec_dout[2],
                                       vec_dout[3], vec_dout[4], vec_dout[5],
                                       vec_dpoynt[0], vec_dpoynt[1], vec_dpoynt[2]);

    std::vector<cuFloatComplex*> vec_hout = memutil.cuMallComplexStack(n_out, ct->size);
    std::vector<float*> vec_hpoynt = memutil.cuMallFloatStack(n_out_poynt, ct->size);
    memutil.cuMemCpComplex(vec_hout, vec_dout, ct->size, false);
    memutil.cuMemCpFloat(vec_hpoynt, vec_dpoynt, ct->size, false);

    gpuErrchk( cudaDeviceReset() );

    _arrCUDACToC3(vec_hout[0], vec_hout[1], vec_hout[2], res->r1x, res->r1y, res->r1z, res->i1x, res->i1y, res->i1z, ct->size);
    _arrCUDACToC3(vec_hout[3], vec_hout[4], vec_hout[5], res->r2x, res->r2y, res->r2z, res->i2x, res->i2y, res->i2z, ct->size);

    for (int i=0; i<ct->size; i++)
    {
        res->r3x[i] = vec_hpoynt[0][i];
        res->r3y[i] = vec_hpoynt[1][i];
        res->r3z[i] = vec_hpoynt[2][i];
    }

    memutil.deallocComplexHost(vec_hout);
    memutil.deallocFloatHost(vec_hpoynt);
}

/**
 * Call FF kernel.
 *
 * Calculate E, H fields on a far-field target surface using CUDA.
 *
 * @param res Pointer to c2Bundlef object.
 * @param source reflparamsf object containing source surface parameters.
 * @param target reflparamsf object containing target surface parameters.
 * @param cs Pointer to reflcontainerf object containing source grids.
 * @param ct Pointer to reflcontainerf object containing target grids.
 * @param currents Pointer to c2Bundlef object containing source currents.
 * @param k Wavenumber of radiation in 1 /mm.
 * @param epsilon Relative permittivity of source surface.
 * @param t_direction Time direction (experimental!).
 * @param nBlocks Number of blocks in GPU grid.
 * @param nThreads Number of threads in a block.
 *
 * @see c2Bundlef
 * @see reflparamsf
 * @see reflcontainerf
 */
void callKernelf_FF(c2Bundlef *res, reflparamsf source, reflparamsf target,
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
    BT = initCUDA(k, epsilon, ct->size, cs->size, t_direction, nBlocks, nThreads);

    MemUtils memutil;

    int n_ds = 4;
    int n_dt = 2;
     
    std::vector<float*> vec_csdat = {cs->x, cs->y, cs->z, cs->area};
    std::vector<float*> vec_ctdat = {ct->x, ct->y};

    std::vector<float*> vec_ds = memutil.cuMallFloat(n_ds, cs->size);
    memutil.cuMemCpFloat(vec_ds, vec_csdat, cs->size); 
    
    std::vector<float*> vec_dt = memutil.cuMallFloat(n_dt, ct->size);
    memutil.cuMemCpFloat(vec_dt, vec_ctdat, ct->size); 

    int n_in = 6;
    int n_out = 6;
    
    std::vector<cuFloatComplex*> vec_hin = memutil.cuMallComplexStack(n_in, cs->size);

    _arrC3ToCUDAC(currents->r1x, currents->r1y, currents->r1z,
                  currents->i1x, currents->i1y, currents->i1z,
                  vec_hin[0], vec_hin[1], vec_hin[2], cs->size);

    _arrC3ToCUDAC(currents->r2x, currents->r2y, currents->r2z,
                  currents->i2x, currents->i2y, currents->i2z,
                  vec_hin[3], vec_hin[4], vec_hin[5], cs->size);
     
    std::vector<cuFloatComplex*> vec_din = memutil.cuMallComplex(n_in, cs->size);
    std::vector<cuFloatComplex*> vec_dout = memutil.cuMallComplex(n_out, ct->size);
    
    memutil.cuMemCpComplex(vec_din, vec_hin, cs->size); 
    memutil.deallocComplexHost(vec_hin);

    GpropagateBeam_4<<<BT[0], BT[1]>>>(vec_ds[0], vec_ds[1], vec_ds[2], vec_ds[3],
                                       vec_dt[0], vec_dt[1],
                                       vec_din[0], vec_din[1], vec_din[2],
                                       vec_din[3], vec_din[4], vec_din[5],
                                       vec_dout[0], vec_dout[1], vec_dout[2],
                                       vec_dout[3], vec_dout[4], vec_dout[5]);

    gpuErrchk( cudaDeviceSynchronize() );
    
    std::vector<cuFloatComplex*> vec_hout = memutil.cuMallComplexStack(n_out, ct->size);
    memutil.cuMemCpComplex(vec_hout, vec_dout, ct->size, false);

    gpuErrchk( cudaDeviceReset() );

    _arrCUDACToC3(vec_hout[0], vec_hout[1], vec_hout[2], res->r1x, res->r1y, res->r1z, res->i1x, res->i1y, res->i1z, ct->size);
    _arrCUDACToC3(vec_hout[3], vec_hout[4], vec_hout[5], res->r2x, res->r2y, res->r2z, res->i2x, res->i2y, res->i2z, ct->size);

    memutil.deallocComplexHost(vec_hout);
}

/**
 * Call scalar kernel.
 *
 * Calculate scalar field on a target surface using CUDA.
 *
 * @param res Pointer to arrC1f object.
 * @param source reflparamsf object containing source surface parameters.
 * @param target reflparamsf object containing target surface parameters.
 * @param cs Pointer to reflcontainerf object containing source grids.
 * @param ct Pointer to reflcontainerf object containing target grids.
 * @param inp Pointer to arrC1f object containing source field.
 * @param k Wavenumber of radiation in 1 /mm.
 * @param epsilon Relative permittivity of source surface.
 * @param t_direction Time direction (experimental!).
 * @param nBlocks Number of blocks in GPU grid.
 * @param nThreads Number of threads in a block.
 *
 * @see arrC1f
 * @see reflparamsf
 * @see reflcontainerf
 */
void callKernelf_scalar(arrC1f *res, reflparamsf source, reflparamsf target,
                                reflcontainerf *cs, reflcontainerf *ct,
                                arrC1f *inp,
                                float k, float epsilon,
                                float t_direction, int nBlocks, int nThreads)
{
    // Generate source and target grids
    generateGridf(source, cs);
    generateGridf(target, ct);

    // Initialize and copy constant memory to device
    std::array<dim3, 2> BT;
    BT = initCUDA(k, epsilon, ct->size, cs->size, t_direction, nBlocks, nThreads);
    //alsd
    // Create pointers to device arrays and allocate/copy source grid and area.
    MemUtils memutil;

    int n_ds = 4;
    int n_dt = 3;
     
    std::vector<float*> vec_csdat = {cs->x, cs->y, cs->z, cs->area};
    std::vector<float*> vec_ctdat = {ct->x, ct->y, ct->z};

    std::vector<float*> vec_ds = memutil.cuMallFloat(n_ds, cs->size);
    memutil.cuMemCpFloat(vec_ds, vec_csdat, cs->size); 
    
    std::vector<float*> vec_dt = memutil.cuMallFloat(n_dt, ct->size);
    memutil.cuMemCpFloat(vec_dt, vec_ctdat, ct->size); 
    
    cuFloatComplex *h_sfs = new cuFloatComplex[cs->size];

    _arrC1ToCUDAC(inp->x, inp->y, h_sfs, cs->size);

    cuFloatComplex *d_sfs;

    gpuErrchk( cudaMalloc((void**)&d_sfs, cs->size * sizeof(cuFloatComplex)) );
    gpuErrchk( cudaMemcpy(d_sfs, h_sfs, cs->size * sizeof(cuFloatComplex), cudaMemcpyHostToDevice) );

    delete h_sfs;
    
    cuFloatComplex *d_sft;

    gpuErrchk( cudaMalloc((void**)&d_sft, ct->size * sizeof(cuFloatComplex)) );

    GpropagateBeam_5<<<BT[0], BT[1]>>>(vec_ds[0], vec_ds[1], vec_ds[2], vec_ds[3],
                                       vec_dt[0], vec_dt[1], vec_dt[2],
                                       d_sfs, d_sft);

    gpuErrchk( cudaDeviceSynchronize() );

    cuFloatComplex *h_sft = new cuFloatComplex[ct->size];
    gpuErrchk( cudaMemcpy(h_sft, d_sft, ct->size * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost) );

    gpuErrchk( cudaDeviceReset() );

    _arrCUDACToC1(h_sft, res->x, res->y, ct->size);

    delete h_sft;
}
