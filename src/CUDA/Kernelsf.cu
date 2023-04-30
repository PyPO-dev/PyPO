#include "InterfaceCUDA.h"

/*! \file Kernelsf.cu
    \brief Kernels for CUDA PO calculations.
    
    Contains kernels for PO calculations. Multiple kernels are defined, each one optimized for a certain calculation.
*/

// Declare constant memory for Device
__constant__ cuFloatComplex con[CSIZE];     // Contains: k, eps, mu0, zeta0, pi, C_l, Time direction, unit, zero, c4 as complex numbers
//__constant__ cuDoubleComplex con[CSIZE];

__constant__ float eye[3][3];      // Identity matrix
__constant__ int g_s;               // Gridsize on source
__constant__ int g_t;               // Gridsize on target

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
 * Debug complex array.
 *
 * Print complex array of size 3.
 *      Useful for debugging.

 * @param arr Array of 3 cuFloatComplex.
 */
__host__ __device__ void _debugArray(cuFloatComplex arr[3])
{
    printf("%e + %ej, %e + %ej, %e + %ej\n", arr[0].x, arr[0].y, arr[1].x, arr[1].y, arr[2].x, arr[2].y);
}

/**
 * Debug real array.
 *
 * Print real valued array of size 3.
 *      Useful for debugging.

 * @param arr Array of 3 float.
 */
__host__ __device__ void _debugArray(float arr[3])
{
    printf("%e, %e, %e\n", arr[0], arr[1], arr[2]);
}

/**
 * Take complex exponential.
 *
 * Take complex exponential by decomposing into sine and cosine.
 *
 * @return res cuFloatComplex number.
 */
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
    float _r_vec[3];        // Distance vector between source and target points
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
        //js[0] = cuCmulf(con[6], d_Jx[i]);
        //js[1] = cuCmulf(con[6], d_Jy[i]);
        //js[2] = cuCmulf(con[6], d_Jz[i]);
        
        js[0] = d_Jx[i];
        js[1] = d_Jy[i];
        js[2] = d_Jz[i];

        //ms[0] = cuCmulf(con[6], d_Mx[i]);
        //ms[1] = cuCmulf(con[6], d_My[i]);
        //ms[2] = cuCmulf(con[6], d_Mz[i]);
        
        ms[0] = d_Mx[i];
        ms[1] = d_My[i];
        ms[2] = d_Mz[i];

        source_point[0] = d_xs[i];
        source_point[1] = d_ys[i];
        source_point[2] = d_zs[i];
        
        source_norm[0] = d_nxs[i];
        source_norm[1] = d_nys[i];
        source_norm[2] = d_nzs[i];

        //printf("%f\n", source_norm[2]);

        diff(point, source_point, r_vec);
        //s_mult(_r_vec, con[6].x, r_vec);
        abs(r_vec, r);

        rc = make_cuFloatComplex(r, 0.);
        r_inv = 1 / r;

        s_mult(r_vec, r_inv, k_hat);

        dot(source_norm, k_hat, norm_dot_k_hat);
        if (norm_dot_k_hat < 0) {continue;}
        //if ((con[6].x * norm_dot_k_hat) < 0) {continue;}

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
                                float (&r_hat)[3], float *d_A, cuFloatComplex (&e)[3])
{
    // Scalars (float & complex float)
    float omega_mu;                       // Angular frequency of field times mu
    float r_hat_in_rp;                 // r_hat dot product r_prime

    // Arrays of floats
    float source_point[3]; // Container for xyz co-ordinates

    // Arrays of complex floats
    cuFloatComplex js[3] = {con[8], con[8], con[8]};      // Build radiation integral
    cuFloatComplex ms[3] = {con[8], con[8], con[8]};      // Build radiation integral

    cuFloatComplex _ctemp[3];
    cuFloatComplex js_tot_factor[3];
    cuFloatComplex ms_tot_factor[3];
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

    for (int n=0; n<3; n++)
    {
        e[n] = cuCsubf(ms_tot_factor[n], js_tot_factor[n]);
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
                      r_hat, d_A, e);

        d_Ext[idx] = e[0];
        d_Eyt[idx] = e[1];
        d_Ezt[idx] = e[2];

        d_Hxt[idx] = e[0];
        d_Hyt[idx] = e[1];
        d_Hzt[idx] = e[2];
    }
}

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
    float *d_xt, *d_yt, *d_zt, *d_nxt, *d_nyt, *d_nzt, *d_nxs, *d_nys, *d_nzs;

    // Allocate target co-ordinate and normal grids
    gpuErrchk( cudaMalloc((void**)&d_xt, ct->size * sizeof(float)) );
    gpuErrchk( cudaMalloc((void**)&d_yt, ct->size * sizeof(float)) );
    gpuErrchk( cudaMalloc((void**)&d_zt, ct->size * sizeof(float)) );

    gpuErrchk( cudaMemcpy(d_xt, ct->x, ct->size * sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_yt, ct->y, ct->size * sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_zt, ct->z, ct->size * sizeof(float), cudaMemcpyHostToDevice) );

    gpuErrchk( cudaMalloc((void**)&d_nxs, cs->size * sizeof(float)) );
    gpuErrchk( cudaMalloc((void**)&d_nys, cs->size * sizeof(float)) );
    gpuErrchk( cudaMalloc((void**)&d_nzs, cs->size * sizeof(float)) );

    gpuErrchk( cudaMemcpy(d_nxs, cs->nx, cs->size * sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_nys, cs->ny, cs->size * sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_nzs, cs->nz, cs->size * sizeof(float), cudaMemcpyHostToDevice) );
    
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

    // Call to KERNEL 0
    GpropagateBeam_0<<<BT[0], BT[1]>>>(d_xs, d_ys, d_zs,
                                d_A, d_xt, d_yt, d_zt,
                                d_nxs, d_nys, d_nzs,
                                d_nxt, d_nyt, d_nzt,
                                d_Jx, d_Jy, d_Jz,
                                d_Mx, d_My, d_Mz,
                                d_Jxt, d_Jyt, d_Jzt,
                                d_Mxt, d_Myt, d_Mzt);
    
    gpuErrchk( cudaDeviceSynchronize() );

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
    BT = _initCUDA(k, epsilon, ct->size, cs->size, t_direction, nBlocks, nThreads);
    //alsd
    // Create pointers to device arrays and allocate/copy source grid and area.
    float *d_xs, *d_ys, *d_zs, *d_A, *d_nxs, *d_nys, *d_nzs;

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

    gpuErrchk( cudaMalloc((void**)&d_nxs, cs->size * sizeof(float)) );
    gpuErrchk( cudaMalloc((void**)&d_nys, cs->size * sizeof(float)) );
    gpuErrchk( cudaMalloc((void**)&d_nzs, cs->size * sizeof(float)) );

    gpuErrchk( cudaMemcpy(d_nxs, cs->nx, cs->size * sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_nys, cs->ny, cs->size * sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_nzs, cs->nz, cs->size * sizeof(float), cudaMemcpyHostToDevice) );
    
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

    // Call to KERNEL 1
    GpropagateBeam_1<<<BT[0], BT[1]>>>(d_xs, d_ys, d_zs,
                                d_A, d_xt, d_yt, d_zt,
                                d_nxs, d_nys, d_nzs,
                                d_Jx, d_Jy, d_Jz,
                                d_Mx, d_My, d_Mz,
                                d_Ext, d_Eyt, d_Ezt,
                                d_Hxt, d_Hyt, d_Hzt);

    gpuErrchk( cudaDeviceSynchronize() );

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
    BT = _initCUDA(k, epsilon, ct->size, cs->size, t_direction, nBlocks, nThreads);

    // Create pointers to device arrays and allocate/copy source grid and area.
    float *d_xs, *d_ys, *d_zs, *d_A, *d_nxs, *d_nys, *d_nzs;

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

    gpuErrchk( cudaMalloc((void**)&d_nxs, cs->size * sizeof(float)) );
    gpuErrchk( cudaMalloc((void**)&d_nys, cs->size * sizeof(float)) );
    gpuErrchk( cudaMalloc((void**)&d_nzs, cs->size * sizeof(float)) );

    gpuErrchk( cudaMemcpy(d_nxs, cs->nx, cs->size * sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_nys, cs->ny, cs->size * sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_nzs, cs->nz, cs->size * sizeof(float), cudaMemcpyHostToDevice) );
    
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

    // Call to KERNEL 2
    GpropagateBeam_2<<<BT[0], BT[1]>>>(d_xs, d_ys, d_zs,
                                   d_A, d_xt, d_yt, d_zt,
                                   d_nxs, d_nys, d_nzs,
                                   d_nxt, d_nyt, d_nzt,
                                   d_Jx, d_Jy, d_Jz,
                                   d_Mx, d_My, d_Mz,
                                   d_Jxt, d_Jyt, d_Jzt,
                                   d_Mxt, d_Myt, d_Mzt,
                                   d_Ext, d_Eyt, d_Ezt,
                                   d_Hxt, d_Hyt, d_Hzt);

    gpuErrchk( cudaDeviceSynchronize() );

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
    BT = _initCUDA(k, epsilon, ct->size, cs->size, t_direction, nBlocks, nThreads);

    // Create pointers to device arrays and allocate/copy source grid and area.
    float *d_xs, *d_ys, *d_zs, *d_A, *d_nxs, *d_nys, *d_nzs;

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
    gpuErrchk( cudaMalloc((void**)&d_xt, ct->size * sizeof(float)) );
    gpuErrchk( cudaMalloc((void**)&d_yt, ct->size * sizeof(float)) );
    gpuErrchk( cudaMalloc((void**)&d_zt, ct->size * sizeof(float)) );

    gpuErrchk( cudaMemcpy(d_xt, ct->x, ct->size * sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_yt, ct->y, ct->size * sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_zt, ct->z, ct->size * sizeof(float), cudaMemcpyHostToDevice) );

    gpuErrchk( cudaMalloc((void**)&d_nxs, cs->size * sizeof(float)) );
    gpuErrchk( cudaMalloc((void**)&d_nys, cs->size * sizeof(float)) );
    gpuErrchk( cudaMalloc((void**)&d_nzs, cs->size * sizeof(float)) );

    gpuErrchk( cudaMemcpy(d_nxs, cs->nx, cs->size * sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_nys, cs->ny, cs->size * sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_nzs, cs->nz, cs->size * sizeof(float), cudaMemcpyHostToDevice) );
    
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

    // Call to KERNEL 3
    GpropagateBeam_3<<<BT[0], BT[1]>>>(d_xs, d_ys, d_zs,
                                d_A, d_xt, d_yt, d_zt,
                                d_nxs, d_nys, d_nzs,
                                d_nxt, d_nyt, d_nzt,
                                d_Jx, d_Jy, d_Jz,
                                d_Mx, d_My, d_Mz,
                                d_Ext, d_Eyt, d_Ezt,
                                d_Hxt, d_Hyt, d_Hzt,
                                d_Prxt, d_Pryt, d_Przt);

    gpuErrchk( cudaDeviceSynchronize() );

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

    gpuErrchk( cudaMemcpy(h_Prxt, d_Prxt, ct->size * sizeof(float), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(h_Pryt, d_Pryt, ct->size * sizeof(float), cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(h_Przt, d_Przt, ct->size * sizeof(float), cudaMemcpyDeviceToHost) );

    // Free Device memory
    gpuErrchk( cudaDeviceReset() );

    _arrCUDACToC3(h_Ext, h_Eyt, h_Ezt, res->r1x, res->r1y, res->r1z, res->i1x, res->i1y, res->i1z, ct->size);
    _arrCUDACToC3(h_Hxt, h_Hyt, h_Hzt, res->r2x, res->r2y, res->r2z, res->i2x, res->i2y, res->i2z, ct->size);

    for (int i=0; i<ct->size; i++)
    {
        res->r3x[i] = h_Prxt[i];
        res->r3y[i] = h_Pryt[i];
        res->r3z[i] = h_Przt[i];
    }

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
    float *d_xt, *d_yt;
    gpuErrchk( cudaMalloc((void**)&d_xt, ct->size * sizeof(float)) );
    gpuErrchk( cudaMalloc((void**)&d_yt, ct->size * sizeof(float)) );

    gpuErrchk( cudaMemcpy(d_xt, ct->x, ct->size * sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_yt, ct->y, ct->size * sizeof(float), cudaMemcpyHostToDevice) );

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

    // Call to KERNEL 1
    GpropagateBeam_4<<<BT[0], BT[1]>>>(d_xs, d_ys, d_zs,
                                d_A, d_xt, d_yt,
                                d_Jx, d_Jy, d_Jz,
                                d_Mx, d_My, d_Mz,
                                d_Ext, d_Eyt, d_Ezt,
                                d_Hxt, d_Hyt, d_Hzt);

    gpuErrchk( cudaDeviceSynchronize() );

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
    
    cuFloatComplex *h_sfs = new cuFloatComplex[cs->size];

    _arrC1ToCUDAC(inp->x, inp->y, h_sfs, cs->size);

    cuFloatComplex *d_sfs;

    // Allocate and copy scalar input field
    gpuErrchk( cudaMalloc((void**)&d_sfs, cs->size * sizeof(cuFloatComplex)) );

    gpuErrchk( cudaMemcpy(d_sfs, h_sfs, cs->size * sizeof(cuFloatComplex), cudaMemcpyHostToDevice) );

    // Delete host arrays for source currents - not needed anymore
    delete h_sfs;
    
    cuFloatComplex *d_sft;

    gpuErrchk( cudaMalloc((void**)&d_sft, ct->size * sizeof(cuFloatComplex)) );

    // Create stopping event for kernel
    cudaEvent_t event;
    gpuErrchk( cudaEventCreateWithFlags(&event, cudaEventDisableTiming) );

    // Call to KERNEL 5
    GpropagateBeam_5<<<BT[0], BT[1]>>>(d_xs, d_ys, d_zs,
                                d_A, d_xt, d_yt, d_zt,
                                d_sfs, d_sft);

    gpuErrchk( cudaDeviceSynchronize() );

    // Allocate Host arrays for scalar res
    cuFloatComplex *h_sft = new cuFloatComplex[ct->size];

    // Copy data back from Device to Host
    gpuErrchk( cudaMemcpy(h_sft, d_sft, ct->size * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost) );

    // Free Device memory
    gpuErrchk( cudaDeviceReset() );

    _arrCUDACToC1(h_sft, res->x, res->y, ct->size);

    // Delete host arrays for target fields
    delete h_sft;
}
//__host__ getComputeCapability

//cudaDeviceProp deviceProp;
//cudaGetDeviceProperties(&deviceProp, dev);
//std::printf("%d.%d\n", deviceProp.major, deviceProp.minor);
