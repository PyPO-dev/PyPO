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

    float PIf = 3.1415926; /* pi */
    float C_L = 2.9979246e8; // m s^-1
    float MU_0 = 1.256637e-6; // kg m s^-2 A^-2
    float EPS_VAC = 1 / (MU_0 * C_L*C_L);
    float EPS = EPS_VAC * epsilon;
    float ZETA = sqrt( MU_0 / EPS);
    float ZETA_INV = 1 / ZETA;

    float prefactor = k*k/(4*PIf);

    // printf("PIf         : %.16g\n", PIf);
    // printf("C_L         : %.16g\n", C_L);
    // printf("MU_0        : %.16g\n", MU_0);
    // printf("EPS_VAC     : %.16g\n", EPS_VAC);
    // printf("EPS         : %.16g\n", EPS);
    // printf("ZETA        : %.16g\n", ZETA);
    // printf("ZETA_INV    : %.16g\n", ZETA_INV);
    // printf("t_direction : %.3g\n", t_direction); 
    
    // printf("k           : %.16g\n", k);
    // printf("prefactor   : %.16g\n", prefactor);


    // Fill ID matrix
    float _eye[3][3] = {};
    _eye[0][0] = 1.;
    _eye[1][1] = 1.;
    _eye[2][2] = 1.;
    
    // Pack constant array
    cuFloatComplex _con[CSIZE] = {make_cuFloatComplex(k, 0.),              // _con[0]
                                    make_cuFloatComplex(prefactor,0),      // _con[1]
                                    make_cuFloatComplex(EPS, 0.),          // _con[2]
                                    make_cuFloatComplex(MU_0, 0.),         // _con[3]
                                    make_cuFloatComplex(ZETA, 0.),         // _con[4]
                                    make_cuFloatComplex(ZETA_INV, 0.),     // _con[5]
                                    make_cuFloatComplex(PIf, 0.),          // _con[6]
                                    make_cuFloatComplex(C_L, 0.),          // _con[7]
                                    make_cuFloatComplex(t_direction, 0.),  // _con[8]
                                    make_cuFloatComplex(0., 1.),           // _con[9]
                                    make_cuFloatComplex(0., 0.),           // _con[10]
                                    make_cuFloatComplex(1., 0.)};          // _con[11]

    
    // Copy constant array to Device constant memory
    gpuErrchk( cudaMemcpyToSymbol(g_s, &gs, sizeof(int)) );
    gpuErrchk( cudaMemcpyToSymbol(g_t, &gt, sizeof(int)) );
    gpuErrchk( cudaMemcpyToSymbol(eye, &_eye, sizeof(_eye)) );
    gpuErrchk( cudaMemcpyToSymbol(con, &_con, CSIZE * sizeof(cuFloatComplex)) );

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
 * @param gmode Int indicating the type of source grid.
 * @param ncx  Int the number of x/u points in the source grid.
 * @param ncy  Int the number of y/v points in the source grid
 * @param d_ei Array of 3 cuFloatComplex, to be filled with E-field at point.
 * @param d_hi Array of 3 cuFloatComplex, to be filled with H-field at point.
 */
__device__ void fieldAtPoint(float *d_xs, float *d_ys, float*d_zs,
                    float *d_nxs, float *d_nys, float *d_nzs,
                    cuFloatComplex *d_Jx, cuFloatComplex *d_Jy, cuFloatComplex *d_Jz,
                    cuFloatComplex *d_Mx, cuFloatComplex *d_My, cuFloatComplex *d_Mz,
                    float (&point)[3], float *d_A, int gmode, int ncx, int ncy,
                    cuFloatComplex (&d_ei)[3], cuFloatComplex (&d_hi)[3])
{
    int i;                // index in grids
    // Scalars (float & complex float)    
    float R;                            // Distance between source and target points
    float R_inv;                        // 1 / R
    float kR;                           // k*R
    float kR_inv;                       // inverse of kR
    cuFloatComplex Green;               // Container for Green's function
    cuFloatComplex js_dot_R;            // Container for inner products between wavevctor and electric currents
    cuFloatComplex ms_dot_R;            // Container for inner products between wavevctor and magnetics currents

    cuFloatComplex kR_inv_sum1;     // Container for the first common complex sum of 1/kR powers
    cuFloatComplex kR_inv_sum2;     // Container for the second common complex sum of 1/kR powers
    cuFloatComplex kR_inv_sum3;     // Container for the third common complex sum of 1/kR powers

    // Arrays of floats
    float source_point[3]; // Container for xyz co-ordinates
    float source_norm[3];  // Container for xyz source normals
    float norm_dot_R_hat;  // Source normal dotted with wavevector direction
    float R_vec[3];        // Distance vector between source and target points
    float R_hat[3];        // Unit distance vector

    // Arrays of complex floats
    cuFloatComplex e_field[3] = {con[10], con[10], con[10]}; // Electric field on target
    cuFloatComplex h_field[3] = {con[10], con[10], con[10]}; // Magnetic field on target
    cuFloatComplex ye_field[3] = {con[10], con[10], con[10]}; // Intermediate electric field due to integral over y/v
    cuFloatComplex yh_field[3] = {con[10], con[10], con[10]}; // Intermediate magnetic field due to integral over y/v
    cuFloatComplex js[3];             // Electric current at source point
    cuFloatComplex ms[3];             // Magnetic current at source point
    cuFloatComplex js_dot_R_R[3];     // Electric current contribution to e-field
    cuFloatComplex ms_dot_R_R[3];    // Magnetic current contribution to h-field
    cuFloatComplex ms_cross_R[3];     // Outer product between ms and R_hat 
    cuFloatComplex js_cross_R[3];     // Outer product between js and R_hat 
    cuFloatComplex e_temp[3];           // Temporary container for intermediate values
    cuFloatComplex h_temp[3];           // Temporary container for intermediate values

    // Integrate over each source point
    // We split the integral into two parts, with the outer loop over the 
    // x/u axis, and the inner loop over the y/v axis
    for(int xu=0; xu<ncx; xu++)
    {
        for (int n=0; n<3; n++) 
        {
            ye_field[n] = con[10]; // Intermediate electric field due to integral over y/v
            yh_field[n] = con[10]; // Intermediate magnetic field due to integral over y/v
        }

        for(int yv=0; yv<ncy; yv++)
        {
            i = xu*ncy + yv;

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


            diff(point, source_point, R_vec);
            
            abs(R_vec, R);
            
            R_inv = 1/R;

            s_mult(R_vec, R_inv, R_hat);
            
            dot(source_norm, R_hat, norm_dot_R_hat);
            
            if ((norm_dot_R_hat < 0) && (con[8].x < 0)) {
                continue;}

            kR = con[0].x * R;
            kR_inv = 1.0f/kR;

            // Calculate the complex sums that appear in the integral
            // con[8] = ∓1 for forward and backward propagation
            // -i/(kR) ∓ 1/(kR)² + i/(kR)³
            kR_inv_sum1 = make_cuFloatComplex(cuCrealf(con[8])*kR_inv*kR_inv, kR_inv*(kR_inv*kR_inv - 1));

            // i/(kR) ± 3/(kR)² - 3i/(kR)³
            kR_inv_sum2 = make_cuFloatComplex(-cuCrealf(con[8])*3*kR_inv*kR_inv, kR_inv*(1 - 3*kR_inv*kR_inv));

            // ∓(i/(kR) ∓ 1/(kR)²)
            kR_inv_sum3 = cuCmulf(con[8], make_cuFloatComplex(-cuCrealf(con[8])*kR_inv*kR_inv, kR_inv));

            // Vector calculations
            // e-field
            // J.Rh
            dot(js, R_hat, js_dot_R);
            
            // (J.Rh)Rh
            s_mult(R_hat, js_dot_R, js_dot_R_R);
            
            // M x Rh
            ext(ms, R_hat, ms_cross_R);
            

            // h-field
            // M.Rh
            dot(ms, R_hat, ms_dot_R);
            
            // (M.Rh)Rh
            s_mult(R_hat, ms_dot_R, ms_dot_R_R);
            
            // j x Rh
            ext(js, R_hat, js_cross_R);
            
            // Green's function
            cuFloatComplex d_Ac = make_cuFloatComplex(d_A[i], 0.);
            
            // k²/(4π) e^{∓ikR} dA
            Green = cuCmulf(cuCmulf(con[1], cuCexpf(cuCmulf(con[8], make_cuFloatComplex(0, kR)))), d_Ac);

            for( int n=0; n<3; n++)
            {
                // If this is an integral over y/el, only add half of 
                // the first and last points
                if ((gmode != 1) && ((yv==0) || (yv==ncy-1)))
                {
                    // 0.5*
                    ye_field[n] = cuCaddf(
                                    cuCmulf(
                                        make_cuFloatComplex(0.5,0), 
                                        cuCmulf(
                                            cuCaddf(
                                                cuCaddf(
                                                    cuCmulf(js[n], kR_inv_sum1), 
                                                    cuCmulf(js_dot_R_R[n], kR_inv_sum2)
                                                )
                                                , 
                                                cuCmulf(ms_cross_R[n], kR_inv_sum3)
                                            ), 
                                            Green
                                        )
                                    ), 
                                    ye_field[n]
                                )  ;
                
                    yh_field[n] = cuCaddf(
                                    cuCmulf(
                                        make_cuFloatComplex(0.5,0), 
                                        cuCmulf(
                                            cuCsubf(
                                                cuCaddf(
                                                    cuCmulf(ms[n], kR_inv_sum1), 
                                                    cuCmulf(ms_dot_R_R[n], kR_inv_sum2)
                                                ), 
                                                cuCmulf(js_cross_R[n], kR_inv_sum3)
                                            ), Green
                                        )
                                    ), yh_field[n]
                                  );
                }
                else  // add the full value of the points
                {
                    ye_field[n] = cuCaddf(
                                    cuCmulf(
                                        cuCaddf(
                                            cuCaddf(
                                                cuCmulf(js[n], kR_inv_sum1), 
                                                cuCmulf(js_dot_R_R[n], kR_inv_sum2)
                                            ),
                                            cuCmulf(ms_cross_R[n], kR_inv_sum3)
                                        ),
                                        Green
                                    ), 
                                    ye_field[n]
                                  );
                
                    yh_field[n] = cuCaddf(
                                    cuCmulf(
                                        cuCsubf(
                                            cuCaddf(
                                                cuCmulf(ms[n], kR_inv_sum1), 
                                                cuCmulf(ms_dot_R_R[n], kR_inv_sum2)
                                            ), 
                                            cuCmulf(js_cross_R[n], kR_inv_sum3)
                                        ), 
                                        Green
                                    ), 
                                    yh_field[n]
                                  );
                }
            }
        }  // End of y/v loop

        for( int n=0; n<3; n++)
        {
            if ((xu==0) || (xu==ncx-1)) // Only add half the point value
            {
                e_field[n] = cuCaddf(
                                cuCmulf(
                                    ye_field[n], 
                                    make_cuFloatComplex(0.5,0)
                                ), 
                                e_field[n]
                            );
                h_field[n] = cuCaddf(
                                cuCmulf(
                                    yh_field[n], 
                                    make_cuFloatComplex(0.5,0)
                                ), 
                                h_field[n]
                            );
            }
            else 
            {
                e_field[n] = cuCaddf(ye_field[n], e_field[n]);
                h_field[n] = cuCaddf(yh_field[n], h_field[n]);
            }
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
 * Calculate JM on PEC target
 *
 * Kernel for calculating J, M currents on a perfectly conducting target surface.
 * This reduces the calculation of the currents to Je = 2 n^Hinc and Jm = -2 n^Einc
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
 * @param gmode Int the mode of source grid.
 * @param ncx   Int number of x/u points in source grid.
 * @param ncy   Int number of y/v points in source grid.
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
__global__ void GpropagateBeam_0_PEC(float *d_xs, float *d_ys, float *d_zs,
                                float *d_A, 
                                float *d_xt, float *d_yt, float *d_zt,
                                float *d_nxs, float *d_nys, float *d_nzs,
                                float *d_nxt, float *d_nyt, float *d_nzt,
                                int gmode, int ncx, int ncy,
                                cuFloatComplex *d_Jx, cuFloatComplex *d_Jy, cuFloatComplex *d_Jz,
                                cuFloatComplex *d_Mx, cuFloatComplex *d_My, cuFloatComplex *d_Mz,
                                cuFloatComplex *d_Jxt, cuFloatComplex *d_Jyt, cuFloatComplex *d_Jzt,
                                cuFloatComplex *d_Mxt, cuFloatComplex *d_Myt, cuFloatComplex *d_Mzt)
{
    float point[3];            // Point on target
    float norms[3];            // Normal vector at point

    // Return containers
    cuFloatComplex d_je[3];    // Electric current
    cuFloatComplex d_jm[3];    // Electric current

    // E and H field containers
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
                    point, d_A, 
                    gmode, ncx, ncy, 
                    d_ei, d_hi);

        ext(norms, d_hi, d_je);

        d_Jxt[idx] = cuCmulf(make_cuFloatComplex(2., 0), d_je[0]);
        d_Jyt[idx] = cuCmulf(make_cuFloatComplex(2., 0), d_je[1]);
        d_Jzt[idx] = cuCmulf(make_cuFloatComplex(2., 0), d_je[2]);

        d_Mxt[idx] = make_cuFloatComplex(0, 0);
        d_Myt[idx] = make_cuFloatComplex(0, 0);
        d_Mzt[idx] = make_cuFloatComplex(0, 0);
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
 * @param gmode Int indicating the type of source grid.
 * @param ncx  Int the number of x/u points in the source grid.
 * @param ncy  Int the number of y/v points in the source grid
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
                                int gmode, int ncx, int ncy,
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
                    point, d_A, gmode, ncx, ncy,
                    d_ei, d_hi);

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
 * Calculate JMEH on PEC target
 *
 * Kernel for calculating J, M, E & H currents on a perfectly conducting target surface.
 * This reduces the calculation of the currents to Je = 2 n^Hinc and Jm = 2 n^Einc
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
 * @param gmode Int indicating the type of source grid.
 * @param ncx  Int the number of x/u points in the source grid.
 * @param ncy  Int the number of y/v points in the source grid
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
__global__ void GpropagateBeam_2_PEC(float *d_xs, float *d_ys, float *d_zs,
                                float *d_A, float *d_xt, float *d_yt, float *d_zt,
                                float *d_nxs, float *d_nys, float *d_nzs,
                                float *d_nxt, float *d_nyt, float *d_nzt,
                                int gmode, int ncx, int ncy,
                                cuFloatComplex *d_Jx, cuFloatComplex *d_Jy, cuFloatComplex *d_Jz,
                                cuFloatComplex *d_Mx, cuFloatComplex *d_My, cuFloatComplex *d_Mz,
                                cuFloatComplex *d_Jxt, cuFloatComplex *d_Jyt, cuFloatComplex *d_Jzt,
                                cuFloatComplex *d_Mxt, cuFloatComplex *d_Myt, cuFloatComplex *d_Mzt,
                                cuFloatComplex *d_Ext, cuFloatComplex *d_Eyt, cuFloatComplex *d_Ezt,
                                cuFloatComplex *d_Hxt, cuFloatComplex *d_Hyt, cuFloatComplex *d_Hzt)
{
    float point[3];            // Point on target
    float norms[3];            // Normal vector at point

    // Return containers
    cuFloatComplex d_je[3];    // Electric current
    cuFloatComplex d_jm[3];    // Electric current

    // E and H field containers
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
                    point, d_A, gmode, ncx, ncy,
                    d_ei, d_hi);

        d_Ext[idx] = d_ei[0];
        d_Eyt[idx] = d_ei[1];
        d_Ezt[idx] = d_ei[2];

        d_Hxt[idx] = d_hi[0];
        d_Hyt[idx] = d_hi[1];
        d_Hzt[idx] = d_hi[2];

        ext(norms, d_hi, d_je);

        d_Jxt[idx] = cuCmulf(make_cuFloatComplex(2., 0), d_je[0]);
        d_Jyt[idx] = cuCmulf(make_cuFloatComplex(2., 0), d_je[1]);
        d_Jzt[idx] = cuCmulf(make_cuFloatComplex(2., 0), d_je[2]);

        ext(norms, d_ei, d_jm);
        
        d_Mxt[idx] = make_cuFloatComplex(0, 0);
        d_Myt[idx] = make_cuFloatComplex(0, 0);
        d_Mzt[idx] = make_cuFloatComplex(0, 0);
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
 * @param gmode Int indicating the type of source grid.
 * @param ncx  Int the number of x/u points in the source grid.
 * @param ncy  Int the number of y/v points in the source grid
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
                                int gmode, int ncx, int ncy,
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
                    point, d_A, gmode, ncx, ncy,
                    d_ei, d_hi);

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
 * @param gmode Int indicating the type of source grid.
 * @param ncx  Int the number of x/u points in the source grid.
 * @param ncy  Int the number of y/v points in the source grid
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
                                int gmode, int ncx, int ncy,
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
                    point, d_A, gmode, ncx, ncy,
                    d_ei, d_hi);

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
 * @param gmode Int indicating the type of source grid.
 * @param ncx  Int the number of x/u points in the source grid.
 * @param ncy  Int the number of y/v points in the source grid
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
                                int gmode, int ncx, int ncy,
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
                    point, d_A, gmode, ncx, ncy,
                    d_ei, d_hi);

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
 * @param d_xs Array containing source point's x-coordinate.
 * @param d_ys Array containing source point's y-coordinate.
 * @param d_zs Array containing source point's z-coordinate.
 * @param d_nxs Array containing source point's normal x-component.
 * @param d_nys Array containing source point's y-component.
 * @param d_nzs Array containing source point's z-component.
 * @param d_Jx Array containing source J x-component.
 * @param d_Jy Array containing source J y-component.
 * @param d_Jz Array containing source J z-component.
 * @param d_Mx Array containing source M x-component.
 * @param d_My Array containing source M y-component.
 * @param d_Mz Array containing source M z-component.
 * @param r_hat Array of 3 float, containing xyz coordinates of target point direction.
 * @param d_A Array containing area elements.
 * @param gmode Int indicating the type of source grid.
 * @param ncx  Int the number of x/u points in the source grid.
 * @param ncy  Int the number of y/v points in the source grid.
 * @param e Array of 3 cuFloatComplex, to be filled with E-field at point.
 * @param h Array of 3 cuFloatComplex, to be filled with H-field at point.
 */
__device__ void farfieldAtPoint(float *d_xs, float *d_ys, float *d_zs, float *d_nxs, float *d_nys, float *d_nzs,
                                cuFloatComplex *d_Jx, cuFloatComplex *d_Jy, cuFloatComplex *d_Jz,
                                cuFloatComplex *d_Mx, cuFloatComplex *d_My, cuFloatComplex *d_Mz,
                                float (&r_hat)[3], float *d_A, int gmode, int ncx, int ncy,
                                cuFloatComplex (&e)[3], cuFloatComplex (&h)[3])
{
    int i;                // index in grids
    // Scalars (float & complex float)
    cuFloatComplex exp;                 // Container for the exponential part of the Green's function
    cuFloatComplex Green;               // Container for Green's function
    cuFloatComplex js_dot_R;            // Container for inner products between wavevctor and electric currents
    cuFloatComplex ms_dot_R;            // Container for inner products between wavevctor and magnetics currents


    // Arrays of floats
    float source_point[3]; // Container for xyz co-ordinates
    float source_norm[3];  // Container for xyz source normals
    float source_point_dot_r_hat; // Container for projection of source point onto r_hat
    float norm_dot_R_hat;  // Source normal dotted with wavevector direction

    // Arrays of complex floats
    cuFloatComplex e_field[3] = {con[10], con[10], con[10]}; // Electric field on target
    cuFloatComplex h_field[3] = {con[10], con[10], con[10]}; // Magnetic field on target
    cuFloatComplex ye_field[3]; // Intermediate electric field due to integral over y/v
    cuFloatComplex yh_field[3]; // Intermediate magnetic field due to integral over y/v
    cuFloatComplex de_field[3] = {con[10], con[10], con[10]}; // Electric field contribution from source point on target
    cuFloatComplex dh_field[3] = {con[10], con[10], con[10]}; // Magnetic field contribution from source point on target
    cuFloatComplex js[3];             // Electric current at source point
    cuFloatComplex ms[3];             // Magnetic current at source point
    cuFloatComplex js_dot_R_R[3];     // Electric current contribution to e-field
    cuFloatComplex ms_dot_R_R[3];    // Magnetic current contribution to h-field
    cuFloatComplex R_cross_ms[3];     // Outer product between R_hat and ms
    cuFloatComplex R_cross_js[3];     // Outer product between R_hat and js
    cuFloatComplex e_temp[3];           // Temporary container for intermediate values
    cuFloatComplex h_temp[3];           // Temporary container for intermediate values

    // Integrate over each source point
    // We split the integral into two parts, with the outer loop over the 
    // x/u axis, and the inner loop over the y/v axis
    for(int xu=0; xu<ncx; xu++)
    {
        // Reset yv integral field to zero.
        for (int n=0; n<3; n++) 
        {
            ye_field[n] = con[10]; // Zero intermediate electric field due to integral over y/v
            yh_field[n] = con[10]; // Zero intermediate magnetic field due to integral over y/v
        }

        for(int yv=0; yv<ncy; yv++)
        {
            i = xu*ncy + yv;

            js[0] = d_Jx[i];
            js[1] = d_Jy[i];
            js[2] = d_Jz[i];

            ms[0] = d_Mx[i];
            ms[1] = d_My[i];
            ms[2] = d_Mz[i];

            //printf("ms      : (%.16g+%.16gi, %.16g+%.16gi, %.16g+%.16gi)\n", ms[0].x, ms[0].y, ms[1].x, ms[1].y, ms[2].x, ms[2].y);
            source_point[0] = d_xs[i];
            source_point[1] = d_ys[i];
            source_point[2] = d_zs[i];
            
            source_norm[0] = d_nxs[i];
            source_norm[1] = d_nys[i];
            source_norm[2] = d_nzs[i];

            dot(source_norm, r_hat, norm_dot_R_hat);
            
            if ((norm_dot_R_hat < 0)) {
                continue;}

            // Vector calculations
            // e-field
            // J.rh
            dot(js, r_hat, js_dot_R);
            
            // (J.rh)rh
            s_mult(r_hat, js_dot_R, js_dot_R_R);

            // rh x M
            ext(r_hat, ms, R_cross_ms);

            // h-field
            // M.rh
            dot(ms, r_hat, ms_dot_R);
            
            // (M.rh)rh
            s_mult(r_hat, ms_dot_R, ms_dot_R_R);
            
            // rh x J
            ext(r_hat, js, R_cross_js);
            
            cuFloatComplex d_Ac = make_cuFloatComplex(d_A[i], 0.);
            
            // r'.rhat
            dot(source_point, r_hat, source_point_dot_r_hat);

            // ∓k  con[8]=∓1, con[0]=k
            //cuCmulf(con[8], con[0]);

            // ∓k (-i) r'.rhat = ±ik r'.rhat 
            //cuCmulf(cuCmulf(con[8], con[0]), make_cuFloatComplex(0, -source_point_dot_r_hat)

            // e^{±i k r'.rhat}
            exp = cuCexpf(cuCmulf(cuCmulf(con[8], con[0]), make_cuFloatComplex(0, -source_point_dot_r_hat)));

            // -(ik²/4π) * dA * e^{±i k r'.rhat}, k^2/4π = con[1]
            // cuCmulf(make_cuFloatComplex(0, -1), cuCmulf(con[1], exp))

            Green = cuCmulf(cuCmulf(make_cuFloatComplex(0, -1), cuCmulf(con[1], exp)), make_cuFloatComplex(d_A[i], 0));
            //printf("Green           : %.16g+%.16gi\n", Green.x, Green.y);

            // Field calculations
            // E = -(ik²/4π) * ( Z(J - (J.r)r) + ∓ r x M ) e^(±i k r'.rh) dA
            // H = -(ik²/4π) * ( (1/Z)(J - (J.r)r) - ∓ r x M ) e^(±i k r'.rh) dA
            for( int n=0; n<3; n++)
            {
                de_field[n] =   cuCmulf(
                                    cuCaddf( // Z (M - (M.rh)rh) + ∓ rh x J
                                        cuCsubf(js[n], js_dot_R_R[n]),
                                        cuCmulf(con[8], R_cross_ms[n])  // ∓ rh x M
                                    ), 
                                    Green
                                );

                dh_field[n] =   cuCmulf(
                                    cuCsubf( // (1/Z) (M - (M.rh)rh) - ∓ rh x J
                                        cuCsubf(ms[n], ms_dot_R_R[n]), // M - (M.rh)rh
                                        cuCmulf(con[8], R_cross_js[n]) // ∓ rh x J
                                        ), 
                                        Green
                                    );
                            
                // If this is an integral over an incomplete period of v, or over y/el, only add half of the first and last points
                if ((gmode != 1) && ((yv==0) || (yv==ncy-1)))
                {
                    // printf("Got gmode %d at endpoint, using trapezoidal endpoints.\n", gmode);
                    ye_field[n] = cuCaddf(
                                        cuCmulf(de_field[n], make_cuFloatComplex(0.5,0)), 
                                        ye_field[n]
                                    );
                
                    yh_field[n] = cuCaddf(
                                        cuCmulf(dh_field[n], make_cuFloatComplex(0.5,0)), 
                                        yh_field[n]
                                    );
                }
                else
                {
                    ye_field[n] = cuCaddf(
                                    de_field[n], 
                                    ye_field[n]
                                );
                
                    yh_field[n] = cuCaddf(
                                    dh_field[n], 
                                    yh_field[n]
                                );
                }
            }
            //printf("e_field      : (%.16g+%.16gi, %.16g+%.16gi, %.16g+%.16gi)\n", e_temp[0].x, e_temp[0].y, e_temp[1].x, e_temp[1].y, e_temp[2].x, e_temp[2].y);
            //printf("h_field      : (%.16g+%.16gi, %.16g+%.16gi, %.16g+%.16gi)\n", h_temp[0].x, h_temp[0].y, h_temp[1].x, h_temp[1].y, h_temp[2].x, h_temp[2].y);
        
        } // End of y/v loop
    
        for( int n=0; n<3; n++)
        {
            if ((xu==0) || (xu==ncx-1)) // Only add half the point value
            {
                e_field[n] = cuCaddf(cuCmulf(ye_field[n], make_cuFloatComplex(0.5,0)), e_field[n]);
                h_field[n] = cuCaddf(cuCmulf(yh_field[n], make_cuFloatComplex(0.5,0)), h_field[n]);
            }
            else 
            {
                e_field[n] = cuCaddf(ye_field[n], e_field[n]);
                h_field[n] = cuCaddf(yh_field[n], h_field[n]);
            }
        }
    
    } // End of x/u loops

    e[0] = e_field[0];
    e[1] = e_field[1];
    e[2] = e_field[2];

    h[0] = h_field[0];
    h[1] = h_field[1];
    h[2] = h_field[2];
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
 * @param gmode Int indicating the type of source grid.
 * @param ncx  Int the number of x/u points in the source grid.
 * @param ncy  Int the number of y/v points in the source grid.
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
void __global__ GpropagateBeam_4(float *d_xs, float *d_ys, float *d_zs, float *d_nxs, float *d_nys, float *d_nzs,
                                float *d_A, float *d_xt, float *d_yt,
                                int gmode, int ncx, int ncy,
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
        farfieldAtPoint(d_xs, d_ys, d_zs, d_nxs, d_nys, d_nzs,
                      d_Jx, d_Jy, d_Jz,
                      d_Mx, d_My, d_Mz,
                      r_hat, d_A, gmode, ncx, ncy,
                      e, h);

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
 * @param gmode Int indicating the type of source grid.
 * @param ncx  Int the number of x/u points in the source grid.
 * @param ncy  Int the number of y/v points in the source grid.
 * @param e Array to be filled with results.
 */
void __device__ scalarfieldAtPoint(float *d_xs, float *d_ys, float *d_zs,
                                   cuFloatComplex *d_sfs, float (&point)[3], 
                                   float *d_A, int gmode, int ncx, int ncy,
                                   cuFloatComplex &e)
{
    int i = 0; // grid index
    
    float r;
    float r_vec[3];
    float source_point[3];
    
    e = con[10];  // initialize field to 0+0j
    cuFloatComplex expo;
    cuFloatComplex cfact;
    cuFloatComplex ye; // intermediate field result

    // printf("gmode %d, ncx %d, ncy %d\n", gmode, ncx, ncy);
    // printf("point: (%.5f, %.5f, %.5f)\n", point[0], point[1], point[2]);


    for(int xu=0; xu<ncx; xu++)
    {
        ye = con[10]; // Intermediate field due to integral over y/v

        //printf("gmode %d, ncx %d, ncy %d, xu %d, i %d\n", gmode, ncx, ncy, xu, i);

        for(int yv=0; yv<ncy; yv++)
        {
            i = xu*ncy + yv;

            // if (i % 100 ==0)
            // {
            //     printf("xu, yv, i, source point: %d, %d, %d, (%.5f, %.5f, %.5f)\n", i, xu, yv, d_xs[i], d_ys[i], d_zs[i]);
            // }
        
            source_point[0] = d_xs[i];
            source_point[1] = d_ys[i];
            source_point[2] = d_zs[i];

            diff(point, source_point, r_vec);

            abs(r_vec, r);

            // if (i % 100 ==0)
            // {
            //     printf("i %d, xu %d, yv %d, point (%.5f, %.5f), r %.5f, r_vec: (%.5f, %.5f, %.5f)\n", i, xu, yv, point[0], point[1], r, r_vec[0], r_vec[1], r_vec[2]);
            // }

            expo = cuCexpf(make_cuFloatComplex(0, con[8].x * con[0].x * r));
            cfact = make_cuFloatComplex(-con[0].x * con[0].x / (4 * con[6].x * r) * d_A[i], 0);

            // if (i % 50 == 0)
            // {
            //     printf("i %d, xu %d, yv %d, point (%.5f, %.5f), d_sfs %.5f+%.5fi, cfact %.5f+%.5fi\n", i, xu, yv, point[0], point[1], d_sfs[i].x, d_sfs[i].y, cfact.x*1e3, cfact.y*1e3);
            // }
            
            // If this is an integral over an incomplete period of v, or over y/el, only add half of the first and last points
            if ((gmode != 1) && ((yv==0) || (yv==ncy-1)))
            {
                ye = cuCaddf(cuCmulf(cuCmulf(cuCmulf(cfact, expo), d_sfs[i]), make_cuFloatComplex(0.5, 0)), ye);
            }
            else
            {
                ye = cuCaddf(cuCmulf(cuCmulf(cfact, expo), d_sfs[i]), ye);
            }
        } // end of y/v loop
        
        if ((xu==0) || (xu==ncx-1)) // Only add half the point value
        {
            e = cuCaddf(cuCmulf(ye, make_cuFloatComplex(0.5,0)), e);
        }
        else 
        {
            e = cuCaddf(ye, e);
        }
    } // end of x/u loop
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
 * @param gmode Int indicating the type of source grid.
 * @param ncx  Int the number of x/u points in the source grid.
 * @param ncy  Int the number of y/v points in the source grid.
 * @param d_xt Array containing target direction x-coordinate.
 * @param d_yt Array containing target direction y-coordinate.
 * @param d_zt Array containing target direction z-coordinate.
 * @param d_sfs Array containing source scalar field.
 * @param d_sft Array to be filled with target scalar field.
 */
void __global__ GpropagateBeam_5(float *d_xs, float *d_ys, float *d_zs,
                                float *d_A, 
                                int gmode, int ncx, int ncy,
                                float *d_xt, float *d_yt, float *d_zt,
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
                      d_sfs, point, d_A, gmode, ncx, ncy, e);

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
    cudaFuncSetCacheConfig(GpropagateBeam_0, cudaFuncCachePreferL1);

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
    GpropagateBeam_0_PEC<<<BT[0], BT[1]>>>(vec_ds[0], vec_ds[1], vec_ds[2], vec_ds[3],
                                       vec_dt[0], vec_dt[1], vec_dt[2],
                                       vec_ds[4], vec_ds[5], vec_ds[6],
                                       vec_dt[3], vec_dt[4], vec_dt[5],
                                       source.gmode, source.n_cells[0], source.n_cells[1],
                                       vec_din[0], vec_din[1], vec_din[2],
                                       vec_din[3], vec_din[4], vec_din[5],
                                       vec_dout[0], vec_dout[1], vec_dout[2],
                                       vec_dout[3], vec_dout[4], vec_dout[5]);
    gpuErrchk( cudaPeekAtLastError() );
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
    cudaFuncSetCacheConfig(GpropagateBeam_1, cudaFuncCachePreferL1);

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
                                       source.gmode, source.n_cells[0], source.n_cells[1],
                                       vec_din[0], vec_din[1], vec_din[2],
                                       vec_din[3], vec_din[4], vec_din[5],
                                       vec_dout[0], vec_dout[1], vec_dout[2],
                                       vec_dout[3], vec_dout[4], vec_dout[5]);
    gpuErrchk( cudaPeekAtLastError() );
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
    cudaFuncSetCacheConfig(GpropagateBeam_2, cudaFuncCachePreferL1);

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
    GpropagateBeam_2_PEC<<<BT[0], BT[1]>>>(vec_ds[0], vec_ds[1], vec_ds[2], vec_ds[3],
                                       vec_dt[0], vec_dt[1], vec_dt[2],
                                       vec_ds[4], vec_ds[5], vec_ds[6],
                                       vec_dt[3], vec_dt[4], vec_dt[5],
                                       source.gmode, source.n_cells[0], source.n_cells[1],
                                       vec_din[0], vec_din[1], vec_din[2],
                                       vec_din[3], vec_din[4], vec_din[5],
                                       vec_dout[0], vec_dout[1], vec_dout[2],
                                       vec_dout[3], vec_dout[4], vec_dout[5],
                                       vec_dout[6], vec_dout[7], vec_dout[8],
                                       vec_dout[9], vec_dout[10], vec_dout[11]);
    gpuErrchk( cudaPeekAtLastError() );
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
    cudaFuncSetCacheConfig(GpropagateBeam_3, cudaFuncCachePreferL1);

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
                                       source.gmode, source.n_cells[0], source.n_cells[1],
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
    cudaFuncSetCacheConfig(GpropagateBeam_4, cudaFuncCachePreferL1);

    MemUtils memutil;

    int n_ds = 7;
    int n_dt = 2;
     
    std::vector<float*> vec_csdat = {cs->x, cs->y, cs->z, cs->nx, cs->ny, cs->nz, cs->area};
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

    GpropagateBeam_4<<<BT[0], BT[1]>>>(vec_ds[0], vec_ds[1], vec_ds[2], vec_ds[3], vec_ds[4], vec_ds[5], vec_ds[6],
                                       vec_dt[0], vec_dt[1],
                                       source.gmode, source.n_cells[0], source.n_cells[1], 
                                       vec_din[0], vec_din[1], vec_din[2],
                                       vec_din[3], vec_din[4], vec_din[5],
                                       vec_dout[0], vec_dout[1], vec_dout[2],
                                       vec_dout[3], vec_dout[4], vec_dout[5]);
    gpuErrchk( cudaPeekAtLastError() );
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

    cudaFuncSetCacheConfig(GpropagateBeam_5, cudaFuncCachePreferL1);

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
                                       source.gmode, source.n_cells[0], source.n_cells[1],
                                       vec_dt[0], vec_dt[1], vec_dt[2],
                                       d_sfs, d_sft);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    cuFloatComplex *h_sft = new cuFloatComplex[ct->size];
    gpuErrchk( cudaMemcpy(h_sft, d_sft, ct->size * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost) );

    gpuErrchk( cudaDeviceReset() );

    _arrCUDACToC1(h_sft, res->x, res->y, ct->size);

    delete h_sft;
}
