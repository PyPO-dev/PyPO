#include <iostream>
#include <vector>
#include <complex>
#include <array>
#include <cmath>
#include <thread>
#include <iomanip>
#include <algorithm>
#include <new>

#include "Utils.h"
#include "Structs.h"
#include "InterfaceReflector.h"

#ifndef __Propagation_h
#define __Propagation_h

/*! \file Propagation.h
    \brief Functions for PO calculations on CPU.

    Provides functions for performing PO calculations on the CPU.
*/

/**
 * Main class for running PO calculations.
 *
 * Contains functions that run PO calculations. All calculations can be performed in a parallel way.
 *
 * @see Utils
 * @see Structs
 * @see InterfaceReflector
 */
template <class T, class U, class V, class W>
class Propagation
{
    T k;                /**<Wavenumber of radiation, double/float.*/  
    int numThreads;     /**<Number of computing threads to employ.*/
    int gs;             /**<Number of cells on source surface.*/
    int gt;             /**<Number of cells on target surface.*/

    int step;           /**<Number of threads per block.*/
                      
    T t_direction;      /**<Time direction (experimental!).*/
                      
    T EPS;              /**<Relative electric permittivity of source medium, double/float.*/
    float C_L;          /**<Speed of light in vacuum, in mm / s.*/
    float MU_0;         /**<Magnetic permeability.*/
    float EPS_VAC;      /**<Vacuum electric permittivity.*/
    float ZETA_0_INV;   /**<Conductance of surrounding medium.*/
    float M_PIf;        /**<Floating point pi (redundant?).*/


    std::complex<T> j;  /**<Complex unit.*/
    std::complex<T> z0; /**<Complex zero.*/

    std::array<std::array<T, 3>, 3> eye; /**<Real-valued 3x3 unit matrix.*/
    
    void joinThreads();

    void _debugArray(T *arr, int idx);
    void _debugArray(std::array<T, 3> arr);
    void _debugArray(std::array<std::complex<T>, 3> arr);

public:

    std::vector<std::thread> threadPool;    /**<Vector of thread objects.*/

    Propagation(T k, int numThreads, int gs, int gt, T epsilon, T t_direction, bool verbose = false);

    // Make T precision utility kit
    Utils<T> ut;    /**<Utils object for vector operations.*/

    // Functions for propagating fields between two surfaces
    void propagateBeam_JM(int start, int stop,
                          V *cs, V *ct,
                          W *currents, U *res);

    void propagateBeam_EH(int start, int stop,
                          V *cs, V *ct,
                          W *currents, U *res);

    void propagateBeam_JMEH(int start, int stop,
                          V *cs, V *ct,
                          W *currents, U *res);

    void propagateBeam_EHP(int start, int stop,
                          V *cs, V *ct,
                          W *currents, U *res);

    std::array<std::array<std::complex<T>, 3>, 2> fieldAtPoint(V *cs, W *currents,
                                              const std::array<T, 3> &point_target);


    void parallelProp_JM(V *cs, V *ct,
                        W *currents, U *res);

    void parallelProp_EH(V *cs, V *ct,
                        W *currents, U *res);

    void parallelProp_JMEH(V *cs, V *ct,
                        W *currents, U *res);

    void parallelProp_EHP(V *cs, V *ct,
                        W *currents, U *res);

    // Functions for calculating angular far-field from reflector directly - no phase term
    void propagateToFarField(int start, int stop,
                              V *cs, V *ct,
                              W *currents, U *res);

    std::array<std::array<std::complex<T>, 3>, 2> farfieldAtPoint(V *cs, W *currents,
                                              const std::array<T, 3> &point_target);

    void parallelFarField(V *cs, V *ct,
                        W *currents, U *res);

    // Scalar propagation
    void propagateScalarBeam(int start, int stop,
                          V *cs, V *ct,
                          W *field, U *res);


    std::complex<T> fieldScalarAtPoint(V *cs, W *field,
                                      const std::array<T, 3> &point_target);

    void parallelPropScalar(V *cs, V *ct,
                            W *field, U *res);

};

/**
 * Constructor.
 *
 * Set important parameters internally, given input.
 *
 * @param k Wavenumber of radiation, double/float.
 * @param numThreads Number of computing threads to employ.
 * @param gs Number of cells on source surface.
 * @param gt Number of cells on target grid.
 * @param epsilon Relative electric permittivity of source surface.
 * @param t_direction Time direction (experimental!).
 * @param verbose Whether to print internal state info.
 */
template <class T, class U, class V, class W>
Propagation<T, U, V, W>::Propagation(T k, int numThreads, int gs, int gt, T epsilon, T t_direction, bool verbose)
{
    this->M_PIf = 3.14159265359f;
    this->C_L = 2.99792458e11f; // mm s^-1
    this->MU_0 = 1.2566370614e-3f; // kg mm s^-2 A^-2
    this->EPS_VAC = 1 / (MU_0 * C_L*C_L);
    this->ZETA_0_INV = 1 / (C_L * MU_0);

    std::complex<T> j(0., 1.);
    std::complex<T> z0(0., 0.);
    this->j = j;
    this->z0 = z0;

    this->k = k;

    this->EPS = epsilon * EPS_VAC; // epsilon is relative permeability

    this->numThreads = numThreads;
    this->gs = gs;
    this->gt = gt;

    this->step = ceil(gt / numThreads);

    threadPool.resize(numThreads);

    this->t_direction = t_direction;

    this->eye[0].fill(0);
    this->eye[1].fill(0);
    this->eye[2].fill(0);

    this->eye[0][0] = 1;
    this->eye[1][1] = 1;
    this->eye[2][2] = 1;

    if (verbose)
    {
        printf("***--------- PO info ---------***\n");
        printf("--- Source         :   %d cells\n", gs);
        printf("--- Target         :   %d cells\n", gt);
        printf("--- Wavenumber     :   %.3f / mm\n", k);
        printf("--- Threads        :   %d\n", numThreads);
        printf("--- Device         :   CPU\n");
        printf("***--------- PO info ---------***\n");
        printf("\n");
    }
}

/**
 * Calculate JM on target.
 *
 * Calculate the J, M currents on a target surface.
 *
 * @param start Index of first loop iteration in parallel block.
 * @param stop Index of last loop iteration in parallel block.
 * @param cs Pointer to reflcontainer or reflcontainerf object containing source grids.
 * @param ct Pointer to reflcontainer or reflcontainerf object containing target grids.
 * @param currents Pointer to c2Bundle or c2Bundlef object containing currents on source.
 * @param res Pointer to c2Bundle or c2Bundlef object, to be filled with calculation results.
 *
 * @see reflcontainer
 * @see reflcontainerf
 * @see c2Bundle
 * @see c2Bundlef
 */
template <class T, class U, class V, class W>
void Propagation<T,U, V, W>::propagateBeam_JM(int start, int stop,
                                              V *cs, V *ct,
                                              W *currents, U *res)
{
    // Scalars (T & complex T)
    std::complex<T> e_dot_p_r_perp;    // E-field - perpendicular reflected POI polarization vector dot product
    std::complex<T> e_dot_p_r_parr;    // E-field - parallel reflected POI polarization vector dot product

    // Arrays of Ts
    std::array<T, 3> S_i_norm;         // Normalized incoming Poynting vector
    std::array<T, 3> p_i_perp;         // Perpendicular incoming POI polarization vector
    std::array<T, 3> p_i_parr;         // Parallel incoming POI polarization vector
    std::array<T, 3> S_r_norm;         // Normalized reflected Poynting vector
    std::array<T, 3> p_r_perp;         // Perpendicular reflected POI polarization vector
    std::array<T, 3> p_r_parr;         // Parallel reflected POI polarization vector
    std::array<T, 3> S_out_n;          // Container for Poynting-normal ext products
    std::array<T, 3> point;            // Point on target
    std::array<T, 3> norms;            // Normal vector at point
    std::array<T, 3> e_out_h_r;        // Real part of E-field - H-field ext product

    // Arrays of complex Ts
    std::array<std::complex<T>, 3> e_r;            // Reflected E-field
    std::array<std::complex<T>, 3> h_r;            // Reflected H-field
    std::array<std::complex<T>, 3> n_out_e_i_r;    // Electric current
    std::array<std::complex<T>, 3> temp1;          // Temporary container 1 for intermediate irrelevant values
    std::array<std::complex<T>, 3> temp2;          // Temporary container 2

    std::array<std::complex<T>, 3> jt;
    std::array<std::complex<T>, 3> mt;

    // Return containers
    std::array<std::array<std::complex<T>, 3>, 2> beam_e_h; // Container for storing fieldAtPoint return

    int jc = 0; // Counter

    for(int i=start; i<stop; i++)
    {

        point[0] = ct->x[i];
        point[1] = ct->y[i];
        point[2] = ct->z[i];

        norms[0] = ct->nx[i];
        norms[1] = ct->ny[i];
        norms[2] = ct->nz[i];

        // Calculate total incoming E and H field at point on target
        beam_e_h = fieldAtPoint(cs, currents, point);

        // Calculate normalized incoming poynting vector.
        ut.conj(beam_e_h[1], temp1);                        // h_conj
        ut.ext(beam_e_h[0], temp1, temp2);                  // e_out_h

        for (int n=0; n<3; n++)
        {
            e_out_h_r[n] = temp2[n].real();                      // e_out_h_r
        }

        //std::cout << this->Et_container.size() << std::endl;

        ut.normalize(e_out_h_r, S_i_norm);                       // S_i_norm

        // Calculate incoming polarization vectors
        ut.ext(S_i_norm, norms, S_out_n);                      // S_i_out_n
        ut.normalize(S_out_n, p_i_perp);                       // p_i_perp
        ut.ext(p_i_perp, S_i_norm, p_i_parr);               // p_i_parr

        // Now calculate reflected poynting vector.
        ut.snell(S_i_norm, norms, S_r_norm);                // S_r_norm

        // Calculate normalised reflected polarization vectors
        ut.ext(S_r_norm, norms, S_out_n);                      // S_r_out_n
        ut.normalize(S_out_n, p_r_perp);                       // p_r_perp
        ut.ext(S_r_norm, p_r_perp, p_r_parr);               // p_r_parr

        // Now, calculate reflected field from target
        ut.dot(beam_e_h[0], p_r_perp, e_dot_p_r_perp);      // e_dot_p_r_perp
        ut.dot(beam_e_h[0], p_r_parr, e_dot_p_r_parr);      // e_dot_p_r_parr

        // Calculate reflected field from reflection matrix
        for(int n=0; n<3; n++)
        {
            e_r[n] = -e_dot_p_r_perp * p_i_perp[n] - e_dot_p_r_parr * p_i_parr[n];

            //this->Et_container[k][i] = e_r[k];
        }

        ut.ext(S_r_norm, e_r, temp1);                       // h_r_temp
        ut.s_mult(temp1, ZETA_0_INV, h_r);                  // h_r

        for(int n=0; n<3; n++)
        {
            temp1[n] = e_r[n] + beam_e_h[0][n]; // e_i_r
            temp2[n] = h_r[n] + beam_e_h[1][n]; // h_i_r
        }

        ut.ext(norms, temp2, jt);
        ut.ext(norms, temp1, n_out_e_i_r);
        ut.s_mult(n_out_e_i_r, -1., mt);

        res->r1x[i] = jt[0].real();
        res->r1y[i] = jt[1].real();
        res->r1z[i] = jt[2].real();

        res->i1x[i] = jt[0].imag();
        res->i1y[i] = jt[1].imag();
        res->i1z[i] = jt[2].imag();

        res->r2x[i] = mt[0].real();
        res->r2y[i] = mt[1].real();
        res->r2z[i] = mt[2].real();

        res->i2x[i] = mt[0].imag();
        res->i2y[i] = mt[1].imag();
        res->i2z[i] = mt[2].imag();
    }
}

/**
 * Calculate EH on target.
 *
 * Calculate the E, H fields on a target surface.
 *
 * @param start Index of first loop iteration in parallel block.
 * @param stop Index of last loop iteration in parallel block.
 * @param cs Pointer to reflcontainer or reflcontainerf object containing source grids.
 * @param ct Pointer to reflcontainer or reflcontainerf object containing target grids.
 * @param currents Pointer to c2Bundle or c2Bundlef object containing currents on source.
 * @param res Pointer to c2Bundle or c2Bundlef object, to be filled with calculation results.
 *
 * @see reflcontainer
 * @see reflcontainerf
 * @see c2Bundle
 * @see c2Bundlef
 */
template <class T, class U, class V, class W>
void Propagation<T, U, V, W>::propagateBeam_EH(int start, int stop,
                                              V *cs, V *ct,
                                              W *currents, U *res)
{
    // Arrays of Ts
    std::array<T, 3> point;            // Point on target

    // Return containers
    std::array<std::array<std::complex<T>, 3>, 2> beam_e_h; // Container for storing fieldAtPoint return

    int jc = 0; // Counter

    for(int i=start; i<stop; i++)
    {

        point[0] = ct->x[i];
        point[1] = ct->y[i];
        point[2] = ct->z[i];

        // Calculate total incoming E and H field at point on target
        beam_e_h = fieldAtPoint(cs, currents, point);

        res->r1x[i] = beam_e_h[0][0].real();
        res->r1y[i] = beam_e_h[0][1].real();
        res->r1z[i] = beam_e_h[0][2].real();

        res->i1x[i] = beam_e_h[0][0].imag();
        res->i1y[i] = beam_e_h[0][1].imag();
        res->i1z[i] = beam_e_h[0][2].imag();

        res->r2x[i] = beam_e_h[1][0].real();
        res->r2y[i] = beam_e_h[1][1].real();
        res->r2z[i] = beam_e_h[1][2].real();

        res->i2x[i] = beam_e_h[1][0].imag();
        res->i2y[i] = beam_e_h[1][1].imag();
        res->i2z[i] = beam_e_h[1][2].imag();
    }
}

/**
 * Calculate JM and EH on target.
 *
 * Calculate the J, M currents and E, H fields on a target surface.
 *
 * @param start Index of first loop iteration in parallel block.
 * @param stop Index of last loop iteration in parallel block.
 * @param cs Pointer to reflcontainer or reflcontainerf object containing source grids.
 * @param ct Pointer to reflcontainer or reflcontainerf object containing target grids.
 * @param currents Pointer to c2Bundle or c2Bundlef object containing currents on source.
 * @param res Pointer to c4Bundle or c4Bundlef object, to be filled with calculation results.
 *
 * @see reflcontainer
 * @see reflcontainerf
 * @see c2Bundle
 * @see c2Bundlef
 * @see c4Bundle
 * @see c4Bundlef
 */
template <class T, class U, class V, class W>
void Propagation<T, U, V, W>::propagateBeam_JMEH(int start, int stop,
                                                V *cs, V *ct,
                                                W *currents, U *res)
{
    // Scalars (T & complex T)
    std::complex<T> e_dot_p_r_perp;    // E-field - perpendicular reflected POI polarization vector dot product
    std::complex<T> e_dot_p_r_parr;    // E-field - parallel reflected POI polarization vector dot product

    // Arrays of Ts
    std::array<T, 3> S_i_norm;         // Normalized incoming Poynting vector
    std::array<T, 3> p_i_perp;         // Perpendicular incoming POI polarization vector
    std::array<T, 3> p_i_parr;         // Parallel incoming POI polarization vector
    std::array<T, 3> S_r_norm;         // Normalized reflected Poynting vector
    std::array<T, 3> p_r_perp;         // Perpendicular reflected POI polarization vector
    std::array<T, 3> p_r_parr;         // Parallel reflected POI polarization vector
    std::array<T, 3> S_out_n;          // Container for Poynting-normal ext products
    std::array<T, 3> point;            // Point on target
    std::array<T, 3> norms;            // Normal vector at point
    std::array<T, 3> e_out_h_r;        // Real part of E-field - H-field ext product

    // Arrays of complex Ts
    std::array<std::complex<T>, 3> e_r;            // Reflected E-field
    std::array<std::complex<T>, 3> h_r;            // Reflected H-field
    std::array<std::complex<T>, 3> n_out_e_i_r;    // Electric current
    std::array<std::complex<T>, 3> temp1;          // Temporary container 1 for intermediate irrelevant values
    std::array<std::complex<T>, 3> temp2;          // Temporary container 2

    std::array<std::complex<T>, 3> jt;
    std::array<std::complex<T>, 3> mt;

    // Return containers
    std::array<std::array<std::complex<T>, 3>, 2> beam_e_h; // Container for storing fieldAtPoint return

    int jc = 0; // Counter

    for(int i=start; i<stop; i++)
    {

        point[0] = ct->x[i];
        point[1] = ct->y[i];
        point[2] = ct->z[i];

        norms[0] = ct->nx[i];
        norms[1] = ct->ny[i];
        norms[2] = ct->nz[i];

        // Calculate total incoming E and H field at point on target
        beam_e_h = fieldAtPoint(cs, currents, point);

        //res->r1x[i] = 0.;//beam_e_h[0][0].real();

        res->r3x[i] = beam_e_h[0][0].real();
        res->r3y[i] = beam_e_h[0][1].real();
        res->r3z[i] = beam_e_h[0][2].real();

        res->i3x[i] = beam_e_h[0][0].imag();
        res->i3y[i] = beam_e_h[0][1].imag();
        res->i3z[i] = beam_e_h[0][2].imag();

        res->r4x[i] = beam_e_h[1][0].real();
        res->r4y[i] = beam_e_h[1][1].real();
        res->r4z[i] = beam_e_h[1][2].real();

        res->i4x[i] = beam_e_h[1][0].imag();
        res->i4y[i] = beam_e_h[1][1].imag();
        res->i4z[i] = beam_e_h[1][2].imag();

        // Calculate normalised incoming poynting vector.
        ut.conj(beam_e_h[1], temp1);                        // h_conj

        ut.ext(beam_e_h[0], temp1, temp2);                  // e_out_h

        for (int n=0; n<3; n++)
        {
            e_out_h_r[n] = temp2[n].real();                      // e_out_h_r
        }

        //std::cout << this->Et_container.size() << std::endl;

        ut.normalize(e_out_h_r, S_i_norm);                       // S_i_norm

        // Calculate incoming polarization vectors
        ut.ext(S_i_norm, norms, S_out_n);                      // S_i_out_n

        ut.normalize(S_out_n, p_i_perp);                       // p_i_perp
        ut.ext(p_i_perp, S_i_norm, p_i_parr);               // p_i_parr

        // Now calculate reflected poynting vector.
        ut.snell(S_i_norm, norms, S_r_norm);                // S_r_norm

        // Calculate normalised reflected polarization vectors
        ut.ext(S_r_norm, norms, S_out_n);                      // S_r_out_n

        ut.normalize(S_out_n, p_r_perp);                       // p_r_perp
        ut.ext(S_r_norm, p_r_perp, p_r_parr);               // p_r_parr

        // Now, calculate reflected field from target
        ut.dot(beam_e_h[0], p_r_perp, e_dot_p_r_perp);      // e_dot_p_r_perp
        ut.dot(beam_e_h[0], p_r_parr, e_dot_p_r_parr);      // e_dot_p_r_parr

        //res->r1x[i] = beam_e_h[0][0].real();

        // Calculate reflected field from reflection matrix
        for(int n=0; n<3; n++)
        {
            e_r[n] = -e_dot_p_r_perp * p_i_perp[n] - e_dot_p_r_parr * p_i_parr[n];

            //this->Et_container[k][i] = e_r[k];
        }

        ut.ext(S_r_norm, e_r, temp1);                       // h_r_temp
        ut.s_mult(temp1, ZETA_0_INV, h_r);                  // h_r

        for(int n=0; n<3; n++)
        {
            temp1[n] = e_r[n] + beam_e_h[0][n]; // e_i_r
            temp2[n] = h_r[n] + beam_e_h[1][n]; // h_i_r
        }

        ut.ext(norms, temp2, jt);
        ut.ext(norms, temp1, n_out_e_i_r);
        ut.s_mult(n_out_e_i_r, -1., mt);



        res->r1x[i] = jt[0].real();
        res->r1y[i] = jt[1].real();
        res->r1z[i] = jt[2].real();

        res->i1x[i] = jt[0].imag();
        res->i1y[i] = jt[1].imag();
        res->i1z[i] = jt[2].imag();

        res->r2x[i] = mt[0].real();
        res->r2y[i] = mt[1].real();
        res->r2z[i] = mt[2].real();

        res->i2x[i] = mt[0].imag();
        res->i2y[i] = mt[1].imag();
        res->i2z[i] = mt[2].imag();
    }
}

/**
 * Calculate reflected EH and P on target.
 *
 * Calculate the reflected E, H fields and P, the reflected Poynting vector field, on a target surface.
 *
 * @param start Index of first loop iteration in parallel block.
 * @param stop Index of last loop iteration in parallel block.
 * @param cs Pointer to reflcontainer or reflcontainerf object containing source grids.
 * @param ct Pointer to reflcontainer or reflcontainerf object containing target grids.
 * @param currents Pointer to c2Bundle or c2Bundlef object containing currents on source.
 * @param res Pointer to c2rBundle or c2rBundlef object, to be filled with calculation results.
 *
 * @see reflcontainer
 * @see reflcontainerf
 * @see c2Bundle
 * @see c2Bundlef
 * @see c2rBundle
 * @see c2rBundlef
 */
template <class T, class U, class V, class W>
void Propagation<T, U, V, W>::propagateBeam_EHP(int start, int stop,
                                                V *cs, V *ct,
                                                W *currents, U *res)
{
    // Scalars (T & complex T)
    std::complex<T> e_dot_p_r_perp;    // E-field - perpendicular reflected POI polarization vector dot product
    std::complex<T> e_dot_p_r_parr;    // E-field - parallel reflected POI polarization vector dot product

    // Arrays of Ts
    std::array<T, 3> S_i_norm;         // Normalized incoming Poynting vector
    std::array<T, 3> p_i_perp;         // Perpendicular incoming POI polarization vector
    std::array<T, 3> p_i_parr;         // Parallel incoming POI polarization vector
    std::array<T, 3> S_r_norm;         // Normalized reflected Poynting vector
    std::array<T, 3> p_r_perp;         // Perpendicular reflected POI polarization vector
    std::array<T, 3> p_r_parr;         // Parallel reflected POI polarization vector
    std::array<T, 3> S_out_n;          // Container for Poynting-normal ext products
    std::array<T, 3> point;            // Point on target
    std::array<T, 3> norms;            // Normal vector at point
    std::array<T, 3> e_out_h_r;        // Real part of E-field - H-field ext product

    // Arrays of complex Ts
    std::array<std::complex<T>, 3> e_r;            // Reflected E-field
    std::array<std::complex<T>, 3> h_r;            // Reflected H-field
    std::array<std::complex<T>, 3> n_out_e_i_r;    // Electric current
    std::array<std::complex<T>, 3> temp1;          // Temporary container 1 for intermediate irrelevant values
    std::array<std::complex<T>, 3> temp2;          // Temporary container 2

    // Return containers
    std::array<std::array<std::complex<T>, 3>, 2> beam_e_h; // Container for storing fieldAtPoint return

    int jc = 0; // Counter

    for(int i=start; i<stop; i++)
    {

        point[0] = ct->x[i];
        point[1] = ct->y[i];
        point[2] = ct->z[i];

        norms[0] = ct->nx[i];
        norms[1] = ct->ny[i];
        norms[2] = ct->nz[i];

        // Calculate total incoming E and H field at point on target
        beam_e_h = fieldAtPoint(cs, currents, point);

        // Calculate normalised incoming poynting vector.
        ut.conj(beam_e_h[1], temp1);                        // h_conj
        ut.ext(beam_e_h[0], temp1, temp2);                  // e_out_h

        for (int n=0; n<3; n++)
        {
            e_out_h_r[n] = temp2[n].real();                      // e_out_h_r
        }

        //std::cout << this->Et_container.size() << std::endl;

        ut.normalize(e_out_h_r, S_i_norm);                       // S_i_norm

        // Calculate incoming polarization vectors
        ut.ext(S_i_norm, norms, S_out_n);                      // S_i_out_n
        ut.normalize(S_out_n, p_i_perp);                       // p_i_perp
        ut.ext(p_i_perp, S_i_norm, p_i_parr);               // p_i_parr

        // Now calculate reflected poynting vector.
        ut.snell(S_i_norm, norms, S_r_norm);                // S_r_norm

        // Calculate normalised reflected polarization vectors
        ut.ext(S_r_norm, norms, S_out_n);                      // S_r_out_n
        ut.normalize(S_out_n, p_r_perp);                       // p_r_perp
        ut.ext(S_r_norm, p_r_perp, p_r_parr);               // p_r_parr

        // Now, calculate reflected field from target
        ut.dot(beam_e_h[0], p_r_perp, e_dot_p_r_perp);      // e_dot_p_r_perp
        ut.dot(beam_e_h[0], p_r_parr, e_dot_p_r_parr);      // e_dot_p_r_parr

        // Calculate reflected field from reflection matrix
        for(int n=0; n<3; n++)
        {
            e_r[n] = -e_dot_p_r_perp * p_i_perp[n] - e_dot_p_r_parr * p_i_parr[n];
        }

        ut.ext(S_r_norm, e_r, temp1);                       // h_r_temp
        ut.s_mult(temp1, ZETA_0_INV, h_r);                  // h_r

        res->r1x[i] = e_r[0].real();
        res->r1y[i] = e_r[1].real();
        res->r1z[i] = e_r[2].real();

        res->i1x[i] = e_r[0].imag();
        res->i1y[i] = e_r[1].imag();
        res->i1z[i] = e_r[2].imag();

        res->r2x[i] = h_r[0].real();
        res->r2y[i] = h_r[1].real();
        res->r2z[i] = h_r[2].real();

        res->i2x[i] = h_r[0].imag();
        res->i2y[i] = h_r[1].imag();
        res->i2z[i] = h_r[2].imag();

        res->r3x[i] = S_r_norm[0];
        res->r3y[i] = S_r_norm[1];
        res->r3z[i] = S_r_norm[2];
    }
}

/**
 * Calculate scalar field on target.
 *
 * Calculate the scalar fields on a target surface.
 *
 * @param start Index of first loop iteration in parallel block.
 * @param stop Index of last loop iteration in parallel block.
 * @param cs Pointer to reflcontainer or reflcontainerf object containing source grids.
 * @param ct Pointer to reflcontainer or reflcontainerf object containing target grids.
 * @param field Pointer to arrC1 or arrC1f object containing field on source.
 * @param res Pointer to arrC1 or arrC1f object, to be filled with calculation results.
 *
 * @see reflcontainer
 * @see reflcontainerf
 * @see c2Bundle
 * @see c2Bundlef
 */
template <class T, class U, class V, class W>
void Propagation<T, U, V, W>::propagateScalarBeam(int start, int stop,
                                                  V *cs, V *ct,
                                                  W *field, U *res)
{
    std::array<T, 3> point_target;
    std::complex<T> ets;

    for(int i=start; i<stop; i++)
    {
        point_target[0] = ct->x[i];
        point_target[1] = ct->y[i];
        point_target[2] = ct->z[i];

        ets = fieldScalarAtPoint(cs, field, point_target);

        res->x[i] = ets.real();
        res->y[i] = ets.imag();
    }
}

/**
 * Calculate field on target.
 *
 * Calculate integrated E and H fields on a target point.
 *
 * @param cs Pointer to reflcontainer or reflcontainerf object containing source grids.
 * @param currents Pointer to c2Bundle or c2Bundlef object containing currents on source.
 * @param point_target Array of 3 double/float containing target point co-ordinate.
 *
 * @see reflcontainer
 * @see reflcontainerf
 * @see c2Bundle
 * @see c2Bundlef
 */
template <class T, class U, class V, class W>
std::array<std::array<std::complex<T>, 3>, 2> Propagation<T, U, V, W>::fieldAtPoint(V *cs,
                                                                             W *currents, const std::array<T, 3> &point_target)
{
    // Scalars (T & complex T)
    T r;                           // Distance between source and target points
    T r_inv;                       // 1 / r
    T omega;                       // Angular frequency of field
    T norm_dot_k_hat;              // Source normal dotted with wavevector direction
    std::complex<T> Green;         // Container for Green's function
    std::complex<T> r_in_s;        // Container for inner products between wavevctor and currents

    // Arrays of Ts
    std::array<T, 3> source_point; // Container for xyz co-ordinates
    std::array<T, 3> source_norm;  // Container for xyz normals.
    std::array<T, 3> r_vec;        // Distance vector between source and target points
    std::array<T, 3> k_hat;        // Unit wavevctor
    std::array<T, 3> k_arr;        // Wavevector

    // Arrays of complex Ts
    std::array<std::complex<T>, 3> e_field;        // Electric field on target
    std::array<std::complex<T>, 3> h_field;        // Magnetic field on target
    std::array<std::complex<T>, 3> js;             // Electric current at source point
    std::array<std::complex<T>, 3> ms;             // Magnetic current at source point
    std::array<std::complex<T>, 3> e_vec_thing;    // Electric current contribution to e-field
    std::array<std::complex<T>, 3> h_vec_thing;    // Magnetic current contribution to h-field
    std::array<std::complex<T>, 3> k_out_ms;       // Outer product between k and ms
    std::array<std::complex<T>, 3> k_out_js;       // Outer product between k and js
    std::array<std::complex<T>, 3> temp;           // Temporary container for intermediate values

    // Return container
    std::array<std::array<std::complex<T>, 3>, 2> e_h_field; // Return container containing e and h-fields

    e_field.fill(z0);
    h_field.fill(z0);

    omega = C_L * k;

    for(int i=0; i<gs; i++)
    {
        source_point[0] = cs->x[i];
        source_point[1] = cs->y[i];
        source_point[2] = cs->z[i];
        
        source_norm[0] = cs->nx[i];
        source_norm[1] = cs->ny[i];
        source_norm[2] = cs->nz[i];

        js[0] = {currents->r1x[i], currents->i1x[i]};
        js[1] = {currents->r1y[i], currents->i1y[i]};
        js[2] = {currents->r1z[i], currents->i1z[i]};

        ms[0] = {currents->r2x[i], currents->i2x[i]};
        ms[1] = {currents->r2y[i], currents->i2y[i]};
        ms[2] = {currents->r2z[i], currents->i2z[i]};

        ut.diff(point_target, source_point, r_vec);
        ut.abs(r_vec, r);

        r_inv = 1 / r;

        ut.s_mult(r_vec, r_inv, k_hat);

        // Check if source point illuminates target point or not.
        ut.dot(source_norm, k_hat, norm_dot_k_hat);
        if (norm_dot_k_hat < 0) {continue;}

        ut.s_mult(k_hat, k, k_arr);

        // e-field
        ut.dot(k_hat, js, r_in_s);
        ut.s_mult(k_hat, r_in_s, temp);
        ut.diff(js, temp, e_vec_thing);

        ut.ext(k_arr, ms, k_out_ms);

        // h-field
        ut.dot(k_hat, ms, r_in_s);
        ut.s_mult(k_hat, r_in_s, temp);
        ut.diff(ms, temp, h_vec_thing);

        ut.ext(k_arr, js, k_out_js);

        //printf("%.16g\n", r);

        Green = exp(this->t_direction * j * k * r) / (4 * M_PIf * r) * cs->area[i] * j;

        for( int n=0; n<3; n++)
        {
            e_field[n] += (-omega * MU_0 * e_vec_thing[n] + k_out_ms[n]) * Green;
            h_field[n] += (-omega * EPS * h_vec_thing[n] - k_out_js[n]) * Green;
        }
        //printf("%.16g, %.16g\n", Green.real(), Green.imag()); // %s is format specifier
    }

    // Pack e and h together in single container
    e_h_field[0] = e_field;
    e_h_field[1] = h_field;

    //std::cout << ut.abs(e_field) << std::endl;

    return e_h_field;
}

/**
 * Calculate scalar field on target.
 *
 * Calculate integrated scalar field on a target point.
 *
 * @param cs Pointer to reflcontainer or reflcontainerf object containing source grids.
 * @param field Pointer to arrC1 or arrC1f object containing currents on source.
 * @param point_target Array of 3 double/float containing target point co-ordinate.
 *
 * @see reflcontainer
 * @see reflcontainerf
 * @see arrC1
 * @see arrC1f
 */
template <class T, class U, class V, class W>
std::complex<T> Propagation<T, U, V, W>::fieldScalarAtPoint(V *cs,
                                   W *field, const std::array<T, 3> &point_target)
{
    std::complex<T> out(0., 0.);
    std::complex<T> j(0., 1.);
    std::complex<T> _field;

    T r;
    std::array<T, 3> r_vec;
    std::array<T, 3> source_point;

    for(int i=0; i<gs; i++)
    {
        source_point[0] = cs->x[i];
        source_point[1] = cs->y[i];
        source_point[2] = cs->z[i];

        _field = {field->x[i], field->y[i]};

        ut.diff(point_target, source_point, r_vec);
        ut.abs(r_vec, r);

        out += - k * k * _field * exp(this->t_direction * j * k * r) / (4 * M_PIf * r) * cs->area[i];

    }
    return out;
}

/**
 * Calculate JM parallel.
 *
 * Run the propagateBeam_JM function in parallel blocks.
 *
 * @param cs Pointer to reflcontainer or reflcontainerf object containing source grids.
 * @param ct Pointer to reflcontainer or reflcontainerf object containing target grids.
 * @param currents Pointer to c2Bundle or c2Bundlef object containing source currents.
 * @param res Pointer to c2Bundle or c2Bundlef object to be filled with calculation results.
 *
 * @see reflcontainer
 * @see reflcontainerf
 * @see c2Bundle
 * @see c2Bundlef
 */
template <class T, class U, class V, class W>
void Propagation<T, U, V, W>::parallelProp_JM(V *cs, V *ct,
                                              W *currents, U *res)
{
    int final_step;

    for(int n=0; n<numThreads; n++)
    {
        //std::cout << n << std::endl;
        if(n == (numThreads-1))
        {
            final_step = gt;
        }

        else
        {
            final_step = (n+1) * step;
        }

        //std::cout << final_step << std::endl;

        threadPool[n] = std::thread(&Propagation::propagateBeam_JM,
                                    this, n * step, final_step,
                                    cs, ct, currents, res);
    }
    joinThreads();
}

/**
 * Calculate EH parallel.
 *
 * Run the propagateBeam_EH function in parallel blocks.
 *
 * @param cs Pointer to reflcontainer or reflcontainerf object containing source grids.
 * @param ct Pointer to reflcontainer or reflcontainerf object containing target grids.
 * @param currents Pointer to c2Bundle or c2Bundlef object containing source currents.
 * @param res Pointer to c2Bundle or c2Bundlef object to be filled with calculation results.
 *
 * @see reflcontainer
 * @see reflcontainerf
 * @see c2Bundle
 * @see c2Bundlef
 */
template <class T, class U, class V, class W>
void Propagation<T, U, V, W>::parallelProp_EH(V *cs, V *ct,
                                              W *currents, U *res)
{
    int final_step;

    for(int n=0; n<numThreads; n++)
    {
        //std::cout << n << std::endl;
        if(n == (numThreads-1))
        {
            final_step = gt;
        }

        else
        {
            final_step = (n+1) * step;
        }

        //std::cout << final_step << std::endl;

        threadPool[n] = std::thread(&Propagation::propagateBeam_EH,
                                    this, n * step, final_step,
                                    cs, ct, currents, res);
    }
    joinThreads();
}

/**
 * Calculate JM and EH parallel.
 *
 * Run the propagateBeam_JMEH function in parallel blocks.
 *
 * @param cs Pointer to reflcontainer or reflcontainerf object containing source grids.
 * @param ct Pointer to reflcontainer or reflcontainerf object containing target grids.
 * @param currents Pointer to c2Bundle or c2Bundlef object containing source currents.
 * @param res Pointer to c4Bundle or c4Bundlef object to be filled with calculation results.
 *
 * @see reflcontainer
 * @see reflcontainerf
 * @see c2Bundle
 * @see c2Bundlef
 * @see c4Bundle
 * @see c4Bundlef
 */
template <class T, class U, class V, class W>
void Propagation<T, U, V, W>::parallelProp_JMEH(V *cs, V *ct,
                                                W *currents, U *res)
{
    int final_step;

    for(int n=0; n<numThreads; n++)
    {
        if(n == (numThreads-1))
        {
            final_step = gt;
        }

        else
        {
            final_step = (n+1) * step;
        }

        threadPool[n] = std::thread(&Propagation::propagateBeam_JMEH,
                                    this, n * step, final_step,
                                    cs, ct, currents, res);
    }
    joinThreads();
}

/**
 * Calculate reflected EH and P parallel.
 *
 * Run the propagateBeam_EHP function in parallel blocks.
 *
 * @param cs Pointer to reflcontainer or reflcontainerf object containing source grids.
 * @param ct Pointer to reflcontainer or reflcontainerf object containing target grids.
 * @param currents Pointer to c2Bundle or c2Bundlef object containing source currents.
 * @param res Pointer to c2rBundle or c2rBundlef object to be filled with calculation results.
 *
 * @see reflcontainer
 * @see reflcontainerf
 * @see c2Bundle
 * @see c2Bundlef
 * @see c2rBundle
 * @see c2rBundlef
 */
template <class T, class U, class V, class W>
void Propagation<T, U, V, W>::parallelProp_EHP(V *cs, V *ct,
                                              W *currents, U *res)
{
    int final_step;

    for(int n=0; n<numThreads; n++)
    {
        //std::cout << n << std::endl;
        if(n == (numThreads-1))
        {
            final_step = gt;
        }

        else
        {
            final_step = (n+1) * step;
        }

        threadPool[n] = std::thread(&Propagation::propagateBeam_EHP,
                                    this, n * step, final_step,
                                    cs, ct, currents, res);
    }
    joinThreads();
}

/**
 * Calculate scalar field parallel.
 *
 * Run the propagateScalarBeam function in parallel blocks.
 *
 * @param cs Pointer to reflcontainer or reflcontainerf object containing source grids.
 * @param ct Pointer to reflcontainer or reflcontainerf object containing target grids.
 * @param field Pointer to arrC1 or arrC1f object containing source currents.
 * @param res Pointer to arrC1 or arrC1f object to be filled with calculation results.
 *
 * @see reflcontainer
 * @see reflcontainerf
 * @see arrC1
 * @see arrC1f
 */
template <class T, class U, class V, class W>
void Propagation<T, U, V, W>::parallelPropScalar(V *cs, V *ct,
                                              W *field, U *res)
{
    int final_step;

    for(int n=0; n<numThreads; n++)
    {
        if(n == (numThreads-1))
        {
            final_step = gt;
        }

        else
        {
            final_step = (n+1) * step;
        }

        threadPool[n] = std::thread(&Propagation::propagateScalarBeam,
                                    this, n * step, final_step,
                                    cs, ct, field, res);
    }
    joinThreads();
}

/**
 * Calculate far-field on target.
 *
 * Calculate integrated E and H fields on a far-field target point.
 *
 * @param start Index of first loop iteration in parallel block.
 * @param stop Index of last loop iteration in parallel block.
 * @param cs Pointer to reflcontainer or reflcontainerf object containing source grids.
 * @param ct Pointer to reflcontainer or reflcontainerf object containing target grids.
 * @param currents Pointer to c2Bundle or c2Bundlef object containing currents on source.
 * @param res Pointer to c2Bundle or c2Bundlef object, to be filled with calculation results.
 *
 * @see reflcontainer
 * @see reflcontainerf
 * @see c2Bundle
 * @see c2Bundlef
 */
template <class T, class U, class V, class W>
void Propagation<T, U, V, W>::propagateToFarField(int start, int stop,
                                              V *cs, V *ct,
                                              W *currents, U *res)
{
    // Scalars (T & complex T)
    T theta;
    T phi;
    T cosEl;

    std::complex<T> e_th;
    std::complex<T> e_ph;

    std::complex<T> e_Az;
    std::complex<T> e_El;

    // Arrays of Ts
    std::array<T, 2> point_ff;            // Angular point on far-field
    std::array<T, 3> r_hat;                // Unit vector in far-field point direction

    // Arrays of complex Ts
    std::array<std::array<std::complex<T>, 3>, 2> eh;            // far-field EH-field

    int jc = 0;
    for(int i=start; i<stop; i++)
    {
        phi     = ct->x[i];
        theta   = ct->y[i];
        cosEl   = std::sqrt(1 - sin(theta) * sin(phi) * sin(theta) * sin(phi));

        r_hat[0] = cos(theta) * sin(phi);
        r_hat[1] = sin(theta) * sin(phi);
        r_hat[2] = cos(phi);

        // Calculate total incoming E and H field at point on target
        eh = farfieldAtPoint(cs, currents, r_hat);

        res->r1x[i] = eh[0][0].real();
        res->r1y[i] = eh[0][1].real();
        res->r1z[i] = eh[0][2].real();

        res->i1x[i] = eh[0][0].imag();
        res->i1y[i] = eh[0][1].imag();
        res->i1z[i] = eh[0][2].imag();

        // TODO: Calculate H far fields
        res->r2x[i] = eh[1][0].real();
        res->r2y[i] = eh[1][1].real();
        res->r2z[i] = eh[1][2].real();
                                      
        res->i2x[i] = eh[1][0].imag();
        res->i2y[i] = eh[1][1].imag();
        res->i2z[i] = eh[1][2].imag();
    }
}

/**
 * Calculate far-field on target.
 *
 * Calculate integrated E and H fields on a far-field target point.
 *
 * @param cs Pointer to reflcontainer or reflcontainerf object containing source grids.
 * @param currents Pointer to c2Bundle or c2Bundlef object containing currents on source.
 * @param r_hat Array of 3 double/float containing target direction angles.
 *
 * @see reflcontainer
 * @see reflcontainerf
 * @see c2Bundle
 * @see c2Bundlef
 */
template <class T, class U, class V, class W>
std::array<std::array<std::complex<T>, 3>, 2> Propagation<T, U, V, W>::farfieldAtPoint(V *cs,
                                                W *currents,
                                                const std::array<T, 3> &r_hat)
{
    // Scalars (T & complex T)
    T omega_mu;                       // Angular frequency of field times mu
    T omega_eps;                       // Angular frequency of field times eps
    T r_hat_in_rp;                 // r_hat dot product r_prime
    std::complex<T> r_in_s;        // Container for inner products between wavevctor and currents
    std::complex<T> expo;
    T area;

    // Arrays of Ts
    std::array<T, 3> source_point; // Container for xyz co-ordinates

    // Arrays of complex Ts
    std::array<std::complex<T>, 3> e;        // Electric field on far-field point
    std::array<std::complex<T>, 3> h;        // Magnetic field on far-field point
    std::array<std::complex<T>, 3> _js;      // Temporary Electric current at source point
    std::array<std::complex<T>, 3> _ms;      // Temporary Magnetic current at source point

    std::array<std::complex<T>, 3> js;      // Build radiation integral
    std::array<std::complex<T>, 3> ms;      // Build radiation integral

    std::array<std::complex<T>, 3> _ctemp;
    std::array<std::complex<T>, 3> js_tot_factor;
    std::array<std::complex<T>, 3> ms_tot_factor;
    std::array<std::complex<T>, 3> js_tot_factor_h;
    std::array<std::complex<T>, 3> ms_tot_factor_h;

    // Output array
    std::array<std::array<std::complex<T>, 3>, 2> out;

    // Matrices
    std::array<std::array<T, 3>, 3> rr_dyad;       // Dyadic product between r_hat - r_hat
    std::array<std::array<T, 3>, 3> eye_min_rr;    // I - rr

    e.fill(z0);
    h.fill(z0);
    js.fill(z0);
    ms.fill(z0);

    omega_mu = C_L * k * MU_0;

    ut.dyad(r_hat, r_hat, rr_dyad);
    ut.matDiff(this->eye, rr_dyad, eye_min_rr);

    for(int i=0; i<gs; i++)
    {
        source_point[0] = cs->x[i];
        source_point[1] = cs->y[i];
        source_point[2] = cs->z[i];

        ut.dot(r_hat, source_point, r_hat_in_rp);

        expo = exp(j * k * r_hat_in_rp);
        area = cs->area[i];

        _js[0] = {currents->r1x[i], currents->i1x[i]};
        _js[1] = {currents->r1y[i], currents->i1y[i]};
        _js[2] = {currents->r1z[i], currents->i1z[i]};

        _ms[0] = {currents->r2x[i], currents->i2x[i]};
        _ms[1] = {currents->r2y[i], currents->i2y[i]};
        _ms[2] = {currents->r2z[i], currents->i2z[i]};

        for (int n=0; n<3; n++)
        {
            js[n] += _js[n] * expo * area;
            ms[n] += _ms[n] * expo * area;
        }
    }

    ut.matVec(eye_min_rr, js, _ctemp);
    ut.s_mult(_ctemp, omega_mu, js_tot_factor);

    ut.ext(r_hat, ms, _ctemp);
    ut.s_mult(_ctemp, k, ms_tot_factor);

    omega_eps = C_L * k * EPS;
    
    ut.matVec(eye_min_rr, ms, _ctemp);
    ut.s_mult(_ctemp, omega_eps, ms_tot_factor_h);

    ut.ext(r_hat, js, _ctemp);
    ut.s_mult(_ctemp, k, js_tot_factor_h);
    
    for (int n=0; n<3; n++)
    {
        e[n] = -js_tot_factor[n] + ms_tot_factor[n];
        h[n] = -ms_tot_factor[n] - js_tot_factor[n];
    }

    out[0] = e;
    out[1] = h;
    return out;
}

/**
 * Calculate far-field parallel.
 *
 * Run the propagateToFarField function in parallel blocks.
 *
 * @param cs Pointer to reflcontainer or reflcontainerf object containing source grids.
 * @param ct Pointer to reflcontainer or reflcontainerf object containing target grids.
 * @param currents Pointer to c2Bundle or c2Bundlef object containing source currents.
 * @param res Pointer to c2Bundle or c2Bundlef object to be filled with calculation results.
 *
 * @see reflcontainer
 * @see reflcontainerf
 * @see c2Bundle
 * @see c2Bundlef
 */
template <class T, class U, class V, class W>
void Propagation<T, U, V, W>::parallelFarField(V *cs, V *ct,
                                              W *currents, U *res)
{
    int final_step;

    for(int n=0; n<numThreads; n++)
    {
        if(n == (numThreads-1))
        {
            final_step = gt;
        }

        else
        {
            final_step = (n+1) * step;
        }

        threadPool[n] = std::thread(&Propagation::propagateToFarField,
                                    this, n * step, final_step,
                                    cs, ct, currents, res);
    }
    joinThreads();
}

template <class T, class U, class V, class W>
void Propagation<T, U, V, W>::joinThreads()
{
    for (std::thread &t : threadPool)
    {
        if (t.joinable())
        {
            t.join();
        }
    }
}

template <class T, class U, class V, class W>
void Propagation<T, U, V, W>::_debugArray(T *arr, int idx)
{
    T toPrint = arr[idx];
    std::cout << "Value of c-style array, element " << idx << ", is : " << toPrint << std::endl;
}

template <class T, class U, class V, class W>
void Propagation<T, U, V, W>::_debugArray(std::array<T, 3> arr)
{
    std::cout << arr[0] << ", " << arr[1] << ", " << arr[2] << std::endl;
}

template <class T, class U, class V, class W>
void Propagation<T, U, V, W>::_debugArray(std::array<std::complex<T>, 3> arr)
{
    std::cout << arr[0].real() << " + " << arr[0].imag() << "j"
                <<  ", " << arr[1].real() << " + " << arr[1].imag() << "j"
                <<  ", " << arr[2].real() << " + " << arr[2].imag() << "j"
                <<  std::endl;
}


#endif
