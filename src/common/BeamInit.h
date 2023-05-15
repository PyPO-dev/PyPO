#include <iostream>
#include <array>
#include <cmath>
#include <complex>

#include "Utils.h"
#include "Structs.h"
#include "InterfaceReflector.h"
#include "Random.h"

#define _USE_MATH_DEFINES

#ifndef __BeamInit_h
#define __BeamInit_h

/*! \file BeamInit.h
    \brief Initialize beam objects.
    
    Initializes ray-trace frames, PO beams, and custom beams.
*/

/** 
 * Initialize ray-trace frame from RTDict or RTDictf.
 *
 * Takes an RTDict or RTDictf and generates a frame object, which can be used 
 *      to initialize a ray-trace.
 *
 * @param rdict RTDict or RTDictf object from which to generate a frame.
 * @param fr Pointer to cframe or cframef object.
 * 
 * @see RTDict
 * @see RTDictf
 * @see cframe
 * @see cframef
 */
template<typename T, typename U, typename V>
void initFrame(T rdict, U *fr);

/** 
 * Initialize Gaussian ray-trace frame from RTDict or RTDictf.
 *
 * Takes a GRTDict or GRTDictf and generates a frame object, which can be used 
 *      to initialize a Gaussian ray-trace.
 *
 * @param grdict GRTDict or GRTDictf object from which to generate a frame.
 * @param fr Pointer to cframe or cframef object.
 * 
 * @see GRTDict
 * @see GRTDictf
 * @see cframe
 * @see cframef
 */
template<typename T, typename U, typename V>
void initRTGauss(T grdict, U *fr);

/**
 * Probability density function for drawing ray positions and tilts from Gaussian.
 *
 * The density function is used for generating Gaussian ray-trace beams.
 * Using rejection sampling, the method draws samples from the Gaussian pdf.
 *
 * @param vars Vector of length 4, containing the xy positions and tilts of the ray to be checked.
 * @param scales Vector of length 4 containing the scale factors along xy and tilts
 */
template<typename T>
T pdfGauss(std::vector<T> vars, std::vector<T> scales);

/** 
 * Initialize Gaussian beam from GPODict or GPODictf.
 *
 * Takes a GPODict or GPODictf and generates two c2Bundle or c2Bundlef objects, which contain the field and 
 *      associated currents and are allocated to passed pointer arguments.
 *
 * @param gdict GPODict or GPODictf object from which to generate a Gaussian beam.
 * @param refldict reflparams or reflparamsf object corresponding to surface on
 *      which to generate the Gaussian beam.
 * @param res_field Pointer to c2Bundle or c2Bundlef object.
 * @param res_current Pointer to c2Bundle or c2Bundlef object.
 *
 * @see GPODict
 * @see GPODictf
 * @see reflparams
 * @see reflparamsf
 * @see c2Bundle
 * @see c2Bundlef
 */
template<typename T, typename U, typename V, typename W, typename G>
void initGauss(T gdict, U refldict, V *res_field, V *res_current);

/** 
 * Initialize scalar Gaussian beam from GPODict or GPODictf.
 *
 * Takes a ScalarGPODict or ScalarGPODictf and generates an arrC1 or arrC1f object.
 *
 * @param gdict ScalarGPODict or ScalarGPODictf object from which to generate a Gaussian beam.
 * @param refldict reflparams or reflparamsf object corresponding to surface on
 *      which to generate the Gaussian beam.
 * @param res_field Pointer to arrC1 or arrC1f object.
 *
 * @see ScalarGPODict
 * @see ScalarGPODictf
 * @see reflparams
 * @see reflparamsf
 * @see arrC1
 * @see arrC1f
 */
template<typename T, typename U, typename V, typename W, typename G>
void initScalarGauss(T sgdict, U refldict, V *res_field);

/** 
 * Calculate currents from electromagnetic field.
 * 
 * Calculate the J and M vectorial currents given a vectorial E and H field.
 *      Can calculate full currents, PMC and PEC surfaces.
 *
 * @param res_field Pointer to c2Bundle or c2Bundlef object.
 * @param res_current Pointer to c2Bundle or c2Bundlef object.
 * @param refldict reflparams or reflparamsf object corresponding to surface on
 *      which to calculate currents.
 * @param mode How to calculate currents. 0 is full currents, 1 is PMC and 2 is PEC.
 *
 * @see c2Bundle
 * @see c2Bundlef
 * @see reflparams
 * @see reflparamsf
 */
template<typename T, typename U, typename V, typename W>
void calcJM(T *res_field, T *res_current, V refldict, int mode);

template<typename T, typename U, typename V>
void initFrame(T rdict, U *fr)
{
    std::array<V, 3> nomChief = {0, 0, 1};
    std::array<V, 3> zero = {0, 0, 0};

    Utils<V> ut;

    int nTot = 1 + rdict.nRing * 4 * rdict.nRays;

    fr->size = nTot;

    V alpha = 0; // Set first circle ray in the right of beam
    V d_alpha = 0;

    if (rdict.nRays > 0) {d_alpha = 2 * M_PI / (4 * rdict.nRays);}

    int n = 1;

    std::array<V, 3> rotation;
    std::array<V, 3> direction;
    
    fr->x[0] = 0.;
    fr->y[0] = 0.;
    fr->z[0] = 0.;

    fr->dx[0] = 0.;
    fr->dy[0] = 0.;
    fr->dz[0] = 1.;

    std::array<V, 3> pos;

    for (int i=1; i<nTot; i++)
    {
        fr->x[i] = rdict.x0 * cos(alpha) / rdict.nRing * n;
        fr->y[i] = rdict.y0 * sin(alpha) / rdict.nRing * n;
        fr->z[i] = 0.;

        rotation[0] = rdict.angy0 * sin(alpha) / rdict.nRing * n;
        rotation[1] = rdict.angx0 * cos(alpha) / rdict.nRing * n;
        rotation[2] = 2 * alpha;

        ut.matRot(rotation, nomChief, zero, direction);

        fr->dx[i] = direction[0];
        fr->dy[i] = direction[1];
        fr->dz[i] = direction[2];
        alpha += d_alpha;

        if (i == int(nTot / rdict.nRing) * n)
        {
            n += 1;
            alpha = 0;
        }
    }
}

template<typename T, typename U, typename V>
void initRTGauss(T grdict, U *fr)
{
    V thres = 3.; // Choose 3-sigma point
    std::array<V, 3> nomChief = {0, 0, 1};
    std::array<V, 3> zero = {0, 0, 0};

    Utils<V> ut;
    Random<V> rando;
    if (grdict.seed != -1) {Random<V> rando(grdict.seed);}

    fr->size = grdict.nRays;
    
    std::array<V, 3> rotation;
    std::array<V, 3> direction;
    std::array<V, 3> pos;

    // Initialize scale vector
    std::vector<V> scales{grdict.x0, grdict.y0, grdict.angx0, grdict.angy0};

    fr->x[0] = 0.;
    fr->y[0] = 0.;
    fr->z[0] = 0.;

    fr->dx[0] = 0.;
    fr->dy[0] = 0.;
    fr->dz[0] = 1.;

    // Start rejection sampling. Use n_suc as succes counter
    int n_suc = 1;
    V yi;
    std::vector<V> xi(4, 0);
    V lower = 0.0;

    while (n_suc < grdict.nRays)
    {
       // First, generate y-value between 0 and 1.
       yi = rando.generateUniform(lower);
       
       // Now, generate vector of xi
       xi = rando.generateUniform(grdict.nRays);

       for (int k = 0; k<4; k++) {xi[k] = xi[k] * thres * scales[k];}
       if (pdfGauss<V>(xi, scales) > yi) 
       {
           // Rotate chief ray by tilt angles found
           rotation  = {xi[3], xi[2], 0};
           ut.matRot(rotation, nomChief, zero, direction);
           //std::cout << ddirection[2] << std::endl;
           pos = {xi[0], xi[1], 0};

           fr->x[n_suc] = pos[0];
           fr->y[n_suc] = pos[1];
           fr->z[n_suc] = pos[2];

           fr->dx[n_suc] = direction[0];
           fr->dy[n_suc] = direction[1];
           fr->dz[n_suc] = direction[2];
           n_suc++;
       }
    }
}

template<typename T>
T pdfGauss(std::vector<T> vars, std::vector<T> scales)
{
    T norm = 1 / (M_PI*M_PI * scales[0] * scales[1] * scales[2] * scales[3]);

    return norm * exp(-vars[0]*vars[0] / (scales[0]*scales[0])) * exp(-vars[1]*vars[1] / (scales[1]*scales[1])) * 
                    exp(-vars[2]*vars[2] / (scales[2]*scales[2])) * exp(-vars[3]*vars[3] / (scales[3]*scales[3]));
}

template<typename T, typename U, typename V, typename W, typename G>
void initGauss(T gdict, U refldict, V *res_field, V *res_current)
{
    int nTot = refldict.n_cells[0] * refldict.n_cells[1];

    W reflc;

    reflc.size = nTot;

    reflc.x = new G[nTot];
    reflc.y = new G[nTot];
    reflc.z = new G[nTot];

    reflc.nx = new G[nTot];
    reflc.ny = new G[nTot];
    reflc.nz = new G[nTot];

    reflc.area = new G[nTot];

    Utils<G> ut;

    bool transform = true;
    generateGrid(refldict, &reflc, transform);


    G zRx      = M_PI * gdict.w0x*gdict.w0x * gdict.n / gdict.lam;
    G zRy      = M_PI * gdict.w0y*gdict.w0y * gdict.n / gdict.lam;
    G k        = 2 * M_PI / gdict.lam;
    
    G wzx;      
    G wzy;     
    G Rzx_inv;  
    G Rzy_inv;  
    G phizx;    
    G phizy;    

    std::complex<G> j(0, 1);

    std::complex<G> field_atPoint;
    std::array<std::complex<G>, 3> efield;
    std::array<G, 3> n_source;
    std::array<std::complex<G>, 3> m;

    for (int i=0; i<nTot; i++)
    {
        wzx      = gdict.w0x * sqrt(1 + (reflc.z[i] / zRx)*(reflc.z[i] / zRx));
        wzy      = gdict.w0y * sqrt(1 + ((reflc.z[i] - gdict.dxyz) / zRy)*((reflc.z[i] - gdict.dxyz) / zRy));
        Rzx_inv  = reflc.z[i] / (reflc.z[i]*reflc.z[i] + zRx*zRx);
        Rzy_inv  = (reflc.z[i] - gdict.dxyz) / ((reflc.z[i] - gdict.dxyz)*(reflc.z[i] - gdict.dxyz) + zRy*zRy);
        phizx    = atan(reflc.z[i] / zRx);
        phizy    = atan((reflc.z[i] - gdict.dxyz) / zRy);
        
        //field_atPoint = gdict.E0 * sqrt(2 / (M_PI * wzx * wzy)) * exp(-(reflc.x[i]/wzx)*(reflc.x[i]/wzx) - (reflc.y[i]/wzy)*(reflc.y[i]/wzy) -
        //        j*M_PI/gdict.lam * (reflc.x[i]*reflc.x[i]*Rzx_inv + reflc.y[i]*reflc.y[i]*Rzy_inv) - j*k*reflc.z[i] + j*(phizx - phizy)*0.5);
        
        field_atPoint = gdict.E0 * exp(-(reflc.x[i]/wzx)*(reflc.x[i]/wzx) - (reflc.y[i]/wzy)*(reflc.y[i]/wzy) -
                j*M_PI/gdict.lam * (reflc.x[i]*reflc.x[i]*Rzx_inv + reflc.y[i]*reflc.y[i]*Rzy_inv) - j*k*reflc.z[i] + j*(phizx - phizy)*0.5);
        
        efield[0] = field_atPoint * gdict.pol[0];
        efield[1] = field_atPoint * gdict.pol[1];
        efield[2] = field_atPoint * gdict.pol[2];

        n_source[0] = reflc.nx[i];
        n_source[1] = reflc.ny[i];
        n_source[2] = reflc.nz[i];

        ut.ext(n_source, efield, m);

        res_field->r1x[i] = efield[0].real();
        res_field->i1x[i] = efield[0].imag();

        res_field->r1y[i] = efield[1].real();
        res_field->i1y[i] = efield[1].imag();

        res_field->r1z[i] = efield[2].real();
        res_field->i1z[i] = efield[2].imag();

        // Set H to zero
        res_field->r2x[i] = 0;
        res_field->i2x[i] = 0;

        res_field->r2y[i] = 0;
        res_field->i2y[i] = 0;

        res_field->r2z[i] = 0;
        res_field->i2z[i] = 0;

        // Fill currents
        res_current->r1x[i] = 0;
        res_current->i1x[i] = 0;

        res_current->r1y[i] = 0;
        res_current->i1y[i] = 0;

        res_current->r1z[i] = 0;
        res_current->i1z[i] = 0;

        // Set H to zero
        res_current->r2x[i] = -2*m[0].real();
        res_current->i2x[i] = -2*m[0].imag();

        res_current->r2y[i] = -2*m[1].real();
        res_current->i2y[i] = -2*m[1].imag();

        res_current->r2z[i] = -2*m[2].real();
        res_current->i2z[i] = -2*m[2].imag();
    }

    delete reflc.x;
    delete reflc.y;
    delete reflc.z;

    delete reflc.nx;
    delete reflc.ny;
    delete reflc.nz;

    delete reflc.area;
}

template<typename T, typename U, typename V, typename W, typename G>
void initScalarGauss(T sgdict, U refldict, V *res_field)
{
    int nTot = refldict.n_cells[0] * refldict.n_cells[1];
    W reflc;

    reflc.size = nTot;
    reflc.x = new G[nTot];
    reflc.y = new G[nTot];
    reflc.z = new G[nTot];

    reflc.nx = new G[nTot];
    reflc.ny = new G[nTot];
    reflc.nz = new G[nTot];

    reflc.area = new G[nTot];
    Utils<G> ut;

    bool transform = true;
    generateGrid(refldict, &reflc, transform);

    G zRx      = M_PI * sgdict.w0x*sgdict.w0x * sgdict.n / sgdict.lam;
    G zRy      = M_PI * sgdict.w0y*sgdict.w0y * sgdict.n / sgdict.lam;
    G k        = 2 * M_PI / sgdict.lam;
    
    G wzx;      
    G wzy;     
    G Rzx_inv;  
    G Rzy_inv;  
    G phizx;    
    G phizy;    
    std::complex<G> j(0, 1);

    std::complex<G> efield;

    for (int i=0; i<nTot; i++)
    {
        wzx      = sgdict.w0x * sqrt(1 + (reflc.z[i] / zRx)*(reflc.z[i] / zRx));
        wzy      = sgdict.w0y * sqrt(1 + ((reflc.z[i] - sgdict.dxyz) / zRy)*((reflc.z[i] - sgdict.dxyz) / zRy));
        Rzx_inv  = reflc.z[i] / (reflc.z[i]*reflc.z[i] + zRx*zRx);
        Rzy_inv  = (reflc.z[i] - sgdict.dxyz) / ((reflc.z[i] - sgdict.dxyz)*(reflc.z[i] - sgdict.dxyz) + zRy*zRy);
        phizx    = atan(reflc.z[i] / zRx);
        phizy    = atan((reflc.z[i] - sgdict.dxyz) / zRy);
        
        efield = sgdict.E0 * sqrt(2 / (M_PI * wzx * wzy)) * exp(-(reflc.x[i]/wzx)*(reflc.x[i]/wzx) - (reflc.y[i]/wzy)*(reflc.y[i]/wzy) -
                j*M_PI/sgdict.lam * (reflc.x[i]*reflc.x[i]*Rzx_inv + reflc.y[i]*reflc.y[i]*Rzy_inv) - j*k*reflc.z[i] + j*(phizx - phizy)*0.5);

        res_field->x[i] = efield.real();
        res_field->y[i] = efield.imag();
    }

    delete reflc.x;
    delete reflc.y;
    delete reflc.z;
    
    delete reflc.nx;
    delete reflc.ny;
    delete reflc.nz;

    delete reflc.area;
}

template<typename T, typename U, typename V, typename W>
void calcJM(T *res_field, T *res_current, V refldict, int mode)
{
    W reflc;
    int nTot = refldict.n_cells[0] * refldict.n_cells[1];

    reflc.size = nTot;

    reflc.x = new U[nTot];
    reflc.y = new U[nTot];
    reflc.z = new U[nTot];

    reflc.nx = new U[nTot];
    reflc.ny = new U[nTot];
    reflc.nz = new U[nTot];

    reflc.area = new U[nTot];

    bool transform = true;
    generateGrid(refldict, &reflc, transform);

    Utils<U> ut;

    std::array<std::complex<U>, 3> field;
    std::array<U, 3> n_source;

    std::array<std::complex<U>, 3> js;
    std::array<std::complex<U>, 3> ms;

    // full currents
    if (mode == 0)
    {
        for (int i=0; i<nTot; i++)
        {
            field[0] = {res_field->r1x[i], res_field->i1x[i]};
            field[1] = {res_field->r1y[i], res_field->i1y[i]};
            field[2] = {res_field->r1z[i], res_field->i1z[i]};

            n_source[0] = reflc.nx[i];
            n_source[1] = reflc.ny[i];
            n_source[2] = reflc.nz[i];

            ut.ext(n_source, field, ms);

            res_current->r2x[i] = -ms[0].real();
            res_current->i2x[i] = -ms[0].imag();

            res_current->r2y[i] = -ms[1].real();
            res_current->i2y[i] = -ms[1].imag();

            res_current->r2z[i] = -ms[2].real();
            res_current->i2z[i] = -ms[2].imag();

            field[0] = {res_field->r2x[i], res_field->i2x[i]};
            field[1] = {res_field->r2y[i], res_field->i2y[i]};
            field[2] = {res_field->r2z[i], res_field->i2z[i]};

            ut.ext(n_source, field, js);

            res_current->r1x[i] = js[0].real();
            res_current->i1x[i] = js[0].imag();

            res_current->r1y[i] = js[1].real();
            res_current->i1y[i] = js[1].imag();

            res_current->r1z[i] = js[2].real();
            res_current->i1z[i] = js[2].imag();
        }
    }

    // PMC
    else if (mode == 1)
    {
        for (int i=0; i<nTot; i++)
        {
            field[0] = {res_field->r1x[i], res_field->i1x[i]};
            field[1] = {res_field->r1y[i], res_field->i1y[i]};
            field[2] = {res_field->r1z[i], res_field->i1z[i]};

            n_source[0] = reflc.nx[i];
            n_source[1] = reflc.ny[i];
            n_source[2] = reflc.nz[i];

            ut.ext(n_source, field, ms);

            res_current->r2x[i] = -2*ms[0].real();
            res_current->i2x[i] = -2*ms[0].imag();

            res_current->r2y[i] = -2*ms[1].real();
            res_current->i2y[i] = -2*ms[1].imag();

            res_current->r2z[i] = -2*ms[2].real();
            res_current->i2z[i] = -2*ms[2].imag();

            res_current->r1x[i] = 0;
            res_current->i1x[i] = 0;

            res_current->r1y[i] = 0;
            res_current->i1y[i] = 0;

            res_current->r1z[i] = 0;
            res_current->i1z[i] = 0;
        }
    }

    // PEC
    else if (mode == 2)
    {
        for (int i=0; i<nTot; i++)
        {
            field[0] = {res_field->r2x[i], res_field->i2x[i]};
            field[1] = {res_field->r2y[i], res_field->i2y[i]};
            field[2] = {res_field->r2z[i], res_field->i2z[i]};

            n_source[0] = reflc.nx[i];
            n_source[1] = reflc.ny[i];
            n_source[2] = reflc.nz[i];

            ut.ext(n_source, field, js);

            res_current->r1x[i] = 2*js[0].real();
            res_current->i1x[i] = 2*js[0].imag();

            res_current->r1y[i] = 2*js[1].real();
            res_current->i1y[i] = 2*js[1].imag();

            res_current->r1z[i] = 2*js[2].real();
            res_current->i1z[i] = 2*js[2].imag();

            res_current->r2x[i] = 0;
            res_current->i2x[i] = 0;

            res_current->r2y[i] = 0;
            res_current->i2y[i] = 0;

            res_current->r2z[i] = 0;
            res_current->i2z[i] = 0;
        }
    }

    delete reflc.x;
    delete reflc.y;
    delete reflc.z;

    delete reflc.nx;
    delete reflc.ny;
    delete reflc.nz;

    delete reflc.area;
}
#endif
