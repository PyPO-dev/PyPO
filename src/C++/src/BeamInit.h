#include <iostream>
#include <vector>
#include <array>
#define _USE_MATH_DEFINES
#include <cmath>
#include <complex>
#include <thread>
#include <iomanip>
#include <algorithm>
#include <new>

#include "Utils.h"
#include "Structs.h"
#include "InterfaceReflector.h"

#ifndef __BeamInit_h
#define __BeamInit_h


template<typename T, typename U, typename V>
void initFrame(T rdict, U *fr);

template<typename T, typename U, typename V, typename W, typename G>
void initGauss(T gdict, U rdict, V *res_field, V *res_current);

template<typename T, typename U, typename V, typename W>
void calcJM(T *res_field, T *res_current, V rdict, int mode);


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

    std::array<V, 3> tChief;
    std::array<V, 3> oChief;
    std::array<V, 3> rotation;
    std::array<V, 3> _direction;
    std::array<V, 3> direction;

    for (int n=0; n<3; n++) {tChief[n] = rdict.tChief[n];}
    for (int n=0; n<3; n++) {oChief[n] = rdict.oChief[n];}

    ut.matRot(tChief, nomChief, zero, direction);

    fr->x[0] = oChief[0];
    fr->y[0] = oChief[1];
    fr->z[0] = oChief[2];

    fr->dx[0] = direction[0];
    fr->dy[0] = direction[1];
    fr->dz[0] = direction[2];

    std::array<V, 3> _pos;
    std::array<V, 3> pos;

    for (int i=1; i<nTot; i++)
    {
        _pos[0] = rdict.a * cos(alpha) / rdict.nRing * n + rdict.oChief[0];
        _pos[1] = rdict.b * sin(alpha) / rdict.nRing * n + rdict.oChief[1];
        _pos[2] = rdict.oChief[2];

        rotation[0] = rdict.angy * sin(alpha) / rdict.nRing * n;
        rotation[1] = rdict.angx * cos(alpha) / rdict.nRing * n;
        rotation[2] = 2 * alpha;

        ut.matRot(rotation, nomChief, zero, _direction);
        ut.matRot(tChief, _direction, zero, direction);

        ut.matRot(tChief, _pos, oChief, pos);

        fr->x[i] = pos[0];
        fr->y[i] = pos[1];
        fr->z[i] = pos[2];

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

template<typename T, typename U, typename V, typename W, typename G>
void initGauss(T gdict, U rdict, V *res_field, V *res_current)
{
    int nTot = rdict.n_cells[0] * rdict.n_cells[1];

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

    bool transform = false;
    generateGrid(rdict, &reflc, transform);

    G zR      = M_PI * gdict.w0*gdict.w0 * gdict.n / gdict.lam;
    G wz      = gdict.w0 * sqrt(1 + (gdict.z / zR)*(gdict.z / zR));
    G Rz_inv  = gdict.z / (gdict.z*gdict.z + zR*zR);
    G phiz    = atan(gdict.z / zR);
    G k       = 2 * M_PI / gdict.lam;

    G r2;

    std::complex<G> j(0, 1);

    std::complex<G> field_atPoint;
    std::array<std::complex<G>, 3> efield;
    std::array<G, 3> n_source;
    std::array<std::complex<G>, 3> m;

    for (int i=0; i<nTot; i++)
    {
        r2 = reflc.x[i]*reflc.x[i] + reflc.y[i]*reflc.y[i];

        field_atPoint = gdict.E0 * gdict.w0/wz * exp(-r2/(wz*wz)) *
                        exp(-j * (k*gdict.z + k*r2*Rz_inv/2 - phiz));

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

template<typename T, typename U, typename V, typename W>
void calcJM(T *res_field, T *res_current, V rdict, int mode)
{
    W reflc;
    int nTot = rdict.n_cells[0] * rdict.n_cells[1];

    reflc.size = nTot;

    reflc.x = new U[nTot];
    reflc.y = new U[nTot];
    reflc.z = new U[nTot];

    reflc.nx = new U[nTot];
    reflc.ny = new U[nTot];
    reflc.nz = new U[nTot];

    reflc.area = new U[nTot];

    bool transform = true;
    generateGrid(rdict, &reflc, transform);

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
