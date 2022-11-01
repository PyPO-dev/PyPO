#include <iostream>
#include <vector>
#include <array>
#define _USE_MATH_DEFINES
#include <cmath>
#include <thread>
#include <iomanip>
#include <algorithm>
#include <new>

#include "Utils.h"
#include "Structs.h"

#ifndef __BeamInit_h
#define __BeamInit_h

template<typename T, typename U, typename V>
void initFrame(T rdict, U *fr);

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
        _pos[2] = 0;

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
#endif
