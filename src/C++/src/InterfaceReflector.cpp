#include <iostream>
#include <vector>
#include <array>
#define _USE_MATH_DEFINES
#include <cmath>

#include "Utils.h"
#include "InterfaceReflector.h"

template<typename T, typename U>
void Parabola_xy(T *parabola, U xu_lo, U xu_up, U yv_lo,
                U yv_up, U a, U b, int ncx, int ncy, int nfac,
                U mat[16])
{
    U dx = (xu_up - xu_lo) / (ncx - 1);
    U dy = (yv_up - yv_lo) / (ncy - 1);

    // Generate using xy parametrisation
    U x;
    U y;
    U prefac;

    U norm;

    Utils<U> ut;
    bool vec = true;

    std::array<U, 3> inp, out;

    for (int i=0; i < ncx; i++)
    {
        x = i * dx + xu_lo;

        for (int j=0; j < ncy; j++)
        {

            y = j * dy + yv_lo;

            parabola->x[i*ncy + j] = x;
            parabola->y[i*ncy  + j] = y;
            parabola->z[i*ncy  + j] = x*x / (a*a) + y*y / (b*b);

            parabola->nx[i*ncy  + j] = -2 * x / (a*a);
            parabola->ny[i*ncy  + j] = -2 * y / (b*b);
            parabola->nz[i*ncy  + j] = 1;

            norm = sqrt(parabola->nx[i*ncy  + j]*parabola->nx[i*ncy  + j] +
                        parabola->ny[i*ncy  + j]*parabola->ny[i*ncy  + j] + 1);

            parabola->nx[i*ncy  + j] = nfac * parabola->nx[i*ncy  + j] / norm;
            parabola->ny[i*ncy  + j] = nfac * parabola->ny[i*ncy  + j] / norm;
            parabola->nz[i*ncy  + j] = nfac * parabola->nz[i*ncy  + j] / norm;

            parabola->area[i*ncy  + j] = norm * dx * dy;

            inp[0] = parabola->x[i*ncy + j];
            inp[1] = parabola->y[i*ncy + j];
            inp[2] = parabola->z[i*ncy + j];

            ut.matVec4(mat, inp, out);

            parabola->x[i*ncy + j] = out[0];
            parabola->y[i*ncy  + j] = out[1];
            parabola->z[i*ncy  + j] = out[2];

            inp[0] = parabola->nx[i*ncy + j];
            inp[1] = parabola->ny[i*ncy + j];
            inp[2] = parabola->nz[i*ncy + j];

            ut.matVec4(mat, inp, out, vec);

            parabola->nx[i*ncy  + j] = out[0];
            parabola->ny[i*ncy  + j] = out[1];
            parabola->nz[i*ncy  + j] = out[2];
        }
    }
}

template<typename T, typename U>
void Parabola_uv(T *parabola, U xu_lo, U xu_up, U yv_lo,
                U yv_up, U a, U b, int ncx, int ncy, int nfac,
                U mat[16])
{
    U du = (xu_up/a - xu_lo/a) / (ncx - 1);
    U dv = (yv_up- yv_lo) / (ncy - 1);

    // Generate using uv parametrisation
    U u;
    U v;
    U prefac;

    Utils<U> ut;
    bool vec = true;

    std::array<U, 3> inp, out;

    for (int i=0; i < ncx; i++)
    {
        u = i * du + xu_lo/a;
        for (int j=0; j < ncy; j++)
        {
            v = (j * dv + yv_lo) * M_PI/180;

            parabola->x[i*ncy + j] = a * u * cos(v);
            parabola->y[i*ncy + j] = b * u * sin(v);
            parabola->z[i*ncy + j] = u * u;

            prefac =  nfac / sqrt(4 * b*b * u*u * cos(v)*cos(v) +
                      4 * a*a * u*u * sin(v)*sin(v) +
                      a*a * b*b);

            parabola->nx[i*ncy + j] = -2 * b * u * cos(v) * prefac;
            parabola->ny[i*ncy + j] = -2 * a * u * sin(v) * prefac;
            parabola->nz[i*ncy + j] = b * a * prefac;

            parabola->area[i*ncy + j] = u * sqrt(4 * b*b * u*u * cos(v)*cos(v) +
                                    4 * a*a * u*u * sin(v)*sin(v) +
                                    a*a * b*b) * du * dv;

            inp[0] = parabola->x[i*ncy + j];
            inp[1] = parabola->y[i*ncy + j];
            inp[2] = parabola->z[i*ncy + j];

            ut.matVec4(mat, inp, out);
            inp = out;
            ut.invmatVec4(mat, inp, out);

            parabola->x[i*ncy + j] = out[0];
            parabola->y[i*ncy  + j] = out[1];
            parabola->z[i*ncy  + j] = out[2];

            inp[0] = parabola->nx[i*ncy + j];
            inp[1] = parabola->ny[i*ncy + j];
            inp[2] = parabola->nz[i*ncy + j];

            ut.matVec4(mat, inp, out, vec);

            inp = out;
            ut.invmatVec4(mat, inp, out, vec);

            parabola->nx[i*ncy  + j] = out[0];
            parabola->ny[i*ncy  + j] = out[1];
            parabola->nz[i*ncy  + j] = out[2];
        }
    }
}

template<typename T, typename U>
void Hyperbola_xy(T *hyperbola, U xu_lo, U xu_up, U yv_lo,
                  U yv_up, U a, U b, U c, int ncx, int ncy, int nfac,
                  U mat[16])
{
    U dx = (xu_up - xu_lo) / (ncx - 1);
    U dy = (yv_up - yv_lo) / (ncy - 1);

    // Generate using xy parametrisation
    U x;
    U y;
    U prefac;

    U norm;

    Utils<U> ut;
    bool vec = true;

    std::array<U, 3> inp, out;

    for (int i=0; i < ncx; i++)
    {
        x = i * dx + xu_lo;

        for (int j=0; j < ncy; j++)
        {
            y = j * dy + yv_lo;

            hyperbola->x[i*ncy + j] = x;
            hyperbola->y[i*ncy  + j] = y;
            hyperbola->z[i*ncy  + j] = c * sqrt(x*x / (a*a) + y*y / (b*b) + 1);

            hyperbola->nx[i*ncy  + j] = -2 * x / (a*a);
            hyperbola->ny[i*ncy  + j] = -2 * y / (b*b);
            hyperbola->nz[i*ncy  + j] = 2 * hyperbola->z[i*ncy  + j] / (c*c);

            norm = sqrt(hyperbola->nx[i*ncy  + j]*hyperbola->nx[i*ncy  + j] +
                        hyperbola->ny[i*ncy  + j]*hyperbola->ny[i*ncy  + j] +
                        hyperbola->nz[i*ncy  + j]*hyperbola->nz[i*ncy  + j]);

            hyperbola->nx[i*ncy  + j] = nfac * hyperbola->nx[i*ncy  + j] / norm;
            hyperbola->ny[i*ncy  + j] = nfac * hyperbola->ny[i*ncy  + j] / norm;
            hyperbola->nz[i*ncy  + j] = nfac * hyperbola->nz[i*ncy  + j] / norm;

            hyperbola->area[i*ncy  + j] = norm * dx * dy;

            inp[0] = hyperbola->x[i*ncy + j];
            inp[1] = hyperbola->y[i*ncy + j];
            inp[2] = hyperbola->z[i*ncy + j];

            ut.matVec4(mat, inp, out);

            hyperbola->x[i*ncy + j] = out[0];
            hyperbola->y[i*ncy  + j] = out[1];
            hyperbola->z[i*ncy  + j] = out[2];

            inp[0] = hyperbola->nx[i*ncy + j];
            inp[1] = hyperbola->ny[i*ncy + j];
            inp[2] = hyperbola->nz[i*ncy + j];

            ut.matVec4(mat, inp, out, vec);

            hyperbola->nx[i*ncy  + j] = out[0];
            hyperbola->ny[i*ncy  + j] = out[1];
            hyperbola->nz[i*ncy  + j] = out[2];
        }
    }
}

template<typename T, typename U>
void Hyperbola_uv(T *hyperbola, U xu_lo, U xu_up, U yv_lo,
                  U yv_up, U a, U b, U c, int ncx, int ncy, int nfac,
                  U mat[16])
{
    U du = (sqrt(xu_up*xu_up/(a*a) + 1) - sqrt(xu_lo*xu_lo/(a*a) + 1)) / (ncx - 1);
    U dv = (yv_up - yv_lo) / (ncy - 1);

    // Generate using xy parametrisation
    U u;
    U v;
    U prefac;

    U norm;

    Utils<U> ut;
    bool vec = true;

    std::array<U, 3> inp, out;

    for (int i=0; i < ncx; i++)
    {
        u = i * du + sqrt(xu_lo*xu_lo/(a*a) + 1);
        for (int j=0; j < ncy; j++)
        {
            v = (j * dv + yv_lo) * M_PI/180;

            hyperbola->x[i*ncy + j] = a * sqrt(u*u - 1) * cos(v);
            hyperbola->y[i*ncy + j] = b * sqrt(u*u - 1) * sin(v);
            hyperbola->z[i*ncy + j] = c * u;

            prefac = nfac / sqrt(b*b * c*c * (u*u - 1) * cos(v)*cos(v) +
                      a*a * c*c * (u*u - 1) * sin(v)*sin(v) + a*a * b*b * u*u);

            hyperbola->nx[i*ncy + j] = -b * c * sqrt(u*u - 1) * cos(v) * prefac;
            hyperbola->ny[i*ncy + j] = -a * c * sqrt(u*u - 1) * sin(v) * prefac;
            hyperbola->nz[i*ncy + j] = b * a * u * prefac;

            hyperbola->area[i*ncy + j] = sqrt(b*b * c*c * (u*u - 1) * cos(v)*cos(v) +
                                        a*a * c*c * (u*u - 1) * sin(v)*sin(v) + a*a * b*b * u*u) * du * dv;

            inp[0] = hyperbola->x[i*ncy + j];
            inp[1] = hyperbola->y[i*ncy + j];
            inp[2] = hyperbola->z[i*ncy + j];

            ut.matVec4(mat, inp, out);

            hyperbola->x[i*ncy + j] = out[0];
            hyperbola->y[i*ncy  + j] = out[1];
            hyperbola->z[i*ncy  + j] = out[2];

            inp[0] = hyperbola->nx[i*ncy + j];
            inp[1] = hyperbola->ny[i*ncy + j];
            inp[2] = hyperbola->nz[i*ncy + j];

            ut.matVec4(mat, inp, out, vec);

            hyperbola->nx[i*ncy  + j] = out[0];
            hyperbola->ny[i*ncy  + j] = out[1];
            hyperbola->nz[i*ncy  + j] = out[2];
        }
    }
}

template<typename T, typename U>
void Ellipse_xy(T *ellipse, U xu_lo, U xu_up, U yv_lo,
                U yv_up, U a, U b, U c, int ncx, int ncy, int nfac,
                U mat[16])
{
    U dx = (xu_up - xu_lo) / (ncx - 1);
    U dy = (yv_up - yv_lo) / (ncy - 1);

    // Generate using xy parametrisation
    U x;
    U y;
    U prefac;

    U norm;

    Utils<U> ut;
    bool vec = true;

    std::array<U, 3> inp, out;

    for (int i=0; i < ncx; i++)
    {
        x = i * dx + xu_lo;

        for (int j=0; j < ncy; j++)
        {
            y = j * dy + yv_lo;

            ellipse->x[i*ncy + j] = x;
            ellipse->y[i*ncy  + j] = y;
            ellipse->z[i*ncy  + j] = c * sqrt(1 - x*x / (a*a) - y*y / (b*b));

            ellipse->nx[i*ncy  + j] = -2 * x / (a*a);
            ellipse->ny[i*ncy  + j] = -2 * y / (b*b);
            ellipse->nz[i*ncy  + j] = 2 * ellipse->z[i*ncy  + j] / (c*c);

            norm = sqrt(ellipse->nx[i*ncy  + j]*ellipse->nx[i*ncy  + j] +
                        ellipse->ny[i*ncy  + j]*ellipse->ny[i*ncy  + j] +
                        ellipse->nz[i*ncy  + j]*ellipse->nz[i*ncy  + j]);

            ellipse->nx[i*ncy  + j] = nfac * ellipse->nx[i*ncy  + j] / norm;
            ellipse->ny[i*ncy  + j] = nfac * ellipse->ny[i*ncy  + j] / norm;
            ellipse->nz[i*ncy  + j] = nfac * ellipse->nz[i*ncy  + j] / norm;

            ellipse->area[i*ncy  + j] = norm * dx * dy;

            inp[0] = ellipse->x[i*ncy + j];
            inp[1] = ellipse->y[i*ncy + j];
            inp[2] = ellipse->z[i*ncy + j];

            ut.matVec4(mat, inp, out);

            ellipse->x[i*ncy + j] = out[0];
            ellipse->y[i*ncy  + j] = out[1];
            ellipse->z[i*ncy  + j] = out[2];

            inp[0] = ellipse->nx[i*ncy + j];
            inp[1] = ellipse->ny[i*ncy + j];
            inp[2] = ellipse->nz[i*ncy + j];

            ut.matVec4(mat, inp, out, vec);

            ellipse->nx[i*ncy  + j] = out[0];
            ellipse->ny[i*ncy  + j] = out[1];
            ellipse->nz[i*ncy  + j] = out[2];
        }
    }
}

template<typename T, typename U>
void Ellipse_uv(T *ellipse, U xu_lo, U xu_up, U yv_lo,
                U yv_up, U a, U b, U c, int ncx, int ncy, int nfac,
                U mat[16])
{
    U du = (asin(xu_up/a) - asin(xu_lo/a)) / (ncx - 1);
    U dv = (yv_up - yv_lo) / (ncy - 1);

    // Generate using xy parametrisation
    U u;
    U v;
    U prefac;

    U norm;

    Utils<U> ut;
    bool vec = true;

    std::array<U, 3> inp, out;

    for (int i=0; i < ncx; i++)
    {
        u = i * du + asin(xu_lo/a);
        for (int j=0; j < ncy; j++)
        {
            v = (j * dv + yv_lo) * M_PI/180;

            ellipse->x[i*ncy + j] = a * sin(u) * cos(v);
            ellipse->y[i*ncy + j] = b * sin(u) * sin(v);
            ellipse->z[i*ncy + j] = c * cos(u);

            prefac = nfac / sqrt(b*b * c*c * sin(u)*sin(u) * cos(v)*cos(v) +
                      a*a * c*c * sin(u)*sin(u) * sin(v)*sin(v) + a*a * b*b * cos(u)*cos(u));

            ellipse->nx[i*ncy + j] = b * c * sin(u) * cos(v) * prefac;
            ellipse->ny[i*ncy + j] = a * c * sin(u) * sin(v) * prefac;
            ellipse->nz[i*ncy + j] = b * a * cos(u) * prefac;

            ellipse->area[i*ncy + j] = sin(u) * sqrt(b*b * c*c * sin(u)*sin(u) * cos(v)*cos(v) +
                                        a*a * c*c * sin(u)*sin(u) * sin(v)*sin(v) +
                                        a*a * b*b * cos(u)*cos(u)) * du * dv;

            inp[0] = ellipse->x[i*ncy + j];
            inp[1] = ellipse->y[i*ncy + j];
            inp[2] = ellipse->z[i*ncy + j];

            ut.matVec4(mat, inp, out);

            ellipse->x[i*ncy + j] = out[0];
            ellipse->y[i*ncy  + j] = out[1];
            ellipse->z[i*ncy  + j] = out[2];

            inp[0] = ellipse->nx[i*ncy + j];
            inp[1] = ellipse->ny[i*ncy + j];
            inp[2] = ellipse->nz[i*ncy + j];

            ut.matVec4(mat, inp, out, vec);

            ellipse->nx[i*ncy  + j] = out[0];
            ellipse->ny[i*ncy  + j] = out[1];
            ellipse->nz[i*ncy  + j] = out[2];
        }
    }
}

template<typename T, typename U>
void Plane_xy(T *plane, U xu_lo, U xu_up, U yv_lo,
              U yv_up, int ncx, int ncy, int nfac,
              U mat[16])
{
    U dx = (xu_up - xu_lo) / (ncx - 1);
    U dy = (yv_up - yv_lo) / (ncy - 1);

    // Generate using xy parametrisation
    U x;
    U y;
    U prefac;

    U norm;

    Utils<U> ut;
    bool vec = true;

    std::array<U, 3> inp, out;

    for (int i=0; i < ncx; i++)
    {
        x = i * dx + xu_lo;
        for (int j=0; j < ncy; j++)
        {
            y = j * dy + yv_lo;

            plane->x[i*ncy + j] = x;
            plane->y[i*ncy  + j] = y;
            plane->z[i*ncy  + j] = 0;

            plane->nx[i*ncy  + j] = 0;
            plane->ny[i*ncy  + j] = 0;
            plane->nz[i*ncy  + j] = nfac * 1;

            plane->area[i*ncy  + j] = dx * dy;

            inp[0] = plane->x[i*ncy + j];
            inp[1] = plane->y[i*ncy + j];
            inp[2] = plane->z[i*ncy + j];

            ut.matVec4(mat, inp, out);

            plane->x[i*ncy + j] = out[0];
            plane->y[i*ncy  + j] = out[1];
            plane->z[i*ncy  + j] = out[2];

            inp[0] = plane->nx[i*ncy + j];
            inp[1] = plane->ny[i*ncy + j];
            inp[2] = plane->nz[i*ncy + j];

            ut.matVec4(mat, inp, out, vec);

            plane->nx[i*ncy  + j] = out[0];
            plane->ny[i*ncy  + j] = out[1];
            plane->nz[i*ncy  + j] = out[2];
        }
    }
}

template<typename T, typename U>
void Plane_uv(T *plane, U xu_lo, U xu_up, U yv_lo,
              U yv_up, int ncx, int ncy, int nfac,
              U mat[16])
{
    U du = (xu_up - xu_lo) / (ncx - 1);
    U dv = (yv_up - yv_lo) / (ncy - 1);

    // Generate using xy parametrisation
    U u;
    U v;

    U norm;

    Utils<U> ut;
    bool vec = true;

    std::array<U, 3> inp, out;

    for (int i=0; i < ncx; i++)
    {
        u = i * du + xu_lo;

        for (int j=0; j < ncy; j++)
        {
            v = (j * dv + yv_lo) * M_PI/180;

            plane->x[i*ncy + j] = u * cos(v);
            plane->y[i*ncy  + j] = u * sin(v);
            plane->z[i*ncy  + j] = 0;

            plane->nx[i*ncy  + j] = 0;
            plane->ny[i*ncy  + j] = 0;
            plane->nz[i*ncy  + j] = nfac * 1;

            plane->area[i*ncy  + j] = u * du * dv;

            inp[0] = plane->x[i*ncy + j];
            inp[1] = plane->y[i*ncy + j];
            inp[2] = plane->z[i*ncy + j];

            ut.matVec4(mat, inp, out);

            plane->x[i*ncy + j] = out[0];
            plane->y[i*ncy  + j] = out[1];
            plane->z[i*ncy  + j] = out[2];

            inp[0] = plane->nx[i*ncy + j];
            inp[1] = plane->ny[i*ncy + j];
            inp[2] = plane->nz[i*ncy + j];

            ut.matVec4(mat, inp, out, vec);

            plane->nx[i*ncy  + j] = out[0];
            plane->ny[i*ncy  + j] = out[1];
            plane->nz[i*ncy  + j] = out[2];
        }
    }
}

template<typename T, typename U>
void Plane_AoE(T *plane, U xu_lo, U xu_up, U yv_lo,
              U yv_up, int ncx, int ncy,
              U mat[16])
{
    U dA = (xu_up - xu_lo) / (ncx - 1);
    U dE = (yv_up - yv_lo) / (ncy - 1);

    // Generate using xy parametrisation
    U Az;
    U El;

    Utils<U> ut;
    bool vec = true;

    std::array<U, 3> inp, out;

    for (int i=0; i < ncx; i++)
    {
        Az = i * dA + xu_lo;

        for (int j=0; j < ncy; j++)
        {
            El = j * dE + yv_lo;

            plane->x[i*ncy + j] = sqrt(Az*Az + El*El) * M_PI/180;
            plane->y[i*ncy  + j] = atan(El / Az);

            if (plane->y[i*ncy  + j] != plane->y[i*ncy  + j])
            {
                plane->y[i*ncy  + j] = 0;
            }

            plane->z[i*ncy  + j] = 0;

            plane->nx[i*ncy  + j] = 0;
            plane->ny[i*ncy  + j] = 0;
            plane->nz[i*ncy  + j] = 1;

            plane->area[i*ncy  + j] = 1;

            inp[0] = plane->x[i*ncy + j];
            inp[1] = plane->y[i*ncy + j];
            inp[2] = plane->z[i*ncy + j];

            ut.matVec4(mat, inp, out);

            plane->x[i*ncy + j] = out[0];
            plane->y[i*ncy  + j] = out[1];
            plane->z[i*ncy  + j] = out[2];

            inp[0] = plane->nx[i*ncy + j];
            inp[1] = plane->ny[i*ncy + j];
            inp[2] = plane->nz[i*ncy + j];

            ut.matVec4(mat, inp, out, vec);

            plane->nx[i*ncy  + j] = out[0];
            plane->ny[i*ncy  + j] = out[1];
            plane->nz[i*ncy  + j] = out[2];
        }
    }
}

extern "C" void generateGrid(reflparams refl, reflcontainer *container)
{
    // For readability, assign new temporary placeholders
    double xu_lo = refl.lxu[0];
    double xu_up = refl.lxu[1];
    double yv_lo = refl.lyv[0];
    double yv_up = refl.lyv[1];

    double a = refl.coeffs[0];
    double b = refl.coeffs[1];
    double c = refl.coeffs[2];

    int ncx = refl.n_cells[0];
    int ncy = refl.n_cells[1];

    int nfac = 1;

    if (refl.flip)
    {
        nfac = -1;
    }

    if (refl.gmode == 0)
    {
        if (refl.type == 0)
        {
            Parabola_xy<reflcontainer, double>(container, xu_lo, xu_up, yv_lo,
                                            yv_up, a, b, ncx, ncy, nfac, refl.transf);
        }

        else if (refl.type == 1)
        {
            Hyperbola_xy<reflcontainer, double>(container, xu_lo, xu_up, yv_lo,
                                            yv_up, a, b, c, ncx, ncy, nfac, refl.transf);
        }

        else if (refl.type == 2)
        {
            Ellipse_xy<reflcontainer, double>(container, xu_lo, xu_up, yv_lo,
                                            yv_up, a, b, c, ncx, ncy, nfac, refl.transf);
        }

        else if (refl.type == 3)
        {
            Plane_xy<reflcontainer, double>(container, xu_lo, xu_up, yv_lo,
                                            yv_up, ncx, ncy, nfac, refl.transf);
        }
    }

    else if (refl.gmode == 1)
    {
        if (refl.type == 0)
        {

            Parabola_uv<reflcontainer, double>(container, xu_lo, xu_up, yv_lo,
                                            yv_up, a, b, ncx, ncy, nfac, refl.transf);
        }

        else if (refl.type == 1)
        {
            Hyperbola_uv<reflcontainer, double>(container, xu_lo, xu_up, yv_lo,
                                            yv_up, a, b, c, ncx, ncy, nfac, refl.transf);
        }

        else if (refl.type == 2)
        {
            Ellipse_uv<reflcontainer, double>(container, xu_lo, xu_up, yv_lo,
                                            yv_up, a, b, c, ncx, ncy, nfac, refl.transf);
        }

        else if (refl.type == 3)
        {
            Plane_uv<reflcontainer, double>(container, xu_lo, xu_up, yv_lo,
                                            yv_up, ncx, ncy, nfac, refl.transf);
        }
    }

    if (refl.gmode == 2)
    {
        Plane_AoE<reflcontainer, double>(container, xu_lo, xu_up, yv_lo,
                                        yv_up, ncx, ncy, refl.transf);
    }
}

extern "C" void generateGridf(reflparamsf refl, reflcontainerf *container)
{
    // For readability, assign new temporary placeholders
    float xu_lo = refl.lxu[0];
    float xu_up = refl.lxu[1];
    float yv_lo = refl.lyv[0];
    float yv_up = refl.lyv[1];

    float a = refl.coeffs[0];
    float b = refl.coeffs[1];
    float c = refl.coeffs[2];

    int ncx = refl.n_cells[0];
    int ncy = refl.n_cells[1];

    int nfac = 1;

    if (refl.flip)
    {
        nfac = -1;
    }

    if (refl.gmode == 0)
    {
        if (refl.type == 0)
        {
            Parabola_xy<reflcontainerf, float>(container, xu_lo, xu_up, yv_lo,
                                            yv_up, a, b, ncx, ncy, nfac, refl.transf);
        }

        else if (refl.type == 1)
        {
            Hyperbola_xy<reflcontainerf, float>(container, xu_lo, xu_up, yv_lo,
                                            yv_up, a, b, c, ncx, ncy, nfac, refl.transf);
        }

        else if (refl.type == 2)
        {
            Ellipse_xy<reflcontainerf, float>(container, xu_lo, xu_up, yv_lo,
                                            yv_up, a, b, c, ncx, ncy, nfac, refl.transf);
        }

        else if (refl.type == 3)
        {
            Plane_xy<reflcontainerf, float>(container, xu_lo, xu_up, yv_lo,
                                            yv_up, ncx, ncy, nfac, refl.transf);
        }
    }

    else if (refl.gmode == 1)
    {
        if (refl.type == 0)
        {

            Parabola_uv<reflcontainerf, float>(container, xu_lo, xu_up, yv_lo,
                                            yv_up, a, b, ncx, ncy, nfac, refl.transf);
        }

        else if (refl.type == 1)
        {
            Hyperbola_uv<reflcontainerf, float>(container, xu_lo, xu_up, yv_lo,
                                            yv_up, a, b, c, ncx, ncy, nfac, refl.transf);
        }

        else if (refl.type == 2)
        {
            Ellipse_uv<reflcontainerf, float>(container, xu_lo, xu_up, yv_lo,
                                            yv_up, a, b, c, ncx, ncy, nfac, refl.transf);
        }

        else if (refl.type == 3)
        {
            Plane_uv<reflcontainerf, float>(container, xu_lo, xu_up, yv_lo,
                                            yv_up, ncx, ncy, nfac, refl.transf);
        }
    }

    else if (refl.gmode == 2)
    {
        Plane_AoE<reflcontainerf, float>(container, xu_lo, xu_up, yv_lo,
                                        yv_up, ncx, ncy, refl.transf);
    }
}
