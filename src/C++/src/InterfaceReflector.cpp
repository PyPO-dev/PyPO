#include <iostream>
#include <vector>
#include <cmath>

#include "InterfaceReflector.h"

template<typename T, typename U> void Parabola_xy(T *parabola, U xu_lo, U xu_up, U yv_lo,
                                                U yv_up, U a, U b, int ncx, int ncy, int nfac)
{
    U _dx = (xu_up - xu_lo) / ncx;
    U _dy = (yv_up - yv_lo) / ncy;

    U dx = (xu_up + _dx - xu_lo) / ncx;
    U dy = (yv_up + _dy - yv_lo) / ncy;

    // Generate using xy parametrisation
    U x;
    U y;
    U prefac;

    U norm;

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
        }
    }
}

template<typename T, typename U> void Parabola_uv(T *parabola, U xu_lo, U xu_up, U yv_lo,
                                                U yv_up, U a, U b, int ncx, int ncy, int nfac)
{
    U _du = (xu_up/a - xu_lo/a) / ncx; // Divide by a to convert aperture radius to u param
    U _dv = (yv_up - yv_lo) / ncy;

    U du = (xu_up/a + _du - xu_lo/a) / ncx;
    U dv = (yv_up + _dv - yv_lo) / ncy;

    // Generate using uv parametrisation
    U u;
    U v;
    U prefac;

    for (int i=0; i < ncx; i++)
    {
        u = i * du + xu_lo/a;
        for (int j=0; j < ncy; j++)
        {
            v = j * dv + yv_lo;

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


        }
    }
}

template<typename T, typename U> void Hyperbola_xy(T *hyperbola, U xu_lo, U xu_up, U yv_lo,
                                                U yv_up, U a, U b, U c, int ncx, int ncy, int nfac)
{
    U _dx = (xu_up - xu_lo) / ncx;
    U _dy = (yv_up - yv_lo) / ncy;

    U dx = (xu_up + _dx - xu_lo) / ncx;
    U dy = (yv_up + _dy - yv_lo) / ncy;

    // Generate using xy parametrisation
    U x;
    U y;
    U prefac;

    U norm;

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
        }
    }
}

template<typename T, typename U> void Hyperbola_uv(T *hyperbola, U xu_lo, U xu_up, U yv_lo,
                                                U yv_up, U a, U b, U c, int ncx, int ncy, int nfac)
{
    U _du = (sqrt(xu_up*xu_up/(a*a) + 1) - sqrt(xu_lo*xu_lo/(a*a) + 1)) / ncx;
    U _dv = (yv_up - yv_lo) / ncy;

    U du = (sqrt(xu_up*xu_up/(a*a) + 1) + _du - sqrt(xu_lo*xu_lo/(a*a) + 1)) / ncx;
    U dv = (yv_up + _dv - yv_lo) / ncy;

    // Generate using xy parametrisation
    U u;
    U v;
    U prefac;

    U norm;

    for (int i=0; i < ncx; i++)
    {
        u = i * du + sqrt(xu_lo*xu_lo/(a*a) + 1);
        for (int j=0; j < ncy; j++)
        {
            v = j * dv + yv_lo;

            hyperbola->x[i*ncy + j] = a * sqrt(u*u - 1) * cos(v);
            hyperbola->y[i*ncy + j] = b * sqrt(u*u - 1) * sin(v);
            hyperbola->z[i*ncy + j] = c * u;

            prefac = nfac / sqrt(b*b * c*c * (u*u -1) * cos(v)*cos(v) +
                      a*a * c*c * (u*u - 1) * sin(v)*sin(v) + a*a + b*b + u*u);

            if(i == 0 && j == 0){printf("%f\n", c);}

            hyperbola->nx[i*ncy + j] = -b * c * sqrt(u*u - 1) * cos(v) * prefac;
            hyperbola->ny[i*ncy + j] = -a * c * sqrt(u*u - 1) * sin(v) * prefac;
            hyperbola->nz[i*ncy + j] = b * a * u * prefac;

            hyperbola->area[i*ncy + j] = sqrt(b*b * c*c * (u*u - 1) * cos(v)*cos(v) +
                                        a*a * c*c * (u*u - 1) * sin(v)*sin(v) + a*a * b*b * u*u) * du * dv;

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

    if (refl.gmode)
    {
        if (refl.type == 0)
        {
            Parabola_xy<reflcontainer, double>(container, xu_lo, xu_up, yv_lo,
                                            yv_up, a, b, ncx, ncy, nfac);
        }

        else if (refl.type == 1)
        {
            Hyperbola_xy<reflcontainer, double>(container, xu_lo, xu_up, yv_lo,
                                            yv_up, a, b, c, ncx, ncy, nfac);
        }
    }

    else
    {
        if (refl.type == 0)
        {

            Parabola_uv<reflcontainer, double>(container, xu_lo, xu_up, yv_lo,
                                            yv_up, a, b, ncx, ncy, nfac);
        }

        else if (refl.type == 1)
        {
            Hyperbola_uv<reflcontainer, double>(container, xu_lo, xu_up, yv_lo,
                                            yv_up, a, b, c, ncx, ncy, nfac);
        }
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

    if (refl.gmode)
    {
        if (refl.type == 0)
        {
            Parabola_xy<reflcontainerf, float>(container, xu_lo, xu_up, yv_lo,
                                            yv_up, a, b, ncx, ncy, nfac);
        }

        else if (refl.type == 1)
        {
            Hyperbola_xy<reflcontainerf, float>(container, xu_lo, xu_up, yv_lo,
                                            yv_up, a, b, c, ncx, ncy, nfac);
        }
    }

    else
    {
        if (refl.type == 0)
        {
            Parabola_uv<reflcontainerf, float>(container, xu_lo, xu_up, yv_lo,
                                            yv_up, a, b, ncx, ncy, nfac);
        }

        else if (refl.type == 1)
        {
            Hyperbola_uv<reflcontainerf, float>(container, xu_lo, xu_up, yv_lo,
                                            yv_up, a, b, c, ncx, ncy, nfac);
        }
    }
}
