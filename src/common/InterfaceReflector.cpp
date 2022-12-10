#include <iostream>
#include <vector>
#include <array>
#define _USE_MATH_DEFINES
#include <cmath>

#include "Utils.h"
#include "InterfaceReflector.h"

// Normalize to [-pi, pi)
template<typename T>
T constrainAngle(T x){
    x = fmod(x + M_PI, 2*M_PI);
    if (x < 0)
        x += 2*M_PI;
    return x - M_PI;
}

template<typename T, typename U>
void transformGrids(T *reflc, int idx, std::array<U, 3> &inp, std::array<U, 3> &out, Utils<U> *ut, U mat[16])
{
    bool vec = true;
    inp[0] = reflc->x[idx];
    inp[1] = reflc->y[idx];
    inp[2] = reflc->z[idx];

    ut->matVec4(mat, inp, out);

    reflc->x[idx] = out[0];
    reflc->y[idx] = out[1];
    reflc->z[idx] = out[2];

    inp[0] = reflc->nx[idx];
    inp[1] = reflc->ny[idx];
    inp[2] = reflc->nz[idx];

    ut->matVec4(mat, inp, out, vec);

    reflc->nx[idx] = out[0];
    reflc->ny[idx] = out[1];
    reflc->nz[idx] = out[2];
}

template<typename T, typename U>
void Parabola_xy(T *parabola, U xu_lo, U xu_up, U yv_lo,
                U yv_up, U xcenter, U ycenter, U a, U b, int ncx, int ncy, int nfac,
                U mat[16], bool transform)
{
    U dx = (xu_up - xu_lo) / (ncx - 1);
    U dy = (yv_up - yv_lo) / (ncy - 1);

    // Generate using xy parametrisation
    U x;
    U y;
    U prefac;

    U norm;

    Utils<U> ut;


    std::array<U, 3> inp, out;

    for (int i=0; i < ncx; i++)
    {
        x = i * dx + xu_lo + xcenter;

        for (int j=0; j < ncy; j++)
        {

            y = j * dy + yv_lo + ycenter;

            int idx = i*ncy + j;

            parabola->x[idx] = x;
            parabola->y[idx] = y;
            parabola->z[idx] = x*x / (a*a) + y*y / (b*b);

            parabola->nx[idx] = -2 * x / (a*a);
            parabola->ny[idx] = -2 * y / (b*b);
            parabola->nz[idx] = 1;

            norm = sqrt(parabola->nx[idx]*parabola->nx[idx] +
                        parabola->ny[idx]*parabola->ny[idx] + 1);

            parabola->nx[idx] = nfac * parabola->nx[idx] / norm;
            parabola->ny[idx] = nfac * parabola->ny[idx] / norm;
            parabola->nz[idx] = nfac * parabola->nz[idx] / norm;

            parabola->area[idx] = norm * dx * dy;

            if (transform)
            {
                transformGrids<T, U>(parabola, idx, inp, out, &ut, mat);
            }
        }
    }
}

template<typename T, typename U>
void Parabola_uv(T *parabola, U xu_lo, U xu_up, U yv_lo,
                U yv_up, U xcenter, U ycenter, U a, U b, int ncx, int ncy, int nfac,
                U mat[16], bool transform)
{
    U du = (xu_up/a - xu_lo/a) / (ncx - 1);
    U dv = (yv_up- yv_lo) / (ncy - 1);

    // Generate using uv parametrisation
    U u;
    U v;
    U prefac;

    U x, y, norm;

    Utils<U> ut;


    std::array<U, 3> inp, out;

    for (int i=0; i < ncx; i++)
    {
        u = i * du + xu_lo/a;
        for (int j=0; j < ncy; j++)
        {
            v = (j * dv + yv_lo) * M_PI/180;
            int idx = i*ncy + j;

            x = a * u * cos(v) + xcenter;
            y = b * u * sin(v) + ycenter;

            parabola->x[idx] = x;
            parabola->y[idx] = y;
            parabola->z[idx] = x*x / (a*a) + y*y / (b*b);

            parabola->nx[idx] = -2 * x / (a*a);
            parabola->ny[idx] = -2 * y / (b*b);
            parabola->nz[idx] = 1;

            norm = sqrt(parabola->nx[idx]*parabola->nx[idx] +
                        parabola->ny[idx]*parabola->ny[idx] + 1);

            parabola->nx[idx] = nfac * parabola->nx[idx] / norm;
            parabola->ny[idx] = nfac * parabola->ny[idx] / norm;
            parabola->nz[idx] = nfac * parabola->nz[idx] / norm;

            parabola->area[idx] = norm * u * du * dv;

            /*
            prefac =  nfac / sqrt(4 * b*b * u*u * cos(v)*cos(v) +
                      4 * a*a * u*u * sin(v)*sin(v) +
                      a*a * b*b);

            parabola->nx[idx] = -2 * b * u * cos(v) * prefac;
            parabola->ny[idx] = -2 * a * u * sin(v) * prefac;
            parabola->nz[idx] = b * a * prefac;

            parabola->area[idx] = u * sqrt(4 * b*b * u*u * cos(v)*cos(v) +
                                    4 * a*a * u*u * sin(v)*sin(v) +
                                    a*a * b*b) * du * dv;
            */
            if (transform)
            {
                transformGrids<T, U>(parabola, idx, inp, out, &ut, mat);
            }
        }
    }
}

template<typename T, typename U>
void Hyperbola_xy(T *hyperbola, U xu_lo, U xu_up, U yv_lo,
                  U yv_up, U xcenter, U ycenter, U a, U b, U c, int ncx, int ncy, int nfac,
                  U mat[16], bool transform)
{
    U dx = (xu_up - xu_lo) / (ncx - 1);
    U dy = (yv_up - yv_lo) / (ncy - 1);

    // Generate using xy parametrisation
    U x;
    U y;
    U prefac;

    U norm;

    Utils<U> ut;


    std::array<U, 3> inp, out;

    for (int i=0; i < ncx; i++)
    {
        x = i * dx + xu_lo + xcenter;

        for (int j=0; j < ncy; j++)
        {
            y = j * dy + yv_lo + ycenter;
            int idx = i*ncy + j;

            hyperbola->x[idx] = x;
            hyperbola->y[idx] = y;
            hyperbola->z[idx] = c * sqrt(x*x / (a*a) + y*y / (b*b) + 1);

            hyperbola->nx[idx] = -2 * x / (a*a);
            hyperbola->ny[idx] = -2 * y / (b*b);
            hyperbola->nz[idx] = 2 * hyperbola->z[idx] / (c*c);

            norm = sqrt(hyperbola->nx[idx]*hyperbola->nx[idx] +
                        hyperbola->ny[idx]*hyperbola->ny[idx] +
                        hyperbola->nz[idx]*hyperbola->nz[idx]);

            hyperbola->nx[idx] = nfac * hyperbola->nx[idx] / norm;
            hyperbola->ny[idx] = nfac * hyperbola->ny[idx] / norm;
            hyperbola->nz[idx] = nfac * hyperbola->nz[idx] / norm;

            hyperbola->area[idx] = norm * dx * dy;

            if (transform)
            {
                transformGrids<T, U>(hyperbola, idx, inp, out, &ut, mat);
            }
        }
    }
}

template<typename T, typename U>
void Hyperbola_uv(T *hyperbola, U xu_lo, U xu_up, U yv_lo,
                  U yv_up, U xcenter, U ycenter, U a, U b, U c, int ncx, int ncy, int nfac,
                  U mat[16], bool transform)
{
    U du = (sqrt(xu_up*xu_up/(a*a) + 1) - sqrt(xu_lo*xu_lo/(a*a) + 1)) / (ncx - 1);
    U dv = (yv_up - yv_lo) / (ncy - 1);

    // Generate using xy parametrisation
    U u;
    U v;
    U prefac;

    U x, y, norm;

    Utils<U> ut;


    std::array<U, 3> inp, out;

    for (int i=0; i < ncx; i++)
    {
        u = i * du + sqrt(xu_lo*xu_lo/(a*a) + 1);
        for (int j=0; j < ncy; j++)
        {
            v = (j * dv + yv_lo) * M_PI/180;
            int idx = i*ncy + j;

            x = a * sqrt(u*u - 1) * cos(v) + xcenter;
            y = b * sqrt(u*u - 1) * sin(v) + ycenter;

            hyperbola->x[idx] = x;
            hyperbola->y[idx] = y;
            hyperbola->z[idx] = c * sqrt(x*x / (a*a) + y*y / (b*b) + 1);

            hyperbola->nx[idx] = -2 * x / (a*a);
            hyperbola->ny[idx] = -2 * y / (b*b);
            hyperbola->nz[idx] = 2 * hyperbola->z[idx] / (c*c);

            norm = sqrt(hyperbola->nx[idx]*hyperbola->nx[idx] +
                        hyperbola->ny[idx]*hyperbola->ny[idx] +
                        hyperbola->nz[idx]*hyperbola->nz[idx]);

            hyperbola->nx[idx] = nfac * hyperbola->nx[idx] / norm;
            hyperbola->ny[idx] = nfac * hyperbola->ny[idx] / norm;
            hyperbola->nz[idx] = nfac * hyperbola->nz[idx] / norm;

            hyperbola->area[idx] = norm * u * du * dv;

            /*
            prefac = nfac / sqrt(b*b * c*c * (u*u - 1) * cos(v)*cos(v) +
                      a*a * c*c * (u*u - 1) * sin(v)*sin(v) + a*a * b*b * u*u);

            hyperbola->nx[idx] = -b * c * sqrt(u*u - 1) * cos(v) * prefac;
            hyperbola->ny[idx] = -a * c * sqrt(u*u - 1) * sin(v) * prefac;
            hyperbola->nz[idx] = b * a * u * prefac;

            hyperbola->area[idx] = sqrt(b*b * c*c * (u*u - 1) * cos(v)*cos(v) +
                                        a*a * c*c * (u*u - 1) * sin(v)*sin(v) + a*a * b*b * u*u) * du * dv;
            */
            if (transform)
            {
                transformGrids<T, U>(hyperbola, idx, inp, out, &ut, mat);
            }
        }
    }
}

template<typename T, typename U>
void Ellipse_xy(T *ellipse, U xu_lo, U xu_up, U yv_lo,
                U yv_up, U xcenter, U ycenter, U a, U b, U c, int ncx, int ncy, int nfac,
                U mat[16], bool transform)
{
    U dx = (xu_up - xu_lo) / (ncx - 1);
    U dy = (yv_up - yv_lo) / (ncy - 1);

    // Generate using xy parametrisation
    U x;
    U y;
    U prefac;

    U norm;

    Utils<U> ut;


    std::array<U, 3> inp, out;

    for (int i=0; i < ncx; i++)
    {
        x = i * dx + xu_lo;

        for (int j=0; j < ncy; j++)
        {
            y = j * dy + yv_lo;
            int idx = i*ncy + j;

            ellipse->x[idx] = x;
            ellipse->y[idx] = y;
            ellipse->z[idx] = c * sqrt(1 - x*x / (a*a) - y*y / (b*b));

            ellipse->nx[idx] = 2 * x / (a*a);
            ellipse->ny[idx] = 2 * y / (b*b);
            ellipse->nz[idx] = 2 * ellipse->z[idx] / (c*c);

            norm = sqrt(ellipse->nx[idx]*ellipse->nx[idx] +
                        ellipse->ny[idx]*ellipse->ny[idx] +
                        ellipse->nz[idx]*ellipse->nz[idx]);

            ellipse->nx[idx] = nfac * ellipse->nx[idx] / norm;
            ellipse->ny[idx] = nfac * ellipse->ny[idx] / norm;
            ellipse->nz[idx] = nfac * ellipse->nz[idx] / norm;

            ellipse->area[idx] = norm * dx * dy;

            if (transform)
            {
                transformGrids<T, U>(ellipse, idx, inp, out, &ut, mat);
            }
        }
    }
}

template<typename T, typename U>
void Ellipse_uv(T *ellipse, U xu_lo, U xu_up, U yv_lo,
                U yv_up, U xcenter, U ycenter, U a, U b, U c, int ncx, int ncy, int nfac,
                U mat[16], bool transform)
{
    U du = (asin(xu_up/a) - asin(xu_lo/a)) / (ncx - 1);
    U dv = (yv_up - yv_lo) / (ncy - 1);

    // Generate using xy parametrisation
    U u;
    U v;
    U prefac;

    U norm;

    Utils<U> ut;


    std::array<U, 3> inp, out;

    for (int i=0; i < ncx; i++)
    {
        u = i * du + asin(xu_lo/a);
        for (int j=0; j < ncy; j++)
        {
            v = (j * dv + yv_lo) * M_PI/180;
            int idx = i*ncy + j;

            ellipse->x[idx] = a * sin(u) * cos(v);
            ellipse->y[idx] = b * sin(u) * sin(v);
            ellipse->z[idx] = c * cos(u);

            prefac = nfac / sqrt(b*b * c*c * sin(u)*sin(u) * cos(v)*cos(v) +
                      a*a * c*c * sin(u)*sin(u) * sin(v)*sin(v) + a*a * b*b * cos(u)*cos(u));

            ellipse->nx[idx] = b * c * sin(u) * cos(v) * prefac;
            ellipse->ny[idx] = a * c * sin(u) * sin(v) * prefac;
            ellipse->nz[idx] = b * a * cos(u) * prefac;

            ellipse->area[idx] = sin(u) * sqrt(b*b * c*c * sin(u)*sin(u) * cos(v)*cos(v) +
                                        a*a * c*c * sin(u)*sin(u) * sin(v)*sin(v) +
                                        a*a * b*b * cos(u)*cos(u)) * du * dv;

            if (transform)
            {
                transformGrids<T, U>(ellipse, idx, inp, out, &ut, mat);
            }
        }
    }
}

template<typename T, typename U>
void Plane_xy(T *plane, U xu_lo, U xu_up, U yv_lo,
              U yv_up, U xcenter, U ycenter, int ncx, int ncy, int nfac,
              U mat[16], bool transform)
{
    U dx = (xu_up - xu_lo) / (ncx - 1);
    U dy = (yv_up - yv_lo) / (ncy - 1);

    // Generate using xy parametrisation
    U x;
    U y;
    U prefac;

    U norm;

    Utils<U> ut;


    std::array<U, 3> inp, out;

    for (int i=0; i < ncx; i++)
    {
        x = i * dx + xu_lo;
        for (int j=0; j < ncy; j++)
        {
            y = j * dy + yv_lo;
            int idx = i*ncy + j;

            plane->x[idx] = x;
            plane->y[idx] = y;
            plane->z[idx] = 0;

            plane->nx[idx] = 0;
            plane->ny[idx] = 0;
            plane->nz[idx] = nfac * 1;

            plane->area[idx] = dx * dy;

            if (transform)
            {
                transformGrids<T, U>(plane, idx, inp, out, &ut, mat);
            }
        }
    }
}

template<typename T, typename U>
void Plane_uv(T *plane, U xu_lo, U xu_up, U yv_lo,
              U yv_up, U xcenter, U ycenter, U a, U b, int ncx, int ncy, int nfac,
              U mat[16], bool transform)
{


    // Generate using xy parametrisation
    U u, du, duv, dux, duy;
    U v, dv, r;

    dux = a / (ncx - 1);
    duy = b / (ncx - 1);

    dv = (yv_up - yv_lo) / (ncy - 1) * M_PI/180;
    
    U majmin = b / a;

    U norm;

    Utils<U> ut;


    std::array<U, 3> inp, out;

    for (int i=0; i < ncx; i++)
    {        
        du = (xu_up - xu_lo) / (ncx - 1);
        u = i * du + xu_lo;

        //dux = a / (ncx - 1);
        //duy = b / (ncx - 1);

        for (int j=0; j < ncy; j++)
        {
            v = (j * dv + yv_lo);
            int idx = i*ncy + j;
          
            //du = dux * cos(v) + duy * sin(v); 
            //u = i * du + xu_lo;
            
            plane->x[idx] = u * cos(v);
            plane->y[idx] = u * sin(v) * majmin;
            plane->z[idx] = 0;

            r = u * sqrt(cos(v)*cos(v) + sin(v)*sin(v)*majmin*majmin);

            plane->nx[idx] = 0;
            plane->ny[idx] = 0;
            plane->nz[idx] = nfac * 1;

            duv = sqrt(dux*dux*cos(v)*cos(v) + duy*duy*sin(v)*sin(v));

            plane->area[idx] = r * duv * dv;// / (a*a*cos(v)*cos(v) + sin(v)*sin(v)*b*b);// * (cos(v) + sin(v)*majmin);// * (cos(v) + sin(v)*majmin);
            //plane->area[idx] = u * du * dv;//a*a * b*b / 2 * dv / (b*b * cos(v)*cos(v) + a*a * sin(v)*sin(v));
            if (transform)
            {
                transformGrids<T, U>(plane, idx, inp, out, &ut, mat);
            }
        }
    }
}

template<typename T, typename U>
void Plane_AoE(T *plane, U xu_lo, U xu_up, U yv_lo,
              U yv_up, U xcenter, U ycenter, int ncx, int ncy,
              U mat[16], bool transform, bool spheric)
{
    U dA = (xu_up - xu_lo) / (ncx - 1);
    U dE = (yv_up - yv_lo) / (ncy - 1);

    // Generate using xy parametrisation
    U Az;
    U El;

    Utils<U> ut;

    std::array<U, 3> inp, out;

    for (int i=0; i < ncx; i++)
    {
        Az = i * dA + xu_lo;

        for (int j=0; j < ncy; j++)
        {
            El = j * dE + yv_lo;
            int idx = i*ncy + j;

            if (spheric)
            {
                plane->x[idx] = sqrt(Az*Az + El*El) * M_PI/180;
                plane->y[idx] = atan2(El, Az);
                //plane->y[idx] = constrainAngle<U>(atan(El / Az));

                    if (plane->y[idx] != plane->y[idx])
                    {
                        plane->y[idx] = 0;
                    }
            }

            else
            {
                plane->x[idx] = Az;
                plane->y[idx] = El;
            }

            plane->z[idx] = 0;

            plane->nx[idx] = 0;
            plane->ny[idx] = 0;
            plane->nz[idx] = 1;

            plane->area[idx] = 1;

            if (transform)
            {
                transformGrids<T, U>(plane, idx, inp, out, &ut, mat);
            }
        }
    }
}

extern "C" void generateGrid(reflparams refl, reflcontainer *container, bool transform, bool spheric)
{
    // For readability, assign new temporary placeholders
    double xu_lo = refl.lxu[0];
    double xu_up = refl.lxu[1];
    double yv_lo = refl.lyv[0];
    double yv_up = refl.lyv[1];

    double xcenter = refl.gcenter[0];
    double ycenter = refl.gcenter[1];

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
                                            yv_up, xcenter, ycenter, a, b, ncx, ncy, nfac, refl.transf, transform);
        }

        else if (refl.type == 1)
        {
            Hyperbola_xy<reflcontainer, double>(container, xu_lo, xu_up, yv_lo,
                                            yv_up, xcenter, ycenter, a, b, c, ncx, ncy, nfac, refl.transf, transform);
        }

        else if (refl.type == 2)
        {
            Ellipse_xy<reflcontainer, double>(container, xu_lo, xu_up, yv_lo,
                                            yv_up, xcenter, ycenter, a, b, c, ncx, ncy, nfac, refl.transf, transform);
        }

        else if (refl.type == 3)
        {
            Plane_xy<reflcontainer, double>(container, xu_lo, xu_up, yv_lo,
                                            yv_up, xcenter, ycenter, ncx, ncy, nfac, refl.transf, transform);
        }
    }

    else if (refl.gmode == 1)
    {
        if (refl.type == 0)
        {

            Parabola_uv<reflcontainer, double>(container, xu_lo, xu_up, yv_lo,
                                            yv_up, xcenter, ycenter, a, b, ncx, ncy, nfac, refl.transf, transform);
        }

        else if (refl.type == 1)
        {
            Hyperbola_uv<reflcontainer, double>(container, xu_lo, xu_up, yv_lo,
                                            yv_up, xcenter, ycenter, a, b, c, ncx, ncy, nfac, refl.transf, transform);
        }

        else if (refl.type == 2)
        {
            Ellipse_uv<reflcontainer, double>(container, xu_lo, xu_up, yv_lo,
                                            yv_up, xcenter, ycenter, a, b, c, ncx, ncy, nfac, refl.transf, transform);
        }

        else if (refl.type == 3)
        {
            Plane_uv<reflcontainer, double>(container, xu_lo, xu_up, yv_lo,
                                            yv_up, xcenter, ycenter, a, b, ncx, ncy, nfac, refl.transf, transform);
        }
    }

    if (refl.gmode == 2)
    {
        Plane_AoE<reflcontainer, double>(container, xu_lo, xu_up, yv_lo,
                                        yv_up, xcenter, ycenter, ncx, ncy, refl.transf, transform, spheric);
    }
}

extern "C" void generateGridf(reflparamsf refl, reflcontainerf *container, bool transform, bool spheric)
{
    // For readability, assign new temporary placeholders
    float xu_lo = refl.lxu[0];
    float xu_up = refl.lxu[1];
    float yv_lo = refl.lyv[0];
    float yv_up = refl.lyv[1];

    float xcenter = refl.gcenter[0];
    float ycenter = refl.gcenter[1];

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
                                            yv_up, xcenter, ycenter, a, b, ncx, ncy, nfac, refl.transf, transform);
        }

        else if (refl.type == 1)
        {
            Hyperbola_xy<reflcontainerf, float>(container, xu_lo, xu_up, yv_lo,
                                            yv_up, xcenter, ycenter, a, b, c, ncx, ncy, nfac, refl.transf, transform);
        }

        else if (refl.type == 2)
        {
            Ellipse_xy<reflcontainerf, float>(container, xu_lo, xu_up, yv_lo,
                                            yv_up, xcenter, ycenter, a, b, c, ncx, ncy, nfac, refl.transf, transform);
        }

        else if (refl.type == 3)
        {
            Plane_xy<reflcontainerf, float>(container, xu_lo, xu_up, yv_lo,
                                            yv_up, xcenter, ycenter, ncx, ncy, nfac, refl.transf, transform);
        }
    }

    else if (refl.gmode == 1)
    {
        if (refl.type == 0)
        {

            Parabola_uv<reflcontainerf, float>(container, xu_lo, xu_up, yv_lo,
                                            yv_up, xcenter, ycenter, a, b, ncx, ncy, nfac, refl.transf, transform);
        }

        else if (refl.type == 1)
        {
            Hyperbola_uv<reflcontainerf, float>(container, xu_lo, xu_up, yv_lo,
                                            yv_up, xcenter, ycenter, a, b, c, ncx, ncy, nfac, refl.transf, transform);
        }

        else if (refl.type == 2)
        {
            Ellipse_uv<reflcontainerf, float>(container, xu_lo, xu_up, yv_lo,
                                            yv_up, xcenter, ycenter, a, b, c, ncx, ncy, nfac, refl.transf, transform);
        }

        else if (refl.type == 3)
        {
            Plane_uv<reflcontainerf, float>(container, xu_lo, xu_up, yv_lo,
                                            yv_up, xcenter, ycenter, a, b, ncx, ncy, nfac, refl.transf, transform);
        }
    }

    else if (refl.gmode == 2)
    {
        Plane_AoE<reflcontainerf, float>(container, xu_lo, xu_up, yv_lo,
                                        yv_up, xcenter, ycenter, ncx, ncy, refl.transf, transform, spheric);
    }
}
