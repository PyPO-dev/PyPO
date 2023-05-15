#include <iostream>
#include <vector>
#include <array>
#define _USE_MATH_DEFINES
#include <cmath>

#include "Utils.h"
#include "InterfaceReflector.h"

/*! \file InterfaceReflector.cpp
    \brief Implementation of reflector objects.
    
    General definitions of reflector objects. Generates reflectors using xy, uv or Azimuth over Elevation parametrization.
        Associated normal vectors and area element sizes are also returned.
*/

/**
 * Transform reflector grids.
 *
 * Applies rotations and translations to reflector, co-ordinate wise.
 *
 * @param reflc Pointer to reflcontainer or reflcontainerf object.
 * @param idx Index of co-ordinate.
 * @param inp Array of 3 double/float.
 * @param out Array of 3 double/float.
 * @param ut Pointer to Utils object.
 * @param mat Array of 16 double/float.
 *
 * @see Utils
 * @see Structs
 * @see reflcontainer
 * @see reflcontainerfa
 */
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

/**
 * Generate paraboloid from xy parametrization.
 *
 * Generate a paraboloid using an xy parametrization. Also generates the normal vectors and area elements.
 *      Applies specified transformations as well, if enabled.
 *
 * @param parabola Pointer to reflcontainer or reflcontainerf object.
 * @param xu_lo Lower limit on x co-ordinate, double/float.
 * @param xu_up Upper limit on x co-ordinate, double/float.
 * @param yv_lo Lower limit on y co-ordinate, double/float.
 * @param yv_up Upper limit on y co-ordinate, double/float.
 * @param xcenter Center x co-ordinate of xy region.
 * @param ycenter Center y co-ordinate of xy region.
 * @param a Scaling factor along x-axis.
 * @param b Scaling factor along y-axis.
 * @param ncx Number of cells along x-axis.
 * @param ncy Number of cells along y-axis.
 * @param nfac Flip normal vectors.
 * @param mat Array of 16 double/float, transformation matrix.
 * @param transform Whether or not to apply transformation to reflector.
 *
 * @see reflcontainer
 * @see reflcontainerf
 */
template<typename T, typename U>
void Parabola_xy(T *parabola, U xu_lo, U xu_up, U yv_lo,
                U yv_up, U xcenter, U ycenter, U a, U b, int ncx, int ncy, int nfac,
                U mat[16], bool transform)
{
    U dx = (xu_up - xu_lo) / (ncx - 1);
    U dy = (yv_up - yv_lo) / (ncy - 1);

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

/**
 * Generate paraboloid from uv parametrization.
 *
 * Generate a paraboloid using a uv parametrization. Also generates the normal vectors and area elements.
 *      Applies specified transformations as well, if enabled.
 *
 * @param parabola Pointer to reflcontainer or reflcontainerf object.
 * @param xu_lo Lower limit on u co-ordinate, double/float.
 * @param xu_up Upper limit on u co-ordinate, double/float.
 * @param yv_lo Lower limit on v co-ordinate, double/float.
 * @param yv_up Upper limit on v co-ordinate, double/float.
 * @param xcenter Center x co-ordinate of uv region.
 * @param ycenter Center y co-ordinate of uv region.
 * @param ecc_uv Eccentricity of uv-generated ellipse in xy-grid, double/float.
 * @param rot_uv Position angle of uv-generated ellipse in xy-grid, double/float.
 * @param a Scaling factor along x-axis.
 * @param b Scaling factor along y-axis.
 * @param ncx Number of cells along u-axis.
 * @param ncy Number of cells along v-axis.
 * @param nfac Flip normal vectors.
 * @param mat Array of 16 double/float, transformation matrix.
 * @param transform Whether or not to apply transformation to reflector.
 *
 * @see reflcontainer
 * @see reflcontainerf
 */
template<typename T, typename U>
void Parabola_uv(T *parabola, U xu_lo, U xu_up, U yv_lo,
                U yv_up, U xcenter, U ycenter, U ecc_uv, U rot_uv, U a, U b, int ncx, int ncy, int nfac,
                U mat[16], bool transform)
{
    U du = (xu_up - xu_lo) / (ncx - 1);
    U dv = (yv_up - yv_lo) / (ncy - 1);

    U majmin = sqrt(1 - ecc_uv*ecc_uv);
    
    U u, duv;
    U v;
    U prefac;

    U x, y, xr, yr, r, norm;

    Utils<U> ut;


    std::array<U, 3> inp, out;

    for (int i=0; i < ncx; i++)
    {
        u = i * du + xu_lo;
        for (int j=0; j < ncy; j++)
        {
            v = (j * dv + yv_lo) * M_PI/180;
            int idx = i*ncy + j;

            x = u * cos(v);
            y = u * sin(v) * majmin;

            r = sqrt(x*x + y*y);

            xr = x * cos(rot_uv) - y * sin(rot_uv);
            yr = x * sin(rot_uv) + y * cos(rot_uv);

            x = xr + xcenter;
            y = yr + ycenter;

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

            duv = sqrt(du*du*cos(v)*cos(v) + du*du*sin(v)*sin(v)*majmin*majmin);
            parabola->area[idx] = norm * r * duv * dv;

            if (transform)
            {
                transformGrids<T, U>(parabola, idx, inp, out, &ut, mat);
            }
        }
    }
}

/**
 * Generate hyperboloid from xy parametrization.
 *
 * Generate a hyperboloid using an xy parametrization. Also generates the normal vectors and area elements.
 *      Applies specified transformations as well, if enabled.
 *
 * @param hyperbola Pointer to reflcontainer or reflcontainerf object.
 * @param xu_lo Lower limit on x co-ordinate, double/float.
 * @param xu_up Upper limit on x co-ordinate, double/float.
 * @param yv_lo Lower limit on y co-ordinate, double/float.
 * @param yv_up Upper limit on y co-ordinate, double/float.
 * @param xcenter Center x co-ordinate of xy region.
 * @param ycenter Center y co-ordinate of xy region.
 * @param a Scaling factor along x-axis.
 * @param b Scaling factor along y-axis.
 * @param c Scaling factor along z-axis
 * @param ncx Number of cells along x-axis.
 * @param ncy Number of cells along y-axis.
 * @param nfac Flip normal vectors.
 * @param mat Array of 16 double/float, transformation matrix.
 * @param transform Whether or not to apply transformation to reflector.
 *
 * @see reflcontainer
 * @see reflcontainerf
 */
template<typename T, typename U>
void Hyperbola_xy(T *hyperbola, U xu_lo, U xu_up, U yv_lo,
                  U yv_up, U xcenter, U ycenter, U a, U b, U c, int ncx, int ncy, int nfac,
                  U mat[16], bool transform)
{
    U dx = (xu_up - xu_lo) / (ncx - 1);
    U dy = (yv_up - yv_lo) / (ncy - 1);

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

/**
 * Generate hyperboloid from uv parametrization.
 *
 * Generate a hyperboloid using a uv parametrization. Also generates the normal vectors and area elements.
 *      Applies specified transformations as well, if enabled.
 *
 * @param hyperbola Pointer to reflcontainer or reflcontainerf object.
 * @param xu_lo Lower limit on u co-ordinate, double/float.
 * @param xu_up Upper limit on u co-ordinate, double/float.
 * @param yv_lo Lower limit on v co-ordinate, double/float.
 * @param yv_up Upper limit on v co-ordinate, double/float.
 * @param xcenter Center x co-ordinate of xy region.
 * @param ycenter Center y co-ordinate of xy region.
 * @param ecc_uv Eccentricity of uv-generated ellipse in xy-grid, double/float.
 * @param rot_uv Position angle of uv-generated ellipse in xy-grid, double/float.
 * @param a Scaling factor along x-axis.
 * @param b Scaling factor along y-axis.
 * @param c Scaling factor along z-axis
 * @param ncx Number of cells along x-axis.
 * @param ncy Number of cells along y-axis.
 * @param nfac Flip normal vectors.
 * @param mat Array of 16 double/float, transformation matrix.
 * @param transform Whether or not to apply transformation to reflector.
 *
 * @see reflcontainer
 * @see reflcontainerf
 */
template<typename T, typename U>
void Hyperbola_uv(T *hyperbola, U xu_lo, U xu_up, U yv_lo,
                  U yv_up, U xcenter, U ycenter, U ecc_uv, U rot_uv, U a, U b, U c, int ncx, int ncy, int nfac,
                  U mat[16], bool transform)
{
    U du = (xu_up - xu_lo) / (ncx - 1);
    U dv = (yv_up - yv_lo) / (ncy - 1);

    U majmin = sqrt(1 - ecc_uv*ecc_uv);
    
    U u, duv;
    U v;
    U prefac;

    U x, y, xr, yr, r, norm;

    Utils<U> ut;

    std::array<U, 3> inp, out;

    for (int i=0; i < ncx; i++)
    {
        u = i * du + xu_lo;
        for (int j=0; j < ncy; j++)
        {
            v = (j * dv + yv_lo) * M_PI/180;
            int idx = i*ncy + j;

            x = u * cos(v);
            y = u * sin(v) * majmin;

            r = sqrt(x*x + y*y);

            xr = x * cos(rot_uv) - y * sin(rot_uv);
            yr = x * sin(rot_uv) + y * cos(rot_uv);

            x = xr + xcenter;
            y = yr + ycenter;

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

            duv = sqrt(du*du*cos(v)*cos(v) + du*du*sin(v)*sin(v)*majmin*majmin);
            hyperbola->area[idx] = norm * r * duv * dv;

            if (transform)
            {
                transformGrids<T, U>(hyperbola, idx, inp, out, &ut, mat);
            }
        }
    }
}

/**
 * Generate ellipsoid from xy parametrization.
 *
 * Generate a ellipsoid using an xy parametrization. Also generates the normal vectors and area elements.
 *      Applies specified transformations as well, if enabled.
 *
 * @param ellipse Pointer to reflcontainer or reflcontainerf object.
 * @param xu_lo Lower limit on x co-ordinate, double/float.
 * @param xu_up Upper limit on x co-ordinate, double/float.
 * @param yv_lo Lower limit on y co-ordinate, double/float.
 * @param yv_up Upper limit on y co-ordinate, double/float.
 * @param xcenter Center x co-ordinate of xy region.
 * @param ycenter Center y co-ordinate of xy region.
 * @param a Scaling factor along x-axis.
 * @param b Scaling factor along y-axis.
 * @param c Scaling factor along z-axis
 * @param ncx Number of cells along x-axis.
 * @param ncy Number of cells along y-axis.
 * @param nfac Flip normal vectors.
 * @param mat Array of 16 double/float, transformation matrix.
 * @param transform Whether or not to apply transformation to reflector.
 *
 * @see reflcontainer
 * @see reflcontainerf
 */
template<typename T, typename U>
void Ellipse_xy(T *ellipse, U xu_lo, U xu_up, U yv_lo,
                U yv_up, U xcenter, U ycenter, U a, U b, U c, int ncx, int ncy, int nfac,
                U mat[16], bool transform)
{
    U dx = (xu_up - xu_lo) / (ncx - 1);
    U dy = (yv_up - yv_lo) / (ncy - 1);

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

/**
 * Generate ellipsoid from uv parametrization.
 *
 * Generate a ellipsoid using a uv parametrization. Also generates the normal vectors and area elements.
 *      Applies specified transformations as well, if enabled.
 *
 * @param ellipse Pointer to reflcontainer or reflcontainerf object.
 * @param xu_lo Lower limit on u co-ordinate, double/float.
 * @param xu_up Upper limit on u co-ordinate, double/float.
 * @param yv_lo Lower limit on v co-ordinate, double/float.
 * @param yv_up Upper limit on v co-ordinate, double/float.
 * @param xcenter Center x co-ordinate of xy region.
 * @param ycenter Center y co-ordinate of xy region.
 * @param ecc_uv Eccentricity of uv-generated ellipse in xy-grid, double/float.
 * @param rot_uv Position angle of uv-generated ellipse in xy-grid, double/float.
 * @param a Scaling factor along x-axis.
 * @param b Scaling factor along y-axis.
 * @param c Scaling factor along z-axis
 * @param ncx Number of cells along x-axis.
 * @param ncy Number of cells along y-axis.
 * @param nfac Flip normal vectors.
 * @param mat Array of 16 double/float, transformation matrix.
 * @param transform Whether or not to apply transformation to reflector.
 *
 * @see reflcontainer
 * @see reflcontainerf
 */
template<typename T, typename U>
void Ellipse_uv(T *ellipse, U xu_lo, U xu_up, U yv_lo,
                U yv_up, U xcenter, U ycenter, U ecc_uv, U rot_uv, U a, U b, U c, int ncx, int ncy, int nfac,
                U mat[16], bool transform)
{
    U du = (xu_up - xu_lo) / (ncx - 1);
    U dv = (yv_up - yv_lo) / (ncy - 1);

    U majmin = sqrt(1 - ecc_uv*ecc_uv);
    U u, duv;
    U v;
    U prefac;

    U x, y, xr, yr, r, norm;

    Utils<U> ut;

    std::array<U, 3> inp, out;

    for (int i=0; i < ncx; i++)
    {
        u = i * du + xu_lo;
        for (int j=0; j < ncy; j++)
        {
            v = (j * dv + yv_lo) * M_PI/180;
            int idx = i*ncy + j;

            x = u * cos(v);
            y = u * sin(v) * majmin;

            r = sqrt(x*x + y*y);

            xr = x * cos(rot_uv) - y * sin(rot_uv);
            yr = x * sin(rot_uv) + y * cos(rot_uv);

            x = xr + xcenter;
            y = yr + ycenter;

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

            duv = sqrt(du*du*cos(v)*cos(v) + du*du*sin(v)*sin(v)*majmin*majmin);
            ellipse->area[idx] = norm * r * duv * dv;
            
            if (transform)
            {
                transformGrids<T, U>(ellipse, idx, inp, out, &ut, mat);
            }
        }
    }
}

/**
 * Generate plane from xy parametrization.
 *
 * Generate a plane using an xy parametrization. Also generates the normal vectors and area elements.
 *      Applies specified transformations as well, if enabled.
 *
 * @param plane Pointer to reflcontainer or reflcontainerf object.
 * @param xu_lo Lower limit on x co-ordinate, double/float.
 * @param xu_up Upper limit on x co-ordinate, double/float.
 * @param yv_lo Lower limit on y co-ordinate, double/float.
 * @param yv_up Upper limit on y co-ordinate, double/float.
 * @param ncx Number of cells along x-axis.
 * @param ncy Number of cells along y-axis.
 * @param nfac Flip normal vectors.
 * @param mat Array of 16 double/float, transformation matrix.
 * @param transform Whether or not to apply transformation to reflector.
 *
 * @see reflcontainer
 * @see reflcontainerf
 */
template<typename T, typename U>
void Plane_xy(T *plane, U xu_lo, U xu_up, U yv_lo,
              U yv_up, int ncx, int ncy, int nfac,
              U mat[16], bool transform)
{
    U dx = (xu_up - xu_lo) / (ncx - 1);
    U dy = (yv_up - yv_lo) / (ncy - 1);

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

/**
 * Generate plane from uv parametrization.
 *
 * Generate a plane using an uv parametrization. Also generates the normal vectors and area elements.
 *      Applies specified transformations as well, if enabled.
 *
 * @param plane Pointer to reflcontainer or reflcontainerf object.
 * @param xu_lo Lower limit on x co-ordinate, double/float.
 * @param xu_up Upper limit on x co-ordinate, double/float.
 * @param yv_lo Lower limit on y co-ordinate, double/float.
 * @param yv_up Upper limit on y co-ordinate, double/float.
 * @param xcenter Center x co-ordinate of xy region.
 * @param ycenter Center y co-ordinate of xy region.
 * @param ecc_uv Eccentricity of uv-generated ellipse in xy-grid, double/float.
 * @param rot_uv Position angle of uv-generated ellipse in xy-grid, double/float.
 * @param ncx Number of cells along x-axis.
 * @param ncy Number of cells along y-axis.
 * @param nfac Flip normal vectors.
 * @param mat Array of 16 double/float, transformation matrix.
 * @param transform Whether or not to apply transformation to reflector.
 *
 * @see reflcontainer
 * @see reflcontainerf
 */
template<typename T, typename U>
void Plane_uv(T *plane, U xu_lo, U xu_up, U yv_lo,
              U yv_up, U xcenter, U ycenter, U ecc_uv, U rot_uv, int ncx, int ncy, int nfac,
              U mat[16], bool transform)
{


    U u, du, duv;
    U v, dv, r;
    U x, xr, y, yr;

    U majmin = sqrt(1 - ecc_uv*ecc_uv);
    
    du = (xu_up - xu_lo) / (ncx - 1);
    dv = (yv_up - yv_lo) / (ncy - 1) * M_PI/180;

    U norm;

    Utils<U> ut;

    std::array<U, 3> inp, out;

    for (int i=0; i < ncx; i++)
    {        
        u = i * du + xu_lo;

        for (int j=0; j < ncy; j++)
        {
            v = (j * dv + yv_lo);
            int idx = i*ncy + j;
            
            // Calculate co-ordinate of ellipse in rest frame
            x = u * cos(v);
            y = u * sin(v) * majmin;

            // Calculate distance of x, y point to origin in rest frame.
            // Rotation preserves area elements
            r = sqrt(x*x + y*y);
            
            // Rotate ellipse in rest frame
            xr = x * cos(rot_uv) - y * sin(rot_uv);
            yr = x * sin(rot_uv) + y * cos(rot_uv);

            // Add center offset to xy grids
            plane->x[idx] = xr + xcenter;
            plane->y[idx] = yr + ycenter;
            plane->z[idx] = 0;

            plane->nx[idx] = 0;
            plane->ny[idx] = 0;
            plane->nz[idx] = nfac * 1;

            // Calculate du along value of v, dv unchanged.
            duv = sqrt(du*du*cos(v)*cos(v) + du*du*sin(v)*sin(v)*majmin*majmin);

            plane->area[idx] = r * duv * dv;            
            if (transform)
            {
                transformGrids<T, U>(plane, idx, inp, out, &ut, mat);
            }
        }
    }
}

/**
 * Generate plane from AoE (on-sky angles) parametrization.
 *
 * Generate a plane using an AoE parametrization. Used for calculating far-fields only.
 *
 * @param plane Pointer to reflcontainer or reflcontainerf object.
 * @param xu_lo Lower limit on Az co-ordinate, double/float.
 * @param xu_up Upper limit on Az co-ordinate, double/float.
 * @param yv_lo Lower limit on El co-ordinate, double/float.
 * @param yv_up Upper limit on El co-ordinate, double/float.
 * @param xcenter Center Az co-ordinate of Az-El region.
 * @param ycenter Center El co-ordinate of Az-El region.
 * @param ncx Number of cells along Az-axis.
 * @param ncy Number of cells along El-axis.
 * @param mat Array of 16 double/float, transformation matrix.
 * @param transform Whether or not to apply transformation to plane.
 * @param spheric Whether or not to apply spherical transformation to plane, for plotting purposes..
 *
 * @see reflcontainer
 * @see reflcontainerf
 */
template<typename T, typename U>
void Plane_AoE(T *plane, U xu_lo, U xu_up, U yv_lo,
              U yv_up, U xcenter, U ycenter, int ncx, int ncy,
              U mat[16], bool transform, bool spheric)
{
    U dA = (xu_up - xu_lo) / (ncx - 1);
    U dE = (yv_up - yv_lo) / (ncy - 1);

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

/**
 * Generate reflector/far-field grids.
 * 
 * Generate x, y, z grids and corresponding normal vectors nx, ny, nz. Also generates area elements.
 *      For far-field grids, generates Az and El grid, leaves z-container at 0 values.
 *
 * @param refl A reflparams object containing reflector parameters.
 * @param container Pointer to reflcontainer object.
 * @param transform Whether or not to apply transformation to reflector.
 * @param spheric Whether or not to convert a far-field grid to spherical co-ordinates, for plotting purposes.
 *
 * @see reflparams
 * @see reflcontainer
 */
void generateGrid(reflparams refl, reflcontainer *container, bool transform, bool spheric)
{
    double xu_lo = refl.lxu[0];
    double xu_up = refl.lxu[1];
    double yv_lo = refl.lyv[0];
    double yv_up = refl.lyv[1];

    double xcenter = refl.gcenter[0];
    double ycenter = refl.gcenter[1];

    double ecc_uv = refl.ecc_uv;
    double rot_uv = refl.rot_uv * M_PI / 180;

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
                                            yv_up, ncx, ncy, nfac, refl.transf, transform);
        }
    }

    else if (refl.gmode == 1)
    {
        if (refl.type == 0)
        {

            Parabola_uv<reflcontainer, double>(container, xu_lo, xu_up, yv_lo,
                                            yv_up, xcenter, ycenter, ecc_uv, rot_uv, a, b, ncx, ncy, nfac, refl.transf, transform);
        }

        else if (refl.type == 1)
        {
            Hyperbola_uv<reflcontainer, double>(container, xu_lo, xu_up, yv_lo,
                                            yv_up, xcenter, ycenter, ecc_uv, rot_uv, a, b, c, ncx, ncy, nfac, refl.transf, transform);
        }

        else if (refl.type == 2)
        {
            Ellipse_uv<reflcontainer, double>(container, xu_lo, xu_up, yv_lo,
                                            yv_up, xcenter, ycenter, ecc_uv, rot_uv, a, b, c, ncx, ncy, nfac, refl.transf, transform);
        }

        else if (refl.type == 3)
        {
            Plane_uv<reflcontainer, double>(container, xu_lo, xu_up, yv_lo,
                                            yv_up, xcenter, ycenter, ecc_uv, rot_uv, ncx, ncy, nfac, refl.transf, transform);
        }
    }

    if (refl.gmode == 2)
    {
        Plane_AoE<reflcontainer, double>(container, xu_lo, xu_up, yv_lo,
                                        yv_up, xcenter, ycenter, ncx, ncy, refl.transf, transform, spheric);
    }
}

/**
 * Generate reflector/far-field grids.
 * 
 * Generate x, y, z grids and corresponding normal vectors nx, ny, nz. Also generates area elements.
 *      For far-field grids, generates Az and El grid, leaves z-container at 0 values.
 *
 * @param refl A reflparamsf object containing reflector parameters.
 * @param container Pointer to reflcontainerf object.
 * @param transform Whether or not to apply transformation to reflector.
 * @param spheric Whether or not to convert a far-field grid to spherical co-ordinates, for plotting purposes.
 *
 * @see reflparamsf
 * @see reflcontainerf
 */
void generateGridf(reflparamsf refl, reflcontainerf *container, bool transform, bool spheric)
{
    // for readability, assign new temporary placeholders
    float xu_lo = refl.lxu[0];
    float xu_up = refl.lxu[1];
    float yv_lo = refl.lyv[0];
    float yv_up = refl.lyv[1];

    float xcenter = refl.gcenter[0];
    float ycenter = refl.gcenter[1];

    float ecc_uv = refl.ecc_uv;
    float rot_uv = refl.rot_uv * M_PI / 180;

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
                                            yv_up, ncx, ncy, nfac, refl.transf, transform);
        }
    }

    else if (refl.gmode == 1)
    {
        if (refl.type == 0)
        {

            Parabola_uv<reflcontainerf, float>(container, xu_lo, xu_up, yv_lo,
                                            yv_up, xcenter, ycenter, ecc_uv, rot_uv, a, b, ncx, ncy, nfac, refl.transf, transform);
        }

        else if (refl.type == 1)
        {
            Hyperbola_uv<reflcontainerf, float>(container, xu_lo, xu_up, yv_lo,
                                            yv_up, xcenter, ycenter, ecc_uv, rot_uv, a, b, c, ncx, ncy, nfac, refl.transf, transform);
        }

        else if (refl.type == 2)
        {
            Ellipse_uv<reflcontainerf, float>(container, xu_lo, xu_up, yv_lo,
                                            yv_up, xcenter, ycenter, ecc_uv, rot_uv, a, b, c, ncx, ncy, nfac, refl.transf, transform);
        }

        else if (refl.type == 3)
        {
            Plane_uv<reflcontainerf, float>(container, xu_lo, xu_up, yv_lo,
                                            yv_up, xcenter, ycenter, ecc_uv, rot_uv, ncx, ncy, nfac, refl.transf, transform);
        }
    }

    else if (refl.gmode == 2)
    {
        Plane_AoE<reflcontainerf, float>(container, xu_lo, xu_up, yv_lo,
                                        yv_up, xcenter, ycenter, ncx, ncy, refl.transf, transform, spheric);
    }
}
