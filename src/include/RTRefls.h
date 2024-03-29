#include <cmath>
#include "Utils.h"

#ifndef __RTRefls_h
#define __RTRefls_h

/*! \file RTRefls.h
    \brief Simple reflector definitions used for PyPO ray-tracer. 
        
    Simple, bare-bones definitions of reflectors for CPU/GPU ray-tracing.
        Use only for raytracing; for PO, use the elaborate definitions.
*/


/** 
 * Simple representation of reflectors and implementations for the Newton-Rhapson method.
 */
template<class T>
class RTRefls
{
public:
    Utils<T> ut;

    static T common1(T t, T xr, T yr, T dxr, T dyr, T a, T b);
    static T common2(T t, T xr, T yr, T dxr, T dyr, T a, T b);
    static T gp(T t, T xr, T yr, T zr, T dxr, T dyr, T dzr, T a, T b, T c);
    static T gh(T t, T xr, T yr, T zr, T dxr, T dyr, T dzr, T a, T b, T c);
    static T ge(T t, T xr, T yr, T zr, T dxr, T dyr, T dzr, T a, T b, T c);
    static T gpl(T t, T xr, T yr, T zr, T dxr, T dyr, T dzr, T a, T b, T c);

    static std::array<T, 3> np(T xr, T yr, T zr, int flip, T a, T b, T c);
    static std::array<T, 3> nh(T xr, T yr, T zr, int flip, T a, T b, T c);
    static std::array<T, 3> ne(T xr, T yr, T zr, int flip, T a, T b, T c);
    static std::array<T, 3> npl(T xr, T yr, T zr, int flip, T a, T b, T c);
};

/**
 * Common factor 1 for all reflectors.
 *
 * Calculate common factor 1. These calculations are done separately as these
 *      factors are common to all reflectors.
 *
 * @param t Scaling factor of ray, double/float.
 * @param xr Current x co-ordinate of ray, double/float.
 * @param yr Current y co-ordinate of ray, double/float.
 * @param dxr Component of ray direction along x-axis, double/float.
 * @param dyr Component of ray direction along y-axis, double/float.
 * @param a Scale factor along x-axis, double/float.
 * @param b Scale factor along y-axis, double/float.
 * @returns z Co-ordinate along z-axis on reflector, corresponding to ray co-ordinate, direction and scaling. Double/float.
 */
template<class T>
inline T RTRefls<T>::common1(T t, T xr, T yr, T dxr, T dyr, T a, T b)
{
    return (xr + t*dxr)*(xr + t*dxr)/(a*a) + (yr + t*dyr)*(yr + t*dyr)/(b*b);
}

/**
 * Common factor 2 for all reflectors.
 *
 * Calculate common factor 2. These calculations are done separately as these
 *      factors are common to all reflectors.
 *
 * @param t Scaling factor of ray, double/float.
 * @param xr Current x co-ordinate of ray, double/float.
 * @param yr Current y co-ordinate of ray, double/float.
 * @param dxr Component of ray direction along x-axis.
 * @param dyr Component of ray direction along y-axis.
 * @param a Scale factor along x-axis, double/float.
 * @param b Scale factor along y-axis, double/float.
 * @returns z Co-ordinate along z-axis on reflector, corresponding to ray co-ordinate, direction and scaling. Double/float.
 */
template<class T>
inline T RTRefls<T>::common2(T t, T xr, T yr, T dxr, T dyr, T a, T b)
{
    return (xr + t*dxr)*2*dxr/(a*a) + (yr + t*dyr)*2*dyr/(b*b);
}

/**
 * Grid paraboloid.
 *
 * Calculate difference between paraboloid z co-ordinate and ray z co-ordinate corresponding to x and y ray co-ordinates.
 *
 * @param t Scaling factor of ray, double/float.
 * @param xr Current x co-ordinate of ray, double/float.
 * @param yr Current y co-ordinate of ray, double/float.
 * @param zr Current z co-ordinate of ray, double/float.
 * @param dxr Component of ray direction along x-axis.
 * @param dyr Component of ray direction along y-axis.
 * @param dzr Component of ray direction along z-axis.
 * @param a Scale factor along x-axis, double/float.
 * @param b Scale factor along y-axis, double/float.
 * @param c Scale factor along z-axis, double/float (hyperboloid/ellipsoid only).
 * @returns dz Difference between reflector and ray co-ordinate along z-axis.
 */
template<class T>
inline T RTRefls<T>::gp(T t, T xr, T yr, T zr, T dxr, T dyr, T dzr, T a, T b, T c)
{
    return t - (zr + t*dzr - RTRefls::common1(t, xr, yr, dxr, dyr, a, b)) /
                (dzr - RTRefls::common2(t, xr, yr, dxr, dyr, a, b));
}

/**
 * Grid hyperboloid.
 *
 * Calculate difference between hyperboloid z co-ordinate and ray z co-ordinate corresponding to x and y ray co-ordinates.
 *
 * @param t Scaling factor of ray, double/float.
 * @param xr Current x co-ordinate of ray, double/float.
 * @param yr Current y co-ordinate of ray, double/float.
 * @param zr Current z co-ordinate of ray, double/float.
 * @param dxr Component of ray direction along x-axis.
 * @param dyr Component of ray direction along y-axis.
 * @param dzr Component of ray direction along z-axis.
 * @param a Scale factor along x-axis, double/float.
 * @param b Scale factor along y-axis, double/float.
 * @param c Scale factor along z-axis, double/float (hyperboloid/ellipsoid only).
 * @returns dz Difference between reflector and ray co-ordinate along z-axis.
 */
template<class T>
inline T RTRefls<T>::gh(T t, T xr, T yr, T zr, T dxr, T dyr, T dzr, T a, T b, T c)
{
    return t - (zr + t*dzr - c*sqrt(RTRefls::common1(t, xr, yr, dxr, dyr, a, b) + 1)) /
                (dzr - c/(2*sqrt(RTRefls::common1(t, xr, yr, dxr, dyr, a, b) + 1)) *
                RTRefls::common2(t, xr, yr, dxr, dyr, a, b));
}

/**
 * Grid ellipsoid.
 *
 * Calculate difference between ellipsoid z co-ordinate and ray z co-ordinate corresponding to x and y ray co-ordinates.
 *
 * @param t Scaling factor of ray, double/float.
 * @param xr Current x co-ordinate of ray, double/float.
 * @param yr Current y co-ordinate of ray, double/float.
 * @param zr Current z co-ordinate of ray, double/float.
 * @param dxr Component of ray direction along x-axis.
 * @param dyr Component of ray direction along y-axis.
 * @param dzr Component of ray direction along z-axis.
 * @param a Scale factor along x-axis, double/float.
 * @param b Scale factor along y-axis, double/float.
 * @param c Scale factor along z-axis, double/float (hyperboloid/ellipsoid only).
 * @returns dz Difference between reflector and ray co-ordinate along z-axis.
 */
template<class T>
inline T RTRefls<T>::ge(T t, T xr, T yr, T zr, T dxr, T dyr, T dzr, T a, T b, T c)
{
    return t - (zr + t*dzr - c*sqrt(1 - RTRefls::common1(t, xr, yr, dxr, dyr, a, b))) /
                (dzr + c/(2*sqrt(1 - RTRefls::common1(t, xr, yr, dxr, dyr, a, b))) *
                RTRefls::common2(t, xr, yr, dxr, dyr, a, b));
}

/**
 * Grid plane.
 *
 * Calculate difference between plane z co-ordinate and ray z co-ordinate corresponding to x and y ray co-ordinates.
 *
 * @param t Scaling factor of ray, double/float.
 * @param xr Current x co-ordinate of ray, double/float.
 * @param yr Current y co-ordinate of ray, double/float.
 * @param zr Current z co-ordinate of ray, double/float.
 * @param dxr Component of ray direction along x-axis.
 * @param dyr Component of ray direction along y-axis.
 * @param dzr Component of ray direction along z-axis.
 * @param a Scale factor along x-axis, double/float.
 * @param b Scale factor along y-axis, double/float.
 * @param c Scale factor along z-axis, double/float (hyperboloid/ellipsoid only).
 * @returns dz Difference between reflector and ray co-ordinate along z-axis.
 */
template<class T>
inline T RTRefls<T>::gpl(T t, T xr, T yr, T zr, T dxr, T dyr, T dzr, T a, T b, T c)
{
    return t - (zr + t*dzr) / dzr;
}

/**
 * Paraboloid normal vectors.
 * 
 * Calculate normal vectors to point on paraboloid.
 *
 * @param xr Current x co-ordinate of ray, double/float.
 * @param yr Current y co-ordinate of ray, double/float.
 * @param zr Current z co-ordinate of ray, double/float.
 * @param flip Direction of normal vectors.
 * @param a Scale factor along x-axis, double/float.
 * @param b Scale factor along y-axis, double/float.
 * @param c Scale factor along z-axis, double/float (hyperboloid/ellipsoid only).
 * @returns out Array of 3 double/float, containing components of normal vector.
 */
template<class T>
inline std::array<T, 3> RTRefls<T>::np(T xr, T yr, T zr, int flip, T a, T b, T c)
{
    std::array<T, 3> out;

    out[0] = -2 * xr / (a*a) * flip;
    out[1] = -2 * yr / (b*b) * flip;
    out[2] = flip;

    T norm = sqrt(out[0]*out[0] + out[1]*out[1] + out[2]*out[2]);

    out[0] = out[0] / norm;
    out[1] = out[1] / norm;
    out[2] = out[2] / norm;

    return out;
}

/**
 * Hyperboloid normal vectors.
 * 
 * Calculate normal vectors to point on hyperboloid.
 *
 * @param xr Current x co-ordinate of ray, double/float.
 * @param yr Current y co-ordinate of ray, double/float.
 * @param zr Current z co-ordinate of ray, double/float.
 * @param flip Direction of normal vectors.
 * @param a Scale factor along x-axis, double/float.
 * @param b Scale factor along y-axis, double/float.
 * @param c Scale factor along z-axis, double/float (hyperboloid/ellipsoid only).
 * @returns out Array of 3 double/float, containing components of normal vector.
 */
template<class T>
inline std::array<T, 3> RTRefls<T>::nh(T xr, T yr, T zr, int flip, T a, T b, T c)
{
    std::array<T, 3> out;

    out[0] = -2 * xr / (a*a) * flip;
    out[1] = -2 * yr / (b*b) * flip;
    out[2] = 2 * zr / (c*c) * flip;

    T norm = sqrt(out[0]*out[0] + out[1]*out[1] + out[2]*out[2]);

    out[0] = out[0] / norm;
    out[1] = out[1] / norm;
    out[2] = out[2] / norm;

    return out;
}

/**
 * Ellipsoid normal vectors.
 * 
 * Calculate normal vectors to point on ellipsoid.
 *
 * @param xr Current x co-ordinate of ray, double/float.
 * @param yr Current y co-ordinate of ray, double/float.
 * @param zr Current z co-ordinate of ray, double/float.
 * @param flip Direction of normal vectors.
 * @param a Scale factor along x-axis, double/float.
 * @param b Scale factor along y-axis, double/float.
 * @param c Scale factor along z-axis, double/float (hyperboloid/ellipsoid only).
 * @returns out Array of 3 double/float, containing components of normal vector.
 */
template<class T>
inline std::array<T, 3> RTRefls<T>::ne(T xr, T yr, T zr, int flip, T a, T b, T c)
{
    std::array<T, 3> out;

    out[0] = 2 * xr / (a*a) * flip;
    out[1] = 2 * yr / (b*b) * flip;
    out[2] = 2 * zr / (c*c) * flip;

    T norm = sqrt(out[0]*out[0] + out[1]*out[1] + out[2]*out[2]);

    out[0] = out[0] / norm;
    out[1] = out[1] / norm;
    out[2] = out[2] / norm;

    return out;
}

/**
 * Plane normal vectors.
 * 
 * Calculate normal vectors to point on plane.
 *      Seems sort of redundant, but implement this way for consistency.
 *
 * @param xr Current x co-ordinate of ray, double/float.
 * @param yr Current y co-ordinate of ray, double/float.
 * @param zr Current z co-ordinate of ray, double/float.
 * @param flip Direction of normal vectors.
 * @param a Scale factor along x-axis, double/float.
 * @param b Scale factor along y-axis, double/float.
 * @param c Scale factor along z-axis, double/float (hyperboloid/ellipsoid only).
 * @returns out Array of 3 double/float, containing components of normal vector.
 */
template<class T>
inline std::array<T, 3> RTRefls<T>::npl(T xr, T yr, T zr, int flip, T a, T b, T c)
{
    std::array<T, 3> out;

    out[0] = 0;
    out[1] = 0;
    out[2] = 1;

    return out;
}
#endif
