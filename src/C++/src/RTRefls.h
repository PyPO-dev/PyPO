// Simple, bare-bones definitions of reflectors for CPU/GPU RT.
// Use only for raytracing; for PO, use the elaborate definitions
// Include derivatives necessary for Newton method
#include <cmath>
#define FLEPS 1.0E-8

#ifndef __RTRefls_h
#define __RTRefls_h

template<class T>
class RTRefls
{
    T a, b, c;

public:
    RTRefls(T a, T b, T c);

    T common1(T t, T xr, T yr, T dxr, T dyr);
    T common2(T t, T xr, T yr, T dxr, T dyr);
    T gp(T t, T xr, T yr, T zr, T dxr, T dyr, T dzr);
    T gh(T t, T xr, T yr, T zr, T dxr, T dyr, T dzr);
    T ge(T t, T xr, T yr, T zr, T dxr, T dyr, T dzr);
    T gpl(T t, T xr, T yr, T zr, T dxr, T dyr, T dzr);

    std::array<T, 3> np(T xr, T yr, T zr, int flip);
    std::array<T, 3> nhe(T xr, T yr, T zr, int flip);
    std::array<T, 3> npl(T xr, T yr, T zr, int flip);
};

template<class T>
RTRefls<T>::RTRefls(T a, T b, T c)
{
    this->a = a;
    this->b = b;
    this->c = c;
}

template<class T>
inline T RTRefls<T>::common1(T t, T xr, T yr, T dxr, T dyr)
{
    return (xr + t*dxr)*(xr + t*dxr)/(a*a) + (yr + t*dyr)*(yr + t*dyr)/(b*b);
}

template<class T>
inline T RTRefls<T>::common2(T t, T xr, T yr, T dxr, T dyr)
{
    return (xr + t*dxr)*2*dxr/(a*a) + (yr + t*dyr)*2*dyr/(b*b);
}

template<class T>
inline T RTRefls<T>::gp(T t, T xr, T yr, T zr, T dxr, T dyr, T dzr)
{
    return t - (zr + t*dzr - common1(t, xr, yr, dxr, dyr)) /
                (dzr - common2(t, xr, yr, dxr, dyr));
}

template<class T>
inline T RTRefls<T>::gh(T t, T xr, T yr, T zr, T dxr, T dyr, T dzr)
{
    return t - (zr + t*dzr - c*sqrt(common1(t, xr, yr, dxr, dyr) + 1)) /
                (dzr - c/(2*sqrt(common1(t, xr, yr, dxr, dyr) + 1)) *
                common2(t, xr, yr, dxr, dyr));
}

template<class T>
inline T RTRefls<T>::ge(T t, T xr, T yr, T zr, T dxr, T dyr, T dzr)
{
    return t - (zr + t*dzr - c*sqrt(1 - common1(t, xr, yr, dxr, dyr))) /
                (dzr + c/(2*sqrt(1 - common1(t, xr, yr, dxr, dyr))) *
                common2(t, xr, yr, dxr, dyr));
}

template<class T>
inline T RTRefls<T>::gpl(T t, T xr, T yr, T zr, T dxr, T dyr, T dzr)
{
    return t - (zr + t*dzr - 11e3) / dzr;
}

template<class T>
inline std::array<T, 3> RTRefls<T>::np(T xr, T yr, T zr, int flip)
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

template<class T>
inline std::array<T, 3> RTRefls<T>::nhe(T xr, T yr, T zr, int flip)
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

template<class T>
inline std::array<T, 3> RTRefls<T>::npl(T xr, T yr, T zr, int flip)
{
    std::array<T, 3> out;

    out[0] = 0;
    out[1] = 0;
    out[2] = 1;

    return out;
}
#endif
