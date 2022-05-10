#include <iostream>
#include <vector>
#include <complex>
#include <array>
#include <cmath>
#include <thread>
#include <iomanip>

#ifndef __Utils_h
#define __Utils_h

/* The Utils class contains often used vector operations. All operations are overloaded to deal with
 * complex and real vectors of length 3 by means of template.
 */

class Utils
{
public:
    // Dot products
    double dot(const std::vector<double> &v1, const std::vector<double> &v2);
    std::complex<double> dot(const std::vector<std::complex<double>> &cv1, const std::vector<std::complex<double>> &cv2);
    std::complex<double> dot(const std::vector<std::complex<double>> &cv1, const std::vector<double> &v2);
    std::complex<double> dot(const std::vector<double> &v1, const std::vector<std::complex<double>> &cv2);
    
    // Overloaded cross products
    std::vector<double> ext(const std::vector<double> &v1, const std::vector<double> &v2);
    std::vector<std::complex<double>> ext(const std::vector<std::complex<double>> &cv1, const std::vector<std::complex<double>> &cv2);
    std::vector<std::complex<double>> ext(const std::vector<std::complex<double>> &cv1, const std::vector<double> &v2);
    std::vector<std::complex<double>> ext(const std::vector<double> &v1, const std::vector<std::complex<double>> &cv2);
    
    // Overloaded absolute value
    double abs(const std::vector<double> &v);
    std::complex<double> abs(const std::vector<std::complex<double>> &cv);
    
    // Difference vectors
    std::vector<double> diff(const std::vector<double> &v1, const std::vector<double> &v2);
    std::vector<std::complex<double>> diff(const std::vector<std::complex<double>> &cv1, const std::vector<std::complex<double>> &cv2);
    
    // Normalization
    std::vector<double> normalize(const std::vector<double> &v);
    std::vector<std::complex<double>> normalize(const std::vector<std::complex<double>> &cv);
    
    // Scalar multiplication
    std::vector<double> s_mult(const std::vector<double> &v, const double &s);
    std::vector<std::complex<double>> s_mult(const std::vector<std::complex<double>> &cv, const std::complex<double> &cs);
    std::vector<std::complex<double>> s_mult(const std::vector<double> &v, const std::complex<double> &cs);
    std::vector<std::complex<double>> s_mult(const std::vector<std::complex<double>> &cv, const double &s);
    
    // Conjugation of complex vector
    std::vector<std::complex<double>> conj(const std::vector<std::complex<double>> &cv);
    
    // Snell's function
    std::vector<std::complex<double>> snell(const std::vector<std::complex<double>> &cvin, const std::vector<double> &normal);
    std::vector<double> snell(const std::vector<double> &vin, const std::vector<double> &normal);
    
};
#endif 

// Real dot-product
inline double Utils::dot(const std::vector<double> &v1, const std::vector<double> &v2)
{
    double out = 0.;
    
    for(int n=0; n<3; n++)
    {
        out += v1[n] * v2[n];
    }
    return out;
}


// Complex hermitian conjugate inner-product
inline std::complex<double> Utils::dot(const std::vector<std::complex<double>> &cv1, const std::vector<std::complex<double>> &cv2)
{
    std::complex<double> out(0., 0.);
    
    for(int n=0; n<3; n++)
    {
        out += std::conj(cv1[n]) * cv2[n];
    }
    return out;
}

// Complex vector - real vector dot-product
inline std::complex<double> Utils::dot(const std::vector<std::complex<double>> &cv1, const std::vector<double> &v2)
{
    std::complex<double> out(0., 0.);
    
    for(int n=0; n<3; n++)
    {
        out += std::conj(cv1[n]) * v2[n];
    }
    return out;
}

// Real vector - complex vector dot-product
inline std::complex<double> Utils::dot(const std::vector<double> &v1, const std::vector<std::complex<double>> &cv2)
{
    std::complex<double> out(0., 0.);
    
    for(int n=0; n<3; n++)
    {
        out += v1[n] * cv2[n];
    }
    return out;
}

// Real cross-product
inline std::vector<double> Utils::ext(const std::vector<double> &v1, const std::vector<double> &v2)
{
    std::vector<double> out(3, 0.);
    out[0] = v1[1]*v2[2] - v1[2]*v2[1];
    out[1] = v1[2]*v2[0] - v1[0]*v2[2];
    out[2] = v1[0]*v2[1] - v1[1]*v2[0];
    
    return out;
}


// Complex conjugate of cross product
inline std::vector<std::complex<double>> Utils::ext(const std::vector<std::complex<double>> &cv1, const std::vector<std::complex<double>> &cv2)
{
    std::vector<std::complex<double>> out(3, (0., 0.));
    out[0] = cv1[1]*cv2[2] - cv1[2]*cv2[1];
    out[1] = cv1[2]*cv2[0] - cv1[0]*cv2[2];
    out[2] = cv1[0]*cv2[1] - cv1[1]*cv2[0];
    
    return out;
}

// Cross product between an complex and a real vector
inline std::vector<std::complex<double>> Utils::ext(const std::vector<std::complex<double>> &cv1, const std::vector<double> &v2)
{
    std::vector<std::complex<double>> out(3, (0., 0.));
    out[0] = cv1[1]*v2[2] - cv1[2]*v2[1];
    out[1] = cv1[2]*v2[0] - cv1[0]*v2[2];
    out[2] = cv1[0]*v2[1] - cv1[1]*v2[0];
    
    return out;
}

// Cross product between a real vector and a complex vector
inline std::vector<std::complex<double>> Utils::ext(const std::vector<double> &v1, const std::vector<std::complex<double>> &cv2)
{
    std::vector<std::complex<double>> out(3, (0., 0.));
    out[0] = v1[1]*cv2[2] - v1[2]*cv2[1];
    out[1] = v1[2]*cv2[0] - v1[0]*cv2[2];
    out[2] = v1[0]*cv2[1] - v1[1]*cv2[0];
    
    return out;
}

// Difference between two real vectors
inline std::vector<double> Utils::diff(const std::vector<double> &v1, const std::vector<double> &v2)
{
    std::vector<double> out(3, 0.);
    
    for(int n=0; n<3; n++)
    {
        out[n] = v1[n] - v2[n];
    }
    return out;
}

// Difference between two complex valued vectors
inline std::vector<std::complex<double>> Utils::diff(const std::vector<std::complex<double>> &cv1, const std::vector<std::complex<double>> &cv2)
{
    std::vector<std::complex<double>> out(3, (0., 0.));
    
    for(int n=0; n<3; n++)
    {
        out[n] = cv1[n] - cv2[n];
    }
    return out;
}

// Absolute value of real vector
inline double Utils::abs(const std::vector<double> &v)
{
    double out = dot(v, v);
    out = std::sqrt(out);
    
    return out;
}

// Absolute value of a complex vector. Still returns a complex number!
inline std::complex<double> Utils::abs(const std::vector<std::complex<double>> &cv)
{
    std::vector<std::complex<double>> cv_conj = conj(cv);
    std::complex<double> out = dot(cv, cv_conj);
    out = std::sqrt(out);
    
    return out;
}

// Return normalized real vector from vector
inline std::vector<double> Utils::normalize(const std::vector<double> &v)
{
    std::vector<double> out(3, 0.);
    double norm = abs(v);
    
    for( int n=0; n<3; n++)
    {
        out[n] = v[n] / norm;
    }
    return out;
}

// Normalize complex vector
inline std::vector<std::complex<double>> Utils::normalize(const std::vector<std::complex<double>> &cv)
{
    std::vector<std::complex<double>> out(3, (0., 0.));
    std::complex<double> cnorm = abs(cv);
    //double norm = cnorm.real();
    
    for( int n=0; n<3; n++)
    {
        out[n] = cv[n] / cnorm;
    }
    return out;
}

// Apply standard real s-multiplication on a real vector
inline std::vector<double> Utils::s_mult(const std::vector<double> &v, const double &s)
{
    std::vector<double> out(3, 0.);
    
    for(int n=0; n<3; n++)
    {
        out[n] = s * v[n];
    }
    return out;
}


// Multiply complex vector by complex scalar
inline std::vector<std::complex<double>> Utils::s_mult(const std::vector<std::complex<double>> &cv, const std::complex<double> &cs)
{
    std::vector<std::complex<double>> out(3, (0., 0.));
    
    for(int n=0; n<3; n++)
    {
        out[n] = cs * cv[n];
    }
    return out;
}

// Multiply real vector by complex scalar
inline std::vector<std::complex<double>> Utils::s_mult(const std::vector<double> &v, const std::complex<double> &cs)
{
    std::vector<std::complex<double>> out(3, (0., 0.));
    
    for(int n=0; n<3; n++)
    {
        out[n] = cs * v[n];
    }
    return out;
}

// Multiply complex vector by real scalar
inline std::vector<std::complex<double>> Utils::s_mult(const std::vector<std::complex<double>> &cv, const double &s)
{
    std::vector<std::complex<double>> out(3, (0., 0.));
    
    for(int n=0; n<3; n++)
    {
        out[n] = s * cv[n];
    }
    return out;
}



// Return complex conjugate of complex vector
inline std::vector<std::complex<double>> Utils::conj(const std::vector<std::complex<double>> &cv)
{
    std::vector<std::complex<double>> out(3, (0., 0.));
    
    for(int n=0; n<3; n++)
    {
        out[n] = std::conj(cv[n]);
    }
    return out;
}

// Calculate refected vector from surface using complex incoming vector and real normal vector to surface
inline std::vector<std::complex<double>> Utils::snell(const std::vector<std::complex<double>> &cvin, const std::vector<double> &normal)
{
    std::vector<std::complex<double>> out(3, (0., 0.));
    std::complex<double> cfactor = 2. * dot(cvin, normal);
    
    std::vector<std::complex<double>> rhs = s_mult(normal, cfactor);
    
    out = diff(cvin, rhs);
    return out;
}

// Calculate refected vector from surface using complex incoming vector and real normal vector to surface
inline std::vector<double> Utils::snell(const std::vector<double> &vin, const std::vector<double> &normal)
{
    std::vector<double> out(3, 0.);
    double factor = 2. * dot(vin, normal);
    
    std::vector<double> rhs = s_mult(normal, factor);
    
    out = diff(vin, rhs);
    return out;
}



