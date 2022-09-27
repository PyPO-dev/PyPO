#include <iostream>
#include <vector>
#include <complex>
#include <array>
#include <cmath>
#include <thread>
#include <iomanip>

#ifndef __Utilsf_h
#define __Utilsf_h

/* The Utils class contains often used vector operations. All operations are overloaded to deal with
 * complex and real vectors of length 3 by means of template.
 */

class Utilsf
{
public:
    // Dot products
    void dot(const std::array<float, 3> &v1, const std::array<float, 3> &v2, float &out);
    void dot(const std::array<std::complex<float>, 3> &cv1, const std::array<std::complex<float>, 3> &cv2, std::complex<float> &out);
    void dot(const std::array<std::complex<float>, 3> &cv1, const std::array<float, 3> &v2, std::complex<float> &out);
    void dot(const std::array<float, 3> &v1, const std::array<std::complex<float>, 3> &cv2, std::complex<float> &out);
    
    // Overloaded cross products
    void ext(const std::array<float, 3> &v1, const std::array<float, 3> &v2, std::array<float, 3> &out);
    void ext(const std::array<std::complex<float>, 3> &cv1, const std::array<std::complex<float>, 3> &cv2, std::array<std::complex<float>, 3> &out);
    void ext(const std::array<std::complex<float>, 3> &cv1, const std::array<float, 3> &v2, std::array<std::complex<float>, 3> &out);
    void ext(const std::array<float, 3> &v1, const std::array<std::complex<float>, 3> &cv2, std::array<std::complex<float>, 3> &out);
    
    // Overloaded absolute value
    void abs(const std::array<float, 3> &v, float &out);
    void abs(const std::array<std::complex<float>, 3> &cv, std::complex<float> &out);
    
    // Difference vectors
    void diff(const std::array<float, 3> &v1, const std::array<float, 3> &v2, std::array<float, 3> &out);
    void diff(const std::array<std::complex<float>, 3> &cv1, const std::array<std::complex<float>, 3> &cv2, std::array<std::complex<float>, 3> &out);
    
    // Normalization
    void normalize(const std::array<float, 3> &v, std::array<float, 3> &out);
    void normalize(const std::array<std::complex<float>, 3> &cv, std::array<std::complex<float>, 3> &out);
    
    // Scalar multiplication
    void s_mult(const std::array<float, 3> &v, const float &s, std::array<float, 3> &out);
    void s_mult(const std::array<std::complex<float>, 3> &cv, const std::complex<float> &cs, std::array<std::complex<float>, 3> &out);
    void s_mult(const std::array<float, 3> &v, const std::complex<float> &cs, std::array<std::complex<float>, 3> &out);
    void s_mult(const std::array<std::complex<float>, 3> &cv, const float &s, std::array<std::complex<float>, 3> &out);
    
    // Conjugation of complex vector
    void conj(const std::array<std::complex<float>, 3> &cv, std::array<std::complex<float>, 3> &out);
    
    // Snell's function
    void snell(const std::array<std::complex<float>, 3> &cvin, const std::array<float, 3> &normal, std::array<std::complex<float>, 3> &out);
    void snell(const std::array<float, 3> &vin, const std::array<float, 3> &normal, std::array<float, 3> &out);
    
    // Dyadic products
    void dyad(const std::array<float, 3> &v1, const std::array<float, 3> &v2, std::array<std::array<float, 3>, 3> &out);
    
    // Real valued matrix-matrix subtraction
    void matDiff(const std::array<std::array<float, 3>, 3> &m1, const std::array<std::array<float, 3>, 3> &m2, std::array<std::array<float, 3>, 3> &out);
    
    // Matrix-vector multiplication
    void matVec(const std::array<std::array<float, 3>, 3> &m1, const std::array<float, 3> &v1, std::array<float, 3> &out);
    void matVec(const std::array<std::array<float, 3>, 3> &m1, const std::array<std::complex<float>, 3> &cv1, std::array<std::complex<float>, 3> &out);
    
};
#endif 

// Real dot-product
inline void Utilsf::dot(const std::array<float, 3> &v1, const std::array<float, 3> &v2, float &out)
{
    out = 0;
    
    for(int n=0; n<3; n++)
    {
        out += v1[n] * v2[n];
    }
}


// Complex hermitian conjugate inner-product
inline void Utilsf::dot(const std::array<std::complex<float>, 3> &cv1, const std::array<std::complex<float>, 3> &cv2, std::complex<float> &out)
{
    out = (0, 0);
    
    for(int n=0; n<3; n++)
    {
        out += std::conj(cv1[n]) * cv2[n];
    }
}

// Complex vector - real vector dot-product
inline void Utilsf::dot(const std::array<std::complex<float>, 3> &cv1, const std::array<float, 3> &v2, std::complex<float> &out)
{
    out = (0, 0);
    
    for(int n=0; n<3; n++)
    {
        out += std::conj(cv1[n]) * v2[n];
    }
}

// Real vector - complex vector dot-product
inline void Utilsf::dot(const std::array<float, 3> &v1, const std::array<std::complex<float>, 3> &cv2, std::complex<float> &out)
{
    out = (0, 0);
    
    for(int n=0; n<3; n++)
    {
        out += v1[n] * cv2[n];
    }
}

// Real cross-product
inline void Utilsf::ext(const std::array<float, 3> &v1, const std::array<float, 3> &v2, std::array<float, 3> &out)
{
    out[0] = v1[1]*v2[2] - v1[2]*v2[1];
    out[1] = v1[2]*v2[0] - v1[0]*v2[2];
    out[2] = v1[0]*v2[1] - v1[1]*v2[0];
}


// Complex conjugate of cross product
inline void Utilsf::ext(const std::array<std::complex<float>, 3> &cv1, const std::array<std::complex<float>, 3> &cv2, std::array<std::complex<float>, 3> &out)
{
    out[0] = cv1[1]*cv2[2] - cv1[2]*cv2[1];
    out[1] = cv1[2]*cv2[0] - cv1[0]*cv2[2];
    out[2] = cv1[0]*cv2[1] - cv1[1]*cv2[0];
}

// Cross product between an complex and a real vector
inline void Utilsf::ext(const std::array<std::complex<float>, 3> &cv1, const std::array<float, 3> &v2, std::array<std::complex<float>, 3> &out)
{
    out[0] = cv1[1]*v2[2] - cv1[2]*v2[1];
    out[1] = cv1[2]*v2[0] - cv1[0]*v2[2];
    out[2] = cv1[0]*v2[1] - cv1[1]*v2[0];
}

// Cross product between a real vector and a complex vector
inline void Utilsf::ext(const std::array<float, 3> &v1, const std::array<std::complex<float>, 3> &cv2, std::array<std::complex<float>, 3> &out)
{
    out[0] = v1[1]*cv2[2] - v1[2]*cv2[1];
    out[1] = v1[2]*cv2[0] - v1[0]*cv2[2];
    out[2] = v1[0]*cv2[1] - v1[1]*cv2[0];
}

// Difference between two real vectors
inline void Utilsf::diff(const std::array<float, 3> &v1, const std::array<float, 3> &v2, std::array<float, 3> &out)
{
    for(int n=0; n<3; n++)
    {
        out[n] = v1[n] - v2[n];
    }
}

// Difference between two complex valued vectors
inline void Utilsf::diff(const std::array<std::complex<float>, 3> &cv1, const std::array<std::complex<float>, 3> &cv2, std::array<std::complex<float>, 3> &out)
{
    for(int n=0; n<3; n++)
    {
        out[n] = cv1[n] - cv2[n];
    }
}

// Absolute value of real vector
inline void Utilsf::abs(const std::array<float, 3> &v, float &out)
{
    dot(v, v, out);
    out = std::sqrt(out);
}

// Absolute value of a complex vector. Still returns a complex number!
inline void Utilsf::abs(const std::array<std::complex<float>, 3> &cv, std::complex<float> &out)
{
    std::array<std::complex<float>, 3> cv_conj;
    conj(cv, cv_conj);
    dot(cv, cv_conj, out);
    out = std::sqrt(out);
}

// Return normalized real vector from vector
inline void Utilsf::normalize(const std::array<float, 3> &v, std::array<float, 3> &out)
{
    float norm;
    abs(v, norm);
    
    for( int n=0; n<3; n++)
    {
        out[n] = v[n] / norm;
    }
}

// Normalize complex vector
inline void Utilsf::normalize(const std::array<std::complex<float>, 3> &cv, std::array<std::complex<float>, 3> &out)
{
    std::complex<float> cnorm;
    abs(cv, cnorm);
    
    for( int n=0; n<3; n++)
    {
        out[n] = cv[n] / cnorm;
    }
}

// Apply standard real s-multiplication on a real vector
inline void Utilsf::s_mult(const std::array<float, 3> &v, const float &s, std::array<float, 3> &out)
{
    for(int n=0; n<3; n++)
    {
        out[n] = s * v[n];
    }
}


// Multiply complex vector by complex scalar
inline void Utilsf::s_mult(const std::array<std::complex<float>, 3> &cv, const std::complex<float> &cs, std::array<std::complex<float>, 3> &out)
{
    for(int n=0; n<3; n++)
    {
        out[n] = cs * cv[n];
    }
}

// Multiply real vector by complex scalar
inline void Utilsf::s_mult(const std::array<float, 3> &v, const std::complex<float> &cs, std::array<std::complex<float>, 3> &out)
{
    for(int n=0; n<3; n++)
    {
        out[n] = cs * v[n];
    }
}

// Multiply complex vector by real scalar
inline void Utilsf::s_mult(const std::array<std::complex<float>, 3> &cv, const float &s, std::array<std::complex<float>, 3> &out)
{
    for(int n=0; n<3; n++)
    {
        out[n] = s * cv[n];
    }
}



// Return complex conjugate of complex vector
inline void Utilsf::conj(const std::array<std::complex<float>, 3> &cv, std::array<std::complex<float>, 3> &out)
{
    for(int n=0; n<3; n++)
    {
        out[n] = std::conj(cv[n]);
    }
}

// Calculate refected vector from surface using complex incoming vector and real normal vector to surface
inline void Utilsf::snell(const std::array<std::complex<float>, 3> &cvin, const std::array<float, 3> &normal, std::array<std::complex<float>, 3> &out)
{
    std::complex<float> cfactor;
    dot(cvin, normal, cfactor);
    
    cfactor = (float)2. * cfactor;
    
    std::array<std::complex<float>, 3> rhs;
    s_mult(normal, cfactor, rhs);
    
    diff(cvin, rhs, out);
}

// Calculate refected vector from surface using complex incoming vector and real normal vector to surface
inline void Utilsf::snell(const std::array<float, 3> &vin, const std::array<float, 3> &normal, std::array<float, 3> &out)
{
    float factor;
    dot(vin, normal, factor);
    
    factor = (float)2. * factor;
    
    std::array<float, 3> rhs;
    s_mult(normal, factor, rhs);
    
    diff(vin, rhs, out);
}

// Calculate Dyadic product between two real vectors
// Returns array of length 3, containing 3 arrays representing ROWS in the resulting matrix
inline void Utilsf::dyad(const std::array<float, 3> &v1, const std::array<float, 3> &v2, std::array<std::array<float, 3>, 3> &out)
{
    for(int n=0; n<3; n++)
    {
        out[n][0] = v1[n] * v2[0];
        out[n][1] = v1[n] * v2[1];
        out[n][2] = v1[n] * v2[2];
    }
}

// Subtract matrix from another matrix element-wise
inline void Utilsf::matDiff(const std::array<std::array<float, 3>, 3> &m1, const std::array<std::array<float, 3>, 3> &m2, std::array<std::array<float, 3>, 3> &out)
{
    for(int n=0; n<3; n++)
    {
        out[n][0] = m1[n][0] - m2[n][0];
        out[n][1] = m1[n][1] - m2[n][1];
        out[n][2] = m1[n][2] - m2[n][2];
    }
}

// Multiply matrix with vector to return vector
inline void Utilsf::matVec(const std::array<std::array<float, 3>, 3> &m1, const std::array<float, 3> &v1, std::array<float, 3> &out)
{
    for(int n=0; n<3; n++)
    {
        out[n] = m1[n][0] * v1[0] + m1[n][1] * v1[1] + m1[n][2] * v1[2];
    }
}

inline void Utilsf::matVec(const std::array<std::array<float, 3>, 3> &m1, const std::array<std::complex<float>, 3> &cv1, std::array<std::complex<float>, 3> &out)
{
    for(int n=0; n<3; n++)
    {
        out[n] = m1[n][0] * cv1[0] + m1[n][1] * cv1[1] + m1[n][2] * cv1[2];
    }
}


