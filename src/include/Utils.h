#include <complex>
#include <array>
#include <cmath>
#include <limits>

#define _USE_MATH_DEFINES

#ifndef __Utils_h
#define __Utils_h

/*! \file Utils.h
    \brief Linear algebra functions for the CPU version of PyPO. 
    
    Contains double/float overloaded functions for doing basic 3D vector operations.
*/

/**
 * Class for basic 3D linear algebra functions.
 *
 * Note that no function returns. All values are stored inside a variable which is passed by reference to the function.
 */
template <typename T> class Utils
{
public:
    // Dot products
    void dot(const std::array<T, 3> &v1, const std::array<T, 3> &v2, T &out);
    void dot(const std::array<std::complex<T>, 3> &cv1, const std::array<std::complex<T>, 3> &cv2, std::complex<T> &out);
    void dot(const std::array<std::complex<T>, 3> &cv1, const std::array<T, 3> &v2, std::complex<T> &out);
    void dot(const std::array<T, 3> &v1, const std::array<std::complex<T>, 3> &cv2, std::complex<T> &out);

    // Overloaded cross products
    void ext(const std::array<T, 3> &v1, const std::array<T, 3> &v2, std::array<T, 3> &out);
    void ext(const std::array<std::complex<T>, 3> &cv1, const std::array<std::complex<T>, 3> &cv2, std::array<std::complex<T>, 3> &out);
    void ext(const std::array<std::complex<T>, 3> &cv1, const std::array<T, 3> &v2, std::array<std::complex<T>, 3> &out);
    void ext(const std::array<T, 3> &v1, const std::array<std::complex<T>, 3> &cv2, std::array<std::complex<T>, 3> &out);

    // Overloaded absolute value
    void abs(const std::array<T, 3> &v, T &out);
    void abs(const std::array<std::complex<T>, 3> &cv, std::complex<T> &out);

    // Difference vectors
    void diff(const std::array<T, 3> &v1, const std::array<T, 3> &v2, std::array<T, 3> &out);
    void diff(const std::array<std::complex<T>, 3> &cv1, const std::array<std::complex<T>, 3> &cv2, std::array<std::complex<T>, 3> &out);

    // Normalization
    void normalize(const std::array<T, 3> &v, std::array<T, 3> &out);
    void normalize(const std::array<std::complex<T>, 3> &cv, std::array<std::complex<T>, 3> &out);

    // Scalar multiplication
    void s_mult(const std::array<T, 3> &v, const T &s, std::array<T, 3> &out);
    void s_mult(const std::array<std::complex<T>, 3> &cv, const std::complex<T> &cs, std::array<std::complex<T>, 3> &out);
    void s_mult(const std::array<T, 3> &v, const std::complex<T> &cs, std::array<std::complex<T>, 3> &out);
    void s_mult(const std::array<std::complex<T>, 3> &cv, const T &s, std::array<std::complex<T>, 3> &out);

    // Conjugation of complex vector
    void conj(const std::array<std::complex<T>, 3> &cv, std::array<std::complex<T>, 3> &out);

    // Snell's function
    void snell(const std::array<std::complex<T>, 3> &cvin, const std::array<T, 3> &normal, std::array<std::complex<T>, 3> &out);
    void snell(const std::array<T, 3> &vin, const std::array<T, 3> &normal, std::array<T, 3> &out);
    void snell_t(const std::array<T, 3> &vin, const std::array<T, 3> &normal, T mu, std::array<T, 3> &out);

    // Dyadic products
    void dyad(const std::array<T, 3> &v1, const std::array<T, 3> &v2, std::array<std::array<T, 3>, 3> &out);

    // Real valued matrix-matrix subtraction
    void matDiff(const std::array<std::array<T, 3>, 3> &m1, const std::array<std::array<T, 3>, 3> &m2, std::array<std::array<T, 3>, 3> &out);

    // Matrix-vector multiplication
    void matVec(const std::array<std::array<T, 3>, 3> &m1, const std::array<T, 3> &v1, std::array<T, 3> &out);
    void matVec(const std::array<std::array<T, 3>, 3> &m1, const std::array<std::complex<T>, 3> &cv1, std::array<std::complex<T>, 3> &out);

    // 4D-only need real-real method
    void matVec4(const T *m1, const std::array<T, 3> &v1, std::array<T, 3> &out, bool vec=false);

    void invmatVec4(const T *m1, const std::array<T, 3> &v1, std::array<T, 3> &out, bool vec=false);
    void matRot(const std::array<T, 3> &rot, const std::array<T, 3> &v1, const std::array<T, 3> &cRot, std::array<T, 3> &out);
};

/**
 * Dot product.
 *
 * Take the dot (inner) product of two real valued arrays of size 3.
 * 
 * @param v1 Array of 3 double/float.
 * @param v2 Array of 3 double/float.
 * @param out Scalar double/float.
 */
template <typename T> inline
void Utils<T>::dot(const std::array<T, 3> &v1, const std::array<T, 3> &v2, T &out)
{
    out = 0;

    for(int n=0; n<3; n++)
    {
        out += v1[n] * v2[n];
    }
}

/**
 * Dot product.
 *
 * Take the dot (inner) product of two complex valued double/float arrays of size 3.
 * 
 * @param cv1 Array of 3 complex double/float.
 * @param cv2 Array of 3 complex double/float.
 * @param out Scalar complex double/float.
 */
template <typename T> inline
void Utils<T>::dot(const std::array<std::complex<T>, 3> &cv1, const std::array<std::complex<T>, 3> &cv2, std::complex<T> &out)
{
    out = (0, 0);

    for(int n=0; n<3; n++)
    {
        out += std::conj(cv1[n]) * cv2[n];
    }
}

/**
 * Dot product.
 *
 * Take the dot (inner) product of one complex valued and one real valued double/float array of size 3.
 * 
 * @param cv1 Array of 3 complex double/float.
 * @param v2 Array of 3 double/float.
 * @param out Scalar complex double/float.
 */
template <typename T> inline
void Utils<T>::dot(const std::array<std::complex<T>, 3> &cv1, const std::array<T, 3> &v2, std::complex<T> &out)
{
    out = (0, 0);

    for(int n=0; n<3; n++)
    {
        out += std::conj(cv1[n]) * v2[n];
    }
}

/**
 * Dot product.
 *
 * Take the dot (inner) product of one real valued and one complex valued double/float array of size 3.
 * 
 * @param v1 Array of 3 double/float.
 * @param cv2 Array of 3 complex double/float.
 * @param out Scalar complex double/float.
 */
template <typename T> inline
void Utils<T>::dot(const std::array<T, 3> &v1, const std::array<std::complex<T>, 3> &cv2, std::complex<T> &out)
{
    out = (0, 0);

    for(int n=0; n<3; n++)
    {
        out += v1[n] * cv2[n];
    }
}

/**
 * Cross product.
 *
 * Take the cross (outer) product of two real valued double/float arrays of size 3.
 * 
 * @param v1 Array of 3 double/float.
 * @param v2 Array of 3 double/float.
 * @param out Array of 3 double/float.
 */
template <typename T> inline
void Utils<T>::ext(const std::array<T, 3> &v1, const std::array<T, 3> &v2, std::array<T, 3> &out)
{
    out[0] = v1[1]*v2[2] - v1[2]*v2[1];
    out[1] = v1[2]*v2[0] - v1[0]*v2[2];
    out[2] = v1[0]*v2[1] - v1[1]*v2[0];
}

/**
 * Cross product.
 *
 * Take the cross (outer) product of two complex valued double/float arrays of size 3.
 * 
 * @param cv1 Array of 3 complex double/float.
 * @param cv2 Array of 3 complex double/float.
 * @param out Array of 3 complex double/float.
 */
template <typename T> inline
void Utils<T>::ext(const std::array<std::complex<T>, 3> &cv1, const std::array<std::complex<T>, 3> &cv2, std::array<std::complex<T>, 3> &out)
{
    out[0] = cv1[1]*cv2[2] - cv1[2]*cv2[1];
    out[1] = cv1[2]*cv2[0] - cv1[0]*cv2[2];
    out[2] = cv1[0]*cv2[1] - cv1[1]*cv2[0];
}

/**
 * Cross product.
 *
 * Take the cross (outer) product of one complex valued and one real valued double/float array of size 3.
 * 
 * @param cv1 Array of 3 complex double/float.
 * @param v2 Array of 3 double/float.
 * @param out Array of 3 complex double/float.
 */
template <typename T> inline
void Utils<T>::ext(const std::array<std::complex<T>, 3> &cv1, const std::array<T, 3> &v2, std::array<std::complex<T>, 3> &out)
{
    out[0] = cv1[1]*v2[2] - cv1[2]*v2[1];
    out[1] = cv1[2]*v2[0] - cv1[0]*v2[2];
    out[2] = cv1[0]*v2[1] - cv1[1]*v2[0];
}

/**
 * Cross product.
 *
 * Take the cross (outer) product of one real valued and one complex valued double/float array of size 3.
 * 
 * @param v1 Array of 3 double/float.
 * @param cv2 Array of 3 complex double/float.
 * @param out Array of 3 complex double/float.
 */
template <typename T> inline
void Utils<T>::ext(const std::array<T, 3> &v1, const std::array<std::complex<T>, 3> &cv2, std::array<std::complex<T>, 3> &out)
{
    out[0] = v1[1]*cv2[2] - v1[2]*cv2[1];
    out[1] = v1[2]*cv2[0] - v1[0]*cv2[2];
    out[2] = v1[0]*cv2[1] - v1[1]*cv2[0];
}

/**
 * Component-wise vector difference.
 *
 * Subtract two real valued vectors of size 3, element-wise.
 * 
 * @param v1 Array of 3 double/float.
 * @param v2 Array of 3 double/float.
 * @param out Array of 3 double/float.
 */
template <typename T> inline
void Utils<T>::diff(const std::array<T, 3> &v1, const std::array<T, 3> &v2, std::array<T, 3> &out)
{
    for(int n=0; n<3; n++)
    {
        out[n] = v1[n] - v2[n];
    }
}

/**
 * Component-wise vector difference.
 *
 * Subtract two complex valued vectors of size 3, element-wise.
 * 
 * @param cv1 Array of 3 complex double/float.
 * @param cv2 Array of 3 complex double/float.
 * @param out Array of 3 complex double/float.
 */
template <typename T> inline
void Utils<T>::diff(const std::array<std::complex<T>, 3> &cv1, const std::array<std::complex<T>, 3> &cv2, std::array<std::complex<T>, 3> &out)
{
    for(int n=0; n<3; n++)
    {
        out[n] = cv1[n] - cv2[n];
    }
}

/**
 * Absolute value.
 *
 * Calculate absolute value of real valued vector of size 3.
 * 
 * @param v Array of 3 double/float.
 * @param out Scalar double/float.
 */
template <typename T> inline
void Utils<T>::abs(const std::array<T, 3> &v, T &out)
{
    dot(v, v, out);
    out = std::sqrt(out);
}

/**
 * Absolute value.
 *
 * Calculate absolute value of complex valued vector of size 3.
 * 
 * @param cv Array of 3 complex double/float.
 * @param out Scalar complex double/float.
 */
template <typename T> inline
void Utils<T>::abs(const std::array<std::complex<T>, 3> &cv, std::complex<T> &out)
{
    std::array<std::complex<T>, 3> cv_conj;
    conj(cv, cv_conj);
    dot(cv, cv_conj, out);
    out = std::sqrt(out);
}

/**
 * Normalize vector.
 *
 * Normalize real valued vector of size 3.
 * 
 * @param v Array of 3 double/float.
 * @param out Array of 3 double/float.
 */
template <typename T> inline
void Utils<T>::normalize(const std::array<T, 3> &v, std::array<T, 3> &out)
{
    T norm;
    abs(v, norm);

    if (norm <= std::numeric_limits<T>::denorm_min())
    {
        norm = std::numeric_limits<T>::denorm_min();
    }

    for( int n=0; n<3; n++)
    {
        out[n] = v[n] / norm;
    }
}

/**
 * Normalize vector.
 *
 * Normalize complex valued vector of size 3.
 * 
 * @param cv Array of 3 complex double/float.
 * @param out Array of 3 complex double/float.
 */
template <typename T> inline
void Utils<T>::normalize(const std::array<std::complex<T>, 3> &cv, std::array<std::complex<T>, 3> &out)
{
    std::complex<T> cnorm;
    abs(cv, cnorm);

    for( int n=0; n<3; n++)
    {
        out[n] = cv[n] / cnorm;
    }
}

/**
 * Scalar multiplication.
 *
 * Multiply real valued vector of size 3 by real scalar, element-wise.
 * 
 * @param v Array of 3 double/float.
 * @param s Scalar double/float.
 * @param out Array of 3 double/float.
 */
template <typename T> inline
void Utils<T>::s_mult(const std::array<T, 3> &v, const T &s, std::array<T, 3> &out)
{
    for(int n=0; n<3; n++)
    {
        out[n] = s * v[n];
    }
}

/**
 * Scalar multiplication.
 *
 * Multiply complex valued vector of size 3 by complex scalar, element-wise.
 * 
 * @param cv Array of 3 complex double/float.
 * @param cs Scalar complex double/float.
 * @param out Array of 3 complex double/float.
 */
template <typename T> inline
void Utils<T>::s_mult(const std::array<std::complex<T>, 3> &cv, const std::complex<T> &cs, std::array<std::complex<T>, 3> &out)
{
    for(int n=0; n<3; n++)
    {
        out[n] = cs * cv[n];
    }
}

/**
 * Scalar multiplication.
 *
 * Multiply real valued vector of size 3 by complex scalar, element-wise.
 * 
 * @param v Array of 3 double/float.
 * @param cs Scalar complex double/float.
 * @param out Array of 3 complex double/float.
 */
template <typename T> inline
void Utils<T>::s_mult(const std::array<T, 3> &v, const std::complex<T> &cs, std::array<std::complex<T>, 3> &out)
{
    for(int n=0; n<3; n++)
    {
        out[n] = cs * v[n];
    }
}

/**
 * Scalar multiplication.
 *
 * Multiply complex valued vector of size 3 by real scalar, element-wise.
 * 
 * @param cv Array of 3 complex double/float.
 * @param s Scalar double/float.
 * @param out Array of 3 complex double/float.
 */
template <typename T> inline
void Utils<T>::s_mult(const std::array<std::complex<T>, 3> &cv, const T &s, std::array<std::complex<T>, 3> &out)
{
    for(int n=0; n<3; n++)
    {
        out[n] = s * cv[n];
    }
}

/**
 * Conjugate.
 *
 * Conjugate complex valued vector of size 3.
 * 
 * @param cv Array of 3 complex double/float.
 * @param out Array of 3 complex double/float.
 */
template <typename T> inline
void Utils<T>::conj(const std::array<std::complex<T>, 3> &cv, std::array<std::complex<T>, 3> &out)
{
    for(int n=0; n<3; n++)
    {
        out[n] = std::conj(cv[n]);
    }
}

/**
 * Snell's law.
 *
 * Calculate reflected direction vector from incoming direction and normal vector.
 * 
 * @param cvin Array of 3 complex double/float, incoming direction vector.
 * @param normal Array of 3 double/float, normal vector of surface.
 * @param out Array of 3 complex double/float.
 */
template <typename T> inline
void Utils<T>::snell(const std::array<std::complex<T>, 3> &cvin, const std::array<T, 3> &normal, std::array<std::complex<T>, 3> &out)
{
    std::complex<T> cfactor;
    dot(cvin, normal, cfactor);

    cfactor = 2. * cfactor;

    std::array<std::complex<T>, 3> rhs;
    s_mult(normal, cfactor, rhs);

    diff(cvin, rhs, out);
}

/**
 * Snell's law for reflection.
 *
 * Calculate reflected direction vector from incoming direction and normal vector.
 * 
 * @param vin Array of 3 double/float, incoming direction vector.
 * @param normal Array of 3 double/float, normal vector of surface.
 * @param out Array of 3 double/float.
 */
template <typename T> inline
void Utils<T>::snell(const std::array<T, 3> &vin, const std::array<T, 3> &normal, std::array<T, 3> &out)
{
    T factor;
    dot(vin, normal, factor);

    factor = 2. * factor;

    std::array<T, 3> rhs;
    s_mult(normal, factor, rhs);

    diff(vin, rhs, out);
}

/**
 * Snell's law refraction.
 *
 * Calculate refracted direction vector from incoming direction and normal vector.
 * 
 * @param vin Array of 3 double/float, incoming direction vector.
 * @param normal Array of 3 double/float, normal vector of surface.
 * @param mu Ratio of n1 to n2.
 * @param out Array of 3 double/float.
 */
template <typename T> inline
void Utils<T>::snell_t(const std::array<T, 3> &vin, const std::array<T, 3> &normal, T mu, std::array<T, 3> &out)
{
    T in_dot_n, factor1;
    std::array<T, 3> term1, term2, temp1, temp2;

    dot(vin, normal, in_dot_n);
    
    factor1 = sqrt(1 - mu*mu * (1 - in_dot_n*in_dot_n));
    s_mult(normal, factor1, term1);

    s_mult(normal, in_dot_n, temp1);
    diff(vin, temp1, temp2);
    s_mult(temp2, mu, term2);

    diff(term1, term2, out);
}

/**
 * Dyadic product.
 *
 * Calculate dyadic product between two real valued double/float vectors of size 3.
 * 
 * @param v1 Array of 3 double/float.
 * @param v2 Array of 3 double/float.
 * @param out Array of 3 double/float, nested inside array of size 3.
 */
template <typename T> inline
void Utils<T>::dyad(const std::array<T, 3> &v1, const std::array<T, 3> &v2, std::array<std::array<T, 3>, 3> &out)
{
    for(int n=0; n<3; n++)
    {
        out[n][0] = v1[n] * v2[0];
        out[n][1] = v1[n] * v2[1];
        out[n][2] = v1[n] * v2[2];
    }
}

/**
 * Matrix difference, element wise.
 *
 * Subtract two 3x3 matrices, element wise.
 * 
 * @param m1 Array of 3 double/float, nested inside array of size 3.
 * @param m2 Array of 3 double/float, nested inside array of size 3.
 * @param out Array of 3 double/float, nested inside array of size 3.
 */
template <typename T> inline
void Utils<T>::matDiff(const std::array<std::array<T, 3>, 3> &m1, const std::array<std::array<T, 3>, 3> &m2, std::array<std::array<T, 3>, 3> &out)
{
    for(int n=0; n<3; n++)
    {
        out[n][0] = m1[n][0] - m2[n][0];
        out[n][1] = m1[n][1] - m2[n][1];
        out[n][2] = m1[n][2] - m2[n][2];
    }
}

/**
 * Matrix-vector product.
 *
 * Multiply a real valued 3x3 matrix and a real valued size 3 vector to generate a new real valued size 3 vector.
 * 
 * @param m1 Array of 3 double/float, nested inside array of size 3.
 * @param v1 Array of 3 double/float.
 * @param out Array of 3 double/float.
 */
template <typename T> inline
void Utils<T>::matVec(const std::array<std::array<T, 3>, 3> &m1, const std::array<T, 3> &v1, std::array<T, 3> &out)
{
    for(int n=0; n<3; n++)
    {
        out[n] = m1[n][0] * v1[0] + m1[n][1] * v1[1] + m1[n][2] * v1[2];
    }
}

/**
 * Matrix-vector product.
 *
 * Multiply a real valued 3x3 matrix and a complex valued size 3 vector to generate a new complex valued size 3 vector.
 * 
 * @param m1 Array of 3 double/float, nested inside array of size 3.
 * @param cv1 Array of 3 complex double/float.
 * @param out Array of 3 complex double/float.
 */
template <typename T> inline
void Utils<T>::matVec(const std::array<std::array<T, 3>, 3> &m1, const std::array<std::complex<T>, 3> &cv1, std::array<std::complex<T>, 3> &out)
{
    for(int n=0; n<3; n++)
    {
        out[n] = m1[n][0] * cv1[0] + m1[n][1] * cv1[1] + m1[n][2] * cv1[2];
    }
}

/**
 * Matrix-vector product.
 *
 * Multiply a real valued 4x4 matrix and a real valued size 3 vector to generate a new real valued size 3 vector.
 *      This function is only used for multiplying 3D vectors by a 4D transformation matrix.
 * 
 * @param m1 Array of 4 double/float, nested inside array of size 4.
 * @param v1 Array of 3 double/float.
 * @param out Array of 3 double/float.
 * @param vec Whether or not to transform v1 as a point or a vector.
 */
template <typename T> inline
void Utils<T>::matVec4(const T *m1, const std::array<T, 3> &v1, std::array<T, 3> &out, bool vec)
{
    if (vec)
    {
        for(int n=0; n<3; n++)
        {
            out[n] = m1[n*4] * v1[0] + m1[1+n*4] * v1[1] + m1[2+n*4] * v1[2];
        }
    }

    else
    {
        for(int n=0; n<3; n++)
        {
            out[n] = m1[n*4] * v1[0] + m1[1+n*4] * v1[1] + m1[2+n*4] * v1[2] + m1[3+n*4];
        }
    }
}

/**
 * Inverse matrix-vector product.
 *
 * Multiply the inverse of a real valued 4x4 matrix by a real valued size 3 vector to generate a new size 3 vector.
 *      This function is only used for multiplying 3D vectors by a 4D transformation matrix.
 * 
 * @param m1 Array of 4 double/float, nested inside array of size 4.
 * @param v1 Array of 3 double/float.
 * @param out Array of 3 double/float.
 * @param vec Whether or not to transform v1 as a point or a vector.
 */
template <typename T> inline
void Utils<T>::invmatVec4(const T *m1, const std::array<T, 3> &v1, std::array<T, 3> &out, bool vec)
{
    if (vec)
    {
        for(int n=0; n<3; n++)
        {
            out[n] = m1[n] * v1[0] + m1[n+4] * v1[1] + m1[n+8] * v1[2];
        }
    }

    else
    {
        T temp;
        for(int n=0; n<3; n++)
        {
            temp = -m1[n]*m1[3] - m1[n+4]*m1[7] - m1[n+8]*m1[11];
            out[n] = m1[n] * v1[0] + m1[n+4] * v1[1] + m1[n+8] * v1[2] + temp;
        }
    }
}

/**
 * Manual vector rotation.
 *  
 * Rotate a real valued size 3 vector without defining a transformation matrix.
 * 
 * @param rot Array of 3 double/float, containing rotation angles.
 * @param v1 Array of 3 double/float.
 * @param cRot Array of 3, center of rotation.
 * @param out Array of 3 double/float.
 */
template <typename T> inline
void Utils<T>::matRot(const std::array<T, 3> &rot, const std::array<T, 3> &v1, const std::array<T, 3> &cRot, std::array<T, 3> &out)
{
    T cosx = cos(rot[0]);
    T cosy = cos(rot[1]);
    T cosz = cos(rot[2]);

    T sinx = sin(rot[0]);
    T siny = sin(rot[1]);
    T sinz = sin(rot[2]);

    T mu = cosz * siny;
    T rho = sinz * siny;

    std::array<T, 3> to_rot;

    for(int n=0; n<3; n++) {to_rot[n] = v1[n] - cRot[n];}

    out[0] = cosz*cosy * to_rot[0] + (mu*sinx - sinz*cosx) * to_rot[1] + (mu*cosx + sinz*sinx) * to_rot[2];
    out[1] = sinz*cosy * to_rot[0] + (rho*sinx + cosz*cosx) * to_rot[1] + (rho*cosx - cosz*sinx) * to_rot[2];
    out[2] = -siny * to_rot[0] + cosy*sinx * to_rot[1] + cosy*cosx * to_rot[2];

    for(int n=0; n<3; n++) {out[n] = out[n] + cRot[n];}
}

#endif
