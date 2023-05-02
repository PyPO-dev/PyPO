#include <math.h>
#include <cuda.h>
#include <cuComplex.h>

/*! \file GUtils.h
    \brief Linear algebra functions for the CUDA version of PyPO. 
    
    Contains float overloaded functions for doing basic 3D vector operations.
        For the CUDA complex valued linear algebra, we employ the cuComplex.h library.
*/

/**
 * Dot product.
 *
 * Take the dot (inner) product of two real valued arrays of size 3.
 * 
 * @param v1 Array of 3 float.
 * @param v2 Array of 3 float.
 * @param out Scalar float.
 */
__device__ __inline__ void dot(float (&v1)[3], float (&v2)[3], float &out)
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
 * Take the dot (inner) product of two complex valued float arrays of size 3.
 * 
 * @param cv1 Array of 3 complex float.
 * @param cv2 Array of 3 complex float.
 * @param out Scalar complex float.
 */
__device__ __inline__ void dot(cuFloatComplex (&cv1)[3], cuFloatComplex (&cv2)[3], cuFloatComplex &out)
{
    out = make_cuFloatComplex(0, 0);

    for(int n=0; n<3; n++)
    {
        out = cuCaddf(cuCmulf(cuConjf(cv1[n]), cv2[n]), out);
    }
}

/**
 * Dot product.
 *
 * Take the dot (inner) product of one complex valued and one real valued float array of size 3.
 * 
 * @param cv1 Array of 3 complex float.
 * @param v2 Array of 3 float.
 * @param out Scalar complex float.
 */
__device__ __inline__ void dot(cuFloatComplex (&cv1)[3], float (&v2)[3], cuFloatComplex &out)
{
    out = make_cuFloatComplex(0, 0);

    for(int n=0; n<3; n++)
    {
        out = cuCaddf(cuCmulf(cuConjf(cv1[n]), make_cuFloatComplex(v2[n], 0)), out);
    }
}

/**
 * Dot product.
 *
 * Take the dot (inner) product of one real valued and one complex valued float array of size 3.
 * 
 * @param v1 Array of 3 float.
 * @param cv2 Array of 3 complex float.
 * @param out Scalar complex float.
 */
__device__ __inline__ void dot(float (&v1)[3], cuFloatComplex (&cv2)[3], cuFloatComplex &out)
{
    out = make_cuFloatComplex(0, 0);

    for(int n=0; n<3; n++)
    {
        out = cuCaddf(cuCmulf(make_cuFloatComplex(v1[n], 0), cv2[n]), out);
    }
}

/**
 * Cross product.
 *
 * Take the cross (outer) product of two real valued float arrays of size 3.
 * 
 * @param v1 Array of 3 float.
 * @param v2 Array of 3 float.
 * @param out Array of 3 float.
 */
__device__ __inline__ void ext(float (&v1)[3], float (&v2)[3], float (&out)[3])
{
    out[0] = v1[1]*v2[2] - v1[2]*v2[1];
    out[1] = v1[2]*v2[0] - v1[0]*v2[2];
    out[2] = v1[0]*v2[1] - v1[1]*v2[0];
}


/**
 * Cross product.
 *
 * Take the cross (outer) product of two complex valued float arrays of size 3.
 * 
 * @param cv1 Array of 3 complex float.
 * @param cv2 Array of 3 complex float.
 * @param out Array of 3 complex float.
 */
__device__ __inline__ void ext(cuFloatComplex (&cv1)[3], cuFloatComplex (&cv2)[3], cuFloatComplex (&out)[3])
{
    out[0] = cuCsubf(cuCmulf(cv1[1], cv2[2]), cuCmulf(cv1[2], cv2[1]));
    out[1] = cuCsubf(cuCmulf(cv1[2], cv2[0]), cuCmulf(cv1[0], cv2[2]));
    out[2] = cuCsubf(cuCmulf(cv1[0], cv2[1]), cuCmulf(cv1[1], cv2[0]));
}

/**
 * Cross product.
 *
 * Take the cross (outer) product of one complex valued and one real valued float array of size 3.
 * 
 * @param cv1 Array of 3 complex float.
 * @param v2 Array of 3 float.
 * @param out Array of 3 complex float.
 */
__device__ __inline__ void ext(cuFloatComplex (&cv1)[3], float (&v2)[3], cuFloatComplex (&out)[3])
{
    out[0] = cuCsubf(cuCmulf(cv1[1], make_cuFloatComplex(v2[2],0)), cuCmulf(cv1[2], make_cuFloatComplex(v2[1],0)));
    out[1] = cuCsubf(cuCmulf(cv1[2], make_cuFloatComplex(v2[0],0)), cuCmulf(cv1[0], make_cuFloatComplex(v2[2],0)));
    out[2] = cuCsubf(cuCmulf(cv1[0], make_cuFloatComplex(v2[1],0)), cuCmulf(cv1[1], make_cuFloatComplex(v2[0],0)));
}

/**
 * Cross product.
 *
 * Take the cross (outer) product of one real valued and one complex valued float array of size 3.
 * 
 * @param v1 Array of 3 float.
 * @param cv2 Array of 3 complex float.
 * @param out Array of 3 complex float.
 */
__device__ __inline__ void ext(float (&v1)[3], cuFloatComplex (&cv2)[3], cuFloatComplex (&out)[3])
{
    out[0] = cuCsubf(cuCmulf(make_cuFloatComplex(v1[1],0), cv2[2]), cuCmulf(make_cuFloatComplex(v1[2],0), cv2[1]));
    out[1] = cuCsubf(cuCmulf(make_cuFloatComplex(v1[2],0), cv2[0]), cuCmulf(make_cuFloatComplex(v1[0],0), cv2[2]));
    out[2] = cuCsubf(cuCmulf(make_cuFloatComplex(v1[0],0), cv2[1]), cuCmulf(make_cuFloatComplex(v1[1],0), cv2[0]));
}

/**
 * Component-wise vector difference.
 *
 * Subtract two real valued vectors of size 3, element-wise.
 * 
 * @param v1 Array of 3 float.
 * @param v2 Array of 3 float.
 * @param out Array of 3 float.
 */
__device__ __inline__ void diff(float (&v1)[3], float (&v2)[3], float (&out)[3])
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
 * @param cv1 Array of 3 complex float.
 * @param cv2 Array of 3 complex float.
 * @param out Array of 3 complex float.
 */
__device__ __inline__ void diff(cuFloatComplex (&cv1)[3], cuFloatComplex (&cv2)[3], cuFloatComplex (&out)[3])
{
    for(int n=0; n<3; n++)
    {
        out[n] = cuCsubf(cv1[n], cv2[n]);
    }
}

/**
 * Absolute value.
 *
 * Calculate absolute value of real valued vector of size 3.
 * 
 * @param v Array of 3 float.
 * @param out Scalar float.
 */
__device__ __inline__ void abs(float (&v)[3], float &out)
{
    dot(v, v, out);
    out = sqrt(out);
}

/**
 * Conjugate.
 *
 * Conjugate complex valued vector of size 3.
 * 
 * @param cv Array of 3 complex float.
 * @param out Array of 3 complex float.
 */
__device__ __inline__ void conja(cuFloatComplex (&cv)[3], cuFloatComplex (&out)[3])
{
    for(int n=0; n<3; n++)
    {
        out[n] = cuConjf(cv[n]);
    }
}

/**
 * Absolute value.
 *
 * Calculate absolute value of complex valued vector of size 3.
 * 
 * @param cv Array of 3 complex float.
 * @param out Scalar complex float.
 */
__device__ __inline__ void abs(cuFloatComplex (&cv)[3], cuFloatComplex &out)
{
    cuFloatComplex cv_conj[3];
    conja(cv, cv_conj);
    dot(cv, cv_conj, out);
    out = make_cuFloatComplex(cuCabsf(out), 0);
}

/**
 * Normalize vector.
 *
 * Normalize real valued vector of size 3.
 * 
 * @param v Array of 3 float.
 * @param out Array of 3 float.
 */
__device__ __inline__ void normalize(float (&v)[3], float (&out)[3])
{
    float norm;
    abs(v, norm);


    if (norm == 0)
    {
        norm = 1;
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
 * @param cv Array of 3 complex float.
 * @param out Array of 3 complex float.
 */
__device__ __inline__ void normalize(cuFloatComplex (&cv)[3], cuFloatComplex (&out)[3])
{
    cuFloatComplex cnorm;
    abs(cv, cnorm);

    for( int n=0; n<3; n++)
    {
        out[n] = cuCdivf(cv[n], cnorm);
    }
}

/**
 * Scalar multiplication.
 *
 * Multiply real valued vector of size 3 by real scalar, element-wise.
 * 
 * @param v Array of 3 float.
 * @param s Scalar float.
 * @param out Array of 3 float.
 */
__device__ __inline__ void s_mult(float (&v)[3], float &s, float (&out)[3])
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
 * @param cv Array of 3 complex float.
 * @param cs Scalar complex float.
 * @param out Array of 3 complex float.
 */
__device__ __inline__ void s_mult(cuFloatComplex (&cv)[3], cuFloatComplex &cs, cuFloatComplex (&out)[3])
{
    for(int n=0; n<3; n++)
    {
        out[n] = cuCmulf(cs, cv[n]);
    }
}

/**
 * Scalar multiplication.
 *
 * Multiply real valued vector of size 3 by complex scalar, element-wise.
 * 
 * @param v Array of 3 float.
 * @param cs Scalar complex float.
 * @param out Array of 3 complex float.
 */
__device__ __inline__ void s_mult(float (&v)[3], cuFloatComplex &cs, cuFloatComplex (&out)[3])
{
    for(int n=0; n<3; n++)
    {
        out[n] = cuCmulf(cs, make_cuFloatComplex(v[n],0));
    }
}

/**
 * Scalar multiplication.
 *
 * Multiply complex valued vector of size 3 by real scalar, element-wise.
 * 
 * @param cv Array of 3 complex float.
 * @param s Scalar float.
 * @param out Array of 3 complex float.
 */
__device__ __inline__ void s_mult(cuFloatComplex (&cv)[3], const float &s, cuFloatComplex (&out)[3])
{
    for(int n=0; n<3; n++)
    {
        out[n] = cuCmulf(make_cuFloatComplex(s,0), cv[n]);
    }
}

/**
 * Snell's law reflection.
 *
 * Calculate reflected direction vector from incoming direction and normal vector.
 * 
 * @param cvin Array of 3 complex float, incoming direction vector.
 * @param normal Array of 3 float, normal vector of surface.
 * @param out Array of 3 complex float.
 */
__device__ __inline__ void snell(cuFloatComplex (&cvin)[3], float (&normal)[3], cuFloatComplex (&out)[3])
{
    cuFloatComplex cfactor;
    dot(cvin, normal, cfactor);

    cfactor = cuCmulf(make_cuFloatComplex(2.,0), cfactor);

    cuFloatComplex rhs[3];
    s_mult(normal, cfactor, rhs);

    diff(cvin, rhs, out);
}

/**
 * Snell's law reflection.
 *
 * Calculate reflected direction vector from incoming direction and normal vector.
 * 
 * @param vin Array of 3 float, incoming direction vector.
 * @param normal Array of 3 float, normal vector of surface.
 * @param out Array of 3 float.
 */
__device__ __inline__ void snell(float (&vin)[3], float (&normal)[3], float (&out)[3])
{
    float factor;
    dot(vin, normal, factor);

    factor = 2. * factor;

    float rhs[3];
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
__device__ __inline__ void snell(float (&vin)[3], float (&normal)[3], float mu, float (&out)[3])
{
    float in_dot_n, factor1;
    float term1[3], term2[3], temp1[3], temp2[3];

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
 * Calculate dyadic product between two real valued float vectors of size 3.
 * 
 * @param v1 Array of 3 float.
 * @param v2 Array of 3 float.
 * @param out Array of 3 float, nested inside array of size 3.
 */
__device__ __inline__ void dyad(float (&v1)[3], float (&v2)[3], float (&out)[3][3])
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
 * @param m1 Array of 3 float, nested inside array of size 3.
 * @param m2 Array of 3 float, nested inside array of size 3.
 * @param out Array of 3 float, nested inside array of size 3.
 */
__device__ __inline__ void matDiff(float (&m1)[3][3], float (&m2)[3][3], float (&out)[3][3])
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
 * @param m1 Array of 3 float, nested inside array of size 3.
 * @param v1 Array of 3 float.
 * @param out Array of 3 float.
 */
__device__ __inline__ void matVec(float (&m1)[3][3], float (&v1)[3], float (&out)[3])
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
 * @param m1 Array of 3 float, nested inside array of size 3.
 * @param cv1 Array of 3 complex float.
 * @param out Array of 3 complex float.
 */
__device__ __inline__ void matVec(float (&m1)[3][3], cuFloatComplex (&cv1)[3], cuFloatComplex (&out)[3])
{
    for(int n=0; n<3; n++)
    {
        out[n] = cuCaddf(cuCmulf(make_cuFloatComplex(m1[n][0],0), cv1[0]),
                cuCaddf(cuCmulf(make_cuFloatComplex(m1[n][1],0), cv1[1]),
                cuCmulf(make_cuFloatComplex(m1[n][2],0), cv1[2])));
    }
}

/**
 * Take complex exponential.
 *
 * Take complex exponential by decomposing into sine and cosine.
 *
 * @return res cuFloatComplex number.
 */
__device__ __inline__ cuFloatComplex expCo(cuFloatComplex z)
{
    cuFloatComplex res;
    float t = exp(z.x);
    float ys = sin(z.y);
    float yc = cos(z.y);
    res.x = t*yc;
    res.y = t*ys;

    return res;
}
