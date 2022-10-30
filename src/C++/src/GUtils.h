#include <math.h>
#include <cuda.h>
#include <cuComplex.h>

// The Utils file contains mostly linear algebra functions as CUDA device functions
// File contains overloads for doubles and floats
// In GUtils, we use CUDA complex library for arithmetic.

// Real dot-product
__device__ __inline__ void dot(double (&v1)[3], double (&v2)[3], double &out)
{
    out = 0;

    for(int n=0; n<3; n++)
    {
        out += v1[n] * v2[n];
    }
}


// Complex hermitian conjugate inner-product
__device__ __inline__ void dot(cuDoubleComplex (&cv1)[3], cuDoubleComplex (&cv2)[3], cuDoubleComplex &out)
{
    out = make_cuDoubleComplex(0, 0);

    for(int n=0; n<3; n++)
    {
        out = cuCadd(cuCmul(cuConj(cv1[n]), cv2[n]), out);
    }
}

// Complex vector - real vector dot-product
__device__ __inline__ void dot(cuDoubleComplex (&cv1)[3], double (&v2)[3], cuDoubleComplex &out)
{
    out = make_cuDoubleComplex(0, 0);

    for(int n=0; n<3; n++)
    {
        out = cuCadd(cuCmul(cuConj(cv1[n]), make_cuDoubleComplex(v2[n], 0)), out);
    }
}

// Real vector - complex vector dot-product
__device__ __inline__ void dot(double (&v1)[3], cuDoubleComplex (&cv2)[3], cuDoubleComplex &out)
{
    out = make_cuDoubleComplex(0, 0);

    for(int n=0; n<3; n++)
    {
        out = cuCadd(cuCmul(make_cuDoubleComplex(v1[n], 0), cv2[n]), out);
    }
}

// Real cross-product
__device__ __inline__ void ext(double (&v1)[3], double (&v2)[3], double (&out)[3])
{
    out[0] = v1[1]*v2[2] - v1[2]*v2[1];
    out[1] = v1[2]*v2[0] - v1[0]*v2[2];
    out[2] = v1[0]*v2[1] - v1[1]*v2[0];
}


// Complex conjugate of cross product
__device__ __inline__ void ext(cuDoubleComplex (&cv1)[3], cuDoubleComplex (&cv2)[3], cuDoubleComplex (&out)[3])
{
    out[0] = cuCsub(cuCmul(cv1[1], cv2[2]), cuCmul(cv1[2], cv2[1]));
    out[1] = cuCsub(cuCmul(cv1[2], cv2[0]), cuCmul(cv1[0], cv2[2]));
    out[2] = cuCsub(cuCmul(cv1[0], cv2[1]), cuCmul(cv1[1], cv2[0]));
}

// Cross product between an complex and a real vector
__device__ __inline__ void ext(cuDoubleComplex (&cv1)[3], double (&v2)[3], cuDoubleComplex (&out)[3])
{
    out[0] = cuCsub(cuCmul(cv1[1], make_cuDoubleComplex(v2[2],0)), cuCmul(cv1[2], make_cuDoubleComplex(v2[1],0)));
    out[1] = cuCsub(cuCmul(cv1[2], make_cuDoubleComplex(v2[0],0)), cuCmul(cv1[0], make_cuDoubleComplex(v2[2],0)));
    out[2] = cuCsub(cuCmul(cv1[0], make_cuDoubleComplex(v2[1],0)), cuCmul(cv1[1], make_cuDoubleComplex(v2[0],0)));
}

// Cross product between a real vector and a complex vector
__device__ __inline__ void ext(double (&v1)[3], cuDoubleComplex (&cv2)[3], cuDoubleComplex (&out)[3])
{
    out[0] = cuCsub(cuCmul(make_cuDoubleComplex(v1[1],0), cv2[2]), cuCmul(make_cuDoubleComplex(v1[2],0), cv2[1]));
    out[1] = cuCsub(cuCmul(make_cuDoubleComplex(v1[2],0), cv2[0]), cuCmul(make_cuDoubleComplex(v1[0],0), cv2[2]));
    out[2] = cuCsub(cuCmul(make_cuDoubleComplex(v1[0],0), cv2[1]), cuCmul(make_cuDoubleComplex(v1[1],0), cv2[0]));
}

// Difference between two real vectors
__device__ __inline__ void diff(double (&v1)[3], double (&v2)[3], double (&out)[3])
{
    for(int n=0; n<3; n++)
    {
        out[n] = v1[n] - v2[n];
    }
}

// Difference between two complex valued vectors
__device__ __inline__ void diff(cuDoubleComplex (&cv1)[3], cuDoubleComplex (&cv2)[3], cuDoubleComplex (&out)[3])
{
    for(int n=0; n<3; n++)
    {
        out[n] = cuCsub(cv1[n], cv2[n]);
    }
}

// Absolute value of real vector
__device__ __inline__ void abs(double (&v)[3], double &out)
{
    dot(v, v, out);
    out = sqrt(out);
}
// Return complex conjugate of complex vector
__device__ __inline__ void conja(cuDoubleComplex (&cv)[3], cuDoubleComplex (&out)[3])
{
    for(int n=0; n<3; n++)
    {
        out[n] = cuConj(cv[n]);
    }
}
// Absolute value of a complex vector. Still returns a complex number!
__device__ __inline__ void abs(cuDoubleComplex (&cv)[3], cuDoubleComplex &out)
{
    cuDoubleComplex cv_conj[3];
    conja(cv, cv_conj);
    dot(cv, cv_conj, out);
    out = make_cuDoubleComplex(cuCabs(out), 0);
}

// Return normalized real vector from vector
__device__ __inline__ void normalize(double (&v)[3], double (&out)[3])
{
    double norm;
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

// Normalize complex vector
__device__ __inline__ void normalize(cuDoubleComplex (&cv)[3], cuDoubleComplex (&out)[3])
{
    cuDoubleComplex cnorm;
    abs(cv, cnorm);

    for( int n=0; n<3; n++)
    {
        out[n] = cuCdiv(cv[n], cnorm);
    }
}

// Apply standard real s-multiplication on a real vector
__device__ __inline__ void s_mult(double (&v)[3], double &s, double (&out)[3])
{
    for(int n=0; n<3; n++)
    {
        out[n] = s * v[n];
    }
}


// Multiply complex vector by complex scalar
__device__ __inline__ void s_mult(cuDoubleComplex (&cv)[3], cuDoubleComplex &cs, cuDoubleComplex (&out)[3])
{
    for(int n=0; n<3; n++)
    {
        out[n] = cuCmul(cs, cv[n]);
    }
}

// Multiply real vector by complex scalar
__device__ __inline__ void s_mult(double (&v)[3], cuDoubleComplex &cs, cuDoubleComplex (&out)[3])
{
    for(int n=0; n<3; n++)
    {
        out[n] = cuCmul(cs, make_cuDoubleComplex(v[n],0));
    }
}

// Multiply complex vector by real scalar
__device__ __inline__ void s_mult(cuDoubleComplex (&cv)[3], const double &s, cuDoubleComplex (&out)[3])
{
    for(int n=0; n<3; n++)
    {
        out[n] = cuCmul(make_cuDoubleComplex(s,0), cv[n]);
    }
}

// Calculate refected vector from surface using complex incoming vector and real normal vector to surface
__device__ __inline__ void snell(cuDoubleComplex (&cvin)[3], double (&normal)[3], cuDoubleComplex (&out)[3])
{
    cuDoubleComplex cfactor;
    dot(cvin, normal, cfactor);

    cfactor = cuCmul(make_cuDoubleComplex(2.,0), cfactor);

    cuDoubleComplex rhs[3];
    s_mult(normal, cfactor, rhs);

    diff(cvin, rhs, out);
}

// Calculate refected vector from surface using complex incoming vector and real normal vector to surface
__device__ __inline__ void snell(double (&vin)[3], double (&normal)[3], double (&out)[3])
{
    double factor;
    dot(vin, normal, factor);

    factor = 2. * factor;

    double rhs[3];
    s_mult(normal, factor, rhs);

    diff(vin, rhs, out);
}

// Calculate Dyadic product between two real vectors
// Returns array of length 3, containing 3 arrays representing ROWS in the resulting matrix
__device__ __inline__ void dyad(double (&v1)[3], double (&v2)[3], double (&out)[3][3])
{
    for(int n=0; n<3; n++)
    {
        out[n][0] = v1[n] * v2[0];
        out[n][1] = v1[n] * v2[1];
        out[n][2] = v1[n] * v2[2];
    }
}

// Subtract matrix from another matrix element-wise
__device__ __inline__ void matDiff(double (&m1)[3][3], double (&m2)[3][3], double (&out)[3][3])
{
    for(int n=0; n<3; n++)
    {
        out[n][0] = m1[n][0] - m2[n][0];
        out[n][1] = m1[n][1] - m2[n][1];
        out[n][2] = m1[n][2] - m2[n][2];
    }
}

// Multiply matrix with vector to return vector
__device__ __inline__ void matVec(double (&m1)[3][3], double (&v1)[3], double (&out)[3])
{
    for(int n=0; n<3; n++)
    {
        out[n] = m1[n][0] * v1[0] + m1[n][1] * v1[1] + m1[n][2] * v1[2];
    }
}

__device__ __inline__ void matVec(double (&m1)[3][3], cuDoubleComplex (&cv1)[3], cuDoubleComplex (&out)[3])
{
    for(int n=0; n<3; n++)
    {
        out[n] = cuCadd(cuCmul(make_cuDoubleComplex(m1[n][0],0), cv1[0]),
                cuCadd(cuCmul(make_cuDoubleComplex(m1[n][1],0), cv1[1]),
                cuCmul(make_cuDoubleComplex(m1[n][2],0), cv1[2])));
    }
}

// OVERLOADS FOR FLOAT

// Real dot-product
__device__ __inline__ void dot(float (&v1)[3], float (&v2)[3], float &out)
{
    out = 0;

    for(int n=0; n<3; n++)
    {
        out += v1[n] * v2[n];
    }
}


// Complex hermitian conjugate inner-product
__device__ __inline__ void dot(cuFloatComplex (&cv1)[3], cuFloatComplex (&cv2)[3], cuFloatComplex &out)
{
    out = make_cuFloatComplex(0, 0);

    for(int n=0; n<3; n++)
    {
        out = cuCaddf(cuCmulf(cuConjf(cv1[n]), cv2[n]), out);
    }
}

// Complex vector - real vector dot-product
__device__ __inline__ void dot(cuFloatComplex (&cv1)[3], float (&v2)[3], cuFloatComplex &out)
{
    out = make_cuFloatComplex(0, 0);

    for(int n=0; n<3; n++)
    {
        out = cuCaddf(cuCmulf(cuConjf(cv1[n]), make_cuFloatComplex(v2[n], 0)), out);
    }
}

// Real vector - complex vector dot-product
__device__ __inline__ void dot(float (&v1)[3], cuFloatComplex (&cv2)[3], cuFloatComplex &out)
{
    out = make_cuFloatComplex(0, 0);

    for(int n=0; n<3; n++)
    {
        out = cuCaddf(cuCmulf(make_cuFloatComplex(v1[n], 0), cv2[n]), out);
    }
}

// Real cross-product
__device__ __inline__ void ext(float (&v1)[3], float (&v2)[3], float (&out)[3])
{
    out[0] = v1[1]*v2[2] - v1[2]*v2[1];
    out[1] = v1[2]*v2[0] - v1[0]*v2[2];
    out[2] = v1[0]*v2[1] - v1[1]*v2[0];
}


// Complex conjugate of cross product
__device__ __inline__ void ext(cuFloatComplex (&cv1)[3], cuFloatComplex (&cv2)[3], cuFloatComplex (&out)[3])
{
    out[0] = cuCsubf(cuCmulf(cv1[1], cv2[2]), cuCmulf(cv1[2], cv2[1]));
    out[1] = cuCsubf(cuCmulf(cv1[2], cv2[0]), cuCmulf(cv1[0], cv2[2]));
    out[2] = cuCsubf(cuCmulf(cv1[0], cv2[1]), cuCmulf(cv1[1], cv2[0]));
}

// Cross product between an complex and a real vector
__device__ __inline__ void ext(cuFloatComplex (&cv1)[3], float (&v2)[3], cuFloatComplex (&out)[3])
{
    out[0] = cuCsubf(cuCmulf(cv1[1], make_cuFloatComplex(v2[2],0)), cuCmulf(cv1[2], make_cuFloatComplex(v2[1],0)));
    out[1] = cuCsubf(cuCmulf(cv1[2], make_cuFloatComplex(v2[0],0)), cuCmulf(cv1[0], make_cuFloatComplex(v2[2],0)));
    out[2] = cuCsubf(cuCmulf(cv1[0], make_cuFloatComplex(v2[1],0)), cuCmulf(cv1[1], make_cuFloatComplex(v2[0],0)));
}

// Cross product between a real vector and a complex vector
__device__ __inline__ void ext(float (&v1)[3], cuFloatComplex (&cv2)[3], cuFloatComplex (&out)[3])
{
    out[0] = cuCsubf(cuCmulf(make_cuFloatComplex(v1[1],0), cv2[2]), cuCmulf(make_cuFloatComplex(v1[2],0), cv2[1]));
    out[1] = cuCsubf(cuCmulf(make_cuFloatComplex(v1[2],0), cv2[0]), cuCmulf(make_cuFloatComplex(v1[0],0), cv2[2]));
    out[2] = cuCsubf(cuCmulf(make_cuFloatComplex(v1[0],0), cv2[1]), cuCmulf(make_cuFloatComplex(v1[1],0), cv2[0]));
}

// Difference between two real vectors
__device__ __inline__ void diff(float (&v1)[3], float (&v2)[3], float (&out)[3])
{
    for(int n=0; n<3; n++)
    {
        out[n] = v1[n] - v2[n];
    }
}

// Difference between two complex valued vectors
__device__ __inline__ void diff(cuFloatComplex (&cv1)[3], cuFloatComplex (&cv2)[3], cuFloatComplex (&out)[3])
{
    for(int n=0; n<3; n++)
    {
        out[n] = cuCsubf(cv1[n], cv2[n]);
    }
}

// Absolute value of real vector
__device__ __inline__ void abs(float (&v)[3], float &out)
{
    dot(v, v, out);
    out = sqrt(out);
}
// Return complex conjugate of complex vector
__device__ __inline__ void conja(cuFloatComplex (&cv)[3], cuFloatComplex (&out)[3])
{
    for(int n=0; n<3; n++)
    {
        out[n] = cuConjf(cv[n]);
    }
}
// Absolute value of a complex vector. Still returns a complex number!
__device__ __inline__ void abs(cuFloatComplex (&cv)[3], cuFloatComplex &out)
{
    cuFloatComplex cv_conj[3];
    conja(cv, cv_conj);
    dot(cv, cv_conj, out);
    out = make_cuFloatComplex(cuCabsf(out), 0);
}

// Return normalized real vector from vector
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

// Normalize complex vector
__device__ __inline__ void normalize(cuFloatComplex (&cv)[3], cuFloatComplex (&out)[3])
{
    cuFloatComplex cnorm;
    abs(cv, cnorm);

    for( int n=0; n<3; n++)
    {
        out[n] = cuCdivf(cv[n], cnorm);
    }
}

// Apply standard real s-multiplication on a real vector
__device__ __inline__ void s_mult(float (&v)[3], float &s, float (&out)[3])
{
    for(int n=0; n<3; n++)
    {
        out[n] = s * v[n];
    }
}


// Multiply complex vector by complex scalar
__device__ __inline__ void s_mult(cuFloatComplex (&cv)[3], cuFloatComplex &cs, cuFloatComplex (&out)[3])
{
    for(int n=0; n<3; n++)
    {
        out[n] = cuCmulf(cs, cv[n]);
    }
}

// Multiply real vector by complex scalar
__device__ __inline__ void s_mult(float (&v)[3], cuFloatComplex &cs, cuFloatComplex (&out)[3])
{
    for(int n=0; n<3; n++)
    {
        out[n] = cuCmulf(cs, make_cuFloatComplex(v[n],0));
    }
}

// Multiply complex vector by real scalar
__device__ __inline__ void s_mult(cuFloatComplex (&cv)[3], const float &s, cuFloatComplex (&out)[3])
{
    for(int n=0; n<3; n++)
    {
        out[n] = cuCmulf(make_cuFloatComplex(s,0), cv[n]);
    }
}

// Calculate refected vector from surface using complex incoming vector and real normal vector to surface
__device__ __inline__ void snell(cuFloatComplex (&cvin)[3], float (&normal)[3], cuFloatComplex (&out)[3])
{
    cuFloatComplex cfactor;
    dot(cvin, normal, cfactor);

    cfactor = cuCmulf(make_cuFloatComplex(2.,0), cfactor);

    cuFloatComplex rhs[3];
    s_mult(normal, cfactor, rhs);

    diff(cvin, rhs, out);
}

// Calculate refected vector from surface using complex incoming vector and real normal vector to surface
__device__ __inline__ void snell(float (&vin)[3], float (&normal)[3], float (&out)[3])
{
    float factor;
    dot(vin, normal, factor);

    factor = 2. * factor;

    float rhs[3];
    s_mult(normal, factor, rhs);

    diff(vin, rhs, out);
}

// Calculate Dyadic product between two real vectors
// Returns array of length 3, containing 3 arrays representing ROWS in the resulting matrix
__device__ __inline__ void dyad(float (&v1)[3], float (&v2)[3], float (&out)[3][3])
{
    for(int n=0; n<3; n++)
    {
        out[n][0] = v1[n] * v2[0];
        out[n][1] = v1[n] * v2[1];
        out[n][2] = v1[n] * v2[2];
    }
}

// Subtract matrix from another matrix element-wise
__device__ __inline__ void matDiff(float (&m1)[3][3], float (&m2)[3][3], float (&out)[3][3])
{
    for(int n=0; n<3; n++)
    {
        out[n][0] = m1[n][0] - m2[n][0];
        out[n][1] = m1[n][1] - m2[n][1];
        out[n][2] = m1[n][2] - m2[n][2];
    }
}

// Multiply matrix with vector to return vector
__device__ __inline__ void matVec(float (&m1)[3][3], float (&v1)[3], float (&out)[3])
{
    for(int n=0; n<3; n++)
    {
        out[n] = m1[n][0] * v1[0] + m1[n][1] * v1[1] + m1[n][2] * v1[2];
    }
}

__device__ __inline__ void matVec(float (&m1)[3][3], cuFloatComplex (&cv1)[3], cuFloatComplex (&out)[3])
{
    for(int n=0; n<3; n++)
    {
        out[n] = cuCaddf(cuCmulf(make_cuFloatComplex(m1[n][0],0), cv1[0]),
                cuCaddf(cuCmulf(make_cuFloatComplex(m1[n][1],0), cv1[1]),
                cuCmulf(make_cuFloatComplex(m1[n][2],0), cv1[2])));
    }
}
