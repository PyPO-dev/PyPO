#include <math.h> 
#include <cuda.h>
#include <cuComplex.h>

// The Utils file contains mostly linear algebra functions as CUDA device functions

// Real dot-product
__device__ void dot(float (&v1)[3], float (&v2)[3], float &out)
{
    out = 0;
    
    for(int n=0; n<3; n++)
    {
        out += v1[n] * v2[n];
    }
}


// Complex hermitian conjugate inner-product
__device__ void dot(cuFloatComplex (&cv1)[3], cuFloatComplex (&cv2)[3], cuFloatComplex &out)
{
    out = make_cuFloatComplex(0, 0);
    
    for(int n=0; n<3; n++)
    {
        out = cuCaddf(cuCmulf(cuConjf(cv1[n]), cv2[n]), out);
    }
}

// Complex vector - real vector dot-product
__device__ void dot(cuFloatComplex (&cv1)[3], float (&v2)[3], cuFloatComplex &out)
{
    out = make_cuFloatComplex(0, 0);
    
    for(int n=0; n<3; n++)
    {
        out = cuCaddf(cuCmulf(cuConjf(cv1[n]), make_cuFloatComplex(v2[n], 0)), out);
    }
}

// Real vector - complex vector dot-product
__device__ void dot(float (&v1)[3], cuFloatComplex (&cv2)[3], cuFloatComplex &out)
{
    out = make_cuFloatComplex(0, 0);
    
    for(int n=0; n<3; n++)
    {
        out = cuCaddf(cuCmulf(make_cuFloatComplex(v1[n], 0), cv2[n]), out);
    }
}

// Real cross-product
__device__ void ext(float (&v1)[3], float (&v2)[3], float (&out)[3])
{
    out[0] = v1[1]*v2[2] - v1[2]*v2[1];
    out[1] = v1[2]*v2[0] - v1[0]*v2[2];
    out[2] = v1[0]*v2[1] - v1[1]*v2[0];
}


// Complex conjugate of cross product
__device__ void ext(cuFloatComplex (&cv1)[3], cuFloatComplex (&cv2)[3], cuFloatComplex (&out)[3])
{
    out[0] = cuCsubf(cuCmulf(cv1[1], cv2[2]), cuCmulf(cv1[2], cv2[1]));
    out[1] = cuCsubf(cuCmulf(cv1[2], cv2[0]), cuCmulf(cv1[0], cv2[2]));
    out[2] = cuCsubf(cuCmulf(cv1[0], cv2[1]), cuCmulf(cv1[1], cv2[0]));
}

// Cross product between an complex and a real vector
__device__ void ext(cuFloatComplex (&cv1)[3], float (&v2)[3], cuFloatComplex (&out)[3])
{
    out[0] = cuCsubf(cuCmulf(cv1[1], make_cuFloatComplex(v2[2],0)), cuCmulf(cv1[2], make_cuFloatComplex(v2[1],0)));
    out[1] = cuCsubf(cuCmulf(cv1[2], make_cuFloatComplex(v2[0],0)), cuCmulf(cv1[0], make_cuFloatComplex(v2[2],0)));
    out[2] = cuCsubf(cuCmulf(cv1[0], make_cuFloatComplex(v2[1],0)), cuCmulf(cv1[1], make_cuFloatComplex(v2[0],0)));
}

// Cross product between a real vector and a complex vector
__device__ void ext(float (&v1)[3], cuFloatComplex (&cv2)[3], cuFloatComplex (&out)[3])
{
    out[0] = cuCsubf(cuCmulf(make_cuFloatComplex(v1[1],0), cv2[2]), cuCmulf(make_cuFloatComplex(v1[2],0), cv2[1]));
    out[1] = cuCsubf(cuCmulf(make_cuFloatComplex(v1[2],0), cv2[0]), cuCmulf(make_cuFloatComplex(v1[0],0), cv2[2]));
    out[2] = cuCsubf(cuCmulf(make_cuFloatComplex(v1[0],0), cv2[1]), cuCmulf(make_cuFloatComplex(v1[1],0), cv2[0]));
}

// Difference between two real vectors
__device__ void diff(float (&v1)[3], float (&v2)[3], float (&out)[3])
{
    for(int n=0; n<3; n++)
    {
        out[n] = v1[n] - v2[n];
    }
}

// Difference between two complex valued vectors
__device__ void diff(cuFloatComplex (&cv1)[3], cuFloatComplex (&cv2)[3], cuFloatComplex (&out)[3])
{
    for(int n=0; n<3; n++)
    {
        out[n] = cuCsubf(cv1[n], cv2[n]);
    }
}

// Absolute value of real vector
__device__ void abs(float (&v)[3], float &out)
{
    dot(v, v, out);
    out = sqrt(out);
}
// Return complex conjugate of complex vector
__device__ void conja(cuFloatComplex (&cv)[3], cuFloatComplex (&out)[3])
{
    for(int n=0; n<3; n++)
    {
        out[n] = cuConjf(cv[n]);
    }
}
// Absolute value of a complex vector. Still returns a complex number!
__device__ void abs(cuFloatComplex (&cv)[3], cuFloatComplex &out)
{
    cuFloatComplex cv_conj[3];
    conja(cv, cv_conj);
    dot(cv, cv_conj, out);
    out = make_cuFloatComplex(cuCabsf(out), 0);
}

// Return normalized real vector from vector
__device__ void normalize(float (&v)[3], float (&out)[3])
{
    float norm;
    abs(v, norm);
    
    for( int n=0; n<3; n++)
    {
        out[n] = v[n] / norm;
    }
}

// Normalize complex vector
__device__ void normalize(cuFloatComplex (&cv)[3], cuFloatComplex (&out)[3])
{
    cuFloatComplex cnorm;
    abs(cv, cnorm);
    
    for( int n=0; n<3; n++)
    {
        out[n] = cuCdivf(cv[n], cnorm);
    }
}

// Apply standard real s-multiplication on a real vector
__device__ void s_mult(float (&v)[3], float &s, float (&out)[3])
{
    for(int n=0; n<3; n++)
    {
        out[n] = s * v[n];
    }
}


// Multiply complex vector by complex scalar
__device__ void s_mult(cuFloatComplex (&cv)[3], cuFloatComplex &cs, cuFloatComplex (&out)[3])
{
    for(int n=0; n<3; n++)
    {
        out[n] = cuCmulf(cs, cv[n]);
    }
}

// Multiply real vector by complex scalar
__device__ void s_mult(float (&v)[3], cuFloatComplex &cs, cuFloatComplex (&out)[3])
{
    for(int n=0; n<3; n++)
    {
        out[n] = cuCmulf(cs, make_cuFloatComplex(v[n],0));
    }
}

// Multiply complex vector by real scalar
__device__ void s_mult(cuFloatComplex (&cv)[3], const float &s, cuFloatComplex (&out)[3])
{
    for(int n=0; n<3; n++)
    {
        out[n] = cuCmulf(make_cuFloatComplex(s,0), cv[n]);
    }
}

// Calculate refected vector from surface using complex incoming vector and real normal vector to surface
__device__ void snell(cuFloatComplex (&cvin)[3], float (&normal)[3], cuFloatComplex (&out)[3])
{
    cuFloatComplex cfactor;
    dot(cvin, normal, cfactor);
    
    cfactor = cuCmulf(make_cuFloatComplex(2.,0), cfactor);
    
    cuFloatComplex rhs[3];
    s_mult(normal, cfactor, rhs);
    
    diff(cvin, rhs, out);
}

// Calculate refected vector from surface using complex incoming vector and real normal vector to surface
__device__ void snell(float (&vin)[3], float (&normal)[3], float (&out)[3])
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
__device__ void dyad(float (&v1)[3], float (&v2)[3], float (&out)[3][3])
{
    for(int n=0; n<3; n++)
    {
        out[n][0] = v1[n] * v2[0];
        out[n][1] = v1[n] * v2[1];
        out[n][2] = v1[n] * v2[2];
    }
}

// Subtract matrix from another matrix element-wise
__device__ void matDiff(float (&m1)[3][3], float (&m2)[3][3], float (&out)[3][3])
{
    for(int n=0; n<3; n++)
    {
        out[n][0] = m1[n][0] - m2[n][0];
        out[n][1] = m1[n][1] - m2[n][1];
        out[n][2] = m1[n][2] - m2[n][2];
    }
}

// Multiply matrix with vector to return vector
__device__ void matVec(float (&m1)[3][3], float (&v1)[3], float (&out)[3])
{
    for(int n=0; n<3; n++)
    {
        out[n] = m1[n][0] * v1[0] + m1[n][1] * v1[1] + m1[n][2] * v1[2];
    }
}

__device__ void matVec(float (&m1)[3][3], cuFloatComplex (&cv1)[3], cuFloatComplex (&out)[3])
{
    for(int n=0; n<3; n++)
    {
        out[n] = cuCaddf(cuCmulf(make_cuFloatComplex(m1[n][0],0), cv1[0]), 
                cuCaddf(cuCmulf(make_cuFloatComplex(m1[n][1],0), cv1[1]),
                cuCmulf(make_cuFloatComplex(m1[n][2],0), cv1[2])));
    }
}


 
