#include <math.h>
#include <cuda.h>
#include <cuComplex.h>

// File contains overloaded arithmetics for complex numbers

__host__ __device__ __inline__ void addCo(cuDoubleComplex &a, cuDoubleComplex &b, cuDoubleComplex &c)
{
    c.x = a.x + b.x;
    c.y = a.y + b.y;
}

__host__ __device__ __inline__ void subCo(cuDoubleComplex &a, cuDoubleComplex &b, cuDoubleComplex &c)
{
    c.x = a.x - b.x;
    c.y = a.y - b.y;
}

__host__ __device__ __inline__ void mulCo(cuDoubleComplex &a, cuDoubleComplex &b, cuDoubleComplex &c)
{
    c.x = a.x*b.x - a.y*b.y;
    c.y = a.x*b.y + a.y*b.x;
}

__host__ __device__ __inline__ void divCo(cuDoubleComplex &a, cuDoubleComplex &b, cuDoubleComplex &c)
{
    c.x = (a.x*b.x + a.y*b.y) / (b.x*b.x + b.y*b.y);
    c.y = (a.x*b.y - a.y*b.x) / (b.x*b.x + b.y*b.y);
}

__host__ __device__ __inline__ void expCo(cuDoubleComplex &a, cuDoubleComplex &b)
{
    double t = exp(a.x);
    double ys = sin(a.y);
    double yc = cos(a.y);

    b.x = t*yc;
    b.y = t*ys;
}

// FLOAT OVERLOADS

__host__ __device__ __inline__ void addCo(cuFloatComplex &a, cuFloatComplex &b, cuFloatComplex &c)
{
    c.x = a.x + b.x;
    c.y = a.y + b.y;
}

__host__ __device__ __inline__ void subCo(cuFloatComplex &a, cuFloatComplex &b, cuFloatComplex &c)
{
    c.x = a.x - b.x;
    c.y = a.y - b.y;
}

__host__ __device__ __inline__ void mulCo(cuFloatComplex &a, cuFloatComplex &b, cuFloatComplex &c)
{
    c.x = a.x*b.x - a.y*b.y;
    c.y = a.x*b.y + a.y*b.x;
}

__host__ __device__ __inline__ void divCo(cuFloatComplex &a, cuFloatComplex &b, cuFloatComplex &c)
{
    c.x = (a.x*b.x + a.y*b.y) / (b.x*b.x + b.y*b.y);
    c.y = (a.x*b.y - a.y*b.x) / (b.x*b.x + b.y*b.y);
}

__host__ __device__ __inline__ void expCo(cuFloatComplex &a, cuFloatComplex &b)
{
    float t = exp(a.x);
    float ys = sin(a.y);
    float yc = cos(a.y);

    b.x = t*yc;
    b.y = t*ys;
}
