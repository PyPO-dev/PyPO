#include <array>
#include <complex>

#ifndef __Structs_h
#define __Structs_h

struct arrC1;
struct arrR3;
struct c2Bundle;
struct c4Bundle;
struct c2rBundle;
struct reflparams;
struct reflcontainer;

typedef struct arrC1 {
    double *rx;
    double *ix;
} arrC1;

typedef struct arrR3 {
    double *x;
    double *y;
    double *z;
} arrR3;

typedef struct c2Bundle {
    double *r1x, *r1y, *r1z;
    double *i1x, *i1y, *i1z;

    double *r2x, *r2y, *r2z;
    double *i2x, *i2y, *i2z;
} c2Bundle;

typedef struct c4Bundle {
    double *r1x, *r1y, *r1z;
    double *i1x, *i1y, *i1z;

    double *r2x, *r2y, *r2z;
    double *i2x, *i2y, *i2z;

    double *r3x, *r3y, *r3z;
    double *i3x, *i3y, *i3z;

    double *r4x, *r4y, *r4z;
    double *i4x, *i4y, *i4z;
} c4Bundle;

typedef struct c2rBundle {
    double *r1x, *r1y, *r1z;
    double *i1x, *i1y, *i1z;

    double *r2x, *r2y, *r2z;
    double *i2x, *i2y, *i2z;

    double *r3x, *r3y, *r3z;
} c2rBundle;

typedef struct reflparams {
    double *coeffs;

    double *lxu;
    double *lyv;

    int *n_cells;

    bool flip;
    int gmode;
    double *gcenter;
    int type;

    double *transf;

} reflparams;

typedef struct reflcontainer {
    int size;

    double *x;
    double *y;
    double *z;

    double *nx;
    double *ny;
    double *nz;

    double *area;
} reflcontainer;

typedef struct cframe {
    int size;

    double *x;
    double *y;
    double *z;

    double *dx;
    double *dy;
    double *dz;
} cframe;

typedef struct RTDict {
    int nRays;
    int nRing;
    double angx;
    double angy;
    double a;
    double b;
    double *tChief;
    double *oChief;
} RTDict;

typedef struct GDict {
    double lam;
    double w0x;
    double w0y;
    double n;
    double E0;
    double z;
    double *pol;
} GDict;

// Have to write explicit types for float and float: ctypes doesnt support templates
struct arrC1f;
struct arrR3f;
struct c2Bundlef;
struct c4Bundlef;
struct c2rBundlef;
struct reflparamsf;
struct reflcontainerf;

typedef struct arrC1f {
    float *rx;
    float *ix;
} arrC1f;

typedef struct arrR3f {
    float *x;
    float *y;
    float *z;
} arrR3f;

typedef struct c2Bundlef {
    float *r1x, *r1y, *r1z;
    float *i1x, *i1y, *i1z;

    float *r2x, *r2y, *r2z;
    float *i2x, *i2y, *i2z;
} c2Bundlef;

typedef struct c4Bundlef {
    float *r1x, *r1y, *r1z;
    float *i1x, *i1y, *i1z;

    float *r2x, *r2y, *r2z;
    float *i2x, *i2y, *i2z;

    float *r3x, *r3y, *r3z;
    float *i3x, *i3y, *i3z;

    float *r4x, *r4y, *r4z;
    float *i4x, *i4y, *i4z;
} c4Bundlef;

typedef struct c2rBundlef {
    float *r1x, *r1y, *r1z;
    float *i1x, *i1y, *i1z;

    float *r2x, *r2y, *r2z;
    float *i2x, *i2y, *i2z;

    float *r3x, *r3y, *r3z;
} c2rBundlef;

typedef struct reflparamsf {
    float *coeffs;

    float *lxu;
    float *lyv;

    int *n_cells;

    bool flip;
    int gmode;
    float *gcenter;
    int type;

    float *transf;

} reflparamsf;

typedef struct reflcontainerf {
    int size;

    float *x;
    float *y;
    float *z;

    float *nx;
    float *ny;
    float *nz;

    float *area;
} reflcontainerf;

typedef struct cframef {
    int size;

    float *x;
    float *y;
    float *z;

    float *dx;
    float *dy;
    float *dz;
} cframef;

typedef struct RTDictf {
    int nRays;
    int nRing;
    float angx;
    float angy;
    float a;
    float b;
    float *tChief;
    float *oChief;
} RTDictf;

typedef struct GDictf {
    float lam;
    float w0x;
    float w0y;
    float n;
    float E0;
    float z;
    float *pol;
} GDictf;

#endif
