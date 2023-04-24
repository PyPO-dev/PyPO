#include <array>
#include <complex>

#ifndef __Structs_h
#define __Structs_h

/*! \file Structs.h
    \brief Structs used within PyPO. 
        
    This file contains all object that are used to either store results from calculations internally
        or pass data back/fetch data from the Python interface. 
        As ctypes does not support templates or overloading, this file contains
        explicit descriptions of double and float variants.
*/
struct arrC1;
struct arrR3;
struct c2Bundle;
struct c4Bundle;
struct c2rBundle;
struct reflparams;
struct reflcontainer;
struct cframe;
struct RTDict;
struct GRTDict;
struct GPODict;
struct ScalarGPODict;

/** 1D array of complex numbers.*/
struct arrC1 {
    double *x;     /**< array of double, representing real part of array.*/
    double *y;     /**< array of double, representing imaginary part of array.*/
};

/** 3D array of real numbers.*/
struct arrR3 {
    double *x;      /**<array of double, x-axis.*/
    double *y;      /**<array of double, y-axis.*/
    double *z;      /**<array of double, z-axis.*/
};

/** Object containing two 3D complex-valued arrays.*/
struct c2Bundle {
    double *r1x,      /**<array of double, field 1, real part, x-axis.*/ 
           *r1y,      /**<array of double, field 1, real part, y-axis.*/ 
           *r1z;      /**<array of double, field 1, real part, z-axis.*/
    double *i1x,      /**<array of double, field 1, imaginary part, x-axis.*/
           *i1y,      /**<array of double, field 1, imaginary part, y-axis.*/ 
           *i1z;      /**<array of double, field 1, imaginary part, z-axis.*/

    double *r2x,      /**<array of double, field 2, real part, x-axis.*/ 
           *r2y,      /**<array of double, field 2, real part, y-axis.*/ 
           *r2z;      /**<array of double, field 2, real part, z-axis.*/
    double *i2x,      /**<array of double, field 2, imaginary part, x-axis.*/ 
           *i2y,      /**<array of double, field 2, imaginary part, y-axis.*/ 
           *i2z;      /**<array of double, field 2, imaginary part, z-axis.*/
};

/** Object containing four 3D complex-valued arrays.*/
struct c4Bundle {
    double *r1x,      /**<array of double, field 1, real part, x-axis.*/ 
           *r1y,      /**<array of double, field 1, real part, y-axis.*/ 
           *r1z;      /**<array of double, field 1, real part, z-axis.*/
    double *i1x,      /**<array of double, field 1, imaginary part, x-axis.*/
           *i1y,      /**<array of double, field 1, imaginary part, y-axis.*/ 
           *i1z;      /**<array of double, field 1, imaginary part, z-axis.*/
                                                                                    
    double *r2x,      /**<array of double, field 2, real part, x-axis.*/ 
           *r2y,      /**<array of double, field 2, real part, y-axis.*/ 
           *r2z;      /**<array of double, field 2, real part, z-axis.*/
    double *i2x,      /**<array of double, field 2, imaginary part, x-axis.*/ 
           *i2y,      /**<array of double, field 2, imaginary part, y-axis.*/ 
           *i2z;      /**<array of double, field 2, imaginary part, z-axis.*/

    double *r3x,      /**<array of double, field 3, real part, x-axis.*/ 
           *r3y,      /**<array of double, field 3, real part, y-axis.*/ 
           *r3z;      /**<array of double, field 3, real part, z-axis.*/
    double *i3x,      /**<array of double, field 3, imaginary part, x-axis.*/
           *i3y,      /**<array of double, field 3, imaginary part, y-axis.*/ 
           *i3z;      /**<array of double, field 3, imaginary part, z-axis.*/
                                                                                    
    double *r4x,      /**<array of double, field 4, real part, x-axis.*/ 
           *r4y,      /**<array of double, field 4, real part, y-axis.*/ 
           *r4z;      /**<array of double, field 4, real part, z-axis.*/
    double *i4x,      /**<array of double, field 4, imaginary part, x-axis.*/ 
           *i4y,      /**<array of double, field 4, imaginary part, y-axis.*/ 
           *i4z;      /**<array of double, field 4, imaginary part, z-axis.*/
};

/** Object containing two 3D complex valued arrays and one 3D real valued array.*/
struct c2rBundle {
    double *r1x,      /**<array of double, field 1, real part, x-axis.*/ 
           *r1y,      /**<array of double, field 1, real part, y-axis.*/ 
           *r1z;      /**<array of double, field 1, real part, z-axis.*/
    double *i1x,      /**<array of double, field 1, imaginary part, x-axis.*/
           *i1y,      /**<array of double, field 1, imaginary part, y-axis.*/ 
           *i1z;      /**<array of double, field 1, imaginary part, z-axis.*/
                                                                                    
    double *r2x,      /**<array of double, field 2, real part, x-axis.*/ 
           *r2y,      /**<array of double, field 2, real part, y-axis.*/ 
           *r2z;      /**<array of double, field 2, real part, z-axis.*/
    double *i2x,      /**<array of double, field 2, imaginary part, x-axis.*/ 
           *i2y,      /**<array of double, field 2, imaginary part, y-axis.*/ 
           *i2z;      /**<array of double, field 2, imaginary part, z-axis.*/

    double *r3x,      /**<array of double, field 3, x-axis.*/ 
           *r3y,      /**<array of double, field 3, y-axis.*/ 
           *r3z;      /**<array of double, field 3, z-axis.*/
};

/** Object containing parameters for generating reflector surfaces.*/
struct reflparams {
    double *coeffs;     /**<array of 3 double. Contains a, b, c coefficients for reflectors.*/

    double *lxu;        /**<array of 2 double. Contains lower and upper x/u limits.*/
    double *lyv;        /**<array of 2 double. Contains lower and upper y/v limits.*/

    int *n_cells;       /**<array of 2 int. Contains gridsize along x/u and y/v axes.*/

    bool flip;          /**<Whether or not to flip normal vectors. Only relevant for quadric surfaces.*/
    int gmode;          /**<How to grid surface. 0 is "xy", 1 is "uv" and 2 is "AoE".*/
    double *gcenter;    /**<array of two double. Contains x and y co-ordinates for center of selection area.*/
    
    double ecc_uv;       /**<Eccentricity of uv-generated xy grid.*/
    double rot_uv;       /**<Position angle of uv-generated xy_grid, w.r.t. x-axis.*/
   
    int type;           /**<What type the reflector is. 0 is paraboloid, 1 is hyperboloid, 2 is ellipsoid and 3 is planar.*/

    double *transf;     /**<array of 16 double. Contains the transformation matrix of reflector surface.*/

};

/** Object containing co-ordinate and normal vector arrays for reflector surfaces.*/
struct reflcontainer {
    int size;       /**<Number of cells on surface.*/

    double *x;      /**<array of double. Contains co-ordinates along x-axis.*/
    double *y;      /**<array of double. Contains co-ordinates along y-axis.*/
    double *z;      /**<array of double. Contains co-ordinates along z-axis.*/

    double *nx;     /**<array of double. Contains normal vector component along x-axis.*/
    double *ny;     /**<array of double. Contains normal vector component along y-axis.*/
    double *nz;     /**<array of double. Contains normal vector component along z-axis.*/

    double *area;     /**<array of double. Contains area element size of surface.*/
};

/** Object containing ray evaluation points and corresponding direction vectors.
 *      Note that the stored direction vector is the reflected, not the incoming, vector.*/
struct cframe {
    int size;       /**<Number of rays in frame.*/

    double *x;     /**<array of double. Contains ray co-ordinates along x-axis.*/
    double *y;     /**<array of double. Contains ray co-ordinates along y-axis.*/
    double *z;     /**<array of double. Contains ray co-ordinates along z-axis.*/
                                                                                              
    double *dx;    /**<array of double. Contains direction component along x-axis.*/
    double *dy;    /**<array of double. Contains direction component along y-axis.*/
    double *dz;    /**<array of double. Contains direction component along z-axis.*/
};

/** Object for initializing a ray-trace frame object.*/
struct RTDict {
    int nRays;      /**<Number of rays to place in cframe.*/
    int nRing;      /**<Number of concentric rings in ray-trace beam.*/
    double angx0;    /**<Apex angle of beam at focus in x-direction, in degrees.*/
    double angy0;    /**<Apex angle of beam at focus in y-direction, in degrees.*/
    double x0;       /**<Semi-major axis of outer ring, in millimeters.*/
    double y0;       /**<Semi-minor axis of outer ring, in millimeters.*/
};

/** Object for initializing a Gaussian ray-trace frame object.*/
struct GRTDict {
    int nRays;      /**<Number of rays to place in cframe.*/
    double angx0;       /**<Beam waist along x-direction, in millimeters.*/
    double angy0;       /**<Beam waist along x-direction, in millimeters.*/
    double x0;       /**<Beam waist along x-direction, in millimeters.*/
    double y0;       /**<Beam waist along x-direction, in millimeters.*/
    int seed;
};

/** Object for initializing a Gaussian beam.*/
struct GPODict {
    double lam;     /**<Wavelength of beam, in millimeters.*/
    double w0x;     /**<Beamwaist size along x-direction, in millimeters.*/
    double w0y;     /**<Beamwaist size along y-direction, in millimeters.*/
    double n;       /**<Refractive index of medium.*/
    double E0;      /**<Peak electric field value.*/
    double dxyz;    /**<Astigmatic distance between x-focus and y-focus, in millimeters.*/
    double *pol;    /**<array of 3 double. Polarization components along x, y and z axes.*/
};

/** Object for initializing a scalar Gaussian beam.*/
struct ScalarGPODict {
    double lam;     /**<Wavelength of beam, in millimeters.*/
    double w0x;     /**<Beamwaist size along x-direction, in millimeters.*/
    double w0y;     /**<Beamwaist size along y-direction, in millimeters.*/
    double n;       /**<Refractive index of medium.*/
    double E0;      /**<Peak electric field value.*/
    double dxyz;    /**<Astigmatic distance between x-focus and y-focus, in millimeters.*/
};

// Have to write explicit types for float and float: ctypes doesnt support templates
struct arrC1f;
struct arrR3f;
struct c2Bundlef;
struct c4Bundlef;
struct c2rBundlef;
struct reflparamsf;
struct reflcontainerf;
struct cframef;
struct RTDictf;
struct GRTDictf;
struct GPODictf;
struct ScalarGPODictf;

/** 1D array of complex numbers.*/
struct arrC1f {
    float *x;     /**< array of float, representing real part of array.*/
    float *y;     /**< array of float, representing imaginary part of array.*/
};

/** 3D array of real numbers.*/
struct arrR3f {
    float *x;      /**<array of float, x-axis.*/
    float *y;      /**<array of float, y-axis.*/
    float *z;      /**<array of float, z-axis.*/
};

/** Object containing two 3D complex-valued arrays.*/
struct c2Bundlef {
    float *r1x,      /**<array of float, field 1, real part, x-axis.*/ 
           *r1y,      /**<array of float, field 1, real part, y-axis.*/ 
           *r1z;      /**<array of float, field 1, real part, z-axis.*/
    float *i1x,      /**<array of float, field 1, imaginary part, x-axis.*/
           *i1y,      /**<array of float, field 1, imaginary part, y-axis.*/ 
           *i1z;      /**<array of float, field 1, imaginary part, z-axis.*/

    float *r2x,      /**<array of float, field 2, real part, x-axis.*/ 
           *r2y,      /**<array of float, field 2, real part, y-axis.*/ 
           *r2z;      /**<array of float, field 2, real part, z-axis.*/
    float *i2x,      /**<array of float, field 2, imaginary part, x-axis.*/ 
           *i2y,      /**<array of float, field 2, imaginary part, y-axis.*/ 
           *i2z;      /**<array of float, field 2, imaginary part, z-axis.*/
};

/** Object containing four 3D complex-valued arrays.*/
struct c4Bundlef {
    float *r1x,      /**<array of float, field 1, real part, x-axis.*/ 
           *r1y,      /**<array of float, field 1, real part, y-axis.*/ 
           *r1z;      /**<array of float, field 1, real part, z-axis.*/
    float *i1x,      /**<array of float, field 1, imaginary part, x-axis.*/
           *i1y,      /**<array of float, field 1, imaginary part, y-axis.*/ 
           *i1z;      /**<array of float, field 1, imaginary part, z-axis.*/
                                                                                    
    float *r2x,      /**<array of float, field 2, real part, x-axis.*/ 
           *r2y,      /**<array of float, field 2, real part, y-axis.*/ 
           *r2z;      /**<array of float, field 2, real part, z-axis.*/
    float *i2x,      /**<array of float, field 2, imaginary part, x-axis.*/ 
           *i2y,      /**<array of float, field 2, imaginary part, y-axis.*/ 
           *i2z;      /**<array of float, field 2, imaginary part, z-axis.*/

    float *r3x,      /**<array of float, field 3, real part, x-axis.*/ 
           *r3y,      /**<array of float, field 3, real part, y-axis.*/ 
           *r3z;      /**<array of float, field 3, real part, z-axis.*/
    float *i3x,      /**<array of float, field 3, imaginary part, x-axis.*/
           *i3y,      /**<array of float, field 3, imaginary part, y-axis.*/ 
           *i3z;      /**<array of float, field 3, imaginary part, z-axis.*/
                                                                                    
    float *r4x,      /**<array of float, field 4, real part, x-axis.*/ 
           *r4y,      /**<array of float, field 4, real part, y-axis.*/ 
           *r4z;      /**<array of float, field 4, real part, z-axis.*/
    float *i4x,      /**<array of float, field 4, imaginary part, x-axis.*/ 
           *i4y,      /**<array of float, field 4, imaginary part, y-axis.*/ 
           *i4z;      /**<array of float, field 4, imaginary part, z-axis.*/
};

/** Object containing two 3D complex valued arrays and one 3D real valued array.*/
struct c2rBundlef {
    float *r1x,      /**<array of float, field 1, real part, x-axis.*/ 
           *r1y,      /**<array of float, field 1, real part, y-axis.*/ 
           *r1z;      /**<array of float, field 1, real part, z-axis.*/
    float *i1x,      /**<array of float, field 1, imaginary part, x-axis.*/
           *i1y,      /**<array of float, field 1, imaginary part, y-axis.*/ 
           *i1z;      /**<array of float, field 1, imaginary part, z-axis.*/
                                                                                    
    float *r2x,      /**<array of float, field 2, real part, x-axis.*/ 
           *r2y,      /**<array of float, field 2, real part, y-axis.*/ 
           *r2z;      /**<array of float, field 2, real part, z-axis.*/
    float *i2x,      /**<array of float, field 2, imaginary part, x-axis.*/ 
           *i2y,      /**<array of float, field 2, imaginary part, y-axis.*/ 
           *i2z;      /**<array of float, field 2, imaginary part, z-axis.*/

    float *r3x,      /**<array of float, field 3, x-axis.*/ 
           *r3y,      /**<array of float, field 3, y-axis.*/ 
           *r3z;      /**<array of float, field 3, z-axis.*/
};

/** Object containing parameters for generating reflector surfaces.*/
struct reflparamsf {
    float *coeffs;     /**<array of 3 float. Contains a, b, c coefficients for reflectors.*/

    float *lxu;        /**<array of 2 float. Contains lower and upper x/u limits.*/
    float *lyv;        /**<array of 2 float. Contains lower and upper y/v limits.*/

    int *n_cells;       /**<array of 2 int. Contains gridsize along x/u and y/v axes.*/

    bool flip;          /**<Whether or not to flip normal vectors. Only relevant for quadric surfaces.*/
    int gmode;          /**<How to grid surface. 0 is "xy", 1 is "uv" and 2 is "AoE".*/
    float *gcenter;    /**<array of two float. Contains x and y co-ordinates for center of selection area.*/
    
    float ecc_uv;       /**<Eccentricity of uv-generated xy grid.*/
    float rot_uv;       /**<Position angle of uv-generated xy_grid, w.r.t. x-axis.*/

    int type;           /**<What type the reflector is. 0 is paraboloid, 1 is hyperboloid, 2 is ellipsoid and 3 is planar.*/

    float *transf;     /**<array of 16 float. Contains the transformation matrix of reflector surface.*/

};

/** Object containing co-ordinate and normal vector arrays for reflector surfaces.*/
struct reflcontainerf {
    int size;       /**<Number of cells on surface.*/

    float *x;      /**<array of float. Contains co-ordinates along x-axis.*/
    float *y;      /**<array of float. Contains co-ordinates along y-axis.*/
    float *z;      /**<array of float. Contains co-ordinates along z-axis.*/

    float *nx;     /**<array of float. Contains normal vector component along x-axis.*/
    float *ny;     /**<array of float. Contains normal vector component along y-axis.*/
    float *nz;     /**<array of float. Contains normal vector component along z-axis.*/

    float *area;     /**<array of float. Contains area element size of surface.*/
};

/** Object containing ray evaluation points and corresponding direction vectors.
 *      Note that the stored direction vector is the reflected, not the incoming, vector.*/
struct cframef {
    int size;       /**<Number of rays in frame.*/

    float *x;     /**<array of float. Contains ray co-ordinates along x-axis.*/
    float *y;     /**<array of float. Contains ray co-ordinates along y-axis.*/
    float *z;     /**<array of float. Contains ray co-ordinates along z-axis.*/
                                                                                              
    float *dx;    /**<array of float. Contains direction component along x-axis.*/
    float *dy;    /**<array of float. Contains direction component along y-axis.*/
    float *dz;    /**<array of float. Contains direction component along z-axis.*/
};

/** Object for initializing a ray-trace frame object.*/
struct RTDictf {
    int nRays;      /**<Number of rays to place in cframe.*/
    int nRing;      /**<Number of concentric rings in ray-trace beam.*/
    float angx0;    /**<Apex angle of beam at focus in x-direction, in degrees.*/
    float angy0;    /**<Apex angle of beam at focus in y-direction, in degrees.*/
    float x0;       /**<Semi-major axis of outer ring, in millimeters.*/
    float y0;       /**<Semi-minor axis of outer ring, in millimeters.*/
    float *tChief; /**<array of 3 float. Tilt of chief, ray, along x, y or z axis, in degrees.*/
    float *oChief; /**<array of 3 float. Co-ordinate of chief ray origin.*/
};

/** Object for initializing a ray-trace frame object.*/
struct GRTDictf {
    int nRays;      /**<Number of rays to place in cframe.*/
    float angx0;       /**<Beam waist along x-direction, in millimeters.*/
    float angy0;       /**<Beam waist along x-direction, in millimeters.*/
    float x0;       /**<Beam waist along x-direction, in millimeters.*/
    float y0;       /**<Beam waist along x-direction, in millimeters.*/
    int seed;
    float *tChief; /**<array of 3 float. Tilt of chief, ray, along x, y or z axis, in degrees.*/
    float *oChief; /**<array of 3 float. Co-ordinate of chief ray origin.*/
};

/** Object for initializing a Gaussian beam.*/
struct GPODictf {
    float lam;     /**<Wavelength of beam, in millimeters.*/
    float w0x;     /**<Beamwaist size along x-direction, in millimeters.*/
    float w0y;     /**<Beamwaist size along y-direction, in millimeters.*/
    float n;       /**<Refractive index of medium.*/
    float E0;      /**<Peak electric field value.*/
    float dxyz;    /**<Astigmatic distance between x-focus and y-focus, in millimeters.*/
    float *pol;    /**<array of 3 float. Polarization components along x, y and z axes.*/
};

/** Object for initializing a scalar Gaussian beam.*/
struct ScalarGPODictf {
    float lam;     /**<Wavelength of beam, in millimeters.*/
    float w0x;     /**<Beamwaist size along x-direction, in millimeters.*/
    float w0y;     /**<Beamwaist size along y-direction, in millimeters.*/
    float n;       /**<Refractive index of medium.*/
    float E0;      /**<Peak electric field value.*/
    float dxyz;    /**<Astigmatic distance between x-focus and y-focus, in millimeters.*/
};
#endif
