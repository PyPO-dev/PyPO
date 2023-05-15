#include "Transform.h"

/*! \file Transform.cpp
    \brief Implementations of transformations for frames and fields/currents.
*/

/**
 * Transform a frame of rays.
 *
 * @param fr Pointer to frame to be transformed.
 * @param mat Pointer to transformation matrix.
 */
void transformRays(cframe *fr, double *mat)
{
    Utils<double> ut;

    bool vec = true;
    std::array<double, 3> inp, out;
    for (int i=0; i<fr->size; i++)
    {
        inp[0] = fr->x[i];
        inp[1] = fr->y[i];
        inp[2] = fr->z[i];

        ut.matVec4(mat, inp, out);
        
        fr->x[i] = out[0];
        fr->y[i] = out[1];
        fr->z[i] = out[2];

        inp[0] = fr->dx[i];
        inp[1] = fr->dy[i];
        inp[2] = fr->dz[i];

        ut.matVec4(mat, inp, out, vec);
        
        fr->dx[i] = out[0];
        fr->dy[i] = out[1];
        fr->dz[i] = out[2];
    }
}

/**
 * Transform a c2Bundle.
 *
 * Because c2Bundle objects are defined on surfaces, the components are only rotated.
 *
 * @param fr Pointer to frame to be transformed.
 * @param mat Pointer to transformation matrix.
 */
void transformFields(c2Bundle *fields, double *mat, int nTot)
{
    Utils<double> ut;
    
    bool vec = true;
    std::array<double, 3> inp, out;
    for (int i=0; i<nTot; i++)
    {
        inp[0] = fields->r1x[i];
        inp[1] = fields->r1y[i];
        inp[2] = fields->r1z[i];

        ut.matVec4(mat, inp, out, vec);
        //printf("%d\n", len);        
        fields->r1x[i] = out[0];
        fields->r1y[i] = out[1];
        fields->r1z[i] = out[2];

        inp[0] = fields->i1x[i];
        inp[1] = fields->i1y[i];
        inp[2] = fields->i1z[i];

        ut.matVec4(mat, inp, out, vec);
        
        fields->i1x[i] = out[0];
        fields->i1y[i] = out[1];
        fields->i1z[i] = out[2];
        
        inp[0] = fields->r2x[i];
        inp[1] = fields->r2y[i];
        inp[2] = fields->r2z[i];

        ut.matVec4(mat, inp, out, vec);
        
        fields->r2x[i] = out[0];
        fields->r2y[i] = out[1];
        fields->r2z[i] = out[2];

        inp[0] = fields->i2x[i];
        inp[1] = fields->i2y[i];
        inp[2] = fields->i2z[i];

        ut.matVec4(mat, inp, out, vec);
        
        fields->i2x[i] = out[0];
        fields->i2y[i] = out[1];
        fields->i2z[i] = out[2];
    }
}
