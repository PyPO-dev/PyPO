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

        //if (inv) {ut.invmatVec4(ctp.transf, inp, out);}
        //else {ut.matVec4(ctp.transf, inp, out);}
        ut.matVec4(mat, inp, out);
        
        fr->x[i] = out[0];
        fr->y[i] = out[1];
        fr->z[i] = out[2];

        inp[0] = fr->dx[i];
        inp[1] = fr->dy[i];
        inp[2] = fr->dz[i];

        //if (inv) {ut.invmatVec4(ctp.transf, inp, out, vec);}
        //else {ut.matVec4(ctp.transf, inp, out, vec);}
        ut.matVec4(mat, inp, out, vec);
        
        fr->dx[i] = out[0];
        fr->dy[i] = out[1];
        fr->dz[i] = out[2];
    }
}
