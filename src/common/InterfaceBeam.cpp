#include "InterfaceBeam.h"

/*! \file InterfaceBeam.cpp
    \brief Implementations for beam initialization interface.
    
    Implementation of interface for initializing ray-trace frames, Gaussian beams and custom beams by calculating currents.
*/

/**
 * Generate ray-trace frame.
 *
 * Generate a ray-trace frame from an RTDict or RTDictf. Wrapper for initFrame.
 * @param rdict RTDict or RTDictf object.
 * @param fr Pointer to cframe or cframef object.
 *
 * @see initFrame()
 * @see RTDict
 * @see RTDictf
 * @see cframe
 * @see cframef
 */
void makeRTframe(RTDict rdict, cframe *fr)
{
    initFrame<RTDict, cframe, double>(rdict, fr);
}

/** 
 * Initialize Gaussian beam from GDict or GDictf.
 *
 * Takes a GDict or GDictf and generates two arrC3 or arrC3f objects, which contain the field and 
 *      associated currents and are allocated to passed pointer arguments. Wrapper for initGauss.
 *
 * @param gdict GDict or GDictf object from which to generate a Gaussian beam.
 * @param plane reflparams or reflparamsf object corresponding to surface on
 *      which to generate the Gaussian beam.
 * @param res_field Pointer to c2Bundle or c2Bundlef object.
 * @param res_current Pointer to c2Bundle or c2Bundlef object.
 *
 * @see initGauss()
 * @see GDict
 * @see GDictf
 * @see reflparams
 * @see reflparamsf
 * @see c2Bundle
 * @see c2Bundlef
 */
void makeGauss(GDict gdict, reflparams plane, c2Bundle *res_field, c2Bundle *res_current)
{
    initGauss<GDict, reflparams, c2Bundle, reflcontainer, double>(gdict, plane, res_field ,res_current);
}

/** 
 * Calculate currents from electromagnetic field.
 * 
 * Calculate the J and M vectorial currents given a vectorial E and H field.
 *      Can calculate full currents, PMC and PEC surfaces. Wrapper for calcJM.
 *
 * @param res_field Pointer to c2Bundle or c2Bundlef object.
 * @param res_current Pointer to c2Bundle or c2Bundlef object.
 * @param refldict reflparams or reflparamsf object corresponding to surface on
 *      which to calculate currents.
 * @param mode How to calculate currents. 0 is full currents, 1 is PMC and 2 is PEC.
 *
 * @see calcJM()
 * @see c2Bundle
 * @see c2Bundlef
 * @see reflparams
 * @see reflparamsf
 */
void calcCurrents(c2Bundle *res_field, c2Bundle *res_current, reflparams refldict, int mode)
{
    calcJM<c2Bundle, double, reflparams, reflcontainer>(res_field, res_current, refldict, mode);
}
