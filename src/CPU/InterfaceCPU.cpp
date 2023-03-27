#include <InterfaceCPU.h>

/*! \file InterfaceCPU.cpp
    \brief Implementation of PO and RT library for CPU.

    Provides double and single precision interface for CPU PO and RT.
*/

/**
 * Calculate JM on surface.
 *
 * Calculate the J and M currents on a surface. 
 *
 * @param res Pointer to c2Bundle object.
 * @param source reflparams object, source surface.
 * @param target refparams object, target surface.
 * @param cs Pointer to reflcontainer, source surface.
 * @param ct Pointer to reflcontainer, target surface.
 * @param currents Pointer to c2Bundle containing currents on source surface.
 * @param k Wavenumber of radiation, in 1 / mm.
 * @param numThreads Number of computing threads to employ.
 * @param epsilon Electrical permittivity of source medium.
 * @param t_direction Time direction of beam (experimental!).
 *
 * @see c2Bundle
 * @see reflparams
 * @see reflcontainer
 */
void propagateToGrid_JM(c2Bundle *res, reflparams source, reflparams target,
                                reflcontainer *cs, reflcontainer *ct,
                                c2Bundle *currents,
                                double k, int numThreads, double epsilon,
                                double t_direction)
{
    Propagation<double, c2Bundle, reflcontainer, c2Bundle> prop(k, numThreads, cs->size, ct->size, epsilon, t_direction);

    // Generate source and target grids
    generateGrid(source, cs);
    generateGrid(target, ct);

    prop.parallelProp_JM(cs, ct, currents, res);
}

/**
 * Calculate EH on surface.
 *
 * Calculate the E and H fields on a surface. 
 *
 * @param res Pointer to c2Bundle object.
 * @param source reflparams object, source surface.
 * @param target refparams object, target surface.
 * @param cs Pointer to reflcontainer, source surface.
 * @param ct Pointer to reflcontainer, target surface.
 * @param currents Pointer to c2Bundle containing currents on source surface.
 * @param k Wavenumber of radiation, in 1 / mm.
 * @param numThreads Number of computing threads to employ.
 * @param epsilon Electrical permittivity of source medium.
 * @param t_direction Time direction of beam (experimental!).
 *
 * @see c2Bundle
 * @see reflparams
 * @see reflcontainer
 */
void propagateToGrid_EH(c2Bundle *res, reflparams source, reflparams target,
                                reflcontainer *cs, reflcontainer *ct,
                                c2Bundle *currents,
                                double k, int numThreads, double epsilon,
                                double t_direction)
{
    Propagation<double, c2Bundle, reflcontainer, c2Bundle> prop(k, numThreads, cs->size, ct->size, epsilon, t_direction);

    // Generate source and target grids
    generateGrid(source, cs);
    generateGrid(target, ct);

    prop.parallelProp_EH(cs, ct, currents, res);
}

/**
 * Calculate JM and EH on surface.
 *
 * Calculate the J, M currents and E, H fields on a surface. 
 *
 * @param res Pointer to c4Bundle object.
 * @param source reflparams object, source surface.
 * @param target refparams object, target surface.
 * @param cs Pointer to reflcontainer, source surface.
 * @param ct Pointer to reflcontainer, target surface.
 * @param currents Pointer to c2Bundle containing currents on source surface.
 * @param k Wavenumber of radiation, in 1 / mm.
 * @param numThreads Number of computing threads to employ.
 * @param epsilon Electrical permittivity of source medium.
 * @param t_direction Time direction of beam (experimental!).
 *
 * @see c4Bundle
 * @see reflparams
 * @see reflcontainer
 * @see c2Bundle
 */
void propagateToGrid_JMEH(c4Bundle *res, reflparams source, reflparams target,
                                reflcontainer *cs, reflcontainer *ct,
                                c2Bundle *currents,
                                double k, int numThreads, double epsilon,
                                double t_direction)
{
    Propagation<double, c4Bundle, reflcontainer, c2Bundle> prop(k, numThreads, cs->size, ct->size, epsilon, t_direction);

    // Generate source and target grids
    generateGrid(source, cs);
    generateGrid(target, ct);

    prop.parallelProp_JMEH(cs, ct, currents, res);
}

/**
 * Calculate EH and P on surface.
 *
 * Calculate the reflected E, H fields and P, the reflected Poynting vector field, on a surface. 
 *
 * @param res Pointer to c2rBundle object.
 * @param source reflparams object, source surface.
 * @param target refparams object, target surface.
 * @param cs Pointer to reflcontainer, source surface.
 * @param ct Pointer to reflcontainer, target surface.
 * @param currents Pointer to c2Bundle containing currents on source surface.
 * @param k Wavenumber of radiation, in 1 / mm.
 * @param numThreads Number of computing threads to employ.
 * @param epsilon Electrical permittivity of source medium.
 * @param t_direction Time direction of beam (experimental!).
 *
 * @see c2rBundle
 * @see reflparams
 * @see reflcontainer
 * @see c2Bundle
 */
void propagateToGrid_EHP(c2rBundle *res, reflparams source, reflparams target,
                                reflcontainer *cs, reflcontainer *ct,
                                c2Bundle *currents,
                                double k, int numThreads, double epsilon,
                                double t_direction)
{
    Propagation<double, c2rBundle, reflcontainer, c2Bundle> prop(k, numThreads, cs->size, ct->size, epsilon, t_direction);

    // Generate source and target grids
    generateGrid(source, cs);
    generateGrid(target, ct);

    prop.parallelProp_EHP(cs, ct, currents, res);
}

/**
 * Calculate scalar field on surface.
 *
 * Calculate the incoming scalar field on a surface. 
 *
 * @param res Pointer to arrC1 object.
 * @param source reflparams object, source surface.
 * @param target refparams object, target surface.
 * @param cs Pointer to reflcontainer, source surface.
 * @param ct Pointer to reflcontainer, target surface.
 * @param field Pointer to arrC1 containing field on source surface.
 * @param k Wavenumber of radiation, in 1 / mm.
 * @param numThreads Number of computing threads to employ.
 * @param epsilon Electrical permittivity of source medium.
 * @param t_direction Time direction of beam (experimental!).
 *
 * @see arrC1
 * @see reflparams
 * @see reflcontainer
 */
void propagateToGrid_scalar(arrC1 *res, reflparams source, reflparams target,
                                reflcontainer *cs, reflcontainer *ct,
                                arrC1 *field,
                                double k, int numThreads, double epsilon,
                                double t_direction)
{
    Propagation<double, arrC1, reflcontainer, arrC1> prop(k, numThreads, cs->size, ct->size, epsilon, t_direction);

    // Generate source and target grids
    generateGrid(source, cs);
    generateGrid(target, ct);

    prop.parallelPropScalar(cs, ct, field, res);
}

/**
 * Calculate EH on far-field surface.
 *
 * Calculate the E, H fields on a far-field surface. 
 *
 * @param res Pointer to c2Bundle object.
 * @param source reflparams object, source surface.
 * @param target refparams object, target surface.
 * @param cs Pointer to reflcontainer, source surface.
 * @param ct Pointer to reflcontainer, target surface.
 * @param currents Pointer to c2Bundle containing currents on source surface.
 * @param k Wavenumber of radiation, in 1 / mm.
 * @param numThreads Number of computing threads to employ.
 * @param epsilon Electrical permittivity of source medium.
 * @param t_direction Time direction of beam (experimental!).
 *
 * @see c2Bundle
 * @see reflparams
 * @see reflcontainer
 */
void propagateToFarField(c2Bundle *res, reflparams source, reflparams target,
                                reflcontainer *cs, reflcontainer *ct,
                                c2Bundle *currents,
                                double k, int numThreads, double epsilon,
                                double t_direction)
{
    Propagation<double, c2Bundle, reflcontainer, c2Bundle> prop(k, numThreads, cs->size, ct->size, epsilon, t_direction);

    // Generate source and target grids
    generateGrid(source, cs);
    generateGrid(target, ct);

    prop.parallelFarField(cs, ct, currents, res);
}

/**
 * Calculate ray-trace frame on surface.
 *
 * Calculate the ray-trace frame on a surface. 
 * 
 * @param ctp reflparams object of target surface.
 * @param fr_in Pointer to cframe object containing input frame.
 * @param fr_out Pointer to cframe object, to be filled with result frame.
 * @param numThreads Number of computing threads to employ.
 * @param epsilon Numerical accuracy of NR method.
 * @param t0 Initial value to try in NR method.
 *
 * @see reflparams
 * @see cframe
 */
void propagateRays(reflparams ctp, cframe *fr_in, cframe *fr_out,
                              int numThreads, double epsilon, double t0)
{
    int nTot = fr_in->size;
    RayTracer<reflparams, cframe, double> RT(numThreads, nTot, epsilon);

    RT.parallelRays(ctp, fr_in, fr_out, t0);
}

/**
 * Calculate JM on surface.
 *
 * Calculate the J and M currents on a surface. 
 *
 * @param res Pointer to c2Bundlef object.
 * @param source reflparamsf object, source surface.
 * @param target refparamsf object, target surface.
 * @param cs Pointer to reflcontainerf, source surface.
 * @param ct Pointer to reflcontainerf, target surface.
 * @param currents Pointer to c2Bundlef containing currents on source surface.
 * @param k Wavenumber of radiation, in 1 / mm.
 * @param numThreads Number of computing threads to employ.
 * @param epsilon Electrical permittivity of source medium.
 * @param t_direction Time direction of beam (experimental!).
 *
 * @see c2Bundlef
 * @see reflparamsf
 * @see reflcontainerf
 */
void propagateToGridf_JM(c2Bundlef *res, reflparamsf source, reflparamsf target,
                                reflcontainerf *cs, reflcontainerf *ct,
                                c2Bundlef *currents,
                                float k, int numThreads, float epsilon,
                                float t_direction)
{
    Propagation<float, c2Bundlef, reflcontainerf, c2Bundlef> prop(k, numThreads, cs->size, ct->size, epsilon, t_direction);

    // Generate source and target grids
    generateGridf(source, cs);
    generateGridf(target, ct);
    
    prop.parallelProp_JM(cs, ct, currents, res);
}

/**
 * Calculate EH on surface.
 *
 * Calculate the E, H fields on a surface. 
 *
 * @param res Pointer to c2Bundlef object.
 * @param source reflparamsf object, source surface.
 * @param target refparamsf object, target surface.
 * @param cs Pointer to reflcontainerf, source surface.
 * @param ct Pointer to reflcontainerf, target surface.
 * @param currents Pointer to c2Bundlef containing currents on source surface.
 * @param k Wavenumber of radiation, in 1 / mm.
 * @param numThreads Number of computing threads to employ.
 * @param epsilon Electrical permittivity of source medium.
 * @param t_direction Time direction of beam (experimental!).
 *
 * @see c2Bundlef
 * @see reflparamsf
 * @see reflcontainerf
 */
void propagateToGridf_EH(c2Bundlef *res, reflparamsf source, reflparamsf target,
                                reflcontainerf *cs, reflcontainerf *ct,
                                c2Bundlef *currents,
                                float k, int numThreads, float epsilon,
                                float t_direction)
{
    Propagation<float, c2Bundlef, reflcontainerf, c2Bundlef> prop(k, numThreads, cs->size, ct->size, epsilon, t_direction);

    // Generate source and target grids
    generateGridf(source, cs);
    generateGridf(target, ct);

    prop.parallelProp_EH(cs, ct, currents, res);
}

/**
 * Calculate JM and EH on surface.
 *
 * Calculate the J, M currents and E, H fields on a surface. 
 *
 * @param res Pointer to c4Bundlef object.
 * @param source reflparamsf object, source surface.
 * @param target refparamsf object, target surface.
 * @param cs Pointer to reflcontainerf, source surface.
 * @param ct Pointer to reflcontainerf, target surface.
 * @param currents Pointer to c2Bundlef containing currents on source surface.
 * @param k Wavenumber of radiation, in 1 / mm.
 * @param numThreads Number of computing threads to employ.
 * @param epsilon Electrical permittivity of source medium.
 * @param t_direction Time direction of beam (experimental!).
 *
 * @see c4Bundlef
 * @see reflparamsf
 * @see reflcontainerf
 * @see c2Bundlef
 */
void propagateToGridf_JMEH(c4Bundlef *res, reflparamsf source, reflparamsf target,
                                reflcontainerf *cs, reflcontainerf *ct,
                                c2Bundlef *currents,
                                float k, int numThreads, float epsilon,
                                float t_direction)
{
    Propagation<float, c4Bundlef, reflcontainerf, c2Bundlef> prop(k, numThreads, cs->size, ct->size, epsilon, t_direction);

    // Generate source and target grids
    generateGridf(source, cs);
    generateGridf(target, ct);

    prop.parallelProp_JMEH(cs, ct, currents, res);
}

/**
 * Calculate EH and P on surface.
 *
 * Calculate the reflected E, H fields and P, the reflected Poynting vector field, on a surface. 
 *
 * @param res Pointer to c2rBundlef object.
 * @param source reflparamsf object, source surface.
 * @param target refparamsf object, target surface.
 * @param cs Pointer to reflcontainerf, source surface.
 * @param ct Pointer to reflcontainerf, target surface.
 * @param currents Pointer to c2Bundlef containing currents on source surface.
 * @param k Wavenumber of radiation, in 1 / mm.
 * @param numThreads Number of computing threads to employ.
 * @param epsilon Electrical permittivity of source medium.
 * @param t_direction Time direction of beam (experimental!).
 *
 * @see c2rBundlef
 * @see reflparamsf
 * @see reflcontainerf
 * @see c2Bundle
 */
void propagateToGridf_EHP(c2rBundlef *res, reflparamsf source, reflparamsf target,
                                reflcontainerf *cs, reflcontainerf *ct,
                                c2Bundlef *currents,
                                float k, int numThreads, float epsilon,
                                float t_direction)
{
    Propagation<float, c2rBundlef, reflcontainerf, c2Bundlef> prop(k, numThreads, cs->size, ct->size, epsilon, t_direction);

    // Generate source and target grids
    generateGridf(source, cs);
    generateGridf(target, ct);

    prop.parallelProp_EHP(cs, ct, currents, res);
}

/**
 * Calculate scalar field on surface.
 *
 * Calculate the incoming scalar field on a surface. 
 *
 * @param res Pointer to arrC1f object.
 * @param source reflparamsf object, source surface.
 * @param target refparamsf object, target surface.
 * @param cs Pointer to reflcontainerf, source surface.
 * @param ct Pointer to reflcontainerf, target surface.
 * @param field Pointer to arrC1f containing currents on source surface.
 * @param k Wavenumber of radiation, in 1 / mm.
 * @param numThreads Number of computing threads to employ.
 * @param epsilon Electrical permittivity of source medium.
 * @param t_direction Time direction of beam (experimental!).
 *
 * @see arrC1f
 * @see reflparamsf
 * @see reflcontainerf
 */
void propagateToGridf_scalar(arrC1f *res, reflparamsf source, reflparamsf target,
                                reflcontainerf *cs, reflcontainerf *ct,
                                arrC1f *field,
                                float k, int numThreads, float epsilon,
                                float t_direction)
{
    Propagation<float, arrC1f, reflcontainerf, arrC1f> prop(k, numThreads, cs->size, ct->size, epsilon, t_direction);

    // Generate source and target grids
    generateGridf(source, cs);
    generateGridf(target, ct);

    prop.parallelPropScalar(cs, ct, field, res);
}

/**
 * Calculate E, H on a far-field surface.
 *
 * Calculate the E, H fields on a far-field surface. 
 *
 * @param res Pointer to c2Bundlef object.
 * @param source reflparamsf object, source surface.
 * @param target refparamsf object, target surface.
 * @param cs Pointer to reflcontainerf, source surface.
 * @param ct Pointer to reflcontainerf, target surface.
 * @param currents Pointer to c2Bundlef containing currents on source surface.
 * @param k Wavenumber of radiation, in 1 / mm.
 * @param numThreads Number of computing threads to employ.
 * @param epsilon Electrical permittivity of source medium.
 * @param t_direction Time direction of beam (experimental!).
 *
 * @see c2Bundlef
 * @see reflparamsf
 * @see reflcontainerf
 */
void propagateToFarFieldf(c2Bundlef *res, reflparamsf source, reflparamsf target,
                                reflcontainerf *cs, reflcontainerf *ct,
                                c2Bundlef *currents,
                                float k, int numThreads, float epsilon,
                                float t_direction)
{
    Propagation<float, c2Bundlef, reflcontainerf, c2Bundlef> prop(k, numThreads, cs->size, ct->size, epsilon, t_direction);

    // Generate source and target grids
    generateGridf(source, cs);
    generateGridf(target, ct);

    prop.parallelFarField(cs, ct, currents, res);
}
