#include "BeamInit.h"
#include "Structs.h"

#ifdef _WIN32
#   define PYPO_DLL __declspec(dllexport)
#else
#   define PYPO_DLL
#endif

#ifndef InterfaceBeam_h
#define InterfaceBeam_h

/*! \file InterfaceBeam.h
    \brief Header for beam initialization interface.
    
    Declaration of interface for initializing ray-trace frames and PO beams.
*/

extern "C"
{
    PYPO_DLL void makeRTframe(RTDict rdict, cframe *fr);
    
    PYPO_DLL void makeGRTframe(GRTDict grdict, cframe *fr);

    PYPO_DLL void makeGauss(GPODict gdict, reflparams plane, c2Bundle *res_field, c2Bundle *res_current);
    
    PYPO_DLL void makeScalarGauss(ScalarGPODict sgdict, reflparams plane, arrC1 *res_field);

    PYPO_DLL void calcCurrents(c2Bundle *res_field, c2Bundle *res_current, reflparams rdict, int mode);
}

#endif
