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
    
    Declaration of interface for initializing ray-trace frames, Gaussian beams and custom beams by calculating currents.
*/

extern "C"
{
    PYPO_DLL void makeRTframe(RTDict rdict, cframe *fr);

    PYPO_DLL void makeGauss(GDict gdict, reflparams plane, c2Bundle *res_field, c2Bundle *res_current);

    PYPO_DLL void calcCurrents(c2Bundle *res_field, c2Bundle *res_current, reflparams rdict, int mode);
}

#endif
