#include "BeamInit.h"
#include "Structs.h"

#ifdef _WIN32
#   define POPPY_DLL __declspec(dllexport)
#else
#   define POPPY_DLL
#endif

#ifndef InterfaceBeam_h
#define InterfaceBeam_h

/*! \file InterfaceBeam.h
    \brief Header for beam initialization interface.
    
    Declaration of interface for initializing ray-trace frames, Gaussian beams and custom beams by calculating currents.
*/

extern "C"
{
    POPPY_DLL void makeRTframe(RTDict rdict, cframe *fr);

    POPPY_DLL void makeGauss(GDict gdict, reflparams plane, c2Bundle *res_field, c2Bundle *res_current);

    POPPY_DLL void calcCurrents(c2Bundle *res_field, c2Bundle *res_current, reflparams rdict, int mode);
}

#endif
