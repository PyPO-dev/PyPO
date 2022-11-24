#include "BeamInit.h"
#include "Structs.h"

#ifdef _WIN32
#   define POPPY_DLL __declspec(dllexport)
#else
#   define POPPY_DLL
#endif

POPPY_DLL extern "C" void makeRTframe(RTDict rdict, cframe *fr)
{
    initFrame<RTDict, cframe, double>(rdict, fr);
}

POPPY_DLL extern "C" void makeGauss(GDict gdict, reflparams plane, c2Bundle *res_field, c2Bundle *res_current)
{
    initGauss<GDict, reflparams, c2Bundle, reflcontainer, double>(gdict, plane, res_field ,res_current);
}

POPPY_DLL extern "C" void calcCurrents(c2Bundle *res_field, c2Bundle *res_current, reflparams rdict, int mode)
{
    calcJM<c2Bundle, double, reflparams, reflcontainer>(res_field, res_current, rdict, mode);
}
