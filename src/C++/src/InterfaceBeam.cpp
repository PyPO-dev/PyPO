#include "BeamInit.h"
#include "Structs.h"

extern "C" void makeRTframe(RTDict rdict, cframe *fr)
{
    initFrame<RTDict, cframe, double>(rdict, fr);
}

extern "C" void makeGauss(GDict gdict, reflparams plane, c2Bundle *res_field, c2Bundle *res_current)
{
    initGauss<GDict, reflparams, c2Bundle, reflcontainer, double>(gdict, plane, res_field ,res_current);
}

extern "C" void calcCurrents(c2Bundle *res_field, c2Bundle *res_current, reflparams rdict, int mode)
{
    calcJM<c2Bundle, double, reflparams, reflcontainer>(res_field, res_current, rdict, mode);
}
