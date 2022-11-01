#include "BeamInit.h"
#include "Structs.h"

extern "C" void makeRTframe(RTDict rdict, cframe *fr)
{
    initFrame<RTDict, cframe, double>(rdict, fr);
}
