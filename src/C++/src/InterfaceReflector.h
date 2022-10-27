#include "Structs.h"

#ifndef __InterfaceReflector_h
#define __InterfaceReflector_h

extern "C"
{
    void generateGrid(reflparams refl, reflcontainer *container);
    void generateGridf(reflparamsf refl, reflcontainerf *container);
}
#endif
