#include "Structs.h"

#ifndef __InterfaceReflector_h
#define __InterfaceReflector_h

extern "C"
{
    void generateGrid(reflparams refl, reflcontainer *container, bool transform=true);
    void generateGridf(reflparamsf refl, reflcontainerf *container, bool transform=true);
}
#endif
