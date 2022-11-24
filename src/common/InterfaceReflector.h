#include "Structs.h"

#ifndef __InterfaceReflector_h
#define __InterfaceReflector_h

#ifdef _WIN32
#   define POPPY_DLL __declspec(dllexport)
#else
#   define POPPY_DLL
#endif

extern "C"
{
    POPPY_DLL void generateGrid(reflparams refl, reflcontainer *container,
                      bool transform=true, bool spheric=true);
    POPPY_DLL void generateGridf(reflparamsf refl, reflcontainerf *container,
                      bool transform=true, bool spheric=true);
}
#endif
