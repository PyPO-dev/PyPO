#include "Structs.h"

#ifndef __InterfaceReflector_h
#define __InterfaceReflector_h

#ifdef _WIN32
#   define POPPY_DLL __declspec(dllexport)
#else
#   define POPPY_DLL
#endif

/*! \file InterfaceReflector.h
    \brief Header for reflector generation interface.
    
    Declaration of interface for reflector objects.
    @see Structs
    @see generateGrid()
    @see generateGridf()
*/
extern "C"
{
    POPPY_DLL void generateGrid(reflparams refl, reflcontainer *container,
                      bool transform=true, bool spheric=true);
    
    POPPY_DLL void generateGridf(reflparamsf refl, reflcontainerf *container,
                      bool transform=true, bool spheric=true);
}
#endif
