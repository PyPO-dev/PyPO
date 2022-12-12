#include "Structs.h"

#ifndef __InterfaceReflector_h
#define __InterfaceReflector_h

#ifdef _WIN32
#   define PYPO_DLL __declspec(dllexport)
#else
#   define PYPO_DLL
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
    PYPO_DLL void generateGrid(reflparams refl, reflcontainer *container,
                      bool transform=true, bool spheric=true);
    
    PYPO_DLL void generateGridf(reflparamsf refl, reflcontainerf *container,
                      bool transform=true, bool spheric=true);
}
#endif
