#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <fstream>
#include <string>
#include <iterator>
#include <iomanip>

#include <cuda.h>
#include <cuComplex.h>

#include "DataHandlerf.h"

#ifndef __GDataHandlerf_h
#define __GDataHandlerf_h

class GDataHandlerf 
{
public:
    const int prec{ 16 }; // Machine precision
    
    DataHandlerf dh;
    
    GDataHandlerf() {};
    std::array<float*, 3> cppToCUDA_3DGrid(std::string &mode);
    std::array<float*, 2> cppToCUDA_2DGrid();
    std::array<cuFloatComplex*, 3> cppToCUDA_Js();
    std::array<cuFloatComplex*, 3> cppToCUDA_Ms();
    void cppToCUDA_area(float *out);
    std::array<float*, 3> cppToCUDA_3Dnormals();
    std::vector<std::array<std::complex<float>, 3>> CUDAToCpp_C(std::array<cuFloatComplex*, 3> CUDA_C, int size);
    std::vector<std::array<float, 3>> CUDAToCpp_R(std::array<float*, 3> CUDA_R, int size);
};
#endif 
 
