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
    const int prec{ 8 }; // Machine precision
    
    DataHandlerf dh;
    
    GDataHandlerf() {};
    __host__ std::array<float*, 3> cppToCUDA_3DGrid(std::string &mode);
    __host__ std::array<float*, 2> cppToCUDA_2DGrid();
    __host__ std::array<cuFloatComplex*, 3> cppToCUDA_Js();
    __host__ std::array<cuFloatComplex*, 3> cppToCUDA_Ms();
    __host__ void cppToCUDA_area(float *out);
    __host__ std::array<float*, 3> cppToCUDA_3Dnormals();
    __host__ std::vector<std::array<std::complex<float>, 3>> CUDAToCpp_C(std::array<cuFloatComplex*, 3> CUDA_C);
    __host__ std::vector<std::array<float, 3>> CUDAToCpp_R(std::array<float*, 3> CUDA_R);
};
#endif 
 
