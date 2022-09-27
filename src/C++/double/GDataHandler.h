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

#include "DataHandler.h"

#ifndef __GDataHandler_h
#define __GDataHandler_h

class GDataHandler 
{
public:
    const int prec{ 16 }; // Machine precision
    
    DataHandler dh;
    
    GDataHandler() {};
    std::array<double*, 3> cppToCUDA_3DGrid(std::string &mode);
    std::array<double*, 2> cppToCUDA_2DGrid();
    std::array<cuDoubleComplex*, 3> cppToCUDA_Js();
    std::array<cuDoubleComplex*, 3> cppToCUDA_Ms();
    void cppToCUDA_area(double *out);
    std::array<double*, 3> cppToCUDA_3Dnormals();
    std::vector<std::array<std::complex<double>, 3>> CUDAToCpp_C(std::array<cuDoubleComplex*, 3> CUDA_C);
    std::vector<std::array<double, 3>> CUDAToCpp_R(std::array<double*, 3> CUDA_R);
};
#endif 
 
