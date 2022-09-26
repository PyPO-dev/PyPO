#include "GDataHandler.h"

// CUDA specific data functions

// Convert real 3D grid to three 1D arrays
std::array<double*, 3> GDataHandler::cppToCUDA_3DGrid(std::string &mode)
{
    std::vector<std::array<double, 3>> data = this->dh.readGrid3D(mode);
    std::array<double*, 3> out;
    
    int m = data.size();
    
    double outx[m];
    double outy[m];
    double outz[m];
    
    for (int i=0; i<m; i++)
    {
        outx[i] = data[i][0];
        outy[i] = data[i][1];
        outz[i] = data[i][2];
    }
    
    out[0] = outx;
    out[1] = outy;
    out[2] = outz;
    
    return out;
} 

// Convert real 2D grid to two 1D arrays
std::array<double*, 2> GDataHandler::cppToCUDA_2DGrid()
{
    std::vector<std::array<double, 2>> data = dh.readGrid2D();
    std::array<double*, 2> out;
    
    int m = data.size();
    
    double outx[m];
    double outy[m];
    
    for (int i=0; i<m; i++)
    {
        outx[i] = data[i][0];
        outy[i] = data[i][1];
    }
    
    out[0] = outx;
    out[1] = outy;
    
    return out;
}  

// Convert J current to CUDA arrays
std::array<cuDoubleComplex*, 3> GDataHandler::cppToCUDA_Js()
{
    std::vector<std::array<std::complex<double>, 3>> data = dh.read_Js();
    std::array<cuDoubleComplex*, 3> out;
    
    int m = data.size();
    
    cuDoubleComplex outx[m];
    cuDoubleComplex outy[m];
    cuDoubleComplex outz[m];
    
    for (int i=0; i<m; i++)
    {
        outx[i] = make_cuDoubleComplex(data[i][0].real(), data[i][0].imag());
        outx[i] = make_cuDoubleComplex(data[i][1].real(), data[i][1].imag());
        outx[i] = make_cuDoubleComplex(data[i][2].real(), data[i][2].imag());
    }
    
    out[0] = outx;
    out[1] = outy;
    out[2] = outz;
    
    return out;
} 

// Convert M current to CUDA arrays
std::array<cuDoubleComplex*, 3> GDataHandler::cppToCUDA_Ms()
{
    std::vector<std::array<std::complex<double>, 3>> data = dh.read_Ms();
    std::array<cuDoubleComplex*, 3> out;
    
    int m = data.size();
    
    cuDoubleComplex outx[m];
    cuDoubleComplex outy[m];
    cuDoubleComplex outz[m];
    
    for (int i=0; i<m; i++)
    {
        outx[i] = make_cuDoubleComplex(data[i][0].real(), data[i][0].imag());
        outx[i] = make_cuDoubleComplex(data[i][1].real(), data[i][1].imag());
        outx[i] = make_cuDoubleComplex(data[i][2].real(), data[i][2].imag());
    }
    
    out[0] = outx;
    out[1] = outy;
    out[2] = outz;
    
    return out;
} 

// Because cannot return raw array, pass and void 

void GDataHandler::cppToCUDA_area(double *out)
{
    std::vector<double> area = dh.readArea();
    int m = area.size();
    
    for (int i=0; i<m; i++)
    {
        out[i] = area[i];
    }
}

std::array<double*, 3> GDataHandler::cppToCUDA_3Dnormals()
{
    std::vector<std::array<double, 3>> data = dh.readNormals();
    std::array<double*, 3> out;
    
    int m = data.size();
    
    double outx[m];
    double outy[m];
    double outz[m];
    
    for (int i=0; i<m; i++)
    {
        outx[i] = data[i][0];
        outy[i] = data[i][1];
        outz[i] = data[i][2];
    }
    
    out[0] = outx;
    out[1] = outy;
    out[2] = outz;
    
    return out;
} 

// Convert complex CUDA grid to complex cpp grid
std::vector<std::array<std::complex<double>, 3>> GDataHandler::CUDAToCpp_C(std::array<cuDoubleComplex*, 3> CUDA_C)
{
    std::vector<std::array<std::complex<double>, 3>> out;
    int n = sizeof(CUDA_C[0])/sizeof(CUDA_C[0][0]);
    
    for (int i=0; i<n; i++)
    {
        std::array<std::complex<double>, 3> arr;

        std::complex<double> x(cuCreal(CUDA_C[0][i]), cuCimag(CUDA_C[0][i]));
        std::complex<double> y(cuCreal(CUDA_C[1][i]), cuCimag(CUDA_C[1][i]));
        std::complex<double> z(cuCreal(CUDA_C[2][i]), cuCimag(CUDA_C[2][i]));
        
        arr[0] = x;
        arr[1] = y;
        arr[2] = z;
        
        out.push_back(arr);
    }
    return out;
}

// Convert real CUDA grid to real cpp grid
std::vector<std::array<double, 3>> GDataHandler::CUDAToCpp_R(std::array<double*, 3> CUDA_R)
{
    std::vector<std::array<double, 3>> out;
    int n = sizeof(CUDA_R[0])/sizeof(CUDA_R[0][0]);
    
    for (int i=0; i<n; i++)
    {
        std::array<double, 3> arr;
        
        arr[0] = CUDA_R[0][i];
        arr[1] = CUDA_R[1][i];
        arr[2] = CUDA_R[2][i];
        
        out.push_back(arr);
    }
    return out;
}
