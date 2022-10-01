#include "GDataHandlerf.h"

// CUDA specific data functions

// Convert real 3D grid to three 1D arrays
std::array<float*, 3> GDataHandlerf::cppToCUDA_3DGrid(std::string &mode)
{
    std::vector<std::array<float, 3>> data = this->dh.readGrid3D(mode);
    std::array<float*, 3> out;
    
    int m = data.size();
    
    float *outx = new float[m];
    float *outy = new float[m];
    float *outz = new float[m];
    
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
std::array<float*, 2> GDataHandlerf::cppToCUDA_2DGrid()
{
    std::vector<std::array<float, 2>> data = dh.readGrid2D();
    std::array<float*, 2> out;
    
    int m = data.size();
    
    float *outx = new float[m];
    float *outy = new float[m];
    
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
std::array<cuFloatComplex*, 3> GDataHandlerf::cppToCUDA_Js()
{
    std::vector<std::array<std::complex<float>, 3>> data = dh.read_Js();
    std::array<cuFloatComplex*, 3> out;
    
    int m = data.size();
    
    cuFloatComplex *outx = new cuFloatComplex[m];
    cuFloatComplex *outy = new cuFloatComplex[m];
    cuFloatComplex *outz = new cuFloatComplex[m];
    
    for (int i=0; i<m; i++)
    {
        outx[i] = make_cuFloatComplex(data[i][0].real(), data[i][0].imag());
        outy[i] = make_cuFloatComplex(data[i][1].real(), data[i][1].imag());
        outz[i] = make_cuFloatComplex(data[i][2].real(), data[i][2].imag());
    }
    
    out[0] = outx;
    out[1] = outy;
    out[2] = outz;
    
    return out;
} 

// Convert M current to CUDA arrays
std::array<cuFloatComplex*, 3> GDataHandlerf::cppToCUDA_Ms()
{
    std::vector<std::array<std::complex<float>, 3>> data = dh.read_Ms();
    std::array<cuFloatComplex*, 3> out;
    
    int m = data.size();
    
    cuFloatComplex *outx = new cuFloatComplex[m];
    cuFloatComplex *outy = new cuFloatComplex[m];
    cuFloatComplex *outz = new cuFloatComplex[m];
    
    for (int i=0; i<m; i++)
    {
        outx[i] = make_cuFloatComplex(data[i][0].real(), data[i][0].imag());
        outy[i] = make_cuFloatComplex(data[i][1].real(), data[i][1].imag());
        outz[i] = make_cuFloatComplex(data[i][2].real(), data[i][2].imag());
    }
    
    out[0] = outx;
    out[1] = outy;
    out[2] = outz;
    
    return out;
} 

// Because cannot return raw array, pass and void 

void GDataHandlerf::cppToCUDA_area(float *out)
{
    std::vector<float> area = dh.readArea();
    int m = area.size();
    
    for (int i=0; i<m; i++)
    {
        out[i] = area[i];
    }
}

std::array<float*, 3> GDataHandlerf::cppToCUDA_3Dnormals()
{
    std::vector<std::array<float, 3>> data = dh.readNormals();
    std::array<float*, 3> out;
    
    int m = data.size();
    
    float *outx = new float[m];
    float *outy = new float[m];
    float *outz = new float[m];
    
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
std::vector<std::array<std::complex<float>, 3>> GDataHandlerf::CUDAToCpp_C(std::array<cuFloatComplex*, 3> CUDA_C, int size)
{
    std::vector<std::array<std::complex<float>, 3>> out;

    for (int i=0; i<size; i++)
    {
        std::array<std::complex<float>, 3> arr;

        std::complex<float> x(cuCrealf(CUDA_C[0][i]), cuCimagf(CUDA_C[0][i]));
        std::complex<float> y(cuCrealf(CUDA_C[1][i]), cuCimagf(CUDA_C[1][i]));
        std::complex<float> z(cuCrealf(CUDA_C[2][i]), cuCimagf(CUDA_C[2][i]));

        arr[0] = x;
        arr[1] = y;
        arr[2] = z;
        
        out.push_back(arr);
    }
    return out;
}

// Convert real CUDA grid to real cpp grid
std::vector<std::array<float, 3>> GDataHandlerf::CUDAToCpp_R(std::array<float*, 3> CUDA_R, int size)
{
    std::vector<std::array<float, 3>> out;
    
    for (int i=0; i<size; i++)
    {
        std::array<float, 3> arr;
        
        arr[0] = CUDA_R[0][i];
        arr[1] = CUDA_R[1][i];
        arr[2] = CUDA_R[2][i];
        
        out.push_back(arr);
    }
    return out;
}
