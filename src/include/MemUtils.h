#include <iostream>
#include <vector>

#ifndef __MemUtils_h
#define __MemUtils_h
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

/*! \file MemUtils.h
    \brief Utility class for CUDA memory allocations.
*/


/**
 * Check CUDA API error status of call.
 *
 * Wrapper for finding errors in CUDA API calls.
 *
 * @param code The errorcode returned from failed API call.
 * @param file The file in which failure occured.
 * @param line The line in file in which error occured.
 * @param abort Exit code upon error.
 */
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/* Utility class for memory allocations/copies between CUDA and host.
 */
class MemUtils
{
    public:
        inline std::vector<float*> cuMallFloat(int &n, int &size);
        inline std::vector<float*> cuMallFloatStack(int &n, int &size);
        inline std::vector<cuFloatComplex*> cuMallComplex(int &n, int &size);
        inline std::vector<cuFloatComplex*> cuMallComplexStack(int &n, int &size);
        
        inline void cuMemCpFloat(std::vector<float*> vecFloat, std::vector<float*> vecData, int &size, bool H2D = true);
        inline void cuMemCpComplex(std::vector<cuFloatComplex*> vecFloat, std::vector<cuFloatComplex*> vecData, int &size, bool H2D = true);

        inline void deallocFloatHost(std::vector<float*> vecFloat);
        inline void deallocComplexHost(std::vector<cuFloatComplex*> vecFloat);
};

inline std::vector<float*> MemUtils::cuMallFloat(int &n, int &size)
{
    std::vector<float*> out(n, nullptr);
    for(int i=0; i<n; i++)
    {
        float *p;
        gpuErrchk( cudaMalloc((void**)&p, size * sizeof(float)) );
        out[i] = p;
    }
    return out;
}

inline std::vector<float*> MemUtils::cuMallFloatStack(int &n, int &size)
{
    std::vector<float*> out(n, nullptr);
    for(int i=0; i<n; i++)
    {
        float *p = new float[size];
        out[i] = p;
    }
    return out;
}

inline std::vector<cuFloatComplex*> MemUtils::cuMallComplex(int &n, int &size)
{
    std::vector<cuFloatComplex*> out(n, nullptr);
    for(int i=0; i<n; i++)
    {
        cuFloatComplex *p;
        gpuErrchk( cudaMalloc((void**)&p, size * sizeof(cuFloatComplex)) );
        out[i] = p;
    }
    return out;
}

inline std::vector<cuFloatComplex*> MemUtils::cuMallComplexStack(int &n, int &size)
{
    std::vector<cuFloatComplex*> out(n, nullptr);
    for(int i=0; i<n; i++)
    {
        cuFloatComplex *p = new cuFloatComplex[size];
        out[i] = p;
    }
    return out;
}

inline void MemUtils::cuMemCpFloat(std::vector<float*> vecFloat, std::vector<float*> vecData, int &size, bool H2D)
{
    int n = vecFloat.size();
    for(int i=0; i<n; i++)
    {
        if(H2D) {gpuErrchk( cudaMemcpy(vecFloat[i], vecData[i], size * sizeof(float), cudaMemcpyHostToDevice) );}
        
        else {gpuErrchk( cudaMemcpy(vecFloat[i], vecData[i], size * sizeof(float), cudaMemcpyDeviceToHost) );}
    }
}

inline void MemUtils::cuMemCpComplex(std::vector<cuFloatComplex*> vecFloat, std::vector<cuFloatComplex*> vecData, int &size, bool H2D)
{
    int n = vecFloat.size();
    for(int i=0; i<n; i++)
    {
        if(H2D) {gpuErrchk( cudaMemcpy(vecFloat[i], vecData[i], size * sizeof(cuFloatComplex), cudaMemcpyHostToDevice) );}
        
        else {gpuErrchk( cudaMemcpy(vecFloat[i], vecData[i], size * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost) );}
    }
}

inline void MemUtils::deallocComplexHost(std::vector<cuFloatComplex*> vecFloat)
{    
    int n = vecFloat.size();
    for(int i=0; i<n; i++)
    {
        delete vecFloat[i];
    }
}

inline void MemUtils::deallocFloatHost(std::vector<float*> vecFloat)
{    
    int n = vecFloat.size();
    for(int i=0; i<n; i++)
    {
        delete vecFloat[i];
    }
}
#endif
