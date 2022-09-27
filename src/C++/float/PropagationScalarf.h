#include <iostream>
#include <vector>
#include <complex>
#include <array>
#include <cmath>
#include <thread>
#include <iomanip>

#include "Utilsf.h"

#ifndef __PropagationScalarf_h
#define __PropagationScalarf_h

class PropagationScalarf
{
    float k;                   // Wavenumber
    int numThreads;             // Number of CPU threads used
    int gridsize_s;             // Flattened gridsize of source grids
    int gridsize_t;             // Flattened gridsize of target grids
    int step;                   // Number of points calculated by n-1 threads.
    float n_ref;
    
    float M_PIf;
    
    Utilsf ut;
    
public:
    std::vector<std::thread> threadPool;
    std::vector<std::complex<float>> field_container;

    PropagationScalarf(float k, int numThreads, int gridsize_s, int gridsize_t, float eps);

    void propagateBeam(int start, int stop,
                       const std::vector<std::array<float, 3>> &grid_target,
                       const std::vector<std::array<float, 3>> &grid_source, 
                       const std::vector<std::complex<float>> &beam_source,
                       const std::vector<float> &source_area);
    
    std::complex<float> fieldAtPoint(const std::vector<std::array<float, 3>> &grid_source,
                                      const std::vector<std::complex<float>> &beam_source,
                                      const std::array<float, 3> &point_target,
                                      const std::vector<float> &source_area);
    
    void parallelProp(const std::vector<std::array<float, 3>> &grid_target,
                      const std::vector<std::array<float, 3>> &grid_source, 
                      const std::vector<std::complex<float>> &beam_source,
                      const std::vector<float> &source_area);
    
    void joinThreads();
};
#endif
    
 
