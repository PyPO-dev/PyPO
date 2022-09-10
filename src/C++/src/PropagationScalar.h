#include <iostream>
#include <vector>
#include <complex>
#include <array>
#include <cmath>
#include <thread>
#include <iomanip>

#include "Utils.h"

#define M_PI           3.14159265358979323846  /* pi */

#ifndef __PropagationScalar_h
#define __PropagationScalar_h

class PropagationScalar
{
    double k;                   // Wavenumber
    int numThreads;             // Number of CPU threads used
    int gridsize_s;             // Flattened gridsize of source grids
    int gridsize_t;             // Flattened gridsize of target grids
    int step;                   // Number of points calculated by n-1 threads.
    
    Utils ut;
    
public:
    std::vector<std::thread> threadPool;
    std::vector<std::complex<double>> field_container;

    PropagationScalar(double k, int numThreads, int gridsize_s, int gridsize_t);

    void propagateBeam(int start, int stop,
                       const std::vector<std::array<double, 3>> &grid_target,
                       const std::vector<std::array<double, 3>> &grid_source, 
                       const std::vector<std::complex<double>> &beam_source,
                       const std::vector<double> &source_area);
    
    std::complex<double> fieldAtPoint(const std::vector<std::array<double, 3>> &grid_source,
                                      const std::vector<std::complex<double>> &beam_source,
                                      const std::array<double, 3> &point_target,
                                      const std::vector<double> &source_area);
    
    void parallelProp(const std::vector<std::array<double, 3>> &grid_target,
                      const std::vector<std::array<double, 3>> &grid_source, 
                      const std::vector<std::complex<double>> &beam_source,
                      const std::vector<double> &source_area);
    
    void joinThreads();
};
#endif
    
 
