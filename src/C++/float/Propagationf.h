#include <iostream>
#include <vector>
#include <complex>
#include <array>
#include <cmath>
#include <thread>
#include <iomanip>
#include <algorithm> 

#include "Utilsf.h"

#ifndef __Propagationf_h
#define __Propagationf_h

class Propagationf 
{
    float k;                   // Wavenumber
    int numThreads;             // Number of CPU threads used
    int gridsize_s;             // Flattened gridsize of source grids
    int gridsize_t;             // Flattened gridsize of target grids

    int step;                   // Number of points calculated by n-1 threads.
                                // Thread n gets slightly different amount, possibly
    float t_direction;         // Time direction. If -1, propagate field forward in time
                                // If 1, propagate backwards in time
    float thres;
    float EPS;
    
    int toPrint;
    
    float M_PIf;
    
    float C_L;
    float MU_0;
    float EPS_VAC;
    float ZETA_0_INV;
    
    std::complex<float> j;
    std::complex<float> z0;
    
    std::array<std::array<float, 3>, 3> eye;
    
public:
    
    std::vector<std::thread> threadPool;
    std::vector<std::array<std::complex<float>, 3>> Et_container;
    std::vector<std::array<std::complex<float>, 3>> Ht_container;
    
    std::vector<std::array<std::complex<float>, 3>> Jt_container;
    std::vector<std::array<std::complex<float>, 3>> Mt_container;
    
    std::vector<std::array<float, 3>> Pr_container;

    Propagationf(float k, int numThreads, int gridsize_s, int gridsize_t, float thres, float epsilon, float t_direction, int toPrint);
    
    Utilsf ut;

    // Functions for propagating fields between two surfaces
    void propagateBeam(int start, int stop,
                       const std::vector<std::array<float, 3>> &grid_target,
                       const std::vector<std::array<float, 3>> &grid_source, 
                       const std::vector<std::array<float, 3>> &norm_target, 
                       const std::vector<std::array<std::complex<float>, 3>> &Js,
                       const std::vector<std::array<std::complex<float>, 3>> &Ms,
                       const std::vector<float> &source_area);

    std::array<std::array<std::complex<float>, 3>, 2> fieldAtPoint(const std::vector<std::array<float, 3>> &grid_source,
                                      const std::vector<std::array<std::complex<float>, 3>> &Js,
                                      const std::vector<std::array<std::complex<float>, 3>> &Ms,
                                      const std::array<float, 3> &point_target,
                                      const std::vector<float> &source_area,
                                      const int start);

    void parallelProp(const std::vector<std::array<float, 3>> &grid_target,
                      const std::vector<std::array<float, 3>> &grid_source, 
                      const std::vector<std::array<float, 3>> &norm_target,
                      const std::vector<std::array<std::complex<float>, 3>> &Js,
                      const std::vector<std::array<std::complex<float>, 3>> &Ms,
                      const std::vector<float> &source_area);
    
    // Functions for calculating angular far-field from reflector directly - no phase term
    void calculateFarField(int start, int stop,
                      const std::vector<std::array<float, 2>> &grid_ff,
                      const std::vector<std::array<float, 3>> &grid_source, 
                      const std::vector<std::array<std::complex<float>, 3>> &Js,
                      const std::vector<std::array<std::complex<float>, 3>> &Ms,
                      const std::vector<float> &source_area);
    
    std::array<std::complex<float>, 3> farfieldAtPoint(const std::vector<std::array<float, 3>> &grid_source,
                                      const std::vector<std::array<std::complex<float>, 3>> &Js,
                                      const std::vector<std::array<std::complex<float>, 3>> &Ms,
                                      const std::array<float, 3> &point_ff,
                                      const std::vector<float> &source_area,
                                      const int start);
    
    void parallelFarField(const std::vector<std::array<float, 2>> &grid_ff,
                      const std::vector<std::array<float, 3>> &grid_source, 
                      const std::vector<std::array<std::complex<float>, 3>> &Js,
                      const std::vector<std::array<std::complex<float>, 3>> &Ms,
                      const std::vector<float> &source_area);

    void joinThreads();
};
#endif
    
