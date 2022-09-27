#include <iostream>
#include <vector>
#include <complex>
#include <array>
#include <cmath>
#include <thread>
#include <iomanip>
#include <algorithm> 

#include "Utils.h"

#define M_PI            3.14159265358979323846  /* pi */
#define C_L             2.99792458e11 // mm s^-1
#define MU_0            1.2566370614e-3 // kg mm s^-2 A^-2
#define EPS_VAC         1 / (MU_0 * C_L*C_L)
#define ZETA_0_INV      1 / (C_L * MU_0)

#ifndef __Propagation_h
#define __Propagation_h

class Propagation 
{
    double k;                   // Wavenumber
    int numThreads;             // Number of CPU threads used
    int gridsize_s;             // Flattened gridsize of source grids
    int gridsize_t;             // Flattened gridsize of target grids

    int step;                   // Number of points calculated by n-1 threads.
                                // Thread n gets slightly different amount, possibly
    double t_direction;         // Time direction. If -1, propagate field forward in time
                                // If 1, propagate backwards in time
    double thres;
    double EPS;
    
    int toPrint;
    
    
    
    std::complex<double> j;
    std::complex<double> z0;
    
    std::array<std::array<double, 3>, 3> eye;
    
public:
    
    std::vector<std::thread> threadPool;
    std::vector<std::array<std::complex<double>, 3>> Et_container;
    std::vector<std::array<std::complex<double>, 3>> Ht_container;
    
    std::vector<std::array<std::complex<double>, 3>> Jt_container;
    std::vector<std::array<std::complex<double>, 3>> Mt_container;
    
    std::vector<std::array<double, 3>> Pr_container;

    Propagation(double k, int numThreads, int gridsize_s, int gridsize_t, double thres, double epsilon, double t_direction, int toPrint);
    
    Utils ut;

    // Functions for propagating fields between two surfaces
    void propagateBeam(int start, int stop,
                       const std::vector<std::array<double, 3>> &grid_target,
                       const std::vector<std::array<double, 3>> &grid_source, 
                       const std::vector<std::array<double, 3>> &norm_target, 
                       const std::vector<std::array<std::complex<double>, 3>> &Js,
                       const std::vector<std::array<std::complex<double>, 3>> &Ms,
                       const std::vector<double> &source_area);

    std::array<std::array<std::complex<double>, 3>, 2> fieldAtPoint(const std::vector<std::array<double, 3>> &grid_source,
                                      const std::vector<std::array<std::complex<double>, 3>> &Js,
                                      const std::vector<std::array<std::complex<double>, 3>> &Ms,
                                      const std::array<double, 3> &point_target,
                                      const std::vector<double> &source_area,
                                      const int start);

    void parallelProp(const std::vector<std::array<double, 3>> &grid_target,
                      const std::vector<std::array<double, 3>> &grid_source, 
                      const std::vector<std::array<double, 3>> &norm_target,
                      const std::vector<std::array<std::complex<double>, 3>> &Js,
                      const std::vector<std::array<std::complex<double>, 3>> &Ms,
                      const std::vector<double> &source_area);
    
    // Functions for calculating angular far-field from reflector directly - no phase term
    void calculateFarField(int start, int stop,
                      const std::vector<std::array<double, 2>> &grid_ff,
                      const std::vector<std::array<double, 3>> &grid_source, 
                      const std::vector<std::array<std::complex<double>, 3>> &Js,
                      const std::vector<std::array<std::complex<double>, 3>> &Ms,
                      const std::vector<double> &source_area);
    
    std::array<std::complex<double>, 3> farfieldAtPoint(const std::vector<std::array<double, 3>> &grid_source,
                                      const std::vector<std::array<std::complex<double>, 3>> &Js,
                                      const std::vector<std::array<std::complex<double>, 3>> &Ms,
                                      const std::array<double, 3> &point_ff,
                                      const std::vector<double> &source_area,
                                      const int start);
    
    void parallelFarField(const std::vector<std::array<double, 2>> &grid_ff,
                      const std::vector<std::array<double, 3>> &grid_source, 
                      const std::vector<std::array<std::complex<double>, 3>> &Js,
                      const std::vector<std::array<std::complex<double>, 3>> &Ms,
                      const std::vector<double> &source_area);

    void joinThreads();
};
#endif
    
