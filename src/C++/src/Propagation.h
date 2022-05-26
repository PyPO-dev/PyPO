#include <iostream>
#include <vector>
#include <complex>
#include <array>
#include <cmath>
#include <thread>
#include <iomanip>
#include <algorithm> 

#include "Utils.h"

# define M_PI           3.14159265358979323846  /* pi */
#define C_L             2.998e11 // mm s^-1
#define MU_0            1.256e-3 // kg mm s^-2 A^-2
#define EPS_ALU         10. * 1 / (MU_0 * C_L*C_L)
#define ZETA_0_INV      1 / (C_L * MU_0)

#ifndef __Propagation_h
#define __Propagation_h

class Propagation 
{
    double k;
    int numThreads;
    int gridsize_s;
    int gridsize_t;

    int step;
    
    double thres;
    
    std::complex<double> j;
    std::complex<double> z0;
public:
    
    std::vector<std::thread> threadPool;
    std::vector<std::vector<std::complex<double>>> Et_container;
    std::vector<std::vector<std::complex<double>>> Ht_container;
    
    std::vector<std::vector<std::complex<double>>> Jt_container;
    std::vector<std::vector<std::complex<double>>> Mt_container;

    Propagation(double k, int numThreads, int gridsize_s, int gridsize_t, double thres);
    
    Utils ut;

    void propagateBeam(int start, int stop,
                       const std::vector<std::vector<double>> &grid_target,
                       const std::vector<std::vector<double>> &grid_source, 
                       const std::vector<std::vector<double>> &norm_target, 
                       const std::vector<std::vector<std::complex<double>>> &Js,
                       const std::vector<std::vector<std::complex<double>>> &Ms,
                       const std::vector<double> &source_area);

    std::vector<std::vector<std::complex<double>>> fieldAtPoint(const std::vector<std::vector<double>> &grid_source,
                                      const std::vector<std::vector<std::complex<double>>> &Js,
                                      const std::vector<std::vector<std::complex<double>>> &Ms,
                                      const std::vector<double> &point_target,
                                      const std::vector<double> &source_area,
                                      const int start);

    void parallelProp(const std::vector<std::vector<double>> &grid_target,
                      const std::vector<std::vector<double>> &grid_source, 
                      const std::vector<std::vector<double>> &norm_target,
                      const std::vector<std::vector<std::complex<double>>> &Js,
                      const std::vector<std::vector<std::complex<double>>> &Ms,
                      const std::vector<double> &source_area);

    void joinThreads();
};
#endif
    
