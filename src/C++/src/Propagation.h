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
    int gridsize_i;
    int gridsize_h;
    int gridsize_e;
    int gridsize_sec;
    int step_h;
    int step_e;
    int step_sec;
    
    double thres;
    
    std::complex<double> j;
    std::complex<double> z0;
    
    std::vector<double> n0;
    std::vector<double> p0;
    double source_area;
    
public:
    
    std::vector<std::thread> threadPool;
    std::vector<std::vector<std::complex<double>>> beam_container_e;
    std::vector<std::vector<std::complex<double>>> beam_container_h;
    
    std::vector<std::vector<std::complex<double>>> j_1_container;
    std::vector<std::vector<std::complex<double>>> m_1_container;
    
    std::vector<std::vector<std::complex<double>>> j_2_container;
    std::vector<std::vector<std::complex<double>>> m_2_container;

    Propagation(double k, int numThreads, 
                int gridsize_i, int gridsize_h, int gridsize_e, int gridsize_sec, 
                std::vector<double> n0, std::vector<double> p0, 
                double &source_area, double thres);
    
    Utils ut;
    
    void propagateBeamFoc(int start, int stop,
                       const std::vector<std::vector<double>> &grid_target,
                       const std::vector<std::vector<double>> &grid_source,
                       const std::vector<double> &source_area);

    void propagateBeam(int start, int stop,
                       const std::vector<std::vector<double>> &grid_target,
                       const std::vector<std::vector<double>> &grid_source, 
                       const std::vector<double> &source_area,
                       const std::vector<std::vector<double>> &ell_n);

    void propagateBeamInit(int start, int stop,
                       const std::vector<std::vector<double>> &grid_target,
                       const std::vector<std::vector<double>> &grid_source, 
                       const std::vector<std::complex<double>> &beam_source,
                       const std::vector<std::vector<double>> &hyp_n);
    
    std::vector<std::vector<std::complex<double>>> fieldAtFoc(const std::vector<std::vector<double>> &grid_source,
                                      const std::vector<double> &point_target,
                                      const std::vector<double> &source_area);

    std::vector<std::vector<std::complex<double>>> fieldAtPoint(const std::vector<std::vector<double>> &grid_source,
                                      const std::vector<double> &point_target,
                                      const std::vector<double> &source_area,
                                      const int start);

    std::vector<std::vector<std::complex<double>>> fieldAtPointInit(const std::vector<std::vector<double>> &grid_source,
                                      const std::vector<std::complex<double>> &beam_source,
                                      const std::vector<double> &point_target,
                                      const double max_beam);
    
    void parallelPropFoc(const std::vector<std::vector<double>> &grid_target,
                      const std::vector<std::vector<double>> &grid_source, 
                      const std::vector<double> &source_area);

    void parallelProp(const std::vector<std::vector<double>> &grid_target,
                      const std::vector<std::vector<double>> &grid_source, 
                      const std::vector<double> &source_area,
                      const std::vector<std::vector<double>> &ell_n);

    void parallelPropInit(const std::vector<std::vector<double>> &grid_target,
                      const std::vector<std::vector<double>> &grid_source, 
                      const std::vector<std::complex<double>> &beam_source,
                      const std::vector<std::vector<double>> &hyp_n);
    
    void joinThreads();
    void emptyContainer();
};
#endif
    
