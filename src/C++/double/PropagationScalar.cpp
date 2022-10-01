#include "PropagationScalar.h"

PropagationScalar::PropagationScalar(double k, int numThreads, int gridsize_s, int gridsize_t, double eps)
{
    this->k = k;

    this->numThreads = numThreads;
    this->gridsize_s = gridsize_s;
    this->gridsize_t = gridsize_t;
    
    this->step = gridsize_t / numThreads;
    
    threadPool.resize(numThreads);
    field_container.resize(gridsize_t);
    
    this->n_ref = std::sqrt(eps);
}

void PropagationScalar::propagateBeam(int start, int stop,
                                const std::vector<std::array<double, 3>> &grid_target,
                                const std::vector<std::array<double, 3>> &grid_source,
                                const std::vector<std::complex<double>> &beam_source,
                                const std::vector<double> &source_area)
{
    for(int i=start; i<stop; i++)
    {
        this->field_container[i] = fieldAtPoint(grid_source, beam_source, grid_target[i], source_area);
        
        if(i % 100 == 0 and start == 0 * this->step)
        {
            std::cout << i << " / " << this->step << std::endl;
        }
    }
}
    
std::complex<double> PropagationScalar::fieldAtPoint(const std::vector<std::array<double, 3>> &grid_source,
                                               const std::vector<std::complex<double>> &beam_source,
                                               const std::array<double, 3> &point_target,
                                               const std::vector<double> &source_area)
{
    std::complex<double> field(0., 0.);
    std::complex<double> j(0., 1.);
    
    double r;
    std::array<double, 3> r_vec;
    
    for(int i=0; i<gridsize_s; i++)
    {
        ut.diff(point_target, grid_source[i], r_vec);
        ut.abs(r_vec, r);    
                    
        field += - k * k * (n_ref*n_ref - 1) * beam_source[i] * exp(-j * k * r) / (4. * M_PI * r) * source_area[i];
        
    }
    return field;
}

void PropagationScalar::parallelProp(const std::vector<std::array<double, 3>> &grid_target,
                               const std::vector<std::array<double, 3>> &grid_source, 
                               const std::vector<std::complex<double>> &beam_source,
                               const std::vector<double> &source_area)
{
    int final_step; 
    
    for(int n=0; n<numThreads; n++)
    {
        if(n == (numThreads-1))
        {
            final_step = gridsize_t;
        }

        else
        {
            final_step = (n+1) * step;
        }
        
        threadPool[n] = std::thread(&PropagationScalar::propagateBeam, 
                                    this, n * step, final_step, 
                                    grid_target, grid_source, beam_source, source_area);
    }
}

void PropagationScalar::joinThreads() 
{
    for (std::thread &t : threadPool) 
    {
        if (t.joinable()) 
        {
            t.join();
        }
    }
}
