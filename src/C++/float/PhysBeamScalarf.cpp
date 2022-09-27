#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <string>
#include <iterator>
#include <thread>
#include <array>

#include "PropagationScalarf.h"
#include "DataHandlerf.h"

/* This program calculates the PO propagation between a source and a target plane in scalar (i.e. incoherent) formalism.
 * 
 * In order to run, the presence of the following .txt files in the POPPy/src/C++/input/ is required:
 * - rFs.txt the real part of the source field illuminating target
 * - iFs.txt the imaginary part of the source field illuminating target
 *
 * - grid_s_(x,y,z).txt the source x,y,z grids
 * - A_s the source area elements corresponding to points x,y,z
 * 
 * - grid_t_(x,y,z).txt the target x,y,z grids
 * 
 * 
 * Author: Arend Moerman
 * For questions, contact: arendmoerman@gmail.com
 */

int main(int argc, char *argv [])
{
    int numThreads  = atoi(argv[1]); // Number of CPU threads to use
    float k        = atof(argv[2]); // Wavenumber of field to be propagated
    float eps      = atof(argv[3]); // Relative permittivity of source medium
    
    // Initialize timer to assess performance
    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end; 
    
    
    std::string source = "s"; 
    std::string target = "t"; 
    
    DataHandlerf handler;
    
    std::vector<float> source_area = handler.readArea();
    
    std::vector<std::array<float, 3>> grid_source = handler.readGrid3D(source);
    std::vector<std::array<float, 3>> grid_target = handler.readGrid3D(target);

    int gridsize_s = grid_source.size();
    int gridsize_t = grid_target.size();

    std::vector<std::complex<float>> field_s = handler.readScalarField();
    
    PropagationScalarf prop(k, numThreads, gridsize_s, gridsize_t, eps);
    // Start timer
    
    begin = std::chrono::steady_clock::now();
    std::cout << "Calculating fields on target..." << std::endl;
    
    prop.parallelProp(grid_target, grid_source, field_s, source_area);
    prop.joinThreads();

    std::vector<std::complex<float>> Ft = prop.field_container;
    
    std::string field_t = "Ft";
    handler.writeScalarOut(Ft, field_t);

    // End timer
    end = std::chrono::steady_clock::now();
    
    std::cout << "Calculation time = " 
        << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() 
        << " [s]\n" << std::endl;

    return 0;
}
 
