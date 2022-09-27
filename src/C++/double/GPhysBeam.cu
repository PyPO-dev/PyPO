#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <string>
#include <iterator>
#include <array>

#include <cuda.h>
#include <cuComplex.h>

//#include "Propagation.cu"

#include "GDataHandler.h"

#include "GUtils.h"

/* This program calculates the PO propagation between a source and a target plane.
 * NOTE: This file contains the CUDA version of PhysBeam
 * 
 * In order to run, the presence of the following .txt files in the POPPy/src/C++/input/ is required:
 * - s_Jr_(x,y,z).txt the real x,y,z components of the source electric current distribution
 * - s_Ji_(x,y,z).txt the imaginary x,y,z components of the source electric current distribution 
 * - s_Mr_(x,y,z).txt the real x,y,z components of the source magnetic current distribution
 * - s_Mi_(x,y,z).txt the imaginary x,y,z components of the source magnetic current distribution
 *
 * - s_(x,y,z).txt the source x,y,z grids
 * - s_n(x,y,z).txt the source nx,ny,nz normal grids
 * - A_s the source area elements corresponding to points x,y,z
 * 
 * - t_(x,y,z).txt the target x,y,z grids
 * - t_n(x,y,z).txt the target nx,ny,nz normal grids
 * 
 * 
 * Author: Arend Moerman
 * For questions, contact: arendmoerman@gmail.com
 */

  

int main(int argc, char *argv [])
{
    int numThreads  = atoi(argv[1]); // Number of GPU threads per block
    int numBlocks   = atoi(argv[2]); // Threshold in dB for propagation performance
    double k        = atof(argv[3]); // Wavenumber of field to be propagated
    int toPrint     = atoi(argv[4]); // 0 for printing J and M, 1 for E and H and 2 for all fields
    
    double epsilon  = atof(argv[5]); // Relative electric permeability
    int prop_mode   = atoi(argv[6]); // Whether to propagate to surface or to far-field
    double t_direction = atof(argv[7]); // Whether to propagate forward or back in time
    
    int gridsize_s  = atoi(argv[8]); // Source gridsize, flattened
    int gridsize_t  = atoi(argv[9]); // Target gridsize, flattened
    
    // Initialize timer to assess performance
    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end; 
    
    
    std::string source = "s"; 
    std::string target = "t"; 
    
    GDataHandler ghandler;
    
    double source_area[gridsize_s];
    
    ghandler.cppToCUDA_area(source_area);

    std::array<double*, 3> grid_source = ghandler.cppToCUDA_3DGrid(source);
    std::array<double*, 3> grid_target3D;
    std::array<double*, 2> grid_target2D;
    std::array<double*, 3> norm_target;

    if (prop_mode == 0)
    {
        grid_target3D = ghandler.cppToCUDA_3DGrid(target);
        norm_target = ghandler.cppToCUDA_3Dnormals();
    }
    
    else if (prop_mode == 1)
    {
        grid_target2D = ghandler.cppToCUDA_2DGrid();
    }

    std::array<cuDoubleComplex*, 3> Js = ghandler.cppToCUDA_Js();
    std::array<cuDoubleComplex*, 3> Ms = ghandler.cppToCUDA_Ms();
    
    /*
    Propagation prop(k, numThreads, gridsize_s, gridsize_t, thres, epsilon, t_direction, toPrint);
    //std::cout << "jelloo" << std::endl;
    // Start timer

    begin = std::chrono::steady_clock::now();
    
    if (prop_mode == 0) {prop.parallelProp(grid_target3D, grid_source, norm_target, Js, Ms, source_area);}
    else if (prop_mode == 1) {prop.parallelFarField(grid_target2D, grid_source, Js, Ms, source_area);}
    
    prop.joinThreads();
    
    if (toPrint == 0)
    {
        std::vector<std::array<std::complex<double>, 3>> Jt = prop.Jt_container;
        std::vector<std::array<std::complex<double>, 3>> Mt = prop.Mt_container;
    
        std::string Jt_file = "Jt";
        std::string Mt_file = "Mt";
        handler.writeOutC(Jt, Jt_file);
        handler.writeOutC(Mt, Mt_file);
    }
    
    else if (toPrint == 1)
    {
        std::vector<std::array<std::complex<double>, 3>> Et = prop.Et_container;
        std::vector<std::array<std::complex<double>, 3>> Ht = prop.Ht_container;
    
        std::string Et_file = "Et";
        std::string Ht_file = "Ht";
        handler.writeOutC(Et, Et_file);
        handler.writeOutC(Ht, Ht_file);
    }
    
    else if (toPrint == 2)
    {
        std::vector<std::array<std::complex<double>, 3>> Jt = prop.Jt_container;
        std::vector<std::array<std::complex<double>, 3>> Mt = prop.Mt_container;
    
        std::string Jt_file = "Jt";
        std::string Mt_file = "Mt";
        handler.writeOutC(Jt, Jt_file);
        handler.writeOutC(Mt, Mt_file);
        
        std::vector<std::array<std::complex<double>, 3>> Et = prop.Et_container;
        std::vector<std::array<std::complex<double>, 3>> Ht = prop.Ht_container;
    
        std::string Et_file = "Et";
        std::string Ht_file = "Ht";
        handler.writeOutC(Et, Et_file);
        handler.writeOutC(Ht, Ht_file);
    }
    
    else if (toPrint == 3)
    {
        std::vector<std::array<double, 3>> Pr = prop.Pr_container;
        std::vector<std::array<std::complex<double>, 3>> Et = prop.Et_container;
        std::vector<std::array<std::complex<double>, 3>> Ht = prop.Ht_container;
        
        std::string Pr_file = "Pr";
        std::string Et_file = "Et";
        std::string Ht_file = "Ht";
        handler.writeOutR(Pr, Pr_file);
        handler.writeOutC(Et, Et_file);
        handler.writeOutC(Ht, Ht_file);
    }
    
    // End timer
    end = std::chrono::steady_clock::now();
    
    std::cout << "Elapsed time: " 
        << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() 
        << " [s]\n" << std::endl;
    */
    return 0;
}
 
