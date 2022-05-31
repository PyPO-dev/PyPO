#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <string>
#include <iterator>
#include <thread>

#include "Propagation.h"
#include "DataHandler.h"

/* This program calculates the PO propagation between a source and a target plane.
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
    int numThreads  = atoi(argv[1]); // Number of CPU threads to use
    double k        = atof(argv[2]); // Wavenumber of field to be propagated
    double thres    = atof(argv[3]); // Threshold in dB for propagation performance
    int toPrint     = atoi(argv[4]); // 0 for printing J and M, 1 for E and H and 2 for all fields
    
    // Initialize timer to assess performance
    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end; 
    
    
    std::string source = "s"; 
    std::string target = "t"; 
    
    DataHandler handler;
    
    std::vector<double> source_area = handler.readArea();
    
    std::vector<std::vector<double>> grid_source = handler.readGrid3D(source);
    std::vector<std::vector<double>> grid_target = handler.readGrid3D(target);

    std::vector<std::vector<double>> norm_target = handler.readNormals();
    
    std::vector<std::vector<std::complex<double>>> Js = handler.read_Js();
    std::vector<std::vector<std::complex<double>>> Ms = handler.read_Ms();
    
    int gridsize_s = grid_source[0].size();
    int gridsize_t = grid_target[0].size();

    Propagation prop(k, numThreads, gridsize_s, gridsize_t, thres);
        
    // Start timer
    
    begin = std::chrono::steady_clock::now();
    std::cout << "Calculating beam on target..." << std::endl;
    prop.parallelProp(grid_target, grid_source, norm_target, Js, Ms, source_area);
    prop.joinThreads();
    
    if (toPrint == 0)
    {
        std::vector<std::vector<std::complex<double>>> Jt = prop.Jt_container;
        std::vector<std::vector<std::complex<double>>> Mt = prop.Mt_container;
    
        std::string Jt_file = "Jt";
        std::string Mt_file = "Mt";
        handler.writeOut(Jt, Jt_file);
        handler.writeOut(Mt, Mt_file);
    }
    
    else if (toPrint == 1)
    {
        std::vector<std::vector<std::complex<double>>> Et = prop.Et_container;
        std::vector<std::vector<std::complex<double>>> Ht = prop.Ht_container;
    
        std::string Et_file = "Et";
        std::string Ht_file = "Ht";
        handler.writeOut(Et, Et_file);
        handler.writeOut(Ht, Ht_file);
    }
    
    else if (toPrint == 2)
    {
        std::vector<std::vector<std::complex<double>>> Jt = prop.Jt_container;
        std::vector<std::vector<std::complex<double>>> Mt = prop.Mt_container;
    
        std::string Jt_file = "Jt";
        std::string Mt_file = "Mt";
        handler.writeOut(Jt, Jt_file);
        handler.writeOut(Mt, Mt_file);
        
        std::vector<std::vector<std::complex<double>>> Et = prop.Et_container;
        std::vector<std::vector<std::complex<double>>> Ht = prop.Ht_container;
    
        std::string Et_file = "Et";
        std::string Ht_file = "Ht";
        handler.writeOut(Et, Et_file);
        handler.writeOut(Ht, Ht_file);
    }
    
    // End timer
    end = std::chrono::steady_clock::now();
    
    std::cout << "Calculation time = " 
        << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() 
        << " [s]\n" << std::endl;

    return 0;
}
