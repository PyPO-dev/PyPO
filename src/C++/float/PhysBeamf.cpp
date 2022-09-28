#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <string>
#include <iterator>
#include <thread>
#include <array>

#include "Propagationf.h"
#include "DataHandlerf.h"

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
    float k        = atof(argv[2]); // Wavenumber of field to be propagated
    float thres    = atof(argv[3]); // Threshold in dB for propagation performance
    int toPrint     = atoi(argv[4]); // 0 for printing J and M, 1 for E and H and 2 for all fields
    
    float epsilon  = atof(argv[5]); // Relative electric permeability
    int prop_mode   = atoi(argv[6]); // Whether to propagate to surface or to far-field
    float t_direction = atof(argv[7]); // Whether to propagate forward or back in time

    std::string source = "s"; 
    std::string target = "t"; 
    
    DataHandlerf handler;
    
    std::vector<float> source_area = handler.readArea();
    
    std::vector<std::array<float, 3>> grid_source = handler.readGrid3D(source);
    std::vector<std::array<float, 3>> grid_target3D;
    std::vector<std::array<float, 2>> grid_target2D;
    std::vector<std::array<float, 3>> norm_target;
    
    int gridsize_t;
    
    if (prop_mode == 0)
    {
        grid_target3D = handler.readGrid3D(target);
        norm_target = handler.readNormals();
        
        gridsize_t = grid_target3D.size();
    }
    
    else if (prop_mode == 1)
    {
        grid_target2D = handler.readGrid2D();
        
        gridsize_t = grid_target2D.size();
    }

    int gridsize_s = grid_source.size();

    std::vector<std::array<std::complex<float>, 3>> Js = handler.read_Js();
    std::vector<std::array<std::complex<float>, 3>> Ms = handler.read_Ms();
    
    Propagationf prop(k, numThreads, gridsize_s, gridsize_t, thres, epsilon, t_direction, toPrint);

    if (prop_mode == 0) {prop.parallelProp(grid_target3D, grid_source, norm_target, Js, Ms, source_area);}
    else if (prop_mode == 1) {prop.parallelFarField(grid_target2D, grid_source, Js, Ms, source_area);}
    
    prop.joinThreads();
    
    if (toPrint == 0)
    {
        std::vector<std::array<std::complex<float>, 3>> Jt = prop.Jt_container;
        std::vector<std::array<std::complex<float>, 3>> Mt = prop.Mt_container;
    
        std::string Jt_file = "Jt";
        std::string Mt_file = "Mt";
        handler.writeOutC(Jt, Jt_file);
        handler.writeOutC(Mt, Mt_file);
    }
    
    else if (toPrint == 1)
    {
        std::vector<std::array<std::complex<float>, 3>> Et = prop.Et_container;
        std::vector<std::array<std::complex<float>, 3>> Ht = prop.Ht_container;
    
        std::string Et_file = "Et";
        std::string Ht_file = "Ht";
        handler.writeOutC(Et, Et_file);
        handler.writeOutC(Ht, Ht_file);
    }
    
    else if (toPrint == 2)
    {
        std::vector<std::array<std::complex<float>, 3>> Jt = prop.Jt_container;
        std::vector<std::array<std::complex<float>, 3>> Mt = prop.Mt_container;
    
        std::string Jt_file = "Jt";
        std::string Mt_file = "Mt";
        handler.writeOutC(Jt, Jt_file);
        handler.writeOutC(Mt, Mt_file);
        
        std::vector<std::array<std::complex<float>, 3>> Et = prop.Et_container;
        std::vector<std::array<std::complex<float>, 3>> Ht = prop.Ht_container;
    
        std::string Et_file = "Et";
        std::string Ht_file = "Ht";
        handler.writeOutC(Et, Et_file);
        handler.writeOutC(Ht, Ht_file);
    }
    
    else if (toPrint == 3)
    {
        std::vector<std::array<float, 3>> Pr = prop.Pr_container;
        std::vector<std::array<std::complex<float>, 3>> Et = prop.Et_container;
        std::vector<std::array<std::complex<float>, 3>> Ht = prop.Ht_container;
        
        std::string Pr_file = "Pr";
        std::string Et_file = "Et";
        std::string Ht_file = "Ht";
        handler.writeOutR(Pr, Pr_file);
        handler.writeOutC(Et, Et_file);
        handler.writeOutC(Ht, Ht_file);
    }
    return 0;
}
