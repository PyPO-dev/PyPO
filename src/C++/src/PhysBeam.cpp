#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <string>
#include <iterator>
#include <thread>

#include "Propagation.h"
#include "DataHandler.h"

/* This program calculates the far field beam pattern.
 * It needs as .txt files:
 * - initial_beam_r.txt -> text file containing the initial real beam grid, on an n x n grid.
 * - initial_beam_i.txt -> text file containing the initial imaginary beam grid, on an n x n grid.
 * - initial_(x/y/z).txt -> three text files, each containing the x,y,z coordinates of the initial beam grid
 * 
 * - hyp_(x/y/z).txt -> three text files, each containing the x,y,z coordinates of the hyperbola
 * 
 * - hyp_(nx/ny/nz).txt -> three text files, each containing the u,v,w coordinates of the normal vectors to hyperbola
 * 
 * - ell_(x/y/z).txt -> three text files, each containing the x,y,z coordinates of the ellipse
 * 
 * - ell_(nx/ny/nz).txt -> three text files, each containing the u,v,w coordinates of the normals to the ellipse
 * 
 * - sec_(x/y/z).txt -> three text files, each containing the x,y,z coordinates of the secondary
 * 
 * - pars.txt - > a text file containing k, dli, dle, n0, p0
 * 
 * NOTE: Internally, position grids and normal vector grids are stored in the following way:
 * 
 *          vector of 3 elements --> vector of gridsize elements --> x coordinate
 * 
 * The beams on the other hand are stored as such:
 * 
 *          vector of gridsize elements --> vector of 3 elememts --> x value
 * 
 * for both the e and h field.
 * 
 * Author: Arend Moerman
 * For questions, contact: arendmoerman@gmail.com
 */

int main(int argc, char *argv [])
{
    int numThreads = atoi(argv[1]);
    double thres = atof(argv[2]);
    
    // Initialize timer to assess performance
    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end; 
    
    
    std::string beamFile = "initial_beam"; 
    std::string A_h = "A_h";
    std::string A_e = "A_e";
    
    DataHandler handler;
    
    std::vector<double> pars = handler.readPars();
    double k = pars[0];

    std::vector<double> area_hyp = handler.readArea(A_h);
    std::vector<double> area_ell = handler.readArea(A_e);
    
    std::vector<std::vector<double>> grid_initial = handler.readGrid3D(0);
    std::vector<std::vector<double>> grid_hyp = handler.readGrid3D(1);
    std::vector<std::vector<double>> grid_ell = handler.readGrid3D(2);
    std::vector<std::vector<double>> grid_sec = handler.readGrid3D(3);
    
    std::vector<std::vector<double>> hyp_n = handler.readNormals(1);
    std::vector<std::vector<double>> ell_n = handler.readNormals(2);
    
    std::vector<std::complex<double>> beam_initial = handler.readBeam(beamFile);
    
    int gridsize_i = grid_initial[0].size();
    int gridsize_h = grid_hyp[0].size();
    int gridsize_e = grid_ell[0].size();
    int gridsize_sec = grid_sec[0].size();
    
    // Initialize n0 from input pars
    std::vector<double> n0;
    for( int i=0; i<3; i++)
    {
        n0.push_back(pars[i+3]);
    }
    
    // Initialize p0, initial polarisation
    std::vector<double> p0;
    for( int i=0; i<3; i++)
    {
        p0.push_back(pars[i+6]);
    }
    
    // Initialize the surface area per point for the initial grid
    double a_i = pars[1] * pars[2];
    
    Propagation prop(k, numThreads, gridsize_i, gridsize_h, gridsize_e, gridsize_sec, n0, p0, a_i, thres);
        
    // Start timer
    
    begin = std::chrono::steady_clock::now();
    std::cout << "    Calculating beam on hyperboloid..." << std::endl;
    prop.parallelPropInit(grid_hyp, grid_initial, beam_initial, hyp_n);
    prop.joinThreads();
    std::vector<std::vector<std::complex<double>>> beam_temp_e_0 = prop.beam_container_e;
    std::vector<std::vector<std::complex<double>>> beam_temp_h_0 = prop.beam_container_h;
    
    std::string hypFile_e_i = "beam_hyp_e_i";
    std::string hypFile_h_i = "beam_hyp_h_i";
    handler.writeBeam(beam_temp_e_0, hypFile_e_i);
    handler.writeBeam(beam_temp_h_0, hypFile_h_i);
    prop.emptyContainer();

    std::cout << std::endl;
    std::cout << "    Calculating beam on ellipsoid..." << std::endl;
    prop.parallelProp(grid_ell, grid_hyp, area_hyp, ell_n);
    prop.joinThreads();
    std::vector<std::vector<std::complex<double>>> beam_temp_e_1 = prop.beam_container_e;
    std::vector<std::vector<std::complex<double>>> beam_temp_h_1 = prop.beam_container_h;
    
    std::string ellFile_e_i = "beam_ell_e_i";
    std::string ellFile_h_i = "beam_ell_h_i";
    handler.writeBeam(beam_temp_e_1, ellFile_e_i);
    handler.writeBeam(beam_temp_h_1, ellFile_h_i);
    prop.emptyContainer();
    
    std::cout << std::endl;
    std::cout << "    Calculating beam on ellipsoid focal plane..." << std::endl;
    prop.parallelPropFoc(grid_sec, grid_ell, area_ell);
    prop.joinThreads();
    std::vector<std::vector<std::complex<double>>> beam_temp_e_2 = prop.beam_container_e;
    std::vector<std::vector<std::complex<double>>> beam_temp_h_2 = prop.beam_container_h;
    
    std::string secFile_e_i = "beam_sec_e_i";
    std::string secFile_h_i = "beam_sec_h_i";
    handler.writeBeam(beam_temp_e_2, secFile_e_i);
    handler.writeBeam(beam_temp_h_2, secFile_h_i);
    prop.emptyContainer();

    // End timer
    end = std::chrono::steady_clock::now();
    
    std::cout << "    Calculation time = " 
        << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() 
        << " [s]" << std::endl;

    return 0;
}
