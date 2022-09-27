#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <fstream>
#include <string>
#include <iterator>
#include <iomanip>

#ifndef __DataHandlerf_h
#define __DataHandlerf_h

class DataHandlerf
{
public:
    const int prec{ 8 }; // Single precision
    
    DataHandlerf() {};
    std::vector<float> readFile(std::ifstream &file, float factor);
    std::vector<float> readPars();
    std::vector<std::array<float, 3>> readGrid3D(std::string &mode);
    std::vector<std::array<float, 2>> readGrid2D();
    std::vector<std::array<std::complex<float>, 3>> read_Js();
    std::vector<std::array<std::complex<float>, 3>> read_Ms();
    std::vector<std::complex<float>> readScalarField();
    std::vector<float> readArea();
    std::vector<std::array<float, 3>> readNormals();
    
    void writeOutC(std::vector<std::array<std::complex<float>, 3>> &out, std::string &fileName);
    void writeOutR(std::vector<std::array<float, 3>> &out, std::string &fileName);
    void writeScalarOut(std::vector<std::complex<float>> &out, std::string &fileName);

};
#endif 
