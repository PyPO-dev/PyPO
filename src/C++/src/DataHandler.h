#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <fstream>
#include <string>
#include <iterator>
#include <iomanip>

#ifndef __DataHandler_h
#define __DataHandler_h

class DataHandler 
{
public:
    const int prec{ 16 }; // Machine precision
    
    DataHandler() {};
    std::vector<double> readFile(std::ifstream &file, double factor);
    std::vector<double> readPars();
    std::vector<std::vector<double>> readGrid3D(std::string &mode);
    std::vector<std::vector<double>> readGrid2D(int prop_mode);
    std::vector<std::vector<std::complex<double>>> read_Js();
    std::vector<std::vector<std::complex<double>>> read_Ms();
    std::vector<double> readArea();
    std::vector<std::vector<double>> readNormals();
    
    void writeOut(std::vector<std::vector<std::complex<double>>> &out, std::string &fileName);
    
};
#endif 
