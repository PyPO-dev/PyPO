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
    std::vector<double> readFile(std::ifstream &file);
    std::vector<std::vector<double>> readGrid3D(int mode);
    std::vector<std::complex<double>> readBeamInit(std::string &fileName);
    std::vector<double> readArea(std::string &fileName);
    std::vector<std::vector<double>> readNormals(int mode);
    
    void writeBeam(std::vector<std::vector<std::complex<double>>> &beam, std::string &fileName);
    
};
#endif 
