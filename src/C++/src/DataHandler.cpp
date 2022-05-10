#include "DataHandler.h"

/**
 * Read from .txt file.
 * 
 * @param file An ifstream object to the .txt file to be read.
 * @return data Container of doubles containing the contents of .txt file.
 */
std::vector<double> DataHandler::readFile(std::ifstream &file)
{
    std::string value;
    std::vector<double> data;
    
    if (file)
    {
        while (file >> value)
        {
            double val = atof(value.c_str());
            data.push_back(val);
        }
    }
    
    return data;
}

/**
 * Read a position grid in 3D Eucliean space.
 * 
 * @param mode Optical element to read. 
 *    mode = 0 corresponds to the initial beam grid.
 *    mode = 1,2,3,...,N-1 corresponds to optical elements.
 *    mode = N corresponds to the terminating surface.
 * @return grid Container of length 3 with at each element the position grid in x,y,z respectively.
 */
std::vector<std::vector<double>> DataHandler::readGrid3D(int &mode) 
{
    std::vector<std::vector<double>> grid;
    
    char fileName = static_cast<char>(mode);
    std::vector<std::string> xyz = {"_x", "_y", "_z"};
    
    for(int k=0; k<3; k++)
    {
        std::ifstream coord;
        coord.open("inputs/" + fileName + xyz[k] + ".txt");
        grid.push_back(readFile(coord));
    
        coord.close();
    }
    
    return grid;
}

/**
 * Read an input beam defined in 3D Euclidean space.
 * 
 * @param fileName Name of input beam.
 * @return beam Container containing complex numbers.
 */
std::vector<std::complex<double>> DataHandler::readBeamInit(std::string &fileName)
{   
    std::vector<std::vector<std::complex<double>>> beam;
    
    std::ifstream r_stream;
    std::ifstream i_stream;

    r_stream.open("inputs/" + fileName + "_r.txt");
    i_stream.open("inputs/" + fileName + "_i.txt");
    
    std::vector<double> beam_r = readFile(r_stream);
    std::vector<double> beam_i = readFile(i_stream);

    r_stream.close();
    i_stream.close();
    
    for (int i=0; i<beam_r.size(); i++)
    {
        std::complex<double> z(beam_r[i], beam_i[i]);
        beam.push_back(z);
    }
    
    return beam;
}

/**
 * Read area of discrete element of optical element.
 * 
 * @param fileName Name of file containing grid of areas.
 * @return are Container with areas corresponding to spatial grid.
 */
std::vector<double> DataHandler::readArea(std::string &fileName)
{
    std::ifstream a_stream;
    
    a_stream.open("inputs/" + fileName + ".txt");
    std::vector<double> area = readFile(a_stream);
    
    return area;
}

/**
 * Read components of normal vectors to optical elements.
 * 
 * @param mode Optical element to read. 
 *    mode = 0 corresponds to the initial beam grid.
 *    mode = 1,2,3,...,N-1 corresponds to optical elements.
 *    mode = N corresponds to the terminating surface.
 * @return grid Container of length 3 with at each element the components nx,ny,nz respectively.
 */
std::vector<std::vector<double>> DataHandler::readNormals(int &mode)
{
    std::vector<std::vector<double>> grid;

    char fileName = static_cast<char>(mode);
    std::vector<std::string> nxyz = {"_nx", "_ny", "_nz"};
    
    for(int k=0; k<3; k++)
    {
        std::ifstream ncoord;
        ncoord.open("inputs/" + fileName + nxyz[k] + ".txt");
        grid.push_back(readFile(ncoord));
    
        ncoord.close();
    }
    
    return grid;
}

void DataHandler::writeBeam(std::vector<std::vector<std::complex<double>>> &beam, std::string &fileName)
{
    std::ofstream out_rx;
    std::ofstream out_ry;
    std::ofstream out_rz;
    
    std::ofstream out_ix;
    std::ofstream out_iy;
    std::ofstream out_iz;
    
    out_rx.open("outputs/" + fileName + "_rx.txt");
    out_ry.open("outputs/" + fileName + "_ry.txt");
    out_rz.open("outputs/" + fileName + "_rz.txt");
    
    out_ix.open("outputs/" + fileName + "_ix.txt");
    out_iy.open("outputs/" + fileName + "_iy.txt");
    out_iz.open("outputs/" + fileName + "_iz.txt");
    
    out_rx << std::setprecision(prec);
    out_ry << std::setprecision(prec);
    out_rz << std::setprecision(prec);
    
    out_ix << std::setprecision(prec);
    out_iy << std::setprecision(prec);
    out_iz << std::setprecision(prec);
    
    for(int i=0; i<beam.size(); i++)
    {
        out_rx << beam[i][0].real() << " ";
        out_ry << beam[i][1].real() << " ";
        out_rz << beam[i][2].real() << " ";
        
        out_ix << beam[i][0].imag() << " ";
        out_iy << beam[i][1].imag() << " ";
        out_iz << beam[i][2].imag() << " ";
    }
    out_rx << std::endl;
    out_ry << std::endl;
    out_rz << std::endl;
    
    out_ix << std::endl;
    out_iy << std::endl;
    out_iz << std::endl;
}



