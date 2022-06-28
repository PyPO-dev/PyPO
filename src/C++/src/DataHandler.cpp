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

std::vector<double> DataHandler::readPars()
{
    std::ifstream pars_stream;
    pars_stream.open("input/pars.txt");
    
    std::string value;
    std::vector<double> pars;
    
    if (pars_stream)
    {
        while (pars_stream >> value)
        {
            double val = atof(value.c_str());
            pars.push_back(val);
        }
    }
    return pars;
}

/**
 * Read a position grid in 3D Eucliean space.
 * 
 * @param mode Optical element to read. 
 *    mode = s corresponds to the initial beam grid.
 *    mode = t corresponds to target grid.
 * @return grid Container of length 3 with at each element the position grid in x,y,z respectively.
 */
std::vector<std::vector<double>> DataHandler::readGrid3D(std::string &mode) 
{
    std::vector<std::vector<double>> grid;

    std::vector<std::string> xyz = {"_x", "_y", "_z"};
    
    for(int k=0; k<3; k++)
    {
        std::ifstream coord;
        coord.open("input/grid_" + mode + xyz[k] + ".txt");
        grid.push_back(readFile(coord));
    
        coord.close();
    }
    
    return grid;
}

/**
 * Read source electric current Js defined in 3D Euclidean space.
 * 
 * @return Js_xyz Container the complex xyz components of Js.
 */
std::vector<std::vector<std::complex<double>>> DataHandler::read_Js()
{   
    
    std::vector<std::vector<std::complex<double>>> Js_xyz;
    
    std::vector<std::string> xyz = {"_x", "_y", "_z"};
    
    for (int k=0; k<3; k++)
    {
        std::vector<std::complex<double>> Js;
        
        std::ifstream rJs_stream;
        std::ifstream iJs_stream;

        rJs_stream.open("input/rJs" + xyz[k] + ".txt");
        iJs_stream.open("input/iJs" + xyz[k] + ".txt");
    
        std::vector<double> rJs = readFile(rJs_stream);
        std::vector<double> iJs = readFile(iJs_stream);
        
        rJs_stream.close();
        iJs_stream.close();
    
        for (int i=0; i<rJs.size(); i++)
        {
            std::complex<double> z(rJs[i], iJs[i]);
            Js.push_back(z);
        }
        Js_xyz.push_back(Js);
    }

    return Js_xyz;
}

/**
 * Read source magnetic current Ms defined in 3D Euclidean space.
 * 
 * @return Ms_xyz Container the complex xyz components of Ms.
 */
std::vector<std::vector<std::complex<double>>> DataHandler::read_Ms()
{   
    
    std::vector<std::vector<std::complex<double>>> Ms_xyz;
    
    std::vector<std::string> xyz = {"_x", "_y", "_z"};
    
    for (int k=0; k<3; k++)
    {
        std::vector<std::complex<double>> Ms;
        
        std::ifstream rMs_stream;
        std::ifstream iMs_stream;

        rMs_stream.open("input/rMs" + xyz[k] + ".txt");
        iMs_stream.open("input/iMs" + xyz[k] + ".txt");
    
        std::vector<double> rMs = readFile(rMs_stream);
        std::vector<double> iMs = readFile(iMs_stream);
        
        rMs_stream.close();
        iMs_stream.close();
    
        for (int i=0; i<rMs.size(); i++)
        {
            std::complex<double> z(rMs[i], iMs[i]);
            Ms.push_back(z);
        }
        Ms_xyz.push_back(Ms);
    }

    return Ms_xyz;
}

/**
 * Read area of discrete element of optical element.
 * 
 * @param fileName Name of file containing grid of areas.
 * @return are Container with areas corresponding to spatial grid.
 */
std::vector<double> DataHandler::readArea()
{
    std::ifstream a_stream;
    
    a_stream.open("input/As.txt");
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
std::vector<std::vector<double>> DataHandler::readNormals()
{
    std::vector<std::vector<double>> grid;

    std::vector<std::string> nxyz = {"_nx", "_ny", "_nz"};
    
    for(int k=0; k<3; k++)
    {
        std::ifstream ncoord;
        ncoord.open("input/norm_t" + nxyz[k] + ".txt");
        grid.push_back(readFile(ncoord));
    
        ncoord.close();
    }
    
    return grid;
}

/**
 * Write J, M or E, H to file.
 * 
 * @param out Container with 3 x-y-z coordinate vectors with complex field values
 * @param fileName String containing name of field to be printed.
 *      Use "Jt", "Mt", "Et" and "Ht" for consistency.
 */

void DataHandler::writeOut(std::vector<std::vector<std::complex<double>>> &out, std::string &fileName)
{
    std::vector<std::string> xyz = {"_x", "_y", "_z"};
    
    for (int k=0; k<3; k++)
    {
    
        std::fstream out_r;
        std::fstream out_i;
    
        out_r.open("output/r" + fileName + xyz[k] + ".txt", std::fstream::out | std::fstream::trunc);
        out_i.open("output/i" + fileName + xyz[k] + ".txt", std::fstream::out | std::fstream::trunc);
        
        out_r << std::setprecision(prec);
        out_i << std::setprecision(prec);

    
        for(int i=0; i<out[k].size(); i++)
        {
            out_r << out[k][i].real() << " ";
            out_i << out[k][i].imag() << " ";
        }
        out_r << std::endl;
        out_i << std::endl;
        
        out_r.close();
        out_i.close();
    }
}



