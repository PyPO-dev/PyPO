#include "DataHandler.h"

/**
 * Read from .txt file.
 * 
 * @param file An ifstream object to the .txt file to be read.
 * @return data Container of doubles containing the contents of .txt file.
 */
std::vector<double> DataHandler::readFile(std::ifstream &file, double factor = 1.)
{
    std::string value;
    std::vector<double> data;
    
    if (file)
    {
        while (file >> value)
        {
            double val = factor * atof(value.c_str());
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
 * @return grid Container of length grid with at each element a 3D array containing x,y,z respectively.
 */
std::vector<std::array<double, 3>> DataHandler::readGrid3D(std::string &mode) 
{
    std::vector<std::array<double, 3>> grid;
    
    std::vector<std::vector<double>> _grid;
    
    std::vector<std::string> xyz = {"_x", "_y", "_z"};
    
    for(int k=0; k<3; k++)
    {
        std::ifstream coord;
        coord.open("input/grid_" + mode + xyz[k] + ".txt");
        _grid.push_back(readFile(coord));
    
        coord.close();
    }
    
    for (int i=0; i<_grid[0].size(); i++)
        {
            std::array<double, 3> pos;
            
            pos[0] = _grid[0][i];
            pos[1] = _grid[1][i];
            pos[2] = _grid[2][i];
            
            grid.push_back(pos);
        }
    
    return grid;
}

/**
 * Read a position grid in 2D Eucliean space. Use for azimuth-elevation far-field.
 * 
 * @param mode Optical element to read. 
 *    mode = s corresponds to the initial beam grid.
 *    mode = t corresponds to target grid.
 * @return grid Container of length 3 with at each element the position grid in x,y,z respectively.
 */
std::vector<std::array<double, 2>> DataHandler::readGrid2D() 
{
    std::vector<std::array<double, 2>> grid;
    std::vector<std::vector<double>> _grid;

    std::array<std::string, 2> azel = {"_th", "_ph"};

    for(int k=0; k<2; k++)
    {
        std::ifstream coord;
        coord.open("input/grid_t" + azel[k] + ".txt");
        _grid.push_back(readFile(coord));
        coord.close();
    }
    
    for (int i=0; i<_grid[0].size(); i++)
        {
            std::array<double, 2> pos;
            
            pos[0] = _grid[0][i];
            pos[1] = _grid[1][i];
            
            grid.push_back(pos);
        }
    
    return grid;
}

/**
 * Read source electric current Js defined in 3D Euclidean space.
 * 
 * @return Js_xyz Container the complex xyz components of Js.
 */
std::vector<std::array<std::complex<double>, 3>> DataHandler::read_Js()
{   
    
    std::vector<std::array<std::complex<double>, 3>> Js_grid;
    
    std::vector<std::vector<std::complex<double>>> _Js_grid;
    
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
        _Js_grid.push_back(Js);
    }
    
    for (int i=0; i<_Js_grid[0].size(); i++)
        {
            std::array<std::complex<double>, 3> arr;
            
            arr[0] = _Js_grid[0][i];
            arr[1] = _Js_grid[1][i];
            arr[2] = _Js_grid[2][i];
            
            Js_grid.push_back(arr);
        }

    return Js_grid;
}

/**
 * Read source magnetic current Ms defined in 3D Euclidean space.
 * 
 * @return Ms_xyz Container the complex xyz components of Ms.
 */
std::vector<std::array<std::complex<double>, 3>> DataHandler::read_Ms()
{   
    
    std::vector<std::array<std::complex<double>, 3>> Ms_grid;
    std::vector<std::vector<std::complex<double>>> _Ms_grid;
    
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
        _Ms_grid.push_back(Ms);
    }

    for (int i=0; i<_Ms_grid[0].size(); i++)
        {
            std::array<std::complex<double>, 3> arr;
            
            arr[0] = _Ms_grid[0][i];
            arr[1] = _Ms_grid[1][i];
            arr[2] = _Ms_grid[2][i];
            
            Ms_grid.push_back(arr);
        }
    
    return Ms_grid;
}

/**
 * Read scalar field from input.
 * 
 * @return field_s Source field illuminating target.
 */
std::vector<std::complex<double>> DataHandler::readScalarField()
{
    std::vector<std::complex<double>> Fs;
    
    std::ifstream rFs_stream;
    std::ifstream iFs_stream;
    
    rFs_stream.open("input/rFs.txt");
    iFs_stream.open("input/iFs.txt");
    
    std::vector<double> rFs = readFile(rFs_stream);
    std::vector<double> iFs = readFile(iFs_stream);
    
    for (int i=0; i<rFs.size(); i++)
    {
        std::complex<double> z(rFs[i], iFs[i]);
        Fs.push_back(z);
    }
    
    return Fs;
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
std::vector<std::array<double, 3>> DataHandler::readNormals()
{
    std::vector<std::array<double, 3>> grid;
    std::vector<std::vector<double>> _grid;

    std::vector<std::string> nxyz = {"_nx", "_ny", "_nz"};
    
    for(int k=0; k<3; k++)
    {
        std::ifstream ncoord;
        ncoord.open("input/norm_t" + nxyz[k] + ".txt");
        _grid.push_back(readFile(ncoord));
    
        ncoord.close();
    }
    
    for (int i=0; i<_grid[0].size(); i++)
        {
            std::array<double, 3> arr;
            
            arr[0] = _grid[0][i];
            arr[1] = _grid[1][i];
            arr[2] = _grid[2][i];
            
            grid.push_back(arr);
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

void DataHandler::writeOutC(std::vector<std::array<std::complex<double>, 3>> &out, std::string &fileName)
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

    
        for(int i=0; i<(out.size() - 1); i++)
        //for(int i=(out.size() - 1); i>0; i--)
        {
            out_r << out[i][k].real() << std::endl;
            out_i << out[i][k].imag() << std::endl;
        }
        out_r << out[out.size() - 1][k].real();
        out_i << out[out.size() - 1][k].imag();
        
        out_r.close();
        out_i.close();
    }
}

void DataHandler::writeOutR(std::vector<std::array<double, 3>> &out, std::string &fileName)
{
    std::vector<std::string> xyz = {"_x", "_y", "_z"};
    
    for (int k=0; k<3; k++)
    {
    
        std::fstream out_r;
    
        out_r.open("output/" + fileName + xyz[k] + ".txt", std::fstream::out | std::fstream::trunc);
        
        out_r << std::setprecision(prec);

    
        for(int i=0; i<(out.size() - 1); i++)
        //for(int i=(out.size() - 1); i>0; i--)
        {
            out_r << out[i][k] << std::endl;
        }
        out_r << out[out.size() - 1][k];
        
        out_r.close();
    }
}

void DataHandler::writeScalarOut(std::vector<std::complex<double>> &out, std::string &fileName)
{
    std::fstream out_r;
    std::fstream out_i;
    
    out_r.open("output/rFt.txt", std::fstream::out | std::fstream::trunc);
    out_i.open("output/iFt.txt", std::fstream::out | std::fstream::trunc);
        
    out_r << std::setprecision(prec);
    out_i << std::setprecision(prec);

    
    for(int i=0; i<(out.size() - 1); i++)
    {
        out_r << out[i].real() << std::endl;
        out_i << out[i].imag() << std::endl;
    }
    out_r << out[out.size() - 1].real();
    out_i << out[out.size() - 1].imag();
        
    out_r.close();
    out_i.close();
}
