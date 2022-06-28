#include "Propagation.h"

Propagation::Propagation(double k, int numThreads, int gridsize_s, int gridsize_t, double thres)
{
    std::complex<double> j(0., 1.);
    std::complex<double> z0(0., 0.);
    this->j = j;
    this->z0 = z0;
    
    this->k = k;

    this->numThreads = numThreads;
    this->gridsize_s = gridsize_s;
    this->gridsize_t = gridsize_t;
    
    this->step = ceil(gridsize_t / numThreads);

    // Convert decibels to E-field values now to save log10 evaluation time
    double exponent = -1 * thres / 20.;
    this->thres = pow(10., exponent);
    
    threadPool.resize(numThreads);
    
    std::vector<std::complex<double>> grid(gridsize_t, z0);
    std::vector<std::vector<std::complex<double>>> grid3D(3, grid);
    
    this->Jt_container = grid3D;
    this->Mt_container = grid3D;
    
    this->Et_container = grid3D;
    this->Ht_container = grid3D;
}


// This function calculates the propagation between source and target
void Propagation::propagateBeam(int start, int stop,
                                const std::vector<std::vector<double>> &grid_target,
                                const std::vector<std::vector<double>> &grid_source,
                                const std::vector<std::vector<double>> &norm_target,
                                const std::vector<std::vector<std::complex<double>>> &Js,
                                const std::vector<std::vector<std::complex<double>>> &Ms,
                                const std::vector<double> &source_area)
{
    // Scalars (double & complex double)
    std::complex<double> e_dot_p_r_perp;    // E-field - perpendicular reflected POI polarization vector dot product
    std::complex<double> e_dot_p_r_parr;    // E-field - parallel reflected POI polarization vector dot product
    
    // Arrays of doubles
    std::array<double, 3> S_i_norm;         // Normalized incoming Poynting vector
    std::array<double, 3> p_i_perp;         // Perpendicular incoming POI polarization vector 
    std::array<double, 3> p_i_parr;         // Parallel incoming POI polarization vector 
    std::array<double, 3> S_r_norm;         // Normalized reflected Poynting vector
    std::array<double, 3> p_r_perp;         // Perpendicular reflected POI polarization vector 
    std::array<double, 3> p_r_parr;         // Parallel reflected POI polarization vector 
    std::array<double, 3> S_out_n;          // Container for Poynting-normal ext products
    std::array<double, 3> point;            // Point on target
    std::array<double, 3> norms;            // Normal vector at point
    std::array<double, 3> e_out_h_r;        // Real part of E-field - H-field ext product

    // Arrays of complex doubles
    std::array<std::complex<double>, 3> e_r;            // Reflected E-field
    std::array<std::complex<double>, 3> h_r;            // Reflected H-field
    std::array<std::complex<double>, 3> n_out_e_i_r;    // Electric current
    std::array<std::complex<double>, 3> n_out_h_i_r;    // Magnetic current
    std::array<std::complex<double>, 3> temp1;          // Temporary container 1 for intermediate irrelevant values
    std::array<std::complex<double>, 3> temp2;          // Temporary container 2
    
    // Return containers
    std::array<std::array<std::complex<double>, 3>, 2> beam_e_h; // Container for storing fieldAtPoint return
    
    for(int i=start; i<stop; i++)
    {
        for(int k=0; k<3; k++)
        {
            point[k] = grid_target[k][i];
            norms[k] = norm_target[k][i];
        }
        
        // Calculate total incoming E and H field at point on target
        beam_e_h = fieldAtPoint(grid_source, Js, Ms, point, source_area, start);
        // Calculate normalised incoming poynting vector.
        ut.conj(beam_e_h[1], temp1);                        // h_conj
        ut.ext(beam_e_h[0], temp1, temp2);                  // e_out_h
        
        for (int k=0; k<3; k++) 
        {
            e_out_h_r[k] = temp2[k].real();                      // e_out_h_r
            
            this->Et_container[k][i] = beam_e_h[0][k];
            this->Ht_container[k][i] = beam_e_h[1][k];
        }

        ut.normalize(e_out_h_r, S_i_norm);                       // S_i_norm                   
        
        // Calculate incoming polarization vectors
        ut.ext(S_i_norm, norms, S_out_n);                      // S_i_out_n
        ut.normalize(S_out_n, p_i_perp);                       // p_i_perp                   
        ut.ext(p_i_perp, S_i_norm, p_i_parr);               // p_i_parr                     
        
        // Now calculate reflected poynting vector.
        ut.snell(S_i_norm, norms, S_r_norm);                // S_r_norm     

        // Calculate normalised reflected polarization vectors
        ut.ext(S_r_norm, norms, S_out_n);                      // S_r_out_n
        ut.normalize(S_out_n, p_r_perp);                       // p_r_perp                   
        ut.ext(S_r_norm, p_r_perp, p_r_parr);               // p_r_parr                     
        
        // Now, calculate reflected field from target
        ut.dot(beam_e_h[0], p_r_perp, e_dot_p_r_perp);      // e_dot_p_r_perp
        ut.dot(beam_e_h[0], p_r_parr, e_dot_p_r_parr);      // e_dot_p_r_parr
        
        
        
        // Calculate reflected field from reflection matrix
        for(int k=0; k<3; k++)
        {
            e_r[k] = -e_dot_p_r_perp * p_i_perp[k] - e_dot_p_r_parr * p_i_parr[k];
            
            //this->Et_container[k][i] = e_r[k];
        }

        ut.ext(S_r_norm, e_r, temp1);                       // h_r_temp
        ut.s_mult(temp1, ZETA_0_INV, h_r);                  // h_r       
        
        // Now calculate j and m
        for(int k=0; k<3; k++)
        {
            temp1[k] = e_r[k] + beam_e_h[0][k]; // e_i_r
            temp2[k] = h_r[k] + beam_e_h[1][k]; // h_i_r
        } 
        
        ut.ext(norms, temp2, n_out_h_i_r);
        ut.ext(norms, temp1, n_out_e_i_r);

        for (int k=0; k<3; k++)
        {
            this->Jt_container[k][i] = n_out_h_i_r[k];
            this->Mt_container[k][i] = -n_out_e_i_r[k];
        }

        if(i % 100 == 0 and start == 0 * this->step)
        {
            //std::cout << p_i_perp[0] << std::endl;
            std::cout << i << " / " << this->step << std::endl;
        }
    }
}

std::array<std::array<std::complex<double>, 3>, 2> Propagation::fieldAtPoint(const std::vector<std::vector<double>> &grid_source,
                                               const std::vector<std::vector<std::complex<double>>> &Js,
                                               const std::vector<std::vector<std::complex<double>>> &Ms,
                                               const std::array<double, 3> &point_target,
                                               const std::vector<double> &source_area,
                                               const int start)
{
    // Scalars (double & complex double)
    double r;                           // Distance between source and target points
    double r_inv;                       // 1 / r
    double omega;                       // Angular frequency of field
    std::complex<double> Green;         // Container for Green's function
    std::complex<double> r_in_s;        // Container for inner products between wavevctor and currents
    
    // Arrays of doubles
    std::array<double, 3> source_point; // Container for xyz co-ordinates
    std::array<double, 3> r_vec;        // Distance vector between source and target points
    std::array<double, 3> k_hat;        // Unit wavevctor
    std::array<double, 3> k_arr;        // Wavevector
    
    // Arrays of complex doubles
    std::array<std::complex<double>, 3> e_field;        // Electric field on target
    std::array<std::complex<double>, 3> h_field;        // Magnetic field on target
    std::array<std::complex<double>, 3> js;             // Electric current at source point
    std::array<std::complex<double>, 3> ms;             // Magnetic current at source point
    std::array<std::complex<double>, 3> e_vec_thing;    // Electric current contribution to e-field
    std::array<std::complex<double>, 3> h_vec_thing;    // Magnetic current contribution to h-field
    std::array<std::complex<double>, 3> k_out_ms;       // Outer product between k and ms
    std::array<std::complex<double>, 3> k_out_js;       // Outer product between k and js
    std::array<std::complex<double>, 3> temp;           // Temporary container for intermediate values
    
    // Return container
    std::array<std::array<std::complex<double>, 3>, 2> e_h_field; // Return container containing e and h-fields
    
    e_field.fill(z0);
    h_field.fill(z0);
    
    for(int i=0; i<gridsize_s; i++)
    {
        for (int k=0; k<3; k++)
        {
            source_point[k] = grid_source[k][i];
            js[k] = Js[k][i];
            ms[k] = Ms[k][i];
        }
        //std::cout << source_point[0] << " " << source_point[1] << " " << source_point[2] << std::endl;
        //std::cout << point_target[0] << " " << point_target[1] << " " << point_target[2] << std::endl;
        
        ut.diff(point_target, source_point, r_vec);
        ut.abs(r_vec, r);                              
        
        r_inv = 1 / r;
        
        ut.s_mult(r_vec, r_inv, k_hat);
        ut.s_mult(k_hat, k, k_arr);
        
        omega = C_L * k;
        
        // e-field
        ut.dot(k_hat, js, r_in_s);
        ut.s_mult(k_hat, r_in_s, temp);
        ut.diff(js, temp, e_vec_thing);
        
        ut.ext(k_arr, ms, k_out_ms);
        
        // h-field
        ut.dot(k_hat, ms, r_in_s);
        ut.s_mult(k_hat, r_in_s, temp);
        ut.diff(ms, temp, h_vec_thing);
        
        ut.ext(k_arr, js, k_out_js);
        
        Green = exp(-j * k * r) / (4 * M_PI * r) * source_area[i] * j;
        
        for( int k=0; k<3; k++)
        {
            e_field[k] += (-omega * MU_0 * e_vec_thing[k] + k_out_ms[k]) * Green;
            h_field[k] += (-omega * EPS_ALU * h_vec_thing[k] - k_out_js[k]) * Green;
        }  
    }
    
    // Pack e and h together in single container 
    e_h_field[0] = e_field;
    e_h_field[1] = h_field;
    
    //std::cout << ut.abs(e_field) << std::endl;
    
    return e_h_field;
}

void Propagation::parallelProp(const std::vector<std::vector<double>> &grid_target,
                               const std::vector<std::vector<double>> &grid_source,
                               const std::vector<std::vector<double>> &norm_target,
                               const std::vector<std::vector<std::complex<double>>> &Js,
                               const std::vector<std::vector<std::complex<double>>> &Ms,
                               const std::vector<double> &source_area)
{
    int final_step; 
    
    std::cout << gridsize_t << std::endl;
    
    
    for(int n=0; n<numThreads; n++)
    {
        if(n == (numThreads-1))
        {
            final_step = gridsize_t;
        }
        /*
        else if(n == 1)
        {
            final_step = (n+1) * step - 2000;
        }
        */
        else
        {
            final_step = (n+1) * step;
        }
        
        std::cout << final_step << std::endl;
        
        threadPool[n] = std::thread(&Propagation::propagateBeam, 
                                    this, n * step, final_step, 
                                    grid_target, grid_source, norm_target, Js, Ms, source_area);
    }
}

void Propagation::joinThreads() 
{
    for (std::thread &t : threadPool) 
    {
        if (t.joinable()) 
        {
            t.join();
        }
    }
}
        
    
                                        
    
