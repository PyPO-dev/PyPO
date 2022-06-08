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
    
    // For performance gain, allocate ALL data structures before the main loop
    std::array<std::array<std::complex<double>, 3>, 2> beam_e_h;
    
    std::array<double, 3> S_i_norm;
    std::array<double, 3> p_i_perp;
    std::array<double, 3> p_i_parr;
    
    std::array<double, 3> S_r_norm;
    std::array<double, 3> p_r_perp;
    std::array<double, 3> p_r_parr;

    std::array<std::complex<double>, 3> e_r;
    std::array<std::complex<double>, 3> h_r;
        
    std::complex<double> e_dot_p_r_perp;
    std::complex<double> e_dot_p_r_parr;
    
    std::array<double, 3> point;
    std::array<double, 3> norms;
    
    std::array<std::complex<double>, 3> n_out_e_i_r; 
    std::array<std::complex<double>, 3> n_out_h_i_r; 
    
    // Utility arrays
    std::array<double, 3> out1;
    std::array<double, 3> out2;
    
    std::array<std::complex<double>, 3> cout1;
    std::array<std::complex<double>, 3> cout2;
    
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
        ut.conj(beam_e_h[1], cout1);                        // h_conj
        ut.ext(beam_e_h[0], cout1, cout2);                  // e_out_h
        
        for (int k=0; k<3; k++) 
        {
            out1[k] = cout2[k].real();                      // e_out_h_r
            
            this->Et_container[k][i] = beam_e_h[0][k];
            this->Ht_container[k][i] = beam_e_h[1][k];
        }

        ut.normalize(out1, S_i_norm);                       // S_i_norm                   
        
        // Calculate incoming polarization vectors
        ut.ext(S_i_norm, norms, out1);                      // S_i_out_n
        ut.normalize(out1, p_i_perp);                       // p_i_perp                   
        ut.ext(p_i_perp, S_i_norm, p_i_parr);               // p_i_parr                     
        
        // Now calculate reflected poynting vector.
        ut.snell(S_i_norm, norms, S_r_norm);                // S_r_norm            
        
        // Calculate normalised reflected polarization vectors
        ut.ext(S_r_norm, norms, out1);                      // S_r_out_n
        ut.normalize(out1, p_r_perp);                       // p_r_perp                   
        ut.ext(S_r_norm, p_r_perp, p_r_parr);               // p_r_parr                     
        
        // Now, calculate reflected field from target
        ut.dot(beam_e_h[0], p_r_perp, e_dot_p_r_perp);      // e_dot_p_r_perp
        ut.dot(beam_e_h[0], p_r_parr, e_dot_p_r_parr);      // e_dot_p_r_parr
        
        // Calculate reflected field from reflection matrix
        for(int k=0; k<3; k++)
        {
            e_r[k] = -e_dot_p_r_perp * p_i_perp[k] - e_dot_p_r_parr * p_i_parr[k];
        }

        ut.ext(S_r_norm, e_r, cout1);                       // h_r_temp
        ut.s_mult(cout1, ZETA_0_INV, h_r);                  // h_r       
        
        // Now calculate j and m
        for(int k=0; k<3; k++)
        {
            cout1[k] = e_r[k] + beam_e_h[0][k];             // e_i_r
            cout2[k] = h_r[k] + beam_e_h[1][k];             // h_i_r
        } 
        
        ut.ext(norms, cout2, n_out_h_i_r);                  // n_out_h_i_r
        ut.ext(norms, cout1, n_out_e_i_r);                  // n_out_e_i_r
        // Fill arrays containing the currents on the target. Note m should be multiplied with - before usage
        for (int k=0; k<3; k++)
        {
            this->Jt_container[k][i] = n_out_h_i_r[k];
            this->Mt_container[k][i] = -n_out_e_i_r[k];
        }
        
        if(i % 100 == 0 and start == 0 * this->step)
        {
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
    
    std::array<std::complex<double>, 3> e_field;
    std::array<std::complex<double>, 3> h_field;

    std::array<double, 3> source_point;
    
    std::array<std::array<std::complex<double>, 3>, 2> e_h_field;
    
    std::array<std::complex<double>, 3> js;
    std::array<std::complex<double>, 3> ms;
    std::array<double, 3> r_vec;
    double r = 0.;
    
    std::complex<double> r_in_s = z0;
    std::array<std::complex<double>, 3> temp2;
    std::array<std::complex<double>, 3> temp1;
    std::array<std::complex<double>, 3> e_vec_thing;
    std::array<std::complex<double>, 3> h_vec_thing;
        
    std::array<std::complex<double>, 3> k_out_ms;

    std::array<std::complex<double>, 3> k_out_js;

    std::complex<double> Green = z0;
    
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
        
        ut.diff(point_target, source_point, r_vec);     // r_vec
        ut.abs(r_vec, r);                               // r
        //std::cout << r << std::endl;
        
        // This outer products are necessary for the e-field. Assume MU_ALU = MU_0
        ut.dot(r_vec, js, r_in_s);                      // r_in_js
        ut.s_mult(r_vec, r_in_s, temp2);                // temp2
        ut.s_mult(js, r*r, temp1);                      // temp1
        ut.diff(temp1, temp2, e_vec_thing);             // e_vec_thing
        
        ut.ext(r_vec, ms, k_out_ms);                    // k_out_ms
        
        // This part is for h-field.
        ut.ext(r_vec, js, k_out_js);                    // k_out_js
        
        ut.dot(r_vec, ms, r_in_s);                      // r_in_ms
        ut.s_mult(r_vec, r_in_s, temp2);                // temp2
        ut.s_mult(ms, r*r, temp1);                      // temp1
        
        ut.diff(temp1, temp2, h_vec_thing);             // h_vec_thing
        
        // Technically every similar term for e and h is calculated here, not just greens function...
        Green = k * exp(-j * k * r) / (4 * M_PI * r*r) * source_area[i] * j;
        
        for( int k=0; k<3; k++)
        {
            e_field[k] += -C_L * MU_0 * e_vec_thing[k] / r * Green + k_out_ms[k] * Green;
            h_field[k] += -C_L * EPS_ALU * h_vec_thing[k] / r * Green - k_out_js[k] * Green;
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
    
    for(int n=0; n<numThreads; n++)
    {
        if(n == (numThreads-1))
        {
            final_step = gridsize_t;
        }
        
        else
        {
            final_step = (n+1) * step;
        }
        
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
        
    
                                        
    
