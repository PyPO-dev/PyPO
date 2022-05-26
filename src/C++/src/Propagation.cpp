#include "Propagation.h"

Propagation::Propagation(double k, int numThreads, int gridsize_s, int gridsize_t, double thres)
{
    this->k = k;

    this->numThreads = numThreads;
    this->gridsize_s = gridsize_s;
    this->gridsize_t = gridsize_t;
    
    this->step = ceil(gridsize_t / numThreads);
    
    // Convert decibels to E-field values now to save log10 evaluation time
    double exponent = -1 * thres / 20.;
    this->thres = pow(10., exponent);
    
    threadPool.resize(numThreads);
    
    j_container.resize(gridsize_t);
    m_container.resize(gridsize_t);

    std::complex<double> j(0., 1.);
    std::complex<double> z0(0., 0.);
    this->j = j;
    this->z0 = z0;
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
    std::vector<std::complex<double>> h_conj(3, z0);
    std::vector<std::complex<double>> e_out_h(3, z0);
    std::vector<std::vector<std::complex<double>>> beam_e_h(3, h_conj);
    
    std::vector<double> S_i_norm(3, 0.);
    std::vector<double> S_i_out_n(3, 0.);
    std::vector<double> p_i_perp(3, 0.);
    std::vector<double> p_i_parr(3, 0.);
    std::vector<double> S_r_norm(3, 0.);
    std::vector<double> S_r_out_n(3, 0.);
    std::vector<double> p_r_perp(3, 0.);
    std::vector<double> p_r_parr(3, 0.);

    std::vector<std::complex<double>> e_r(3, z0);
        
    std::complex<double> e_dot_p_r_perp = z0;
    std::complex<double> e_dot_p_r_parr = z0;
    
    std::vector<double> point(3, 0.);
    std::vector<double> norms(3, 0.);
    
    std::vector<double> e_out_h_r(3, 0.);
    
    std::vector<std::complex<double>> h_r_temp(3, z0);
    std::vector<std::complex<double>> h_r(3, z0);

    std::vector<std::complex<double>> e_i_r(3, z0);
    std::vector<std::complex<double>> h_i_r(3, z0);
    std::vector<std::complex<double>> n_out_e_i_r(3, z0); 
    std::vector<std::complex<double>> n_out_h_i_r(3, z0); 
    
    for(int i=start; i<stop; i++)
    {
        point[0] = grid_target[0][i];
        point[1] = grid_target[1][i];
        point[2] = grid_target[2][i];

        norms[0] = norm_target[0][i];
        norms[1] = norm_target[1][i];
        norms[2] = norm_target[2][i];
        
        // Calculate total incoming E and H field at point on target
        beam_e_h = fieldAtPoint(grid_source, Js, Ms, point, source_area, start);
        
        // Calculate normalised incoming poynting vector.
        h_conj = ut.conj(beam_e_h[1]);
        e_out_h = ut.ext(beam_e_h[0], h_conj);
        
        for (int k=0; k<3; k++) 
        {
            e_out_h_r[k] = e_out_h[k].real();
            
            this->Et_container[k][i] = beam_e_h[0][k];
            this->Ht_container[k][i] = beam_e_h[1][k];
        }
        
        S_i_norm = ut.normalize(e_out_h_r); // Missing factor of 0.5: normalized anyways right?
        
        // Calculate incoming polarization vectors
        S_i_out_n = ut.ext(S_i_norm, norms);
        p_i_perp = ut.normalize(S_i_out_n);
        p_i_parr = ut.ext(p_i_perp, S_i_norm);
        
        // Now calculate reflected poynting vector.
        S_r_norm = ut.snell(S_i_norm, norms);
        
        // Calculate normalised reflected polarization vectors
        S_r_out_n = ut.ext(S_r_norm, norms);
        p_r_perp = ut.normalize(S_r_out_n);
        p_r_parr = ut.ext(S_r_norm, p_r_perp);
        
        // Now, calculate reflected field from target
        e_dot_p_r_perp = ut.dot(beam_e_h[0], p_r_perp);
        e_dot_p_r_parr = ut.dot(beam_e_h[0], p_r_parr);
        
        // Calculate reflected field from reflection matrix
        for(int k=0; k<3; k++)
        {
            e_r[k] = -e_dot_p_r_perp * p_i_perp[k] - e_dot_p_r_parr * p_i_parr[k];
        }

        h_r_temp = ut.ext(S_r_norm, e_r);
        h_r = ut.s_mult(h_r_temp, ZETA_0_INV);
        
        // Now calculate j and m
        for(int k=0; k<3; k++)
        {
            e_i_r[k] = e_r[k] + beam_e_h[0][k];
            h_i_r[k] = h_r[k] + beam_e_h[1][k];
        }
        
        n_out_h_i_r = ut.ext(norms, h_i_r);
        n_out_e_i_r = ut.ext(norms, e_i_r);
        // Fill arrays containing the currents on the target. Note m should be multiplied with - before usage
        for (int k=0; k<3; k++)
        {
            this->Jt_container[k][i] = n_out_h_i_r[k]
            this->Mt_container[k][i] = n_out_e_i_r[k]
        }
        
        if(i % 100 == 0 and start == 0 * step_e)
        {
            int toPrint = gridsize_e / numThreads;
            std::cout << "    " << i << " / " << toPrint << std::endl;
        }
    }
}

std::vector<std::vector<std::complex<double>>> Propagation::fieldAtPoint(const std::vector<std::vector<double>> &grid_source,
                                               const std::vector<std::vector<std::complex<double>>> &Js,
                                               const std::vector<std::vector<std::complex<double>>> &Ms,
                                               const std::vector<double> &point_target,
                                               const std::vector<double> &source_area,
                                               const int start)
{
    std::vector<std::complex<double>> e_field(3, z0);
    std::vector<std::complex<double>> h_field(3, z0);

    std::vector<double> source_point(3, 0.);
    std::vector<double> source_norm(3, 0.);
    
    std::vector<std::vector<std::complex<double>>> e_h_field(2, e_field);
    std::vector<std::complex<double>> j(3, z0);
    std::vector<std::complex<double>> m(3, z0);
    std::vector<double> r_vec(3, 0.);
    double r = 0.;
    
    std::complex<double> r_in_j = z0;
    std::vector<std::complex<double>> temp2_e(3, z0);
    std::vector<std::complex<double>> temp1_e(3, z0);
    std::vector<std::complex<double>> e_vec_thing(3, z0);
        
    std::vector<std::complex<double>> k_out_m(3, z0);

    std::vector<std::complex<double>> k_out_j(3, z0);
        
    std::complex<double> r_in_m = z0;
    std::vector<std::complex<double>> temp2_h(3, z0);
    std::vector<std::complex<double>> temp1_h(3, z0);
    std::vector<std::complex<double>> h_vec_thing(3, z0);
    std::complex<double> Green = z0;
    
    for(int i=0; i<gridsize_s; i++)
    {
        for (int k=0; k<3; k++)
        {
            source_point[k] = grid_source[k][i];
            source_norm[k] = norm_source[k][i];
        
            js[k] = Js[k][i];
            ms[k] = Ms[k][i];
        }
        r_vec = ut.diff(point_target, source_point);
        r = ut.abs(r_vec);
        
        // This outer products are necessary for the e-field. Assume MU_ALU = MU_0
        r_in_js = ut.dot(r_vec, js);
        temp2_e = ut.s_mult(r_vec, r_in_js);
        temp1_e = ut.s_mult(js, r*r);
        e_vec_thing = ut.diff(temp1_e, temp2_e);
        
        k_out_ms = ut.ext(r_vec, ms);
        
        // This part is for h-field.
        k_out_js = ut.ext(r_vec, js);
        
        r_in_ms = ut.dot(r_vec, ms);
        temp2_h = ut.s_mult(r_vec, r_in_ms);
        temp1_h = ut.s_mult(ms, r*r);
        h_vec_thing = ut.diff(temp1_h, temp2_h);
        
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
    return e_h_field;
}

void Propagation::parallelProp(const std::vector<std::vector<double>> &grid_target,
                               const std::vector<std::vector<double>> &grid_source,
                               const std::vector<std::vector<double>> &norm_target,
                               const std::vector<std::vector<std::complex<double>>> &Js,
                               const std::vector<std::vector<std::complex<double>>> &Ms,
                               const std::vector<double> &source_area)
{
    // Resize J, M, E, H containers.
    Jt_container.resize(gridsize_t);
    Mt_container.resize(gridsize_t);
    
    Et_container.resize(gridsize_t);
    Ht_container.resize(gridsize_t);
    
    int final_step; 
    
    for(int n=0; n<numThreads; n++)
    {
        if(n == (numThreads-1))
        {
            final_step = gridsize_e;
        }
        
        else
        {
            final_step = (n+1) * step_e;
        }
        
        threadPool[n] = std::thread(&Propagation::propagateBeam, 
                                    this, n * step_e, final_step, 
                                    grid_target, grid_source, norm_target, norm_source, Js, Ms, source_area);
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
        
    
                                        
    
