#include "Propagation.h"

Propagation::Propagation(double k, int numThreads, 
                         int gridsize_i, int gridsize_h, int gridsize_e, int gridsize_sec,
                         std::vector<double> n0, std::vector<double> p0, 
                         double &source_area, double thres)
{
    this->k = k;

    this->numThreads = numThreads;
    this->gridsize_i = gridsize_i;
    this->gridsize_h = gridsize_h;
    this->gridsize_e = gridsize_e;
    this->gridsize_sec = gridsize_sec;
    
    this->step_h = ceil(gridsize_h / numThreads);
    this->step_e = ceil(gridsize_e / numThreads);
    this->step_sec = ceil(gridsize_sec / numThreads);
    
    // Convert decibels to E-field values now to save log10 evaluation time
    double exponent = -1 * thres / 20.;
    this->thres = pow(10., exponent);
    
    threadPool.resize(numThreads);
    
    j_1_container.resize(gridsize_h);
    m_1_container.resize(gridsize_h);
    
    j_2_container.resize(gridsize_e);
    m_2_container.resize(gridsize_e);

    std::complex<double> j(0., 1.);
    std::complex<double> z0(0., 0.);
    this->j = j;
    this->z0 = z0;
    
    this->n0 = n0;
    this->p0 = p0;
    this->source_area = source_area;
}

// This function calculates the propagation between ellipse and focal plane
void Propagation::propagateBeamFoc(int start, int stop,
                                const std::vector<std::vector<double>> &grid_target,
                                const std::vector<std::vector<double>> &grid_source,
                                const std::vector<double> &source_area)
{
    // For performance gain, allocate ALL data structures before the main loop
    std::vector<std::complex<double>> h_conj(3, z0);
    std::vector<std::complex<double>> e_out_h(3, z0);
    std::vector<double> e_out_h_r(3, 0.);
    std::vector<std::vector<std::complex<double>>> beam_e_h(3, h_conj);
    std::vector<double> point(3, 0.);
    
    for(int i=start; i<stop; i++)
    {
        point[0] = grid_target[0][i];
        point[1] = grid_target[1][i];
        point[2] = grid_target[2][i];

        beam_e_h = fieldAtFoc(grid_source, point, source_area);
        
        this->beam_container_e[i] = beam_e_h[0];
        this->beam_container_h[i] = beam_e_h[1];
        
        // Calculate normalised incoming poynting vector for GO determination in python.
        h_conj = ut.conj(beam_e_h[1]);
        e_out_h = ut.ext(beam_e_h[0], h_conj);

        for (int n=0; n<3; n++) 
        {
            e_out_h_r[n] = e_out_h[n].real();
        }

        if(i % 100 == 0 and start == 0 * step_sec)
        {
            int toPrint = gridsize_sec / numThreads;
            std::cout << "    " << i << " / " << toPrint << std::endl;
        }
    }
}

std::vector<std::vector<std::complex<double>>> Propagation::fieldAtFoc(const std::vector<std::vector<double>> &grid_source,
                                               const std::vector<double> &point_target,
                                               const std::vector<double> &source_area)
{
    std::vector<std::complex<double>> e_field(3, z0);
    std::vector<std::complex<double>> h_field(3, z0);
    
    std::vector<std::vector<std::complex<double>>> e_h_field(2, e_field);
    
    // For performance gain, initialize ALL data structures oputside of loop
    // Re-assign values as we go
    
    std::vector<std::complex<double>> j_2(3, z0);
    std::vector<std::complex<double>> m_2(3, z0);
    std::vector<double> r_vec(3, 0.);
    double r = 0.;
    std::complex<double> r_in_j2 = z0;
    std::vector<std::complex<double>> temp2_e(3, z0);
    std::vector<std::complex<double>> temp1_e(3, z0);
    std::vector<std::complex<double>> e_vec_thing(3, z0);
    std::vector<std::complex<double>> k_out_m2(3, z0); 
    std::vector<std::complex<double>> k_out_j2(3, z0); 
    std::complex<double> r_in_m2 = z0;
    std::vector<std::complex<double>> temp2_h(3, z0);
    std::vector<std::complex<double>> temp1_h(3, z0);
    std::vector<std::complex<double>> h_vec_thing(3, z0);
    std::complex<double> Green = z0;
    
    std::vector<double> source_point(3, 0.);
    
    for(int i=0; i<gridsize_e; i++)
    {
        source_point[0] = grid_source[0][i];
        source_point[1] = grid_source[1][i];
        source_point[2] = grid_source[2][i];
        
        j_2 = this->j_2_container[i];
        m_2 = this->m_2_container[i];
        
        r_vec = ut.diff(point_target, source_point);
        r = ut.abs(r_vec);
        
        // These outer products are necessary for the e-field. Assume MU_ALU = MU_0
        r_in_j2 = ut.dot(r_vec, j_2);
        temp2_e = ut.s_mult(r_vec, r_in_j2);
        temp1_e = ut.s_mult(j_2, r*r);
        e_vec_thing = ut.diff(temp1_e, temp2_e);
        
        k_out_m2 = ut.ext(r_vec, m_2);
        
        // This part is for h-field.
        k_out_j2 = ut.ext(r_vec, j_2);
        
        r_in_m2 = ut.dot(r_vec, m_2);
        temp2_h = ut.s_mult(r_vec, r_in_m2);
        temp1_h = ut.s_mult(m_2, r*r);
        h_vec_thing = ut.diff(temp1_h, temp2_h);
        
        // Technically every similar term for e and h is calculated here, not just greens function...
        Green = k * exp(-j * k * r) / (4 * M_PI * r*r) * source_area[i] * j;
        
        for( int n=0; n<3; n++)
        {
            e_field[n] += -C_L * MU_0 * e_vec_thing[n] / r * Green + k_out_m2[n] * Green;
            h_field[n] += -C_L * EPS_ALU * h_vec_thing[n] / r * Green - k_out_j2[n] * Green;
        }  
    }
    e_h_field[0] = e_field;
    e_h_field[1] = h_field;
    return e_h_field;
}

// This function calculates the propagation between hyperbola and ellipse
void Propagation::propagateBeam(int start, int stop,
                                const std::vector<std::vector<double>> &grid_target,
                                const std::vector<std::vector<double>> &grid_source,
                                const std::vector<double> &source_area,
                                const std::vector<std::vector<double>> &ell_n)
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
    
    for(int i=start; i<stop; i++)
    {
        point[0] = grid_target[0][i];
        point[1] = grid_target[1][i];
        point[2] = grid_target[2][i];

        norms[0] = ell_n[0][i];
        norms[1] = ell_n[1][i];
        norms[2] = ell_n[2][i];

        beam_e_h = fieldAtPoint(grid_source, point, source_area, start);
        
        this->beam_container_e[i] = beam_e_h[0];
        this->beam_container_h[i] = beam_e_h[1];
        
        // Calculate normalised incoming poynting vector.
        h_conj = ut.conj(beam_e_h[1]);
        e_out_h = ut.ext(beam_e_h[0], h_conj);
        
        for (int n=0; n<3; n++) 
        {
            e_out_h_r[n] = e_out_h[n].real();
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
        
        // Now, calculate reflected field from ellipse
        e_dot_p_r_perp = ut.dot(beam_e_h[0], p_r_perp);
        e_dot_p_r_parr = ut.dot(beam_e_h[0], p_r_parr);
        
        // Calculate reflected field from reflection matrix
        for(int n=0; n<3; n++)
        {
            e_r[n] = -e_dot_p_r_perp * p_i_perp[n] - e_dot_p_r_parr * p_i_parr[n];
        }

        h_r_temp = ut.ext(S_r_norm, e_r);
        h_r = ut.s_mult(h_r_temp, ZETA_0_INV);
        
        // Now calculate j_2 and m_2
        for(int n=0; n<3; n++)
        {
            e_i_r[n] = e_r[n] + beam_e_h[0][n];
            h_i_r[n] = h_r[n] + beam_e_h[1][n];
        }
        
        // Fill arrays containing the currents on the ellipse for propagation to focal plane. Note m_2 should be multiplied with - before usage
        this->j_2_container[i] = ut.ext(norms, h_i_r);
        n_out_e_i_r = ut.ext(norms, e_i_r);
        
        this->m_2_container[i] = ut.s_mult(n_out_e_i_r, -1.);
        
        
        if(i % 100 == 0 and start == 0 * step_e)
        {
            int toPrint = gridsize_e / numThreads;
            std::cout << "    " << i << " / " << toPrint << std::endl;
        }
    }
}

std::vector<std::vector<std::complex<double>>> Propagation::fieldAtPoint(const std::vector<std::vector<double>> &grid_source,
                                               const std::vector<double> &point_target,
                                               const std::vector<double> &source_area,
                                               const int start)
{
    std::vector<std::complex<double>> e_field(3, z0);
    std::vector<std::complex<double>> h_field(3, z0);

    std::vector<double> source_point(3, 0.);
    
    std::vector<std::vector<std::complex<double>>> e_h_field(2, e_field);
    std::vector<std::complex<double>> j_1(3, z0);
    std::vector<std::complex<double>> m_1(3, z0);
    std::vector<double> r_vec(3, 0.);
    double r = 0.;
    
    std::complex<double> r_in_j1 = z0;
    std::vector<std::complex<double>> temp2_e(3, z0);
    std::vector<std::complex<double>> temp1_e(3, z0);
    std::vector<std::complex<double>> e_vec_thing(3, z0);
        
    std::vector<std::complex<double>> k_out_m1(3, z0);

    std::vector<std::complex<double>> k_out_j1(3, z0);
        
    std::complex<double> r_in_m1 = z0;
    std::vector<std::complex<double>> temp2_h(3, z0);
    std::vector<std::complex<double>> temp1_h(3, z0);
    std::vector<std::complex<double>> h_vec_thing(3, z0);
    std::complex<double> Green = z0;
    
    for(int i=0; i<gridsize_h; i++)
    {
        source_point[0] = grid_source[0][i];
        source_point[1] = grid_source[1][i];
        source_point[2] = grid_source[2][i];
        
        j_1 = this->j_1_container[i];
        m_1 = this->m_1_container[i];
        
        r_vec = ut.diff(point_target, source_point);
        r = ut.abs(r_vec);
        
        // This outer products are necessary for the e-field. Assume MU_ALU = MU_0
        r_in_j1 = ut.dot(r_vec, j_1);
        temp2_e = ut.s_mult(r_vec, r_in_j1);
        temp1_e = ut.s_mult(j_1, r*r);
        e_vec_thing = ut.diff(temp1_e, temp2_e);
        
        k_out_m1 = ut.ext(r_vec, m_1);
        
        // This part is for h-field.
        k_out_j1 = ut.ext(r_vec, j_1);
        
        r_in_m1 = ut.dot(r_vec, m_1);
        temp2_h = ut.s_mult(r_vec, r_in_m1);
        temp1_h = ut.s_mult(m_1, r*r);
        h_vec_thing = ut.diff(temp1_h, temp2_h);
        
        // Technically every similar term for e and h is calculated here, not just greens function...
        Green = k * exp(-j * k * r) / (4 * M_PI * r*r) * source_area[i] * j;
        
        for( int n=0; n<3; n++)
        {
            e_field[n] += -C_L * MU_0 * e_vec_thing[n] / r * Green + k_out_m1[n] * Green;
            h_field[n] += -C_L * EPS_ALU * h_vec_thing[n] / r * Green - k_out_j1[n] * Green;
        }  
    }
    e_h_field[0] = e_field;
    e_h_field[1] = h_field;
    return e_h_field;
}

// Use this function to propagate from source plane to hyperbola. It stores the incoming and reflected e and h fields on hyperbola.
// Also, it stores the reflected poynting vectors from hyperbola
void Propagation::propagateBeamInit(int start, int stop,
                                const std::vector<std::vector<double>> &grid_target,
                                const std::vector<std::vector<double>> &grid_source,
                                const std::vector<std::complex<double>> &beam_source,
                                const std::vector<std::vector<double>> &hyp_n)
{
    /*
    std::vector<double> n0_out_p0 = ut.ext(n0, p0);
    std::vector<double> m0_hat = ut.s_mult(n0_out_p0, -2.);
    */
    // Calculate inverse of maximum element of beam
    double max_beam =  1. / abs(*std::max_element(beam_source.begin(), beam_source.end(),
                           [](const std::complex<double>& a, const std::complex<double>& b)
                             { return abs(a) < abs(b); }));

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
    
    for(int i=start; i<stop; i++)
    {
        point[0] = grid_target[0][i];
        point[1] = grid_target[1][i];
        point[2] = grid_target[2][i];

        norms[0] = hyp_n[0][i];
        norms[1] = hyp_n[1][i];
        norms[2] = hyp_n[2][i];
        
        beam_e_h = fieldAtPointInit(grid_source, beam_source, point, max_beam);
        
        this->beam_container_e[i] = beam_e_h[0];
        this->beam_container_h[i] = beam_e_h[1];
        
        // Calculate normalised incoming poynting vector.
        h_conj = ut.conj(beam_e_h[1]);
        e_out_h = ut.ext(beam_e_h[0], h_conj);

        for (int n=0; n<3; n++) 
        {
            e_out_h_r[n] = e_out_h[n].real();
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
        
        // Now, calculate reflected field from hyperbola.
        e_dot_p_r_perp = ut.dot(beam_e_h[0], p_r_perp);
        e_dot_p_r_parr = ut.dot(beam_e_h[0], p_r_parr);
        
        // Calculate reflected field from reflection matrix
        for(int n=0; n<3; n++)
        {
            e_r[n] = -e_dot_p_r_perp * p_i_perp[n] - e_dot_p_r_parr * p_i_parr[n];
        }

        h_r_temp = ut.ext(S_r_norm, e_r);
        h_r = ut.s_mult(h_r_temp, ZETA_0_INV);
        
        // Now calculate j_1 and m_1
        for(int n=0; n<3; n++)
        {
            e_i_r[n] = e_r[n] + beam_e_h[0][n];
            h_i_r[n] = h_r[n] + beam_e_h[1][n];
        }
        
        // Fill arrays containing the currents on the hyperbola for propagation to ellipse
        this->j_1_container[i] = ut.ext(norms, h_i_r);
        n_out_e_i_r = ut.ext(norms, e_i_r);
        
        this->m_1_container[i] = ut.s_mult(n_out_e_i_r, -1.);

        if(i % 100 == 0 and start == 0*step_h)
        {
            int toPrint = gridsize_h / numThreads;
            std::cout << "    " << i << " / " << toPrint << std::endl;
        }
    }
}

// Function to calculate field on hyperbola point due to source plane
// Output structure -->
// Vector of 2 elements, e and h field -->
// Vector of 3 elements, x y z components of the e/h-field
// complex number: component value.

std::vector<std::vector<std::complex<double>>> Propagation::fieldAtPointInit(const std::vector<std::vector<double>> &grid_source,
                                               const std::vector<std::complex<double>> &beam_source,
                                               const std::vector<double> &point_target,
                                               const double max_beam)
{
    std::vector<std::complex<double>> e_field(3, z0);
    std::vector<std::complex<double>> h_field(3, z0);

    std::vector<std::vector<std::complex<double>>> e_h_field(2, e_field);
    std::vector<double> source_point(3, 0.);
    std::vector<std::complex<double>> e_vec(3, z0);
    
    std::vector<std::complex<double>> n0_out_e(3, z0);
    std::vector<std::complex<double>> m0_hat(3, z0);
        
    std::vector<double> r_vec(3, 0.);
    double r = 0.;

    std::vector<std::complex<double>> k_out_m(3, z0);

    std::complex<double> r_in_m0 = z0;
    std::vector<std::complex<double>> temp2(3, z0);
    std::vector<std::complex<double>> temp1(3, z0);
    std::vector<std::complex<double>> h_vec_thing(3, z0);
    std::complex<double> Green = z0;

    for(int i=0; i<gridsize_i; i++)
    {
        source_point[0] = grid_source[0][i];
        source_point[1] = grid_source[1][i];
        source_point[2] = grid_source[2][i];
        
        // Create vectorial e field using initial polarization
        e_vec = ut.s_mult(p0, beam_source[i]);
        
        // Check if x-component of field is below thres
        if ((max_beam * abs(e_vec[0])) < thres)
        {
            continue;
        }
        
        n0_out_e = ut.ext(n0, e_vec);
        m0_hat = ut.s_mult(n0_out_e, -2.);
        
        r_vec = ut.diff(point_target, source_point);
        r = ut.abs(r_vec);
        
        // This outer product is necessary for the e-field
        k_out_m = ut.ext(r_vec, m0_hat);
        
        // This part is for h-field
        r_in_m0 = ut.dot(r_vec, m0_hat);
        temp2 = ut.s_mult(r_vec, r_in_m0);
        temp1 = ut.s_mult(m0_hat, r*r);
        h_vec_thing = ut.diff(temp1, temp2);
        
        // Technically every similar term for e and h are calculated here, not just greens function...
        Green = k * exp(-j * k * r) / (4 * M_PI * r*r) * source_area * j;
        
        for( int n=0; n<3; n++)
        {
            e_field[n] +=  k_out_m[n] * Green;
            h_field[n] += -C_L * EPS_ALU * h_vec_thing[n] / r * Green;
        }    
    }
    e_h_field[0] = e_field;
    e_h_field[1] = h_field;
    return e_h_field;
}

void Propagation::parallelPropFoc(const std::vector<std::vector<double>> &grid_target,
                               const std::vector<std::vector<double>> &grid_source, 
                               const std::vector<double> &source_area)
{
    // Resize beam containers. Note, e and h in beam container refer to e and h field, whereas e in gridsize refers to ellipse.
    beam_container_e.resize(gridsize_sec);
    beam_container_h.resize(gridsize_sec);
    
    int final_step; 
    
    for(int n=0; n<numThreads; n++)
    {
        if(n == (numThreads-1))
        {
            final_step = gridsize_sec;
        }
        
        else
        {
            final_step = (n+1) * step_sec;
        }
        
        threadPool[n] = std::thread(&Propagation::propagateBeamFoc, 
                                    this, n * step_sec, final_step, 
                                    grid_target, grid_source, source_area);
    }
}

void Propagation::parallelProp(const std::vector<std::vector<double>> &grid_target,
                               const std::vector<std::vector<double>> &grid_source, 
                               const std::vector<double> &source_area,
                               const std::vector<std::vector<double>> &ell_n)
{
    // Resize beam containers. Note, e and h in beam container refer to e and h field, whereas e in gridsize refers to ellipse.
    beam_container_e.resize(gridsize_e);
    beam_container_h.resize(gridsize_e);
    
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
                                    grid_target, grid_source, source_area, ell_n);
    }
}


// Function to propagate the beam from source plane to hyperbola. Takes scalar
// area and just one single normal and polarization vector.
void Propagation::parallelPropInit(const std::vector<std::vector<double>> &grid_target,
                               const std::vector<std::vector<double>> &grid_source, 
                               const std::vector<std::complex<double>> &beam_source,
                               const std::vector<std::vector<double>> &hyp_n)
{
    // Resize beam containers. Note, e and h in beam container refer to e and h field, whereas h in gridsize refers to hyperbola.
    beam_container_e.resize(gridsize_h);
    beam_container_h.resize(gridsize_h);
    
    int final_step; 
    
    for(int n=0; n<numThreads; n++)
    {
        if(n == (numThreads-1))
        {
            final_step = gridsize_h;
        }
        
        else
        {
            final_step = (n+1) * step_h;
        }
        
        threadPool[n] = std::thread(&Propagation::propagateBeamInit, 
                                    this, n * step_h, final_step, 
                                    grid_target, grid_source, beam_source, hyp_n);
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

void Propagation::emptyContainer()
{
    std::vector<std::complex<double>> z0(3, j);
    std::fill(beam_container_e.begin(), beam_container_e.end(), z0);
    std::fill(beam_container_h.begin(), beam_container_h.end(), z0);
}
        
    
                                        
    
