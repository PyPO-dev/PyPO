#include "Propagation.h"

Propagation::Propagation(double k, int numThreads, int gridsize_s, int gridsize_t, double thres, double epsilon, double t_direction)
{
    std::complex<double> j(0., 1.);
    std::complex<double> z0(0., 0.);
    this->j = j;
    this->z0 = z0;
    
    this->k = k;
    
    this->EPS = epsilon * EPS_VAC; // epsilon is relative permeability

    this->numThreads = numThreads;
    this->gridsize_s = gridsize_s;
    this->gridsize_t = gridsize_t;
    
    this->step = ceil(gridsize_t / numThreads);

    // Convert decibels to E-field values now to save log10 evaluation time
    double exponent = -1 * thres / 20.;
    this->thres = pow(10., exponent);
    
    threadPool.resize(numThreads);
    std::array<std::complex<double>, 3> arr;
    
    std::vector<std::array<std::complex<double>, 3>> grid3D(gridsize_t, arr);
    /*
    this->Jt_container = grid3D;
    this->Mt_container = grid3D;
    
    this->Et_container = grid3D;
    this->Ht_container = grid3D;
    */
    this->Jt_container.resize(gridsize_t);
    this->Mt_container.resize(gridsize_t);
    
    this->Et_container.resize(gridsize_t);
    this->Ht_container.resize(gridsize_t);
    
    this->t_direction = t_direction;
    
    std::array<std::array<double, 3>, 3> eye;
    eye[0].fill(0.);
    eye[1].fill(0.);
    eye[2].fill(0.);
    
    eye[0][0] = 1.;
    eye[1][1] = 1.;
    eye[2][2] = 1.;
}


// This function calculates the propagation between source and target
void Propagation::propagateBeam(int start, int stop,
                                const std::vector<std::array<double, 3>> &grid_target,
                                const std::vector<std::array<double, 3>> &grid_source,
                                const std::vector<std::array<double, 3>> &norm_target,
                                const std::vector<std::array<std::complex<double>, 3>> &Js,
                                const std::vector<std::array<std::complex<double>, 3>> &Ms,
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
    std::array<std::complex<double>, 3> temp1;          // Temporary container 1 for intermediate irrelevant values
    std::array<std::complex<double>, 3> temp2;          // Temporary container 2
    
    // Return containers
    std::array<std::array<std::complex<double>, 3>, 2> beam_e_h; // Container for storing fieldAtPoint return
    
    for(int i=start; i<stop; i++)
    {
        
        point = grid_target[i];
        norms = norm_target[i];
        
        // Calculate total incoming E and H field at point on target
        beam_e_h = fieldAtPoint(grid_source, Js, Ms, point, source_area, start);
        
        
        
        // Calculate normalised incoming poynting vector.
        ut.conj(beam_e_h[1], temp1);                        // h_conj
        ut.ext(beam_e_h[0], temp1, temp2);                  // e_out_h
        
        for (int n=0; n<3; n++) 
        {
            e_out_h_r[n] = temp2[n].real();                      // e_out_h_r
        }
            
        this->Et_container[i] = beam_e_h[0];
        this->Ht_container[i] = beam_e_h[1];
        //std::cout << this->Et_container.size() << std::endl;
        
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
        for(int n=0; n<3; n++)
        {
            e_r[n] = -e_dot_p_r_perp * p_i_perp[n] - e_dot_p_r_parr * p_i_parr[n];
            
            //this->Et_container[k][i] = e_r[k];
        }

        ut.ext(S_r_norm, e_r, temp1);                       // h_r_temp
        ut.s_mult(temp1, ZETA_0_INV, h_r);                  // h_r       
        
        // Now calculate j and m
        for(int n=0; n<3; n++)
        {
            temp1[n] = e_r[n] + beam_e_h[0][n]; // e_i_r
            temp2[n] = h_r[n] + beam_e_h[1][n]; // h_i_r
        } 
        //std::cout << "check" << std::endl;
        ut.ext(norms, temp2, this->Jt_container[i]);
        ut.ext(norms, temp1, n_out_e_i_r);

        ut.s_mult(n_out_e_i_r, -1., this->Mt_container[i]);//this->Mt_container[i] = -n_out_e_i_r;
        //std::cout << "jooo" << std::endl;
        if(i % 100 == 0 and start == 0 * this->step)
        {
            //std::cout << p_i_perp[0] << std::endl;
            std::cout << i << " / " << this->step << std::endl;
        }
    }
}

std::array<std::array<std::complex<double>, 3>, 2> Propagation::fieldAtPoint(const std::vector<std::array<double, 3>> &grid_source,
                                               const std::vector<std::array<std::complex<double>, 3>> &Js,
                                               const std::vector<std::array<std::complex<double>, 3>> &Ms,
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
    
    omega = C_L * k;
    
    for(int i=0; i<gridsize_s; i++)
    {
        ut.diff(point_target, grid_source[i], r_vec);
        ut.abs(r_vec, r);                              
        
        r_inv = 1 / r;
        
        ut.s_mult(r_vec, r_inv, k_hat);
        ut.s_mult(k_hat, k, k_arr);
        
        // e-field
        ut.dot(k_hat, Js[i], r_in_s);
        ut.s_mult(k_hat, r_in_s, temp);
        ut.diff(Js[i], temp, e_vec_thing);
        
        ut.ext(k_arr, Ms[i], k_out_ms);
        
        // h-field
        ut.dot(k_hat, Ms[i], r_in_s);
        ut.s_mult(k_hat, r_in_s, temp);
        ut.diff(Ms[i], temp, h_vec_thing);
        
        ut.ext(k_arr, Js[i], k_out_js);
        
        Green = exp(this->t_direction * j * k * r) / (4 * M_PI * r) * source_area[i] * j;
        
        for( int n=0; n<3; n++)
        {
            e_field[n] += (-omega * MU_0 * e_vec_thing[n] + k_out_ms[n]) * Green;
            h_field[n] += (-omega * EPS * h_vec_thing[n] - k_out_js[n]) * Green;
        }  
    }
    
    // Pack e and h together in single container 
    e_h_field[0] = e_field;
    e_h_field[1] = h_field;
    
    //std::cout << ut.abs(e_field) << std::endl;
    
    return e_h_field;
}

void Propagation::parallelProp(const std::vector<std::array<double, 3>> &grid_target,
                               const std::vector<std::array<double, 3>> &grid_source,
                               const std::vector<std::array<double, 3>> &norm_target,
                               const std::vector<std::array<std::complex<double>, 3>> &Js,
                               const std::vector<std::array<std::complex<double>, 3>> &Ms,
                               const std::vector<double> &source_area)
{
    int final_step; 
    
    //std::cout << gridsize_t << std::endl;
    
    
    for(int n=0; n<numThreads; n++)
    {
        //std::cout << n << std::endl;
        if(n == (numThreads-1))
        {
            final_step = gridsize_t;
        }

        else
        {
            final_step = (n+1) * step;
        }
        
        //std::cout << final_step << std::endl;
        
        threadPool[n] = std::thread(&Propagation::propagateBeam, 
                                    this, n * step, final_step, 
                                    grid_target, grid_source, norm_target, Js, Ms, source_area);
    }
}

// Far-field functions to calculate E-vector in far-field
void Propagation::calculateFarField(int start, int stop,
                                const std::vector<std::array<double, 2>> &grid_ff,
                                const std::vector<std::array<double, 3>> &grid_source,
                                const std::vector<std::array<std::complex<double>, 3>> &Js,
                                const std::vector<std::array<std::complex<double>, 3>> &Ms,
                                const std::vector<double> &source_area)
{
    // Scalars (double & complex double)
    double theta;
    double phi;
    double cosEl;
    
    std::complex<double> e_th;
    std::complex<double> e_ph;
    
    std::complex<double> e_Az;
    std::complex<double> e_El;

    // Arrays of doubles
    std::array<double, 2> point_ff;            // Angular point on far-field
    std::array<double, 3> r_hat;                // Unit vector in far-field point direction

    // Arrays of complex doubles
    std::array<std::complex<double>, 3> e;            // far-field E-field
    for(int i=start; i<stop; i++)
    {
        theta = grid_ff[i][0];
        phi = grid_ff[i][1];
        cosEl = std::sqrt(1 - sin(theta) * sin(phi) * sin(theta) * sin(phi));
        
        r_hat[0] = cos(theta) * sin(phi);
        r_hat[1] = sin(theta) * sin(phi);
        r_hat[2] = cos(phi);
        
        double test;
        ut.abs(r_hat, test);
        //std::cout << r_hat[2] << std::endl; 
        
        // Calculate total incoming E and H field at point on target
        e = farfieldAtPoint(grid_source, Js, Ms, r_hat, source_area, start);
        /*
        // Convert to spherical
        e_th = cos(theta) * (cos(phi) * e[0] + sin(phi) * e[1]) - sin(theta) * e[2];
        
        e_ph = -sin(phi) * e[0] + cos(phi) * e[1];
        
        // Convert to AoE
        
        e_Az = cos(phi) * e_th - cos(theta) * sin(phi) * e_ph;
        e_El = cos(theta) * sin(phi) * e_th + cos(phi) * e_ph;
        
        e[0] = e_Az / cosEl;
        e[1] = e_El / cosEl;
        e[2] = z0;
        */
        this->Et_container[i] = e;

        if(i % 100 == 0 and start == 0 * this->step)
        {
            std::cout << i << " / " << this->step << std::endl;
        }
    }
}

std::array<std::complex<double>, 3> Propagation::farfieldAtPoint(const std::vector<std::array<double, 3>> &grid_source,
                                               const std::vector<std::array<std::complex<double>, 3>> &Js,
                                               const std::vector<std::array<std::complex<double>, 3>> &Ms,
                                               const std::array<double, 3> &r_hat,
                                               const std::vector<double> &source_area,
                                               const int start)
{
    // Scalars (double & complex double)
    double omega_mu;                       // Angular frequency of field times mu
    double r_hat_in_rp;                 // r_hat dot product r_prime
    std::complex<double> r_in_s;        // Container for inner products between wavevctor and currents
    
    // Arrays of doubles
    std::array<double, 3> source_point; // Container for xyz co-ordinates
    
    // Arrays of complex doubles
    std::array<std::complex<double>, 3> e;        // Electric field on far-field point
    std::array<std::complex<double>, 3> _js;      // Temporary Electric current at source point
    std::array<std::complex<double>, 3> _ms;      // Temporary Magnetic current at source point
    
    std::array<std::complex<double>, 3> js;      // Build radiation integral
    std::array<std::complex<double>, 3> ms;      // Build radiation integral
    
    std::array<std::complex<double>, 3> _ctemp;
    std::array<std::complex<double>, 3> js_tot_factor;
    std::array<std::complex<double>, 3> ms_tot_factor;

    // Matrices
    std::array<std::array<double, 3>, 3> rr_dyad;       // Dyadic product between r_hat - r_hat
    std::array<std::array<double, 3>, 3> eye_min_rr;    // I - rr
    
    e.fill(z0);
    js.fill(z0);
    ms.fill(z0);
    
    omega_mu = C_L * k * MU_0;
    
    ut.dyad(r_hat, r_hat, rr_dyad);
    ut.matDiff(this->eye, rr_dyad, eye_min_rr);
    
    for(int i=0; i<gridsize_s; i++)
    {
        
        ut.dot(r_hat, grid_source[i], r_hat_in_rp);
        
        std::complex<double> expo = exp(j * k * r_hat_in_rp);
        
        for (int n=0; n<3; n++)
        {
            js[n] += Js[i][n] * expo * source_area[i];
            ms[n] += Ms[i][n] * expo * source_area[i];
        }
    }
    ut.matVec(eye_min_rr, js, _ctemp);
    ut.s_mult(_ctemp, omega_mu, js_tot_factor);
    
    ut.ext(r_hat, ms, _ctemp);
    ut.s_mult(_ctemp, k, ms_tot_factor);
    
    for (int n=0; n<3; n++)
    {
        e[n] = -js_tot_factor[n] + ms_tot_factor[n];
    }
    
    return e;
}

void Propagation::parallelFarField(const std::vector<std::array<double, 2>> &grid_ff,
                               const std::vector<std::array<double, 3>> &grid_source,
                               const std::vector<std::array<std::complex<double>, 3>> &Js,
                               const std::vector<std::array<std::complex<double>, 3>> &Ms,
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

        threadPool[n] = std::thread(&Propagation::calculateFarField, 
                                    this, n * step, final_step, 
                                    grid_ff, grid_source, Js, Ms, source_area);
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
        
    
                                        
    
